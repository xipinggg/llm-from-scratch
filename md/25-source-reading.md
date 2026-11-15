*第二十五章 · 源码导读*

# 源码导读：读一个真实的注意力 kernel

第 13 章讲了引擎怎么做注意力：分页、分区、后端选择。这一章换一个角度，把一段真实的 kernel 代码摊开，逐行读它——包括那段最有名的"在线 softmax"。读完之后，你应该能从代码里认出本页讲过的每一个概念，也能看清教学版与生产版之间差的那一层工程。

## 读 kernel 之前：先找出五个坐标

GPU kernel 的代码看起来密，但结构几乎固定。拿到任何一段 kernel（Triton、CUDA 都一样），先找五样东西，剩下的都是填空：

| 坐标 | 在 Triton 里长什么样 | 回答什么问题 |
| --- | --- | --- |
| 程序标识 | `tl.program_id(0/1/2)` | 我这个线程块负责哪一块活？（batch、head、行块） |
| 指针与步长 | 裸指针 + `stride_x` 参数 | 数据在显存里怎么排？我要读的偏移是哪个 |
| 块大小 | `BLOCK_M`/`BLOCK_N`/`HEAD_DIM`（`tl.constexpr`） | 一次搬多大的瓦片？这是性能调优的主要旋钮 |
| 掩码 | `offs_m[:, None] >= start_n + offs_n` | 哪些位置越界、哪些位置不许看（因果） |
| 归约 | `tl.max(..., 1)`、`tl.sum(..., 1)` | 沿哪个轴做归约？结果形状会少一维 |

把注意力 kernel 套进这五个坐标：程序标识对应"第几个请求 × 第几个头 × 第几个行块"；块大小是 $B_M\times B_N$ 的瓦片；掩码负责因果与变长；归约只有两处——softmax 的 max 和 sum。剩下的全部是循环与累加。

## 教学版：Triton 官方教程的内层循环

下面这段来自 Triton 官方教程 `06-fused-attention.py`（MIT 许可），是流式（FlashAttention）注意力最核心的部分。为了让行数可控，去掉了 stage 分支、Hopper 专属路径和 FP8 分支，只保留主线：

```text
for start_n in range(lo, hi, BLOCK_N):        # 沿 KV 方向逐块扫
    k = load_k(start_n)                        # [HEAD_DIM, BLOCK_N]
    qk = tl.dot(q, k)                          # [BLOCK_M, BLOCK_N] 分数瓦片
    qk = qk * qk_scale + tl.where(mask, 0, -1.0e6)   # 缩放 + 因果掩码
    m_ij = tl.maximum(m_i, tl.max(qk, 1))      # 本块与历史的最大值取大
    qk = qk - m_ij[:, None]                    # 减去新的最大值
    p = tl.math.exp2(qk)                       # 指数（2 为底，见下）
    alpha = tl.math.exp2(m_i - m_ij)           # 历史累加项的修正因子
    l_ij = tl.sum(p, 1)                        # 本块的指数和
    acc = acc * alpha[:, None]                 # 历史输出按新最大值重标定
    v = load_v(start_n)                        # [BLOCK_N, HEAD_DIM]
    acc = tl.dot(p, v, acc)                    # 本块贡献累加进去
    l_i = l_i * alpha + l_ij                   # 更新累计分母
    m_i = m_ij                                 # 更新运行最大值
```

十行代码，四个变量。它们就是整章讲的东西在汇编层的形状：

- `q` 是查询瓦片（$B_M\times d$），一个线程块只负责 $B_M$ 行——对应第 5 章的"第 19 位"只跟踪一行；
- `k`、`v` 沿 KV 方向循环加载——这就是"遍读历史"；分页版这里会换成"按 block table 取物理块"；
- `acc` 是未归一化的输出累加器（$B_M\times d$），`l_i` 是分母，`m_i` 是运行中的最大值；
- 循环结束时输出 `acc / l_i`，结果与"一次算完整个 softmax"完全等价。

## 在线 softmax：为什么这四个变量就够

保持两个不变量，整套算法就成立了。设已经扫过的分数字段为 $S$、当前最大值为 $m$：

$$
\mathrm{acc}=\sum_{j\in S} e^{s_j-m}\,v_j,\qquad l=\sum_{j\in S} e^{s_j-m}
$$

于是 $\mathrm{acc}/l = \sum_j \mathrm{softmax}(s)_j\,v_j$——分子分母同乘同一个因子，归一化把它消掉了。来一个新块时，如果最大值从 $m$ 变成 $m'$，旧的和只需要乘一个修正因子：

$$
\mathrm{acc}\leftarrow \mathrm{acc}\cdot e^{m-m'},\qquad l\leftarrow l\cdot e^{m-m'}+l_{\text{new}}
$$

这就是代码里的 `alpha`。没有它，分段计算时旧数据会被旧的最大值缩放，结果就错了；有了它，扫描顺序（先扫长块还是短块、是否分区）都不影响最终结果——这是 split-K 能与单块计算合并的前提（第 13 章的分区合并也是同一个式子）。

还有个小技巧解释了代码为什么用 `exp2` 而不是 `exp`：$e^{x}=2^{x\log_2 e}$，把常数 $\log_2 e$ 提前乘进 `qk_scale`，每次指数就变成一次硬件友好的 $2^x$。数值上完全等价，速度上省掉每条指令里的乘常数。

## 生产版比教学版多什么

教学版假设三件事：KV 连续存放、每个头各自一份 KV、一个请求独占一个 kernel。真实引擎里这三条全不成立，所以 vLLM 的统一注意力 kernel（`triton_unified_attention.py`）的参数表长这样——每个额外的参数都对应一处工程：

| 额外的东西 | 参数/分支 | 解决什么 |
| --- | --- | --- |
| 分页寻址 | `block_tables_ptr`、`seq_lens_ptr`、`stride_k_cache_*` | KV 散落在物理块里，按 block table 查再取 |
| GQA / MQA | `num_queries_per_kv`（`constexpr`） | 多个查询头共享一份 KV，取 K/V 时按 `q_head // group` 映射 |
| 变长批 | `query_start_len_ptr`、`BLOCK_Q` | 一个 batch 里每个请求长度不同，用前缀和定位边界 |
| split-K | `NUM_SEGMENTS_PER_SEQ`、`segm_output_ptr`、`segm_max_ptr`、`segm_expsum_ptr`、`reduce_segments` | 长上下文切成多段并行算，再用 $e^{m_i-M}$ 合并 |
| 量化 | `KV_QUANT_MODE`、`k_scale`/`v_scale`、per-token scale 缓存 | KV 以 fp8/int8 存储，读时反量化 |
| 掩码变体 | `USE_CAUSAL`、`SLIDING_WINDOW`、`USE_SINKS`、`USE_ALIBI_SLOPES`、`USE_SOFTCAP`、`CHUNK_LOOKBACK` | 窗口、sink、位置偏置、logit 上限、块内局部注意力 |
| 混合前缀 | `USE_MM_PREFIX`、`mm_prefix_range_ptr` | 多模态前缀需要双向注意力，与因果部分并存 |
| 硬件路径 | `IS_3D`、`USE_TD`、`warp_specialize` | 分两段式高带宽布局、Tensor Descriptor、warp 专用化 |

把这些叠起来看，就明白为什么"读源码"比"读论文"难：论文里的 20 行公式，落到工程里是 500 行带分支的 kernel，而且分支数量随支持的特性线性增长。反过来，只要抓住那 20 行主线，其余的都能归类到这张表里。

## 怎么验证自己读懂了

四个从易到难的检查，不依赖特定显卡：

1. **用解释器模式跑教学版。**Triton 支持 `TRITON_INTERPRET=1` 在 CPU 上逐行解释执行——慢，但能 `print` 中间张量。把 `BLOCK_N` 设成 4，打印每一轮之后的 `acc`、`l_i`、`m_i`，和手算的 softmax 对照。
2. **对参考实现。**用 `torch.nn.functional.scaled_dot_product_attention` 当基准，比较最大相对误差；再故意把 `alpha` 写成 1，看误差如何随上下文长度爆炸——这会让你彻底记住那个修正因子。
3. **数一遍访存。**按 $B_M$、$B_N$ 推出每个线程块读了多少字节，和第 10 章的带宽公式对一遍；如果和 ncu 的 DRAM 读放大差异很大，多半是掩码或块大小用错了。
4. **找一个真实 bug。**读 vLLM 的 issue：分页 kernel 的 off-by-one（最后一块不满 16 个 token）、变长批的边界、split-K 合并时忘了 `alpha`，都是真实发生过的。能看懂这些 issue 的复现步骤，就算读进去了。

> **这一章的要点**
>
> 读 kernel 先找五个坐标（程序标识、指针步长、块大小、掩码、归约），再找四个变量（q、K/V 瓦片、acc、l_i/m_i）。在线 softmax 用一个修正因子 `alpha = exp(m_i - m_ij)` 把"分段算"变成"和一次算等价"，这是流式注意力、split-K 与分区合并共用的同一件事。生产版多出来的每一行，几乎都能对应到本页某一章的某个概念。

> **到此为止**
>
> 这一章是代码层的终点。如果还想往下走，两条路最实在：一是把教学版 kernel 改成分页版（加 block table 与 GQA），二是在有 GPU 的机器上跑一遍 `vllm bench serve`，把第 12 章的参数表变成你自己机器上的数字。最后一章（第 26 章）是一张研究前沿地图，可以当"接下来该读什么"的目录。

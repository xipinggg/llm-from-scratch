*第二十六章 · 前瞻*

# 研究前沿：推理引擎正在往哪走

前面的章节讲的都是"已经定型的机制"。这一章换一个角度：2025–2026 年，做推理系统的人正在解决什么问题、用什么办法、拿到了多大的数字、还剩什么没解决。它更像一张地图而不是教程——读完之后，你应该能判断一篇新论文在打哪一场仗，也能看出哪些"优化"其实是在把瓶颈换个地方。

## 为什么重心从 kernel 移到了数据移动

2022–2024 年，推理性能的提升主要来自把单个算子写快：FlashAttention 把注意力的访存从平方降到线性、PagedAttention 把显存碎片压掉、连续批处理把 GPU 的空闲填满。这些优化现在已经被吃到极限——主流的三个生产引擎（vLLM、SGLang、TensorRT-LLM）在这些点上的差距通常只剩个位数百分比。

而 roofline 那笔账没有变：一次 decode 要把 16 GB 权重从显存读一遍，算力只用掉 2%–3%。**把 kernel 写快解决不了"读得多"**。于是 2025 年之后的研究集中在四个动作上：

| 动作 | 典型手段 | 收益量级 | 成熟度 |
| --- | --- | --- | --- |
| 搬得少 | KV 量化、稀疏注意力、GQA/MLA、线性注意力 | 2–50× | 已进模型结构 |
| 搬得快 | NVLink、RDMA、分层存储、异步拷贝 | 4–30× | 已产品化 |
| 搬得巧 | PD 分离、前缀缓存、跨实例共享、按会话调度 | 1.5–6× | 产品化中 |
| 少搬一次 | 投机解码、MTP、批内复用 | 1.5–3× | 已进默认配置 |

这张表也是下面八条主线的索引：每一条都在打其中一行。

## 主线一：PD 分离，从"拆开"到"按 SLO 自动配比"

预填充吃算力、解码吃带宽，放在同一张卡上必然有一半资源闲着。2024 年的 DistServe、Splitwise 证明了"拆开"有效，2025 年之后的问题变成了三件：拆开之后比例怎么定、KV 怎么搬、异构硬件上怎么搬。

- **Mooncake**（FAST'25 最佳论文）把整台服务改造成以 KVCache 为中心：prefill 集群与 decode 集群分离，同时把推理集群里闲置的 CPU、DRAM、SSD、NIC 组成一个独立的 KV 缓存池，再由全局缓存与调度器在 SLO 约束下最大化吞吐。真实流量下有效请求处理能力提升 **59%–498%**，已在数千节点上每天处理 **1000 亿 token**。[来源](https://www.tsinghua.edu.cn/en/info/1245/14138.htm)
- **Dynamo 与 llm-d** 把这件事产品化：Planner Profiler 先测出本机型的性能曲线，SLA Planner 再按目标 SLO 自动匹配 prefill:decode 的实例比例；KV 通过 NIXL 在 RDMA 上搬；多模态进一步拆成 **EPD**（encode–prefill–decode）三段。[Dynamo 会议](https://www.nvidia.com/en-us/on-demand/session/other25-dynamoday14/)、[机制说明](https://github.com/rohitg00/ai-engineering-from-scratch/blob/a509402c0b5455d11ddc2b5fa83cc8c0d2e0692b/phases/17-infrastructure-and-production/17-disaggregated-prefill-decode/docs/en.md)
- **异构 PD** 是 2026 年的新方向：AsymFlow 让 CPU 承担预填充/解码分离后的部分工作，其 CPU 侧 decode 注意力算子快 **2 倍**（[ICPP'26](https://dl.acm.org/doi/10.1145/3832810.3832866)）；PiKV-Serve 面向边缘-云链路，把 KV 按层流水线迁移、与计算重叠（[IEEE](https://ieeexplore.ieee.org/document/11746543)）。

**还没解决**：KV 传输的尾延迟（一次 1.28 GB 的跨节点搬运就是几十毫秒，见第 11 章的分层表）、prefill:decode 比例对负载突变不鲁棒、跨数据中心基本不可行。

## 主线二：KV cache 从"副产品"变成"一等存储"

这是过去两年变化最大的一条。核心认识是：KV 是可以被**寻址、复用、分层、跨实例共享**的对象，而不是某个请求的私有中间变量。

- **LMCache** 把卸载做成独立服务：在 Dynamo 里以 DaemonSet 跑，通过跨 Pod CUDA IPC 与 worker 共享张量（[文档](https://docs.dynamo.nvidia.com/dynamo/v1.6.1/kubernetes/kv-cache-offloading/deploy-lm-cache-mp)）；2026 年又做了**多节点 P2P CPU 内存共享**，专门解决"多实例各攒各的缓存"——四个实例时缓存命中率会掉到 1/4（[博客](https://blog.lmcache.ai/en/2026/01/21/p2p-1/)）。
- **全局 KV 感知调度**：Axon（SIGOPS ATC'26）把"上下文在哪台机器上"变成调度变量——会话级的上下文可以在 GPU 常驻缓存与 CPU 内存之间迁移，优先走 PCIe 恢复、必要时才做 GPU-GPU 迁移，已实现在 vLLM 之上，用生产 agent 轨迹在 4 台 A100 上评测（[论文](https://dl.acm.org/doi/10.1145/3828523.3857095)）。
- **反向的声音**：vAttention 认为分页本身是性能负担——它用操作系统式虚拟内存加 CUDA VMM 做"按需映射的连续 KV"，并引用实测：paged prefill kernel 比非分页慢约 **42%**，部分场景整体延迟高约 **5%**（[论文](https://arxiv.org/html/2405.04437)）。这条线提醒：分页解决碎片，但把复杂度转嫁给了 kernel。

**还没解决**：缓存一致性（权重版本、LoRA、精度）、失效语义、跨租户的计时侧信道（第 11 章安全一节）、跨集群共享的可行性。

## 主线三：架构与系统的协同设计

模型结构正在**为了服务成本而改变**，这是 2025 年最值得注意的趋势。三例子：

- **DSA（DeepSeek Sparse Attention）**：V3.2-Exp 用两级注意力——先由 lightning indexer 给全量缓存打分、每层保留 top-2048，再对这 2048 个 token 做稀疏注意力（[技术说明](https://www.amd.com/en/developer/resources/technical-articles/2025/amd-instinct-mi335x-gpu-supports-deepseek-v3-2-exp.html)）。
- **瓶颈搬家**：长上下文下 decode 的时间花在"每层扫一遍全量缓存"的索引器上，注意力每层都要等它。NeurIPS'26 的多层索引器论文进一步指出：相邻层选出的 token 高度重叠，因此索引器可以跨层共享（[论文](https://neurips.cc/virtual/2026/loc/atlanta/173747)）。
- **线性与混合注意力**（Qwen3-Next 的 Gated DeltaNet 层、Mamba/RWKV 类状态递推）：KV 不再随长度增长，但 decode 每步要读写固定大小的状态矩阵。按 3 层 GDN + 1 层全注意力的配置估算，GDN 的 decode 算术强度只有约 **0.87 FLOP/B**——瓶颈从"读 KV"变成"读写状态"。

**还没解决**：稀疏索引器自身的高效 kernel、混合架构的缓存语义（循环状态与 KV 并存）、训练与推理在稀疏模式上的一致性。

## 主线四：投机解码从论文走进默认配置

- **EAGLE-3** 不再训练独立草稿模型，而是在目标模型内部层挂一个只有 **2%–5%** 大小的草稿头，在特征级与词元级同时预测；Google Vertex AI 用 SGLang 大规模部署，解码提速约 **2–3 倍**（[博客](https://docs.cloud.google.com/vertex-ai/docs/blog/posts/from-research-to-production-accelerate-oss-llm-with-eagle-3-on-vertex)）。
- **MTP（多 token 预测）**从 DeepSeek-V3 起成为新模型的标配头，推理引擎把它当"内置投机"。
- 研究点转到了**交互项**：接受率随 batch 增大而衰减、被拒绝的草稿 token 不能写进 KV 缓存（否则下一次命中就错，见第 11 章前缀缓存一节）、与结构化输出/约束解码的先后顺序。

## 主线五：MoE 与大规模并行

- **wide EP**：DeepSeek 的 large-scale EP、DeepEP 通信库把专家并行推到几十卡；TensorRT-LLM 在 GB200 上做了 EP32 的实测，**在线 EPLB** 周期性重排专家放置，配合专家副本把最热的 rank 打散（[技术博客](https://nvidia.github.io/TensorRT-LLM/1.2.0rc5/blogs/tech_blog/blog4_Scaling_Expert_Parallelism_in_TensorRT-LLM.html)）。
- 研究点：all-to-all 与注意力的重叠、路由负载偏斜、专家副本的收益与显存代价、EP 规模上去之后的故障恢复（FailSafe、ReviveMoE 一类工作）。

## 主线六：kernel、编译与精度

- **FlashAttention-4** 是"算法与流水线协同设计"，针对 Blackwell 的非对称扩展（tcgen05、warp specialization、软硬件协同的 softmax 重缩放）。[论文](https://ar5iv.labs.arxiv.org/html/2603.05451)、[CuTe DSL 实现](https://pypi.org/project/fa4-jax/0.1.0/)
- **手写 kernel 仍不可替代**：CuTe DSL 让"只针对某代 GPU 的高性能库"变得可行，比如只支持 SM120 的 NVFP4 GEMM / MoE / paged attention 库，直接对接 vLLM 与 SGLang（[b12x](https://pypi.org/project/b12x/0.1.0/)）。
- **编译与自动调优的代价被正式对待**：TensorRT-LLM 的一个改动显示，首次 prefill 要编译 42–59 个 Triton 变体、每 rank 约 **31 秒**；改成提前预编译后降到约 **5 秒**（[PR](https://github.com/NVIDIA/TensorRT-LLM/pull/19843)）。
- 研究点：JIT 冷启动、CUDA graph 与动态形状的冲突、FP8/FP4 的 scale 布局、注意力后端与批次形状的兼容性。

## 主线七：长上下文与 agent 负载

- **Decode Context Parallelism（DCP）**：decode 阶段把 KV 按序列切到多卡，用更多 GPU 换更长上下文与更大并发；vLLM 在 2026 年专门写了博客讲它在长上下文 agent 场景下的收益（[博客](https://vllm.ai/blog/2026-08-07-decode-context-parallelism)）。
- **负载形态变了**：长思维链让推理从"算力受限"转向"容量受限"——KV 随思考长度线性增长，系统刻画显示 8B 到 671B 的最优并行策略都因此改变（[IEEE 2026](https://ieeexplore.ieee.org/document/11617776)）；KV 剪枝开始针对推理负载做层与时间自适应（Lethe，[AAAI'26](https://ojs.aaai.org/index.php/AAAI/article/view/40036)）。
- **容量分层**：短上下文走低延迟实例、中长上下文走平衡配置、超长上下文走专用池（[实践综述](https://introl.com/blog/long-context-llm-infrastructure-million-token-windows-guide)）。

## 主线八：调度、公平与可运维性

- **体验公平**：AAAI'26 的 Experiential Fairness 指出，按资源分配的公平性与用户感知的流式体验并不一致，提出以"体验"为目标的调度（[论文](https://ojs.aaai.org/index.php/AAAI/article/view/40946/44907)）。
- **故障与恢复**：FailSafe 处理不规则 GPU 可用性下的张量并行；ReviveMoE 面向大规模 MoE 的快速恢复（概述见[这份运维对照](https://raw.githubusercontent.com/anthony-chaudhary/fak/refs/tags/v0.43.0/docs/industry-scorecard/operability.md)）。
- **容量规划用模拟器**：Vidur 只做一次 profiling，就能在 CPU 上模拟调度策略、投机解码与并行配置的端到端性能（[工具](https://pypi.org/project/vidur/0.0.17/)）。

## 一张研究地图

| 方向 | 核心问题 | 代表工作 | 量化收益 | 成熟度 |
| --- | --- | --- | --- | --- |
| PD 分离 | 比例怎么定、KV 怎么搬 | Mooncake、Dynamo、llm-d | 有效容量 1.6–6× | 生产 |
| KV 存储层 | 缓存放哪、谁能用 | LMCache、HiCache、Axon | 命中率从 1/N 到接近 1 | 生产/试点 |
| 非分页内存 | 分页的 kernel 代价 | vAttention | prefill +42%（反例） | 研究 |
| 原生稀疏 | 选谁、谁选 | DSA、NSA、多层索引器 | 长上下文成本显著下降 | 生产（V3.2/GLM-5） |
| 线性/混合 | 状态的读写成本 | Qwen3-Next、Mamba | KV 不随长度增长 | 生产 |
| 投机与 MTP | 接受率、与缓存的一致性 | EAGLE-3、MTP | 解码 2–3× | 生产 |
| MoE 大 EP | 通信与负载均衡 | DeepEP、EPLB | 热点 rank 被打散（在线 EPLB） | 生产 |
| kernel 与编译 | 异步硬件、冷启动 | FA4、CuTe DSL、NVFP4 | 启动 31s→5s | 生产/研究 |
| 长上下文/agent | 容量与会话级 SLO | DCP、Lethe、Axon | 并发提升（视链路与长度） | 生产/试点 |
| 调度与公平 | 尾延迟、体验、恢复 | Vidur、FailSafe、体验公平 | 取决于负载 | 研究 |

## 三个判断

> **一、重心从单个算子移到数据移动**
> 单算子优化的空间已经吃尽，剩下的大头是"KV 放在哪、谁来搬、什么时候搬、按什么规则分给谁"。这也是为什么过去两年最亮眼的结果多来自系统论文而不是 kernel 论文。

> **二、模型与系统开始共同设计**
> DSA、MTP、混合线性注意力都是"为了服务成本而改结构"；反过来，引擎也在为这些结构定制 kernel（稀疏索引器、状态缓存、MLA 吸收）。两边不再是上下游，而是同一个设计问题的两半。

> **三、目标从吞吐变成延迟承诺、公平与成本**
> 多租户、agent 长会话、推理模型的长时间思考，都在迫使调度器处理"尾延迟、缓存命中率、单位会话成本"，而不是只盯 tokens/s。可承诺的延迟正在成为比峰值吞吐更重要的指标。

> **反面提醒：没有免费的优化**
> 本页出现过三次"瓶颈搬家"：稀疏把瓶颈搬到索引器，线性注意力搬到状态读写，PD 分离把瓶颈搬到 KV 传输。看到一个优化，先问它把成本挪到了哪里、那个地方会不会成为新的墙。

## 如果你要做研究：八个可验证的空白

1. **多层索引器的选择一致性**：相邻层重叠多少才算"可以共享"？需要的是可度量的选择相似度，以及共享之后的精度损失上界。
2. **混合架构的缓存语义**：GDN/Mamba 的循环状态与 KV 并存时，"前缀缓存"该怎么定义？状态的更新不是只追加，复用条件要重新推。
3. **跨集群 KV 迁移的 SLO 可行性**：在 50 ms 量级的广域链路上，什么规模的请求值得搬、什么规模不如重算？目前只有单机房数据。
4. **投机 + 前缀缓存 + 约束解码的三方交互**：三者对"哪些 token 合法"的判断顺序不同，组合起来的一致性还没有系统论证。
5. **稀疏注意力的 prefill kernel**：现有稀疏优化大多针对 decode；prefill 阶段块稀疏在张量核心上的利用率仍是开放问题。
6. **会话级容量规划**：agent 的 SLO 不是"每个请求多快"，而是"这个会话多久能推进完"，需要新的度量与调度目标。
7. **联合基准**：长上下文 + agent + 投机的组合负载缺一个公认基准，现有 benchmark 各自只覆盖一角。
8. **共享缓存的安全**：前缀缓存的计时侧信道已有攻击，但"可证明的防护 + 可接受的性能损失"还没有落地答案。

## 怎么读这些论文：五个问题

| 问题 | 为什么重要 | 看哪里 |
| --- | --- | --- |
| baseline 是谁 | 打 vLLM 默认配置和打 TRT-LLM 调优配置，结论可能相反 | 实验设置那一节 |
| 度量是什么 | TTFT、TPOT、goodput、$/百万 token 是四个不同的目标 | 指标定义与负载描述 |
| 负载从哪来 | 合成 trace 与真实 trace 的最优配置常常不同 | workload 一节 |
| 隐藏成本是多少 | 编译时间、显存占用、迁移带宽、冷启动往往不进主表 | 结果表之外的脚注 |
| 能复现吗 | 代码/配置/trace 是否公开，决定结论能不能迁移到你的机器 | artifact 与附录 |

> **这一章的用法**
>
> 把这张地图当目录：遇到新论文，先定位它打的是哪一行、把成本挪到了哪里；做选型时，先问自己的负载落在哪个区间（长上下文?agent?高并发短请求?），再挑对应主线的结论。本书前面所有章节给的数字都是"机制不变时的账"，这一章给的是"机制正在变的方向"——两者一起看，才不会把某一年的最优配置当成永恒答案。

> **关于来源与时效**
>
> 本章的数字都来自 2025–2026 年的公开论文、官方文档与工程博客，链接附在正文里（离线打开时链接不可点，但不影响阅读）。这个领域半年就会变一轮：看到具体倍数时，先确认它对应的模型规模、硬件与负载；看到"生产可用"时，先确认它指的是引擎默认配置还是某家公司的内部部署。

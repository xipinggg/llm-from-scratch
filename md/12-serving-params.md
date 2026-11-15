*第十二章 · 服务参数*

# 服务参数：vLLM 与 SGLang 调参

前两章算的是算力、显存和带宽这些硬指标。这一章算另一种账：同一个模型、同一张卡，参数配错能让吞吐差 3–10 倍，而且多数时候不报错，只会让人以为硬件不行。下面把 vLLM 与 SGLang 最常调的十几个参数按用途拆开讲。

## 参数按用途分成四类

服务端的参数看着杂乱，其实只分四层，每层回答一个不同的问题：

> **容量 · 能装多少 KV**
> `gpu-memory-utilization`（vLLM）`--mem-fraction-static`（SGLang）
> 配错的症状：并发上不去、频繁抢占、启动 OOM。

> **调度 · 每步先算谁**
> `max-num-seqs`、`max-num-batched-tokens`（vLLM）`--max-running-requests`、`--chunked-prefill-size`、`--schedule-policy`（SGLang）
> 配错的症状：TTFT 高、ITL 抖、GPU 空转。

> **并行 · 多卡怎么分**
> `-tp` / `-pp` / `-dp`、`enable_expert_parallel`（vLLM）`--tp-size` / `--pp-size` / `--dp-size` / `--ep-size`（SGLang）
> 配错的症状：通信打满、扩展效率低。

> **精度 · 几个 bit 换多少速度**
> `--quantization`、`--kv-cache-dtype`（两家通用）
> 配错的症状：质量掉、长上下文明显变差。

> **所有参数最后都落在三个量上**
>
> 所有参数最终都影响三个数字：能同时挂几条序列（容量）、每一步算多少 token（调度）、每条序列等多久（延迟）。调参就是在这三者之间取舍，没有免费的档位。

## 显存怎么分：权重、KV 池、激活

显存不是被一个参数决定的，它被切成三块：模型权重、KV cache 池、以及运行时开销（激活、CUDA graph 的捕获缓冲、通信 buffer）。前两块是你能选的，第三块是框架替你留的。

$$
f_{\text{static}} = \frac{W + P}{M_{\text{gpu}}},\qquad P = f_{\text{static}}\cdot M_{\text{gpu}} - W\qquad (12.1)
$$

这是 SGLang `--mem-fraction-static` 的定义：**权重加 KV 池占整张卡的比例**。剩下的 $1-f$ 就是留给激活、graph 缓冲和碎片的余量。以 H100 80GB 跑 Llama-3.1-8B（BF16 权重 16.06GB）为例：

**一个 f 值决定多少并发（H100 80GB，Llama-3.1-8B）**

> **SGLang：按比例切**
> `--mem-fraction-static` 是一个**声明**：我打算让权重和 KV 池占这么多，余下的框架自己用。它是静态的、由你负责留余量，调大一点就多几条并发，调大到越界就是启动或运行期 OOM。

> **vLLM V1：按实测切**
> CUDA/ROCm 上 vLLM 先加载权重、跑 warmup、捕获 CUDA graph，**再按剩下多少空闲显存去开 KV 池**（可扩展 KV cache）。所以 `gpu-memory-utilization` 在 V1 里默认就是 1.0，通常根本不用调。

> **两个框架的同名参数不能互抄数值**
>
> 把 SGLang 的 0.90 抄给 vLLM V1 的 `gpu-memory-utilization`，会白丢 10% 的 KV 池——因为 V1 的 1.0 意思是"把实测完剩下的都给 KV"，不是"占满 100% 显存"。反过来把 vLLM 的 1.0 抄给 SGLang，会连激活余量一起吃掉，长 prompt 直接 OOM。旧教程里的 0.9 是 V0 时代的经验值。

## 并发上限由三个约束决定

你设的 `max-num-seqs`（SGLang 叫 `--max-running-requests`）只是一个**上限**，真正的并发是三个约束里最小的那个：

$$
N_{\text{real}} = \min\Big(N_{\text{seqs}},\ \Big\lfloor \frac{P}{C\cdot T} \Big\rfloor,\ N_{\text{batch budget}}\Big)\qquad (12.2)
$$

其中 $C$ 是每 token 的 KV 字节数（第 11 章算过，Llama-3 8B 是 128 KB），$T$ 是上下文长度。这个式子带来一个反直觉的结论：**并发和上下文长度成反比**。同一张卡，8K 上下文能挂 52 条，32K 就只有 13 条，128K 只剩 3 条。

**同一个池（55.9GB）在不同上下文下的并发**

> **所以先问一句：真的需要 128K 吗**
>
> `max-model-len` / `--context-length` 不是"越大越好"的开关，它直接决定 KV 单价。多数线上业务的 95 分位长度只有几千 token，把窗口设到 32K 只是在为少数长请求牺牲多数短请求的并发。常见做法是：主集群按 8K–16K 配，长上下文单独起一个小实例池。

## 每步预算：max-num-batched-tokens 与 chunked prefill

调度器每一步只做一件事：**在 token 预算内挑一批活**。预算就是 `max-num-batched-tokens`（SGLang 里是 `--max-prefill-tokens` 与 `--chunked-prefill-size` 共同作用）。它决定了 prefill 和 decode 怎么抢同一块算力。

$$
B \;\ge\; \underbrace{D_{\text{decode}}}_{\text{先排：每条序列 1 token}} \;+\; \underbrace{\min\big(s_{\text{chunk}},\ B - D_{\text{decode}}\big)}_{\text{剩多少排多少 prefill，装不下就切块}}\qquad (12.3)
$$

vLLM V1 默认开启 chunked prefill，并且调度顺序是**先 decode、后 prefill**：这样解码请求不会因为别人一句 3 万 token 的长文而卡住几秒，同时同一批里既有计算密集（prefill）又有带宽密集（decode），GPU 反而更满。

| batch tokens | ITL 峰值抖动 | TTFT（2K prompt） | 适合 |
| --- | --- | --- | --- |
| 2048 | ≈ 137 ms | ≈ 0.2 s | 聊天，最在意顺滑 |
| 8192（常用） | ≈ 546 ms | ≈ 0.15 s | 通用在线服务 |
| 16384 | ≈ 1.1 s | ≈ 0.14 s | 长 prompt 批处理 |
| 32768 | ≈ 2.2 s | ≈ 0.13 s | 整段文档一次灌入 |

抖动按 prefill 吞吐 ≈ 15000 token/s 估算（H100 上 8B 模型的量级）；实际值取决于模型、序列长度分布和是否命中前缀缓存。真实系统里你看到的往往是"几十到几百毫秒"的毛刺，而不是整段阻塞。

> **一个会让服务起不来的坑**
>
> 如果你**关闭**了 chunked prefill（vLLM 用 `--no-enable-chunked-prefill`，SGLang 把 `--chunked-prefill-size` 设成 −1），那么 `max-num-batched-tokens` 必须**大于** `max_model_len`，否则一条最长请求都排不进去，服务启动即崩。这个约束在文档里只有一行小字，但踩中率极高。

## 前缀缓存：省下的是 prefill

多轮对话、RAG、few-shot 提示里，每次请求的开头往往一模一样。前缀缓存（vLLM 的 `--enable-prefix-caching`，SGLang 的 radix cache）把已经算过的 KV 块按 token 前缀存起来，下一句直接复用，**跳过的正是最贵的 prefill**。

**一段 2000 token 的系统提示，被 1000 个请求复用**

缓存是**按块对齐**的：vLLM 默认块大小 16 token，SGLang 的 `--page-size` 默认 1。也就是说前缀哪怕只差中间一个 token，从那里往后全部不命中。这带来两个实操结论：

- **把稳定内容放最前面。**系统提示、工具定义、few-shot 示例放开头，用户这句话放最后——顺序反了缓存就全废。
- **调度策略要配合。**SGLang 默认 `fcfs`（先来先服务）会把相同前缀的请求打散；`--schedule-policy lpm`（最长前缀匹配优先）让同前缀请求挤在一起，命中率明显更高，代价是调度器多算一点。

## 抢占：日志里的 retracted 与 preempted

当 KV 池真的满了，调度器不能拒绝已经在跑的请求，只能**把一些请求的 KV 腾出来**，等有空位再补回去。这就是抢占。vLLM V1 的默认策略是 **RECOMPUTE**（丢掉 KV、之后重算），而不是 V0 时代的 SWAP（换到 CPU 内存再换回来）：

```text
WARNING Sequence group 0 is preempted by PreemptionMode.RECOMPUTE mode
        because there is not enough KV cache space.
```

看到它不代表服务坏了，但**频率**很重要。偶尔一次是正常波动；持续出现说明容量或并发设错了。对策按代价从小到大排：

| 做法 | 动作 | 代价 |
| --- | --- | --- |
| 降并发 | 调小 `max-num-seqs` / `--max-running-requests` | 吞吐下降，但延迟更稳 |
| 降单请求长度 | 调小 `max-model-len` / `--context-length` | 长请求被拒或截断 |
| 加容量 | 调大 `gpu-memory-utilization`（SGLang 调大 `--mem-fraction-static`） | 离 OOM 更近，激活余量变少 |
| 加卡 | 提高 TP 或 PP，让每卡权重更小、KV 更多 | 通信开销上升 |

## 并行方式：TP、PP、DP、EP 怎么选

四个词的差别只在"切什么"。

| 策略 | 切什么 | 什么时候用 | 主要代价 |
| --- | --- | --- | --- |
| **TP** 张量并行 | 每层的矩阵按列/行切开 | 单卡装不下模型；想给 KV 腾显存 | 每层都要 all-reduce，卡越多通信越贵 |
| **PP** 流水并行 | 按层切成若干段 | 跨机器；层多而每层窄的模型 | 流水气泡，单请求延迟变高 |
| **DP** 数据并行 | 整个模型复制一份，各跑各的请求 | 单卡装得下、要的是总吞吐 | 显存利用率低（每份都存全量权重） |
| **EP** 专家并行 | MoE 的不同专家分到不同卡 | DeepSeek / Qwen-MoE 这类稀疏模型 | token 路由带来 all-to-all 通信 |

> **先看单卡放不放得下**
>
> **单卡放得下、目标是吞吐 → 优先 DP，而不是 TP。**TP 的每层 all-reduce 是同步开销，会拖慢每一步；DP 各卡完全独立，只共享一个路由层。SGLang 文档的建议也是：显存够时用 DP 扩吞吐，用它的 Model Gateway / Router 把请求分到多个实例上。真正必须用 TP 的场景只有一个：**模型大到一张卡放不下**。

还有一条容易被忽略的资源：CPU。**每块 GPU 至少配 2 + N 个物理核**（1 个 API 进程 + 1 个调度进程 + 每卡 1 个 worker），开了超线程还要再乘 2。DP 或多 API 进程时用这个式子：

$$
\text{物理核} \ge A + DP + N + (DP > 1\ ?\ 1 : 0)\qquad (12.4)
$$

比如 8 卡、DP=4、TP=2：4 个 API + 4 个调度 + 8 个 worker + 1 个 DP 协调 = **17 个进程**。CPU 不够时 GPU 利用率上不去，而 `nvidia-smi` 看起来一切正常——这是"吞吐低但 GPU 空转"最常见的原因。

## 精度：权重 4bit 与 KV 8bit 各换来什么

量化要分两个互相独立的方面看：

- **权重量化**（`--quantization fp8` / AWQ / GPTQ / marlin）：减少"每步要搬的权重"，主要加速**小 batch 的解码**。
- **KV 量化**（`--kv-cache-dtype fp8`）：减少"每 token 的缓存单价"，直接把并发翻倍，代价是长上下文下注意力精度变差。

**同一张 H100 上的三种配置**

> **量化的坑在评测集之外**
>
> 困惑度只涨 0.1 的 4bit 模型，可能在长文档检索、数字比较、代码缩进这类任务上掉得很明显——因为这些恰恰依赖"低比特下最先丢掉的细节"。上线前必须用自己的业务集做 A/B，而不是只看通用 benchmark。

## 启动与编译：-O2、CUDA graph、enforce-eager

这几个参数的共同目的是**把启动时间换成稳态速度**，理解它们的顺序比记名字有用：

- `-O0 / -O1 / -O2 / -O3`：vLLM 的优化档位，默认 **O2**（更多编译区间、FULL_AND_PIECEWISE cudagraph）。O0 启动最快、稳态最慢；O3 目前等同 O2。
- `--enforce-eager`：跳过编译与 CUDA graph 捕获，启动最快——适合开发调试，或者用来测量"启动时间里有几秒是编译"。
- `VLLM_CACHE_ROOT`（默认 `~/.cache/vllm`）：torch.compile 产物在这里复用，可以拷进镜像，重启就不再重编译。
- `--kv-cache-memory`：把上次日志里的精确值填回来，跳过 profiling 与 graph 显存估算。只在"同 GPU、同初始空闲显存"时有效，换了硬件必须去掉重新测。
- SGLang 的 `--cuda-graph-max-bs-decode`：默认 CUDA graph 只覆盖较小的 batch，大 TP 时加到 512/768 能吃到稳态收益，但要同时调低 `--mem-fraction-static` 让出捕获缓冲。
- 多节点 TP 偶发死锁时，`--disable-cuda-graph` 是常用的止血开关（先跑通再优化）。

## 观测：五个指标

调参不能靠猜。把下面五个指标接进监控，90% 的问题都能定位：

| 指标 | 健康值 | 异常含义与动作 |
| --- | --- | --- |
| **KV 池利用率**`token usage` / `gpu_cache_usage_perc` | > 0.9 | < 0.9 且仍有排队 → 容量分得太保守；长期贴着 1.0 → 池不够，开始抢占 |
| **排队请求数**`#queue-req` / `num_requests_waiting` | 短时 > 0 | 长期 > 0 是吞吐不足（加卡 / DP），不是延迟问题 |
| **抢占次数**`preempted` / `retracted_reqs` | ≈ 0 | 持续出现 → 降并发或加容量；每分钟一次属正常波动 |
| **TTFT p99**首 token 延迟 | 按业务定 | 高 → 加大 batch budget、开前缀缓存、减少排队 |
| **ITL p99**相邻 token 间隔 | 聊天 < 100 ms | 抖动大 → 调小 batch budget / chunk size，检查抢占 |

## 排障：症状与对应的参数

排障时先确定问题出在哪一层，再动对应的参数。下面八种症状覆盖了大部分事故：

| 症状与原因 | 动作 |
| --- | --- |
| **启动就 OOM**（还没接请求）——权重加激活超了 | 开 TP=2；SGLang 把 `--mem-fraction-static` 降到 0.80 / 0.70 |
| **prefill 阶段 OOM**（长 prompt 进来时）——一步吞太多 | 降 `--chunked-prefill-size` 到 4096 / 2048；降 batch budget |
| **decode 阶段 OOM**（并发升高时）——KV 池被吃满 | 降 `--max-running-requests` / `max-num-seqs` |
| **TTFT 高但 GPU 不满**——排队多、前缀没命中 | 开前缀缓存 + `--schedule-policy lpm`；加大 budget；DP 扩实例 |
| **ITL 抖动大**——长 prefill 混进了 decode 的步 | 调小 batch budget / chunk size；确认 chunked prefill 开着 |
| **吞吐低、GPU 空转**——CPU 核不足或调度太保守 | 补物理核；`--schedule-conservativeness` 降到 0.3 |
| **同前缀请求反复重算**——缓存没命中 | 稳定内容前置；按块对齐（16 token）；换 lpm 调度 |
| **多节点 TP 卡死**——CUDA graph 与通信冲突 | 先用 `--disable-cuda-graph` 跑通 |

关于 `--schedule-conservativeness`（SGLang 默认 1.0）：它表示调度器的保守程度。调大更保守，不容易 OOM 但并发上不去；调小更激进。判据是日志里的 token usage：低于 0.9 而仍有排队，就调到 0.3；反过来频繁出现 `KV cache pool is full. Retract requests.`，就抬到 1.3。

## 两份可直接使用的启动命令

场景：单张 H100 80GB，Llama-3.1-8B-Instruct，8K 上下文，在线聊天，希望并发高、ITL 稳。每个参数为什么这么设，见命令后面的对照表。

```bash
# vLLM（V1）—— 单卡 H100，Llama-3.1-8B，8K 上下文
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --served-model-name llama8b \
  --max-model-len 8192 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 8192 \
  --enable-prefix-caching \
  --kv-cache-dtype fp8 \
  --port 8000
```

```bash
# SGLang —— 同样的模型与硬件
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --context-length 8192 \
  --mem-fraction-static 0.90 \
  --max-running-requests 128 \
  --chunked-prefill-size 8192 \
  --schedule-policy lpm \
  --enable-metrics \
  --port 30000
```

| 参数 | 为什么这么设 |
| --- | --- |
| `--max-model-len 8192` | 上下文窗口直接决定 KV 单价；开满 128K 会把并发压到个位数 |
| `--max-num-seqs 128` | 并发上限；实际的瓶颈通常是 KV 池，这个数只是护栏 |
| `--max-num-batched-tokens 8192` | 每步预算；ITL 与 TTFT 的取舍点，见上面的抖动表 |
| `--enable-prefix-caching` | V1 默认已开，显式写出来是自文档化，也避免被旧启动脚本覆盖 |
| `--kv-cache-dtype fp8` | KV 单价减半，并发直接翻倍；先在长文任务上验收精度 |
| `--mem-fraction-static 0.90` | 权重 + KV 池占 90%，余 10% 留给激活与 CUDA graph |
| `--chunked-prefill-size 8192` | 单块 prefill 上限，决定 ITL 峰值抖动（约 546 ms） |
| `--schedule-policy lpm` | 最长前缀优先；共享系统提示的场景命中率明显更高 |
| `--enable-metrics` | 不开就没有 `token usage`，前四项观测指标全部失效 |

这两份配置算不上最优，只是一个可以解释的起点：先按上表测出基线，再逐个参数做单变量实验，一次只改一个。记下 TTFT、ITL 和吞吐三个数——这比记住任何"推荐值"都有用。

## 常用参数速查

| 参数 | 管什么 | 调大 / 调小 |
| --- | --- | --- |
| `gpu-memory-utilization` `V` | 显存硬上限；V1 默认 1.0，按实测剩余分 KV | 调低是给别的进程留显存，代价是 KV 池变小 |
| `--mem-fraction-static` `S` | 权重 + KV 池占整卡的比例 | 调大并发更高、激活余量更少；调小更稳 |
| `max-num-seqs` `V` / `--max-running-requests` `S` | 同时在跑的序列上限 | 调大吞吐升、ITL 升；调小延迟稳、吞吐降 |
| `max-num-batched-tokens` `V` | 每步 token 预算 | 调大 TTFT 更好、ITL 更抖；调小反之 |
| `--chunked-prefill-size` `S` | 单块 prefill 大小（−1 关闭） | 调大长 prompt 更快；调小省激活、ITL 更稳 |
| `--schedule-conservativeness` `S` | 调度胆量，默认 1.0 | 调大更保守、不易 OOM；调小更激进、并发更高 |
| `--schedule-policy` `S` | 选请求的顺序，默认 fcfs | 共享前缀多就换 `lpm`（更吃缓存，调度开销更大） |
| `--quantization` `V/S` | 权重精度（fp8 / awq / gptq / marlin） | 越低越省显存与带宽，质量风险越高 |
| `--kv-cache-dtype` `V/S` | KV 精度（fp8_e4m3 / fp8_e5m2） | fp8 让并发翻倍，长文精度略降 |
| `-tp` `V` / `--tp-size` `S` | 张量并行度 | 调大单卡权重更小、通信更贵；单卡放得下时别用 |
| `-dp` `V` / `--dp-size` `S` | 数据并行度（复制模型扩吞吐） | 调大总吞吐升、每份都要全量权重 |
| `--enforce-eager` `V` | 跳过编译与 CUDA graph | 只开给调试：启动快、稳态明显变慢 |
| `--enable-metrics` `S` | 暴露 Prometheus 指标 | 建议常开，否则本章的观测全部无从谈起 |

## 默认值会变，不要背数字

这一章所有默认值都对应 2025 年前后的 vLLM V1 与 SGLang 主干版本。这两套系统迭代很快，**同名参数在不同大版本里语义都可能变**——最典型的就是 `gpu_memory_utilization`：V0 时代它是"占显存的比例，默认 0.92"，V1 改成"可选硬上限，默认 1.0，KV 按实测剩余分配"。你照抄三年前的博客，就会白丢 10% 的 KV 池。

> **查证的正确姿势**
>
> 想知道某个参数在你装的版本里是什么语义，看两个地方：一是 `--help`（它永远对应当前版本）；二是官方仓库里的文档源文件（`docs/configuration/optimization.md` 这类 Markdown），比网页搜索到的二手总结可靠得多。改完参数记得回到观测指标验证，而不是凭默认值推断行为。

> **小结**
>
> 显存只有三块：权重、KV 池和激活，所有容量参数最终都在决定给 KV 池留多少。并发取三个约束里最小的那个——你设的上限、池除以单价、batch 预算，而单价与上下文长度成正比，所以把窗口设小往往比加卡更有效。调度预算决定 TTFT 与 ITL 的取舍，chunked prefill 让两者同批共存，但 budget 太大会让 ITL 出现几百毫秒的抖动。单卡放得下时优先用 DP 扩吞吐，TP 只用来解决"装不下"；量化则要区分权重和 KV cache 这两件事。

> **下一章**
>
> 参数调好，只是让这台机器跑得有效率。可这些数字究竟是怎么变成 GPU 上的指令的？第 13 章往下钻一层：调用链、每步的 kernel、CUDA graph 与多卡通信。

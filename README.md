# 大模型是怎么想出一个字的

一份从零讲到大模型完整推理过程的单文件图文教程，所有"逐层演算"的数字都来自本地真实训练的小模型。

**在线阅读**：[xipinggg.github.io/llm-from-scratch](https://xipinggg.github.io/llm-from-scratch/)（Markdown 版静态站点，27 章）

**离线阅读**：直接双击 `index.html`（单文件、离线、无外部依赖，含 10 个可交互演示）

## 内容结构（27 章，含第 0 章导言、四章附录与一章前沿）

| 章 | 标题 | 内容 |
| --- | --- | --- |
| 00 | 怎么读这份材料 | 四条阅读路线、材料约定与数字的读法 |
| 01 | 全景：一次推理的七个阶段 | 从字符串到概率分布，先看完整流程再拆开 |
| 02 | 必备数学：只有六件事 | 向量、点积、矩阵乘、softmax、交叉熵、梯度下降 |
| 03 | 分词：把文字变成整数 | BPE 如何切分世界，为什么它总被吐槽（含交互） |
| 04 | 嵌入：把整数变成向量 | 查表、相加，以及位置信息为什么必须存在（RoPE 交互） |
| 05 | 注意力：20 个位置怎么互相看 | QKV、打分、softmax、加权求和、四种读法与注意力家族（真数字 + 热力图） |
| 06 | 残差与归一化：让 96 层也能训 | 为什么加回去，为什么先归一化再进子层 |
| 07 | 前馈网络：模型把知识存在哪 | 两层矩阵、一个非线性，占掉三分之二的参数 |
| 08 | 完整前向：从字符到概率 | 20 个 token 穿过 2 层，最后落在 17 个候选上 |
| 09 | 选哪个字：温度、top-k、top-p | 从概率分布里抽样的全部工程细节（采样器交互） |
| 10 | 推理引擎：把 FLOPs 变成毫秒 | prefill/decode、roofline、FlashAttention、量化、投机解码 |
| 11 | KV cache 深潜 | 只追加的不变量与四条优化路线、缓存公式与逐层账本、并发与带宽、MHA/MQA/GQA/MLA 与矩阵吸收、KV 量化、分页、前缀缓存（键/基数树/命中率）与它和 KV 池的七种互动、跨层共享（YOCO/CLA）、稀疏（StreamingLLM/H2O/SnapKV/Quest/InfLLM/NSA）、卸载到内存/磁盘（页粒度、预取、写回策略）、PD 分离与传输、压缩与卸载两个计算器、精度/一致性与安全、2025 前沿、压缩手段叠加计算器 |
| 12 | 服务参数：vLLM 与 SGLang 调参 | 显存预算与 f 值、并发上限、调度预算、前缀缓存、抢占、并行、排障表、可抄命令 |
| 13 | GPU 执行：一次 decode 怎么落到硬件 | 五层调用链、kernel 清单、注意力 kernel 与分页、H2D、流与同步、CUDA graph、all-reduce（交互） |
| 14 | 推理加速：量化、投机解码与约束解码 | 仿射量化与粒度（交互）、SmoothQuant、GPTQ/AWQ、Marlin、投机解码、grammar 掩码 |
| 15 | 长上下文：位置外推、KV 压缩与 1M 的账 | PI/NTK/YaRN、滑窗与混合层、KV 淘汰、1M 的三笔成本、检索 vs 长上下文、评测 |
| 16 | 架构变体：MoE、MLA、Mamba 与 MTP | MoE 路由与负载均衡、MLA 的低秩 KV、线性注意力、多 token 预测、没变的部分 |
| 17 | 训练：从下一个字到对齐人类 | 预训练、缩放定律、SFT、RLHF、DPO、RLVR/GRPO |
| 18 | 推理模型：让模型多想一想 | CoT、测试时计算缩放、自一致性、PRM/MCTS、GRPO 细节、R1 配方、蒸馏 |
| 19 | 从零实现：200 行 numpy | 前向、手写反向、Adam、训练与生成，完整可运行 |
| 20 | Agent：让模型动手做事 | 工具调用约定、ReAct 循环、上下文预算（交互）、记忆分层、失败模式与评测 |
| 21 | Harness：模型外面那一圈代码 | 主循环、工具设计、权限与沙箱、压缩策略、子代理、事件流、与推理层的接口 |
| 22 | 规模地图与常见误解 | 参数量/FLOPs/显存估算、十个常见误解、延伸阅读 |
| 23 | 习题与动手实验 | 十二道题（答案折叠）与五个本机可跑的实验 |
| 24 | 术语表 | 按主题排列的八十个术语，一句话一个 |
| 25 | 源码导读：读一个真实的注意力 kernel | Triton 教学版逐行读在线 softmax，对照 vLLM 生产版多出的分页/GQA/split-K/量化 |
| 26 | 研究前沿：推理引擎正在往哪走 | PD 分离与 KV 传输、KV 作为一等存储、架构-系统协同、投机与 MTP、MoE 大 EP、kernel 与编译、长上下文与 agent、调度与公平；附研究地图与八个可验证空白 |

## 目录结构

```
index.html              # 构建产物，唯一交付文件（约 1.6 MB，含内联 KaTeX 字体）
src/shell.html          # 页面骨架
src/parts/*.html        # 各章正文，按文件名排序拼接
src/app.css             # 设计系统（暖纸浅色 + 深色主题）
src/app.js              # 交互脚本（无依赖）
build/numbers.json      # 小模型导出的真实数字（前向 trace、注意力矩阵、分布）
build/bpe.json          # 中文 BPE 合并表
build/toy_gpt.npz       # 训练好的权重
build/train.log         # 训练日志
tools/toy_gpt.py        # 训练 + 数值梯度校验 + trace 导出（本页数字的来源）
tools/gpt_min.py        # 第 19 章展示的可运行精简版（约 30 秒训练完）
tools/make_bpe.py       # 从语料训练 BPE，产出 build/bpe.json
tools/render_math.js    # 用 KaTeX 把 \(..\) 预渲染成静态 HTML
tools/build.py          # 构建：拼接、编号、内联、生成导航与封面卡片
tools/preview.py        # 单章预览（调试排版用）
```

## 重新构建

```bash
PY=/Users/fengxiping/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
NODE=/Users/fengxiping/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/node/bin/node

$PY tools/toy_gpt.py          # 重新训练小模型并导出 build/numbers.json（可选）
$PY tools/make_bpe.py         # 重新生成 BPE 表（可选）
$PY tools/build.py            # 生成 index.html
```

构建期完成的事情：KaTeX 公式渲染成静态 HTML（运行时不加载 KaTeX JS）、woff2 字体转 base64 内联、章节编号与侧栏导航生成、`build/*.json` 注入 `#page-data`。

## 模型规格

字符级 GPT：`V=17, T=40, d_model=8, layers=2, heads=2, head_dim=4, FFN=16`，
pre-LN + 可学习绝对位置 + GELU + 权重共享输出头，可训练参数 **1608**。
Adam + 余弦学习率训练 12000 步，最终损失 0.182（困惑度 1.2）。

示例 trace：`"the cat sat on the m"` → 下一个字符 `a`，概率 0.9987。

## Markdown 版（分支 `markdown`）

本分支把 27 章转换成纯 Markdown，方便检索、引用和导入其他笔记工具：

- `md/`：逐章 Markdown（`00-hero.md` … `26-research-frontier.md`）+ 目录 `md/README.md`
- `md/book.md`：全部章节的合集
- `tools/html2md.py`：转换器（读 `src/parts/*.html`，输出 `md/`）

重新生成：

```bash
$PY tools/html2md.py
```

转换规则：公式转成 `$…$` / `$$…$$`（右对齐编号写成 `\qquad (5.1)`）、表格转 GFM、演算面板转两列表、`.compare`/`.keyidea`/`.note` 转引用块、代码块保留语言标记；`<svg>` 流程图与交互演示不导出（Markdown 无法承载），对应位置留下说明。

## 在线文档站（分支 `markdown`）

`docs/` 是由 Markdown 生成的静态站点（GitHub Pages 从 `main` 分支的 `/docs` 发布）（每章一页、左侧目录、公式在构建期渲染成静态 HTML），可直接用 GitHub Pages 托管：

```bash
$PY tools/build_site.py     # 读取 md/ 生成 docs/
```

在线地址：https://xipinggg.github.io/llm-from-scratch/（内容与 `docs/` 一致）

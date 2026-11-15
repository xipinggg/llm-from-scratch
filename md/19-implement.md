*第十九章 · 从零实现*

# 从零实现：200 行 numpy

读懂和写出来是两件事。这一章把前面所有章节拼成一份能直接运行的代码：没有 PyTorch，没有自动微分，梯度全部手推。它训练出的模型就是本页所有数字的来源。

## 为什么值得亲手写一遍

用 PyTorch 搭一个 GPT 只要十几行，但那样会错过几件最容易忽略的事：

- **梯度的形状**：反向传播里每个矩阵乘法的维度为什么会变成那样？只有手写一遍才知道 `dW = Xᵀ dY` 是怎么来的；
- **残差与归一化的耦合**：LayerNorm 的反向公式里有两项残差修正，写错一项梯度就全错（作者本人踩过，见下文）；
- **一个 epoch 到底在做什么**：损失、学习率、batch、梯度累积、参数更新，这些词在框架里都是黑盒。

> **五段代码分别对应哪几章**
>
> 语料与初始化（第 3–4 章）→ 基础算子与归一化（第 2、6 章）→ 前向（第 5、7、8 章）→ 反向（链式法则）→ 训练与生成（第 9、11 章）。这份实现对每个张量都保留中间量，所以后面才能导出第 5–8 章那张逐步演算表。

> **与本页数字的关系**
>
> 本页第 5–11 章的数字由 `tools/toy_gpt.py`（同一份代码的完整版，多一个 trace 导出）生成。下面这份 `tools/gpt_min.py` 是它的精简版，算法完全一致，最终损失在 0.16–0.18（随机初始化与数据顺序略有不同）。运行 `python3 gpt_min.py` 约 30 秒。

## 完整代码

先把代码分四段贴出来，每段后面解释关键行。文件在 `tools/gpt_min.py`。

① 模块说明、语料与初始化

```python
"""
GPT-min：一个 200 行以内、只用 numpy 的完整小型语言模型。
前向、手写反向传播、Adam、训练、采样全在这里。

架构：字符级，V=17，T=40，d_model=8，2 层，2 头（每头 4 维），FFN=16
      pre-LN + 可学习绝对位置 + GELU + 权重共享输出头
运行：python3 gpt_min.py           （约 30 秒，损失降到 0.16 附近）
"""
import math
import numpy as np

# ---------------------------------------------------------------- 1. 语料
CORPUS = """\
the cat sat on the mat.
the cat sat on the log.
a cat sat on the mat.
the dog sat on the mat.
the dog sat on the log.
the man sat on the mat.
the man sat on the log.
she sat on the mat.
he sat on the log.
the cat ate the rat.
the rat ate the mat.
the dog ate the rat.
the cat is on the mat.
the dog is on the log.
the man is on the mat.
the mat is on the log.
a man sat on the log.
the cat and the dog sat on the mat.
the cat ran to the man.
the dog ran to the cat.
the man ran to the cat.
i sat on the mat.
we sat on the log.
sit on the mat.
a cat and a dog sat on the log.
the rat ran to the mat.
the man and the cat sat on the log.
she ran to the man.
he ran to the dog.
the dog and the man ate the rat.
on the mat sat a cat.
on the log sat a dog.
the cat sat.
the dog ran.
the man ate.
the rat ran.
the end.
"""

CORPUS = CORPUS.replace("\n", "")
VOCAB = sorted(set(CORPUS))
V = len(VOCAB)
STOI = {c: i for i, c in enumerate(VOCAB)}
ITOS = {i: c for c, i in STOI.items()}
data = np.array([STOI[c] for c in CORPUS], dtype=np.int64)

T, D, L, H = 40, 8, 2, 2
DH = D // H
F = 16
EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)
rng = np.random.default_rng(20241115)

# ---------------------------------------------------------------- 2. 初始化
def init_params(rng):
    def randn(*shape, std=0.08):
        return rng.normal(0.0, std, size=shape)

    p = {"wte": randn(V, D, std=0.3), "wpe": randn(T, D, std=0.05)}
    for l in range(L):
        p[f"ln1g{l}"], p[f"ln1b{l}"] = np.ones(D), np.zeros(D)
        p[f"wq{l}"] = randn(D, D, std=0.35)
        p[f"wk{l}"] = randn(D, D, std=0.35)
        p[f"wv{l}"] = randn(D, D, std=0.35)
        p[f"wo{l}"] = randn(D, D, std=0.25)
        p[f"ln2g{l}"], p[f"ln2b{l}"] = np.ones(D), np.zeros(D)
        p[f"w1{l}"] = randn(D, F, std=0.35)
        p[f"b1{l}"] = np.zeros(F)
        p[f"w2{l}"] = randn(F, D, std=0.25)
        p[f"b2{l}"] = np.zeros(D)
    p["lnfg"], p["lnfb"] = np.ones(D), np.zeros(D)
    return p

# ---------------------------------------------------------------- 3. 基础算子
def layer_norm(x, g, b):
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc ** 2).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + EPS)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd)

def layer_norm_backward(dy, xhat, rstd, g):
    n = xhat.shape[-1]
    dg = (dy * xhat).sum(0)
    db = dy.sum(0)
    dxhat = dy * g
    dvar = (dxhat * xhat).sum(-1, keepdims=True) * (-0.5) * rstd ** 2
    dmu = (-dxhat * rstd).sum(-1, keepdims=True) \
          + dvar * (-2.0 / n) * (xhat / rstd).sum(-1, keepdims=True)
    dx = dxhat * rstd + dvar * 2.0 * (xhat / rstd) / n + dmu / n
    return dx, dg, db

def gelu(x):
    t = np.tanh(GELU_C * (x + 0.044715 * x ** 3))
    return 0.5 * x * (1.0 + t), t

def gelu_backward(dy, x, t):
    dt = 0.5 * x * (1.0 - t ** 2) * GELU_C * (1.0 + 3 * 0.044715 * x ** 2)
    return dy * (0.5 * (1.0 + t) + dt)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def causal_mask(tn):
    return np.triu(np.full((tn, tn), -1e9), k=1)
```

② 前向传播

```python
# ---------------------------------------------------------------- 4. 前向
def forward(p, ids, keep=False):
    B, tn = ids.shape
    x = p["wte"][ids] + p["wpe"][None, :tn, :]
    cache = {"ids": ids, "layers": []}
    for l in range(L):
        lc = {}
        a, (xhat, rstd) = layer_norm(x, p[f"ln1g{l}"], p[f"ln1b{l}"])
        q, k, v = a @ p[f"wq{l}"], a @ p[f"wk{l}"], a @ p[f"wv{l}"]
        lc.update(a=a, xhat=xhat, rstd=rstd, q=q, k=k, v=v)
        outs, atts = [], []
        for h in range(H):
            sl = slice(h * DH, (h + 1) * DH)
            sc = q[:, :, sl] @ k[:, :, sl].transpose(0, 2, 1) / math.sqrt(DH)
            at = softmax(sc + causal_mask(tn)[None], -1)
            outs.append(at @ v[:, :, sl])
            atts.append(at)
        o = np.concatenate(outs, -1) @ p[f"wo{l}"]
        x = x + o
        lc.update(att=np.stack(atts, 1), o=o, x1=x)
        b, (bhat, brstd) = layer_norm(x, p[f"ln2g{l}"], p[f"ln2b{l}"])
        u = b @ p[f"w1{l}"] + p[f"b1{l}"]
        act, gt = gelu(u)
        y = act @ p[f"w2{l}"] + p[f"b2{l}"]
        x = x + y
        lc.update(b=b, bhat=bhat, brstd=brstd, u=u, act=act, gelu_t=gt, y=y, x2=x)
        cache["layers"].append(lc)
    hf, (fh, frstd) = layer_norm(x, p["lnfg"], p["lnfb"])
    logits = hf @ p["wte"].T          # 权重共享：输出头就是 wte
    cache.update(hf=hf, hf_xhat=fh, hf_rstd=frstd)
    return (logits, cache) if keep else logits
```

③ 手写反向传播

```python
# ---------------------------------------------------------------- 5. 反向
def backward(p, cache, dlogits):
    g = {k: np.zeros_like(v) for k, v in p.items()}
    g["wte"] += dlogits.reshape(-1, V).T @ cache["hf"].reshape(-1, D)
    dhf = dlogits @ p["wte"]
    dx, g["lnfg"], g["lnfb"] = layer_norm_backward(
        dhf.reshape(-1, D), cache["hf_xhat"].reshape(-1, D),
        cache["hf_rstd"].reshape(-1, 1), p["lnfg"])
    dx = dx.reshape(cache["hf"].shape)

    for l in reversed(range(L)):
        lc = cache["layers"][l]
        # 前馈支路：y = act @ w2 + b2
        dy = dx
        g[f"w2{l}"] += lc["act"].reshape(-1, F).T @ dy.reshape(-1, D)
        g[f"b2{l}"] += dy.reshape(-1, D).sum(0)
        du = gelu_backward(dy @ p[f"w2{l}"].T, lc["u"], lc["gelu_t"])
        g[f"w1{l}"] += lc["b"].reshape(-1, D).T @ du.reshape(-1, F)
        g[f"b1{l}"] += du.reshape(-1, F).sum(0)
        dln2, g[f"ln2g{l}"], g[f"ln2b{l}"] = layer_norm_backward(
            (du @ p[f"w1{l}"].T).reshape(-1, D), lc["bhat"].reshape(-1, D),
            lc["brstd"].reshape(-1, 1), p[f"ln2g{l}"])
        dx = dy + dln2.reshape(dy.shape)      # 残差：两条支路相加

        # 注意力支路：o = concat(heads) @ wo
        do = dx
        cat = np.concatenate([lc["att"][:, h] @ lc["v"][:, :, h * DH:(h + 1) * DH]
                              for h in range(H)], -1)
        g[f"wo{l}"] += cat.reshape(-1, D).T @ do.reshape(-1, D)
        docat = do @ p[f"wo{l}"].T
        dq, dk, dv = np.zeros_like(lc["q"]), np.zeros_like(lc["k"]), np.zeros_like(lc["v"])
        for h in range(H):
            sl = slice(h * DH, (h + 1) * DH)
            at, douth = lc["att"][:, h], docat[:, :, sl]
            dp = douth @ lc["v"][:, :, sl].transpose(0, 2, 1)
            dv[:, :, sl] = at.transpose(0, 2, 1) @ douth
            ds = at * (dp - (dp * at).sum(-1, keepdims=True))   # softmax 反向
            dq[:, :, sl] = ds @ lc["k"][:, :, sl] / math.sqrt(DH)
            dk[:, :, sl] = ds.transpose(0, 2, 1) @ lc["q"][:, :, sl] / math.sqrt(DH)
        g[f"wq{l}"] += lc["a"].reshape(-1, D).T @ dq.reshape(-1, D)
        g[f"wk{l}"] += lc["a"].reshape(-1, D).T @ dk.reshape(-1, D)
        g[f"wv{l}"] += lc["a"].reshape(-1, D).T @ dv.reshape(-1, D)
        da = dq @ p[f"wq{l}"].T + dk @ p[f"wk{l}"].T + dv @ p[f"wv{l}"].T
        dln1, g[f"ln1g{l}"], g[f"ln1b{l}"] = layer_norm_backward(
            da.reshape(-1, D), lc["xhat"].reshape(-1, D),
            lc["rstd"].reshape(-1, 1), p[f"ln1g{l}"])
        dx = dx + dln1.reshape(da.shape)

    np.add.at(g["wte"], cache["ids"], dx)          # token id 会重复，必须累加
    g["wpe"][:cache["ids"].shape[1]] += dx.sum(0)
    return g
```

④ 训练循环与生成

```python
# ---------------------------------------------------------------- 6. 训练
def train(steps=12000, batch=16, lr=6e-3, warmup=200):
    p = init_params(rng)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    vv = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.99, 1e-8
    losses = []
    for step in range(1, steps + 1):
        # 随机截一段语料，输入 ids，标签是 ids 右移一位
        ix = rng.integers(0, len(data) - T - 1, size=batch)
        ids = np.stack([data[i:i + T] for i in ix])
        targets = np.stack([data[i + 1:i + 1 + T] for i in ix])

        logits, cache = forward(p, ids, keep=True)
        lg = logits - logits.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        loss = -np.take_along_axis(logp, targets[..., None], -1).mean()
        losses.append(float(loss))

        # 交叉熵对 logits 的梯度就是 p - 1[y]
        dlogits = np.exp(logp)
        np.put_along_axis(dlogits, targets[..., None],
                          np.take_along_axis(dlogits, targets[..., None], -1) - 1.0, -1)
        dlogits /= ids.size
        grads = backward(p, cache, dlogits)

        lr_t = lr * min(1.0, step / warmup) * \
               (0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))) * 0.9 + 0.1)
        for k in p:                                   # Adam
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            vv[k] = b2 * vv[k] + (1 - b2) * grads[k] ** 2
            mhat = m[k] / (1 - b1 ** step)
            vhat = vv[k] / (1 - b2 ** step)
            p[k] -= lr_t * mhat / (np.sqrt(vhat) + eps)
        if step % 250 == 0 or step == steps:
            print(f"step {step:5d}  loss {np.mean(losses[-250:]):.4f}", flush=True)
    return p

# ---------------------------------------------------------------- 7. 生成
def sample(p, text, n=60, temp=0.8):
    s = text
    for _ in range(n):
        ids = np.array([[STOI[c] for c in s[-T:]]])
        lg = forward(p, ids)[0, -1] / temp
        nxt = int(rng.choice(V, p=softmax(lg)))
        s += ITOS[nxt]
        if s.endswith(". ") and len(s) > len(text) + 8:
            break
    return s

if __name__ == "__main__":
    p = train()
    for prompt in ["the cat ", "the dog sat on the ", "a man "]:
        print(repr(sample(p, prompt)))
```

## 三段最值得盯住的代码

**第一，权重共享输出头。**`logits = hf @ p["wte"].T`。输出层没有独立参数：token 嵌入矩阵转置一下就当分类器用。这样 17×8 的矩阵既是"输入的查表"又是"输出的打分表"，参数省一半（GPT-2、Llama 都这么做）。代价是 `wte` 的梯度来自两处——反向开头那份来自 logits，结尾 `np.add.at` 那份来自嵌入——必须相加。

**第二，softmax 的反向。**正向是 `at = softmax(sc)`，反向只有一行：

$$
\frac{\partial L}{\partial s_i} = a_i\Big(\frac{\partial L}{\partial a_i} - \sum_j \frac{\partial L}{\partial a_j}a_j\Big)
    19.1
$$

写成代码就是 `ds = at * (dp - (dp * at).sum(-1))`。注意括号里那项必须沿**同一行**求和（`axis=-1`），因为 softmax 是在"这一行所有 key 的分数"之间做的归一化。

**第三，LayerNorm 的反向。**正向是 $\hat x = (x-\mu)/\sigma$，但 $x$ 同时影响 $\mu$ 和 $\sigma$，所以反向有三条路径汇合：

$$
dx = \frac{1}{\sigma}\Big( d\hat x - \mathrm{mean}(d\hat x) - \hat x \cdot \mathrm{mean}(d\hat x \odot \hat x)\Big)
    19.2
$$

代码里没有直接写这一行，而是用 `dvar` 与 `dmu` 两个中间量把它展开——两种写法等价，后者少做两次归约。作者第一版把 `dvar` 的系数写成了 `rstd**3`（正确是 `rstd**2`），结果训练损失卡在 3.2 不动；用数值梯度一查，只有这一项相对误差 17，其余全是 1e-7 量级。

## 怎么确认自己写对了：数值梯度

手写反向最大的风险是"看起来能训、其实是错的"。最可靠的验收方式是数值梯度：对每个参数扰动一点点，用损失的变化率逼近导数。

$$
\frac{\partial L}{\partial w} \approx \frac{L(w+\delta) - L(w-\delta)}{2\delta}
    19.3
$$

取 $\delta=10^{-5}$，随机抽查每个参数的 12 个分量，计算相对误差。健康的值应该在 $10^{-7}$ 到 $10^{-6}$ 之间；如果某个参数到 $10^{-2}$ 以上，说明那一项的链式法则写错了。`toy_gpt.py` 里的实测结果：

**梯度校验（tools/toy_gpt.py 的 grad_check）**

| 项 | 值 |
| --- | --- |
| 抽查方式 | 每个参数矩阵随机抽 12 个分量，中心差分 δ=1e-5 |
| 最差相对误差 | 3.01e-07 （修复 LayerNorm bug 之前是 17.6） |
| 结论 | 解析梯度与数值梯度在浮点精度内一致 |

> **三个真实踩过的坑**
>
> **1. LayerNorm 的 dvar 系数。**推导时容易把 $1/\sigma^3$ 和 $1/\sigma^2$ 搞混，写错后损失不降但也不报错。
> **2. 重复 token 的嵌入梯度。**一句话里同一个字符出现多次时，`g["wte"][ids] += dx` 只会保留最后一次写入（NumPy 花式索引不累加），必须用 `np.add.at`。
> **3. 因果掩码的位置。**掩码要在 softmax **之前**加（加 -1e9 使其概率趋近 0），加在之后等于没加。

## 跑起来是什么样

训练 12000 步、batch 16、学习率 6e-3 余弦衰减，损失从 3.5 降到 0.16 左右。温度 0.8 采样几个开头：

```text
step   250  loss 2.1043
step  2000  loss 0.3494
step  6000  loss 0.1873
step 12000  loss 0.1600

"the cat "                → "the cat sat on the log."
"the dog sat on the "     → "the dog sat on the mat."
"a man "                  → "a man sat on the log."
```

它没有"理解"任何东西，只是把 775 个字符的统计规律压进了 1608 个参数里。但这套机制——查表、注意力、残差、归一化、softmax、交叉熵、反向传播——和你手机里那个千亿参数的模型完全同构，只是规模差了八个数量级。

## 从 200 行到 2000 行：还缺什么

| 能力 | 玩具版的做法 | 生产版的做法 |
| --- | --- | --- |
| 词表 | 17 个字符 | 10 万级 BPE，含多语言与代码 |
| 位置编码 | 可学习绝对位置 | RoPE（第 4 章） |
| 注意力 | MHA，2 头 | GQA/MQA + FlashAttention（第 10 章） |
| 归一化 | LayerNorm | RMSNorm，无 bias |
| 前馈 | GELU MLP | SwiGLU + MoE（第 7 章） |
| 精度 | float64 | BF16 前向 + FP32 主权重 |
| 并行 | 无 | DP/TP/PP/ZeRO（第 17 章） |
| 推理 | 每次重算全序列 | KV cache + 连续批处理（第 10 章） |

> **小结**
>
> 一个语言模型拆开来看就是"查表 → 若干次（注意力 → 前馈）→ 打分"，前向和反向都能手写；手写的反向必须用数值梯度验收，否则错了也看不出来。规模改变的是工程，不是原理——第 22 章把规模带来的现象和常见误解收个尾。

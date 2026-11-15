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

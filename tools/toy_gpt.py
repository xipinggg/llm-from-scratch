"""
Toy GPT used to produce every number shown in the teaching page.

Architecture (deliberately tiny so every vector fits on screen):
  vocab V=~18 chars, block size T=40, d_model=8, layers=2, heads=2 (head_dim=4), FFN=16
  pre-LN, learned absolute position embeddings, GELU(tanh approx), tied output head.

Everything (forward, manual backprop, Adam) is numpy only, so the page can show the
complete implementation. Gradients are checked against finite differences.
"""
import json
import math
import numpy as np

RNG = np.random.default_rng(20241115)

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

T = 40
D = 8
L = 2
H = 2
DH = D // H
F = 16
EPS = 1e-5
GELU_C = math.sqrt(2.0 / math.pi)

data = np.array([STOI[c] for c in CORPUS], dtype=np.int64)


def init_params(rng):
    def randn(*shape, std=0.08):
        return rng.normal(0.0, std, size=shape)

    p = {
        "wte": randn(V, D, std=0.3),          # token embedding
        "wpe": randn(T, D, std=0.05),         # position embedding
    }
    for l in range(L):
        p[f"ln1g{l}"] = np.ones(D)
        p[f"ln1b{l}"] = np.zeros(D)
        p[f"wq{l}"] = randn(D, D, std=0.35)
        p[f"wk{l}"] = randn(D, D, std=0.35)
        p[f"wv{l}"] = randn(D, D, std=0.35)
        p[f"wo{l}"] = randn(D, D, std=0.25)
        p[f"ln2g{l}"] = np.ones(D)
        p[f"ln2b{l}"] = np.zeros(D)
        p[f"w1{l}"] = randn(D, F, std=0.35)
        p[f"b1{l}"] = np.zeros(F)
        p[f"w2{l}"] = randn(F, D, std=0.25)
        p[f"b2{l}"] = np.zeros(D)
    p["lnfg"] = np.ones(D)
    p["lnfb"] = np.zeros(D)
    return p


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
    dmu = (-dxhat * rstd).sum(-1, keepdims=True) + dvar * (-2.0 / n) * (xhat / rstd).sum(-1, keepdims=True)
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


def causal_mask(Tn):
    return np.triu(np.full((Tn, Tn), -1e9), k=1)


def forward(p, ids, keep=False):
    """ids: (B,T) int. Returns logits (B,T,V) and cache when keep=True."""
    B, Tn = ids.shape
    x = p["wte"][ids] + p["wpe"][None, :Tn, :]
    cache = {"ids": ids, "x0": x, "layers": []}
    for l in range(L):
        lc = {}
        a, (xhat, rstd) = layer_norm(x, p[f"ln1g{l}"], p[f"ln1b{l}"])
        lc.update(a=a, xhat=xhat, rstd=rstd)
        q = a @ p[f"wq{l}"]
        k = a @ p[f"wk{l}"]
        v = a @ p[f"wv{l}"]
        lc.update(q=q, k=k, v=v)
        outs, atts = [], []
        for h in range(H):
            sl = slice(h * DH, (h + 1) * DH)
            qh, kh, vh = q[:, :, sl], k[:, :, sl], v[:, :, sl]
            sc = qh @ kh.transpose(0, 2, 1) / math.sqrt(DH)
            sc = sc + causal_mask(Tn)[None]
            at = softmax(sc, -1)
            outs.append(at @ vh)
            atts.append(at)
        att = np.stack(atts, 1)                        # (B,H,T,T)
        o = np.concatenate(outs, -1) @ p[f"wo{l}"]
        lc.update(att=att, o=o)
        x = x + o
        lc["x1"] = x
        b, (bhat, brstd) = layer_norm(x, p[f"ln2g{l}"], p[f"ln2b{l}"])
        u = b @ p[f"w1{l}"] + p[f"b1{l}"]
        act, gt = gelu(u)
        y = act @ p[f"w2{l}"] + p[f"b2{l}"]
        lc.update(b=b, bhat=bhat, brstd=brstd, u=u, act=act, gelu_t=gt, y=y)
        x = x + y
        lc["x2"] = x
        cache["layers"].append(lc)
    hf, (fh, frstd) = layer_norm(x, p["lnfg"], p["lnfb"])
    logits = hf @ p["wte"].T
    cache.update(hf=hf, hf_xhat=fh, hf_rstd=frstd, logits=logits)
    return (logits, cache) if keep else logits


def backward(p, cache, dlogits):
    g = {k: np.zeros_like(v) for k, v in p.items()}
    hf, xhat_f, rstd_f = cache["hf"], cache["hf_xhat"], cache["hf_rstd"]
    g["wte"] += dlogits.reshape(-1, V).T @ hf.reshape(-1, D)
    dhf = dlogits @ p["wte"]
    dx, g["lnfg"], g["lnfb"] = layer_norm_backward(dhf.reshape(-1, D), xhat_f.reshape(-1, D),
                                                   rstd_f.reshape(-1, 1), p["lnfg"])
    dx = dx.reshape(hf.shape)
    for l in reversed(range(L)):
        lc = cache["layers"][l]
        # --- FFN branch
        dy = dx
        g[f"w2{l}"] += lc["act"].reshape(-1, F).T @ dy.reshape(-1, D)
        g[f"b2{l}"] += dy.reshape(-1, D).sum(0)
        dact = dy @ p[f"w2{l}"].T
        du = gelu_backward(dact, lc["u"], lc["gelu_t"])
        g[f"w1{l}"] += lc["b"].reshape(-1, D).T @ du.reshape(-1, F)
        g[f"b1{l}"] += du.reshape(-1, F).sum(0)
        db = du @ p[f"w1{l}"].T
        dln2, g[f"ln2g{l}"], g[f"ln2b{l}"] = layer_norm_backward(db.reshape(-1, D), lc["bhat"].reshape(-1, D),
                                                                lc["brstd"].reshape(-1, 1), p[f"ln2g{l}"])
        dx = dy + dln2.reshape(dy.shape)
        # --- attention branch
        do = dx
        g[f"wo{l}"] += np.concatenate([lc["att"][:, h] @ lc["v"][:, :, h * DH:(h + 1) * DH] for h in range(H)], -1).reshape(-1, H * DH).T @ do.reshape(-1, D)
        docat = do @ p[f"wo{l}"].T
        dq = np.zeros_like(lc["q"]); dk = np.zeros_like(lc["k"]); dv = np.zeros_like(lc["v"])
        for h in range(H):
            sl = slice(h * DH, (h + 1) * DH)
            at = lc["att"][:, h]
            douth = docat[:, :, sl]
            dp = douth @ lc["v"][:, :, sl].transpose(0, 2, 1)
            dv[:, :, sl] = at.transpose(0, 2, 1) @ douth
            ds = at * (dp - (dp * at).sum(-1, keepdims=True))
            dq[:, :, sl] = ds @ lc["k"][:, :, sl] / math.sqrt(DH)
            dk[:, :, sl] = ds.transpose(0, 2, 1) @ lc["q"][:, :, sl] / math.sqrt(DH)
        g[f"wq{l}"] += lc["a"].reshape(-1, D).T @ dq.reshape(-1, D)
        g[f"wk{l}"] += lc["a"].reshape(-1, D).T @ dk.reshape(-1, D)
        g[f"wv{l}"] += lc["a"].reshape(-1, D).T @ dv.reshape(-1, D)
        da = dq @ p[f"wq{l}"].T + dk @ p[f"wk{l}"].T + dv @ p[f"wv{l}"].T
        dln1, g[f"ln1g{l}"], g[f"ln1b{l}"] = layer_norm_backward(da.reshape(-1, D), lc["xhat"].reshape(-1, D),
                                                                lc["rstd"].reshape(-1, 1), p[f"ln1g{l}"])
        dx = dx + dln1.reshape(da.shape)
    np.add.at(g["wte"], cache["ids"], dx)
    g["wpe"][:cache["ids"].shape[1]] += dx.sum(0)
    return g


def grad_check(p, ids, targets, n=12, delta=1e-5):
    def loss_of(pp):
        lg = forward(pp, ids)
        lg = lg - lg.max(-1, keepdims=True)
        lp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        return -np.take_along_axis(lp, targets[..., None], -1).mean()

    logits, cache = forward(p, ids, keep=True)
    lg = logits - logits.max(-1, keepdims=True)
    probs = np.exp(lg - np.log(np.exp(lg).sum(-1, keepdims=True)))
    dlogits = probs.copy()
    np.put_along_axis(dlogits, targets[..., None],
                      np.take_along_axis(dlogits, targets[..., None], -1) - 1.0, -1)
    dlogits /= ids.size
    grads = backward(p, cache, dlogits)
    worst = 0.0
    keys = [k for k in p if k.startswith(("w", "ln", "b"))]
    for k in keys:
        flat = p[k].reshape(-1)
        for idx in RNG.choice(flat.size, size=min(n, flat.size), replace=False):
            old = flat[idx]
            flat[idx] = old + delta; up = loss_of(p)
            flat[idx] = old - delta; dn = loss_of(p)
            flat[idx] = old
            num = (up - dn) / (2 * delta)
            ana = grads[k].reshape(-1)[idx]
            err = abs(num - ana) / max(1e-8, abs(num) + abs(ana))
            worst = max(worst, err)
    return worst


def train(steps=12000, batch=16, lr=6e-3, warmup=200):
    p = init_params(RNG)
    m = {k: np.zeros_like(v) for k, v in p.items()}
    vv = {k: np.zeros_like(v) for k, v in p.items()}
    b1, b2, eps = 0.9, 0.99, 1e-8
    losses = []
    for step in range(1, steps + 1):
        ix = RNG.integers(0, len(data) - T - 1, size=batch)
        ids = np.stack([data[i:i + T] for i in ix])
        targets = np.stack([data[i + 1:i + 1 + T] for i in ix])
        logits, cache = forward(p, ids, keep=True)
        lg = logits - logits.max(-1, keepdims=True)
        logp = lg - np.log(np.exp(lg).sum(-1, keepdims=True))
        loss = -np.take_along_axis(logp, targets[..., None], -1).mean()
        losses.append(float(loss))
        probs = np.exp(logp)
        dlogits = probs.copy()
        np.put_along_axis(dlogits, targets[..., None],
                          np.take_along_axis(dlogits, targets[..., None], -1) - 1.0, -1)
        dlogits /= ids.size
        grads = backward(p, cache, dlogits)
        lr_t = lr * min(1.0, step / warmup) * (0.5 * (1 + math.cos(math.pi * min(1.0, step / steps))) * 0.9 + 0.1)
        for k in p:
            m[k] = b1 * m[k] + (1 - b1) * grads[k]
            vv[k] = b2 * vv[k] + (1 - b2) * grads[k] ** 2
            mhat = m[k] / (1 - b1 ** step)
            vhat = vv[k] / (1 - b2 ** step)
            p[k] -= lr_t * mhat / (np.sqrt(vhat) + eps)
        if step % 250 == 0 or step == steps:
            print(f"step {step:5d}  loss {np.mean(losses[-250:]):.4f}", flush=True)
    return p, losses


def predict(p, text, temp=1.0):
    ids = np.array([[STOI[c] for c in text[-T:]]])
    lg = forward(p, ids)[0, -1] / temp
    pr = softmax(lg)
    return lg, pr


def sample(p, text, n=60, temp=0.8, rng=RNG):
    s = text
    for _ in range(n):
        _, pr = predict(p, s, temp)
        nxt = int(rng.choice(V, p=pr))
        s += ITOS[nxt]
        if s.endswith(". ") and len(s) > len(text) + 8:
            break
    return s


def trace(p, text):
    ids = np.array([[STOI[c] for c in text]])
    logits, c = forward(p, ids, keep=True)
    tn = ids.shape[1]
    out = {
        "text": text,
        "tokens": list(text),
        "ids": ids[0].tolist(),
        "attn": [[np.round(c["layers"][l]["att"][0, h], 3).tolist() for h in range(H)] for l in range(L)],
        "layers": [],
        "final": {
            "hf": np.round(c["hf"][0, -1], 4).tolist(),
            "hf_pre": np.round(c["hf_xhat"][0, -1], 4).tolist(),
            "logits": np.round(logits[0, -1], 4).tolist(),
            "probs": np.round(softmax(logits[0, -1]), 5).tolist(),
        },
    }
    for l in range(L):
        lc = c["layers"][l]
        x_in = c["x0"][0, -1] if l == 0 else c["layers"][l - 1]["x2"][0, -1]
        xhat = lc["xhat"][0, -1]
        mu = x_in.mean()
        var = ((x_in - mu) ** 2).mean()
        q = lc["q"][0, -1]; k = lc["k"][0]; v = lc["v"][0]
        if l == 0:
            x_in = c["x0"][0, -1]
        out["layers"].append({
            "x_in": np.round(x_in, 4).tolist(),
            "mu": round(float(mu), 4),
            "var": round(float(var), 5),
            "sigma": round(float(math.sqrt(var + EPS)), 4),
            "xhat": np.round(xhat, 4).tolist(),
            "q": np.round(q, 4).tolist(),
            "head_q": [np.round(q[h * DH:(h + 1) * DH], 4).tolist() for h in range(H)],
            "head_k": [np.round(k[:, h * DH:(h + 1) * DH], 4).tolist() for h in range(H)],
            "head_v": [np.round(v[:, h * DH:(h + 1) * DH], 4).tolist() for h in range(H)],
            "scores": [np.round((q[h * DH:(h + 1) * DH] @ k[:, h * DH:(h + 1) * DH].T) / math.sqrt(DH), 3).tolist() for h in range(H)],
            "weights": [np.round(lc["att"][0, h, -1], 4).tolist() for h in range(H)],
            "head_out": [np.round((lc["att"][0, h, -1] @ v[:, h * DH:(h + 1) * DH]), 4).tolist() for h in range(H)],
            "attn_concat": np.round(np.concatenate([lc["att"][0, h, -1] @ v[:, h * DH:(h + 1) * DH] for h in range(H)]), 4).tolist(),
            "o": np.round(lc["o"][0, -1], 4).tolist(),
            "x1": np.round(lc["x1"][0, -1], 4).tolist(),
            "b": np.round(lc["b"][0, -1], 4).tolist(),
            "u": np.round(lc["u"][0, -1], 4).tolist(),
            "act": np.round(lc["act"][0, -1], 4).tolist(),
            "y": np.round(lc["y"][0, -1], 4).tolist(),
            "x2": np.round(lc["x2"][0, -1], 4).tolist(),
        })
    return out


def main():
    print("vocab:", "".join(VOCAB).replace(" ", "␣"), "V =", V)
    print("corpus chars:", len(CORPUS), " distinct:", V)
    p = init_params(RNG)
    ids = np.array([[STOI[c] for c in "the cat sat on the ma"]])
    tg = np.array([[STOI[c] for c in "he cat sat on the ma" + "t"]])
    err = grad_check(p, ids, tg)
    print(f"gradient check: worst relative error = {err:.2e}")

    p, losses = train()
    print("\nfinal train loss", round(float(np.mean(losses[-200:])), 4),
          " ppl", round(float(np.exp(np.mean(losses[-200:]))), 3))
    for ctx in ["the ", "the cat ", "the cat sat on the ", "a "]:
        lg, pr = predict(p, ctx)
        top = np.argsort(-pr)[:4]
        print(ctx, "->", [(ITOS[i], round(float(pr[i]), 3)) for i in top])
    print("\nsample:", repr(sample(p, "the cat ", 60, 0.8)))

    ctx = "the cat sat on the m"
    tr = trace(p, ctx)
    top = np.argsort(-np.array(tr["final"]["probs"]))[:4]
    print("trace ctx:", ctx, "-> next:", [(ITOS[i], tr["final"]["probs"][i]) for i in top])

    payload = {
        "vocab": VOCAB, "V": V, "T": T, "D": D, "L": L, "H": H, "DH": DH, "F": F,
        "corpus": CORPUS, "corpus_len": len(CORPUS),
        "losses": [round(l, 4) for l in losses[::20]],
        "final_loss": round(float(np.mean(losses[-200:])), 4),
        "wte": np.round(p["wte"], 4).tolist(),
        "wpe": np.round(p["wpe"][:T], 4).tolist(),
        "trace": tr,
        "dists": {},
        "grad_err": f"{err:.1e}",
    }
    for ctx2 in ["the ", "the cat ", "the cat sat on the ", "a "]:
        lg, pr = predict(p, ctx2)
        payload["dists"][ctx2] = {
            "logits": np.round(lg, 4).tolist(),
            "probs": np.round(pr, 6).tolist(),
        }
    np.savez("/Users/fengxiping/projects/dsh/llm/build/toy_gpt.npz", **{k: v for k, v in p.items()})
    with open("/Users/fengxiping/projects/dsh/llm/build/numbers.json", "w") as f:
        json.dump(payload, f, ensure_ascii=False)
    print("wrote build/numbers.json")


if __name__ == "__main__":
    main()

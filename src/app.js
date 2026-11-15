/* ============================================================
   交互脚本 · 无依赖，直接操作 DOM
   ============================================================ */
(function () {
  "use strict";

  const $ = (s, el) => (el || document).querySelector(s);
  const $$ = (s, el) => Array.from((el || document).querySelectorAll(s));
  const DATA = JSON.parse($("#page-data").textContent || "{}");
  const LLM = (DATA.llm || {});
  const fmt = (x, n) => Number(x).toFixed(n === undefined ? 4 : n);
  const pct = (x) => (100 * x).toFixed(1) + "%";
  const showChar = (c) => (c === " " ? "␣" : c === "\n" ? "↵" : c);

  /* ---------- 主题 ---------- */
  const themeBtn = $("#theme-btn");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    try { localStorage.setItem("llm-theme", t); } catch (e) {}
  }
  (function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem("llm-theme"); } catch (e) {}
    if (!saved) saved = window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    applyTheme(saved);
  })();
  if (themeBtn) themeBtn.addEventListener("click", () => {
    applyTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    $$("canvas.rope-canvas").forEach(drawRope);
  });

  /* ---------- 进度条 ---------- */
  const bar = $("#progress"), num = $("#progress-num");
  function onScroll() {
    const h = document.documentElement;
    const max = h.scrollHeight - h.clientHeight;
    const p = max > 0 ? Math.min(1, h.scrollTop / max) : 0;
    if (bar) bar.style.width = (p * 100).toFixed(2) + "%";
    if (num) num.textContent = Math.round(p * 100) + "%";
  }
  document.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  /* ---------- 移动端目录 ---------- */
  const menuBtn = $("#menu-btn"), sidebar = $("#sidebar");
  if (menuBtn) menuBtn.addEventListener("click", () => document.body.classList.toggle("nav-open"));
  if (sidebar) sidebar.addEventListener("click", (e) => {
    if (e.target.closest("a")) document.body.classList.remove("nav-open");
  });
  document.addEventListener("click", (e) => {
    if (document.body.classList.contains("nav-open") && !e.target.closest("#sidebar") && !e.target.closest("#menu-btn"))
      document.body.classList.remove("nav-open");
  });

  /* ---------- 滚动高亮 ---------- */
  const links = new Map();
  $$("#toc a").forEach((a) => {
    const id = a.getAttribute("href").slice(1);
    if (id) links.set(id, a);
  });
  const targets = $$("section.chapter, section.chapter h3[id]");
  let current = null;
  function setActive(id) {
    if (id === current) return;
    current = id;
    $$("#toc a.active").forEach((a) => a.classList.remove("active"));
    const a = links.get(id);
    if (!a) return;
    a.classList.add("active");
    $$("#toc li.open").forEach((li) => li.classList.remove("open"));
    const li = a.closest("li");
    if (li && li.querySelector(".subs")) li.classList.add("open");
    const nav = a.closest("li").parentElement;
    if (nav && sidebar) {
      const r = a.getBoundingClientRect(), sr = sidebar.getBoundingClientRect();
      if (r.top < sr.top + 60 || r.bottom > sr.bottom - 40) a.scrollIntoView({ block: "center" });
    }
  }
  const spy = new IntersectionObserver((entries) => {
    let best = null;
    for (const e of entries) {
      if (!e.isIntersecting) continue;
      if (!best || e.boundingClientRect.top < best.boundingClientRect.top) best = e;
    }
    if (best) setActive(best.target.id);
  }, { rootMargin: "-72px 0px -62% 0px", threshold: [0, 1] });
  targets.forEach((t) => spy.observe(t));
  window.addEventListener("hashchange", () => {
    const id = location.hash.slice(1);
    if (id) setActive(id);
  });

  /* ---------- 代码复制 ---------- */
  $$("pre.code").forEach((pre) => {
    const btn = document.createElement("button");
    btn.className = "copy";
    btn.type = "button";
    btn.textContent = "复制";
    btn.addEventListener("click", () => {
      const code = pre.querySelector("code");
      const txt = (code ? code.innerText : pre.innerText);
      const done = () => { btn.textContent = "已复制"; setTimeout(() => (btn.textContent = "复制"), 1200); };
      if (navigator.clipboard) navigator.clipboard.writeText(txt).then(done, done);
      else done();
    });
    pre.appendChild(btn);
  });

  /* ---------- 工具：softmax ---------- */
  function softmax(logits, T) {
    const t = T === undefined ? 1 : Math.max(1e-3, T);
    const m = Math.max.apply(null, logits);
    const ex = logits.map((z) => Math.exp((z - m) / t));
    const s = ex.reduce((a, b) => a + b, 0);
    return ex.map((v) => v / s);
  }

  /* ============================================================
     演示 1 · 采样器
     ============================================================ */
  (function sampler() {
    const box = $('[data-demo="sampler"]');
    if (!box || !LLM.dists) return;
    const ctxSel = $("#sm-ctx"), tIn = $("#sm-t"), kIn = $("#sm-k"), pIn = $("#sm-p");
    const tOut = $("#sm-t-out"), kOut = $("#sm-k-out"), pOut = $("#sm-p-out");
    const barsEl = $("#sm-bars"), note = $("#sm-note");
    const drawBtn = $("#sm-draw"), resetBtn = $("#sm-reset");
    const vocab = LLM.vocab || [];
    let picked = null;

    const ctxs = Object.keys(LLM.dists);
    ctxSel.innerHTML = "";
    ctxs.forEach((c) => {
      const o = document.createElement("option");
      o.value = c;
      o.textContent = "「" + c.replace(/ /g, "␣") + "」";
      ctxSel.appendChild(o);
    });

    function dist() {
      const base = LLM.dists[ctxSel.value] || LLM.dists[ctxs[0]];
      const T = parseFloat(tIn.value);
      const k = parseInt(kIn.value, 10);
      const p = parseFloat(pIn.value);
      let probs = softmax(base.logits, T);
      const order = probs.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
      const keep = new Set();
      let acc = 0;
      for (let j = 0; j < order.length; j++) {
        if (j >= k) break;
        keep.add(order[j][1]);
        acc += order[j][0];
        if (acc >= p) break;
      }
      const cut = probs.map((v, i) => (keep.has(i) ? v : 0));
      const s = cut.reduce((a, b) => a + b, 0) || 1;
      return { raw: probs, probs: cut.map((v) => v / s), keep, T, k, p };
    }

    function render(hl) {
      const d = dist();
      const vocabIdx = d.probs.map((v, i) => i).sort((a, b) => d.probs[b] - d.probs[a]);
      const max = Math.max.apply(null, d.probs.concat([1e-9]));
      barsEl.innerHTML = "";
      const shown = vocabIdx.slice(0, 8);
      const rest = vocabIdx.slice(8);
      shown.forEach((i) => {
        const w = d.probs[i];
        const row = document.createElement("div");
        row.className = "bar-row" + (d.keep.has(i) ? "" : " cut") + (hl === i ? " pick" : "");
        row.innerHTML =
          '<div class="tk">' + showChar(vocab[i]) + "</div>" +
          '<div class="track"><div class="fill" style="width:' + (100 * w / max).toFixed(2) + '%"></div></div>' +
          '<div class="pc">' + (w >= 0.00005 ? pct(w) : "0%") + "</div>";
        barsEl.appendChild(row);
      });
      if (rest.length) {
        const sum = rest.reduce((a, i) => a + d.probs[i], 0);
        const row = document.createElement("div");
        row.className = "bar-row rest" + (rest.indexOf(hl) >= 0 ? " pick" : "");
        row.innerHTML =
          '<div class="tk">…</div>' +
          '<div class="track"><div class="fill" style="width:' + (100 * sum / max).toFixed(2) + '%"></div></div>' +
          '<div class="pc">' + rest.length + " 个 " + pct(sum) + "</div>";
        barsEl.appendChild(row);
      }
      const kept = d.probs.filter((v) => v > 0).length;
      const logitsKept = d.raw.map((v, i) => (d.keep.has(i) ? v : null)).filter((v) => v !== null);
      let H = 0;
      d.probs.forEach((v) => { if (v > 0) H -= v * Math.log(v); });
      note.innerHTML = "有效候选 <b>" + kept + "</b> 个（候选上限 " + d.k + "，累计概率阈值 " + fmt(d.p, 2) +
        "）；温度 " + fmt(d.T, 2) + "；分布熵 " + fmt(H, 3) + " nat ⇒ 困惑度 " + fmt(Math.exp(H), 2) +
        "。" + (hl !== null && hl !== undefined ? " 这次抽到 <b>「" + showChar(vocab[hl]) + "」</b>。" : "");
    }

    function draw() {
      const d = dist();
      let r = Math.random(), acc = 0, pick = d.probs.length - 1;
      for (let i = 0; i < d.probs.length; i++) {
        acc += d.probs[i];
        if (r <= acc) { pick = i; break; }
      }
      picked = pick;
      render(pick);
    }

    if (ctxSel) ctxSel.addEventListener("change", () => { picked = null; render(); });
    [tIn, kIn, pIn].forEach((el) => el && el.addEventListener("input", () => { picked = null; render(); }));
    if (drawBtn) drawBtn.addEventListener("click", draw);
    if (resetBtn) resetBtn.addEventListener("click", () => {
      tIn.value = 1; kIn.value = 17; pIn.value = 1; picked = null; sync(); render();
    });

    function sync() {
      if (tOut) tOut.textContent = fmt(tIn.value, 2);
      if (kOut) kOut.textContent = kIn.value >= 17 ? "不限" : kIn.value;
      if (pOut) pOut.textContent = fmt(pIn.value, 2);
    }
    [tIn, kIn, pIn].forEach((el) => el && el.addEventListener("input", sync));
    sync();
    render();
  })();

  /* ============================================================
     演示 2 · 注意力热力图
     ============================================================ */
  (function heat() {
    const box = $('[data-demo="heat"]');
    if (!box || !LLM.trace) return;
    const grid = $("#heat-grid"), tip = $("#heat-tip");
    const layerSeg = $("#heat-layer"), headSeg = $("#heat-head");
    const tokens = LLM.trace.tokens || [];
    const attn = LLM.trace.attn;
    let L = 0, H = 0, userPicked = false;

    function draw() {
      const M = attn[L][H];
      const N = M.length;
      const cols = "22px repeat(" + N + ", 26px)";
      grid.style.gridTemplateColumns = cols;
      grid.innerHTML = "";
      const corner = document.createElement("div");
      corner.className = "corner";
      grid.appendChild(corner);
      for (let j = 0; j < N; j++) {
        const v = document.createElement("div");
        v.className = "vlab";
        v.textContent = showChar(tokens[j] || "?");
        v.title = "键（被看的位置）" + j;
        grid.appendChild(v);
      }
      for (let i = 0; i < N; i++) {
        const h = document.createElement("div");
        h.className = "hlab";
        h.textContent = showChar(tokens[i] || "?");
        h.title = "查询（正在生成的位置）" + i;
        grid.appendChild(h);
        const rowMax = Math.max.apply(null, M[i]);
        for (let j = 0; j < N; j++) {
          const w = M[i][j];
          const a = rowMax > 0 ? Math.pow(w / rowMax, 0.65) : 0;
          const c = document.createElement("div");
          c.className = "hcell";
          c.style.background = "color-mix(in srgb, var(--accent) " + (a * 100).toFixed(1) + "%, var(--surface))";
          c.dataset.v = fmt(w, 4);
          c.dataset.t = "查询位置 " + i + "「" + showChar(tokens[i]) + "」→ 键位置 " + j + "「" + showChar(tokens[j]) + "」：" + fmt(w, 4);
          grid.appendChild(c);
        }
      }
      if (tip) tip.textContent = "第 " + (L + 1) + " 层 · 第 " + (H + 1) + " 个头：每一行是一个位置的注意力分布，行内和为 1。把鼠标移到格子上看具体数值。";
    }

    grid.addEventListener("mouseover", (e) => {
      const c = e.target.closest(".hcell");
      if (c && tip) tip.textContent = c.dataset.t;
    });

    function syncSeg(seg, attr) {
      if (!seg) return;
      $$("button", seg).forEach((b) => b.classList.toggle("on", +b.dataset[attr] === (attr === "l" ? L : H)));
    }
    if (layerSeg) layerSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      L = +b.dataset.l; userPicked = true; syncSeg(layerSeg, "l"); draw();
    });
    if (headSeg) headSeg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      H = +b.dataset.h; userPicked = true; syncSeg(headSeg, "h"); draw();
    });
    draw();

    // 首次滚动到场时提示"两种头长得不一样"
    const io = new IntersectionObserver((es) => {
      es.forEach((en) => {
        if (en.isIntersecting && !userPicked && headSeg) { syncSeg(headSeg, "h"); }
      });
    }, { threshold: 0.4 });
    io.observe(box);
  })();

  /* ============================================================
     演示 3 · BPE 分词
     ============================================================ */
  (function bpe() {
    const box = $('[data-demo="bpe"]');
    if (!box) return;
    const input = $("#bpe-in"), out = $("#bpe-out"), stats = $("#bpe-stats");
    const merges = (DATA.bpe && DATA.bpe.merges) || [];
    const rank = new Map();
    merges.forEach((pair, i) => rank.set(pair[0] + "\u0001" + pair[1], i));

    function encode(str) {
      let chars = Array.from(str);
      if (!chars.length) return [];
      let tokens = chars.slice();
      let merged = tokens.map(() => false);
      let guard = 0;
      while (guard++ < 4000) {
        let bestRank = Infinity, bestAt = -1;
        for (let i = 0; i < tokens.length - 1; i++) {
          const r = rank.get(tokens[i] + "\u0001" + tokens[i + 1]);
          if (r !== undefined && r < bestRank) { bestRank = r; bestAt = i; }
        }
        if (bestAt < 0) break;
        tokens.splice(bestAt, 2, tokens[bestAt] + tokens[bestAt + 1]);
        merged.splice(bestAt, 2, true);
      }
      return tokens.map((t, i) => ({ t: t, m: merged[i] }));
    }

    function render() {
      const str = input.value.slice(0, 200);
      const toks = encode(str);
      out.innerHTML = "";
      toks.forEach((tk) => {
        const s = document.createElement("span");
        s.className = "bpe-token" + (tk.m ? " merged" : "");
        s.textContent = tk.t.replace(/ /g, "␣").replace(/\n/g, "↵");
        s.title = JSON.stringify(tk.t);
        out.appendChild(s);
      });
      const bytes = new TextEncoder().encode(str).length;
      const chars = Array.from(str).length;
      stats.innerHTML =
        "<span>字符 <b>" + chars + "</b></span>" +
        "<span>token <b>" + toks.length + "</b></span>" +
        "<span>压缩率 <b>" + (chars / (toks.length || 1)).toFixed(2) + "</b> 字/token</span>" +
        "<span>UTF-8 字节 <b>" + bytes + "</b></span>" +
        "<span>合并表 <b>" + merges.length + "</b> 条</span>";
    }

    if (input) {
      input.addEventListener("input", render);
      render();
    }
  })();

  /* ============================================================
     演示 4 · KV cache 显存计算器
     ============================================================ */
  (function kv() {
    const box = $('[data-demo="kv"]');
    if (!box) return;
    const sel = $("#kv-model"), L = $("#kv-L"), HKV = $("#kv-hkv"), D = $("#kv-d"),
      C = $("#kv-ctx"), B = $("#kv-batch"), P = $("#kv-prec");
    const presets = {
      "llama3-8b":  { l: 32, hkv: 8,  d: 128, ctx: 8192, params: 8.03e9, heads: 32 },
      "llama2-7b":  { l: 32, hkv: 32, d: 128, ctx: 8192, params: 6.74e9, heads: 32 },
      "mistral-7b": { l: 32, hkv: 8,  d: 128, ctx: 8192, params: 7.24e9, heads: 32 },
      "qwen25-72b": { l: 80, hkv: 8,  d: 128, ctx: 8192, params: 72.7e9, heads: 64 },
      "gpt3-175b":  { l: 96, hkv: 96, d: 128, ctx: 8192, params: 175e9,  heads: 96 },
    };
    // 权重/带宽用十进制 GB，KV 的 KB/MB 用 1024 进制（与正文 "128KB/token" 一致）
    const decFmt = (n) => n >= 1e9 ? fmt(n / 1e9, 2) + " GB" : fmt(n / 1e6, 1) + " MB";
    const binFmt = (n) => n >= 1024 ** 3 ? fmt(n / 1024 ** 3, 2) + " GB"
      : n >= 1024 ** 2 ? fmt(n / 1024 ** 2, 1) + " MB"
      : fmt(n / 1024, n / 1024 >= 10 ? 0 : 1) + " KB";
    function calc() {
      const layers = +L.value, hkv = +HKV.value, d = +D.value, ctx = +C.value,
        batch = +B.value, bytes = +P.value;
      const preset = presets[sel.value];
      const params = preset ? preset.params : 8.03e9;
      const heads = preset ? preset.heads : Math.max(hkv, 32);
      const weights = params * bytes;
      const per = 2 * layers * hkv * d * bytes;      // 每 token 字节
      const seq = per * ctx;                          // 单序列
      const totalKV = seq * batch;                    // batch 总缓存
      const read = weights + totalKV;                 // 每步读取字节
      const tStep = read / 3.35e12;                   // H100 3.35TB/s
      const tput = batch / tStep;
      const usable = 80e9, reserve = 2e9;
      const conc = Math.max(0, Math.floor((usable - weights - reserve) / seq));
      const mhaPer = 2 * layers * heads * d * bytes;

      $("#kv-v0").textContent = decFmt(weights);
      $("#kv-v2").textContent = binFmt(per);
      $("#kv-v3").textContent = binFmt(seq);
      $("#kv-v4").textContent = binFmt(totalKV);
      $("#kv-v5").textContent = decFmt(read);
      $("#kv-v6").textContent = fmt(tput, 0) + " tok/s";
      $("#kv-v7").textContent = weights > usable - reserve ? "放不下" : conc + " 路";
      $("#kv-oL").textContent = layers;
      $("#kv-ohkv").textContent = hkv;
      $("#kv-od").textContent = d;
      $("#kv-obatch").textContent = batch;
      $("#kv-octx").textContent = ctx >= 1024 ? fmt(ctx / 1024, 0) + "K" : String(ctx);

      const parts = [];
      if (weights > usable - reserve) {
        parts.push("权重 " + decFmt(weights) + " 已经超过单张 80GB 卡，必须张量并行切到多卡——此时每卡的 KV 也被切到 1/TP。");
      } else {
        parts.push("这张卡上权重占 " + decFmt(weights) + "，留给 KV 的约 " +
          decFmt(usable - weights - reserve) + "，所以每序列 " + binFmt(seq) +
          " 的 " + (ctx >= 1024 ? fmt(ctx / 1024, 0) + "K" : ctx + " token") + " 上下文能开 <b>" + conc + " 路</b>。");
      }
      if (hkv < heads) {
        parts.push("同规模若用 MHA（KV 头数 " + heads + "），每 token 要 " + binFmt(mhaPer) +
          "，是现在的 <b>" + fmt(mhaPer / per, 1) + " 倍</b>。");
      }
      parts.push("每步读 " + decFmt(read) + "，按 H100 3.35TB/s 算 " + fmt(tStep * 1000, 1) +
        " ms/步，聚合上限约 <b>" + fmt(tput, 0) + " tok/s</b>，单流约 " + fmt(tStep > 0 ? 1 / tStep : 0, 0) +
        " tok/s——batch 越大总吞吐越高、单流越慢。");
      $("#kv-note").innerHTML = parts.join(" ");
    }
    function onInput() { sel.value = "custom"; calc(); }
    if (sel) sel.addEventListener("change", () => {
      const p = presets[sel.value];
      if (!p) return;
      L.value = p.l; HKV.value = p.hkv; D.value = p.d; C.value = p.ctx; B.value = 1;
      calc();
    });
    [L, HKV, D, C, B, P].forEach((el) => el && el.addEventListener("input", onInput));
    calc();
  })();

  /* ============================================================
     演示 6 · 上下文预算条（第 16 章 agent 循环的窗口账）
     ============================================================ */
  (function ctx() {
    const box = $('[data-demo="ctx"]');
    if (!box) return;
    const W = $("#cx-win"), S = $("#cx-sys"), T = $("#cx-tool"), K = $("#cx-think"), N = $("#cx-turn");
    const bar = $("#cx-bar"), sysSeg = $("i.sys", bar), histSeg = $("i.hist", bar), freeSeg = $("i.free", bar);
    const legend = $("#cx-legend"), note = $("#cx-note");
    const oSys = $("#cx-sys-out"), oTool = $("#cx-tool-out"), oThink = $("#cx-think-out"), oTurn = $("#cx-turn-out");
    const pct = (x) => fmt(x * 100, 0);
    const num = (n) => Math.round(n).toLocaleString("en-US");
    const winLabel = (v) => ({ 32768: "32K", 131072: "128K", 200000: "200K", 1048576: "1M" })[+v] || fmt(+v / 1000, 0) + "K";

    function calc() {
      const win = +W.value, sys = +S.value, tool = +T.value, think = +K.value, turn = +N.value;
      const grow = tool + think;                  // 每轮净增
      const hist = grow * turn;                   // 历史累计
      const thr = 0.8 * win;                      // 压缩阈值
      const used = sys + hist;
      const free = Math.max(0, win - used);

      oSys.textContent = fmt(sys, 0);
      oTool.textContent = fmt(tool, 0);
      oThink.textContent = fmt(think, 0);
      oTurn.textContent = fmt(turn, 0);

      bar.style.setProperty("--thr", "80%");
      sysSeg.style.width = (sys / win * 100).toFixed(2) + "%";
      histSeg.style.width = (Math.min(hist, Math.max(0, win - sys)) / win * 100).toFixed(2) + "%";
      freeSeg.style.width = (free / win * 100).toFixed(2) + "%";
      histSeg.classList.toggle("over", used > thr);

      const nThr = Math.ceil((thr - sys) / grow);
      const nFull = Math.ceil((win - sys) / grow);

      legend.innerHTML =
        '<span><i style="background:color-mix(in srgb,var(--accent) 74%,var(--surface))"></i>系统 + 工具 schema <b>' +
          num(sys) + '</b>（' + pct(sys / win) + '%）</span>' +
        '<span><i style="background:color-mix(in srgb,var(--accent) 28%,var(--surface))"></i>' + fmt(turn, 0) +
          ' 轮历史 <b>' + num(hist) + '</b>（' + pct(hist / win) + '%）</span>' +
        '<span><i style="background:var(--surface-2);border:1px solid var(--rule)"></i>剩余 <b>' + num(free) + '</b></span>' +
        '<span>每轮净增 <b>' + num(grow) + '</b> = 工具 ' + num(tool) + ' + 推理 ' + num(think) + '</span>';

      const parts = [];
      if (turn > nFull) {
        parts.push("第 <b>" + nFull + "</b> 轮就撞满了 " + winLabel(win) + " 窗口，现在第 " + num(turn) +
          " 轮已经<b>溢出</b>：harness 只剩一条路——在工具结果进上下文之前先压缩，否则这一步直接报错。");
      } else if (turn > nThr) {
        parts.push("第 <b>" + nThr + "</b> 轮就过了 80% 阈值（" + num(thr) + " token），现在第 " +
          fmt(turn, 0) + " 轮还没压——下一次工具输出就会顶到窗口边缘，<b>必须压缩</b>。");
      } else if (nThr > 120) {
        parts.push("窗口 " + winLabel(win) + " 下，" + fmt(turn, 0) + " 轮离阈值还很远（第 " + nThr + " 轮才触发压缩）。");
      } else {
        parts.push("按当前参数，第 <b>" + nThr + "</b> 轮踩到 80% 阈值（" + num(thr) +
          " token）必须压缩历史；完全不压缩的话，第 <b>" + nFull + "</b> 轮撞满 " + winLabel(win) + " 窗口。");
      }
      if (grow > win * 0.02) {
        parts.push("一轮吃掉窗口的 <b>" + fmt(grow / win * 100, 1) + "%</b>——上下文预算要按步算，不是按任务算。");
      }
      note.innerHTML = parts.join(" ");
    }
    [W, S, T, K, N].forEach((el) => el && el.addEventListener("input", calc));
    calc();
  })();

  /* ============================================================
     演示 · 提交开销与 GPU 空转
     ============================================================ */
  (function launch() {
    const box = $('[data-demo="launch"]');
    if (!box) return;
    const N = $("#lg-n"), TK = $("#lg-tk"), TL = $("#lg-tl"), seg = $("#lg-graph");
    const oN = $("#lg-on"), oTk = $("#lg-otk"), oTl = $("#lg-otl");
    const gpuEl = $("#lg-gpu"), cpuEl = $("#lg-cpu"), wallEl = $("#lg-wall"),
          idleEl = $("#lg-idle"), tpEl = $("#lg-tput"), note = $("#lg-note");
    let graph = 0;
    const ms = (us) => us >= 1000 ? fmt(us / 1000, 2) + " 毫秒" : fmt(us, 0) + " 微秒";

    function calc() {
      const n = +N.value, tk = +TK.value, tl = +TL.value;
      const gpu = n * tk;                    // 微秒
      const cpu = graph ? 30 : n * tl;       // 一次回放约几十微秒
      const wall = Math.max(gpu, cpu);
      const busy = wall ? gpu / wall : 0;

      oN.textContent = n;
      oTk.textContent = tk + " 微秒";
      oTl.textContent = graph ? "回放一次" : tl + " 微秒";
      gpuEl.textContent = ms(gpu);
      cpuEl.textContent = ms(cpu);
      wallEl.textContent = ms(wall);
      idleEl.textContent = fmt(busy * 100, 0) + "%";
      tpEl.textContent = wall ? fmt(1e6 / wall, 0) + " tok/s" : "—";

      const parts = [];
      if (cpu > gpu) {
        parts.push("CPU 提交（" + ms(cpu) + "）超过 GPU 执行（" + ms(gpu) +
          "），属于<b>提交受限</b>：GPU 有 " + fmt((1 - busy) * 100, 0) + "% 的时间在空等下一步。");
        if (!graph) parts.push("把 CUDA graph 打开，几百次启动会被压成一次回放，这段空档基本消失。");
      } else {
        parts.push("GPU 执行（" + ms(gpu) + "）是瓶颈，忙碌占比 " + fmt(busy * 100, 0) +
          "%：每步 " + n + " 个 kernel、每个 " + tk + " 微秒。");
        if (!graph) parts.push("这时开不开 graph 差别不大——CPU 的提交时间已经被藏进 GPU 的执行时间里了。");
        if (tk <= 8) parts.push("每个 kernel 只有几微秒，说明算子很碎：融合成一个能同时省掉提交和调度开销。");
      }
      if (graph) parts.push("graph 不改变 kernel 本身的速度，只去掉 CPU 与驱动的重复开销。");
      note.innerHTML = parts.join(" ");
    }

    [N, TK, TL].forEach((el) => el && el.addEventListener("input", calc));
    if (seg) seg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      seg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      graph = +b.dataset.g;
      calc();
    }));
    calc();
  })();

  /* ============================================================
     演示 5 · RoPE 相对位置
     ============================================================ */
  function drawRope(canvas) {
    if (!canvas || !canvas.getContext) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth || 640, h = 260;
    canvas.width = w * dpr; canvas.height = h * dpr;
    canvas.style.height = h + "px";
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, w, h);
    const css = getComputedStyle(document.documentElement);
    const ink = css.getPropertyValue("--ink-2").trim() || "#444";
    const faint = css.getPropertyValue("--faint").trim() || "#999";
    const accent = css.getPropertyValue("--accent").trim() || "#3b4cb8";
    const accent2 = css.getPropertyValue("--accent-2").trim() || "#a34a12";
    const surface = css.getPropertyValue("--surface").trim() || "#fff";
    const delta = canvas._delta === undefined ? 7 : canvas._delta;
    const headDim = 8;
    const base = 10000;
    const freqs = [];
    for (let i = 0; i < headDim / 2; i++) freqs.push(Math.pow(base, -2 * i / headDim));

    // 左侧：二维平面上两个旋转向量（只用最低频那一对）
    const cx = w * 0.22, cy = h * 0.5, R = Math.min(w * 0.16, 88);
    const theta = freqs[0];
    const a1 = theta * 0, a2 = theta * delta;
    g.strokeStyle = faint; g.lineWidth = 1;
    g.beginPath(); g.arc(cx, cy, R, 0, Math.PI * 2); g.stroke();
    g.beginPath(); g.moveTo(cx - R - 8, cy); g.lineTo(cx + R + 8, cy);
    g.moveTo(cx, cy - R - 8); g.lineTo(cx, cy + R + 8); g.stroke();

    function arrow(ang, color) {
      const x = cx + R * Math.cos(ang), y = cy - R * Math.sin(ang);
      g.strokeStyle = color; g.lineWidth = 2.4;
      g.beginPath(); g.moveTo(cx, cy); g.lineTo(x, y); g.stroke();
      g.fillStyle = color;
      g.beginPath(); g.arc(x, y, 4, 0, Math.PI * 2); g.fill();
    }
    arrow(a1, accent2);
    arrow(a2, accent);
    g.fillStyle = ink; g.font = "11px ui-monospace, monospace";
    g.fillText("位置 m 的 q", cx - R - 4, cy - R - 14);
    g.fillText("位置 n 的 k", cx + R - 36, cy + R + 18);

    // 右侧：点积随相对距离变化
    const x0 = w * 0.42, x1 = w - 18, y0 = 36, y1 = h - 34;
    g.strokeStyle = faint;
    g.beginPath(); g.moveTo(x0, y1); g.lineTo(x1, y1); g.moveTo(x0, y0); g.lineTo(x0, y1); g.stroke();
    g.fillStyle = faint;
    g.fillText("相对距离 Δ", x1 - 78, y1 + 18);
    g.save(); g.translate(x0 - 8, y0 + 4); g.rotate(-Math.PI / 2); g.fillText("dot(q,k)", 0, 0); g.restore();
    const D = 120;
    const val = (d) => freqs.reduce((s, th) => s + Math.cos(th * d), 0) / freqs.length;
    g.strokeStyle = accent; g.lineWidth = 2; g.beginPath();
    for (let d = 0; d <= D; d++) {
      const px = x0 + (x1 - x0) * d / D;
      const py = y1 - (y1 - y0) * (val(d) + 1) / 2;
      d === 0 ? g.moveTo(px, py) : g.lineTo(px, py);
    }
    g.stroke();
    g.fillStyle = surface; g.strokeStyle = accent2; g.lineWidth = 2;
    const px = x0 + (x1 - x0) * delta / D;
    const py = y1 - (y1 - y0) * (val(delta) + 1) / 2;
    g.beginPath(); g.arc(px, py, 5, 0, Math.PI * 2); g.fill(); g.stroke();
    g.fillStyle = ink; g.font = "12px ui-monospace, monospace";
    g.fillText("Δ = " + delta + " → 平均点积 " + val(delta).toFixed(3), x0 + 8, y0 + 4);
  }

  (function rope() {
    const canvas = $(".rope-canvas");
    if (!canvas) return;
    const slider = $("#rope-d");
    const out = $("#rope-d-out");
    function sync() {
      canvas._delta = +slider.value;
      if (out) out.textContent = slider.value;
      drawRope(canvas);
    }
    if (slider) slider.addEventListener("input", sync);
    window.addEventListener("resize", () => drawRope(canvas));
    sync();
  })();
  /* ============================================================
     演示 · 量化粒度与重建误差
     ============================================================ */
  (function quant() {
    const box = $('[data-demo="quant"]');
    if (!box) return;
    const bitsSeg = $("#q-bits"), grp = $("#q-group");
    const cv = $("#q-canvas");
    const oGrp = $("#q-group-out"), oMax = $("#q-max"), oRmse = $("#q-rmse"), oCos = $("#q-cos"),
          oStep = $("#q-step"), note = $("#q-note");
    const canvas = cv;
    let bits = 4;

    // 96 个确定性权重：大部分在 ±0.9 之间，另有两个离群值
    const W = [];
    for (let i = 0; i < 96; i++) {
      let v = 0.52 * Math.sin(i * 0.61) + 0.33 * Math.cos(i * 0.23) + 0.10 * Math.sin(i * 1.7);
      if (i === 17) v = 6.0;
      if (i === 70) v = -5.2;
      W.push(v);
    }

    function quantize(groupSize) {
      const qmax = Math.pow(2, bits - 1) - 1;         // 8→127, 4→7, 3→3
      const out = new Array(W.length);
      const scales = [];
      for (let g = 0; g < W.length; g += groupSize) {
        const end = Math.min(g + groupSize, W.length);
        let m = 0;
        for (let i = g; i < end; i++) m = Math.max(m, Math.abs(W[i]));
        const s = m / qmax || 1;
        scales.push(s);
        for (let i = g; i < end; i++) {
          const q = Math.max(-qmax, Math.min(qmax, Math.round(W[i] / s)));
          out[i] = q * s;
        }
      }
      return { rec: out, scales };
    }

    function metrics(rec) {
      let maxErr = 0, se = 0, dot = 0, na = 0, nb = 0;
      for (let i = 0; i < W.length; i++) {
        const e = Math.abs(W[i] - rec[i]);
        if (e > maxErr) maxErr = e;
        se += e * e;
        dot += W[i] * rec[i];
        na += W[i] * W[i];
        nb += rec[i] * rec[i];
      }
      return { maxErr, rmse: Math.sqrt(se / W.length), cos: dot / Math.sqrt(na * nb) };
    }

    function draw(rec, groupSize) {
      const dpr = window.devicePixelRatio || 1;
      const w = canvas.clientWidth || 600, h = canvas.clientHeight || 220;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const cs = getComputedStyle(document.documentElement);
      const ink = cs.getPropertyValue("--muted").trim() || "#666";
      const accent = cs.getPropertyValue("--accent").trim() || "#3b5bdb";
      const rule = cs.getPropertyValue("--rule").trim() || "#ddd";
      const bad = cs.getPropertyValue("--bad").trim() || "#b03030";

      const padL = 34, padR = 8, padT = 14, padB = 20;
      const iw = w - padL - padR, ih = h - padT - padB;
      const yMax = 6.6, yMin = -6.2;
      const X = (i) => padL + (i + 0.5) * (iw / W.length);
      const Y = (v) => padT + (yMax - v) / (yMax - yMin) * ih;

      // 网格与零线
      ctx.strokeStyle = rule; ctx.lineWidth = 1;
      [6, 3, 0, -3, -6].forEach((v) => {
        ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(w - padR, Y(v)); ctx.stroke();
        ctx.fillStyle = ink; ctx.font = "10px monospace"; ctx.textAlign = "right";
        ctx.fillText(String(v), padL - 4, Y(v) + 3);
      });

      // 分组边界
      ctx.setLineDash([3, 3]);
      ctx.strokeStyle = rule;
      for (let g = groupSize; g < W.length; g += groupSize) {
        ctx.beginPath(); ctx.moveTo(X(g) - (iw / W.length) / 2, padT); ctx.lineTo(X(g) - (iw / W.length) / 2, h - padB); ctx.stroke();
      }
      ctx.setLineDash([]);

      // 重建值：从零线竖干
      ctx.strokeStyle = accent; ctx.lineWidth = 1.4;
      for (let i = 0; i < W.length; i++) {
        ctx.beginPath(); ctx.moveTo(X(i), Y(0)); ctx.lineTo(X(i), Y(rec[i])); ctx.stroke();
      }

      // 原值点
      ctx.fillStyle = ink;
      for (let i = 0; i < W.length; i++) {
        ctx.beginPath(); ctx.arc(X(i), Y(W[i]), 1.9, 0, Math.PI * 2); ctx.fill();
      }

      // 离群值标注
      [[17, 6.0], [70, -5.2]].forEach(([i, v]) => {
        ctx.fillStyle = bad;
        ctx.beginPath(); ctx.arc(X(i), Y(v), 2.6, 0, Math.PI * 2); ctx.fill();
      });
    }

    function calc() {
      const g = +grp.value;
      const { rec, scales } = quantize(g);
      const m = metrics(rec);
      const sorted = scales.slice().sort((a, b) => a - b);
      const medStep = sorted[Math.floor(sorted.length / 2)] || 0;
      const maxStep = sorted[sorted.length - 1] || 0;
      draw(rec, g);
      oGrp.textContent = g >= W.length ? g + "（整行一组）" : g + "（每组 " + g + " 个）";
      oMax.textContent = fmt(m.maxErr, 3);
      oRmse.textContent = fmt(m.rmse, 4);
      oCos.textContent = fmt(m.cos, 4);
      oStep.textContent = fmt(medStep, 2) + " / " + fmt(maxStep, 2);

      const qmax = Math.pow(2, bits - 1) - 1;
      const parts = [];
      parts.push("int" + bits + " 的步长由每组的最大值决定：整行一组时就是 6.0 撑出来的 " + fmt(maxStep, 2) + "。");
      if (g >= W.length) {
        parts.push("绝对值小于 " + fmt(maxStep / 2, 2) + " 的权重全部被压成 0——图上这些点只剩一根贴着零线的竖干，原值已经丢了。");
      } else {
        parts.push("分组到 " + g + " 之后，中位步长降到 " + fmt(medStep, 3) +
          "，误差主要集中在那两个含离群值的组（步长仍是 " + fmt(maxStep, 2) +
          "）。所以实际方案还要单独处理离群值：混合精度保留少数通道、AWQ 的通道缩放、或者直接裁剪。");
      }
      if (bits === 3) parts.push("int3 的步长又翻倍，分组只能缓解、救不回来——这就是 3bit 通常要换成更复杂的量化算法的原因。");
      note.innerHTML = parts.join(" ");
    }

    if (bitsSeg) bitsSeg.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      bitsSeg.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      bits = +b.dataset.b;
      calc();
    }));
    if (grp) grp.addEventListener("input", calc);
    window.addEventListener("resize", () => calc());
    calc();
  })();
  /* ============================================================
     演示 · 压缩手段叠加（注意力方案 / 精度 / 淘汰比例）
     ============================================================ */
  (function ck() {
    const box = $('[data-demo="ck"]');
    if (!box) return;
    const scheme = $("#ck-scheme"), prec = $("#ck-prec"), keepSeg = $("#ck-keep");
    const oTok = $("#ck-token"), oConc = $("#ck-conc"), oRead = $("#ck-read"), oTp = $("#ck-tput"), note = $("#ck-note");
    let s = "gqa", p = "fp16", k = 1;
    const L = 32, ctx = 8192, gpu = 80e9, weights = 16.06e9, act = 2e9, bw = 3.35e12;

    const KB = (b) => {
      const kb = b / 1024;
      return kb >= 1024 ? fmt(kb / 1024, kb / 1024 >= 10 ? 0 : 1) + " MB" : fmt(kb, kb >= 100 ? 0 : 1) + " KB";
    };
    const GB = (b) => fmt(b / 1e9, 2) + " GB";

    function base(schemeKey) {
      if (schemeKey === "mha") return 2 * L * 32 * 128 * 2;      // 32 头，128 维
      if (schemeKey === "mla") return L * (512 + 64) * 2;        // 低秩潜向量 + RoPE 分量
      return 2 * L * 8 * 128 * 2;                                 // GQA 8 组
    }

    function calc() {
      const precMul = p === "fp8" ? 0.5 : p === "int4" ? 0.25 : 1;
      const perTokRaw = base(s) * precMul;
      const perTok = perTokRaw * k;                 // 淘汰后实际存储/读取
      const totalPerSeq = perTok * ctx;
      const conc = Math.max(0, Math.floor((gpu - weights - act) / totalPerSeq));
      const read = weights + totalPerSeq;           // batch 1
      const tps = bw / read;

      oTok.textContent = KB(perTok);
      oConc.textContent = conc + " 路";
      oRead.textContent = GB(read);
      oTp.textContent = fmt(tps, 0) + " tok/s";

      const parts = [];
      const mha = base("mha") * precMul * k;
      const saved = 1 - perTok / mha;
      parts.push("相对同规模的 MHA" + (precMul !== 1 || k !== 1 ? "（同样的精度与保留比例）" : "") +
        "，每 token 的字节少了 " + fmt(saved * 100, 0) + "%。");
      if (k < 1) parts.push("淘汰不改变历史已经算过的算力，只减少显存与每步读取；代价是被丢掉的 KV 再也回不来。");
      if (s === "mla") parts.push("MLA 的收益来自\"只存 576 维潜向量\"，decode 时不解出 K/V——这是结构性的省法，不与精度、淘汰冲突。");
      if (p === "int4") parts.push("4bit 的 KV 对精度更敏感，通常要配合分组缩放与 sink 保护。");
      parts.push("并发按 80GB 卡、权重 16.06GB、激活预留 2GB 估算；每步读取 = 权重 + 一条序列的 KV。");
      note.innerHTML = parts.join(" ");
    }

    function bindSeg(el, apply) {
      if (!el) return;
      el.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
        el.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
        apply(b.dataset);
        calc();
      }));
    }
    bindSeg(scheme, (d) => { s = d.s; });
    bindSeg(prec, (d) => { p = d.p; });
    bindSeg(keepSeg, (d) => { k = +d.k; });
    calc();
  })();
  /* ============================================================
     演示 · 卸载值不值（换回 vs 重算）
     ============================================================ */
  (function offload() {
    const box = $('[data-demo="offload"]');
    if (!box) return;
    const tier = $("#of-tier"), nIn = $("#of-n"), hIn = $("#of-h");
    const nOut = $("#of-n-out"), hOut = $("#of-h-out");
    const oLoad = $("#of-load"), oRe = $("#of-recomp"), oAvg = $("#of-avg"), oSteps = $("#of-steps"), note = $("#of-note");
    let bw = 64e9;
    const PER_TOK = 128 * 1024, STEP = 5e-3, TFLOPS = 400e12, FLOPS_PER_TOK = 16e9;

    const sec = (s) => (s >= 1 ? fmt(s, 2) + " 秒" : fmt(s * 1e3, s * 1e3 >= 100 ? 0 : 1) + " 毫秒");
    const kTok = (n) => (n >= 1024 ? fmt(n / 1024, n >= 10240 ? 0 : 1) + "K" : String(n));

    function calc() {
      const n = +nIn.value, h = +hIn.value / 100;
      const S = n * PER_TOK;
      const load = S / bw;
      const re = (n * FLOPS_PER_TOK) / TFLOPS;
      const avg = h * load + (1 - h) * re;
      nOut.textContent = kTok(n);
      hOut.textContent = Math.round(h * 100) + "%";
      oLoad.textContent = sec(load);
      oRe.textContent = sec(re);
      oAvg.textContent = sec(avg);
      oSteps.textContent = fmt(load / STEP, load / STEP >= 10 ? 0 : 1) + " 步";

      const parts = [];
      if (load < re) {
        parts.push("换回比重算快 " + fmt(re / load, 1) + " 倍——只要命中就有收益。");
      } else {
        parts.push("换回比重算还慢，除非前缀一定会被反复命中，否则这笔缓存不该建。");
      }
      if (load / STEP > 1) {
        parts.push("但换回要占用 " + fmt(load / STEP, 1) + " 个 decode 步的时长，必须靠预取与计算重叠，或设置超时后改走重算。");
      }
      if (bw < 1e10) parts.push("NVMe 层的换回时间在百毫秒量级，只适合冷数据预热与离线复用，不适合在线热路径。");
      note.innerHTML = parts.join(" ");
    }

    tier.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
      tier.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
      bw = +b.dataset.b * 1e9;
      calc();
    }));
    nIn.addEventListener("input", calc);
    hIn.addEventListener("input", calc);
    calc();
  })();
})();

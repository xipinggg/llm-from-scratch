// 把一批 TeX 渲染成静态 HTML（构建期完成，页面运行时不依赖 KaTeX JS）
const katex = require("katex");

let buf = "";
process.stdin.on("data", (d) => (buf += d));
process.stdin.on("end", () => {
  const items = JSON.parse(buf);
  const out = items.map(({ tex, display }) => {
    try {
      return katex.renderToString(tex, {
        displayMode: !!display,
        throwOnError: false,
        strict: "ignore",
        output: "html",
        trust: false,
      });
    } catch (e) {
      return `<span style="color:#c00">[TeX error: ${String(e.message).slice(0, 80)}]</span>`;
    }
  });
  process.stdout.write(JSON.stringify(out));
});

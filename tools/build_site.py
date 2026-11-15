#!/usr/bin/env python3
"""把 md/ 里生成的 Markdown 渲染成一个静态站点（docs/，供 GitHub Pages 从 /docs 发布）。

设计目标是给 GitHub Pages 用：每章一页、左侧目录、公式构建期渲染成
静态 HTML（不依赖 KaTeX 运行时 JS）。Markdown 的子集正好就是
tools/html2md.py 会产出的那些结构。

用法：
    python3 tools/build_site.py
"""

import html
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build as B  # 复用 render_math / katex_css / NODE / ROOT

ROOT = B.ROOT
MD = os.path.join(ROOT, "md")
OUT = os.path.join(ROOT, "docs")

SENT_D = "\x02D{}\x03"
SENT_I = "\x02I{}\x03"

# ---------- 数学公式：先抽出来，构建期统一渲染 ----------

math_items = []          # [(tex, display)]
math_store = {}          # sentinel -> index


def extract_math(text):
    def keep(display):
        def sub(m):
            idx = len(math_items)
            math_items.append((m.group(1).strip(), display))
            key = (SENT_D if display else SENT_I).format(idx)
            math_store[key] = idx
            return key
        return sub

    text = re.sub(r"\$\$(.+?)\$\$", keep(True), text, flags=re.S)
    text = re.sub(r"(?<!\$)\$([^$\n]+?)\$(?!\$)", keep(False), text)
    return text


# ---------- 行内 ----------

def inline(text):
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![*\w])\*([^*\n]+)\*(?!\*)", r"<em>\1</em>", text)

    def restore(m):
        idx = math_store.get(m.group(0))
        if idx is None:
            return m.group(0)
        return math_html[idx]
    text = re.sub("\x02[DI][0-9]+\x03", restore, text)
    return text


math_html = []

# ---------- 块级 ----------

LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|[\s:|-]*$")


def is_table_start(lines, i):
    return (lines[i].strip().startswith("|") and i + 1 < len(lines)
            and TABLE_SEP.match(lines[i + 1] or "") is not None)


def split_row(line):
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def render_table(rows):
    head, body = rows[0], rows[2:]
    out = ['<div class="tablewrap"><table>', "<thead><tr>"]
    out += ["<th>" + inline(c) + "</th>" for c in split_row(head)]
    out.append("</tr></thead><tbody>")
    for r in body:
        out.append("<tr>" + "".join("<td>" + inline(c) + "</td>" for c in split_row(r)) + "</tr>")
    out.append("</tbody></table></div>")
    return "".join(out)


def render_blocks(lines):
    out = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        s = line.strip()

        if not s:
            i += 1
            continue

        # 代码块
        if s.startswith("```"):
            lang = s[3:].strip()
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            cls = ' class="language-' + lang + '"' if lang else ""
            out.append("<pre><code" + cls + ">" + html.escape("\n".join(buf)) + "</code></pre>")
            continue

        # 独立公式占位
        m = re.fullmatch("\x02D[0-9]+\x03", s)
        if m:
            idx = math_store[s]
            out.append('<div class="equation">' + math_html[idx] + "</div>")
            i += 1
            continue

        # 标题
        hm = re.match(r"^(#{1,4})\s+(.*)$", s)
        if hm:
            lvl = len(hm.group(1))
            out.append(f"<h{lvl}>{inline(hm.group(2))}</h{lvl}>")
            i += 1
            continue

        # 分隔线
        if re.fullmatch(r"-{3,}", s):
            out.append("<hr>")
            i += 1
            continue

        # 表格
        if is_table_start(lines, i):
            rows = []
            while i < n and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            out.append(render_table(rows))
            continue

        # 引用块（可含列表）
        if s.startswith(">"):
            buf = []
            while i < n and lines[i].strip().startswith(">"):
                buf.append(re.sub(r"^>\s?", "", lines[i].strip()))
                i += 1
            inner = render_blocks(buf)
            out.append("<blockquote>" + inner + "</blockquote>")
            continue

        # 列表
        if LIST_RE.match(line):
            ordered = LIST_RE.match(line).group(2).endswith(".")
            items = []
            while i < n and LIST_RE.match(lines[i]):
                mm = LIST_RE.match(lines[i])
                items.append(mm.group(3))
                i += 1
                # 续行（缩进或普通行，直到空行/新结构）
                while (i < n and lines[i].strip() and not LIST_RE.match(lines[i])
                       and not lines[i].strip().startswith((">", "```", "|", "#"))
                       and not re.fullmatch(r"-{3,}", lines[i].strip())
                       and not re.fullmatch("\x02D[0-9]+\x03", lines[i].strip())):
                    items[-1] += " " + lines[i].strip()
                    i += 1
            tag = "ol" if ordered else "ul"
            out.append("<" + tag + ">" + "".join("<li>" + inline(t) + "</li>" for t in items) + "</" + tag + ">")
            continue

        # 段落
        buf = [s]
        i += 1
        while (i < n and lines[i].strip()
               and not lines[i].strip().startswith(("```", ">", "|", "#"))
               and not LIST_RE.match(lines[i])
               and not re.fullmatch(r"-{3,}", lines[i].strip())
               and not re.fullmatch("\x02D[0-9]+\x03", lines[i].strip())):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")

    return "\n".join(out)


# ---------- 站点 ----------

CSS = """
:root{--bg:#fbf8f1;--surface:#fffdf8;--ink:#23201a;--ink2:#4b463d;--muted:#8b8272;
--rule:#e6dece;--accent:#26407a;--code:#1e2028;--code-ink:#e6e3da}
@media (prefers-color-scheme:dark){:root{--bg:#14140f;--surface:#1b1b15;--ink:#e8e3d6;
--ink2:#c3bcac;--muted:#8b8474;--rule:#2e2d24;--accent:#9db2e8;--code:#0e0f12;--code-ink:#dcd7c9}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:16px;line-height:1.85;
font-family:"PingFang SC","Songti SC",-apple-system,"Segoe UI",sans-serif}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.layout{display:flex;max-width:1180px;margin:0 auto;padding:0 24px}
nav{width:260px;flex:0 0 260px;padding:34px 22px 60px 0;position:sticky;top:0;align-self:flex-start;
max-height:100vh;overflow:auto;border-right:1px solid var(--rule)}
nav .brand{font-family:"SF Mono",monospace;font-size:11px;letter-spacing:.12em;
text-transform:uppercase;color:var(--muted);margin-bottom:14px}
nav a{display:block;padding:3px 0;color:var(--ink2);font-size:13.5px;line-height:1.5}
nav a.on{color:var(--accent);font-weight:600}
main{flex:1;min-width:0;padding:34px 0 80px 44px;max-width:780px}
h1{font-size:31px;line-height:1.35;margin:.2em 0 .8em;font-weight:600}
h2{font-size:21px;margin:2em 0 .7em;padding-top:.2em;border-top:1px solid var(--rule);padding-top:18px}
h3{font-size:17px;margin:1.6em 0 .5em}
p{margin:.85em 0;color:var(--ink2)}
strong{color:var(--ink)}
blockquote{margin:18px 0;padding:12px 16px;border-left:3px solid var(--accent);
background:var(--surface);border-radius:0 8px 8px 0}
blockquote p{margin:.35em 0}
blockquote ul{margin:.4em 0;padding-left:20px}
code{font-family:"SF Mono",ui-monospace,monospace;font-size:.86em;background:var(--surface);
border:1px solid var(--rule);border-radius:4px;padding:1px 5px}
pre{background:var(--code);color:var(--code-ink);padding:14px 16px;border-radius:10px;
overflow:auto;font-size:13px;line-height:1.65}
pre code{background:none;border:0;padding:0;color:inherit;font-size:inherit}
.tablewrap{overflow-x:auto;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{border-bottom:1px solid var(--rule);padding:8px 10px;text-align:left;vertical-align:top}
th{font-family:"SF Mono",monospace;font-size:11px;letter-spacing:.06em;text-transform:uppercase;
color:var(--muted);border-bottom:1px solid var(--rule)}
hr{border:0;border-top:1px solid var(--rule);margin:28px 0}
.equation{margin:20px 0;overflow-x:auto;text-align:center}
ul,ol{padding-left:24px}li{margin:.35em 0;color:var(--ink2)}
.crumb{font-family:"SF Mono",monospace;font-size:11px;letter-spacing:.1em;color:var(--muted);
text-transform:uppercase;margin-bottom:6px}
.pager{display:flex;justify-content:space-between;gap:16px;margin-top:56px;padding-top:18px;
border-top:1px solid var(--rule);font-size:13.5px}
.pager a{max-width:48%}
@media (max-width:900px){
 .layout{display:block;padding:0 18px}
 nav{position:static;width:auto;max-height:none;border-right:0;border-bottom:1px solid var(--rule);
 padding:20px 0}
 nav a{display:inline-block;margin-right:14px}
 main{padding:24px 0 60px;max-width:none}
 h1{font-size:25px}
}
"""

PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · 大模型是怎么想出一个字的</title>
<link rel="stylesheet" href="katex.css">
<link rel="stylesheet" href="style.css">
</head>
<body>
<div class="layout">
<nav>
  <div class="brand">LLM · From Scratch</div>
  {nav}
</nav>
<main>
  <div class="crumb">{crumb}</div>
  {body}
  <div class="pager">{pager}</div>
</main>
</div>
</body>
</html>
"""


def title_of(text):
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    hits = re.findall(r"(?m)^#\s+(.+)$", text)
    return hits[-1].strip() if hits else ""


def strip_head(text, title):
    """去掉开头的章号行与重复的章名标题（合集不处理）。"""
    lines = text.split("\n")
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and re.fullmatch(r"\*[^*]+\*", lines[i].strip()):
        i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i < len(lines) and lines[i].strip() == "# " + title:
        i += 1
    return "\n".join(lines[i:])


def kicker_of(text):
    m = re.search(r"(?m)^\*(第[^*]+)\*$", text)
    return m.group(1).strip() if m else ""


def slug(name):
    return name[:-3] + ".html"


def main():
    files = [f for f in sorted(os.listdir(MD)) if f.endswith(".md")]
    chapters = [f for f in files if re.match(r"\d\d-", f)]
    if not chapters:
        sys.exit("no chapter markdown found; run tools/html2md.py first")

    raw = {f: open(os.path.join(MD, f), encoding="utf-8").read() for f in files}

    # 1) 抽公式并批量渲染
    protected = {f: extract_math(t) for f, t in raw.items()}
    global math_html
    math_html = B.render_math([{"tex": t, "display": d} for t, d in math_items])

    # 2) 目录
    nav_items = [("index.html", "目录与说明", "")]
    for f in chapters:
        nav_items.append((slug(f), title_of(raw[f]), f[:2]))
    nav_items.append(("book.html", "合集（单页）", ""))

    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "style.css"), "w", encoding="utf-8").write(CSS)
    open(os.path.join(OUT, "katex.css"), "w", encoding="utf-8").write(B.katex_css())

    pages = [("index.html", "README.md", "目录")] + [(slug(f), f, title_of(raw[f])) for f in chapters]
    pages.append(("book.html", "book.md", "合集"))
    order = [p[0] for p in pages]

    for idx, (out_name, src_name, label) in enumerate(pages):
        src_text = protected[src_name]
        if out_name not in ("index.html", "book.html"):
            src_text = strip_head(src_text, label)
            body = "<h1>" + html.escape(label) + "</h1>\n" + render_blocks(src_text.split("\n"))
        else:
            body = render_blocks(src_text.split("\n"))
        nav = []
        for href, title, num in nav_items:
            text = (num + " · " + title) if num else title
            cls = ' class="on"' if href == out_name else ""
            nav.append(f'<a href="{href}"{cls}>{html.escape(text)}</a>')

        pager = []
        if idx > 0:
            prev = pages[idx - 1]
            pager.append(f'<a href="{prev[0]}">← {html.escape(prev[2])}</a>')
        else:
            pager.append("<span></span>")
        if idx < len(pages) - 1:
            nxt = pages[idx + 1]
            pager.append(f'<a href="{nxt[0]}" style="text-align:right">{html.escape(nxt[2])} →</a>')
        else:
            pager.append("<span></span>")

        crumb = kicker_of(raw[src_name]) or "目录"
        page = PAGE.format(title=label, nav="\n  ".join(nav), body=body,
                           pager="".join(pager), crumb=html.escape(crumb))
        open(os.path.join(OUT, out_name), "w", encoding="utf-8").write(page)
        print(f"  {out_name:34s} {len(page) // 1024:4d} KB  {label}")

    print(f"docs: {len(pages)} pages | math: {len(math_items)}")


if __name__ == "__main__":
    main()

"""把 src/ 的片段拼成单文件 index.html（KaTeX 静态渲染 + 字体内联，完全离线）。"""
import base64
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
PARTS = os.path.join(SRC, "parts")
KATEX = os.path.join(ROOT, "tools", "node_modules", "katex", "dist")
NODE = "/Users/fengxiping/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/node/bin/node"

MATH_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)
MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)
PH = "\x01{}\x02"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def render_math(items):
    if not items:
        return []
    js = os.path.join(ROOT, "tools", "render_math.js")
    out = subprocess.run([NODE, js], input=json.dumps(items), capture_output=True,
                         text=True, cwd=os.path.join(ROOT, "tools"))
    if out.returncode != 0:
        sys.exit("katex render failed:\n" + out.stderr[:2000])
    return json.loads(out.stdout)


def extract_math(text):
    items = []
    store = {}

    def take(m, display):
        idx = len(items)
        items.append({"tex": m.group(1).strip(), "display": display})
        tok = PH.format(f"M{idx}")
        store[tok] = idx
        return tok

    text = MATH_DISPLAY.sub(lambda m: take(m, True), text)
    text = MATH_INLINE.sub(lambda m: take(m, False), text)
    return text, items


def strip_tags(s):
    s = re.sub(r"<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_sections(body):
    """返回 [{id,num,title,desc,start,end,h3:[{id,title}]}]"""
    marks = list(re.finditer(r'<section class="chapter"([^>]*)>', body))
    secs = []
    for i, m in enumerate(marks):
        attrs = m.group(1)
        end = marks[i + 1].start() if i + 1 < len(marks) else len(body)
        sec = {
            "id": (re.search(r'id="([^"]+)"', attrs) or [None, ""])[1],
            "num": (re.search(r'data-num="([^"]*)"', attrs) or [None, ""])[1],
            "title": (re.search(r'data-title="([^"]*)"', attrs) or [None, ""])[1],
            "desc": (re.search(r'data-desc="([^"]*)"', attrs) or [None, ""])[1],
            "start": m.start(), "end": end, "h3": [],
        }
        secs.append(sec)
    return secs


def number_headings(body, secs):
    """给没有 id 的 h3 补 id，返回新 body。"""
    out, cursor = [], 0
    for sec in secs:
        out.append(body[cursor:sec["start"]])
        chunk = body[sec["start"]:sec["end"]]
        idx = 0

        def repl(m):
            nonlocal idx
            attrs, inner = m.group(1), m.group(2)
            got = re.search(r'id="([^"]+)"', attrs)
            if got:
                hid = got.group(1)
            else:
                hid = f"{sec['id']}-{idx + 1}"
                attrs = attrs + f' id="{hid}"'
            sec["h3"].append({"id": hid, "title": strip_tags(re.sub(PH.replace("{}", r"\d+").replace("\x01", "\\x01").replace("\x02", "\\x02"), "", inner))})
            idx += 1
            return f"<h3{attrs}>{inner}</h3>"

        chunk = re.sub(r"<h3([^>]*)>(.*?)</h3>", repl, chunk, flags=re.S)
        out.append(chunk)
        cursor = sec["end"]
    out.append(body[cursor:])
    return "".join(out)


def nav_html(secs):
    rows = []
    for s in secs:
        subs = ""
        if s["h3"]:
            subs = '<ul class="subs">' + "".join(
                f'<li><a href="#{h["id"]}">{h["title"] or "小节"}</a></li>' for h in s["h3"]) + "</ul>"
        rows.append(
            f'<li><a href="#{s["id"]}"><span class="n">{s["num"]}</span><span>{s["title"]}</span></a>{subs}</li>')
    return "\n".join(rows)


def cards_html(secs):
    rows = []
    for s in secs:
        rows.append(
            f'<a href="#{s["id"]}"><span class="num">第 {s["num"]} 章</span>'
            f'<span class="t">{s["title"]}</span><span class="d">{s["desc"]}</span></a>')
    return "\n".join(rows)


def katex_css():
    css = read(os.path.join(KATEX, "katex.min.css"))
    fonts = os.path.join(KATEX, "fonts")

    def repl(m):
        block = m.group(0)
        w = re.search(r"url\(fonts/([^)]+\.woff2)\)", block)
        if not w:
            return block
        path = os.path.join(fonts, w.group(1))
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        block = re.sub(r"src:[^;}]+", f'src:url(data:font/woff2;base64,{b64}) format("woff2")', block, count=1)
        return block

    return re.sub(r"@font-face\{[^}]*\}", repl, css)


def main():
    names = sorted(n for n in os.listdir(PARTS) if n.endswith(".html"))
    parts = [read(os.path.join(PARTS, n)) for n in names]
    body = "\n".join(parts)

    body, items = extract_math(body)
    secs = parse_sections(body)
    body = number_headings(body, secs)

    rendered = render_math(items)

    def put(m):
        tok = m.group(0)
        idx = int(tok.strip("\x01\x02")[1:])
        return rendered[idx]

    body = re.sub("\x01M\\d+\x02", put, body)

    shell = read(os.path.join(SRC, "shell.html"))
    data = json.load(open(os.path.join(ROOT, "build", "numbers.json"), encoding="utf-8"))
    bpe_path = os.path.join(ROOT, "build", "bpe.json")
    if os.path.exists(bpe_path):
        data = {"llm": data, "bpe": json.load(open(bpe_path, encoding="utf-8"))}

    css = read(os.path.join(SRC, "app.css")) + "\n" + katex_css()
    extra_path = os.path.join(SRC, "extra.css")
    if os.path.exists(extra_path):
        css += "\n" + read(extra_path)
    js = read(os.path.join(SRC, "app.js"))

    html = (shell
            .replace("{{BODY}}", body)
            .replace("{{NAV}}", nav_html(secs))
            .replace("{{CARDS}}", cards_html(secs))
            .replace("{{CSS}}", css)
            .replace("{{JS}}", js)
            .replace("{{DATA}}", json.dumps(data, ensure_ascii=False, separators=(",", ":"))))

    out = os.path.join(ROOT, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out}  {len(html)/1024:.0f} KB  sections={len(secs)}  math={len(items)}")


if __name__ == "__main__":
    main()

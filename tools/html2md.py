# -*- coding: utf-8 -*-
"""把 src/parts/*.html 转成 Markdown（用于仓库的 md 分支）。

用法：python3 tools/html2md.py
输出：md/NN-slug.md（逐章）、md/README.md（目录）、md/book.md（合集）
"""
import os, re, sys, html
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARTS = os.path.join(ROOT, "src", "parts")
OUT = os.path.join(ROOT, "md")

VOID = {"br", "img", "input", "hr", "meta", "link", "source", "wbr", "col",
        "path", "circle", "rect", "line", "polyline", "polygon", "ellipse",
        "stop", "use", "image", "animate", "text", "tspan"}
SKIP_TAGS = {"svg", "canvas", "select", "button", "script", "style"}
SKIP_VOID = {"input", "img", "br", "hr", "meta", "link"}


class Node:
    __slots__ = ("tag", "attrs", "children")
    def __init__(self, tag, attrs):
        self.tag = tag
        self.attrs = dict(attrs)
        self.children = []


class DomParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("root", [])
        self.stack = [self.root]
        self.skip_depth = 0

    BLOCK_START = {"p", "div", "section", "h1", "h2", "h3", "h4", "h5", "h6",
                   "table", "ul", "ol", "pre", "figure", "details", "blockquote"}

    def _implied_close(self, tag):
        st = self.stack
        def top():
            return st[-1].tag if len(st) > 1 else None
        if top() == "p" and tag in self.BLOCK_START:
            st.pop()
        if tag == "li":
            while len(st) > 1 and st[-1].tag not in ("ul", "ol"):
                st.pop()
        if tag == "tr":
            while len(st) > 1 and st[-1].tag not in ("table", "thead", "tbody"):
                st.pop()
        if tag in ("td", "th"):
            while len(st) > 1 and st[-1].tag != "tr":
                st.pop()

    def handle_starttag(self, tag, attrs):
        if self.skip_depth:
            if tag in SKIP_TAGS:
                self.skip_depth += 1
            return
        if tag in SKIP_VOID:
            return
        if tag in SKIP_TAGS:
            self.skip_depth = 1
            return
        self._implied_close(tag)
        node = Node(tag, attrs)
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        if self.skip_depth:
            return
        if tag in SKIP_TAGS or tag in SKIP_VOID:
            return
        self.stack[-1].children.append(Node(tag, attrs))

    def handle_endtag(self, tag):
        if self.skip_depth:
            if tag in SKIP_TAGS:
                self.skip_depth -= 1
            return
        if tag in VOID:
            return
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].tag == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        if self.skip_depth:
            return
        self.stack[-1].children.append(data)


def cls(node):
    return (node.attrs.get("class") or "").split()


def has(node, name):
    return name in cls(node)


def text_of(node):
    """纯文本（用于摘要、标题）"""
    out = []
    for c in node.children:
        if isinstance(c, str):
            out.append(c)
        else:
            out.append(text_of(c))
    return "".join(out)


MATH_INLINE = re.compile(r"\\\((.+?)\\\)", re.S)
MATH_DISPLAY = re.compile(r"\\\[(.+?)\\\]", re.S)


def inline(text):
    """把源码里的行内标记渲染成 Markdown 片段。"""
    text = text.replace("\u00a0", " ")
    # 数学：\( x \) -> $x$；\[ x \] -> $$x$$（极少出现在行内）
    text = MATH_INLINE.sub(lambda m: "$" + m.group(1).strip() + "$", text)
    text = MATH_DISPLAY.sub(lambda m: "$$" + m.group(1).strip() + "$$", text)
    # 压缩空白（不 strip，保留边界空格，交给段落层处理）
    text = re.sub(r"[ \t\r\n]+", " ", text)
    return text


def flat(s):
    return re.sub(r"\s*\n\s*", " ", s).strip()


def escape_cell(s):
    s = s.replace("|", "\\|")
    return re.sub(r"\s+", " ", s).strip()


class Renderer:
    def __init__(self):
        self.out = []

    # ---------- 行内 ----------
    def inline_children(self, node):
        buf = []
        for c in node.children:
            if isinstance(c, str):
                buf.append(inline(c))
            else:
                buf.append(self.inline_node(c))
        s = "".join(buf)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        return s.strip()

    def inline_node(self, n):
        t = n.tag
        if t in ("b", "strong"):
            inner = self.inline_children(n)
            return "**" + inner + "**" if inner else ""
        if t in ("i", "em"):
            inner = self.inline_children(n)
            return "*" + inner + "*" if inner else ""
        if t == "code":
            inner = text_of(n).replace("`", "'")
            return "`" + inner + "`"
        if t == "a":
            inner = self.inline_children(n)
            href = n.attrs.get("href", "")
            return "[" + inner + "](" + href + ")" if href else inner
        if t == "br":
            return "  \n"
        if t == "span":
            c = cls(n)
            if "shape" in c:
                return "`" + text_of(n).strip() + "`"
            if "tok" in c:
                return "`" + text_of(n).strip() + "`"
            inner = self.inline_children(n)
            if "eqno" in c:
                return " (" + text_of(n).strip() + ")"
            return inner
        if t == "sup":
            return "<sup>" + text_of(n) + "</sup>"
        if t == "sub":
            return "<sub>" + text_of(n) + "</sub>"
        if t in ("svg", "canvas", "input", "select", "button", "style", "script"):
            return ""
        return self.inline_children(n)

    # ---------- 块 ----------
    def block(self, node, depth=0):
        for c in node.children:
            if isinstance(c, str):
                s = inline(c)
                if s:
                    self.out.append(s)
                continue
            self.block_node(c, depth)

    def block_node(self, n, depth=0):
        t = n.tag
        c = cls(n)

        if t == "h1":
            self.out.append("# " + flat(self.inline_children(n)))
            self.out.append("")
            return
        if t == "div" and "hero-meta" in c:
            chips = [text_of(g).strip() for g in n.children
                     if not isinstance(g, str) and has(g, "chip")]
            if chips:
                self.out.append(" · ".join(chips))
                self.out.append("")
            return
        if t == "div" and "eyebrow" in c:
            self.out.append("*" + flat(text_of(n)) + "*")
            self.out.append("")
            return
        if t in ("section", "div") and ("chapter-head" in c or "chapter" in c or "hero" in c):
            if "chapter-head" in c:
                self.chapter_head(n)
            else:
                self.block(n, depth)
            return

        if t == "h2":
            self.out.append("# " + flat(self.inline_children(n)))
            self.out.append("")
            return
        if t == "h3":
            self.out.append("## " + flat(self.inline_children(n)))
            self.out.append("")
            return
        if t == "h4":
            self.out.append("### " + flat(self.inline_children(n)))
            self.out.append("")
            return
        if t == "p":
            if "lead" in c:
                self.out.append("> " + self.inline_children(n))
                self.out.append("")
                return
            s = self.inline_children(n)
            if s:
                self.out.append(s)
                self.out.append("")
            return
        if t in ("ul", "ol"):
            self.list_block(n)
            return
        if t == "table":
            self.table(n)
            return
        if t == "div":
            if "equation" in c:
                self.equation(n)
                return
            if "calc" in c:
                self.calc(n)
                return
            if "compare" in c:
                self.compare_cards(n)
                return
            if any(k in c for k in ("keyidea", "note", "warn", "modern")):
                self.callout(n)
                return
            if "demo" in c:
                self.demo(n)
                return
            if "tablewrap" in c:
                self.block(n, depth)
                return
            if "steps" in c:
                self.list_block(n)
                return
            self.block(n, depth)
            return
        if t == "pre":
            self.code(n)
            return
        if t == "figure":
            self.figure(n)
            return
        if t == "details":
            self.details(n)
            return
        if t == "blockquote":
            self.out.append("> " + self.inline_children(n))
            self.out.append("")
            return
        if t in ("svg", "canvas", "style", "script"):
            return
        if t in ("b", "strong", "i", "em", "code", "a", "span", "cite", "small"):
            s_inline = self.inline_node(n)
            if s_inline:
                self.out.append(s_inline)
                self.out.append("")
            return
        self.block(n, depth)

    def chapter_head(self, n):
        para = []
        for c in n.children:
            if isinstance(c, str):
                continue
            if has(c, "chapter-kicker"):
                self.out.append("*" + text_of(c).strip() + "*")
                self.out.append("")
            elif c.tag == "h2":
                self.out.append("# " + flat(self.inline_children(c)))
                self.out.append("")
            elif c.tag == "p":
                s = self.inline_children(c)
                if s:
                    para.append(s)
        for s in para:
            self.out.append(s)
            self.out.append("")

    def equation(self, n):
        tex = ""
        for c in n.children:
            if isinstance(c, str):
                tex += c
            elif not has(c, "eqno"):
                tex += text_of(c)
        m = MATH_DISPLAY.search(tex) or MATH_INLINE.search(tex)
        body = m.group(1).strip() if m else tex.strip()
        eqno = ""
        for c in n.children:
            if not isinstance(c, str) and has(c, "eqno"):
                eqno = text_of(c).strip()
        if eqno:
            body = body + r"\qquad (" + eqno + ")"
        self.out.append("$$")
        self.out.append(body)
        self.out.append("$$")
        self.out.append("")

    def calc(self, n):
        rows = []
        for c in n.children:
            if isinstance(c, str):
                continue
            if has(c, "calc-title"):
                self.out.append("**" + self.inline_children(c) + "**")
                self.out.append("")
            elif has(c, "calc-row"):
                label = val = ""
                for g in c.children:
                    if isinstance(g, str):
                        continue
                    if has(g, "calc-label"):
                        label = self.inline_children(g)
                    elif has(g, "calc-val"):
                        val = self.inline_children(g)
                rows.append((label, val))
        if rows:
            self.out.append("| 项 | 值 |")
            self.out.append("| --- | --- |")
            for a, b in rows:
                self.out.append("| " + escape_cell(a) + " | " + escape_cell(b) + " |")
            self.out.append("")

    def compare_cards(self, n):
        for c in n.children:
            if isinstance(c, str) or c.tag != "div":
                continue
            title = ""
            rest = []
            for g in c.children:
                if isinstance(g, str):
                    continue
                if g.tag == "h4":
                    title = self.inline_children(g)
                elif g.tag == "p":
                    rest.append(self.inline_children(g))
                elif g.tag in ("ul", "ol"):
                    for li in g.children:
                        if not isinstance(li, str) and li.tag == "li":
                            rest.append("- " + self.inline_children(li))
            self.out.append("> **" + title + "**")
            for r in rest:
                for line in r.split("\n"):
                    self.out.append("> " + line if line else ">")
            self.out.append("")

    def callout(self, n):
        label = ""
        bodies = []
        for c in n.children:
            if isinstance(c, str):
                continue
            if c.tag == "b" and not label:
                label = self.inline_children(c)
            elif c.tag == "p":
                bodies.append(self.inline_children(c))
            elif c.tag in ("ul", "ol"):
                for li in c.children:
                    if not isinstance(li, str) and li.tag == "li":
                        bodies.append("- " + self.inline_children(li))
        if label:
            self.out.append("> **" + label + "**")
            self.out.append(">")
        for b in bodies:
            for line in b.split("\n"):
                self.out.append("> " + line if line else ">")
        if label or bodies:
            self.out.append("")

    def list_block(self, n):
        ordered = n.tag == "ol"
        i = 0
        for c in n.children:
            if isinstance(c, str) or c.tag != "li":
                continue
            i += 1
            s = self.inline_children(c).replace("\n", " ")
            if ordered:
                self.out.append(str(i) + ". " + s)
            else:
                self.out.append("- " + s)
        if i:
            self.out.append("")

    def table(self, n):
        header, rows = [], []
        for sec in n.children:
            if isinstance(sec, str):
                continue
            if sec.tag == "thead":
                for tr in sec.children:
                    if isinstance(tr, str) or tr.tag != "tr":
                        continue
                    header = [escape_cell(self.inline_children(td))
                              for td in tr.children
                              if not isinstance(td, str) and td.tag in ("th", "td")]
            elif sec.tag == "tbody":
                for tr in sec.children:
                    if isinstance(tr, str) or tr.tag != "tr":
                        continue
                    rows.append([escape_cell(self.inline_children(td))
                                 for td in tr.children
                                 if not isinstance(td, str) and td.tag in ("td", "th")])
        if not header and rows:
            header = ["" for _ in rows[0]]
        if not header:
            return
        self.out.append("| " + " | ".join(header) + " |")
        self.out.append("| " + " | ".join(["---"] * len(header)) + " |")
        for r in rows:
            r = r + [""] * (len(header) - len(r))
            self.out.append("| " + " | ".join(r[:len(header)]) + " |")
        self.out.append("")

    def code(self, n):
        raw = text_of(n)
        first = raw.strip().split("\n")[0]
        lang = "text"
        is_shell = ("vllm serve" in raw or "sglang" in raw or "pip install" in raw
                    or re.search(r"(?m)^\s*--[a-z]", raw) or re.match(r"^\$ ", first))
        if is_shell:
            lang = "bash"
        elif re.match(r"^\{", first):
            lang = "json"
        elif ("def " in raw or "np." in raw or re.match(r"^(import|from) ", first)
              or re.match(r"^class ", first)):
            lang = "python"
        self.out.append("```" + lang)
        self.out.append(raw.rstrip("\n"))
        self.out.append("```")
        self.out.append("")

    def figure(self, n):
        cap = ""
        for c in n.children:
            if isinstance(c, str):
                continue
            if c.tag == "figcaption":
                cap = self.inline_children(c)
        if cap:
            self.out.append("*图：" + cap + "*")
            self.out.append("")

    def details(self, n):
        q = ""
        body = []
        for c in n.children:
            if isinstance(c, str):
                continue
            if c.tag == "summary":
                q = self.inline_children(c)
            elif has(c, "qa-body") or c.tag == "div":
                sub = Renderer()
                sub.block(c)
                body.append("\n".join(x for x in sub.out if x != ""))
        self.out.append("**" + q + "**")
        self.out.append("")
        for b in body:
            lines = [ln for ln in b.split("\n") if ln.strip()]
            for i, line in enumerate(lines):
                if i == 1:
                    self.out.append(">")
                self.out.append("> " + line)
        self.out.append("")

    def demo(self, n):
        title = ""
        hint = ""
        for c in n.children:
            if isinstance(c, str):
                continue
            if has(c, "demo-head"):
                for g in c.children:
                    if isinstance(g, str):
                        continue
                    if g.tag == "b":
                        title = self.inline_children(g)
                    elif has(g, "hint"):
                        hint = self.inline_children(g)
        self.out.append("> **交互演示" + ("：" + title if title else "") + "**" +
                        ("（" + hint + "）" if hint else "") +
                        " —— 在 HTML 版里可直接拖动 / 点击。")
        self.out.append("")


def render_part(path):
    src = open(path, encoding="utf-8").read()
    p = DomParser()
    p.feed(src)
    r = Renderer()
    r.block(p.root)
    lines = []
    blank = 0
    for line in r.out:
        if line.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        lines.append(line.rstrip())
    text = "\n".join(lines).strip() + "\n"
    text = text.replace(" {{CARDS}}", "> 章节总览见 [README.md](README.md)。").replace("{{CARDS}}", "> 章节总览见 [README.md](README.md)。").replace("{{NAV}}", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def main():
    os.makedirs(OUT, exist_ok=True)
    parts = sorted(f for f in os.listdir(PARTS) if f.endswith(".html"))
    metas = []
    for f in parts:
        src = open(os.path.join(PARTS, f), encoding="utf-8").read()
        m = re.search(r'data-num="([^"]+)"[^>]*data-title="([^"]+)"[^>]*data-desc="([^"]*)"', src)
        num, title, desc = (m.group(1), m.group(2), m.group(3)) if m else ("", "", "")
        md = render_part(os.path.join(PARTS, f))
        outname = f.replace(".html", ".md")
        open(os.path.join(OUT, outname), "w", encoding="utf-8").write(md)
        metas.append((num, title, desc, outname, len(md)))
        print("  %-34s %6d chars  %s" % (outname, len(md), title))

    # 目录
    book = ["# 大模型是怎么想出一个字的", "",
            "一份从零讲到大模型完整推理过程的单文件教程的 **Markdown 版**。所有“逐层演算”的数字都来自本地真实训练的小模型（2 层、宽度 8、1608 个参数），固定种子可复现；交互演示只存在于 HTML 版。", "",
            "> 完整版是仓库根目录的 `index.html`（离线单文件，含 10 个可交互演示）。本目录是方便阅读、检索与引用而生成的纯文本版。", "",
            "## 目录", "", "| 章 | 标题 | 简介 |", "| --- | --- | --- |"]
    for num, title, desc, outname, _ in metas:
        book.append("| %s | [%s](%s) | %s |" % (num, title, outname, desc.replace("|", "\\|")))
    book.append("")
    book.append("合集：[book.md](book.md)（全部章节拼接，便于全文搜索）。")
    book.append("")
    open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write("\n".join(book))

    # 合集
    chunks = []
    for num, title, desc, outname, _ in metas:
        chunks.append(open(os.path.join(OUT, outname), encoding="utf-8").read())
    full = "\n\n---\n\n".join(chunks)
    open(os.path.join(OUT, "book.md"), "w", encoding="utf-8").write(
        "# 大模型是怎么想出一个字的（合集）\n\n" + full)
    print("md files:", len(metas), "| book.md:", len(full), "chars")


if __name__ == "__main__":
    main()

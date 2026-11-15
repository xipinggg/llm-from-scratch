"""生成单章预览：python3 tools/preview.py ch05 [out]（用于无头截图检查排版）。"""
import re, sys, os
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
html = open(os.path.join(root, "index.html"), encoding="utf-8").read()
target = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else f"/tmp/preview-{target}.html"
marks = [m.start() for m in re.finditer(r'<section class="chapter" id="([^"]+)"', html)]
ids = [m.group(1) for m in re.finditer(r'<section class="chapter" id="([^"]+)"', html)]
if target not in ids:
    sys.exit(f"no such section: {target}; have {ids}")
i = ids.index(target)
start = marks[i]
end = marks[i + 1] if i + 1 < len(marks) else len(html)
body = html[:html.index('<section class="hero"')]
sec = html[start:end]
foot = html[end:] if end < len(html) else ""
new = body + sec + foot
open(out, "w", encoding="utf-8").write(new)
print("wrote", out, len(new) // 1024, "KB")

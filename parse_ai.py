import sys, os, re
with open("ai_output_raw.txt") as f: content = f.read()
pattern = r"=== FILE: (.+?) ===\n(.*?)=== END FILE ==="
matches = re.findall(pattern, content, re.DOTALL)
allowed = ("src/", "config/", "docs/")
count = 0
for path, body in matches:
    path = path.strip()
    if not any(path.startswith(p) for p in allowed): continue
    dirpath = os.path.dirname(path)
    if dirpath: os.makedirs(dirpath, exist_ok=True)
    open(path, "w").write(body)
    print(f"Wrote: {path}", file=sys.stderr)
    count += 1
print(count)

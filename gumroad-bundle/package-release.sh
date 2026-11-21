#!/usr/bin/env bash
# Create only the buyer-facing bundle; never archive the gateway source checkout.
set -euo pipefail

VERSION="1.0.0"
OUT_DIR="${OUT_DIR:-/tmp/free-llm-gateway-starter-kit}"
ARCHIVE="${OUT_DIR}/free-llm-gateway-starter-kit-v${VERSION}.zip"
FILES=(CHANGELOG.md GUMROAD_LISTING.md README.md SETUP_GUIDE.md .env.starter quickstart.sh package-release.sh)
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

die() { printf '[package] ERROR: %s\n' "$*" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || die "python3 is required."

cd "${BUNDLE_DIR}"

for file in "${FILES[@]}"; do
    [ -f "${file}" ] || die "Missing ${file}; run this from gumroad-bundle/."
done

if [ "${1:-}" = "--self-test" ]; then
    python3 - <<'PY'
import re, sys

# Keep in sync with the scanner below.
PATTERNS = [
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"xai-[A-Za-z0-9]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{24,}",
]

def leaks(text):
    return [p for p in PATTERNS if re.search(p, text)]

dirty = 'PROXY_API_KEY="sk-abcdefghijklmnopqrstuvwxyz012345"\n'
clean = 'PROXY_API_KEY=""\n# openssl rand -hex 32\n'
ok = leaks(dirty) and not leaks(clean)
print(f"package-release self-test: {'OK' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
    exit $?
fi

python3 - "${ARCHIVE}" "${FILES[@]}" <<'PY'
from __future__ import annotations

from hashlib import sha256
import re
from pathlib import Path
from sys import argv
from zipfile import ZIP_DEFLATED, ZipFile

archive = Path(argv[1])
files = [Path(value) for value in argv[2:]]
entries = [f"gumroad-bundle/{file.name}" for file in files]

archive.parent.mkdir(parents=True, exist_ok=True)

SECRET_PATTERNS = [
    r"sk-[A-Za-z0-9_\-]{16,}",
    r"gh[pousr]_[A-Za-z0-9]{20,}",
    r"xai-[A-Za-z0-9]{16,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|password|token)\s*[=:]\s*[\"']?[A-Za-z0-9_\-]{24,}",
]

for file in files:
    # ponytail: this script holds the patterns themselves; scanning it self-matches.
    if file.name == "package-release.sh":
        continue
    text = file.read_text(encoding="utf-8", errors="replace")
    hits = [p for p in SECRET_PATTERNS if re.search(p, text)]
    if hits:
        raise SystemExit(f"[package] ERROR: {file.name} looks like it holds a credential: {hits}")

with ZipFile(archive, "w", compression=ZIP_DEFLATED, compresslevel=9) as bundle:
    for file, entry in zip(files, entries, strict=True):
        bundle.write(file, entry)

with ZipFile(archive) as bundle:
    missing = set(entries) - set(bundle.namelist())
    if missing:
        raise SystemExit(f"[package] ERROR: archive omitted {sorted(missing)!r}")
    bad_file = bundle.testzip()
    if bad_file:
        raise SystemExit(f"[package] ERROR: corrupt archive entry {bad_file}")

digest = sha256(archive.read_bytes()).hexdigest()
checksum = archive.with_suffix(archive.suffix + ".sha256")
checksum.write_text(f"{digest}  {archive.name}\n", encoding="ascii")
print(f"[package] verified {archive}")
print(f"[package] sha256 {checksum}")
PY

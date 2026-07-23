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

python3 - "${ARCHIVE}" "${FILES[@]}" <<'PY'
from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from sys import argv
from zipfile import ZIP_DEFLATED, ZipFile

archive = Path(argv[1])
files = [Path(value) for value in argv[2:]]
entries = [f"gumroad-bundle/{file.name}" for file in files]

archive.parent.mkdir(parents=True, exist_ok=True)
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

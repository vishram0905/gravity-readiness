# AI-assisted code (team statement and per-step log: docs/AI_USE_LOG.md).
"""Download the Human Balance Evaluation Database (PhysioNet, ODC-By 1.0) and verify SHA-256 sums.

Usage:
    python scripts/fetch_data.py --dest data/hbedb            # full database (~370 MB)
    python scripts/fetch_data.py --dest data/hbedb --only BDS00001 BDS00002
Then: export HBEDB_DIR=<dest>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE = "https://physionet.org/files/hbedb/1.0.0/"
ALWAYS = ("SHA256SUMS.txt", "RECORDS", "BDSinfo.txt")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path) -> Path:
    out = dest / name
    if not out.exists():
        tmp = out.with_suffix(out.suffix + ".part")
        with urllib.request.urlopen(BASE + name, timeout=120) as r, open(tmp, "wb") as fh:
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        tmp.replace(out)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--only", nargs="*", help="record names (e.g. BDS00001); default all")
    a = ap.parse_args(argv)
    a.dest.mkdir(parents=True, exist_ok=True)
    for name in ALWAYS:
        fetch(name, a.dest)
    sums = {}
    for line in (a.dest / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, fname = line.split(maxsplit=1)
            sums[fname.strip()] = digest
    records = a.only or (a.dest / "RECORDS").read_text().split()
    wanted = list(ALWAYS[1:]) + [f"{r}{ext}" for r in records for ext in (".hea", ".dat")]
    unlisted = [n for n in wanted if n not in sums]
    if unlisted:
        print(f"No checksum in SHA256SUMS.txt for: {unlisted[:5]}{' ...' if len(unlisted) > 5 else ''}",
              file=sys.stderr)
        return 1
    bad = []
    for i, name in enumerate(wanted, 1):
        path = fetch(name, a.dest)
        if sha256(path) != sums[name]:
            bad.append(name)
            path.unlink()
        if i % 200 == 0:
            print(f"{i}/{len(wanted)} files", flush=True)
    if bad:
        print(f"SHA-256 mismatch (deleted, re-run to retry): {bad}", file=sys.stderr)
        return 1
    print(f"OK: {len(wanted)} files verified in {a.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

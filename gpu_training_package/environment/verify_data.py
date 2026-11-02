#!/usr/bin/env python3
"""Create or verify a SHA-256 inventory of the bundled TLC input files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def inventory(data: Path) -> dict:
    entries = {}
    for path in sorted(p for p in data.rglob("*") if p.is_file()):
        if path.is_symlink():
            raise ValueError(f"Input must be an ordinary file: {path}")
        entries[path.relative_to(data).as_posix()] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    if not entries:
        raise ValueError(f"No input files found in {data}")
    return {"schema_version": 1, "description": "NYC TLC yellow taxi processed input files", "files": entries}


def verify(data: Path, manifest: Path) -> None:
    expected = json.loads(manifest.read_text())
    if expected.get("schema_version") != 1 or not expected.get("files"):
        raise ValueError("Invalid or empty data inventory")
    found = inventory(data)
    if found != expected:
        left, right = expected["files"], found["files"]
        changed = [name for name in sorted(set(left) | set(right)) if left.get(name) != right.get(name)]
        raise ValueError(f"Input inventory mismatch: {changed[:20]}")
    print(f"Verified {len(found['files'])} input files against {manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["create", "check"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "create":
        if args.manifest.exists():
            raise SystemExit("Refusing to overwrite an existing inventory")
        args.manifest.write_text(json.dumps(inventory(args.data), indent=2, sort_keys=True) + "\n")
        print(f"Created {args.manifest}", flush=True)
    else:
        verify(args.data, args.manifest)


if __name__ == "__main__":
    main()

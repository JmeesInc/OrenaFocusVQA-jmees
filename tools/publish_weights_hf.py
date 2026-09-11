#!/usr/bin/env python3
"""Publish the staged container weights to a public Hugging Face model repo.

The trained weights are too large for GitHub (LoRA adapters are ~215 MB each, the
Mask2Former detectors ~823 MB), so they live on the Hub and the containers pull them in.

    python tools/publish_weights_hf.py --track FRAME --repo <org>/OrenaFocusVQA-jmees-weights

Run `container/stage_resources.sh` first: this uploads whatever is under
`<TRACK>/container/resources/`, preserving the directory names the container expects.
Members that share an adapter (FRAME `m1`/`m2`) are uploaded once and recorded as an alias.

Requires `huggingface_hub` and a write token (`huggingface-cli login`, or HF_TOKEN).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(member: Path) -> str:
    """Identify a member by its weight file, so duplicates can be detected."""
    for name in ("adapter_model.safetensors", "model_final.pth", "model.safetensors"):
        f = member / name
        if f.is_file():
            return md5(f)
    files = sorted(p for p in member.rglob("*") if p.is_file())
    return hashlib.md5("".join(f"{p.name}:{p.stat().st_size}" for p in files).encode()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=["FRAME", "SEGMENT", "PROCEDURE"])
    ap.add_argument("--repo", required=True, help="target Hugging Face model repo id")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    res = ROOT / args.track / "container" / "resources"
    if not res.is_dir():
        print(f"ERROR: {res} does not exist. Run container/stage_resources.sh first.", file=sys.stderr)
        return 1
    members = sorted(p for p in res.iterdir() if p.is_dir())
    if not members:
        print(f"ERROR: no member directories under {res}", file=sys.stderr)
        return 1

    seen: dict[str, str] = {}
    upload: list[tuple[Path, str]] = []
    manifest: dict[str, dict[str, str]] = {}
    for m in members:
        fp = fingerprint(m)
        if fp in seen:
            manifest[m.name] = {"alias_of": seen[fp], "md5": fp}
            print(f"  = {m.name:20s} identical to {seen[fp]} (uploaded once)")
            continue
        seen[fp] = m.name
        upload.append((m, f"{args.track.lower()}/{m.name}"))
        manifest[m.name] = {"path": f"{args.track.lower()}/{m.name}", "md5": fp}
        size = sum(p.stat().st_size for p in m.rglob("*") if p.is_file())
        print(f"  + {m.name:20s} {size / 2**20:7.0f} MiB  md5 {fp}")

    (ROOT / args.track / "container" / "weights_manifest.json").write_text(
        json.dumps({"repo": args.repo, "members": manifest}, indent=2) + "\n"
    )
    print(f"wrote {args.track}/container/weights_manifest.json")
    if args.dry_run:
        print("dry run: nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    for src, dest in upload:
        print(f"uploading {src.name} -> {dest}")
        api.upload_folder(repo_id=args.repo, repo_type="model", folder_path=str(src), path_in_repo=dest)
    api.upload_file(
        repo_id=args.repo, repo_type="model",
        path_or_fileobj=str(ROOT / args.track / "container" / "weights_manifest.json"),
        path_in_repo=f"{args.track.lower()}/weights_manifest.json",
    )
    print(f"done: https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Upload the collected frontier-model (GPT-5.4 / Claude Sonnet 4.6) generations to a
PRIVATE Hugging Face dataset repo.

The frontier comparison is routed through a commercial API and is not bit-for-bit
reproducible, so we release the raw generations. This script creates (if needed) a
private dataset repo and uploads the merged CSV.

Run it yourself with a Hugging Face *write* token (it is not stored in the repo):

    HF_TOKEN=hf_xxx python scripts/upload_frontier_to_hf.py
    # or:  python scripts/upload_frontier_to_hf.py --token hf_xxx --repo Despina/frontier-re-generations

Make the repo public later from the Hugging Face web UI (Settings -> Change visibility)
when the paper is accepted.
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = REPO_ROOT / "runs" / "frontier_generations" / "all_frontier_generations.csv"


def main():
    ap = argparse.ArgumentParser(description="Upload frontier generations to a private HF dataset")
    ap.add_argument("--repo", default="Despina/frontier-re-generations",
                    help="target HF dataset repo id (default: Despina/frontier-re-generations)")
    ap.add_argument("--file", default=str(DEFAULT_FILE), help="path to the merged frontier CSV")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"),
                    help="HF write token (or set HF_TOKEN)")
    ap.add_argument("--public", action="store_true", help="create the repo public (default: private)")
    args = ap.parse_args()

    if not args.token:
        sys.exit("ERROR: no Hugging Face token. Set HF_TOKEN or pass --token (a write token).")
    f = Path(args.file)
    if not f.exists():
        sys.exit(f"ERROR: {f} not found. Generate frontier outputs first "
                 f"(scripts/run_frontier_generations.py).")

    from huggingface_hub import HfApi
    api = HfApi(token=args.token)
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"Uploading {f} ({f.stat().st_size/1e6:.1f} MB) -> {args.repo} ...")
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=args.repo, repo_type="dataset",
                    commit_message="Add frontier-model RE generations")
    vis = "public" if args.public else "private"
    print(f"Done. {vis} dataset at https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

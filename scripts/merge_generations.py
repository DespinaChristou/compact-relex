#!/usr/bin/env python3
"""
Merge per-dataset generation shards into a single generations.csv.

Expected layout:
  generations/
    <dataset_name>/
      generations_shard_0.csv
      generations_shard_1.csv

Output (per dataset):
  generations/
    <dataset_name>/
      generations.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_dataset_folder(dataset_dir: Path, *, shards: tuple[int, ...], out_name: str, overwrite: bool) -> bool:
    shard_paths = [dataset_dir / f"generations_shard_{i}.csv" for i in shards]
    existing = [p for p in shard_paths if p.exists()]

    if not existing:
        print(f"skip  : {dataset_dir.name} (no shard files found)")
        return False

    out_path = dataset_dir / out_name
    if out_path.exists() and not overwrite:
        print(f"skip  : {dataset_dir.name} ({out_name} exists; use --overwrite)")
        return False

    frames: list[pd.DataFrame] = []
    columns_ref: list[str] | None = None

    print(f"\nDataset: {dataset_dir.name} ({len(existing)} shard files)")
    for p in existing:
        df = pd.read_csv(p)
        print(f"    {p.name}: {len(df):,} rows")
        if columns_ref is None:
            columns_ref = list(df.columns)
        elif list(df.columns) != columns_ref:
            raise ValueError(
                f"[{dataset_dir.name}] Column mismatch in {p.name}.\n"
                f"Expected: {columns_ref}\n"
                f"Got     : {list(df.columns)}"
            )
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False, encoding="utf-8")
    print(f"merged: {dataset_dir.name} -> {out_path} ({len(merged):,} rows)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("runs") / "generations",
        help="Root folder containing per-dataset subfolders (default: generations)",
    )
    ap.add_argument(
        "--shards",
        default="0,1",
        help="Comma-separated shard indices to merge (default: 0,1)",
    )
    ap.add_argument(
        "--out",
        default="generations.csv",
        help="Output filename to write inside each dataset folder (default: generations.csv)",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    root: Path = args.root
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--root must be an existing directory: {root}")

    shards = tuple(int(x.strip()) for x in args.shards.split(",") if x.strip() != "")

    dataset_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not dataset_dirs:
        raise SystemExit(f"No dataset subfolders found under: {root}")

    merged_any = False
    for d in dataset_dirs:
        merged_any |= merge_dataset_folder(d, shards=shards, out_name=args.out, overwrite=args.overwrite)

    if not merged_any:
        raise SystemExit("No outputs written (nothing to merge, or all outputs already existed).")


if __name__ == "__main__":
    main()
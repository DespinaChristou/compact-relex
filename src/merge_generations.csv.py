"""
Merge per-worker generation shard files into a single file per evaluation dataset.

This script complements src.run_generations, which writes shard files to avoid
concurrent writes when running multiple workers (job_count > 1).

Inputs (per eval dataset)
-------------------------
For each eval dataset name (from configs/generations.yaml: eval_datasets[].name),
we expect files under:

  runs/<output_subdir>/<eval_dataset_name>/generations_shard_*.{csv|parquet}

The extension is determined by configs/generations.yaml: output.format.

Outputs (per eval dataset)
--------------------------
We write:

  runs/<output_subdir>/<eval_dataset_name>/generations.{csv|parquet}

Enhancements / safety
---------------------
- Uses the YAML config to determine:
  - output directory
  - eval dataset list
  - output format (csv/parquet)
- Validates that required columns exist in merged output.
- Optional deduplication (exact-row dedup) and stable sorting.
- Can merge only a subset of eval datasets via CLI.
- Emits helpful warnings if shard files are missing.

Usage
-----
python -m src.merge_generations --config configs/generations.yaml

Optional:
  --only tacred,gids        merge only these eval datasets (names)
  --dedup                  drop exact duplicate rows
  --sort                   sort merged output for stable downstream diffs
  --strict                 fail if any expected eval dataset has no shard files
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml


REQUIRED_COLUMNS = [
    "eval_dataset_name",
    "prompt_0_shot",
    "prompt_2_shot",
    "relation",
    "gen_type",
    "model_id",
    "tuned_dataset_name",
    "model_shot",
    "prompt_shot",
    "generated_relation",
]


SORT_COLUMNS = [
    # group/context
    "eval_dataset_name",
    "tuned_dataset_name",
    "model_id",
    "model_shot",
    "prompt_shot",
    "gen_type",
    # content
    "relation",
    "generated_relation",
]


def _read_config(path: str) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _get_output_root(cfg: Dict[str, Any]) -> Path:
    runs_dir = Path(cfg["paths"]["runs_dir"])
    out_subdir = str(cfg["paths"].get("output_subdir", "generations"))
    return runs_dir / out_subdir


def _get_output_format(cfg: Dict[str, Any]) -> str:
    out_cfg = dict(cfg.get("output", {}))
    fmt = str(out_cfg.get("format", "csv")).strip().lower()
    if fmt not in {"csv", "parquet"}:
        raise ValueError("config.output.format must be one of: csv | parquet")
    return fmt


def _eval_dataset_names(cfg: Dict[str, Any]) -> List[str]:
    ds = list(cfg.get("eval_datasets", []))
    names = [str(d.get("name", "")).strip() for d in ds]
    if any(not n for n in names):
        raise ValueError(f"Found empty eval_datasets[].name entry: {names}")
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate eval_datasets[].name entries found: {names}")
    return names


def _parse_only(only: Optional[str]) -> Optional[set[str]]:
    if not only:
        return None
    parts = [p.strip() for p in only.split(",")]
    parts = [p for p in parts if p]
    return set(parts) if parts else None


def _glob_shards(eval_dir: Path, fmt: str) -> List[Path]:
    return sorted(eval_dir.glob(f"generations_shard_*.{fmt}"))


def _read_shard(path: Path, fmt: str) -> pd.DataFrame:
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported format: {fmt}")


def _write_merged(df: pd.DataFrame, out_path: Path, fmt: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(out_path, index=False, encoding="utf-8")
        return
    if fmt == "parquet":
        df.to_parquet(out_path, index=False)
        return
    raise ValueError(f"Unsupported format: {fmt}")


def _validate_columns(df: pd.DataFrame, required: Sequence[str], *, context: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"[{context}] Missing required columns: {missing}. Available: {list(df.columns)}")


def _merge_one_eval_dataset(
    *,
    eval_name: str,
    eval_dir: Path,
    fmt: str,
    dedup: bool,
    sort_rows: bool,
) -> Path:
    shard_paths = _glob_shards(eval_dir, fmt)

    if not shard_paths:
        raise FileNotFoundError(f"No shard files found for eval dataset '{eval_name}' under: {eval_dir}")

    frames: List[pd.DataFrame] = []
    for p in shard_paths:
        df = _read_shard(p, fmt)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)

    # Basic schema validation (prevents silently merging wrong/old shard files).
    _validate_columns(merged, REQUIRED_COLUMNS, context=f"{eval_name}:merged")

    # Sanity: all rows should point to the directory's eval_name (unless you intentionally mix).
    # We do not require strict equality by default, but it's a strong signal if something is off.
    if "eval_dataset_name" in merged.columns:
        distinct = set(str(x).strip() for x in merged["eval_dataset_name"].dropna().unique().tolist())
        if distinct and (distinct != {eval_name}):
            # Keep as warning-like behavior by raising only if it's clearly wrong:
            # If it contains more than one dataset, that likely indicates wrong merges.
            raise ValueError(
                f"[{eval_name}] Merged shards contain eval_dataset_name values {sorted(distinct)}, "
                f"expected only '{eval_name}'. Check your shard placement."
            )

    if dedup:
        merged = merged.drop_duplicates()

    if sort_rows:
        cols = [c for c in SORT_COLUMNS if c in merged.columns]
        if cols:
            merged = merged.sort_values(cols, kind="mergesort").reset_index(drop=True)

    out_path = eval_dir / f"generations.{fmt}"
    _write_merged(merged, out_path, fmt)
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/generations.yaml")
    ap.add_argument(
        "--only",
        default=None,
        help="Comma-separated eval dataset names to merge (e.g., 'tacred,gids'). Default: all from config.",
    )
    ap.add_argument(
        "--dedup",
        action="store_true",
        help="Drop exact duplicate rows after concatenation (useful if some shards were re-run).",
    )
    ap.add_argument(
        "--sort",
        action="store_true",
        help="Sort merged output for stable diffs (by dataset/model/shot/gen_type...).",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any expected eval dataset has no shard files.",
    )
    args = ap.parse_args()

    cfg = _read_config(args.config)
    fmt = _get_output_format(cfg)
    out_root = _get_output_root(cfg)

    expected_eval_names = _eval_dataset_names(cfg)
    only_set = _parse_only(args.only)
    if only_set is not None:
        unknown = sorted(only_set - set(expected_eval_names))
        if unknown:
            raise ValueError(f"--only contains unknown eval dataset names: {unknown}. Known: {expected_eval_names}")
        eval_names = [n for n in expected_eval_names if n in only_set]
    else:
        eval_names = expected_eval_names

    any_errors = False
    for eval_name in eval_names:
        eval_dir = out_root / eval_name
        try:
            out_path = _merge_one_eval_dataset(
                eval_name=eval_name,
                eval_dir=eval_dir,
                fmt=fmt,
                dedup=bool(args.dedup),
                sort_rows=bool(args.sort),
            )
            print(f"✅ merged: {eval_name} -> {out_path}")
        except FileNotFoundError as e:
            if args.strict:
                raise
            any_errors = True
            print(f"⚠️  skip: {eval_name} ({e})")

    if any_errors and not args.strict:
        print("Some eval datasets had no shards; re-run with --strict to fail instead.")


if __name__ == "__main__":
    main()
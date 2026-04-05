#!/usr/bin/env python3
"""
Evaluation runner — reads generation CSVs, computes all metrics, and writes:

  runs/evaluation/per_dataset_metrics.csv    (one row per model-config × dataset × gen_type)
  runs/evaluation/per_class/<dataset>/<model_config>.csv   (per-label breakdown)

Usage:
    python scripts/run_evaluation.py --config configs/eval.yaml
    python scripts/run_evaluation.py --config configs/eval.yaml --datasets conll04 biographical
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval import (
    compute_per_class_metrics,
    evaluate_slice,
    normalize_relation,
)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(cfg_path: str) -> Path:
    """Resolve a path relative to REPO_ROOT."""
    p = Path(cfg_path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p


def _model_config_key(row: dict) -> str:
    """Deterministic string key for a unique model configuration."""
    return (
        f"{row['model_id']}|{row['tuned_dataset_name']}"
        f"|ms{row['model_shot']}|ps{row['prompt_shot']}"
        f"|{row['gen_type']}"
    )


def _dedup_dataframe(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Detect and fix shard-level duplication.

    The generation pipeline writes per-shard CSVs that are later merged.
    If a shard was accidentally merged twice, some model-config groups will
    have 2× the expected row count.  We detect this by comparing group sizes
    and keep only the first occurrence of each (prompt_hash, model_config) pair.

    IMPORTANT: we do NOT use blanket drop_duplicates() — many test examples
    legitimately share the same (relation, generated_relation) pair.
    The prompt_hash column distinguishes individual test examples.
    """
    group_cols = ["model_id", "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"]
    sizes = df.groupby(group_cols).size()

    # Find the expected (modal) group size per gen_type
    needs_dedup = False
    for gen_type in df["gen_type"].unique():
        gt_mask = sizes.index.get_level_values("gen_type") == gen_type
        gt_sizes = sizes[gt_mask]
        if gt_sizes.nunique() > 1:
            modal_size = int(gt_sizes.mode().iloc[0])
            outliers = gt_sizes[gt_sizes != modal_size]
            # Only flag if some groups are exact multiples (shard duplication)
            for idx, sz in outliers.items():
                if sz > modal_size and sz % modal_size == 0:
                    needs_dedup = True
                    multiplier = sz // modal_size
                    print(
                        f"  ⚠ {dataset_name} [{gen_type}]: group {idx} has {sz} rows "
                        f"(expected ~{modal_size}, {multiplier}× duplicate shards detected)"
                    )

    if needs_dedup:
        before = len(df)
        # Use prompt_hash + group_cols to identify true duplicates
        dedup_cols = ["prompt_hash"] + group_cols + ["relation"]
        dedup_cols = [c for c in dedup_cols if c in df.columns]
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        after = len(df)
        if before != after:
            print(f"  → shard dedup: {before:,} → {after:,} (removed {before - after:,} rows)")

    # Final sanity check
    sizes_after = df.groupby(group_cols).size()
    for gen_type in df["gen_type"].unique():
        gt_mask = sizes_after.index.get_level_values("gen_type") == gen_type
        gt_sizes = sizes_after[gt_mask]
        if gt_sizes.nunique() > 1:
            print(
                f"  ℹ {dataset_name} [{gen_type}]: group sizes after dedup — "
                f"min={gt_sizes.min()}, max={gt_sizes.max()}, unique={gt_sizes.nunique()}"
            )

    return df


def _get_allowed_labels(df: pd.DataFrame) -> set:
    """Extract the full set of gold relation labels for schema-valid rate."""
    labels = set(df["relation"].dropna().unique())
    return labels


def _matched_prompt_shot(model_shot: int) -> int:
    """Return the matched prompt_shot for a given model_shot."""
    return model_shot  # 0→0, 2→2


# ────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    dataset_name: str,
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    per_class_dir: Path,
) -> pd.DataFrame:
    """Evaluate all model configurations for one eval dataset.

    Returns a DataFrame of per-config metrics (one row each).
    """
    csv_path = generations_dir / dataset_name / "generations.csv"
    if not csv_path.exists():
        print(f"  ✗ {dataset_name}: no generations.csv found, skipping")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"Evaluating: {dataset_name}")
    print(f"{'='*60}")

    t0 = time.time()

    # --- Load data ---
    # We load the metric columns PLUS prompt_0_shot (as a unique example identifier).
    # The prompt text is hashed immediately to save memory, then the column is dropped.
    usecols = cfg.get("processing", {}).get("usecols", None)
    # Always include prompt_0_shot for dedup even if not in config usecols
    load_cols = list(usecols) if usecols else None
    if load_cols and "prompt_0_shot" not in load_cols:
        load_cols = load_cols + ["prompt_0_shot"]
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)

    # Read in chunks and concatenate to handle large files
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=load_cols, chunksize=chunk_size):
        # Hash the prompt for memory-efficient dedup key
        chunk["prompt_hash"] = chunk["prompt_0_shot"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()[:12]
        )
        chunk = chunk.drop(columns=["prompt_0_shot"], errors="ignore")
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # --- Deduplication ---
    df = _dedup_dataframe(df, dataset_name)

    # --- Get allowed labels and exclusion list ---
    allowed_labels = _get_allowed_labels(df)
    exclude_labels_list = cfg.get("exclude_labels", {}).get(dataset_name, [])
    exclude_labels = set(exclude_labels_list) if exclude_labels_list else None
    do_normalize = cfg.get("normalize", True)

    print(f"  Gold labels: {len(allowed_labels)}, exclude: {exclude_labels or 'none'}")

    # --- Build allowed (model_shot, prompt_shot) pairs ---
    policy = cfg.get("prompt_shot_policy", "matched")
    available_combos = df[["model_shot", "prompt_shot"]].drop_duplicates().values.tolist()

    if policy == "matched":
        valid_combos = {(ms, _matched_prompt_shot(ms)) for ms, ps in available_combos}
        # Only keep combos that actually exist in data
        valid_combos = {(ms, ps) for ms, ps in valid_combos if [ms, ps] in available_combos}
    else:
        valid_combos = {(ms, ps) for ms, ps in available_combos}

    print(f"  Shot combos ({policy}): {sorted(valid_combos)}")

    # --- Group and evaluate ---
    group_cols = ["model_id", "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"]
    results = []

    grouped = df.groupby(group_cols)
    n_groups = len(grouped)
    print(f"  Processing {n_groups} model configurations...")

    per_class_dataset_dir = per_class_dir / dataset_name
    per_class_dataset_dir.mkdir(parents=True, exist_ok=True)

    for i, (key, group) in enumerate(grouped):
        model_id, tuned_ds, m_shot, p_shot, gen_type = key

        gold = group["relation"].tolist()
        pred = group["generated_relation"].tolist()

        # Evaluate all metrics
        metrics = evaluate_slice(
            gold=gold,
            pred=pred,
            allowed_labels=allowed_labels,
            exclude_labels=exclude_labels,
            normalize=do_normalize,
        )

        row = {
            "eval_dataset_name": dataset_name,
            "model_id": model_id,
            "tuned_dataset_name": tuned_ds,
            "model_shot": int(m_shot),
            "prompt_shot": int(p_shot),
            "gen_type": gen_type,
            **metrics,
        }
        results.append(row)

        # Per-class breakdown (only for primary gen_type + matched shots)
        primary_gt = cfg.get("primary_gen_type", "gen_constrained")
        is_matched = (int(m_shot), int(p_shot)) in valid_combos
        if gen_type == primary_gt and is_matched:
            gold_n = [normalize_relation(g) for g in gold] if do_normalize else gold
            pred_n = [normalize_relation(p) for p in pred] if do_normalize else pred
            pc_df = compute_per_class_metrics(gold_n, pred_n, exclude_labels=exclude_labels)
            if not pc_df.empty:
                safe_name = f"{model_id}__{tuned_ds}__ms{int(m_shot)}_ps{int(p_shot)}.csv"
                pc_df.to_csv(per_class_dataset_dir / safe_name, index=False)

    elapsed = time.time() - t0
    print(f"  ✓ Done: {len(results)} configs evaluated in {elapsed:.1f}s")

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on generation outputs")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval config")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Specific datasets to evaluate (default: all found in generations_dir)"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing per_dataset_metrics.csv"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)

    generations_dir = _resolve_path(cfg["paths"]["generations_dir"])
    output_dir = _resolve_path(cfg["paths"]["output_dir"])
    per_class_dir = _resolve_path(cfg["paths"]["per_class_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)
    per_class_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "per_dataset_metrics.csv"
    if output_csv.exists() and not args.overwrite:
        print(f"Output already exists: {output_csv}")
        print("Use --overwrite to regenerate.")
        return

    # Discover datasets
    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d.name for d in generations_dir.iterdir()
            if d.is_dir() and (d / "generations.csv").exists()
        )

    print(f"Datasets to evaluate: {dataset_names}")
    print(f"Output: {output_csv}")
    print(f"Per-class: {per_class_dir}")

    # Evaluate each dataset
    all_results = []
    t_total = time.time()

    for ds_name in dataset_names:
        result_df = evaluate_dataset(
            dataset_name=ds_name,
            generations_dir=generations_dir,
            cfg=cfg,
            output_dir=output_dir,
            per_class_dir=per_class_dir,
        )
        if not result_df.empty:
            all_results.append(result_df)

    if not all_results:
        print("\nNo results produced. Check that generation files exist.")
        return

    final_df = pd.concat(all_results, ignore_index=True)

    # Sort for readability
    final_df = final_df.sort_values(
        ["eval_dataset_name", "model_id", "tuned_dataset_name",
         "model_shot", "prompt_shot", "gen_type"]
    ).reset_index(drop=True)

    final_df.to_csv(output_csv, index=False)
    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(final_df)} rows written to {output_csv}")
    print(f"Total time: {time.time()-t_total:.1f}s")
    print(f"{'='*60}")

    # Summary statistics
    primary_gt = cfg.get("primary_gen_type", "gen_constrained")
    primary = final_df[final_df["gen_type"] == primary_gt]
    if not primary.empty:
        print(f"\nSummary ({primary_gt}, all configs):")
        print(f"  Mean micro-F1:  {primary['micro_f1'].mean():.4f}")
        print(f"  Mean macro-F1:  {primary['macro_f1'].mean():.4f}")
        print(f"  Mean schema-valid rate: {primary['schema_valid_rate'].mean():.4f}")
        print(f"  Mean malformed rate:    {primary['malformed_rate'].mean():.4f}")


if __name__ == "__main__":
    main()

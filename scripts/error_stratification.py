#!/usr/bin/env python3
"""
Error stratification analysis — quantifies error types by model scale.

Produces:
  runs/evaluation/statistical/
    error_stratification.csv      — error counts by category × scale group
    error_stratification_pct.csv  — same as percentages

Supports the qualitative error analysis in Section 4.5 with quantitative data.

Usage:
    python scripts/error_stratification.py --config configs/eval.yaml
    python scripts/error_stratification.py --config configs/eval.yaml --datasets conll04 biographical
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval import normalize_relation


# ═══════════════════════════════════════════════════════════════════════
# Error classification
# ═══════════════════════════════════════════════════════════════════════

# Negative / "Other" labels that indicate under-prediction
NEGATIVE_LABELS = {
    "other", "no_relation", "no relation", "none", "na", "n/a",
    "no_relation", "unanswerable", "false",
}

# Scale groups
SCALE_GROUPS = {
    "SmolLM2-360M-Instruct": ("sub-billion", 0.36),
    "Qwen2.5-0.5B-Instruct": ("sub-billion", 0.5),
    "SmolLM3-3B": ("3B", 3.0),
    "Qwen2.5-3B-Instruct": ("3B", 3.0),
    "Llama-3.2-3B-Instruct": ("3B", 3.2),
}


def classify_error(
    gold: str,
    pred: str,
    allowed_labels: Set[str],
) -> str:
    """Classify a single (gold, pred) error into categories.

    Categories:
      - "correct"         : gold == pred
      - "near_neighbor"   : pred is a valid label but wrong (same schema)
      - "default_negative": pred is a negative/other label when gold is positive
      - "hallucinated"    : pred is not in the allowed label set
      - "negative_miss"   : pred is positive but gold is negative (false positive)
      - "other_error"     : catchall
    """
    if gold == pred:
        return "correct"

    gold_is_negative = gold in NEGATIVE_LABELS
    pred_is_negative = pred in NEGATIVE_LABELS
    pred_is_valid = pred in allowed_labels

    # Default to negative: model predicts Other/no_relation when gold is a real relation
    if not gold_is_negative and pred_is_negative:
        return "default_negative"

    # Hallucinated label: prediction not in allowed set
    if not pred_is_valid and not pred_is_negative:
        return "hallucinated"

    # Near-neighbor: both gold and pred are valid labels, just wrong
    if pred_is_valid and not pred_is_negative and not gold_is_negative:
        return "near_neighbor"

    # False positive: gold is negative but pred is a real relation
    if gold_is_negative and not pred_is_negative:
        return "false_positive"

    return "other_error"


# ═══════════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════════

def analyze_dataset(
    dataset_name: str,
    generations_dir: Path,
    cfg: dict,
) -> pd.DataFrame:
    """Analyze error types for one dataset across all model configs."""
    csv_path = generations_dir / dataset_name / "generations.csv"
    if not csv_path.exists():
        print(f"  ✗ {dataset_name}: no generations.csv")
        return pd.DataFrame()

    print(f"\n  Analyzing: {dataset_name}")
    t0 = time.time()

    usecols = cfg.get("processing", {}).get("usecols", None)
    load_cols = list(usecols) if usecols else None
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=load_cols, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Filter to constrained + matched
    mask = (df["gen_type"] == "gen_constrained") & (df["model_shot"] == df["prompt_shot"])
    df = df[mask].copy()

    if df.empty:
        return pd.DataFrame()

    do_normalize = cfg.get("normalize", True)
    if do_normalize:
        df["relation"] = df["relation"].apply(normalize_relation)
        df["generated_relation"] = df["generated_relation"].apply(normalize_relation)

    # Get allowed labels
    allowed_labels = set(df["relation"].unique())
    allowed_labels_norm = {normalize_relation(l) for l in allowed_labels} if do_normalize else allowed_labels

    # Classify each prediction
    df["error_type"] = df.apply(
        lambda row: classify_error(row["relation"], row["generated_relation"], allowed_labels_norm),
        axis=1,
    )

    # Add scale group
    df["scale_group"] = df["model_id"].map(lambda m: SCALE_GROUPS.get(m, ("unknown", 0))[0])
    df["params_b"] = df["model_id"].map(lambda m: SCALE_GROUPS.get(m, ("unknown", 0))[1])

    # Aggregate
    rows = []
    for (model_id, tuned_ds, m_shot, scale_group), group in df.groupby(
        ["model_id", "tuned_dataset_name", "model_shot", "scale_group"]
    ):
        total = len(group)
        error_counts = group["error_type"].value_counts().to_dict()

        rows.append({
            "dataset": dataset_name,
            "model_id": model_id,
            "tuned_dataset_name": tuned_ds,
            "model_shot": int(m_shot),
            "scale_group": scale_group,
            "params_b": SCALE_GROUPS.get(model_id, ("unknown", 0))[1],
            "total": total,
            "correct": error_counts.get("correct", 0),
            "near_neighbor": error_counts.get("near_neighbor", 0),
            "default_negative": error_counts.get("default_negative", 0),
            "hallucinated": error_counts.get("hallucinated", 0),
            "false_positive": error_counts.get("false_positive", 0),
            "other_error": error_counts.get("other_error", 0),
        })

    elapsed = time.time() - t0
    print(f"    ✓ {len(rows)} configs in {elapsed:.1f}s")

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Error stratification analysis")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--datasets", nargs="*", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generations_dir = Path(cfg["paths"]["generations_dir"])
    if not generations_dir.is_absolute():
        generations_dir = REPO_ROOT / generations_dir

    output_dir = REPO_ROOT / "runs" / "evaluation" / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        dataset_names = args.datasets
    else:
        dataset_names = sorted(
            d.name for d in generations_dir.iterdir()
            if d.is_dir() and (d / "generations.csv").exists()
        )

    print(f"Error Stratification Analysis")
    print(f"{'='*60}")
    print(f"  Datasets: {dataset_names}")

    all_results = []
    for ds_name in dataset_names:
        result = analyze_dataset(ds_name, generations_dir, cfg)
        if not result.empty:
            all_results.append(result)

    if not all_results:
        print("\nNo results.")
        return

    full = pd.concat(all_results, ignore_index=True)

    # Save raw counts
    full.to_csv(output_dir / "error_stratification.csv", index=False)

    # Compute percentages
    error_cols = ["correct", "near_neighbor", "default_negative", "hallucinated", "false_positive", "other_error"]
    pct = full.copy()
    for col in error_cols:
        pct[f"{col}_pct"] = (pct[col] / pct["total"] * 100).round(2)

    pct.to_csv(output_dir / "error_stratification_pct.csv", index=False)

    # Summary by scale group
    print(f"\n{'='*60}")
    print("Summary by Scale Group (across all datasets)")
    print(f"{'='*60}")

    summary = full.groupby("scale_group")[error_cols + ["total"]].sum()
    for col in error_cols:
        summary[f"{col}_pct"] = (summary[col] / summary["total"] * 100).round(2)

    print(summary[["total"] + [f"{c}_pct" for c in error_cols]].to_string())

    # Summary by tuning regime
    print(f"\n{'='*60}")
    print("Summary by Tuning Regime")
    print(f"{'='*60}")

    tuned_display = cfg.get("tuning_regime_display", {})
    summary_tuned = full.copy()
    summary_tuned["regime"] = summary_tuned["tuned_dataset_name"].map(
        lambda x: tuned_display.get(x, x)
    )
    summary_t = summary_tuned.groupby("regime")[error_cols + ["total"]].sum()
    for col in error_cols:
        summary_t[f"{col}_pct"] = (summary_t[col] / summary_t["total"] * 100).round(2)

    print(summary_t[["total"] + [f"{c}_pct" for c in error_cols]].to_string())

    print(f"\n  → Saved to {output_dir / 'error_stratification.csv'}")
    print(f"  → Saved to {output_dir / 'error_stratification_pct.csv'}")


if __name__ == "__main__":
    main()

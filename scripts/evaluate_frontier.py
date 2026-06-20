#!/usr/bin/env python3
"""Evaluate frontier LLM generations and produce Table 7 metrics.

Reads per-dataset generations_merged.csv files (avoiding the corrupted
all_frontier_generations.csv), computes micro-F1 per dataset per model,
then macro-averages across dataset groups (General / Literary).
"""

import sys, os, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import evaluate_slice, normalize_relation, GEN_SCHEMA_ENUMERATED, GEN_GENERIC

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
FRONTIER_DIR = ROOT / "runs" / "frontier_generations"

GENERAL_DATASETS = [
    "tacred", "semeval2010_task8", "conll04", "nyt11",
    "gids", "re_docred", "rebel",
]
LITERARY_DATASETS = ["biographical", "pg_fiction"]
ALL_DATASETS = GENERAL_DATASETS + LITERARY_DATASETS

MODELS = ["GPT-5.4", "Claude Sonnet 4.6", "Gemini 2.5 Pro"]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_all_frontier_data() -> pd.DataFrame:
    """Read per-dataset generations_merged.csv files into one DataFrame."""
    frames = []
    for ds in ALL_DATASETS:
        csv_path = FRONTIER_DIR / ds / "generations_merged.csv"
        if not csv_path.exists():
            print(f"  WARNING: missing {csv_path}")
            continue
        df = pd.read_csv(csv_path, encoding="utf-8", dtype=str, encoding_errors="replace")
        df["eval_dataset_name"] = ds  # ensure consistent naming
        # Drop rows with missing critical columns
        df = df.dropna(subset=["model_id", "relation", "generated_relation"])
        frames.append(df)
        print(f"  {ds}: {len(df):,} rows, models: {sorted(df['model_id'].unique())}")
    return pd.concat(frames, ignore_index=True)


def main():
    print("Loading frontier generation data...")
    df = load_all_frontier_data()
    print(f"\nTotal rows: {len(df):,}")
    print(f"Models found: {sorted(df['model_id'].unique())}")
    print(f"Datasets found: {sorted(df['eval_dataset_name'].unique())}")
    print(f"Gen types: {sorted(df['gen_type'].unique())}")

    # We only care about gen_constrained, prompt_shot=0
    df = df[df["gen_type"] == GEN_SCHEMA_ENUMERATED].copy()
    print(f"\nAfter filtering to gen_constrained: {len(df):,} rows")

    # ---------------------------------------------------------------------------
    # Per-dataset, per-model evaluation
    # ---------------------------------------------------------------------------
    results = []
    for ds_name in ALL_DATASETS:
        ds_df = df[df["eval_dataset_name"] == ds_name]
        if ds_df.empty:
            print(f"\n  SKIP {ds_name}: no data")
            continue

        # Get allowed labels from this dataset's gold relations
        all_gold = ds_df["relation"].tolist()
        allowed_labels = set(all_gold)

        for model_name in sorted(ds_df["model_id"].unique()):
            slice_df = ds_df[ds_df["model_id"] == model_name]
            gold = slice_df["relation"].tolist()
            pred = slice_df["generated_relation"].tolist()

            metrics = evaluate_slice(
                gold=gold,
                pred=pred,
                allowed_labels=allowed_labels,
                exclude_labels=None,
                normalize=True,
            )

            row = {
                "model_id": model_name,
                "eval_dataset_name": ds_name,
                "gen_type": GEN_SCHEMA_ENUMERATED,
                "prompt_shot": 0,
                **metrics,
            }
            results.append(row)
            print(f"  {model_name:25s} | {ds_name:20s} | micro_f1={metrics['micro_f1']:.4f} | schema_valid={metrics['schema_valid_rate']:.4f} | n={metrics['support']}")

    results_df = pd.DataFrame(results)

    # ---------------------------------------------------------------------------
    # Table 7: Macro-average F1 across dataset groups
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TABLE 7: Frontier Model Comparison (Constrained, 0-shot)")
    print("=" * 80)

    table7_rows = []
    for model_name in MODELS:
        model_df = results_df[results_df["model_id"] == model_name]
        if model_df.empty:
            print(f"  WARNING: No results for {model_name}")
            continue

        gen_df = model_df[model_df["eval_dataset_name"].isin(GENERAL_DATASETS)]
        lit_df = model_df[model_df["eval_dataset_name"].isin(LITERARY_DATASETS)]

        gen_f1 = gen_df["micro_f1"].mean() if len(gen_df) > 0 else float("nan")
        lit_f1 = lit_df["micro_f1"].mean() if len(lit_df) > 0 else float("nan")
        all_f1 = model_df["micro_f1"].mean() if len(model_df) > 0 else float("nan")

        gen_schema = gen_df["schema_valid_rate"].mean() if len(gen_df) > 0 else float("nan")
        lit_schema = lit_df["schema_valid_rate"].mean() if len(lit_df) > 0 else float("nan")

        row = {
            "Model": model_name,
            "General Avg F1": round(gen_f1, 4),
            "Literary Avg F1": round(lit_f1, 4),
            "Overall Avg F1": round(all_f1, 4),
            "Gen Schema Valid %": round(gen_schema * 100, 1),
            "Lit Schema Valid %": round(lit_schema * 100, 1),
            "N General Datasets": len(gen_df),
            "N Literary Datasets": len(lit_df),
        }
        table7_rows.append(row)

        print(f"\n  {model_name}:")
        print(f"    General Avg F1:   {gen_f1:.4f}  (across {len(gen_df)} datasets)")
        print(f"    Literary Avg F1:  {lit_f1:.4f}  (across {len(lit_df)} datasets)")
        print(f"    Overall Avg F1:   {all_f1:.4f}")
        print(f"    Gen Schema Valid: {gen_schema*100:.1f}%")
        print(f"    Lit Schema Valid: {lit_schema*100:.1f}%")

    # ---------------------------------------------------------------------------
    # Per-dataset breakdown
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("PER-DATASET BREAKDOWN")
    print("=" * 80)

    pivot = results_df.pivot_table(
        index="eval_dataset_name",
        columns="model_id",
        values="micro_f1",
        aggfunc="first",
    )
    # Reorder
    ds_order = [d for d in ALL_DATASETS if d in pivot.index]
    pivot = pivot.reindex(ds_order)
    col_order = [m for m in MODELS if m in pivot.columns]
    pivot = pivot[col_order]
    print(f"\n{pivot.round(4).to_string()}")

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    out_dir = ROOT / "runs" / "evaluation" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-dataset metrics
    frontier_metrics_path = out_dir / "frontier_metrics.csv"
    results_df.to_csv(frontier_metrics_path, index=False)
    print(f"\nSaved per-dataset metrics: {frontier_metrics_path}")

    # Table 7 summary
    table7_path = out_dir / "table7_frontier_comparison.csv"
    pd.DataFrame(table7_rows).to_csv(table7_path, index=False)
    print(f"Saved Table 7 summary: {table7_path}")

    # Pivot table
    pivot_path = out_dir / "frontier_pivot_f1.csv"
    pivot.round(4).to_csv(pivot_path)
    print(f"Saved pivot table: {pivot_path}")


if __name__ == "__main__":
    main()

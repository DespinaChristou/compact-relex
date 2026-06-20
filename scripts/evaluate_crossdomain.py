#!/usr/bin/env python3
"""Evaluate cross-domain SLM generations and produce a transfer analysis table.

Reads from runs/generations_crossdomain/ (GenTune→literary, LitTune→general)
and combines with existing in-domain results from runs/evaluation/per_dataset_metrics.csv
to produce a comparison showing domain transfer degradation.

Usage:
  python scripts/evaluate_crossdomain.py
  python scripts/evaluate_crossdomain.py --merge_shards   # merge shard files first
"""

import sys, argparse, glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import evaluate_slice, normalize_relation, GEN_SCHEMA_ENUMERATED, GEN_GENERIC

CROSSDOMAIN_DIR = ROOT / "runs" / "generations_crossdomain"

GENERAL_DATASETS = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids", "re_docred", "rebel"]
LITERARY_DATASETS = ["biographical", "pg_fiction"]

# Cross-domain combos we care about
CROSS_COMBOS = {
    "re_gentune": LITERARY_DATASETS,    # GenTune models on literary datasets
    "re_littune": GENERAL_DATASETS,     # LitTune models on general datasets
}


def merge_shards():
    """Merge shard CSVs into generations_merged.csv per dataset dir."""
    for ds_dir in sorted(CROSSDOMAIN_DIR.iterdir()):
        if not ds_dir.is_dir():
            continue
        shard_files = sorted(ds_dir.glob("generations_shard_*.csv"))
        if not shard_files:
            continue
        frames = []
        for sf in shard_files:
            df = pd.read_csv(sf, dtype=str, encoding_errors="replace", on_bad_lines="skip")
            frames.append(df)
        merged = pd.concat(frames, ignore_index=True)
        out = ds_dir / "generations_merged.csv"
        merged.to_csv(out, index=False)
        print(f"  Merged {len(shard_files)} shards -> {out.name} ({len(merged):,} rows)")


def load_crossdomain_data() -> pd.DataFrame:
    """Load all cross-domain generation results."""
    frames = []
    all_datasets = set(GENERAL_DATASETS + LITERARY_DATASETS)

    for ds_name in sorted(all_datasets):
        ds_dir = CROSSDOMAIN_DIR / ds_name
        if not ds_dir.exists():
            continue

        # Try merged first, fall back to shard files
        merged_csv = ds_dir / "generations_merged.csv"
        if merged_csv.exists():
            csv_path = merged_csv
        else:
            shard_files = sorted(ds_dir.glob("generations_shard_*.csv"))
            if not shard_files:
                continue
            # Read shards directly
            for sf in shard_files:
                df = pd.read_csv(sf, dtype=str, encoding_errors="replace", on_bad_lines="skip")
                df["eval_dataset_name"] = ds_name
                df = df.dropna(subset=["model_id", "relation", "generated_relation"])
                frames.append(df)
            continue

        df = pd.read_csv(csv_path, dtype=str, encoding_errors="replace", on_bad_lines="skip")
        df["eval_dataset_name"] = ds_name
        df = df.dropna(subset=["model_id", "relation", "generated_relation"])
        frames.append(df)
        print(f"  {ds_name}: {len(df):,} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def evaluate_crossdomain(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-dataset per-model metrics for cross-domain generations."""
    results = []

    for (ds_name, model_id, tuned_ds, model_shot, prompt_shot, gen_type), slice_df in df.groupby(
        ["eval_dataset_name", "model_id", "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"]
    ):
        # Only evaluate cross-domain combos
        if tuned_ds in CROSS_COMBOS and ds_name in CROSS_COMBOS[tuned_ds]:
            gold = slice_df["relation"].tolist()
            pred = slice_df["generated_relation"].tolist()
            allowed_labels = set(gold)

            metrics = evaluate_slice(
                gold=gold, pred=pred,
                allowed_labels=allowed_labels,
                normalize=True,
            )

            results.append({
                "eval_dataset_name": ds_name,
                "model_id": model_id,
                "tuned_dataset_name": tuned_ds,
                "model_shot": model_shot,
                "prompt_shot": prompt_shot,
                "gen_type": gen_type,
                **metrics,
            })
            print(f"  {model_id:25s} | {tuned_ds:12s} | {ds_name:20s} | {gen_type:16s} | ps{prompt_shot} | F1={metrics['micro_f1']:.4f}")

    return pd.DataFrame(results)


def build_transfer_table(cross_df: pd.DataFrame, indomain_df: pd.DataFrame) -> pd.DataFrame:
    """Build comparison table: in-domain F1 vs cross-domain F1 vs delta."""
    # Focus on gen_constrained for cleaner comparison
    cross_c = cross_df[cross_df["gen_type"] == GEN_SCHEMA_ENUMERATED].copy()
    indomain_c = indomain_df[indomain_df["gen_type"] == GEN_SCHEMA_ENUMERATED].copy()

    rows = []

    # For each model config that has cross-domain results
    for (model_id, tuned_ds, model_shot, prompt_shot), grp in cross_c.groupby(
        ["model_id", "tuned_dataset_name", "model_shot", "prompt_shot"]
    ):
        # Cross-domain F1 by domain
        if tuned_ds == "re_gentune":
            # GenTune on literary = cross-domain; GenTune on general = in-domain
            cross_datasets = LITERARY_DATASETS
            indomain_datasets = GENERAL_DATASETS
            cross_domain_name = "Literary"
            indomain_domain_name = "General"
        elif tuned_ds == "re_littune":
            cross_datasets = GENERAL_DATASETS
            indomain_datasets = LITERARY_DATASETS
            cross_domain_name = "General"
            indomain_domain_name = "Literary"
        else:
            continue

        cross_f1s = grp[grp["eval_dataset_name"].isin(cross_datasets)]["micro_f1"]
        cross_avg = cross_f1s.mean() if len(cross_f1s) > 0 else float("nan")

        # Get matching in-domain performance from original eval
        indomain_match = indomain_c[
            (indomain_c["model_id"] == model_id) &
            (indomain_c["tuned_dataset_name"] == tuned_ds) &
            (indomain_c["model_shot"] == str(model_shot)) &
            (indomain_c["prompt_shot"] == str(prompt_shot))
        ]
        indomain_f1s = indomain_match[indomain_match["eval_dataset_name"].isin(indomain_datasets)]["micro_f1"]
        indomain_avg = indomain_f1s.mean() if len(indomain_f1s) > 0 else float("nan")

        delta = cross_avg - indomain_avg if not (np.isnan(cross_avg) or np.isnan(indomain_avg)) else float("nan")

        rows.append({
            "model_id": model_id,
            "tuned_dataset_name": tuned_ds,
            "model_shot": model_shot,
            "prompt_shot": prompt_shot,
            "in_domain": indomain_domain_name,
            "in_domain_avg_f1": round(indomain_avg, 4),
            "cross_domain": cross_domain_name,
            "cross_domain_avg_f1": round(cross_avg, 4),
            "delta_f1": round(delta, 4),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_shards", action="store_true", help="Merge shard files first")
    args = parser.parse_args()

    if args.merge_shards:
        print("Merging shard files...")
        merge_shards()

    print("\nLoading cross-domain generation data...")
    df = load_crossdomain_data()

    if df.empty:
        print("\nNo cross-domain data found yet.")
        print("Run generations first:")
        print("  python -m src.run_generations --config configs/generations_crossdomain.yaml")
        print("\nThen re-run this script:")
        print("  python scripts/evaluate_crossdomain.py --merge_shards")
        return

    print(f"\nTotal cross-domain rows: {len(df):,}")
    print(f"Models: {sorted(df['model_id'].unique())}")
    print(f"Tuned datasets: {sorted(df['tuned_dataset_name'].unique())}")
    print(f"Eval datasets: {sorted(df['eval_dataset_name'].unique())}")

    print("\nEvaluating cross-domain slices...")
    cross_metrics = evaluate_crossdomain(df)

    # Load in-domain metrics
    indomain_path = ROOT / "runs" / "evaluation" / "per_dataset_metrics.csv"
    if indomain_path.exists():
        indomain_df = pd.read_csv(indomain_path, dtype={"model_shot": str, "prompt_shot": str})
        print(f"\nLoaded {len(indomain_df)} in-domain evaluation rows")

        transfer_table = build_transfer_table(cross_metrics, indomain_df)

        if not transfer_table.empty:
            print("\n" + "=" * 90)
            print("DOMAIN TRANSFER TABLE (gen_constrained)")
            print("=" * 90)
            print(transfer_table.sort_values(["tuned_dataset_name", "model_id", "model_shot", "prompt_shot"]).to_string(index=False))

            # Summary: average delta by tuned regime
            print("\n--- Average Transfer Delta by Tuning Regime ---")
            summary = transfer_table.groupby("tuned_dataset_name").agg(
                avg_in_domain_f1=("in_domain_avg_f1", "mean"),
                avg_cross_domain_f1=("cross_domain_avg_f1", "mean"),
                avg_delta=("delta_f1", "mean"),
            ).round(4)
            print(summary.to_string())
    else:
        print(f"\nWARNING: In-domain metrics not found at {indomain_path}")
        transfer_table = pd.DataFrame()

    # Save outputs
    out_dir = ROOT / "runs" / "evaluation" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    cross_metrics_path = out_dir / "crossdomain_metrics.csv"
    cross_metrics.to_csv(cross_metrics_path, index=False)
    print(f"\nSaved cross-domain metrics: {cross_metrics_path}")

    if not transfer_table.empty:
        transfer_path = out_dir / "crossdomain_transfer_table.csv"
        transfer_table.to_csv(transfer_path, index=False)
        print(f"Saved transfer table: {transfer_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate DAPT case study: compare Llama-3.2-3B with and without DAPT.

Reads DAPT generations from runs/generations_dapt/ and compares against
existing non-DAPT results from runs/evaluation/per_dataset_metrics.csv.

Usage:
  python scripts/evaluate_dapt_casestudy.py
  python scripts/evaluate_dapt_casestudy.py --merge_shards
"""

import sys, argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.eval import evaluate_slice

DAPT_DIR = ROOT / "runs" / "generations_dapt"

GENERAL_DATASETS = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids", "re_docred", "rebel"]
LITERARY_DATASETS = ["biographical", "pg_fiction"]
ALL_DATASETS = GENERAL_DATASETS + LITERARY_DATASETS


def merge_shards():
    """Merge shard CSVs into generations_merged.csv per dataset dir."""
    for ds_dir in sorted(DAPT_DIR.iterdir()):
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


def _fix_nan_predictions(df: pd.DataFrame, ds_name: str) -> pd.DataFrame:
    """Handle NaN generated_relation values.

    For pg_fiction: rows with relation='none' and NaN generated_relation are
    correct predictions (model correctly abstains). Map NaN → 'none' so these
    count as hits.  All other NaN predictions → '' (counts as wrong).

    For other datasets: NaN predictions → '' (counts as wrong).
    """
    mask_nan = df["generated_relation"].isna()
    if not mask_nan.any():
        return df

    if ds_name == "pg_fiction":
        # NaN prediction on a 'none' gold row → correct (map to 'none')
        none_gold = mask_nan & (df["relation"].str.lower().str.strip() == "none")
        df.loc[none_gold, "generated_relation"] = "none"
        # NaN prediction on a non-'none' gold row → wrong (map to '')
        df.loc[mask_nan & ~none_gold, "generated_relation"] = ""
    else:
        df.loc[mask_nan, "generated_relation"] = ""

    return df


def load_dapt_data() -> pd.DataFrame:
    frames = []
    for ds_name in ALL_DATASETS:
        ds_dir = DAPT_DIR / ds_name
        if not ds_dir.exists():
            continue
        merged_csv = ds_dir / "generations_merged.csv"
        if not merged_csv.exists():
            # Try shards directly
            shard_files = sorted(ds_dir.glob("generations_shard_*.csv"))
            for sf in shard_files:
                df = pd.read_csv(sf, dtype=str, encoding_errors="replace", on_bad_lines="skip")
                df["eval_dataset_name"] = ds_name
                df = _fix_nan_predictions(df, ds_name)
                df = df.dropna(subset=["model_id", "relation"])
                frames.append(df)
            continue

        df = pd.read_csv(merged_csv, dtype=str, encoding_errors="replace", on_bad_lines="skip")
        df["eval_dataset_name"] = ds_name
        df = _fix_nan_predictions(df, ds_name)
        df = df.dropna(subset=["model_id", "relation"])
        frames.append(df)
        print(f"  {ds_name}: {len(df):,} rows")

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge_shards", action="store_true")
    args = parser.parse_args()

    if args.merge_shards:
        print("Merging shard files...")
        merge_shards()

    print("\nLoading DAPT generation data...")
    df = load_dapt_data()

    if df.empty:
        print("\nNo DAPT data found yet.")
        print("Run fine-tuning first:")
        print("  python -m src.run_all --config configs/experiments_dapt_casestudy.yaml --stage finetune")
        print("Then generate:")
        print("  python -m src.run_generations --config configs/generations_dapt_casestudy.yaml")
        print("Then evaluate:")
        print("  python scripts/evaluate_dapt_casestudy.py --merge_shards")
        return

    print(f"\nTotal rows: {len(df):,}")
    print(f"Models: {sorted(df['model_id'].unique())}")
    print(f"Tuned datasets: {sorted(df['tuned_dataset_name'].unique())}")

    # Filter to gen_constrained
    df = df[df["gen_type"] == "gen_constrained"]
    print(f"After filtering to gen_constrained: {len(df):,} rows")

    # Evaluate each slice
    dapt_results = []
    for (ds_name, model_id, tuned_ds, ms, ps), grp in df.groupby(
        ["eval_dataset_name", "model_id", "tuned_dataset_name", "model_shot", "prompt_shot"]
    ):
        gold = grp["relation"].tolist()
        pred = grp["generated_relation"].tolist()

        metrics = evaluate_slice(gold=gold, pred=pred, allowed_labels=set(gold), normalize=True)
        dapt_results.append({
            "eval_dataset_name": ds_name,
            "model_id": model_id,
            "tuned_dataset_name": tuned_ds,
            "model_shot": ms,
            "prompt_shot": ps,
            "gen_type": "gen_constrained",
            **metrics,
        })
        print(f"  {model_id:40s} | {tuned_ds:12s} | {ds_name:20s} | F1={metrics['micro_f1']:.4f}")

    dapt_df = pd.DataFrame(dapt_results)

    # Load non-DAPT Llama results for comparison
    indomain_path = ROOT / "runs" / "evaluation" / "per_dataset_metrics.csv"
    if not indomain_path.exists():
        print(f"\nWARNING: No existing metrics at {indomain_path}")
        return

    existing = pd.read_csv(indomain_path, dtype={"model_shot": str, "prompt_shot": str})
    llama_base = existing[
        (existing["model_id"] == "Llama-3.2-3B-Instruct") &
        (existing["gen_type"] == "gen_constrained") &
        (existing["model_shot"] == "0") &
        (existing["prompt_shot"] == "0")
    ].copy()

    print(f"\nLoaded {len(llama_base)} existing Llama non-DAPT rows")

    # Build comparison table
    print("\n" + "=" * 100)
    print("DAPT CASE STUDY: Llama-3.2-3B-Instruct (DAPT vs No-DAPT), 0-shot, gen_constrained")
    print("=" * 100)

    for tuned_ds in ["re_gentune", "re_littune", "re_mixtune"]:
        dapt_slice = dapt_df[dapt_df["tuned_dataset_name"] == tuned_ds]
        base_slice = llama_base[llama_base["tuned_dataset_name"] == tuned_ds]

        if dapt_slice.empty:
            continue

        print(f"\n--- {tuned_ds} ---")
        print(f"{'Dataset':<22s} {'No-DAPT F1':>12s} {'DAPT F1':>12s} {'Delta':>10s} {'Direction':>12s}")
        print("-" * 70)

        for ds_name in ALL_DATASETS:
            dapt_row = dapt_slice[dapt_slice["eval_dataset_name"] == ds_name]
            base_row = base_slice[base_slice["eval_dataset_name"] == ds_name]

            dapt_f1 = dapt_row["micro_f1"].iloc[0] if len(dapt_row) > 0 else float("nan")
            base_f1 = base_row["micro_f1"].iloc[0] if len(base_row) > 0 else float("nan")

            if np.isnan(dapt_f1) and np.isnan(base_f1):
                continue

            delta = dapt_f1 - base_f1 if not (np.isnan(dapt_f1) or np.isnan(base_f1)) else float("nan")
            direction = ""
            if not np.isnan(delta):
                if delta > 0.01:
                    direction = "IMPROVED"
                elif delta < -0.01:
                    direction = "DEGRADED"
                else:
                    direction = "~same"

            base_str = f"{base_f1:.4f}" if not np.isnan(base_f1) else "N/A"
            dapt_str = f"{dapt_f1:.4f}" if not np.isnan(dapt_f1) else "N/A"
            delta_str = f"{delta:+.4f}" if not np.isnan(delta) else "N/A"

            domain = "LIT" if ds_name in LITERARY_DATASETS else "GEN"
            print(f"  {ds_name:<20s} {base_str:>12s} {dapt_str:>12s} {delta_str:>10s} {direction:>12s}  [{domain}]")

        # Averages
        for domain_name, domain_ds in [("General", GENERAL_DATASETS), ("Literary", LITERARY_DATASETS)]:
            dapt_avg = dapt_slice[dapt_slice["eval_dataset_name"].isin(domain_ds)]["micro_f1"].mean()
            base_avg = base_slice[base_slice["eval_dataset_name"].isin(domain_ds)]["micro_f1"].mean()
            delta_avg = dapt_avg - base_avg if not (np.isnan(dapt_avg) or np.isnan(base_avg)) else float("nan")

            base_str = f"{base_avg:.4f}" if not np.isnan(base_avg) else "N/A"
            dapt_str = f"{dapt_avg:.4f}" if not np.isnan(dapt_avg) else "N/A"
            delta_str = f"{delta_avg:+.4f}" if not np.isnan(delta_avg) else "N/A"

            print(f"  {'>>> ' + domain_name + ' Avg':<20s} {base_str:>12s} {dapt_str:>12s} {delta_str:>10s}")

    # Save outputs
    out_dir = ROOT / "runs" / "evaluation" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    dapt_metrics_path = out_dir / "dapt_casestudy_metrics.csv"
    dapt_df.to_csv(dapt_metrics_path, index=False)
    print(f"\nSaved DAPT metrics: {dapt_metrics_path}")


if __name__ == "__main__":
    main()

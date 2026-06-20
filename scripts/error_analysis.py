#!/usr/bin/env python3
"""
Qualitative error analysis support — surfaces candidate examples for Table 8.

For each literary dataset, finds representative error examples:
  1. Implicit relation failures (model misses subtextual cues)
  2. Near-neighbour label confusions (schema-valid but wrong)
  3. Hallucinated relations (outside the schema)
  4. Coreference / long-context failures

Also computes the top confused label pairs per dataset for analysis.

Usage:
    python scripts/error_analysis.py --config configs/eval.yaml
    python scripts/error_analysis.py --config configs/eval.yaml --datasets biographical pg_fiction
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval import normalize_relation, top_confused_pairs, GEN_SCHEMA_ENUMERATED, GEN_GENERIC


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def _display_regime(tuned_ds: str, cfg: dict) -> str:
    return cfg.get("tuning_regime_display", {}).get(tuned_ds, tuned_ds)


# ═══════════════════════════════════════════════════════════════════
# Error categorisation
# ═══════════════════════════════════════════════════════════════════

def categorise_error(
    gold: str,
    pred: str,
    gold_norm: str,
    pred_norm: str,
    allowed_labels_norm: set,
) -> str:
    """Classify an error into one of the analysis categories."""
    if pred_norm == "":
        return "malformed_empty"
    if len(pred_norm) > 60:
        return "malformed_verbose"
    if pred_norm not in allowed_labels_norm:
        return "hallucinated_relation"
    if pred_norm in allowed_labels_norm and pred_norm != gold_norm:
        return "schema_valid_wrong"
    return "other"


# ═══════════════════════════════════════════════════════════════════
# Main analysis
# ═══════════════════════════════════════════════════════════════════

def analyse_dataset(
    dataset_name: str,
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    max_examples_per_category: int = 20,
):
    """Run error analysis for one dataset."""

    csv_path = generations_dir / dataset_name / "generations.csv"
    if not csv_path.exists():
        print(f"  ✗ {dataset_name}: not found, skipping")
        return

    print(f"\n{'='*60}")
    print(f"Error analysis: {dataset_name}")
    print(f"{'='*60}")

    # Load with prompts (needed for text snippets in Table 8)
    usecols = [
        "eval_dataset_name", "prompt_0_shot", "relation", "gen_type",
        "model_id", "tuned_dataset_name", "model_shot", "prompt_shot",
        "generated_relation",
    ]
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Filter to primary settings
    gt = cfg.get("primary_gen_type", GEN_SCHEMA_ENUMERATED)
    df = df[df["gen_type"] == gt].copy()
    df = df[df["model_shot"] == df["prompt_shot"]].copy()

    # Deduplicate
    df = df.drop_duplicates()

    # Normalise labels
    df["gold_norm"] = df["relation"].apply(normalize_relation)
    df["pred_norm"] = df["generated_relation"].apply(normalize_relation)

    allowed_labels = set(df["relation"].dropna().unique())
    allowed_norm = {normalize_relation(l) for l in allowed_labels}

    # Mark errors
    df["is_correct"] = df["gold_norm"] == df["pred_norm"]
    errors = df[~df["is_correct"]].copy()

    print(f"  Total predictions: {len(df):,}")
    print(f"  Errors: {len(errors):,} ({len(errors)/len(df)*100:.1f}%)")

    if errors.empty:
        print("  No errors to analyse.")
        return

    # Categorise errors
    errors["error_category"] = errors.apply(
        lambda r: categorise_error(
            r["relation"], r["generated_relation"],
            r["gold_norm"], r["pred_norm"],
            allowed_norm,
        ),
        axis=1,
    )

    # Summary
    cat_counts = errors["error_category"].value_counts()
    print(f"\n  Error categories:")
    for cat, cnt in cat_counts.items():
        print(f"    {cat}: {cnt:,} ({cnt/len(errors)*100:.1f}%)")

    # ── Top confused label pairs ──
    gold_list = errors["gold_norm"].tolist()
    pred_list = errors["pred_norm"].tolist()
    confused = top_confused_pairs(gold_list, pred_list, top_k=15)

    confusion_df = pd.DataFrame(confused, columns=["gold_label", "predicted_label", "count"])
    confusion_path = output_dir / f"{dataset_name}_top_confusions.csv"
    confusion_df.to_csv(confusion_path, index=False)
    print(f"\n  Top confused pairs → {confusion_path}")
    print(confusion_df.head(10).to_string(index=False))

    # ── Sample error examples per category ──
    all_examples = []

    for category in cat_counts.index:
        cat_errors = errors[errors["error_category"] == category]

        # Sample diverse examples (different models if possible)
        sampled = cat_errors.groupby("model_id").apply(
            lambda g: g.sample(min(3, len(g)), random_state=42)
        ).reset_index(drop=True)

        if len(sampled) > max_examples_per_category:
            sampled = sampled.sample(max_examples_per_category, random_state=42)

        for _, row in sampled.iterrows():
            # Extract a short text snippet from the prompt
            prompt = str(row.get("prompt_0_shot", ""))
            # Try to extract the sentence from the prompt
            snippet = _extract_sentence_from_prompt(prompt)

            all_examples.append({
                "dataset": dataset_name,
                "error_category": category,
                "gold_relation": row["relation"],
                "predicted_relation": row["generated_relation"],
                "model_id": row["model_id"],
                "tuning_regime": _display_regime(row["tuned_dataset_name"], cfg),
                "prompt_style": f"{int(row['model_shot'])}-shot",
                "text_snippet": snippet[:300] if snippet else "",
            })

    examples_df = pd.DataFrame(all_examples)
    examples_path = output_dir / f"{dataset_name}_error_examples.csv"
    examples_df.to_csv(examples_path, index=False)
    print(f"\n  Error examples ({len(examples_df)}) → {examples_path}")

    # ── Per-model error rate summary ──
    model_error_rates = df.groupby(
        ["model_id", "tuned_dataset_name", "model_shot"]
    ).agg(
        total=("is_correct", "count"),
        correct=("is_correct", "sum"),
    ).reset_index()
    model_error_rates["error_rate"] = (
        1 - model_error_rates["correct"] / model_error_rates["total"]
    ).round(4)
    model_error_path = output_dir / f"{dataset_name}_model_error_rates.csv"
    model_error_rates.to_csv(model_error_path, index=False)
    print(f"  Model error rates → {model_error_path}")


def _extract_sentence_from_prompt(prompt: str) -> str:
    """Extract the main sentence from a prompt template.

    Looks for patterns like 'Sentence: "..."' or text between quotes.
    """
    import re

    # Try to find Sentence: "..." pattern
    match = re.search(r'Sentence:\s*"([^"]+)"', prompt)
    if match:
        return match.group(1).strip()

    # Try to find text between first pair of double quotes
    match = re.search(r'"([^"]{20,})"', prompt)
    if match:
        return match.group(1).strip()

    # Fallback: take first 200 chars after removing boilerplate
    lines = prompt.strip().split("\n")
    for line in lines:
        line = line.strip()
        if len(line) > 30 and not line.startswith("You are") and not line.startswith("Extract"):
            return line[:200]

    return prompt[:200]


# ═══════════════════════════════════════════════════════════════════
# Cross-model comparison for Table 8
# ═══════════════════════════════════════════════════════════════════

def build_cross_model_error_table(
    dataset_name: str,
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    n_examples: int = 50,
):
    """Find examples where different tuning regimes disagree.

    Specifically: cases where GenTune is wrong but LitTune is correct,
    or vice versa. These illustrate domain specialization effects.
    """
    csv_path = generations_dir / dataset_name / "generations.csv"
    if not csv_path.exists():
        return

    usecols = [
        "eval_dataset_name", "prompt_0_shot", "relation", "gen_type",
        "model_id", "tuned_dataset_name", "model_shot", "prompt_shot",
        "generated_relation",
    ]
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)

    # Filter
    gt = cfg.get("primary_gen_type", GEN_SCHEMA_ENUMERATED)
    df = df[(df["gen_type"] == gt) & (df["model_shot"] == df["prompt_shot"])].copy()
    df = df.drop_duplicates()

    df["gold_norm"] = df["relation"].apply(normalize_relation)
    df["pred_norm"] = df["generated_relation"].apply(normalize_relation)
    df["correct"] = df["gold_norm"] == df["pred_norm"]

    # For each unique prompt (approximated by prompt_0_shot + relation),
    # check if different regimes give different answers
    df["example_key"] = df["prompt_0_shot"].str[:100] + "|" + df["relation"]

    # Find examples where one regime is right and another wrong
    disagreements = []

    for example_key, example_group in df.groupby("example_key"):
        regimes_correct = example_group.groupby("tuned_dataset_name")["correct"].any()

        if len(regimes_correct) < 2:
            continue

        # Case: one regime correct, another wrong
        any_correct = regimes_correct.any()
        any_wrong = (~regimes_correct).any()

        if any_correct and any_wrong:
            row = example_group.iloc[0]
            correct_regimes = list(regimes_correct[regimes_correct].index)
            wrong_regimes = list(regimes_correct[~regimes_correct].index)

            # Get the predictions from each regime
            pred_by_regime = {}
            for _, r in example_group.iterrows():
                key = _display_regime(r["tuned_dataset_name"], cfg)
                if key not in pred_by_regime:
                    pred_by_regime[key] = r["generated_relation"]

            snippet = _extract_sentence_from_prompt(str(row["prompt_0_shot"]))

            disagreements.append({
                "dataset": dataset_name,
                "text_snippet": snippet[:300],
                "gold_relation": row["relation"],
                "correct_regimes": ", ".join(_display_regime(r, cfg) for r in correct_regimes),
                "wrong_regimes": ", ".join(_display_regime(r, cfg) for r in wrong_regimes),
                **{f"pred_{k}": v for k, v in pred_by_regime.items()},
            })

            if len(disagreements) >= n_examples:
                break

    if disagreements:
        dis_df = pd.DataFrame(disagreements)
        dis_path = output_dir / f"{dataset_name}_regime_disagreements.csv"
        dis_df.to_csv(dis_path, index=False)
        print(f"\n  Regime disagreements ({len(dis_df)}) → {dis_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Qualitative error analysis")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument(
        "--datasets", nargs="*", default=None,
        help="Datasets to analyse (default: literary datasets)"
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    generations_dir = _resolve_path(cfg["paths"]["generations_dir"])
    output_dir = _resolve_path(cfg["paths"]["output_dir"]) / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Default to literary datasets (the focus of Section 4.5)
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = cfg["dataset_groups"]["literary"]
        # Also include a couple of general datasets for contrast
        datasets = datasets + ["tacred", "conll04"]

    for ds in datasets:
        analyse_dataset(ds, generations_dir, cfg, output_dir)
        build_cross_model_error_table(ds, generations_dir, cfg, output_dir)

    print(f"\n{'='*60}")
    print(f"Error analysis complete. Outputs in {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Statistical significance tests for the Compact RelEx paper.

Produces:
  runs/evaluation/statistical/
    bootstrap_cis.csv              — per model-config × dataset bootstrap 95% CIs
    pairwise_significance.csv      — paired bootstrap + McNemar for key comparisons
    summary_cis_for_table3.csv     — domain-avg CIs for main results table
    label_complexity.csv           — F1 vs. number of relation types per dataset

Usage:
    python scripts/statistical_tests.py --config configs/eval.yaml
    python scripts/statistical_tests.py --config configs/eval.yaml --comparisons-only
    python scripts/statistical_tests.py --config configs/eval.yaml --n-bootstrap 5000

Requires generation CSVs in runs/generations/ (for per-example bootstrap).
Falls back to per_dataset_metrics.csv for summary analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml

# Ensure repo root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.eval import normalize_relation, compute_micro_metrics


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_N_BOOTSTRAP = 10_000
CONFIDENCE_LEVEL = 0.95
SEED = 42

# Key pairwise comparisons for significance testing
# Each tuple: (name, model_config_A, model_config_B, domain_filter)
# model_config = (model_id, tuned_dataset_name, model_shot)
KEY_COMPARISONS = [
    # Best SLM vs. frontier (literary domain)
    (
        "Best SLM vs GPT-5.4 (literary)",
        ("SmolLM3-3B", "re_littune", 0),
        ("GPT-5.4", "frontier", 0),
        "literary",
    ),
    (
        "Best SLM vs Claude Sonnet 4.6 (literary)",
        ("SmolLM3-3B", "re_littune", 0),
        ("Claude-Sonnet-4.6", "frontier", 0),
        "literary",
    ),
    # MixTune vs. GenTune (general domain) — best 3B model
    (
        "Llama MixTune vs GenTune 2s (general)",
        ("Llama-3.2-3B-Instruct", "re_mixtune", 2),
        ("Llama-3.2-3B-Instruct", "re_gentune", 2),
        "general",
    ),
    # MixTune vs. LitTune (literary domain) — best 3B model
    (
        "Llama MixTune vs LitTune 0s (literary)",
        ("Llama-3.2-3B-Instruct", "re_mixtune", 2),
        ("Llama-3.2-3B-Instruct", "re_littune", 0),
        "literary",
    ),
    # 2-shot vs. 0-shot (MixTune, best sub-billion)
    (
        "SmolLM2-360M MixTune 2s vs 0s (overall)",
        ("SmolLM2-360M-Instruct", "re_mixtune", 2),
        ("SmolLM2-360M-Instruct", "re_mixtune", 0),
        "all",
    ),
    # Scale effect: 360M vs. 3B under same regime
    (
        "SmolLM2-360M vs Llama-3B MixTune 2s (overall)",
        ("SmolLM2-360M-Instruct", "re_mixtune", 2),
        ("Llama-3.2-3B-Instruct", "re_mixtune", 2),
        "all",
    ),
    # Best sub-billion vs. best 3B (overall)
    (
        "Best sub-B (Qwen-0.5B GenTune 2s) vs best 3B (Llama MixTune 2s)",
        ("Qwen2.5-0.5B-Instruct", "re_gentune", 2),
        ("Llama-3.2-3B-Instruct", "re_mixtune", 2),
        "all",
    ),
]

DATASET_GROUPS = {
    "general": [
        "tacred", "semeval2010_task8", "conll04", "nyt11",
        "gids", "re_docred", "rebel",
    ],
    "literary": ["biographical", "pg_fiction"],
}
DATASET_GROUPS["all"] = DATASET_GROUPS["general"] + DATASET_GROUPS["literary"]


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

def load_generation_data(
    dataset_name: str,
    generations_dir: Path,
    usecols: List[str],
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """Load generation CSV for a single dataset, adding prompt_hash for dedup."""
    csv_path = generations_dir / dataset_name / "generations.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    load_cols = list(usecols)
    if "prompt_0_shot" not in load_cols:
        load_cols.append("prompt_0_shot")

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=load_cols, chunksize=chunk_size):
        chunk["prompt_hash"] = chunk["prompt_0_shot"].apply(
            lambda x: hashlib.md5(str(x).encode()).hexdigest()[:12]
        )
        chunk.drop(columns=["prompt_0_shot"], errors="ignore", inplace=True)
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Dedup (keep first occurrence per prompt_hash per config)
    group_cols = ["model_id", "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"]
    dedup_cols = ["prompt_hash"] + group_cols + ["relation"]
    dedup_cols = [c for c in dedup_cols if c in df.columns]
    df = df.drop_duplicates(subset=dedup_cols, keep="first")

    return df


def get_matched_constrained(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to gen_constrained + model_shot==prompt_shot."""
    mask = (df["gen_type"] == "gen_constrained") & (df["model_shot"] == df["prompt_shot"])
    return df[mask].copy()


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap confidence intervals
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_micro_f1(
    gold: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence: float = CONFIDENCE_LEVEL,
    rng: np.random.Generator = None,
    exclude_labels: Optional[Set[str]] = None,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for micro-F1.

    Parameters
    ----------
    gold, pred : arrays of normalized label strings (same length)
    n_bootstrap : number of bootstrap iterations
    confidence : confidence level (e.g. 0.95)
    rng : numpy random generator
    exclude_labels : labels to exclude

    Returns
    -------
    dict with point_f1, ci_lower, ci_upper, ci_width, std
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(gold)
    assert n == len(pred), f"Length mismatch: {len(gold)} vs {len(pred)}"

    if n == 0:
        return {"point_f1": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0, "std": 0.0}

    # Filter out excluded labels
    if exclude_labels:
        mask = np.array([g not in exclude_labels for g in gold])
        gold = gold[mask]
        pred = pred[mask]
        n = len(gold)
        if n == 0:
            return {"point_f1": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "ci_width": 0.0, "std": 0.0}

    # Point estimate
    correct = (gold == pred)
    point_f1 = correct.sum() / n  # micro-F1 = accuracy for single-label

    # Bootstrap
    f1_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        f1_samples[i] = correct[idx].sum() / n

    alpha = 1 - confidence
    ci_lower = float(np.percentile(f1_samples, 100 * alpha / 2))
    ci_upper = float(np.percentile(f1_samples, 100 * (1 - alpha / 2)))

    return {
        "point_f1": round(float(point_f1), 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "ci_width": round(ci_upper - ci_lower, 6),
        "std": round(float(np.std(f1_samples)), 6),
    }


# ═══════════════════════════════════════════════════════════════════════
# Paired bootstrap test
# ═══════════════════════════════════════════════════════════════════════

def paired_bootstrap_test(
    gold: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Paired bootstrap test: is model A significantly better than model B?

    Tests H0: F1(A) <= F1(B) via one-sided paired bootstrap.
    Also reports two-sided p-value.

    Returns
    -------
    dict with delta_f1, p_value_two_sided, p_value_one_sided, ci_lower, ci_upper
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    n = len(gold)
    assert n == len(pred_a) == len(pred_b)

    correct_a = (gold == pred_a).astype(float)
    correct_b = (gold == pred_b).astype(float)
    diffs = correct_a - correct_b

    observed_delta = diffs.mean()

    # Bootstrap the mean difference
    delta_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta_samples[i] = diffs[idx].mean()

    # Two-sided p-value: 2 × min(P(δ* ≤ 0), P(δ* ≥ 0))
    # This is symmetric regardless of which model is better.
    frac_le_zero = float(np.mean(delta_samples <= 0))
    frac_ge_zero = float(np.mean(delta_samples >= 0))
    p_two_sided = 2 * min(frac_le_zero, frac_ge_zero)
    p_two_sided = min(p_two_sided, 1.0)

    # One-sided: P(delta <= 0 | H0) — tests whether A is significantly better
    p_one_sided = frac_le_zero

    ci_lower = float(np.percentile(delta_samples, 2.5))
    ci_upper = float(np.percentile(delta_samples, 97.5))

    return {
        "delta_f1": round(float(observed_delta), 6),
        "p_two_sided": round(p_two_sided, 6),
        "p_one_sided": round(p_one_sided, 6),
        "ci_lower": round(ci_lower, 6),
        "ci_upper": round(ci_upper, 6),
        "significant_05": p_two_sided < 0.05,
        "significant_01": p_two_sided < 0.01,
    }


# ═══════════════════════════════════════════════════════════════════════
# McNemar's test
# ═══════════════════════════════════════════════════════════════════════

def mcnemar_test(
    gold: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
) -> Dict[str, float]:
    """McNemar's test for paired nominal data.

    Builds the 2x2 contingency table:
        A correct, B correct   | A correct, B wrong
        A wrong,   B correct   | A wrong,   B wrong

    Tests whether the off-diagonal counts differ significantly.

    Returns
    -------
    dict with n_both_correct, n_both_wrong, n_a_only, n_b_only,
         chi2, p_value, odds_ratio
    """
    correct_a = (gold == pred_a)
    correct_b = (gold == pred_b)

    n_both_correct = int(np.sum(correct_a & correct_b))
    n_both_wrong = int(np.sum(~correct_a & ~correct_b))
    n_a_only = int(np.sum(correct_a & ~correct_b))  # A right, B wrong
    n_b_only = int(np.sum(~correct_a & correct_b))  # A wrong, B right

    # McNemar's chi-squared (with continuity correction)
    denom = n_a_only + n_b_only
    if denom == 0:
        chi2 = 0.0
        p_value = 1.0
    else:
        chi2 = (abs(n_a_only - n_b_only) - 1) ** 2 / denom
        # Approximate p-value from chi2(1) using complementary error function
        # For chi2 with df=1: p = erfc(sqrt(chi2/2))
        import math
        p_value = math.erfc(math.sqrt(chi2 / 2))

    odds_ratio = n_a_only / n_b_only if n_b_only > 0 else float("inf")

    return {
        "n_both_correct": n_both_correct,
        "n_both_wrong": n_both_wrong,
        "n_a_only_correct": n_a_only,
        "n_b_only_correct": n_b_only,
        "mcnemar_chi2": round(chi2, 4),
        "mcnemar_p_value": round(p_value, 6),
        "odds_ratio": round(odds_ratio, 4),
        "mcnemar_significant_05": p_value < 0.05,
    }


# ═══════════════════════════════════════════════════════════════════════
# Domain-average bootstrap CI (for Table 3 annotations)
# ═══════════════════════════════════════════════════════════════════════

def bootstrap_domain_avg_f1(
    per_dataset_f1s: Dict[str, Dict],
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    rng: np.random.Generator = None,
) -> Dict[str, float]:
    """Bootstrap CI for macro-averaged F1 across datasets in a domain.

    per_dataset_f1s: {dataset_name: {"gold": array, "pred": array}}
    The domain average is the mean of per-dataset micro-F1 values.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)

    datasets = list(per_dataset_f1s.keys())
    n_datasets = len(datasets)

    # Point estimate
    point_f1s = []
    for ds in datasets:
        g, p = per_dataset_f1s[ds]["gold"], per_dataset_f1s[ds]["pred"]
        point_f1s.append((g == p).sum() / len(g))
    point_avg = np.mean(point_f1s)

    # Bootstrap: resample within each dataset, then average
    avg_samples = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ds_f1s = []
        for ds in datasets:
            g = per_dataset_f1s[ds]["gold"]
            p = per_dataset_f1s[ds]["pred"]
            n = len(g)
            idx = rng.integers(0, n, size=n)
            ds_f1 = (g[idx] == p[idx]).sum() / n
            ds_f1s.append(ds_f1)
        avg_samples[i] = np.mean(ds_f1s)

    ci_lower = float(np.percentile(avg_samples, 2.5))
    ci_upper = float(np.percentile(avg_samples, 97.5))

    return {
        "point_avg_f1": round(float(point_avg), 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_width": round(ci_upper - ci_lower, 4),
    }


# ═══════════════════════════════════════════════════════════════════════
# Label complexity analysis
# ═══════════════════════════════════════════════════════════════════════

def label_complexity_analysis(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Analyze F1 vs number of relation types per dataset.

    Uses per_dataset_metrics.csv (no generation CSVs needed).
    """
    # Count unique gold labels per dataset from per-class files or estimate
    # For now, use a hardcoded mapping from the dataset configs
    label_counts = {
        "tacred": 41,
        "semeval2010_task8": 19,
        "conll04": 5,
        "nyt11": 24,
        "gids": 5,
        "re_docred": 96,
        "rebel": 219,
        "biographical": 13,
        "pg_fiction": 9,
    }

    matched = metrics_df[
        (metrics_df["gen_type"] == "gen_constrained")
        & (metrics_df["model_shot"] == metrics_df["prompt_shot"])
    ].copy()

    rows = []
    for ds in matched["eval_dataset_name"].unique():
        ds_data = matched[matched["eval_dataset_name"] == ds]
        n_labels = label_counts.get(ds, -1)
        rows.append({
            "dataset": ds,
            "n_labels": n_labels,
            "mean_micro_f1": round(ds_data["micro_f1"].mean(), 4),
            "std_micro_f1": round(ds_data["micro_f1"].std(), 4),
            "min_micro_f1": round(ds_data["micro_f1"].min(), 4),
            "max_micro_f1": round(ds_data["micro_f1"].max(), 4),
            "n_configs": len(ds_data),
            "domain": "literary" if ds in DATASET_GROUPS["literary"] else "general",
        })

    result = pd.DataFrame(rows).sort_values("n_labels")

    # Compute correlation
    if len(result) > 2:
        # Pearson correlation (numpy-only)
        x = result["n_labels"].values.astype(float)
        y = result["mean_micro_f1"].values.astype(float)
        r_pearson = float(np.corrcoef(x, y)[0, 1])

        # Spearman via rank correlation
        def _rankdata(arr):
            order = arr.argsort()
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(arr) + 1, dtype=float)
            return ranks
        r_spearman = float(np.corrcoef(_rankdata(x), _rankdata(y))[0, 1])

        print(f"\n  Label complexity correlations:")
        print(f"    Pearson r = {r_pearson:.4f}")
        print(f"    Spearman ρ = {r_spearman:.4f}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════

def run_bootstrap_cis(
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compute bootstrap CIs for every model-config × dataset combination."""
    rng = np.random.default_rng(SEED)
    usecols = cfg.get("processing", {}).get("usecols", None)
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)

    all_rows = []
    datasets = sorted(
        d.name for d in generations_dir.iterdir()
        if d.is_dir() and (d / "generations.csv").exists()
    )

    for ds_name in datasets:
        print(f"\n  Bootstrap CIs: {ds_name}")
        t0 = time.time()

        df = load_generation_data(ds_name, generations_dir, usecols or [], chunk_size)
        if df.empty:
            print(f"    ✗ No data, skipping")
            continue

        matched = get_matched_constrained(df)
        if matched.empty:
            print(f"    ✗ No matched constrained data, skipping")
            continue

        exclude_labels_list = cfg.get("exclude_labels", {}).get(ds_name, [])
        exclude_labels = set(exclude_labels_list) if exclude_labels_list else None
        do_normalize = cfg.get("normalize", True)

        group_cols = ["model_id", "tuned_dataset_name", "model_shot"]
        for key, group in matched.groupby(group_cols):
            model_id, tuned_ds, m_shot = key

            gold = group["relation"].values
            pred = group["generated_relation"].values

            if do_normalize:
                gold = np.array([normalize_relation(g) for g in gold])
                pred = np.array([normalize_relation(p) for p in pred])

            ci = bootstrap_micro_f1(
                gold, pred, n_bootstrap=n_bootstrap, rng=rng,
                exclude_labels=exclude_labels,
            )

            all_rows.append({
                "eval_dataset_name": ds_name,
                "model_id": model_id,
                "tuned_dataset_name": tuned_ds,
                "model_shot": int(m_shot),
                **ci,
            })

        elapsed = time.time() - t0
        n_configs = matched[group_cols].drop_duplicates().shape[0]
        print(f"    ✓ {n_configs} configs in {elapsed:.1f}s")

    result = pd.DataFrame(all_rows)
    if not result.empty:
        out_path = output_dir / "bootstrap_cis.csv"
        result.to_csv(out_path, index=False)
        print(f"\n  → Saved {len(result)} rows to {out_path}")

    return result


def run_pairwise_tests(
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> pd.DataFrame:
    """Run paired bootstrap + McNemar for key comparisons.

    For comparisons involving frontier models (which aren't in generation CSVs),
    we skip per-example tests and note this.
    """
    rng = np.random.default_rng(SEED)
    usecols = cfg.get("processing", {}).get("usecols", None)
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)
    do_normalize = cfg.get("normalize", True)

    # Pre-load all datasets into memory (filtered to constrained+matched)
    print("\n  Loading generation data for pairwise tests...")
    dataset_cache: Dict[str, pd.DataFrame] = {}
    for ds_name in DATASET_GROUPS["all"]:
        df = load_generation_data(ds_name, generations_dir, usecols or [], chunk_size)
        if not df.empty:
            matched = get_matched_constrained(df)
            if do_normalize:
                matched["relation"] = matched["relation"].apply(normalize_relation)
                matched["generated_relation"] = matched["generated_relation"].apply(normalize_relation)
            dataset_cache[ds_name] = matched
            print(f"    ✓ {ds_name}: {len(matched)} rows")
        else:
            print(f"    ✗ {ds_name}: no data")

    all_rows = []
    for comp_name, config_a, config_b, domain_filter in KEY_COMPARISONS:
        print(f"\n  Comparison: {comp_name}")
        model_a, tuned_a, shot_a = config_a
        model_b, tuned_b, shot_b = config_b

        # Check if either is a frontier model (not in generation CSVs)
        is_frontier_a = tuned_a == "frontier"
        is_frontier_b = tuned_b == "frontier"

        if is_frontier_a or is_frontier_b:
            print(f"    ⚠ Frontier model comparison — skipping per-example tests")
            print(f"      (Frontier models evaluated via API; no per-example alignment)")
            all_rows.append({
                "comparison": comp_name,
                "model_a": f"{model_a} {tuned_a} {shot_a}s",
                "model_b": f"{model_b} {tuned_b} {shot_b}s",
                "domain": domain_filter,
                "test_type": "N/A (frontier)",
                "n_examples": 0,
                "delta_f1": None,
                "p_two_sided": None,
                "p_one_sided": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant_05": None,
                "note": "Frontier model — per-example alignment not available",
            })
            continue

        # Collect aligned examples across datasets in the domain
        datasets_for_domain = DATASET_GROUPS.get(domain_filter, DATASET_GROUPS["all"])

        all_gold = []
        all_pred_a = []
        all_pred_b = []

        for ds_name in datasets_for_domain:
            if ds_name not in dataset_cache:
                continue
            ds_df = dataset_cache[ds_name]

            # Get predictions for config A
            mask_a = (
                (ds_df["model_id"] == model_a)
                & (ds_df["tuned_dataset_name"] == tuned_a)
                & (ds_df["model_shot"] == shot_a)
            )
            df_a = ds_df[mask_a].copy()

            # Get predictions for config B
            mask_b = (
                (ds_df["model_id"] == model_b)
                & (ds_df["tuned_dataset_name"] == tuned_b)
                & (ds_df["model_shot"] == shot_b)
            )
            df_b = ds_df[mask_b].copy()

            if df_a.empty or df_b.empty:
                print(f"    ⚠ {ds_name}: missing data for one config, skipping")
                continue

            # Align by prompt_hash
            df_a = df_a.set_index("prompt_hash")
            df_b = df_b.set_index("prompt_hash")
            common_idx = df_a.index.intersection(df_b.index)

            if len(common_idx) == 0:
                print(f"    ⚠ {ds_name}: no overlapping examples")
                continue

            all_gold.extend(df_a.loc[common_idx, "relation"].values)
            all_pred_a.extend(df_a.loc[common_idx, "generated_relation"].values)
            all_pred_b.extend(df_b.loc[common_idx, "generated_relation"].values)

        if not all_gold:
            print(f"    ✗ No aligned examples found")
            continue

        gold_arr = np.array(all_gold)
        pred_a_arr = np.array(all_pred_a)
        pred_b_arr = np.array(all_pred_b)

        print(f"    Aligned examples: {len(gold_arr)}")

        # Paired bootstrap test
        pb = paired_bootstrap_test(gold_arr, pred_a_arr, pred_b_arr, n_bootstrap, rng)

        # McNemar's test
        mc = mcnemar_test(gold_arr, pred_a_arr, pred_b_arr)

        tuned_display = cfg.get("tuning_regime_display", {})
        all_rows.append({
            "comparison": comp_name,
            "model_a": f"{model_a} {tuned_display.get(tuned_a, tuned_a)} {shot_a}s",
            "model_b": f"{model_b} {tuned_display.get(tuned_b, tuned_b)} {shot_b}s",
            "domain": domain_filter,
            "n_examples": len(gold_arr),
            "f1_a": round(float((gold_arr == pred_a_arr).sum() / len(gold_arr)), 4),
            "f1_b": round(float((gold_arr == pred_b_arr).sum() / len(gold_arr)), 4),
            # Paired bootstrap
            "delta_f1": pb["delta_f1"],
            "pb_p_two_sided": pb["p_two_sided"],
            "pb_p_one_sided": pb["p_one_sided"],
            "pb_ci_lower": pb["ci_lower"],
            "pb_ci_upper": pb["ci_upper"],
            "pb_significant_05": pb["significant_05"],
            "pb_significant_01": pb["significant_01"],
            # McNemar
            "mcnemar_chi2": mc["mcnemar_chi2"],
            "mcnemar_p": mc["mcnemar_p_value"],
            "mcnemar_significant_05": mc["mcnemar_significant_05"],
            "n_a_only": mc["n_a_only_correct"],
            "n_b_only": mc["n_b_only_correct"],
            "odds_ratio": mc["odds_ratio"],
        })

        sig_str = "***" if pb["p_two_sided"] < 0.001 else ("**" if pb["p_two_sided"] < 0.01 else ("*" if pb["p_two_sided"] < 0.05 else "n.s."))
        print(f"    ΔF1 = {pb['delta_f1']:+.4f}, p = {pb['p_two_sided']:.4f} ({sig_str})")
        print(f"    McNemar χ² = {mc['mcnemar_chi2']:.2f}, p = {mc['mcnemar_p_value']:.4f}")

    result = pd.DataFrame(all_rows)
    if not result.empty:
        out_path = output_dir / "pairwise_significance.csv"
        result.to_csv(out_path, index=False)
        print(f"\n  → Saved {len(result)} comparisons to {out_path}")

    return result


def run_domain_avg_cis(
    generations_dir: Path,
    cfg: dict,
    output_dir: Path,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
) -> pd.DataFrame:
    """Compute bootstrap CIs for domain-averaged F1 (for Table 3 annotations)."""
    rng = np.random.default_rng(SEED)
    usecols = cfg.get("processing", {}).get("usecols", None)
    chunk_size = cfg.get("processing", {}).get("chunk_size", 500_000)
    do_normalize = cfg.get("normalize", True)

    # Load all datasets
    dataset_data: Dict[str, pd.DataFrame] = {}
    for ds_name in DATASET_GROUPS["all"]:
        df = load_generation_data(ds_name, generations_dir, usecols or [], chunk_size)
        if not df.empty:
            matched = get_matched_constrained(df)
            if do_normalize:
                matched["relation"] = matched["relation"].apply(normalize_relation)
                matched["generated_relation"] = matched["generated_relation"].apply(normalize_relation)
            dataset_data[ds_name] = matched

    # For each model config, compute domain-avg CI
    all_rows = []
    configs_seen = set()

    for ds_name, ds_df in dataset_data.items():
        group_cols = ["model_id", "tuned_dataset_name", "model_shot"]
        for key, _ in ds_df.groupby(group_cols):
            configs_seen.add(key)

    print(f"\n  Computing domain-avg CIs for {len(configs_seen)} model configs...")

    for model_id, tuned_ds, m_shot in sorted(configs_seen):
        for domain_name, domain_datasets in DATASET_GROUPS.items():
            if domain_name == "all":
                continue  # skip "all" — we do general + literary + overall

            per_ds = {}
            for ds_name in domain_datasets:
                if ds_name not in dataset_data:
                    continue
                ds_df = dataset_data[ds_name]
                mask = (
                    (ds_df["model_id"] == model_id)
                    & (ds_df["tuned_dataset_name"] == tuned_ds)
                    & (ds_df["model_shot"] == m_shot)
                )
                sub = ds_df[mask]
                if not sub.empty:
                    per_ds[ds_name] = {
                        "gold": sub["relation"].values,
                        "pred": sub["generated_relation"].values,
                    }

            if not per_ds:
                continue

            ci = bootstrap_domain_avg_f1(per_ds, n_bootstrap, rng)
            tuned_display = cfg.get("tuning_regime_display", {})

            all_rows.append({
                "model_id": model_id,
                "tuned_dataset_name": tuned_ds,
                "tuning_display": tuned_display.get(tuned_ds, tuned_ds),
                "model_shot": int(m_shot),
                "domain": domain_name,
                "n_datasets": len(per_ds),
                **ci,
            })

    # Also compute overall avg
    for model_id, tuned_ds, m_shot in sorted(configs_seen):
        per_ds = {}
        for ds_name in DATASET_GROUPS["all"]:
            if ds_name not in dataset_data:
                continue
            ds_df = dataset_data[ds_name]
            mask = (
                (ds_df["model_id"] == model_id)
                & (ds_df["tuned_dataset_name"] == tuned_ds)
                & (ds_df["model_shot"] == m_shot)
            )
            sub = ds_df[mask]
            if not sub.empty:
                per_ds[ds_name] = {
                    "gold": sub["relation"].values,
                    "pred": sub["generated_relation"].values,
                }

        if not per_ds:
            continue

        ci = bootstrap_domain_avg_f1(per_ds, n_bootstrap, rng)
        tuned_display = cfg.get("tuning_regime_display", {})

        all_rows.append({
            "model_id": model_id,
            "tuned_dataset_name": tuned_ds,
            "tuning_display": tuned_display.get(tuned_ds, tuned_ds),
            "model_shot": int(m_shot),
            "domain": "overall",
            "n_datasets": len(per_ds),
            **ci,
        })

    result = pd.DataFrame(all_rows)
    if not result.empty:
        out_path = output_dir / "summary_cis_for_table3.csv"
        result.to_csv(out_path, index=False)
        print(f"\n  → Saved {len(result)} rows to {out_path}")

    return result


# ═══════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Statistical significance tests")
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--comparisons-only", action="store_true",
                        help="Only run pairwise comparisons (skip per-config CIs)")
    parser.add_argument("--cis-only", action="store_true",
                        help="Only run bootstrap CIs (skip pairwise tests)")
    parser.add_argument("--label-complexity-only", action="store_true",
                        help="Only run label complexity analysis")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generations_dir = Path(cfg["paths"]["generations_dir"])
    if not generations_dir.is_absolute():
        generations_dir = REPO_ROOT / generations_dir

    output_dir = REPO_ROOT / "runs" / "evaluation" / "statistical"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Statistical Tests Pipeline")
    print(f"{'='*60}")
    print(f"  Bootstrap iterations: {args.n_bootstrap}")
    print(f"  Confidence level: {CONFIDENCE_LEVEL}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Label complexity (always fast, uses pre-computed metrics)
    if not args.comparisons_only and not args.cis_only:
        print("\n" + "="*60)
        print("Label Complexity Analysis")
        print("="*60)
        metrics_path = REPO_ROOT / "runs" / "evaluation" / "per_dataset_metrics.csv"
        if metrics_path.exists():
            metrics_df = pd.read_csv(metrics_path)
            lc = label_complexity_analysis(metrics_df)
            lc.to_csv(output_dir / "label_complexity.csv", index=False)
            print(f"\n  → Saved to {output_dir / 'label_complexity.csv'}")
            print(lc.to_string(index=False))
        else:
            print(f"  ✗ {metrics_path} not found")

    if args.label_complexity_only:
        return

    # Bootstrap CIs
    if not args.comparisons_only:
        print("\n" + "="*60)
        print("Bootstrap Confidence Intervals (per config × dataset)")
        print("="*60)
        run_bootstrap_cis(generations_dir, cfg, output_dir, args.n_bootstrap)

        print("\n" + "="*60)
        print("Domain-Average Bootstrap CIs (for Table 3)")
        print("="*60)
        run_domain_avg_cis(generations_dir, cfg, output_dir, args.n_bootstrap)

    # Pairwise tests
    if not args.cis_only:
        print("\n" + "="*60)
        print("Pairwise Significance Tests")
        print("="*60)
        run_pairwise_tests(generations_dir, cfg, output_dir, args.n_bootstrap)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

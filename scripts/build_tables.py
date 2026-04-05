#!/usr/bin/env python3
"""
Build paper tables from per_dataset_metrics.csv.

Produces CSV files under  runs/evaluation/tables/  corresponding to:
  - Table 3: main_summary.csv          (30-model matrix: General/Lit/Overall Avg F1)
  - Table 4: cross_domain_transfer.csv (GenTune/LitTune/MixTune domain analysis)
  - Table 5: prompt_delta.csv          (2-shot minus 0-shot deltas)
  - Table 9: efficiency_tradeoffs.csv  (params, size, latency, normalised F1)

Tables 6 (DaptTune), 7 (Frontier), 8 (Qualitative) are handled separately
or require manual input.

Usage:
    python scripts/build_tables.py --config configs/eval.yaml
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


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_path(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else REPO_ROOT / pp


def _load_metrics(cfg: dict) -> pd.DataFrame:
    output_dir = _resolve_path(cfg["paths"]["output_dir"])
    csv = output_dir / "per_dataset_metrics.csv"
    if not csv.exists():
        raise FileNotFoundError(
            f"{csv} not found. Run scripts/run_evaluation.py first."
        )
    return pd.read_csv(csv)


def _domain_for_dataset(dataset_name: str, cfg: dict) -> str:
    groups = cfg["dataset_groups"]
    if dataset_name in groups["general"]:
        return "general"
    elif dataset_name in groups["literary"]:
        return "literary"
    return "unknown"


def _filter_primary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Filter to primary gen_type and matched prompt-shot only."""
    gt = cfg.get("primary_gen_type", "gen_constrained")
    out = df[df["gen_type"] == gt].copy()

    policy = cfg.get("prompt_shot_policy", "matched")
    if policy == "matched":
        out = out[out["model_shot"] == out["prompt_shot"]].copy()

    return out


def _display_name(tuned_ds: str, cfg: dict) -> str:
    return cfg.get("tuning_regime_display", {}).get(tuned_ds, tuned_ds)


# ═══════════════════════════════════════════════════════════════════
# TABLE 3: Main Summary (30-model matrix)
# ═══════════════════════════════════════════════════════════════════

def build_table3(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """30 rows: (base_model × prompt_style × tuning_regime) with domain-avg F1."""

    primary = _filter_primary(df, cfg)
    groups = cfg["dataset_groups"]

    rows = []
    model_meta = cfg.get("model_metadata", {})

    for (model_id, m_shot, tuned_ds), grp in primary.groupby(
        ["model_id", "model_shot", "tuned_dataset_name"]
    ):
        general_datasets = [d for d in groups["general"] if d in grp["eval_dataset_name"].values]
        literary_datasets = [d for d in groups["literary"] if d in grp["eval_dataset_name"].values]

        gen_f1s = grp[grp["eval_dataset_name"].isin(general_datasets)]["micro_f1"]
        lit_f1s = grp[grp["eval_dataset_name"].isin(literary_datasets)]["micro_f1"]
        all_f1s = grp["micro_f1"]

        general_avg = gen_f1s.mean() if len(gen_f1s) > 0 else np.nan
        literary_avg = lit_f1s.mean() if len(lit_f1s) > 0 else np.nan
        overall_avg = all_f1s.mean() if len(all_f1s) > 0 else np.nan

        meta = model_meta.get(model_id, {})

        rows.append({
            "base_model": model_id,
            "params": meta.get("param_label", ""),
            "prompt_style": f"{int(m_shot)}-shot",
            "tuning_regime": _display_name(tuned_ds, cfg),
            "general_avg_f1": round(general_avg, 4) if not np.isnan(general_avg) else np.nan,
            "literature_avg_f1": round(literary_avg, 4) if not np.isnan(literary_avg) else np.nan,
            "overall_avg_f1": round(overall_avg, 4) if not np.isnan(overall_avg) else np.nan,
        })

    result = pd.DataFrame(rows)

    # Sort: by model param size, then prompt style, then tuning regime
    param_order = {"360M": 0, "0.5B": 1, "3B": 2}
    regime_order = {"GenTune": 0, "LitTune": 1, "MixTune": 2}
    result["_param_sort"] = result["params"].map(param_order).fillna(99)
    result["_regime_sort"] = result["tuning_regime"].map(regime_order).fillna(99)
    result["_shot_sort"] = result["prompt_style"].map({"0-shot": 0, "2-shot": 1}).fillna(99)

    result = result.sort_values(
        ["base_model", "_param_sort", "_shot_sort", "_regime_sort"]
    ).drop(columns=["_param_sort", "_regime_sort", "_shot_sort"]).reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# TABLE 4: Cross-Domain Transfer by Tuning Regime
# ═══════════════════════════════════════════════════════════════════

def build_table4(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """3 rows: GenTune, LitTune, MixTune — averaged across all models + prompt styles."""

    primary = _filter_primary(df, cfg)
    groups = cfg["dataset_groups"]

    rows = []
    for tuned_ds in ["re_gentune", "re_littune", "re_mixtune"]:
        subset = primary[primary["tuned_dataset_name"] == tuned_ds]
        if subset.empty:
            continue

        # For each model config, compute domain averages, then average across configs
        config_rows = []
        for (model_id, m_shot), config_grp in subset.groupby(["model_id", "model_shot"]):
            gen_f1s = config_grp[config_grp["eval_dataset_name"].isin(groups["general"])]["micro_f1"]
            lit_f1s = config_grp[config_grp["eval_dataset_name"].isin(groups["literary"])]["micro_f1"]

            gen_avg = gen_f1s.mean() if len(gen_f1s) > 0 else np.nan
            lit_avg = lit_f1s.mean() if len(lit_f1s) > 0 else np.nan

            config_rows.append({"gen_avg": gen_avg, "lit_avg": lit_avg})

        config_df = pd.DataFrame(config_rows)
        avg_gen = config_df["gen_avg"].mean()
        avg_lit = config_df["lit_avg"].mean()
        overall = np.nanmean([avg_gen, avg_lit]) if not (np.isnan(avg_gen) and np.isnan(avg_lit)) else np.nan

        # Cross-domain gap (per paper definition):
        #   GenTune: General - Literature (positive = better on trained domain)
        #   LitTune: Literature - General (positive = better on trained domain)
        #   MixTune: |General - Literature| (absolute imbalance)
        if tuned_ds == "re_gentune":
            gap = avg_gen - avg_lit if not (np.isnan(avg_gen) or np.isnan(avg_lit)) else np.nan
        elif tuned_ds == "re_littune":
            gap = avg_lit - avg_gen if not (np.isnan(avg_gen) or np.isnan(avg_lit)) else np.nan
        else:  # mixtune
            gap = abs(avg_gen - avg_lit) if not (np.isnan(avg_gen) or np.isnan(avg_lit)) else np.nan

        rows.append({
            "tuning_regime": _display_name(tuned_ds, cfg),
            "avg_on_general": round(avg_gen, 4) if not np.isnan(avg_gen) else np.nan,
            "avg_on_literature": round(avg_lit, 4) if not np.isnan(avg_lit) else np.nan,
            "cross_domain_gap": round(gap, 4) if not np.isnan(gap) else np.nan,
            "overall_avg": round(overall, 4) if not np.isnan(overall) else np.nan,
        })

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════
# TABLE 5: Prompt-Conditioned Tuning Delta (2-shot - 0-shot)
# ═══════════════════════════════════════════════════════════════════

def build_table5(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """15 rows: (base_model × tuning_regime) — deltas between 2-shot and 0-shot tuning."""

    primary = _filter_primary(df, cfg)
    groups = cfg["dataset_groups"]

    rows = []
    for (model_id, tuned_ds), model_grp in primary.groupby(
        ["model_id", "tuned_dataset_name"]
    ):
        shot0 = model_grp[model_grp["model_shot"] == 0]
        shot2 = model_grp[model_grp["model_shot"] == 2]

        if shot0.empty or shot2.empty:
            continue

        def _avg_metric(sub, metric, dataset_list):
            vals = sub[sub["eval_dataset_name"].isin(dataset_list)][metric]
            return vals.mean() if len(vals) > 0 else np.nan

        gen_f1_0 = _avg_metric(shot0, "micro_f1", groups["general"])
        gen_f1_2 = _avg_metric(shot2, "micro_f1", groups["general"])
        lit_f1_0 = _avg_metric(shot0, "micro_f1", groups["literary"])
        lit_f1_2 = _avg_metric(shot2, "micro_f1", groups["literary"])
        all_f1_0 = shot0["micro_f1"].mean()
        all_f1_2 = shot2["micro_f1"].mean()
        prec_0 = shot0["micro_precision"].mean()
        prec_2 = shot2["micro_precision"].mean()
        rec_0 = shot0["micro_recall"].mean()
        rec_2 = shot2["micro_recall"].mean()

        def _delta(a, b):
            if np.isnan(a) or np.isnan(b):
                return np.nan
            return round(a - b, 4)

        rows.append({
            "base_model": model_id,
            "tuning_regime": _display_name(tuned_ds, cfg),
            "delta_general_avg_f1": _delta(gen_f1_2, gen_f1_0),
            "delta_literature_avg_f1": _delta(lit_f1_2, lit_f1_0),
            "delta_overall_avg_f1": _delta(all_f1_2, all_f1_0),
            "delta_precision": _delta(prec_2, prec_0),
            "delta_recall": _delta(rec_2, rec_0),
        })

    result = pd.DataFrame(rows)
    return result


# ═══════════════════════════════════════════════════════════════════
# TABLE 9: Efficiency & Deployment Trade-offs
# ═══════════════════════════════════════════════════════════════════

def build_table9(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """One row per best-performing model variant, plus model metadata."""

    primary = _filter_primary(df, cfg)
    groups = cfg["dataset_groups"]
    model_meta = cfg.get("model_metadata", {})

    # Find best config per model_id (by overall avg F1)
    model_scores = []
    for (model_id, m_shot, tuned_ds), grp in primary.groupby(
        ["model_id", "model_shot", "tuned_dataset_name"]
    ):
        lit_f1s = grp[grp["eval_dataset_name"].isin(groups["literary"])]["micro_f1"]
        all_f1s = grp["micro_f1"]

        lit_avg = lit_f1s.mean() if len(lit_f1s) > 0 else np.nan
        overall_avg = all_f1s.mean() if len(all_f1s) > 0 else np.nan

        meta = model_meta.get(model_id, {})

        model_scores.append({
            "model": f"{model_id} ({_display_name(tuned_ds, cfg)}, {int(m_shot)}-shot)",
            "model_id": model_id,
            "params_b": meta.get("params_b", np.nan),
            "checkpoint_gb": meta.get("checkpoint_gb", np.nan),
            "avg_inference_latency_ms": meta.get("avg_inference_latency_ms", np.nan),
            "literature_avg_f1": round(lit_avg, 4) if not np.isnan(lit_avg) else np.nan,
            "overall_avg_f1": round(overall_avg, 4) if not np.isnan(overall_avg) else np.nan,
        })

    result = pd.DataFrame(model_scores)

    if not result.empty:
        # Ensure numeric types
        for col in ["params_b", "checkpoint_gb", "overall_avg_f1", "literature_avg_f1"]:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")

        # Normalized measures
        result["f1_per_b_params"] = (
            result["overall_avg_f1"] / result["params_b"]
        ).round(4)
        mask = result["checkpoint_gb"].notna() & (result["checkpoint_gb"] > 0)
        result["f1_per_gb"] = np.nan
        result.loc[mask, "f1_per_gb"] = (
            result.loc[mask, "overall_avg_f1"] / result.loc[mask, "checkpoint_gb"]
        ).round(4)

    # Sort by overall F1 descending
    result = result.sort_values("overall_avg_f1", ascending=False).reset_index(drop=True)

    return result


# ═══════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Full per-dataset grid (for Figure 1 source data)
# ═══════════════════════════════════════════════════════════════════

def build_full_grid(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """30 models × 9 datasets grid of micro-F1 (source data for heatmap)."""

    primary = _filter_primary(df, cfg)

    # Build a label for each model config
    primary = primary.copy()
    primary["model_config"] = (
        primary["model_id"] + " | "
        + primary["tuned_dataset_name"].map(
            lambda x: cfg.get("tuning_regime_display", {}).get(x, x)
        )
        + " | " + primary["model_shot"].astype(str) + "-shot"
    )

    pivot = primary.pivot_table(
        index="model_config",
        columns="eval_dataset_name",
        values="micro_f1",
        aggfunc="first",
    )

    # Reorder columns by domain group
    groups = cfg["dataset_groups"]
    col_order = [c for c in groups["general"] if c in pivot.columns] + \
                [c for c in groups["literary"] if c in pivot.columns]
    pivot = pivot[col_order]

    return pivot.round(4)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build paper tables from evaluation metrics")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    df = _load_metrics(cfg)

    tables_dir = _resolve_path(cfg["paths"]["output_dir"]) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Table 3
    t3 = build_table3(df, cfg)
    t3.to_csv(tables_dir / "table3_main_summary.csv", index=False)
    print(f"Table 3: {len(t3)} rows → {tables_dir / 'table3_main_summary.csv'}")
    print(t3.to_string(index=False))
    print()

    # Table 4
    t4 = build_table4(df, cfg)
    t4.to_csv(tables_dir / "table4_cross_domain_transfer.csv", index=False)
    print(f"Table 4: {len(t4)} rows → {tables_dir / 'table4_cross_domain_transfer.csv'}")
    print(t4.to_string(index=False))
    print()

    # Table 5
    t5 = build_table5(df, cfg)
    t5.to_csv(tables_dir / "table5_prompt_delta.csv", index=False)
    print(f"Table 5: {len(t5)} rows → {tables_dir / 'table5_prompt_delta.csv'}")
    print(t5.to_string(index=False))
    print()

    # Table 9
    t9 = build_table9(df, cfg)
    t9.to_csv(tables_dir / "table9_efficiency_tradeoffs.csv", index=False)
    print(f"Table 9: {len(t9)} rows → {tables_dir / 'table9_efficiency_tradeoffs.csv'}")
    print(t9.head(10).to_string(index=False))
    print()

    # Full grid (Figure 1 source data)
    grid = build_full_grid(df, cfg)
    grid.to_csv(tables_dir / "full_grid_f1.csv")
    print(f"Full grid: {grid.shape} → {tables_dir / 'full_grid_f1.csv'}")
    print(grid.to_string())


if __name__ == "__main__":
    main()

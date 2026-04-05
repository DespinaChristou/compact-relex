#!/usr/bin/env python3
"""
Build paper figures from per_dataset_metrics.csv.

Produces:
  - figures/heatmap_all_models_f1.pdf   (Figure 1: 30×9 heatmap)
  - figures/scaling_trends_overall.pdf  (Figure 2: F1 vs model scale)

Usage:
    python scripts/build_figures.py --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
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
        raise FileNotFoundError(f"{csv} not found. Run scripts/run_evaluation.py first.")
    return pd.read_csv(csv)


def _filter_primary(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    gt = cfg.get("primary_gen_type", "gen_constrained")
    out = df[df["gen_type"] == gt].copy()
    policy = cfg.get("prompt_shot_policy", "matched")
    if policy == "matched":
        out = out[out["model_shot"] == out["prompt_shot"]].copy()
    return out


def _display_regime(tuned_ds: str, cfg: dict) -> str:
    return cfg.get("tuning_regime_display", {}).get(tuned_ds, tuned_ds)


# ═══════════════════════════════════════════════════════════════════
# FIGURE 1: Heatmap of per-dataset F1 for all 30 tuned SLMs
# ═══════════════════════════════════════════════════════════════════

def build_heatmap(df: pd.DataFrame, cfg: dict, out_path: Path):
    primary = _filter_primary(df, cfg)
    groups = cfg["dataset_groups"]

    # Build row labels
    primary = primary.copy()
    primary["row_label"] = (
        primary["model_id"]
        + " | "
        + primary["tuned_dataset_name"].map(lambda x: _display_regime(x, cfg))
        + " | "
        + primary["model_shot"].astype(int).astype(str) + "-shot"
    )

    # Pivot to matrix
    col_order = (
        [c for c in groups["general"] if c in primary["eval_dataset_name"].unique()]
        + [c for c in groups["literary"] if c in primary["eval_dataset_name"].unique()]
    )

    pivot = primary.pivot_table(
        index="row_label",
        columns="eval_dataset_name",
        values="micro_f1",
        aggfunc="first",
    )
    pivot = pivot.reindex(columns=col_order)

    # Sort rows by overall average F1 descending for readability
    pivot["_avg"] = pivot.mean(axis=1)
    pivot = pivot.sort_values("_avg", ascending=True)  # ascending so best is at top in imshow
    pivot = pivot.drop(columns=["_avg"])

    # Prettify column names
    col_display = {
        "tacred": "TACRED",
        "semeval2010_task8": "SemEval",
        "conll04": "CoNLL04",
        "nyt11": "NYT-11",
        "gids": "GIDS",
        "re_docred": "Re-DocRED",
        "rebel": "REBEL",
        "biographical": "Biograph.",
        "pg_fiction": "PG-Fiction",
    }
    pivot = pivot.rename(columns=col_display)

    # Plot
    fig_height = max(8, len(pivot) * 0.35)
    fig_width = max(10, len(pivot.columns) * 1.1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Micro-F1", "shrink": 0.6},
        annot_kws={"fontsize": 7},
    )

    # Domain separator line
    n_general = len([c for c in groups["general"] if col_display.get(c, c) in pivot.columns])
    ax.axvline(x=n_general, color="black", linewidth=2)

    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title("Per-Dataset Micro-F1 for All Tuned SLMs", fontsize=13, pad=12)

    # Labels
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45, ha="right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 1 → {out_path} ({pivot.shape[0]}×{pivot.shape[1]})")


# ═══════════════════════════════════════════════════════════════════
# FIGURE 2: Scaling Trends — Overall Avg F1 vs Model Size
# ═══════════════════════════════════════════════════════════════════

def build_scaling_plot(df: pd.DataFrame, cfg: dict, out_path: Path):
    primary = _filter_primary(df, cfg)
    model_meta = cfg.get("model_metadata", {})

    # Compute overall avg F1 per (model_id, model_shot, tuned_dataset_name)
    records = []
    for (model_id, m_shot, tuned_ds), grp in primary.groupby(
        ["model_id", "model_shot", "tuned_dataset_name"]
    ):
        overall_f1 = grp["micro_f1"].mean()
        meta = model_meta.get(model_id, {})
        params_b = meta.get("params_b", np.nan)

        records.append({
            "model_id": model_id,
            "params_b": params_b,
            "prompt_style": f"{int(m_shot)}-shot",
            "tuning_regime": _display_regime(tuned_ds, cfg),
            "overall_avg_f1": overall_f1,
        })

    plot_df = pd.DataFrame(records)

    if plot_df.empty:
        print("  No data for scaling plot.")
        return

    # Plot: one line per (tuning_regime, prompt_style)
    fig, ax = plt.subplots(figsize=(10, 6))

    regime_colors = {"GenTune": "#2196F3", "LitTune": "#4CAF50", "MixTune": "#FF9800"}
    shot_markers = {"0-shot": "o", "2-shot": "s"}
    shot_linestyles = {"0-shot": "--", "2-shot": "-"}

    for (regime, shot), sub in plot_df.groupby(["tuning_regime", "prompt_style"]):
        sub = sub.sort_values("params_b")
        color = regime_colors.get(regime, "gray")
        marker = shot_markers.get(shot, "D")
        ls = shot_linestyles.get(shot, "-")

        ax.plot(
            sub["params_b"],
            sub["overall_avg_f1"],
            marker=marker,
            linestyle=ls,
            color=color,
            label=f"{regime} ({shot})",
            markersize=7,
            linewidth=1.5,
            alpha=0.85,
        )

    ax.set_xlabel("Parameters (Billions)", fontsize=12)
    ax.set_ylabel("Overall Average F1", fontsize=12)
    ax.set_title("Scaling Trends: Overall Avg F1 by Model Size", fontsize=13, pad=12)

    # X-axis: log-ish scale with specific ticks
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}B" if x >= 1 else f"{x*1000:.0f}M"))
    ax.set_xlim(0.2, 5)

    ax.legend(fontsize=9, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 2 → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# BONUS: Schema-valid rate comparison (gen_open vs gen_constrained)
# ═══════════════════════════════════════════════════════════════════

def build_schema_valid_comparison(df: pd.DataFrame, cfg: dict, out_path: Path):
    """Bar chart comparing schema-valid rates across gen_open vs gen_constrained."""

    policy = cfg.get("prompt_shot_policy", "matched")
    sub = df.copy()
    if policy == "matched":
        sub = sub[sub["model_shot"] == sub["prompt_shot"]]

    # Average schema-valid rate per (model_id, gen_type)
    avg = sub.groupby(["model_id", "gen_type"])["schema_valid_rate"].mean().reset_index()
    pivot = avg.pivot(index="model_id", columns="gen_type", values="schema_valid_rate")

    if pivot.empty:
        print("  No data for schema-valid comparison.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(pivot))
    width = 0.35

    if "gen_open" in pivot.columns:
        ax.bar(x - width/2, pivot["gen_open"], width, label="Open", color="#FF7043", alpha=0.85)
    if "gen_constrained" in pivot.columns:
        ax.bar(x + width/2, pivot["gen_constrained"], width, label="Constrained", color="#42A5F5", alpha=0.85)

    ax.set_xlabel("Model")
    ax.set_ylabel("Schema-Valid Output Rate")
    ax.set_title("Schema-Valid Rate: Open vs Constrained Generation")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Schema-valid comparison → {out_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Build paper figures from evaluation metrics")
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    df = _load_metrics(cfg)

    figures_dir = _resolve_path(cfg["paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Heatmap
    build_heatmap(df, cfg, figures_dir / "heatmap_all_models_f1.pdf")

    # Figure 2: Scaling trends
    build_scaling_plot(df, cfg, figures_dir / "scaling_trends_overall.pdf")

    # Bonus: Schema-valid rate comparison
    build_schema_valid_comparison(df, cfg, figures_dir / "schema_valid_comparison.pdf")

    print("\nAll figures generated.")


if __name__ == "__main__":
    main()

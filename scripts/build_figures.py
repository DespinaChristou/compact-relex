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

from src.eval import GEN_SCHEMA_ENUMERATED, GEN_GENERIC


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
    gt = cfg.get("primary_gen_type", GEN_SCHEMA_ENUMERATED)
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

    # SmolLM3-3B MixTune 0-shot is reported under the pre-specified DEFAULT protocol,
    # where this reasoning model emits <think> tokens in place of a label and scores 0
    # (Section 4). per_dataset_metrics.csv stores the post-hoc "reasoning-disabled"
    # recovery (~0.18); for the headline heatmap we show the default-primary value (0) so
    # the figure agrees with Table 3 and the full-results matrix. The 0.18 recovery is
    # reported only in the text/footnotes as a post-hoc rescue.
    _default_zero = (
        (primary["model_id"] == "SmolLM3-3B")
        & (primary["tuned_dataset_name"] == "re_mixtune")
        & (primary["model_shot"] == 0)
    )
    primary.loc[_default_zero, "micro_f1"] = 0.0

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
    """Within-family scale plot. Model scale is confounded with family (tokenizer,
    pretraining, instruction tuning), so we connect lines ONLY within a family
    (Qwen2.5 0.5B->3B; SmolLM 360M->3B) and draw Llama-3.2-3B as an isolated 3B
    point (no sub-billion counterpart). We deliberately do NOT draw a single trend
    across the unrelated architectures, to avoid implying a continuous scaling law."""
    from matplotlib.lines import Line2D

    primary = _filter_primary(df, cfg)
    model_meta = cfg.get("model_metadata", {})

    # Pre-specified exclusion (Section 4): the two 0-shot decoding/template artifacts.
    _anom = (
        ((primary["model_id"] == "SmolLM3-3B")
         & (primary["tuned_dataset_name"] == "re_mixtune") & (primary["model_shot"] == 0))
        | ((primary["model_id"] == "Qwen2.5-3B-Instruct")
           & (primary["tuned_dataset_name"] == "re_gentune") & (primary["model_shot"] == 0))
    )
    primary = primary[~_anom].copy()

    FAMILY = {"SmolLM2-360M-Instruct": "SmolLM", "SmolLM3-3B": "SmolLM",
              "Qwen2.5-0.5B-Instruct": "Qwen", "Qwen2.5-3B-Instruct": "Qwen",
              "Llama-3.2-3B-Instruct": "Llama"}
    PAIRS = {"Qwen": ("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct"),
             "SmolLM": ("SmolLM2-360M-Instruct", "SmolLM3-3B")}
    FAM_COLOR = {"Qwen": "#2C7FB8", "SmolLM": "#41AB5D", "Llama": "#E6550D"}

    def pbf(m):
        return model_meta.get(m, {}).get("params_b", np.nan)

    # regime-average F1 per (model, regime, shot)
    rec = (primary.groupby(["model_id", "tuned_dataset_name", "model_shot"])["micro_f1"]
                  .mean().reset_index())
    rec["params_b"] = rec["model_id"].map(pbf)

    regimes = [("re_gentune", "GenTune"), ("re_littune", "LitTune"), ("re_mixtune", "MixTune")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)

    def style(shot, color):
        # 0-shot: dashed line, open marker; 2-shot: solid line, filled marker
        return dict(linestyle=("--" if shot == 0 else "-"),
                    marker=("o" if shot == 0 else "s"),
                    color=color, mec=color,
                    mfc=("white" if shot == 0 else color), ms=8, lw=1.7)

    for ax, (reg, rn) in zip(axes, regimes):
        sub = rec[rec["tuned_dataset_name"] == reg]
        # within-family lines (connect only same-family sizes)
        for fam, (sm, bg) in PAIRS.items():
            for shot in (0, 2):
                pts = sub[(sub["model_id"].isin([sm, bg])) & (sub["model_shot"] == shot)].sort_values("params_b")
                if len(pts) == 2:
                    ax.plot(pts["params_b"], pts["micro_f1"], zorder=3, **style(shot, FAM_COLOR[fam]))
                elif len(pts) == 1:  # one endpoint excluded as a 0-shot anomaly -> lone point
                    st = style(shot, FAM_COLOR[fam]); st.pop("linestyle"); st.pop("lw")
                    ax.plot(pts["params_b"], pts["micro_f1"], zorder=3, **st)
        # Llama: isolated 3B points, no connecting line
        for _, r in sub[sub["model_id"] == "Llama-3.2-3B-Instruct"].iterrows():
            st = style(int(r["model_shot"]), FAM_COLOR["Llama"]); st.pop("linestyle"); st.pop("lw")
            ax.plot(r["params_b"], r["micro_f1"], zorder=4, **st)

        ax.set_xscale("log")
        ax.minorticks_off()  # drop the log minor-tick labels (3e-1, 2e0, ...) that clash
        ax.set_xticks([0.36, 0.5, 3.0])
        ax.set_xticklabels(["360M", "0.5B", "3B"])
        ax.set_xlim(0.30, 4.0)
        ax.set_title(rn, fontsize=12)
        ax.set_xlabel("Parameters (log scale)", fontsize=11)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Avg positive-class micro-F1", fontsize=11)
    legend = [
        Line2D([], [], color=FAM_COLOR["Qwen"], marker="s", lw=1.7, label=r"Qwen2.5 (0.5B$\to$3B)"),
        Line2D([], [], color=FAM_COLOR["SmolLM"], marker="s", lw=1.7, label=r"SmolLM (360M$\to$3B)"),
        Line2D([], [], color=FAM_COLOR["Llama"], marker="s", ls="none", label="Llama-3.2-3B (3B only)"),
        Line2D([], [], color="gray", ls="--", marker="o", mfc="white", label="0-shot"),
        Line2D([], [], color="gray", ls="-", marker="s", label="2-shot"),
    ]
    axes[-1].legend(handles=legend, fontsize=8.5, loc="lower right", frameon=True)
    fig.suptitle("Within-family scaling: lines connect same-family sizes; the three 3B points are distinct "
                 "architectures, not a continuous scaling curve", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    if GEN_GENERIC in pivot.columns:
        ax.bar(x - width/2, pivot[GEN_GENERIC], width, label="Open", color="#FF7043", alpha=0.85)
    if GEN_SCHEMA_ENUMERATED in pivot.columns:
        ax.bar(x + width/2, pivot[GEN_SCHEMA_ENUMERATED], width, label="Constrained", color="#42A5F5", alpha=0.85)

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

#!/usr/bin/env python3
"""
Build Figure 3 (constrained vs. open generation) from per_dataset_metrics.csv.

Reproduces the two-panel figure used in the paper:
  (a) per-dataset Delta F1 (open - constrained) with SEM error bars and a dashed
      line at the overall mean;
  (b) breakdown by model scale (sub-billion vs. 3B).

Matched prompt shots only; the two anomalous 0-shot configurations
(SmolLM3-3B MixTune, Qwen2.5-3B GenTune) are excluded, consistent with the paper.

Note: this now reads the canonical runs/evaluation/per_dataset_metrics.csv. The old
per_dataset_metrics_corrected.csv was a manual pg_fiction patch (empty/abstained
predictions credited as the "none" class); that correction is now reproduced by the
pipeline itself via src/eval.py:normalize_relation, so the separate file is retired.

Usage:
    python scripts/build_constrained_vs_open.py \
        --metrics runs/evaluation/per_dataset_metrics.csv \
        --out "Compact Relex Journal Paper/figures/constrained_vs_open.pdf"
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GEN_SCHEMA_ENUMERATED = "gen_constrained"  # paper: schema-enumerated prompting
GEN_GENERIC = "gen_open"                   # paper: generic prompting

# Canonical, paper-consistent display names (fixes the old "GDS"/"DocRED" labels).
DISPLAY = {
    "tacred": "TACRED",
    "semeval2010_task8": "SemEval",
    "conll04": "CoNLL04",
    "nyt11": "NYT11",
    "gids": "GIDS",
    "re_docred": "Re-DocRED",
    "rebel": "REBEL",
    "biographical": "Biographical",
    "pg_fiction": "PG-Fiction",
}
ORDER = list(DISPLAY.keys())

ANOMALIES = [
    ("SmolLM3-3B", "re_mixtune", 0),
    ("Qwen2.5-3B-Instruct", "re_gentune", 0),
]


def load_deltas(metrics_csv: Path):
    d = pd.read_csv(metrics_csv, dtype={"model_shot": int, "prompt_shot": int})
    d = d[d.model_shot == d.prompt_shot].copy()  # matched shots
    for mid, reg, ms in ANOMALIES:
        d = d[~((d.model_id == mid) & (d.tuned_dataset_name == reg) & (d.model_shot == ms))]
    piv = d.pivot_table(
        index=["eval_dataset_name", "model_id", "tuned_dataset_name", "model_shot"],
        columns="gen_type", values="micro_f1",
    ).dropna(subset=[GEN_GENERIC, GEN_SCHEMA_ENUMERATED])
    piv["delta"] = piv[GEN_GENERIC] - piv[GEN_SCHEMA_ENUMERATED]

    # per dataset: mean delta + SEM (in percentage points)
    per_ds = {}
    for ds in ORDER:
        sub = piv.xs(ds, level="eval_dataset_name")["delta"]
        per_ds[ds] = (100 * sub.mean(), 100 * sub.std(ddof=1) / np.sqrt(len(sub)))

    flat = piv.reset_index()
    sub1b = flat[flat.model_id.isin(["SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"])]["delta"]
    t3b = flat[flat.model_id.isin(["SmolLM3-3B", "Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"])]["delta"]
    by_scale = {
        "Sub-billion\n(360M, 0.5B)": (100 * sub1b.mean(), 100 * sub1b.std(ddof=1) / np.sqrt(len(sub1b))),
        "3B": (100 * t3b.mean(), 100 * t3b.std(ddof=1) / np.sqrt(len(t3b))),
    }
    overall = 100 * piv["delta"].mean()
    return per_ds, by_scale, overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default="runs/evaluation/per_dataset_metrics.csv")
    ap.add_argument("--out", default="Compact Relex Journal Paper/figures/constrained_vs_open.pdf")
    args = ap.parse_args()

    per_ds, by_scale, overall = load_deltas(Path(args.metrics))

    POS, NEG, ACCENT = "#2C7FB8", "#D7301F", "#444444"

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [2.5, 1]}
    )

    # ---- Panel (a): per-dataset delta, sorted descending, horizontal bars ----
    items = sorted(per_ds.items(), key=lambda kv: kv[1][0], reverse=True)
    labels = [DISPLAY[k] for k, _ in items]
    vals = [v[0] for _, v in items]
    errs = [v[1] for _, v in items]
    y = np.arange(len(labels))[::-1]  # largest at top
    colors = [POS if v >= 0 else NEG for v in vals]
    axA.barh(y, vals, xerr=errs, color=colors, alpha=0.9,
             error_kw=dict(ecolor=ACCENT, capsize=2, lw=0.8))
    axA.axvline(overall, ls="--", color=ACCENT, lw=1.2,
                label=f"Overall mean: {overall:+.1f} pp")
    axA.axvline(0, color="black", lw=0.6)
    axA.set_yticks(y)
    axA.set_yticklabels(labels, fontsize=9)
    axA.set_xlabel(r"$\Delta$ positive-class micro-F1 (generic $-$ schema-enum.), pp", fontsize=10)
    axA.set_title("(a) Per-dataset effect", fontsize=11)
    axA.legend(fontsize=8, loc="lower right", frameon=False)
    for yi, v, e in zip(y, vals, errs):
        axA.text(v + (0.4 if v >= 0 else -0.4) + (e if v >= 0 else -e), yi,
                 f"{v:+.1f}", va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
    axA.margins(x=0.18)
    axA.grid(axis="x", alpha=0.25)

    # ---- Panel (b): by model scale ----
    bl = list(by_scale.keys())
    bv = [by_scale[k][0] for k in bl]
    be = [by_scale[k][1] for k in bl]
    x = np.arange(len(bl))
    axB.bar(x, bv, yerr=be, color=["#74A9CF", "#2C7FB8"], alpha=0.9,
            error_kw=dict(ecolor=ACCENT, capsize=3, lw=0.8), width=0.6)
    axB.axhline(overall, ls="--", color=ACCENT, lw=1.2)
    axB.set_xticks(x)
    axB.set_xticklabels(bl, fontsize=9)
    axB.set_ylabel(r"$\Delta$ pos.-class micro-F1 (pp)", fontsize=10)
    axB.set_title("(b) By model scale", fontsize=11)
    for xi, v in zip(x, bv):
        axB.text(xi, v + 0.15, f"{v:+.1f}", ha="center", va="bottom", fontsize=9)
    axB.text(0.97, overall, f"mean {overall:+.1f}", transform=axB.get_yaxis_transform(),
             ha="right", va="bottom", fontsize=7.5, color=ACCENT)
    axB.set_ylim(0, max(bv) * 1.35)
    axB.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure 3 -> {out}")
    print(f"  overall mean = {overall:+.2f} pp; "
          f"sub-billion = {by_scale[list(by_scale)[0]][0]:+.2f}; 3B = {by_scale[list(by_scale)[1]][0]:+.2f}")


if __name__ == "__main__":
    main()

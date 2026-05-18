#!/usr/bin/env python3
"""
Build the full 30×9 per-dataset F1 results matrix for the appendix.

Produces:
  runs/evaluation/tables/
    full_results_matrix.csv        — 30 configs × 9 datasets, micro-F1
    full_results_matrix.tex        — LaTeX table for the appendix

Usage:
    python scripts/build_full_results_table.py --config configs/eval.yaml
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


DATASET_ORDER = [
    "tacred", "semeval2010_task8", "conll04", "nyt11",
    "gids", "re_docred", "rebel",
    "biographical", "pg_fiction",
]

DATASET_SHORT = {
    "tacred": "TACRED",
    "semeval2010_task8": "SemEval",
    "conll04": "CoNLL04",
    "nyt11": "NYT11",
    "gids": "GDS",
    "re_docred": "DocRED",
    "rebel": "REBEL",
    "biographical": "Biogr.",
    "pg_fiction": "PG-Fic.",
}

MODEL_SHORT = {
    "SmolLM2-360M-Instruct": "SmolLM2-360M",
    "Qwen2.5-0.5B-Instruct": "Qwen2.5-0.5B",
    "SmolLM3-3B": "SmolLM3-3B",
    "Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Llama-3.2-3B-Instruct": "Llama-3.2-3B",
}

MODEL_ORDER = [
    "SmolLM2-360M-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "SmolLM3-3B",
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct",
]

TUNED_ORDER = ["re_gentune", "re_littune", "re_mixtune"]
SHOT_ORDER = [0, 2]


def build_matrix(metrics_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build the 30×9 F1 matrix from per_dataset_metrics.csv."""
    matched = metrics_df[
        (metrics_df["gen_type"] == "gen_constrained")
        & (metrics_df["model_shot"] == metrics_df["prompt_shot"])
    ].copy()

    tuned_display = cfg.get("tuning_regime_display", {})

    rows = []
    for model_id in MODEL_ORDER:
        for tuned_ds in TUNED_ORDER:
            for shot in SHOT_ORDER:
                config_data = matched[
                    (matched["model_id"] == model_id)
                    & (matched["tuned_dataset_name"] == tuned_ds)
                    & (matched["model_shot"] == shot)
                ]

                row = {
                    "model": MODEL_SHORT.get(model_id, model_id),
                    "regime": tuned_display.get(tuned_ds, tuned_ds),
                    "shot": f"{shot}s",
                    "config": f"{MODEL_SHORT.get(model_id, model_id)} {tuned_display.get(tuned_ds, tuned_ds)} {shot}s",
                }

                for ds in DATASET_ORDER:
                    ds_row = config_data[config_data["eval_dataset_name"] == ds]
                    if not ds_row.empty:
                        row[ds] = round(ds_row.iloc[0]["micro_f1"], 4)
                    else:
                        row[ds] = None

                # Compute domain averages
                gen_vals = [row[ds] for ds in DATASET_ORDER[:7] if row.get(ds) is not None]
                lit_vals = [row[ds] for ds in DATASET_ORDER[7:] if row.get(ds) is not None]
                all_vals = gen_vals + lit_vals

                row["general_avg"] = round(np.mean(gen_vals), 4) if gen_vals else None
                row["literary_avg"] = round(np.mean(lit_vals), 4) if lit_vals else None
                row["overall_avg"] = round(np.mean(all_vals), 4) if all_vals else None

                rows.append(row)

    return pd.DataFrame(rows)


def to_latex(matrix: pd.DataFrame, cfg: dict) -> str:
    """Generate LaTeX table from the matrix."""
    tuned_display = cfg.get("tuning_regime_display", {})

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.1}")

    # 13 columns: config + 9 datasets + 3 averages
    col_spec = "l" + "c" * 12
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    ds_headers = " & ".join(DATASET_SHORT[ds] for ds in DATASET_ORDER)
    lines.append(
        r"\textbf{Configuration} & "
        + ds_headers
        + r" & \textbf{Gen.} & \textbf{Lit.} & \textbf{All} \\"
    )
    lines.append(r"\midrule")

    # Find best per-column for bolding
    best_per_col = {}
    for col in DATASET_ORDER + ["general_avg", "literary_avg", "overall_avg"]:
        vals = matrix[col].dropna()
        if not vals.empty:
            best_per_col[col] = vals.max()

    # Data rows, grouped by model
    prev_model = None
    for _, row in matrix.iterrows():
        if row["model"] != prev_model and prev_model is not None:
            lines.append(r"\midrule")
        prev_model = row["model"]

        config_str = f"{row['model']} {row['regime']} {row['shot']}"
        cells = [config_str]

        for col in DATASET_ORDER + ["general_avg", "literary_avg", "overall_avg"]:
            val = row[col]
            if val is None or pd.isna(val):
                cells.append("--")
            else:
                s = f"{val:.4f}"
                if col in best_per_col and abs(val - best_per_col[col]) < 1e-6:
                    s = r"\textbf{" + s + "}"
                cells.append(s)

        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Full per-dataset micro-F1 results for all 30 SLM configurations "
        r"under constrained generation with matched prompt shots. "
        r"Gen.\ = general-domain average (7 datasets), "
        r"Lit.\ = literary average (2 datasets), "
        r"All = overall average (9 datasets). "
        r"Best values per column are shown in \textbf{bold}.}"
    )
    lines.append(r"\label{tab:full_results_matrix}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/eval.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    metrics_path = REPO_ROOT / "runs" / "evaluation" / "per_dataset_metrics.csv"
    if not metrics_path.exists():
        print(f"✗ {metrics_path} not found. Run scripts/run_evaluation.py first.")
        sys.exit(1)

    metrics_df = pd.read_csv(metrics_path)
    output_dir = REPO_ROOT / "runs" / "evaluation" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Building full results matrix...")
    matrix = build_matrix(metrics_df, cfg)

    # Save CSV
    csv_path = output_dir / "full_results_matrix.csv"
    matrix.to_csv(csv_path, index=False)
    print(f"  → CSV: {csv_path}")

    # Save LaTeX
    latex = to_latex(matrix, cfg)
    tex_path = output_dir / "full_results_matrix.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"  → LaTeX: {tex_path}")

    # Print summary
    print(f"\nMatrix shape: {matrix.shape[0]} configs × {len(DATASET_ORDER)} datasets")
    print(f"\nOverall Avg F1 range: {matrix['overall_avg'].min():.4f} — {matrix['overall_avg'].max():.4f}")

    # Show top 5
    top5 = matrix.nlargest(5, "overall_avg")[["config", "overall_avg", "general_avg", "literary_avg"]]
    print(f"\nTop 5 configs by Overall Avg F1:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()

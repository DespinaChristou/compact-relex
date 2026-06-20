#!/usr/bin/env python3
"""
Disentangle model scale from model family for the scaling discussion.

Parameter count is confounded with family, tokenizer, pretraining data, and
instruction tuning. With only two broad scales (sub-billion, 3B) and no
sub-billion Llama, the only WITHIN-FAMILY scale contrasts are:
  * Qwen2.5  : 0.5B  -> 3B   (same generation; clean)
  * SmolLM   : SmolLM2-360M -> SmolLM3-3B  (same lab, but crosses v2 -> v3)
Llama-3.2-3B has no sub-billion counterpart, so it cannot inform a within-family
scale slope.

We report, on the primary schema-enumerated / matched-shot subset (excluding the
two pre-specified 0-shot anomalies, consistent with Figure 2):
  (1) within-family Delta F1 (3B - small), paired by (regime, shot, dataset),
      overall and per regime, with dataset-clustered bootstrap CIs;
  (2) a family-controlled regression of micro-F1 on log10(params): the NAIVE
      cross-family slope vs. the FAMILY-CONTROLLED (family fixed-effects) slope,
      to show how much of the apparent "scaling" is between-family confound.

Run:  python scripts/analyze_scale_family.py --config configs/eval.yaml
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
from src.eval import GEN_SCHEMA_ENUMERATED  # primary gen type

FAMILY = {
    "SmolLM2-360M-Instruct": "SmolLM",
    "SmolLM3-3B": "SmolLM",
    "Qwen2.5-0.5B-Instruct": "Qwen",
    "Qwen2.5-3B-Instruct": "Qwen",
    "Llama-3.2-3B-Instruct": "Llama",
}
SMALL = {"SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"}  # sub-billion
ANOMALIES = {  # (model_id, regime, shot) -- pre-specified exclusions (Section 4)
    ("SmolLM3-3B", "re_mixtune", 0),
    ("Qwen2.5-3B-Instruct", "re_gentune", 0),
}
# within-family scale pairs: family -> (small_model, big_model)
PAIRS = {"Qwen": ("Qwen2.5-0.5B-Instruct", "Qwen2.5-3B-Instruct"),
         "SmolLM": ("SmolLM2-360M-Instruct", "SmolLM3-3B")}
REGIME = {"re_gentune": "GenTune", "re_littune": "LitTune", "re_mixtune": "MixTune"}


def load(cfg):
    df = pd.read_csv(REPO_ROOT / cfg["paths"]["output_dir"] / "per_dataset_metrics.csv")
    df = df[df.gen_type == GEN_SCHEMA_ENUMERATED].copy()
    df = df[df.model_shot == df.prompt_shot].copy()             # matched shots
    df = df[~df.apply(lambda r: (r.model_id, r.tuned_dataset_name, r.model_shot) in ANOMALIES, axis=1)]
    df["family"] = df.model_id.map(FAMILY)
    df["params_b"] = df.model_id.map(cfg["model_metadata"])  # placeholder, fixed below
    meta = cfg["model_metadata"]
    df["params_b"] = df.model_id.map(lambda m: meta[m]["params_b"])
    df["logp"] = np.log10(df["params_b"])
    return df


def within_family_delta(df, family, rng, n_boot=10000):
    """Paired 3B - small Delta micro-F1 by (regime, shot, dataset); dataset-clustered bootstrap CI."""
    small, big = PAIRS[family]
    s = df[df.model_id == small][["tuned_dataset_name", "model_shot", "eval_dataset_name", "micro_f1"]]
    b = df[df.model_id == big][["tuned_dataset_name", "model_shot", "eval_dataset_name", "micro_f1"]]
    m = s.merge(b, on=["tuned_dataset_name", "model_shot", "eval_dataset_name"], suffixes=("_s", "_b"))
    m["delta"] = m.micro_f1_b - m.micro_f1_s
    overall = m.delta.mean()
    # dataset-clustered bootstrap
    ds = m.eval_dataset_name.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ds, size=len(ds), replace=True)
        boots.append(pd.concat([m[m.eval_dataset_name == d] for d in pick]).delta.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    per_regime = {REGIME[r]: g.delta.mean() for r, g in m.groupby("tuned_dataset_name")}
    return dict(family=family, small=small, big=big, n_pairs=len(m),
                delta=overall, ci=(lo, hi), per_regime=per_regime)


def ols_slope(df, controls):
    """Slope on logp in micro_f1 ~ logp + sum(C(c)); returns the logp coefficient."""
    X = [np.ones(len(df)), df["logp"].values]
    names = ["const", "logp"]
    for c in controls:
        d = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float)
        for col in d.columns:
            X.append(d[col].values); names.append(col)
    Xm = np.column_stack(X)
    beta, *_ = np.linalg.lstsq(Xm, df["micro_f1"].values, rcond=None)
    return beta[names.index("logp")]


def slope_ci(df, controls, rng, n_boot=10000):
    ds = df.eval_dataset_name.unique()
    base = ols_slope(df, controls)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ds, size=len(ds), replace=True)
        bs = pd.concat([df[df.eval_dataset_name == d] for d in pick])
        try:
            boots.append(ols_slope(bs, controls))
        except Exception:
            pass
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(REPO_ROOT / args.config))
    cfg["paths"]["output_dir"] = Path(cfg["paths"]["output_dir"])
    df = load(cfg)
    rng = np.random.default_rng(args.seed)

    print("=" * 72)
    print("WITHIN-FAMILY SCALE CONTRASTS  (3B - small, positive-class micro-F1)")
    print("  paired by (regime, shot, dataset); dataset-clustered bootstrap 95% CI")
    print("=" * 72)
    for fam in ["Qwen", "SmolLM"]:
        r = within_family_delta(df, fam, rng, args.iters)
        pr = "  ".join(f"{k} {v:+.3f}" for k, v in r["per_regime"].items())
        print(f"\n{fam}: {r['small']} -> {r['big']}  (n={r['n_pairs']} paired cells)")
        print(f"  overall  Delta = {r['delta']:+.3f}   95% CI [{r['ci'][0]:+.3f}, {r['ci'][1]:+.3f}]")
        print(f"  by regime: {pr}")

    print("\n" + "=" * 72)
    print("FAMILY-CONTROLLED REGRESSION:  micro_f1 ~ log10(params) + controls")
    print("  slope = Delta F1 per 10x parameters; dataset-clustered bootstrap 95% CI")
    print("=" * 72)
    naive = slope_ci(df, ["tuned_dataset_name", "model_shot"], rng, args.iters)
    ctrl = slope_ci(df, ["family", "tuned_dataset_name", "model_shot"], rng, args.iters)
    print(f"\n  NAIVE (no family control)     slope = {naive[0]:+.3f}  95% CI [{naive[1]:+.3f}, {naive[2]:+.3f}]")
    print(f"  FAMILY-CONTROLLED (family FE) slope = {ctrl[0]:+.3f}  95% CI [{ctrl[1]:+.3f}, {ctrl[2]:+.3f}]")
    print(f"\n  -> within-family scale effect is {ctrl[0]/naive[0]*100:.0f}% of the naive cross-family slope"
          if naive[0] else "")

    # cross-family 'best 3B vs best sub-billion' numbers currently in the text (for context)
    print("\n" + "=" * 72)
    print("CONTEXT: best-in-class overall avg F1 by scale group and regime (cross-family)")
    print("=" * 72)
    for reg, rn in REGIME.items():
        sub = df[df.tuned_dataset_name == reg]
        for shot in [0, 2]:
            g = sub[sub.model_shot == shot]
            if g.empty:
                continue
            agg = g.groupby("model_id").micro_f1.mean()
            small_best = agg[[m for m in agg.index if m in SMALL]].max() if any(m in SMALL for m in agg.index) else np.nan
            big_best = agg[[m for m in agg.index if m not in SMALL]].max() if any(m not in SMALL for m in agg.index) else np.nan
            print(f"  {rn:8s} {shot}-shot: best sub-billion {small_best:.3f} | best 3B {big_best:.3f} | gap {big_best-small_best:+.3f}")


if __name__ == "__main__":
    main()

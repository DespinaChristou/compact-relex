#!/usr/bin/env python3
"""
Compute exact bootstrap CIs and p-values for Table 11 (pairwise significance)
under POSITIVE-CLASS micro-F1 (no-relation class excluded).

Run locally (the generation CSVs are multi-GB and exceed the cowork sandbox's
45s limit). Reads the matched-shot, constrained generations and reports, for each
of the five comparisons, the dataset-macro-averaged Delta(positive-class F1),
its 95% percentile bootstrap CI, and a two-sided bootstrap p-value.

Usage:
    python scripts/compute_table11_cis.py            # 10,000 iterations
    python scripts/compute_table11_cis.py --iters 10000 --seed 42
"""
from __future__ import annotations
import argparse, glob, os, re
import numpy as np
import pandas as pd

GEN = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids", "re_docred", "rebel"]
LIT = ["biographical", "pg_fiction"]
CATCH = {"", "none", "other", "no_relation", "na"}

def isc(x: str) -> bool:
    return str(x).strip().lower() in CATCH

def norm(x: str) -> str:
    s = str(x).strip().split("\n")[0].strip().lower()
    return re.sub(r"\s+", " ", s)

# (model_id, tuned_dataset_name, shot) for every config referenced in Table 11
NEEDED = {
    ("SmolLM2-360M-Instruct", "re_mixtune", 2),
    ("SmolLM2-360M-Instruct", "re_mixtune", 0),
    ("Qwen2.5-0.5B-Instruct", "re_gentune", 2),
    ("Llama-3.2-3B-Instruct", "re_mixtune", 2),
    ("Llama-3.2-3B-Instruct", "re_gentune", 2),
    ("Llama-3.2-3B-Instruct", "re_littune", 0),
}

# Table 11 rows: (label, domain-datasets, configA, configB)
COMPARISONS = [
    ("SmolLM2-360M Mix 2s vs 0s", GEN + LIT,
     ("SmolLM2-360M-Instruct", "re_mixtune", 2), ("SmolLM2-360M-Instruct", "re_mixtune", 0)),
    ("Qwen-0.5B Gen 2s vs Llama-3B Mix 2s", GEN,
     ("Qwen2.5-0.5B-Instruct", "re_gentune", 2), ("Llama-3.2-3B-Instruct", "re_mixtune", 2)),
    ("SmolLM2-360M vs Llama-3B Mix 2s", GEN + LIT,
     ("SmolLM2-360M-Instruct", "re_mixtune", 2), ("Llama-3.2-3B-Instruct", "re_mixtune", 2)),
    ("Llama-3B MixTune vs GenTune 2s", GEN,
     ("Llama-3.2-3B-Instruct", "re_mixtune", 2), ("Llama-3.2-3B-Instruct", "re_gentune", 2)),
    ("Llama-3B MixTune 2s vs LitTune 0s", LIT,
     ("Llama-3.2-3B-Instruct", "re_mixtune", 2), ("Llama-3.2-3B-Instruct", "re_littune", 0)),
]

USECOLS = ["relation", "generated_relation", "model_id",
           "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"]

def load_indicators(gen_root: str):
    """For each (config, dataset) in NEEDED, build boolean tp/fp/fn arrays for
    positive-class F1, aligned by row order (the SLM eval order is identical
    across configs within a dataset)."""
    arr = {}  # (model, regime, shot, ds) -> (tp, fp, fn) np.bool arrays
    for ds in GEN + LIT:
        path = os.path.join(gen_root, ds, "generations.csv")
        if not os.path.exists(path):
            shards = sorted(glob.glob(os.path.join(gen_root, ds, "generations_shard_*.csv")))
            if not shards:
                print(f"  WARNING: no generations for {ds}"); continue
            frames = [pd.read_csv(s, usecols=USECOLS, dtype=str, keep_default_na=False) for s in shards]
            d = pd.concat(frames, ignore_index=True)
        else:
            d = pd.read_csv(path, usecols=USECOLS, dtype=str, keep_default_na=False)
        d = d[(d.gen_type == "gen_constrained") & (d.model_shot == d.prompt_shot)]
        for (mid, reg, ms), g in d.groupby(["model_id", "tuned_dataset_name", "model_shot"]):
            try:
                ms = int(ms)
            except ValueError:
                continue
            if (mid, reg, ms) not in NEEDED:
                continue
            gold = g["relation"].map(norm).to_numpy()
            pred = g["generated_relation"].map(norm).to_numpy()
            gpos = np.array([not isc(x) for x in gold])
            ppos = np.array([not isc(x) for x in pred])
            corr = gold == pred
            arr[(mid, reg, ms, ds)] = (gpos & corr, ppos & ~corr, gpos & ~corr)
    return arr

def f1(tp, fp, fn, idx):
    TP, FP, FN = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0

def bootstrap(arr, dss, A, B, n_iter, rng):
    per = {}
    for ds in dss:
        ka, kb = (A[0], A[1], A[2], ds), (B[0], B[1], B[2], ds)
        if ka in arr and kb in arr:
            per[ds] = (arr[ka], arr[kb])
    if not per:
        return None
    base = np.mean([f1(*per[d][0], np.arange(len(per[d][0][0])))
                    - f1(*per[d][1], np.arange(len(per[d][1][0]))) for d in per])
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        vals = []
        for d in per:
            Aa, Bb = per[d]
            m = len(Aa[0]); idx = rng.integers(0, m, m)
            vals.append(f1(*Aa, idx) - f1(*Bb, idx))
        diffs[i] = np.mean(vals)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, lo, hi, p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_root", default="runs/generations")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("Loading generations (this reads the large CSVs once)...")
    arr = load_indicators(args.gen_root)
    rng = np.random.default_rng(args.seed)
    print(f"\n{'Comparison':40s} {'Domain':8s} {'dF1':>8} {'95% CI':>20} {'p':>10}")
    for label, dss, A, B in COMPARISONS:
        dom = "All" if dss == GEN + LIT else ("General" if dss == GEN else "Literary")
        r = bootstrap(arr, dss, A, B, args.iters, rng)
        if r is None:
            print(f"{label:40s} {dom:8s}  (missing data)"); continue
        base, lo, hi, p = r
        ps = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{label:40s} {dom:8s} {base:+.4f} [{lo:+.3f}, {hi:+.3f}] {ps:>10}")

if __name__ == "__main__":
    main()

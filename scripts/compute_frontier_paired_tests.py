#!/usr/bin/env python3
"""
Paired SLM-vs-frontier significance tests for the headline comparisons.

Reviewers asked for paired intervals/tests on the SLM-versus-frontier comparison
(the paper's headline), which an earlier draft omitted claiming per-example
alignment was unavailable for API-evaluated systems. It IS available: every
frontier API call is logged with prompt_hash = md5(prompt_0_shot)[:12]
(scripts/run_frontier_generations.py), and the SLM generation rows carry
prompt_0_shot, so the two are joinable per example. This script reconstructs that
alignment and runs a paired, dataset-clustered bootstrap on positive-class
micro-F1 (the same procedure as scripts/compute_table11_cis.py), aligning each
tuned SLM to each frontier system example-by-example via the prompt hash.

Run locally (reads the multi-GB generation CSVs):
    python scripts/compute_frontier_paired_tests.py --iters 10000 --seed 42

Outputs, for each headline comparison: the dataset-macro Delta(positive-class F1)
= F1(SLM) - F1(frontier), its 95% percentile CI, a two-sided bootstrap p-value,
and the per-example join hit rate.
"""
from __future__ import annotations
import argparse
import hashlib
import re

import numpy as np
import pandas as pd

CATCH = {"", "none", "other", "no_relation", "na", "nan", "null"}
GEN = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids", "re_docred", "rebel"]
LIT = ["biographical", "pg_fiction"]

# (label, datasets, (slm_model_id, regime, shot), frontier_model_id)
COMPARISONS = [
    ("Llama-3.2-3B GenTune 2s vs GPT-5.4 (General)", GEN, ("Llama-3.2-3B-Instruct", "re_gentune", 2), "GPT-5.4"),
    ("Llama-3.2-3B GenTune 2s vs Claude (General)", GEN, ("Llama-3.2-3B-Instruct", "re_gentune", 2), "Claude Sonnet 4.6"),
    ("Llama-3.2-3B GenTune 0s vs GPT-5.4 (General, demo-matched)", GEN, ("Llama-3.2-3B-Instruct", "re_gentune", 0), "GPT-5.4"),
    ("Llama-3.2-3B GenTune 0s vs Claude (General, demo-matched)", GEN, ("Llama-3.2-3B-Instruct", "re_gentune", 0), "Claude Sonnet 4.6"),
    ("SmolLM3-3B LitTune 0s vs GPT-5.4 (Literary)", LIT, ("SmolLM3-3B", "re_littune", 0), "GPT-5.4"),
    ("SmolLM3-3B LitTune 0s vs Claude (Literary)", LIT, ("SmolLM3-3B", "re_littune", 0), "Claude Sonnet 4.6"),
]

SLM_GEN = "runs/generations/{ds}/generations.csv"
FRONTIER = "runs/frontier_generations/all_frontier_generations.csv"


def norm(x: str) -> str:
    s = str(x).strip().split("\n")[0].strip().lower()
    return re.sub(r"\s+", " ", s)


def indicators(gold, pred, ph, pre):
    """Positive-class tp/fp/fn boolean arrays keyed by prompt hash."""
    g = gold.map(norm); p = pred.map(norm)
    gpos = ~g.isin(CATCH); ppos = ~p.isin(CATCH); corr = (g == p).values
    return pd.DataFrame({"ph": ph.values,
                         f"{pre}tp": (gpos & corr).values,
                         f"{pre}fp": (ppos.values & ~corr),
                         f"{pre}fn": (gpos.values & ~corr)})


def load_frontier():
    fr = pd.read_csv(FRONTIER, encoding="latin-1",
                     usecols=["eval_dataset_name", "prompt_hash", "relation",
                              "gen_type", "model_id", "generated_relation"], dtype=str)
    return fr[fr.gen_type == "gen_constrained"]


def slm_indicators(ds, mid, reg, shot):
    rows = []
    for ch in pd.read_csv(SLM_GEN.format(ds=ds), encoding="latin-1",
                          usecols=["prompt_0_shot", "relation", "generated_relation", "model_id",
                                   "tuned_dataset_name", "model_shot", "prompt_shot", "gen_type"],
                          dtype=str, on_bad_lines="skip", chunksize=400000):
        ch = ch[(ch.model_id == mid) & (ch.tuned_dataset_name == reg) & (ch.model_shot == str(shot))
                & (ch.prompt_shot == str(shot)) & (ch.gen_type == "gen_constrained")]
        if len(ch):
            rows.append(ch)
    d = pd.concat(rows)
    ph = d.prompt_0_shot.map(lambda p: hashlib.md5(str(p).encode()).hexdigest()[:12])
    return indicators(d.relation, d.generated_relation, ph, "s")


def f1(tp, fp, fn, idx):
    TP, FP, FN = tp[idx].sum(), fp[idx].sum(), fn[idx].sum()
    P = TP / (TP + FP) if TP + FP else 0.0
    R = TP / (TP + FN) if TP + FN else 0.0
    return 2 * P * R / (P + R) if P + R else 0.0


def paired(dss, cfg, frm, fr, n_iter, rng):
    per, hits = {}, []
    for ds in dss:
        s = slm_indicators(ds, *cfg).set_index("ph")
        f = indicators(fr[(fr.model_id == frm) & (fr.eval_dataset_name == ds)].relation,
                       fr[(fr.model_id == frm) & (fr.eval_dataset_name == ds)].generated_relation,
                       fr[(fr.model_id == frm) & (fr.eval_dataset_name == ds)].prompt_hash, "f").set_index("ph")
        m = s.join(f, how="inner").dropna()
        hits.append(len(m) / max(len(f), 1))
        per[ds] = tuple(m[c].values.astype(float) for c in ["stp", "sfp", "sfn", "ftp", "ffp", "ffn"])
    full = lambda ds, k: f1(*per[ds][k:k + 3], np.arange(len(per[ds][0])))
    base = np.mean([full(ds, 0) - full(ds, 3) for ds in per])
    sF = np.mean([full(ds, 0) for ds in per]); fF = np.mean([full(ds, 3) for ds in per])
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        v = []
        for ds in per:
            a = per[ds]; n = len(a[0]); idx = rng.integers(0, n, n)
            v.append(f1(*a[:3], idx) - f1(*a[3:], idx))
        diffs[i] = np.mean(v)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return base, lo, hi, p, sF, fF, float(np.mean(hits))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    fr = load_frontier()
    rng = np.random.default_rng(args.seed)
    print(f"{'Comparison':58s} {'SLM':>6} {'Front':>6} {'dF1':>7} {'95% CI':>20} {'p':>8} {'join':>6}")
    for label, dss, cfg, frm in COMPARISONS:
        base, lo, hi, p, sF, fF, hit = paired(dss, cfg, frm, fr, args.iters, rng)
        ps = "<0.001" if p < 0.001 else f"{p:.3f}"
        print(f"{label:58s} {sF:6.3f} {fF:6.3f} {base:+7.3f} [{lo:+.3f},{hi:+.3f}] {ps:>8} {100*hit:5.1f}%")


if __name__ == "__main__":
    main()

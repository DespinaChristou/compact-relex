#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build the interactive findings dashboard (dashboard/index.html).

A single self-contained HTML page (no external JS/CSS dependencies) that presents
the paper's headline results, the full 30-configuration x 9-benchmark matrix, the
frontier / encoder / DAPT comparisons, efficiency trade-offs, and statistical
significance. Reads the current evaluation tables under runs/evaluation and injects
the authoritative full-test-set frontier / encoder / DAPT numbers reported in the
paper (the stale runs/evaluation/tables/frontier_*.csv are an earlier subsampled
run and are intentionally NOT used here).

Run from the repo root (matching the other scripts/build_*.py tools):

    python scripts/build_dashboard.py

Output: dashboard/index.html
"""
import csv
import json
import os
import statistics as _st
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "runs", "evaluation")
TABLES = os.path.join(EVAL, "tables")
STAT = os.path.join(EVAL, "statistical")
OUT_DIR = os.path.join(ROOT, "dashboard")
OUT = os.path.join(OUT_DIR, "index.html")

# The HTML/CSS/JS template lives next to this script.
import sys
sys.path.insert(0, HERE)
from _dashboard_template import TEMPLATE


def rd(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fnum(x):
    if x is None or x == "" or x == "--":
        return None
    try:
        return round(float(x), 4)
    except ValueError:
        return None


# ----------------------------------------------------------------------------
# Model metadata
# ----------------------------------------------------------------------------
MODEL_META = {
    "SmolLM2-360M":   {"family": "SmolLM",  "params": 0.36, "scale": "sub-billion", "disp": "SmolLM2-360M"},
    "Qwen2.5-0.5B":   {"family": "Qwen2.5", "params": 0.50, "scale": "sub-billion", "disp": "Qwen2.5-0.5B"},
    "SmolLM3-3B":     {"family": "SmolLM",  "params": 3.00, "scale": "3B",          "disp": "SmolLM3-3B"},
    "Qwen2.5-3B":     {"family": "Qwen2.5", "params": 3.00, "scale": "3B",          "disp": "Qwen2.5-3B"},
    "Llama-3.2-3B":   {"family": "Llama",   "params": 3.20, "scale": "3B",          "disp": "Llama-3.2-3B"},
}

GEN_DS = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids", "re_docred", "rebel"]
LIT_DS = ["biographical", "pg_fiction"]
DS_LABEL = {
    "tacred": "TACRED", "semeval2010_task8": "SemEval", "conll04": "CoNLL04",
    "nyt11": "NYT11", "gids": "GIDS", "re_docred": "Re-DocRED", "rebel": "REBEL",
    "biographical": "Biographical", "pg_fiction": "PG-Fiction",
}

# ----------------------------------------------------------------------------
# 1. Full 30-config x 9-dataset matrix (current data)
# ----------------------------------------------------------------------------
matrix = []
for r in rd(os.path.join(TABLES, "full_results_matrix.csv")):
    m = r["model"]
    meta = MODEL_META[m]
    cells = {ds: fnum(r.get(ds)) for ds in GEN_DS + LIT_DS}
    flag = None
    if m == "SmolLM3-3B" and r["regime"] == "MixTune" and r["shot"] == "0s":
        flag = "think"   # emits <think> tokens -> 0 under default protocol (rescues to 0.18)
    if m == "Qwen2.5-3B" and r["regime"] == "GenTune" and r["shot"] == "0s":
        flag = "schema"  # wrong-schema labels without demonstrations
    matrix.append({
        "model": meta["disp"], "family": meta["family"], "params": meta["params"],
        "scale": meta["scale"], "regime": r["regime"], "shot": r["shot"],
        "cells": cells,
        "gen": fnum(r.get("general_avg")),
        "lit": fnum(r.get("literary_avg")),
        "overall": fnum(r.get("overall_avg")),
        "flag": flag,
    })

# ----------------------------------------------------------------------------
# 2. Authoritative frontier / encoder numbers (paper, full test sets)
#    NOTE: the runs/evaluation/tables/frontier_*.csv files are an earlier
#    SUBSAMPLED evaluation and give higher numbers; the paper's headline uses
#    the full-test-set figures encoded below.
# ----------------------------------------------------------------------------
frontier_general = {
    "GPT-5.4":            {"tacred":0.531,"semeval2010_task8":0.710,"conll04":0.926,"nyt11":0.649,"gids":0.791,"re_docred":0.559,"rebel":0.684,"avg":0.693},
    "Claude Sonnet 4.6":  {"tacred":0.445,"semeval2010_task8":0.709,"conll04":0.919,"nyt11":0.692,"gids":0.799,"re_docred":0.523,"rebel":0.547,"avg":0.662},
}
frontier_literary = {
    "GPT-5.4":            {"biographical":0.832,"pg_fiction":0.324,"avg":0.578,"macro":0.50,"valid":0.867},
    "Claude Sonnet 4.6":  {"biographical":0.725,"pg_fiction":0.334,"avg":0.530,"macro":0.41,"valid":0.903},
}
best_slm = {"tacred":0.711,"semeval2010_task8":0.886,"conll04":0.995,"nyt11":0.828,"gids":0.883,
            "re_docred":0.743,"rebel":0.921,"biographical":0.917,"pg_fiction":0.760,
            "gen_avg":0.853,"lit_avg":0.838}
best_slm_cfg = {"tacred":"SmolLM3-3B GenTune 2s","semeval2010_task8":"Qwen2.5-3B GenTune 2s",
                "conll04":"several (>=0.995)","nyt11":"Llama-3.2-3B GenTune 2s","gids":"Qwen2.5-0.5B GenTune 2s",
                "re_docred":"Llama-3.2-3B GenTune 2s","rebel":"Qwen2.5-3B GenTune 2s",
                "biographical":"SmolLM3-3B LitTune 0s","pg_fiction":"SmolLM3-3B LitTune 2s"}

encoder = [  # paper Table (RoBERTa entity-marker baseline), positive-class micro-F1
    {"ds":"tacred","rob_base":0.645,"rob_large":0.653,"best_slm":0.711,"gpt":0.531,"claude":0.445},
    {"ds":"semeval2010_task8","rob_base":0.870,"rob_large":0.895,"best_slm":0.886,"gpt":0.710,"claude":0.709},
    {"ds":"conll04","rob_base":1.000,"rob_large":1.000,"best_slm":0.995,"gpt":0.926,"claude":0.919},
    {"ds":"nyt11","rob_base":0.817,"rob_large":0.699,"best_slm":0.828,"gpt":0.649,"claude":0.692},
    {"ds":"gids","rob_base":0.850,"rob_large":0.856,"best_slm":0.883,"gpt":0.791,"claude":0.799},
    {"ds":"re_docred","rob_base":0.713,"rob_large":0.682,"best_slm":0.743,"gpt":0.559,"claude":0.523},
    {"ds":"rebel","rob_base":0.888,"rob_large":0.917,"best_slm":0.921,"gpt":0.684,"claude":0.547},
    {"ds":"biographical","rob_base":0.906,"rob_large":None,"best_slm":0.917,"gpt":0.832,"claude":0.725},
    {"ds":"pg_fiction","rob_base":0.686,"rob_large":None,"best_slm":0.760,"gpt":0.324,"claude":0.334},
]
encoder_avg = {"rob_base_gen":0.826,"rob_large_gen":0.814,"best_slm_gen":0.853,"gpt_gen":0.693,"claude_gen":0.662,
               "rob_base_lit":0.796,"best_slm_lit":0.838,"gpt_lit":0.578,"claude_lit":0.530}

dapt = [
    {"model":"Llama-3.2-3B LitTune 0s","dapt":False,"bio":0.912,"pg":0.740,"lit":0.826,"delta":None},
    {"model":"+ LitBank DAPT, LitTune 0s","dapt":True,"bio":0.912,"pg":0.742,"lit":0.827,"delta":0.001},
    {"model":"Llama-3.2-3B MixTune 0s","dapt":False,"bio":0.900,"pg":0.716,"lit":0.808,"delta":None},
    {"model":"+ LitBank DAPT, MixTune 0s","dapt":True,"bio":0.901,"pg":0.716,"lit":0.809,"delta":0.001},
]

# Efficiency (paper Table 10, curated representative rows)
efficiency = [
    {"cfg":"SmolLM2-360M MixTune 2s","params":0.36,"size":"~0.3 GB","gpu":18,"cpu":120,"lit":0.760,"avg":0.750,"f1b":2.08,"scale":"sub-billion"},
    {"cfg":"Qwen2.5-0.5B GenTune 2s","params":0.50,"size":"~0.5 GB","gpu":22,"cpu":180,"lit":None,"avg":0.828,"f1b":1.66,"scale":"sub-billion"},
    {"cfg":"Qwen2.5-0.5B MixTune 2s","params":0.50,"size":"~0.5 GB","gpu":22,"cpu":180,"lit":0.773,"avg":0.801,"f1b":1.60,"scale":"sub-billion"},
    {"cfg":"SmolLM3-3B LitTune 0s","params":3.00,"size":"~2.0 GB","gpu":45,"cpu":850,"lit":0.833,"avg":0.833,"f1b":0.28,"scale":"3B"},
    {"cfg":"Llama-3.2-3B MixTune 2s","params":3.20,"size":"~2.2 GB","gpu":45,"cpu":900,"lit":0.825,"avg":0.826,"f1b":0.28,"scale":"3B"},
    {"cfg":"Llama-3.2-3B GenTune 2s","params":3.20,"size":"~2.2 GB","gpu":45,"cpu":900,"lit":None,"avg":0.844,"f1b":0.28,"scale":"3B"},
]
efficiency_frontier = [
    {"cfg":"GPT-5.4 (0-shot)","params":None,"size":"API","gpu":None,"cpu":None,"lit":0.578,"avg":0.667,"f1b":None,"scale":"frontier"},
    {"cfg":"Claude Sonnet 4.6 (0-shot)","params":None,"size":"API","gpu":None,"cpu":None,"lit":0.530,"avg":0.632,"f1b":None,"scale":"frontier"},
]

# ----------------------------------------------------------------------------
# 3. Prompt-effect deltas (current table5)
# ----------------------------------------------------------------------------
prompt_deltas = []
for r in rd(os.path.join(TABLES, "table5_prompt_delta.csv")):
    m = r["base_model"].replace("-Instruct", "")
    d_over = fnum(r.get("delta_overall_avg_f1"))
    flag = None
    if m == "SmolLM3-3B" and r["tuning_regime"] == "MixTune":
        flag = "think"
    if m == "Qwen2.5-3B" and r["tuning_regime"] == "GenTune":
        flag = "schema"
    prompt_deltas.append({
        "model": MODEL_META[m]["disp"] if m in MODEL_META else m,
        "scale": MODEL_META[m]["scale"] if m in MODEL_META else "3B",
        "regime": r["tuning_regime"], "delta": d_over, "flag": flag,
    })

sub_deltas = [d["delta"] for d in prompt_deltas if d["scale"] == "sub-billion" and d["delta"] is not None]
big_deltas = [d["delta"] for d in prompt_deltas if d["scale"] == "3B" and d["flag"] is None and d["delta"] is not None]
prompt_summary = {
    "sub_mean": round(sum(sub_deltas) / len(sub_deltas), 3),
    "big_mean": round(sum(big_deltas) / len(big_deltas), 3),
}

# ----------------------------------------------------------------------------
# 4. Dataset difficulty (label complexity)
# ----------------------------------------------------------------------------
difficulty = []
for r in rd(os.path.join(STAT, "label_complexity.csv")):
    difficulty.append({
        "ds": r["dataset"], "label": DS_LABEL.get(r["dataset"], r["dataset"]),
        "n_labels": int(r["n_labels"]), "f1": fnum(r["mean_micro_f1"]),
        "min": fnum(r["min_micro_f1"]), "max": fnum(r["max_micro_f1"]),
        "domain": r["domain"],
    })

# ----------------------------------------------------------------------------
# 5. Within-family scaling + significance callouts
# ----------------------------------------------------------------------------
scaling = {
    "Qwen2.5": [
        {"params":0.50,"label":"0.5B","overall":0.8015},
        {"params":3.00,"label":"3B","overall":0.7995},
    ],
    "SmolLM": [
        {"params":0.36,"label":"360M","overall":0.7499},
        {"params":3.00,"label":"3B","overall":0.8329},
    ],
}
scaling_note = {"qwen_delta":0.037, "qwen_ci":"[+0.009, +0.067]", "qwen_gen":-0.004,
                "smol_delta":0.132, "logslope":0.129, "logslope_ci":"[+0.077, +0.196]"}

significance = [
    {"title":"2-shot tuning transforms sub-billion models",
     "detail":"SmolLM2-360M MixTune: +0.161 F1 from 0->2 shot (0.660 -> 0.820).",
     "stat":"p < 0.001", "kind":"pos"},
    {"title":"A sub-billion model matches the best 3B",
     "detail":"Qwen2.5-0.5B GenTune 2s vs Llama-3.2-3B MixTune 2s: +0.007 F1 in favour of the 0.5B model.",
     "stat":"p < 0.001", "kind":"pos"},
    {"title":"One mixed model matches the literary specialist",
     "detail":"Llama MixTune 2s vs LitTune 0s on literary RE: delta = -0.0001.",
     "stat":"not significant", "kind":"neutral"},
    {"title":"Specialists keep a small in-domain edge",
     "detail":"Llama GenTune 2s vs MixTune 2s on general RE: +0.017 for the specialist.",
     "stat":"p < 0.001", "kind":"pos"},
]

# ----------------------------------------------------------------------------
# 5b. Generic (open) vs schema-enumerated (constrained) prompting.
#     delta = gen_open - gen_constrained in percentage points, matched prompt
#     shots, the two decoding-artifact 0-shot configs excluded. Mirrors
#     scripts/build_constrained_vs_open.py (overall ~+3.2 pp, generic wins 8/9).
# ----------------------------------------------------------------------------
_CO_ANOM = {("SmolLM3-3B", "re_mixtune", "0"), ("Qwen2.5-3B-Instruct", "re_gentune", "0")}
_SUB_IDS = {"SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct"}
_B3_IDS = {"SmolLM3-3B", "Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"}

_pair = defaultdict(dict)
for r in rd(os.path.join(EVAL, "per_dataset_metrics.csv")):
    if r["model_shot"] != r["prompt_shot"]:                     # matched shots only
        continue
    if (r["model_id"], r["tuned_dataset_name"], r["model_shot"]) in _CO_ANOM:
        continue
    key = (r["eval_dataset_name"], r["model_id"], r["tuned_dataset_name"], r["model_shot"])
    _pair[key][r["gen_type"]] = float(r["micro_f1"])

_co_raw = []  # (dataset, model_id, delta_pp)
for (ds, mid, _reg, _ms), gt in _pair.items():
    if "gen_open" in gt and "gen_constrained" in gt:
        _co_raw.append((ds, mid, (gt["gen_open"] - gt["gen_constrained"]) * 100.0))

def _sem(v):
    return (_st.stdev(v) / len(v) ** 0.5) if len(v) > 1 else 0.0

co_perdataset = []
for ds in GEN_DS + LIT_DS:
    vals = [d for dd, _m, d in _co_raw if dd == ds]
    co_perdataset.append({
        "ds": ds, "label": DS_LABEL[ds],
        "delta": round(_st.mean(vals), 2), "sem": round(_sem(vals), 2), "n": len(vals),
        "domain": "literary" if ds in LIT_DS else "general",
    })
_co_sub = [d for _d, mm, d in _co_raw if mm in _SUB_IDS]
_co_b3 = [d for _d, mm, d in _co_raw if mm in _B3_IDS]
constrained_open = {
    "perDataset": co_perdataset,
    "overall": round(_st.mean([d for _d, _m, d in _co_raw]), 2),
    "sub": round(_st.mean(_co_sub), 2),
    "big": round(_st.mean(_co_b3), 2),
    "nPos": sum(1 for x in co_perdataset if x["delta"] > 0),
    "n": len(co_perdataset),
    "maxN": max(x["n"] for x in co_perdataset),
}

# ----------------------------------------------------------------------------
# 6. KPIs + headline + findings
# ----------------------------------------------------------------------------
kpis = [
    {"value":"0.83","unit":"General-Avg F1","label":"Best sub-billion model",
     "sub":"Qwen2.5-0.5B (0.5B params) vs 0.69 GPT-5.4 &middot; 0.66 Claude"},
    {"value":"+26&ndash;30","unit":"F1 points","label":"Literary lead over frontier",
     "sub":"Tuned 3B SLMs vs GPT-5.4 / Claude on literary RE"},
    {"value":"30","unit":"configurations","label":"Controlled tuning grid",
     "sub":"5 base models &times; 3 regimes &times; 2 prompt styles"},
    {"value":"2.08","unit":"F1 per billion params","label":"Efficiency champion",
     "sub":"SmolLM2-360M &mdash; ~7&times; any 3B model"},
    {"value":"~18 ms","unit":"per extraction","label":"On a single RTX 4090",
     "sub":"Sub-billion models; ~120 ms CPU-only"},
    {"value":"9","unit":"benchmarks","label":"General + literary",
     "sub":"7 general-domain + 2 literary RE datasets"},
]

headline = {
    "categories": ["General-domain Avg", "Literary Avg"],
    "series": [
        {"name":"Best tuned SLM", "color":"#10b981",
         "values":[best_slm["gen_avg"], best_slm["lit_avg"]]},
        {"name":"RoBERTa (in-domain)", "color":"#6366f1",
         "values":[encoder_avg["rob_base_gen"], encoder_avg["rob_base_lit"]]},
        {"name":"GPT-5.4 (0-shot)", "color":"#f59e0b",
         "values":[frontier_general["GPT-5.4"]["avg"], frontier_literary["GPT-5.4"]["avg"]]},
        {"name":"Claude Sonnet 4.6 (0-shot)", "color":"#ef4444",
         "values":[frontier_general["Claude Sonnet 4.6"]["avg"], frontier_literary["Claude Sonnet 4.6"]["avg"]]},
    ],
}

findings = [
    {"icon":"target","title":"Small beats frontier on general RE",
     "body":"Every well-tuned SLM in the grid, including the 0.5B Qwen2.5, surpasses zero-shot GPT-5.4 (0.69) and Claude Sonnet 4.6 (0.66) on general-domain RE. The best tuned model reaches 0.844.",
     "stat":"0.83 vs 0.69"},
    {"icon":"book","title":"A decisive literary lead","body":"On the two literary benchmarks, tuned 3B SLMs reach 0.833 average F1 versus 0.578 (GPT-5.4) and 0.530 (Claude), a 26-30 point margin that survives de-leaking and holds under macro-F1.",
     "stat":"+26-30 F1"},
    {"icon":"mix","title":"Data composition rivals scale","body":"Domain specialists win in-domain (GenTune 0.772 general, LitTune 0.789 literary), yet a single MixTune model covers both with a domain-balance gap of only 0.010, giving up just ~2 points.",
     "stat":"0.010 gap"},
    {"icon":"prompt","title":"2-shot rescues the smallest models","body":"Sub-billion models gain +0.13 F1 on average from 2-shot prompt-conditioned tuning (significant at p<0.001); at 3B the effect nearly vanishes (+0.005), as larger models absorb the schema from fine-tuning alone.",
     "stat":"+0.13 vs +0.005"},
    {"icon":"scale","title":"Scale buys surprisingly little","body":"In the cleanest same-generation contrast (Qwen2.5 0.5B->3B), a 6x scale-up adds only +0.037 overall F1 and nothing on general-domain RE, so sub-billion models remain strong deployment options.",
     "stat":"+0.037 for 6x"},
    {"icon":"check","title":"It's task adaptation, not decoding","body":"An in-domain RoBERTa encoder also clears both frontier systems on every benchmark, so the SLM advantage stems from task-specific adaptation rather than generative decoding or raw scale. A LitBank DAPT case study adds <=0.001.",
     "stat":"RoBERTa 0.826 > 0.69"},
]

DATA = {
    "meta": {
        "title": "Sub-Billion, Super-Frontier",
        "subtitle": "Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction",
        "authors": "Despina Christou &middot; Grigorios Tsoumakas",
        "affil": "School of Informatics, Aristotle University of Thessaloniki &middot; Archimedes, Athena Research Center",
        "arxiv": "https://arxiv.org/abs/2606.22606",
        "github": "https://github.com/DespinaChristou/compact-relex",
        "hf": "https://huggingface.co/Despina/Qwen2.5-0.5B-Instruct-re_gentune-2-shot",
    },
    "kpis": kpis,
    "headline": headline,
    "findings": findings,
    "matrix": matrix,
    "genDs": GEN_DS, "litDs": LIT_DS, "dsLabel": DS_LABEL,
    "frontierGeneral": frontier_general,
    "frontierLiterary": frontier_literary,
    "bestSlm": best_slm, "bestSlmCfg": best_slm_cfg,
    "encoder": encoder, "encoderAvg": encoder_avg,
    "dapt": dapt,
    "efficiency": efficiency, "efficiencyFrontier": efficiency_frontier,
    "promptDeltas": prompt_deltas, "promptSummary": prompt_summary,
    "constrainedOpen": constrained_open,
    "difficulty": difficulty,
    "scaling": scaling, "scalingNote": scaling_note,
    "significance": significance,
}

# ----------------------------------------------------------------------------
# Inject data into the HTML template and write the dashboard
# ----------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data_json = json.dumps(DATA, ensure_ascii=False, separators=(",", ":"))
    html = TEMPLATE
    html = html.replace("{PSUB}", "%.3f" % prompt_summary["sub_mean"])
    html = html.replace("{PBIG}", "%.3f" % prompt_summary["big_mean"])
    html = html.replace("__DATA__", data_json)

    assert "__DATA__" not in html, "data placeholder not replaced"
    assert "{PSUB}" not in html and "{PBIG}" not in html, "text placeholder left"

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)

    print("Wrote %s (%.0f KB) from %d configurations." % (
        os.path.relpath(OUT, ROOT), len(html) / 1024, len(matrix)))


if __name__ == "__main__":
    main()
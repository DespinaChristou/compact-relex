#!/usr/bin/env python3
"""
Compact encoder-classifier RE baseline (reviewer-requested).

Trains one sequence-classification encoder per dataset on the SAME processed data
the generative SLMs use, with entity markers ([E1]..[/E1], [E2]..[/E2]) and a
[CLS] -> softmax head over each dataset's relation set. Scored with the identical
positive-class micro/macro-F1 as the generative models (src/eval.py), and reports
parameter count + single-example GPU latency for the efficiency comparison.

This answers the reviewer question "why generate if a 100-400M encoder is faster
and more accurate?" with numbers on the exact data — and surfaces the encoder's
structural limits (one classifier per schema; a fixed, closed label set that
cannot emit unseen relations; schema-valid by construction).

Usage:
    pip install sentencepiece            # needed by deberta-v3 tokenizer
    python scripts/train_encoder_baseline.py --config configs/encoder_baseline.yaml
    python scripts/train_encoder_baseline.py --datasets conll04 tacred   # subset
    python scripts/train_encoder_baseline.py --model microsoft/deberta-v3-large

Outputs: runs/encoder_baseline/<dataset>.json, <dataset>_preds.csv,
         runs/encoder_baseline/encoder_baseline_summary.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.hf_utils import get_hf_token
from src.eval import evaluate_slice

MARKERS = ["[E1]", "[/E1]", "[E2]", "[/E2]"]


def _find_span(s, ent):
    """Locate the entity mention in the text, tolerating tokenization whitespace
    (mentions are space-tokenized, e.g. 'Willowdell , Darke County', while the raw
    text is not). Returns (start, end) char offsets, or (-1, -1) if not found."""
    ent = str(ent).strip()
    if not ent:
        return -1, -1
    i = s.find(ent)
    if i >= 0:
        return i, i + len(ent)
    pat = r"\s*".join(re.escape(tok) for tok in ent.split())
    m = re.search(pat, s)
    return (m.start(), m.end()) if m else (-1, -1)


def _mark(text, e1, e2, t1, t2, style):
    """Insert entity markers ([E1]..[/E1], [E2]..[/E2]) inline at the mentions
    (typed when available). Falls back to appending the mentions only when one
    cannot be located or the two spans overlap."""
    s = str(text)

    def opener(lab, typ):
        return f"[{lab}] {typ} :" if (style == "typed" and typ and str(typ) != "None") else f"[{lab}]"

    a1, b1 = _find_span(s, e1)
    a2, b2 = _find_span(s, e2)
    overlap = a1 >= 0 and a2 >= 0 and not (b1 <= a2 or b2 <= a1)
    if a1 < 0 or a2 < 0 or overlap:
        return f"{s} {opener('E1', t1)} {e1} [/E1] {opener('E2', t2)} {e2} [/E2]"
    (pa, pb, pl, pt), (qa, qb, ql, qt) = sorted([(a1, b1, "E1", t1), (a2, b2, "E2", t2)])
    return (s[:pa] + f" {opener(pl, pt)} " + s[pa:pb] + f" [/{pl}] " +
            s[pb:qa] + f" {opener(ql, qt)} " + s[qa:qb] + f" [/{ql}] " + s[qb:])


def _build_texts(ds, style):
    has_t = "entity1Type" in ds.column_names
    t1s = ds["entity1Type"] if has_t else [None] * len(ds)
    t2s = ds["entity2Type"] if has_t else [None] * len(ds)
    return [_mark(tx, e1, e2, a, b, style)
            for tx, e1, e2, a, b in zip(ds["text"], ds["entity1"], ds["entity2"], t1s, t2s)]


def _latency_ms(model, tok, max_len, sample, n_warm, n_run):
    model.eval()
    enc = tok(sample, truncation=True, max_length=max_len, return_tensors="pt")
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        for _ in range(n_warm):
            model(**enc)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(n_run):
            model(**enc)
        torch.cuda.synchronize()
    return round(1000 * (time.time() - t0) / n_run, 2)


def run_one(d, cfg, token):
    name, repo, sub = d["name"], d["repo"], d.get("subset")
    t, style = cfg["train"], cfg.get("marker_style", "typed")

    def load(split):
        return load_dataset(repo, sub, split=split, token=token) if sub else load_dataset(repo, split=split, token=token)

    tr, te = load("train"), load("test")
    cap = t.get("max_train_samples")
    if cap and len(tr) > cap:
        tr = tr.shuffle(seed=t["seed"]).select(range(int(cap)))

    labels = sorted(set(str(x) for x in tr["relation"]))
    lab2id = {l: i for i, l in enumerate(labels)}
    id2lab = {i: l for l, i in lab2id.items()}

    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True, token=token)
    tok.add_special_tokens({"additional_special_tokens": MARKERS})

    tr_enc = tok(_build_texts(tr, style), truncation=True, max_length=t["max_length"])
    te_enc = tok(_build_texts(te, style), truncation=True, max_length=t["max_length"])
    tr_ds = Dataset.from_dict({**tr_enc, "labels": [lab2id[str(x)] for x in tr["relation"]]})
    te_ds = Dataset.from_dict({**te_enc})

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model"], num_labels=len(labels), token=token)
    model.resize_token_embeddings(len(tok))

    args = TrainingArguments(
        output_dir=str(REPO_ROOT / cfg["output_dir"] / name),
        num_train_epochs=t["epochs"],
        learning_rate=float(t["learning_rate"]),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        bf16=bool(t.get("bf16", True)) and torch.cuda.is_available(),
        fp16=bool(t.get("fp16", False)),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=t["seed"],
    )
    set_seed(t["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds,
                      data_collator=DataCollatorWithPadding(tok))
    train_out = trainer.train()

    pred_ids = trainer.predict(te_ds).predictions.argmax(-1)
    pred = [id2lab[int(i)] for i in pred_ids]
    gold = [str(x) for x in te["relation"]]

    # Did-not-learn / collapse detector: near-constant predictions, or a final training
    # loss that is NaN or still near the random baseline (ln(num_labels)), means the run
    # did not converge -- flag it loudly so a degenerate run never reaches the paper.
    import math
    from collections import Counter
    _dist = Counter(pred)
    _top, _n = _dist.most_common(1)[0]
    _loss = getattr(train_out, "training_loss", float("nan"))
    _rand = math.log(max(len(labels), 2))
    if len(_dist) <= 1 or _n / max(len(pred), 1) > 0.95 or _loss != _loss or _loss > 0.9 * _rand:
        print(f"  !! WARNING [{name}]: model did not learn -- predicts '{_top}' for "
              f"{100 * _n / max(len(pred), 1):.0f}% of test ({len(_dist)} distinct labels); "
              f"final train loss={_loss:.3f} vs random baseline {_rand:.3f}. Lower the learning "
              f"rate or use a more stable encoder (e.g. RoBERTa); scores are not trustworthy.",
              flush=True)

    neg = set(cfg["eval"]["negative_labels"])
    m = evaluate_slice(gold, pred, negative_labels=neg, normalize=True)

    lcfg = cfg.get("latency", {})
    lat = None
    if lcfg.get("enabled", True) and torch.cuda.is_available():
        lat = _latency_ms(model, tok, t["max_length"], _build_texts(te, style)[0],
                          lcfg.get("n_warmup", 5), lcfg.get("n_runs", 50))

    res = {
        "dataset": name, "model": cfg["model"], "n_labels": len(labels),
        "n_train": len(tr), "n_test": len(te),
        "pos_micro_f1": m["micro_f1"], "pos_macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "gpu_latency_ms": lat,
        # the classifier can only emit labels seen in training; report the share of
        # test gold labels that are even reachable (an honest ceiling for the encoder).
        "test_labels_in_train_schema": round(
            sum(1 for g in gold if str(g) in set(labels)) / max(len(gold), 1), 4),
    }
    outdir = REPO_ROOT / cfg["output_dir"]
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"{name}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    pd.DataFrame({"relation": gold, "predicted": pred}).to_csv(outdir / f"{name}_preds.csv", index=False)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/encoder_baseline.yaml")
    ap.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset names to run.")
    ap.add_argument("--model", default=None, help="Override the encoder model.")
    ap.add_argument("--output_dir", default=None,
                    help="Override output dir (use a distinct one per model / per GPU split).")
    ap.add_argument("--batch_size", type=int, default=None,
                    help="Override per_device_train_batch_size (raise to 32-48 on A100/H100).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model:
        cfg["model"] = args.model
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.batch_size:
        cfg["train"]["per_device_train_batch_size"] = args.batch_size
    token = get_hf_token(token=None, token_env=cfg["hf"]["token_env"])

    dss = cfg["datasets"]
    if args.datasets:
        dss = [d for d in dss if d["name"] in args.datasets]

    print(f"Encoder baseline: model={cfg['model']} | {len(dss)} dataset(s) | "
          f"CUDA={'yes' if torch.cuda.is_available() else 'NO (CPU)'}", flush=True)
    summary = []
    for d in dss:
        print(f"\n===== {d['name']} =====", flush=True)
        try:
            r = run_one(d, cfg, token)
            summary.append(r)
            print(f"  {d['name']}: pos-micro-F1={r['pos_micro_f1']:.3f}  pos-macro-F1={r['pos_macro_f1']:.3f}  "
                  f"params={r['params_M']}M  lat={r['gpu_latency_ms']}ms  ({r['n_labels']} labels)", flush=True)
        except Exception as e:
            print(f"  {d['name']} FAILED: {type(e).__name__}: {e}", flush=True)

    if summary:
        out = REPO_ROOT / cfg["output_dir"] / "encoder_baseline_summary.csv"
        pd.DataFrame(summary).to_csv(out, index=False)
        print(f"\nSummary -> {out}", flush=True)


if __name__ == "__main__":
    main()

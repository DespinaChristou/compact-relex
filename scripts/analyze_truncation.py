#!/usr/bin/env python3
"""
Quantify prompt/sequence token lengths and truncation, by dataset x prompt
condition x tokenizer, for the reviewer's truncation-reporting request.

Two caps matter in this pipeline:
  * TRAINING  — src/train.py tokenizes (prompt + gold label) with
                truncation=True, max_length = finetune.max_seq_length = 1024
                (configs/experiments.yaml). All five tokenizers truncate on the
                RIGHT, and the gold label is the final token span, so a sequence
                above 1024 loses its trailing query/answer region.
  * INFERENCE — src/run_generations.py tokenizes the chat-wrapped prompt with
                truncation=True and NO max_length, so the cap is each
                tokenizer's model_max_length (8192-131072 here).

We reconstruct both faithfully by importing the very functions the pipeline
uses (_build_chat_texts for training, _build_chat_prompt_text for inference) and
measuring on the real stored prompts (prompt_0_shot / prompt_2_shot) in the
merged generation CSVs. The eval prompts use the identical template and
demonstration count as training; training sequences additionally append the
gold label, which we include. Inference inputs use the primary schema-enumerated
(constrained) system prompt with the per-dataset allowed-label set reconstructed
from the test-split gold relations, exactly as run_generations builds it.

Outputs:
  runs/evaluation/truncation_stats.csv  — one row per dataset x condition x tokenizer
  plus a console summary.

Run (offline; tokenizers are read from the local HF cache):
    python scripts/analyze_truncation.py
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Read tokenizers from the local cache; do not hit the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from transformers import AutoTokenizer
from src.train import _build_chat_texts
from src.run_generations import _build_chat_prompt_text

# ── Pipeline constants (mirror configs/) ────────────────────────────────────
TRAIN_MAX_LEN = 1024  # finetune.max_seq_length (configs/experiments.yaml)
TRAIN_SYSTEM = "You are an expert relation extraction system. Answer with only the relation label."

SYSTEM_OPEN = (
    "You are a relation extraction system. Be concise and direct. "
    "Output ONLY the relation type that holds between the two mentioned entities. "
    "Do not output any explanation, punctuation, or extra. text—only the label."
)
SYSTEM_CONSTRAINED_TMPL = (
    "You are a relation extraction system. Be concise and direct. "
    "Output ONLY ONE relation type that holds between the two mentioned entities. "
    "You MUST choose exactly one label from this allowed set: {allowed_labels} "
    "Do not output any explanation, punctuation, or extra. text—only the label."
)

# base model id -> HF repo for the tokenizer, with offline fallbacks (cached
# fine-tuned checkpoints share the base tokenizer).
TOKENIZERS = {
    "SmolLM2-360M": ["HuggingFaceTB/SmolLM2-360M-Instruct"],
    "Qwen2.5-0.5B": ["Qwen/Qwen2.5-0.5B-Instruct"],
    "SmolLM3-3B": ["HuggingFaceTB/SmolLM3-3B",
                    "Despina/SmolLM3-3B-re_mixtune-2-shot",
                    "Despina/SmolLM3-3B-re_gentune-0-shot"],
    "Qwen2.5-3B": ["Qwen/Qwen2.5-3B-Instruct"],
    "Llama-3.2-3B": ["meta-llama/Llama-3.2-3B-Instruct"],
}

DATASETS = ["tacred", "semeval2010_task8", "conll04", "nyt11", "gids",
            "re_docred", "rebel", "biographical", "pg_fiction"]


def _cache_snapshot(repo: str) -> str | None:
    """Resolve a repo id to its local HF cache snapshot dir (offline-safe)."""
    import glob
    hub = os.path.expanduser("~/.cache/huggingface/hub")
    key = "models--" + repo.replace("/", "--")
    snaps = glob.glob(os.path.join(hub, key, "snapshots", "*"))
    # prefer a snapshot that actually contains tokenizer files
    for s in snaps:
        if os.path.exists(os.path.join(s, "tokenizer.json")) or \
           os.path.exists(os.path.join(s, "tokenizer_config.json")):
            return s
    return snaps[0] if snaps else None


def load_tokenizer(name: str):
    last = None
    for repo in TOKENIZERS[name]:
        targets = [repo]
        snap = _cache_snapshot(repo)
        if snap:
            targets.append(snap)  # explicit local path (handles tokenizer-only checkpoints)
        for target in targets:
            for fast in (True, False):
                try:
                    return AutoTokenizer.from_pretrained(target, use_fast=fast, trust_remote_code=True)
                except Exception as e:  # noqa: BLE001
                    last = e
    print(f"  !! could not load tokenizer {name}: {type(last).__name__}: {str(last)[:120]}")
    return None


def unique_examples(csv_path: Path, cap: int | None) -> pd.DataFrame:
    """Stream the merged CSV and keep one row per unique test example."""
    # NB: read as UTF-8 (the encoding run_generations writes). Reading as latin-1
    # mojibakes multi-byte chars in literary text (curly quotes/accents) into several
    # 1-byte chars, which inflates token counts and overstates truncation.
    seen: dict[str, tuple[str, str]] = {}
    for ch in pd.read_csv(csv_path, encoding="utf-8", encoding_errors="replace",
                          usecols=["prompt_0_shot", "prompt_2_shot", "relation"],
                          dtype=str, on_bad_lines="skip", chunksize=300_000):
        ch = ch.dropna(subset=["prompt_0_shot"])
        for p0, p2, rel in zip(ch.prompt_0_shot, ch.prompt_2_shot, ch.relation):
            if p0 not in seen:
                seen[p0] = (p2 if isinstance(p2, str) else p0, rel if isinstance(rel, str) else "")
        if cap and len(seen) >= cap:
            break
    rows = [(p0, p2, rel) for p0, (p2, rel) in seen.items()]
    if cap:
        rows = rows[:cap]
    return pd.DataFrame(rows, columns=["prompt_0_shot", "prompt_2_shot", "relation"])


def allowed_str_for(df: pd.DataFrame) -> str:
    uniq = sorted({str(x).strip() for x in df.relation if str(x).strip()})
    return ", ".join(uniq)


def tok_lengths(tokenizer, texts: list[str]) -> np.ndarray:
    # add_special_tokens=False: the chat template already inserts them; avoids a
    # spurious extra BOS. (Runtime adds at most 1 token beyond this.)
    enc = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)
    return np.array([len(ids) for ids in enc["input_ids"]], dtype=np.int64)


def pct(mask: np.ndarray) -> float:
    return 100.0 * float(np.mean(mask)) if len(mask) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="runs/generations")
    ap.add_argument("--out", default="runs/evaluation/truncation_stats.csv")
    ap.add_argument("--max_examples", type=int, default=None,
                    help="cap unique examples per dataset (default: all)")
    args = ap.parse_args()

    gen_dir = REPO_ROOT / args.gen_dir
    toks = {}
    print("Loading tokenizers (offline)...")
    for name in TOKENIZERS:
        t = load_tokenizer(name)
        if t is not None:
            toks[name] = t
            print(f"  OK {name:13s} model_max={t.model_max_length} trunc_side={t.truncation_side}")
    if not toks:
        sys.exit("No tokenizers could be loaded.")

    records = []
    for ds in DATASETS:
        csv_path = gen_dir / ds / "generations.csv"
        if not csv_path.exists():
            print(f"  (skip {ds}: no CSV)")
            continue
        ex = unique_examples(csv_path, args.max_examples)
        allowed = allowed_str_for(ex)
        n_labels = len(allowed.split(", ")) if allowed else 0
        print(f"\n{ds}: {len(ex)} unique examples, {n_labels} allowed labels")

        for tname, tk in toks.items():
            sys_constrained = SYSTEM_CONSTRAINED_TMPL.format(allowed_labels=allowed)
            for cond, pcol in [("0-shot", "prompt_0_shot"), ("2-shot", "prompt_2_shot")]:
                prompts = list(ex[pcol].fillna(ex["prompt_0_shot"]))
                rels = list(ex["relation"])

                # INFERENCE input (primary = schema-enumerated / constrained)
                inf_texts = [_build_chat_prompt_text(tokenizer=tk, system_prompt=sys_constrained, user_prompt=p)
                             for p in prompts]
                n_inf = tok_lengths(tk, inf_texts)

                # TRAINING sequence proxy (prompt + gold label), train system prompt.
                # _build_chat_texts returns (prompt_text, full_text) in one call.
                chat_pairs = [
                    _build_chat_texts(tokenizer=tk, system_prompt=TRAIN_SYSTEM,
                                      user_prompt=p, assistant_answer=str(r))
                    for p, r in zip(prompts, rels)
                ]
                n_prompt = tok_lengths(tk, [c[0] for c in chat_pairs])  # prompt only
                n_train = tok_lengths(tk, [c[1] for c in chat_pairs])   # prompt + label

                trunc_train = n_train > TRAIN_MAX_LEN
                label_cut = (n_prompt >= TRAIN_MAX_LEN) & trunc_train  # whole label dropped
                records.append({
                    "dataset": ds, "condition": cond, "tokenizer": tname,
                    "n_examples": len(ex), "n_allowed_labels": n_labels,
                    "model_max_length": int(tk.model_max_length),
                    "truncation_side": tk.truncation_side,
                    # inference input length (schema-enumerated, chat-wrapped)
                    "infer_median": int(np.median(n_inf)),
                    "infer_p95": int(np.percentile(n_inf, 95)),
                    "infer_max": int(n_inf.max()),
                    "pct_infer_gt_modelmax": round(pct(n_inf > tk.model_max_length), 3),
                    "pct_infer_gt_1024": round(pct(n_inf > TRAIN_MAX_LEN), 3),
                    # training sequence length (prompt + label)
                    "train_median": int(np.median(n_train)),
                    "train_p95": int(np.percentile(n_train, 95)),
                    "train_max": int(n_train.max()),
                    "pct_train_trunc_1024": round(pct(trunc_train), 3),
                    "pct_train_label_fully_cut": round(pct(label_cut), 3),
                })
            print(f"  {tname:13s} done")

    out = pd.DataFrame(records)
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}  ({len(out)} rows)")

    # Console summary: worst-case truncation per dataset (max over tokenizers/conditions)
    print("\n=== Training-cap (1024) truncation, worst tokenizer/condition per dataset ===")
    g = (out.groupby("dataset")
            .agg(train_max=("train_max", "max"),
                 pct_train_trunc_1024=("pct_train_trunc_1024", "max"),
                 infer_max=("infer_max", "max"),
                 pct_infer_gt_modelmax=("pct_infer_gt_modelmax", "max"))
            .reindex(DATASETS).dropna())
    print(g.to_string())


if __name__ == "__main__":
    main()

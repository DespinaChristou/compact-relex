#!/usr/bin/env python3
"""
Generate the missing off-diagonal cell of the shot design:
0-shot-tuned checkpoints evaluated with a 2-SHOT prompt (model_shot=0, prompt_shot=2).

The main pipeline never produced this cell because run_generations enforces
model_shot=0 -> prompt_shot=0 (src/run_generations.py:_allowed_prompt_shots_for_model_shot).
Together with the existing (0,0), (2,0), (2,2) cells it completes the 2x2 of
{tuning shot} x {prompt shot}, enabling a clean factorial decomposition of
training-time vs. inference-time demonstrations (Section 4.2 of the paper).

Scope: all 5 base models x 3 regimes (15 zero-shot checkpoints), each on its
regime's eval datasets, both gen types, greedy decoding with SmolLM3 reasoning
disabled and robust <think>-stripping (same corrected settings as the rerun tool).

Usage
-----
Generate (shard across GPUs, like scripts/rerun_broken_generations.py):
    CUDA_VISIBLE_DEVICES=0 python scripts/generate_offdiagonal.py --batch_size 32 --shard_count 2 --shard_index 0
    CUDA_VISIBLE_DEVICES=1 python scripts/generate_offdiagonal.py --batch_size 32 --shard_count 2 --shard_index 1

For a cheap 2x2 estimate, cap examples per dataset (deterministic subsample):
    python scripts/generate_offdiagonal.py --batch_size 32 --max_per_dataset 2000

Splice the (0,2) rows into the merged eval generations (run LOCALLY, no GPU; points
at the in-repo runs/ where the merged CSVs live):
    python scripts/generate_offdiagonal.py --splice_only --runs_dir runs

Then re-run scripts/run_evaluation.py --overwrite: per_dataset_metrics.csv will
gain the (model_shot=0, prompt_shot=2) rows, completing the 2x2.
"""
from __future__ import annotations

import argparse
import logging
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.hf_utils import get_hf_token
from src.train import _unravel_fewshot_prompt_to_messages
from src.run_generations import (
    _batched_indices,
    _compute_allowed_labels_for_eval_dataset,
    _format_allowed_labels,
    _load_hf_split,
    _load_model_and_tokenizer,
    _model_repo_for_run,
    _normalize_label,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("offdiag")

MODELS = ["SmolLM2-360M-Instruct", "Qwen2.5-0.5B-Instruct", "SmolLM3-3B",
          "Qwen2.5-3B-Instruct", "Llama-3.2-3B-Instruct"]
REGIMES = [("re_gentune", "gentune"), ("re_littune", "littune"), ("re_mixtune", "mixtune")]
NO_THINK = {"SmolLM3-3B"}            # disable reasoning mode for SmolLM3
MODEL_SHOT = 0
PROMPT_SHOT = 2                      # the off-diagonal: 0-shot checkpoint, 2-shot prompt

# ── robust postprocess (mirrors scripts/rerun_broken_generations.py) ──
_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_ANSWER = re.compile(r"^(answer\s*:)\s*", re.IGNORECASE)


def _postprocess(text: str) -> str:
    t = _THINK_OPEN.sub("", _THINK_CLOSED.sub("", text or ""))
    for line in t.splitlines():
        line = line.strip()
        if line:
            return _ANSWER.sub("", line).strip()
    return ""


def _build_prompt_text(tokenizer, system_prompt: str, user_prompt: str, no_think: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "system", "content": system_prompt},
                    *_unravel_fewshot_prompt_to_messages(user_prompt)]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if no_think:
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError:
                messages[0]["content"] = system_prompt.rstrip() + " /no_think"
                return tokenizer.apply_chat_template(messages, **kw)
        return tokenizer.apply_chat_template(messages, **kw)
    return f"{system_prompt}\n\n{user_prompt}\n"


@torch.inference_mode()
def _generate(model, tokenizer, prompt_texts: List[str], max_new_tokens: int) -> List[str]:
    enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    plen = int(enc["input_ids"].shape[1])
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    return [_postprocess(t) for t in tokenizer.batch_decode(out[:, plen:], skip_special_tokens=True)]


def _resolve_eval_datasets(cfg: dict, group: str) -> List[Dict[str, Any]]:
    eds = list(cfg.get("eval_datasets", []))
    gv = cfg["eval_groups"][group]
    if isinstance(gv, str) and gv.strip().lower() == "all":
        return eds
    allowed = {str(x).strip() for x in gv}
    return [d for d in eds if str(d["name"]).strip() in allowed]


def _key(model: str) -> str:
    return model.split("-")[0].lower()[:7]


def _splice_offdiag(merged_dir: Path, eval_name: str, df_new: pd.DataFrame,
                    model_id: str, tuned: str) -> None:
    """Append this spec's (0,2) rows to the merged CSV (idempotent; backs up once)."""
    target = merged_dir / eval_name / "generations.csv"
    if not target.exists():
        log.warning(f"  [splice] {target} missing — skip")
        return
    df = pd.read_csv(target)
    backup = target.with_suffix(".csv.offdiagbak")
    if not backup.exists():
        df.to_csv(backup, index=False, encoding="utf-8")
    add = df_new[(df_new.model_id == model_id) & (df_new.tuned_dataset_name == tuned)
                 & (df_new.model_shot == MODEL_SHOT) & (df_new.prompt_shot == PROMPT_SHOT)]
    drop = ((df.model_id == model_id) & (df.tuned_dataset_name == tuned)
            & (df.model_shot == MODEL_SHOT) & (df.prompt_shot == PROMPT_SHOT))
    out = pd.concat([df[~drop], add[df.columns]], ignore_index=True)
    out.to_csv(target, index=False, encoding="utf-8")
    log.info(f"  [splice] {eval_name}: {model_id}/{tuned} +{len(add)} rows "
             f"(replaced {int(drop.sum())} existing)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generations.yaml")
    ap.add_argument("--out_subdir", default="generations_offdiag")
    ap.add_argument("--runs_dir", default=None,
                    help="Override runs/ location (set --runs_dir runs when splicing locally).")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_per_dataset", type=int, default=None,
                    help="Subsample to at most N examples per eval dataset "
                         "(deterministic, seed 42; same subset on every shard). "
                         "Default: full test set. Use e.g. 2000 for a cheap 2x2 estimate.")
    ap.add_argument("--shard_count", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--only_model", default=None, choices=MODELS)
    ap.add_argument("--splice_only", action="store_true",
                    help="No GPU: splice generated (0,2) rows from --out_subdir into the merged CSVs.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    runs_dir = (Path(args.runs_dir) if args.runs_dir
                else REPO_ROOT / cfg["paths"]["runs_dir"]).resolve()
    out_root = runs_dir / args.out_subdir
    merged_dir = runs_dir / str(cfg["paths"].get("output_subdir", "generations"))
    log.info(f"runs_dir = {runs_dir}")

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard_index must be in [0, shard_count)")

    models = [m for m in MODELS if (args.only_model is None or m == args.only_model)]

    # ── splice-only: no model load ──
    if args.splice_only:
        for model in models:
            for tuned, grp in REGIMES:
                for d in _resolve_eval_datasets(cfg, grp):
                    eval_name = str(d["name"])
                    shards = sorted((out_root / eval_name).glob(f"generations_{_key(model)}_{grp}*.csv"))
                    if not shards:
                        continue
                    df_all = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
                    _splice_offdiag(merged_dir, eval_name, df_all, model, tuned)
        log.info("Splice-only done.")
        return

    token = get_hf_token(token=cfg["hf"].get("token"), token_env=cfg["hf"].get("token_env", "HF_TOKEN"))
    org = str(cfg["hf"]["org_or_user"])
    prompts = cfg.get("prompts", {})
    system_open = str(prompts["system_open"]).strip()
    sys_constrained_tmpl = str(prompts["system_constrained_template"]).strip()
    cf = dict(cfg.get("label_sets", {}).get("compute_from", {}))
    csplit, rcol, cmax = str(cf.get("split", "test")), str(cf.get("relation_column", "relation")), cf.get("max_rows", None)

    log.info("CUDA available" if torch.cuda.is_available() else "CPU only (slow)")

    for model in models:
        no_think = model in NO_THINK
        for tuned, grp in REGIMES:
            repo = _model_repo_for_run(org_or_user=org, model_id=model,
                                       tuned_dataset_name=tuned, model_shot=MODEL_SHOT)
            evs = _resolve_eval_datasets(cfg, grp)
            log.info(f"=== {model} / {tuned} / ms0->ps2 (no_think={no_think}) | "
                     f"{len(evs)} datasets | loading {repo}")
            model_obj, tok = _load_model_and_tokenizer(model_repo=repo, token=token)
            for d in evs:
                eval_name = str(d["name"])
                ds = _load_hf_split(repo=str(d["repo"]), subset=d.get("subset"),
                                    split=str(d.get("split", "test")), token=token)
                p0, p2 = list(ds["prompt_0_shot"]), list(ds["prompt_2_shot"])
                gold = [_normalize_label(x) for x in ds["relation"]]
                user = list(ds["prompt_2_shot"])          # prompt_shot = 2
                allowed = _format_allowed_labels(_compute_allowed_labels_for_eval_dataset(
                    repo=str(d["repo"]), subset=d.get("subset"), compute_split=csplit,
                    relation_column=rcol, max_rows=cmax, token=token))
                sys_constrained = sys_constrained_tmpl.format(allowed_labels=allowed)

                # Optional deterministic subsample (same indices on every shard), then shard.
                sel = list(range(len(user)))
                if args.max_per_dataset and len(sel) > args.max_per_dataset:
                    sel = sorted(random.Random(42).sample(sel, args.max_per_dataset))
                assigned = [sel[k] for k in range(len(sel)) if k % args.shard_count == args.shard_index]
                nb = math.ceil(len(assigned) / args.batch_size) if assigned else 0
                log.info(f"  [{eval_name}] {len(assigned)}/{len(sel)} examples "
                         f"(subsample of {len(user)}; shard {args.shard_index}/{args.shard_count}), {nb} batches")
                rows: List[Dict[str, Any]] = []
                t0 = time.time()
                for bn, pos in enumerate(_batched_indices(len(assigned), args.batch_size), 1):
                    idxs = [assigned[p] for p in pos]
                    batch = [user[i] for i in idxs]
                    for gt, sysp in (("gen_open", system_open), ("gen_constrained", sys_constrained)):
                        preds = _generate(model_obj, tok,
                                          [_build_prompt_text(tok, sysp, x, no_think) for x in batch],
                                          args.max_new_tokens)
                        for j, pr in enumerate(preds):
                            i = idxs[j]
                            rows.append({
                                "eval_dataset_name": eval_name, "prompt_0_shot": p0[i], "prompt_2_shot": p2[i],
                                "relation": gold[i], "gen_type": gt, "model_id": model,
                                "tuned_dataset_name": tuned, "model_shot": MODEL_SHOT,
                                "prompt_shot": PROMPT_SHOT, "generated_relation": pr})
                    if bn == 1 or bn % 20 == 0 or bn == nb:
                        rate = (bn * args.batch_size) / max(time.time() - t0, 1e-6)
                        log.info(f"    batch {bn}/{nb} (~{rate:.1f} ex/s); sample={rows[-1]['generated_relation']!r}")
                suffix = f"_shard_{args.shard_index}" if args.shard_count > 1 else ""
                op = out_root / eval_name / f"generations_{_key(model)}_{grp}{suffix}.csv"
                op.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(rows).to_csv(op, index=False, encoding="utf-8")
                log.info(f"  [{eval_name}] wrote {len(rows)} rows -> {op}")
            del model_obj, tok
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    log.info("Done.")


if __name__ == "__main__":
    main()

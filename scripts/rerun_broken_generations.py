#!/usr/bin/env python3
"""
Regenerate ONLY the two broken 0-shot generation configs:

  1) SmolLM3-3B           tuned on re_mixtune, model_shot=0
       Symptom: every output is "<think>" (SmolLM3 reasoning mode was left on;
       _postprocess_relation kept only the first line, i.e. the opening tag).
       Eval datasets: ALL 9 (mixtune group -> "all").

  2) Qwen2.5-3B-Instruct  tuned on re_gentune, model_shot=0
       Symptom: run-on / out-of-schema labels ("Live or demolished", ...).
       Under-converged checkpoint rambles; max_new_tokens=128 + first-line-only
       postprocessing stored the whole ramble.
       Eval datasets: 7 general (gentune group).

Both are model_shot=0 -> only prompt_shot=0 is valid.

Fixes applied here vs. the original run:
  - greedy decoding (do_sample=False) instead of do_sample=True + temperature=1e-3
  - SmolLM3 reasoning disabled (enable_thinking=False / "/no_think" fallback)
  - robust postprocessing that strips <think>...</think> blocks (closed or not)
  - smaller max_new_tokens budget (labels are short)

It reuses repos / prompts / label-set logic from configs/generations.yaml and the
helpers in src/run_generations.py, so it stays consistent with the main pipeline.

Output (default): runs/generations_rerun/<eval_dataset>/generations.csv
                  (same long/tidy schema as the main pipeline)

To patch the corrected rows back into the merged CSVs in runs/generations/, add
--splice. That drops the matching (model_id, tuned_dataset_name, model_shot=0)
rows from each affected runs/generations/<eval>/generations.csv and appends the
new ones, after writing a .bak backup of each file it touches.

Usage:
    python scripts/rerun_broken_generations.py --config configs/generations.yaml
    python scripts/rerun_broken_generations.py --config configs/generations.yaml --splice
    python scripts/rerun_broken_generations.py --only smollm3   # or: qwen
"""
from __future__ import annotations

import argparse
import logging
import math
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rerun")

# ---------------------------------------------------------------------------
# The two broken configs (all are model_shot=0 -> prompt_shot=0).
# `group` selects eval datasets via config.eval_groups.
# `no_think` disables SmolLM3 reasoning mode in the chat template.
# ---------------------------------------------------------------------------
RERUN_SPECS = [
    {"key": "smollm3", "model_id": "SmolLM3-3B",
     "tuned": "re_mixtune", "group": "mixtune", "no_think": True},
    {"key": "qwen", "model_id": "Qwen2.5-3B-Instruct",
     "tuned": "re_gentune", "group": "gentune", "no_think": False},
]

MODEL_SHOT = 0
PROMPT_SHOT = 0

# ---------------------------------------------------------------------------
# Robust postprocessing: strip <think> blocks (closed OR unclosed), then take
# the first non-empty line and drop a leading "Answer:".
# ---------------------------------------------------------------------------
_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*$", re.DOTALL | re.IGNORECASE)
_ANSWER_PREFIX = re.compile(r"^(answer\s*:)\s*", re.IGNORECASE)


def _postprocess(text: str) -> str:
    t = text or ""
    t = _THINK_CLOSED.sub("", t)
    t = _THINK_OPEN.sub("", t)  # handle a think block that ran out of tokens
    for line in t.splitlines():
        line = line.strip()
        if line:
            return _ANSWER_PREFIX.sub("", line).strip()
    return ""


def _build_prompt_text(tokenizer, system_prompt: str, user_prompt: str, no_think: bool) -> str:
    """Chat-templated prompt, optionally disabling reasoning for SmolLM3."""
    if getattr(tokenizer, "chat_template", None):
        messages = [
            {"role": "system", "content": system_prompt},
            *_unravel_fewshot_prompt_to_messages(user_prompt),
        ]
        kw = dict(tokenize=False, add_generation_prompt=True)
        if no_think:
            try:
                return tokenizer.apply_chat_template(messages, enable_thinking=False, **kw)
            except TypeError:
                # Older/other templates: fall back to the "/no_think" soft switch.
                messages[0]["content"] = system_prompt.rstrip() + " /no_think"
                return tokenizer.apply_chat_template(messages, **kw)
        return tokenizer.apply_chat_template(messages, **kw)
    return f"{system_prompt}\n\n{user_prompt}\n"


@torch.inference_mode()
def _generate(model, tokenizer, prompt_texts: List[str], max_new_tokens: int) -> List[str]:
    enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy / deterministic
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    gen_ids = out[:, prompt_len:]
    return [_postprocess(t) for t in tokenizer.batch_decode(gen_ids, skip_special_tokens=True)]


def _resolve_eval_datasets(cfg: dict, group: str) -> List[Dict[str, Any]]:
    eval_datasets = list(cfg.get("eval_datasets", []))
    group_value = cfg["eval_groups"][group]
    if isinstance(group_value, str) and group_value.strip().lower() == "all":
        return eval_datasets
    allowed = {str(x).strip() for x in group_value}
    return [d for d in eval_datasets if str(d["name"]).strip() in allowed]


def _splice_into_merged(merged_dir: Path, eval_name: str, new_rows: pd.DataFrame,
                        model_id: str, tuned: str) -> None:
    """Drop the broken (model_id, tuned, model_shot=0) rows from the merged CSV and append new ones."""
    target = merged_dir / eval_name / "generations.csv"
    if not target.exists():
        log.warning(f"  [splice] {target} not found — skipping.")
        return
    df = pd.read_csv(target)
    mask = (
        (df["model_id"] == model_id)
        & (df["tuned_dataset_name"] == tuned)
        & (df["model_shot"] == MODEL_SHOT)
    )
    # General eval datasets are shared by BOTH specs (smollm3-mixtune AND qwen-gentune
    # both eval e.g. tacred), so a rerun file may contain rows from both. Keep only the
    # rows belonging to THIS spec before appending — makes the splice idempotent.
    new_match = new_rows[
        (new_rows["model_id"] == model_id)
        & (new_rows["tuned_dataset_name"] == tuned)
        & (new_rows["model_shot"] == MODEL_SHOT)
    ]
    backup = target.with_suffix(".csv.bak")
    if not backup.exists():
        df.to_csv(backup, index=False, encoding="utf-8")
    kept = df[~mask]
    combined = pd.concat([kept, new_match[df.columns]], ignore_index=True)
    combined.to_csv(target, index=False, encoding="utf-8")
    log.info(f"  [splice] {eval_name}: removed {int(mask.sum())} broken rows, "
             f"added {len(new_match)} -> {target} (backup: {backup.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/generations.yaml")
    ap.add_argument("--out_subdir", default="generations_rerun",
                    help="Output dir under runs/ for the corrected rows.")
    ap.add_argument("--runs_dir", default=None,
                    help="Override the runs/ location (default: config paths.runs_dir = ../runs). "
                         "Set this when splicing locally where the merged CSVs live in the repo, e.g. --runs_dir runs")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--only", choices=[s["key"] for s in RERUN_SPECS], default=None,
                    help="Run only one of the two specs.")
    ap.add_argument("--shard_count", type=int, default=1,
                    help="Split each dataset's examples across this many workers (one per GPU).")
    ap.add_argument("--shard_index", type=int, default=0,
                    help="This worker's shard in [0..shard_count-1].")
    ap.add_argument("--splice", action="store_true",
                    help="Patch corrected rows back into runs/generations/<eval>/generations.csv (with .bak backups).")
    ap.add_argument("--splice_only", action="store_true",
                    help="No GPU/generation: splice already-generated rows from --out_subdir into the merged CSVs. "
                         "Run this LOCALLY after downloading runs/<out_subdir>/ from the pod.")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    runs_dir = (Path(args.runs_dir) if args.runs_dir
                else REPO_ROOT / cfg["paths"]["runs_dir"]).resolve()
    log.info(f"runs_dir = {runs_dir}")
    out_root = runs_dir / args.out_subdir
    merged_dir = runs_dir / str(cfg["paths"].get("output_subdir", "generations"))

    specs_all = [s for s in RERUN_SPECS if (args.only is None or s["key"] == args.only)]

    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("--shard_index must be in [0, shard_count)")
    if args.splice and args.shard_count > 1:
        raise SystemExit("Do not use --splice with --shard_count>1 (concurrent writers would "
                         "corrupt the merged CSV). Generate all shards, then run --splice_only.")

    # ---- splice-only path: read saved rerun shard CSVs, patch merged CSVs, no model load ----
    if args.splice_only:
        for spec in specs_all:
            for d in _resolve_eval_datasets(cfg, spec["group"]):
                eval_name = str(d["name"])
                shards = sorted((out_root / eval_name).glob("generations*.csv"))
                if not shards:
                    log.warning(f"  [splice_only] no generations*.csv under {out_root / eval_name} — skipping.")
                    continue
                df_all = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
                _splice_into_merged(merged_dir, eval_name, df_all, spec["model_id"], spec["tuned"])
        log.info("Splice-only done.")
        return

    token = get_hf_token(token=cfg["hf"].get("token"), token_env=cfg["hf"].get("token_env", "HF_TOKEN"))
    org_or_user = str(cfg["hf"]["org_or_user"])

    # Prompts + label-set config (mirror the main pipeline).
    prompts_cfg = cfg.get("prompts", {})
    system_open = str(prompts_cfg["system_open"]).strip()
    system_constrained_tmpl = str(prompts_cfg["system_constrained_template"]).strip()

    ls_cfg = cfg.get("label_sets", {})
    compute_from = dict(ls_cfg.get("compute_from", {}))
    compute_split = str(compute_from.get("split", "test"))
    relation_column = str(compute_from.get("relation_column", "relation"))
    compute_max_rows = compute_from.get("max_rows", None)

    if torch.cuda.is_available():
        log.info(f"CUDA: {torch.cuda.device_count()} device(s)")
    else:
        log.warning("CUDA not available — running on CPU (slow).")

    for spec in specs_all:
        model_id, tuned, group, no_think = spec["model_id"], spec["tuned"], spec["group"], spec["no_think"]
        eval_datasets = _resolve_eval_datasets(cfg, group)
        repo = _model_repo_for_run(org_or_user=org_or_user, model_id=model_id,
                                   tuned_dataset_name=tuned, model_shot=MODEL_SHOT)
        log.info(f"=== {spec['key']}: {model_id} / {tuned} / ms{MODEL_SHOT} "
                 f"(no_think={no_think}) | {len(eval_datasets)} eval datasets ===")
        log.info(f"  Loading {repo} ...")
        model, tokenizer = _load_model_and_tokenizer(model_repo=repo, token=token)

        for d in eval_datasets:
            eval_name = str(d["name"])
            ds = _load_hf_split(repo=str(d["repo"]), subset=d.get("subset"),
                                split=str(d.get("split", "test")), token=token)
            prompt_col = f"prompt_{PROMPT_SHOT}_shot"
            prompt_0 = list(ds["prompt_0_shot"])
            prompt_2 = list(ds["prompt_2_shot"])
            gold = [_normalize_label(x) for x in ds["relation"]]
            user_prompts = list(ds[prompt_col])

            allowed = _format_allowed_labels(_compute_allowed_labels_for_eval_dataset(
                repo=str(d["repo"]), subset=d.get("subset"), compute_split=compute_split,
                relation_column=relation_column, max_rows=compute_max_rows, token=token))
            system_constrained = system_constrained_tmpl.format(allowed_labels=allowed)

            # This worker's slice of examples (one shard per GPU).
            assigned = [i for i in range(len(user_prompts)) if i % args.shard_count == args.shard_index]
            n = len(assigned)
            n_batches = math.ceil(n / args.batch_size) if n else 0
            log.info(f"  [{eval_name}] {n:,}/{len(user_prompts):,} examples "
                     f"(shard {args.shard_index}/{args.shard_count}), {n_batches} batches")
            rows: List[Dict[str, Any]] = []
            t0 = time.time()
            for bnum, pos in enumerate(_batched_indices(n, args.batch_size), 1):
                idxs = [assigned[p] for p in pos]
                batch = [user_prompts[i] for i in idxs]
                for gen_type, system in (("gen_open", system_open),
                                         ("gen_constrained", system_constrained)):
                    texts = [_build_prompt_text(tokenizer, system, p, no_think) for p in batch]
                    preds = _generate(model, tokenizer, texts, args.max_new_tokens)
                    for j, pred in enumerate(preds):
                        i = idxs[j]
                        rows.append({
                            "eval_dataset_name": eval_name,
                            "prompt_0_shot": prompt_0[i],
                            "prompt_2_shot": prompt_2[i],
                            "relation": gold[i],
                            "gen_type": gen_type,
                            "model_id": model_id,
                            "tuned_dataset_name": tuned,
                            "model_shot": MODEL_SHOT,
                            "prompt_shot": PROMPT_SHOT,
                            "generated_relation": pred,
                        })
                if bnum == 1 or bnum % 20 == 0 or bnum == n_batches:
                    rate = (bnum * args.batch_size) / max(time.time() - t0, 1e-6)
                    log.info(f"    batch {bnum}/{n_batches} (~{rate:.1f} ex/s); "
                             f"sample pred={rows[-1]['generated_relation']!r}")

            df_new = pd.DataFrame(rows)
            # Filename carries the spec key (general datasets are run by BOTH specs, so a
            # bare name would clobber) and the shard index (one writer per GPU).
            shard_suffix = f"_shard_{args.shard_index}" if args.shard_count > 1 else ""
            out_path = out_root / eval_name / f"generations_{spec['key']}{shard_suffix}.csv"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df_new.to_csv(out_path, index=False, encoding="utf-8")
            log.info(f"  [{eval_name}] wrote {len(df_new)} rows -> {out_path}")

            if args.splice:
                _splice_into_merged(merged_dir, eval_name, df_new, model_id, tuned)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log.info("Done.")


if __name__ == "__main__":
    main()

"""
Generate relation labels ("relation extraction as generation") for fine-tuned checkpoints.

This script evaluates *fine-tuned* RE checkpoints on the ORIGINAL evaluation datasets
(TACRED, SemEval, etc.) using the dataset-provided prompts.

Key ideas
---------
We must distinguish two different "shot" concepts:

1) model_shot:
   The few-shot setting the model checkpoint was *fine-tuned* under.
   This is part of the model repo name:
     {org_or_user}/{model_id}-{tuned_dataset_name}-{model_shot}-shot

2) prompt_shot:
   The prompt variant we use at *inference* time.
   We select the corresponding column from the evaluation dataset:
     prompt_0_shot or prompt_2_shot

Evaluation constraints (requested)
---------------------------------
A) Prompt-shot compatibility with the tuned checkpoint:
   - model_shot = 0  -> evaluate ONLY with prompt_shot = 0
   - model_shot = 2  -> evaluate with prompt_shot in {0, 2}

B) Which eval datasets to run for each tuned dataset name:
   This is config-driven:
     - tuned_dataset_to_group: maps tuned_dataset_name -> group key
     - eval_groups: maps group key -> list of eval dataset names (or "all")

   Required behavior:
     - re_littune models -> only littune eval datasets
     - re_gentune models -> only gentune eval datasets
     - re_mixtune models -> all eval datasets

Generation types
----------------
We produce two generations for each example:

- gen_open:
    system prompt is generic (no label-set restriction).

- gen_constrained:
    system prompt includes an explicit allowed label set for the evaluation dataset.

Allowed labels for gen_constrained are computed per evaluation dataset by loading a
configurable split (label_sets.compute_from.split) and taking unique relation labels
from label_sets.compute_from.relation_column. If a label is None/NaN/empty, it is
normalized to the literal string label "none".

Output format (LONG / tidy)
---------------------------
We write one shard file per eval dataset per worker:

  runs/<output_subdir>/<eval_dataset_name>/generations_shard_<job_index>.<ext>

Where <ext> is controlled by config.output.format:
  - csv
  - parquet

Each row is one generation:
  - eval_dataset_name
  - prompt_0_shot
  - prompt_2_shot
  - relation
  - gen_type              ("gen_open" | "gen_constrained")
  - model_id
  - tuned_dataset_name    ("re_gentune" | "re_littune" | "re_mixtune")
  - model_shot            (0 | 2)
  - prompt_shot           (0 | 2)
  - generated_relation

Parallelism / sharding
----------------------
We shard jobs deterministically using (i % job_count == job_index). This is job-level
parallelism; each worker writes only to its own shard files.

For CSV we append incrementally.

For Parquet we cannot reliably "append" to the same file in-place; instead we buffer
rows in memory per job and then read+concat+rewrite the shard file. This is safe but
can be slower for very large runs. If that becomes a bottleneck, we can switch parquet
to write per-job part files and merge them later.

Usage
-----
python -m src.run_generations --config configs/generations.yaml
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterable

import pandas as pd
import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.hf_utils import get_hf_token
from src.train import _unravel_fewshot_prompt_to_messages


# -------------------------
# Sharding helper
# -------------------------
def _is_assigned(global_index: int, job_index: int, job_count: int) -> bool:
    """Deterministic sharding: each worker takes items where (i % job_count == job_index)."""
    return (global_index % job_count) == job_index


# -------------------------
# Label normalization
# -------------------------
def _is_nan_like(x: Any) -> bool:
    """True if x is None / NaN float / empty-ish string."""
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "nan", "null", "none"}:
        return True
    return False


def _normalize_label(x: Any) -> str:
    """
    Normalize relation labels.

    Requirement: if None/NaN/empty is seen, map it to literal string label "none".
    """
    if _is_nan_like(x):
        return "none"
    s = str(x).strip()
    return s if s else "none"


def _format_allowed_labels(labels: List[str]) -> str:
    """Deterministic comma-separated allowed label list for constrained prompting."""
    uniq = sorted({str(x).strip() for x in labels if str(x).strip()})
    return ", ".join(uniq)


# -------------------------
# HF loading helpers
# -------------------------
def _load_hf_split(*, repo: str, subset: Optional[str], split: str, token: str) -> Dataset:
    """
    Load a split from HF Datasets with an optional dataset configuration (subset).

    - If subset is null/None/"default", we pass None as the dataset config name.
    """
    subset_arg = None if (subset is None or subset == "default") else subset
    return load_dataset(repo, subset_arg, split=split, token=token)


def _compute_allowed_labels_for_eval_dataset(
    *,
    repo: str,
    subset: Optional[str],
    compute_split: str,
    relation_column: str,
    max_rows: Optional[int],
    token: str,
) -> List[str]:
    """
    Compute the allowed labels for an evaluation dataset by reading `compute_split`
    and collecting unique labels from `relation_column`.

    Any missing/NaN/empty label values are normalized to "none".
    """
    ds = _load_hf_split(repo=repo, subset=subset, split=compute_split, token=token)

    if max_rows is not None:
        ds = ds.select(range(min(int(max_rows), len(ds))))

    if relation_column not in ds.column_names:
        raise ValueError(
            f"[label_sets] relation_column='{relation_column}' not found in {repo} "
            f"({subset or 'default'}) split={compute_split}. Available: {ds.column_names}"
        )

    labels = [_normalize_label(x) for x in ds[relation_column]]
    return sorted(set(labels))


# -------------------------
# Prompt building
# -------------------------
def _build_chat_prompt_text(*, tokenizer, system_prompt: str, user_prompt: str) -> str:
    """
    Build the exact text passed to model.generate.

    We use the model's chat template when available and unravel the dataset few-shot format
    into alternating user/assistant turns (for demonstrations).
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [
            {"role": "system", "content": system_prompt},
            *(_unravel_fewshot_prompt_to_messages(user_prompt)),
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Fallback for non-chat models.
    return f"{system_prompt}\n\n{user_prompt}\n"


def _postprocess_relation(text: str) -> str:
    """Best-effort cleanup to enforce 'label only' outputs."""
    t = (text or "").strip()
    if not t:
        return ""
    t = t.splitlines()[0].strip()
    t = re.sub(r"^(answer\s*:)\s*", "", t, flags=re.IGNORECASE)
    return t.strip()


# -------------------------
# Generation
# -------------------------
@dataclass(frozen=True)
class GenParams:
    batch_size: int
    max_new_tokens: int
    do_sample: bool
    temperature: float
    top_p: float
    repetition_penalty: float


def _batched_indices(n: int, batch_size: int) -> Iterable[List[int]]:
    """Yield index batches over [0..n)."""
    for i in range(0, n, batch_size):
        yield list(range(i, min(i + batch_size, n)))


def _load_model_and_tokenizer(model_repo: str, token: str):
    """Load model+tokenizer for inference with device_map='auto'."""
    tokenizer = AutoTokenizer.from_pretrained(model_repo, use_fast=True, token=token, padding_side='left')
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"[{model_repo}] tokenizer has no pad_token and no eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_repo,
        token=token,
        device_map="auto",
        dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )
    model.eval()
    return model, tokenizer


@torch.inference_mode()
def _generate_relations(
    *,
    model,
    tokenizer,
    prompt_texts: List[str],
    gen: GenParams,
) -> List[str]:
    """
    Generate labels for a batch of prompt texts.

    We decode only the continuation tokens (not the prompt itself).
    """
    enc = tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    prompt_len = int(enc["input_ids"].shape[1])

    out = model.generate(
        **enc,
        max_new_tokens=int(gen.max_new_tokens),
        do_sample=bool(gen.do_sample),
        temperature=float(gen.temperature),
        top_p=float(gen.top_p),
        repetition_penalty=float(gen.repetition_penalty),
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_ids = out[:, prompt_len:]
    texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return [_postprocess_relation(t) for t in texts]


# -------------------------
# Repo naming + evaluation policy
# -------------------------
def _model_repo_for_run(*, org_or_user: str, model_id: str, tuned_dataset_name: str, model_shot: int) -> str:
    """
    Fine-tuned checkpoint naming convention:
      {org_or_user}/{model_id}-{tuned_dataset_name}-{model_shot}-shot
    """
    return f"{org_or_user}/{model_id}-{tuned_dataset_name}-{int(model_shot)}-shot"


def _allowed_prompt_shots_for_model_shot(model_shot: int) -> List[int]:
    """Enforce: 0-> [0], 2-> [0,2]."""
    ms = int(model_shot)
    if ms == 0:
        return [0]
    if ms == 2:
        return [0, 2]
    raise ValueError(f"Unsupported model_shot={model_shot}. Expected 0 or 2.")


def _resolve_allowed_eval_names(
    *,
    tuned_dataset_name: str,
    eval_datasets: List[Dict[str, Any]],
    eval_groups_cfg: Dict[str, Any],
    tuned_to_group_cfg: Dict[str, Any],
) -> Optional[set[str]]:
    """
    Resolve which eval dataset names are allowed for a tuned dataset name.

    Cautious behavior:
      - tuned_dataset_name must exist in tuned_dataset_to_group
      - group key must exist in eval_groups
      - group members must exist in eval_datasets[].name
      - special value 'all' means: allow all eval datasets (return None)
    """
    eval_names = {str(d.get("name", "")).strip() for d in eval_datasets}
    if "" in eval_names:
        raise ValueError("Found eval_datasets entry with empty name.")

    tuned = str(tuned_dataset_name).strip()
    if tuned not in tuned_to_group_cfg:
        raise ValueError(
            f"tuned_dataset_name='{tuned}' not found in config.tuned_dataset_to_group. "
            f"Available: {sorted(map(str, tuned_to_group_cfg.keys()))}"
        )

    group_key = str(tuned_to_group_cfg[tuned]).strip()
    if group_key not in eval_groups_cfg:
        raise ValueError(
            f"Group '{group_key}' (from tuned_dataset_to_group['{tuned}']) not found in config.eval_groups. "
            f"Available: {sorted(map(str, eval_groups_cfg.keys()))}"
        )

    group_value = eval_groups_cfg[group_key]

    if isinstance(group_value, str) and group_value.strip().lower() == "all":
        return None

    if not isinstance(group_value, list):
        raise ValueError(
            f"config.eval_groups['{group_key}'] must be a list of eval dataset names or 'all'. Got: {type(group_value)}"
        )

    allowed = {str(x).strip() for x in group_value if str(x).strip()}
    unknown = sorted(allowed - eval_names)
    if unknown:
        raise ValueError(
            f"config.eval_groups['{group_key}'] contains names not present in eval_datasets[].name: {unknown}. "
            f"Known eval_datasets names: {sorted(eval_names)}"
        )
    return allowed


# -------------------------
# Output writing (csv/parquet)
# -------------------------
def _append_rows_to_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    """Append rows to CSV; write header only if file does not exist."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False, encoding="utf-8")


def _append_rows_to_parquet(path: Path, rows: List[Dict[str, Any]]) -> None:
    """
    Safe-but-slower append for Parquet:
      - read existing shard parquet if present
      - concat new rows
      - rewrite shard parquet

    This avoids concurrent writes because each worker writes to its own shard file.
    """
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(rows)

    if path.exists():
        df_old = pd.read_parquet(path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_parquet(path, index=False)


def _append_rows(path: Path, rows: List[Dict[str, Any]], fmt: str) -> None:
    fmt2 = str(fmt).strip().lower()
    if fmt2 == "csv":
        _append_rows_to_csv(path, rows)
        return
    if fmt2 == "parquet":
        _append_rows_to_parquet(path, rows)
        return
    raise ValueError(f"Unsupported output.format='{fmt}'. Expected 'csv' or 'parquet'.")


# -------------------------
# Main
# -------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    # ap.add_argument("--config", required=True, help="Path to configs/generations.yaml")
    ap.add_argument("--config", help="Path to configs/generations.yaml")
    ap.add_argument("--job_index", type=int, default=0, help="Shard index in [0..job_count-1]")
    ap.add_argument("--job_count", type=int, default=1, help="Number of shards / parallel workers")
    args = ap.parse_args()

    # if args.job_count < 1:
    #     raise ValueError("--job_count must be >= 1")
    # if not (0 <= args.job_index < args.job_count):
    #     raise ValueError("--job_index must be in [0..job_count-1]")

    cfg = yaml.safe_load(Path("configs/generations.yaml").read_text(encoding="utf-8"))
    # cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    # HF auth
    token = get_hf_token(token=cfg["hf"].get("token", None), token_env=cfg["hf"].get("token_env", "HF_TOKEN"))
    org_or_user = str(cfg["hf"]["org_or_user"])

    # Output location
    runs_dir = Path(cfg["paths"]["runs_dir"])
    out_root = runs_dir / str(cfg["paths"].get("output_subdir", "generations"))
    out_root.mkdir(parents=True, exist_ok=True)

    # Output configuration
    out_cfg = dict(cfg.get("output", {}))
    out_format = str(out_cfg.get("format", "csv")).strip().lower()
    if out_format not in {"csv", "parquet"}:
        raise ValueError("config.output.format must be one of: csv | parquet")
    if not bool(out_cfg.get("sharded", True)):
        raise ValueError("config.output.sharded=false is not supported (use merge script to produce final files).")

    # Evaluation datasets
    eval_datasets = list(cfg.get("eval_datasets", []))
    eval_names_list = [str(d.get("name", "")).strip() for d in eval_datasets]
    if any(not n for n in eval_names_list):
        raise ValueError(f"Found empty eval_datasets[].name entry: {eval_names_list}")
    if len(set(eval_names_list)) != len(eval_names_list):
        raise ValueError(f"Duplicate eval_datasets[].name entries found: {eval_names_list}")

    # Tuned datasets (used for model repo naming + family filtering)
    tuned_datasets_raw = cfg.get("tuned_datasets", [])
    if not isinstance(tuned_datasets_raw, list) or not tuned_datasets_raw:
        raise ValueError("Expected 'tuned_datasets' to be a non-empty list of names (e.g., ['re_gentune', ...]).")
    tuned_dataset_names = [str(x).strip() for x in tuned_datasets_raw if str(x).strip()]
    if not tuned_dataset_names:
        raise ValueError("Expected 'tuned_datasets' to contain at least one non-empty name.")

    # model_shots are the tuned checkpoint shots we will load (0-shot-tuned, 2-shot-tuned)
    model_shots = [int(s) for s in cfg.get("shots", [0, 2])]
    for ms in model_shots:
        if ms not in (0, 2):
            raise ValueError(f"Only model_shot values [0,2] are supported. Got: {model_shots}")

    # Models
    model_ids: List[str] = list(cfg.get("models", {}).get("ids", []))
    include_dapt_models = bool(cfg.get("models", {}).get("include_dapt_models", False))

    # Generation params
    gen_cfg = cfg.get("generation", {})
    gen_params = GenParams(
        batch_size=int(gen_cfg.get("batch_size", 16)),
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 8)),
        do_sample=bool(gen_cfg.get("do_sample", False)),
        temperature=float(gen_cfg.get("temperature", 0.0)),
        top_p=float(gen_cfg.get("top_p", 1.0)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
    )

    # Prompts
    prompts_cfg = cfg.get("prompts", {})
    system_open = str(prompts_cfg.get("system_open", "")).strip()
    system_constrained_tmpl = str(prompts_cfg.get("system_constrained_template", "")).strip()
    if not system_open:
        raise ValueError("Missing prompts.system_open in config.")
    if not system_constrained_tmpl:
        raise ValueError("Missing prompts.system_constrained_template in config.")

    # Label-set computation config
    label_sets_cfg = cfg.get("label_sets", {})
    label_sets_enabled = bool(label_sets_cfg.get("enabled", True))
    compute_from = dict(label_sets_cfg.get("compute_from", {}))
    compute_split = str(compute_from.get("split", "train"))
    relation_column = str(compute_from.get("relation_column", "relation"))
    compute_max_rows = compute_from.get("max_rows", None)

    # Eval groups config for tuned-family restrictions
    eval_groups_cfg = cfg.get("eval_groups", None)
    tuned_to_group_cfg = cfg.get("tuned_dataset_to_group", None)
    if not isinstance(eval_groups_cfg, dict) or not eval_groups_cfg:
        raise ValueError("Missing or invalid config.eval_groups (expected a non-empty mapping).")
    if not isinstance(tuned_to_group_cfg, dict) or not tuned_to_group_cfg:
        raise ValueError("Missing or invalid config.tuned_dataset_to_group (expected a non-empty mapping).")

    # ------------------------------------------------------------
    # Precompute allowed labels per eval dataset (once).
    # ------------------------------------------------------------
    allowed_labels_by_eval: Dict[str, str] = {}
    if label_sets_enabled:
        for d in eval_datasets:
            eval_name = str(d["name"])
            eval_repo = str(d["repo"])
            eval_subset = d.get("subset", None)

            labels = _compute_allowed_labels_for_eval_dataset(
                repo=eval_repo,
                subset=eval_subset,
                compute_split=compute_split,
                relation_column=relation_column,
                max_rows=compute_max_rows,
                token=token,
            )
            allowed_labels_by_eval[eval_name] = _format_allowed_labels(labels)

    # ------------------------------------------------------------
    # Build job list and shard it.
    #
    # Job = (tuned_dataset_name, eval_dataset_dict, model_shot, prompt_shot)
    # - prompt_shot values depend on model_shot
    # - eval datasets depend on tuned_dataset_name via eval_groups
    # ------------------------------------------------------------
    all_jobs: List[Tuple[str, Dict[str, Any], int, int]] = []
    skipped: List[Tuple[str, str]] = []  # (tuned_dataset_name, eval_dataset_name)

    for tuned_name in tuned_dataset_names:
        allowed_eval = _resolve_allowed_eval_names(
            tuned_dataset_name=tuned_name,
            eval_datasets=eval_datasets,
            eval_groups_cfg=eval_groups_cfg,
            tuned_to_group_cfg=tuned_to_group_cfg,
        )

        for d in eval_datasets:
            eval_name = str(d.get("name", "")).strip()
            if allowed_eval is not None and eval_name not in allowed_eval:
                skipped.append((tuned_name, eval_name))
                continue

            for model_shot in model_shots:
                for prompt_shot in _allowed_prompt_shots_for_model_shot(model_shot):
                    all_jobs.append((tuned_name, d, int(model_shot), int(prompt_shot)))

    print(f"Planned generation jobs: {len(all_jobs)} (format={out_format}, shards={args.job_count})")
    if skipped:
        by_tuned: Dict[str, List[str]] = {}
        for tn, en in skipped:
            by_tuned.setdefault(tn, []).append(en)
        for tn, ens in by_tuned.items():
            print(f"Filtered out eval datasets for {tn}: {', '.join(sorted(set(ens)))}")

    # ------------------------------------------------------------
    # Execute jobs
    # ------------------------------------------------------------
    for job_i, (tuned_name, d, model_shot, prompt_shot) in enumerate(all_jobs):
        if args.job_count > 1 and not _is_assigned(job_i, args.job_index, args.job_count):
            continue

        eval_name = str(d["name"])
        eval_repo = str(d["repo"])
        eval_subset = d.get("subset", None)
        eval_split = str(d.get("split", "test"))

        print(
            f"\n=== Generations: tuned_on={tuned_name} | eval={eval_name} | split={eval_split} | "
            f"model_shot={model_shot} | prompt_shot={prompt_shot} ==="
        )

        # Sharded output path (per eval dataset, per worker/job_index)
        out_dir = out_root / eval_name
        out_path = out_dir / f"generations_shard_{args.job_index}.{out_format}"

        # Load eval dataset split
        ds = _load_hf_split(repo=eval_repo, subset=eval_subset, split=eval_split, token=token)

        prompt_col = f"prompt_{prompt_shot}_shot"
        required_cols = {"prompt_0_shot", "prompt_2_shot", "relation", prompt_col}
        missing = required_cols - set(ds.column_names)
        if missing:
            raise ValueError(
                f"[{eval_name}] Missing columns {sorted(missing)} in {eval_repo} ({eval_subset or 'default'}) split={eval_split}. "
                f"Available: {ds.column_names}"
            )

        # Base fields (kept for every row in long format)
        prompt_0_all = list(ds["prompt_0_shot"])
        prompt_2_all = list(ds["prompt_2_shot"])
        relation_all = [_normalize_label(x) for x in ds["relation"]]
        prompts_for_inference = list(ds[prompt_col])

        # Constrained prompt system string (allowed labels computed per eval dataset)
        allowed_str = allowed_labels_by_eval.get(eval_name, "")
        system_constrained = (
            system_constrained_tmpl.format(allowed_labels=allowed_str) if label_sets_enabled else ""
        )

        # Iterate models one-by-one to keep VRAM usage predictable
        for base_mid in model_ids:
            model_variants: List[Tuple[str, str]] = [(base_mid, base_mid)]
            if include_dapt_models:
                model_variants.append((f"{base_mid}-lit-dapt", f"{base_mid}-lit-dapt"))

            for model_id_for_repo, model_id_for_col in model_variants:
                model_repo = _model_repo_for_run(
                    org_or_user=org_or_user,
                    model_id=model_id_for_repo,
                    tuned_dataset_name=tuned_name,
                    model_shot=model_shot,
                )
                print(f"  - Model: {model_repo}")

                model, tokenizer = _load_model_and_tokenizer(model_repo=model_repo, token=token)

                for idxs in _batched_indices(len(prompts_for_inference), gen_params.batch_size):
                    batch_prompts = [prompts_for_inference[i] for i in idxs]

                    # ---- gen_open
                    open_prompt_texts = [
                        _build_chat_prompt_text(tokenizer=tokenizer, system_prompt=system_open, user_prompt=p)
                        for p in batch_prompts
                    ]
                    open_preds = _generate_relations(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_texts=open_prompt_texts,
                        gen=gen_params,
                    )

                    open_rows: List[Dict[str, Any]] = []
                    for j, pred in enumerate(open_preds):
                        i = idxs[j]
                        open_rows.append(
                            {
                                "eval_dataset_name": str(eval_name),
                                "prompt_0_shot": prompt_0_all[i],
                                "prompt_2_shot": prompt_2_all[i],
                                "relation": relation_all[i],
                                "gen_type": "gen_open",
                                "model_id": str(model_id_for_col),
                                "tuned_dataset_name": str(tuned_name),
                                "model_shot": int(model_shot),
                                "prompt_shot": int(prompt_shot),
                                "generated_relation": pred,
                            }
                        )
                    _append_rows(Path(out_path), open_rows, out_format)

                    # ---- gen_constrained
                    if label_sets_enabled and system_constrained.strip():
                        constrained_prompt_texts = [
                            _build_chat_prompt_text(
                                tokenizer=tokenizer,
                                system_prompt=system_constrained,
                                user_prompt=p,
                            )
                            for p in batch_prompts
                        ]
                        constrained_preds = _generate_relations(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_texts=constrained_prompt_texts,
                            gen=gen_params,
                        )
                    else:
                        constrained_preds = [""] * len(batch_prompts)

                    constrained_rows: List[Dict[str, Any]] = []
                    for j, pred in enumerate(constrained_preds):
                        i = idxs[j]
                        constrained_rows.append(
                            {
                                "eval_dataset_name": str(eval_name),
                                "prompt_0_shot": prompt_0_all[i],
                                "prompt_2_shot": prompt_2_all[i],
                                "relation": relation_all[i],
                                "gen_type": "gen_constrained",
                                "model_id": str(model_id_for_col),
                                "tuned_dataset_name": str(tuned_name),
                                "model_shot": int(model_shot),
                                "prompt_shot": int(prompt_shot),
                                "generated_relation": pred,
                            }
                        )
                    _append_rows(Path(out_path), constrained_rows, out_format)

                # Free GPU memory between model loads
                del model
                del tokenizer
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(f"    appended: {out_path}")

        print(f"✅ Done (appended): {out_path}")


if __name__ == "__main__":
    main()
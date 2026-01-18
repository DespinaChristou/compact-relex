from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import math
import shutil

import yaml
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from src.hf_utils import get_hf_token, push_model_dir


@dataclass(frozen=True)
class DaptCorpus:
    """
    One raw-text corpus used for domain-adaptive pretraining (continued causal LM training).

    DAPT is causal language modeling (next-token prediction). There is no task "accuracy";
    we monitor train loss and (optionally) eval loss / perplexity on a held-out split.

    - split: the training split name (usually "train")
    - eval_split: optional validation split name (usually "validation")
    - text_column: which column contains raw text
    """
    repo: str
    split: str
    text_column: str
    eval_split: Optional[str] = None


def _mark_done(run_dir: Path) -> None:
    (run_dir / "_DONE").write_text("ok", encoding="utf-8")


def _is_done(run_dir: Path) -> bool:
    return (run_dir / "_DONE").exists()


def _load_and_stack_corpora(corpora: List[DaptCorpus], *, which_split: str):
    """
    Load and concatenate corpora into one dataset with a single `text` column.

    which_split:
      - "train": use corpus.split
      - "eval":  use corpus.eval_split (must be set for all corpora)
    """
    datasets = []
    for c in corpora:
        split_name = c.split if which_split == "train" else c.eval_split
        if which_split == "eval" and not split_name:
            raise ValueError(f"Corpus {c.repo} is missing eval_split, but eval was requested.")

        ds = load_dataset(c.repo, split=split_name)

        if c.text_column not in ds.column_names:
            raise ValueError(
                f"Column '{c.text_column}' not found in {c.repo} split={split_name}. "
                f"Available: {ds.column_names}"
            )

        ds = ds.select_columns([c.text_column]).rename_column(c.text_column, "text")
        datasets.append(ds)

    return concatenate_datasets(datasets)


def _group_texts(examples: Dict[str, Any], block_size: int) -> Dict[str, Any]:
    concatenated = {}
    for k in examples.keys():
        concatenated[k] = sum(examples[k], [])
    total_length = len(concatenated["input_ids"])
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    result = {
        k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def run_dapt_job(
        *,
        base_model: str,
        model_id: str,
        output_dir: str,
        corpora: List[DaptCorpus],
        max_seq_length: int,
        learning_rate: float,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        weight_decay: float,
        save_steps: int,
        logging_steps: int,
        hf_token: Optional[str] = None,
        hf_token_env: Optional[str] = None,
        hf_org_or_user: str,
        hf_private: bool,
        eval_strategy: str = "no",
        eval_steps: Optional[int] = None,
        max_steps: Optional[int] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    DAPT (Domain-Adaptive Pretraining):
      - continue causal LM training on domain corpora (LitBank + BookCorpus)
      - (optional) evaluate on held-out corpus splits to track eval_loss/perplexity
      - save and push a private HF checkpoint named:
          {model_id}-lit-dapt

    TensorBoard:
      - logs are written to {run_dir}/tb
      - we copy them into the pushed artifact under `runs/` so HF can render a TB tab.

    If `max_steps` is provided (>0), it overrides `num_train_epochs` so the run has a
    predictable compute budget (useful for cloud cost control).
    """
    if not hf_private:
        raise ValueError("This project requires pushing to private repos; set hf.private=true in config.")

    run_name = f"{model_id}-lit-dapt"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if _is_done(run_dir):
        return {"local_dir": str(run_dir), "hf_repo": f"{hf_org_or_user}/{run_name}"}

    token = get_hf_token(token=hf_token, token_env=hf_token_env)

    tb_dir = run_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(base_model, token=token)

    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    pre = preprocessing or {}
    num_proc = int(pre.get("num_proc", 1))
    tok_bs = int(pre.get("tokenize_batch_size", 1000))
    grp_bs = int(pre.get("group_batch_size", 1000))

    train_raw = _load_and_stack_corpora(corpora, which_split="train")
    if max_train_samples is not None:
        train_raw = train_raw.select(range(min(int(max_train_samples), len(train_raw))))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], return_special_tokens_mask=False, truncation=False)

    train_tok = train_raw.map(
        tokenize_fn,
        batched=True,
        batch_size=tok_bs,
        num_proc=num_proc if num_proc > 1 else None,
        remove_columns=["text"],
        desc=f"[DAPT:{model_id}] tokenizing train",
    )
    train_lm = train_tok.map(
        lambda b: _group_texts(b, max_seq_length),
        batched=True,
        batch_size=grp_bs,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"[DAPT:{model_id}] grouping train",
    )

    eval_lm = None
    if str(eval_strategy).lower() != "no":
        if any(c.eval_split is None for c in corpora):
            raise ValueError("dapt.eval_strategy != 'no' but at least one corpus is missing eval_split.")
        eval_raw = _load_and_stack_corpora(corpora, which_split="eval")
        if max_eval_samples is not None:
            eval_raw = eval_raw.select(range(min(int(max_eval_samples), len(eval_raw))))

        eval_tok = eval_raw.map(
            tokenize_fn,
            batched=True,
            batch_size=tok_bs,
            num_proc=num_proc if num_proc > 1 else None,
            remove_columns=["text"],
            desc=f"[DAPT:{model_id}] tokenizing eval",
        )
        eval_lm = eval_tok.map(
            lambda b: _group_texts(b, max_seq_length),
            batched=True,
            batch_size=grp_bs,
            num_proc=num_proc if num_proc > 1 else None,
            desc=f"[DAPT:{model_id}] grouping eval",
        )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=str(run_dir),
        overwrite_output_dir=False,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=int(max_steps) if (max_steps is not None and int(max_steps) > 0) else -1,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        logging_dir=str(tb_dir),
        report_to=["tensorboard"],
        bf16=True,
        fp16=False,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_lm,
        eval_dataset=eval_lm,
        data_collator=collator,
    )
    trainer.train()

    if eval_lm is not None:
        metrics = trainer.evaluate()
        if "eval_loss" in metrics and metrics["eval_loss"] is not None:
            metrics["eval_perplexity"] = math.exp(metrics["eval_loss"])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    model.save_pretrained(str(run_dir))
    tokenizer.save_pretrained(str(run_dir))

    # Copy TB logs into the pushed artifact directory so HF can render them.
    # HF typically recognizes `runs/` for TensorBoard event files.
    hf_runs_dir = run_dir / "runs"
    hf_runs_dir.mkdir(parents=True, exist_ok=True)
    if tb_dir.exists():
        shutil.copytree(tb_dir, hf_runs_dir / "tb", dirs_exist_ok=True)

    hf_repo = f"{hf_org_or_user}/{run_name}"
    push_model_dir(
        local_dir=str(run_dir),
        repo_id=hf_repo,
        token=token,
        commit_message=f"DAPT: {run_name}",
    )

    _mark_done(run_dir)
    return {"local_dir": str(run_dir), "hf_repo": hf_repo}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    corpora = [DaptCorpus(**c) for c in cfg["dapt"]["corpora"]]
    runs_dir = Path(cfg["paths"]["runs_dir"]) / cfg["dapt"]["output_subdir"]
    runs_dir.mkdir(parents=True, exist_ok=True)

    hf = cfg["hf"]

    for m in cfg["models"]:
        run_dapt_job(
            base_model=m["base_model"],
            model_id=m["id"],
            output_dir=str(runs_dir),
            corpora=corpora,
            max_seq_length=int(cfg["dapt"]["max_seq_length"]),
            learning_rate=float(cfg["dapt"]["learning_rate"]),
            num_train_epochs=int(cfg["dapt"]["num_train_epochs"]),
            per_device_train_batch_size=int(cfg["dapt"]["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(cfg["dapt"]["gradient_accumulation_steps"]),
            warmup_ratio=float(cfg["dapt"]["warmup_ratio"]),
            weight_decay=float(cfg["dapt"]["weight_decay"]),
            save_steps=int(cfg["dapt"]["save_steps"]),
            logging_steps=int(cfg["dapt"]["logging_steps"]),
            hf_token=hf.get("token", None),
            hf_token_env=hf.get("token_env", None),
            hf_org_or_user=cfg["hf"]["org_or_user"],
            hf_private=bool(cfg["hf"]["private"]),
            eval_strategy=str(cfg["dapt"].get("eval_strategy", "no")),
            eval_steps=cfg["dapt"].get("eval_steps", None),
            preprocessing=dict(cfg["dapt"].get("preprocessing", {})),
            max_train_samples=cfg["dapt"].get("max_train_samples", None),
            max_eval_samples=cfg["dapt"].get("max_eval_samples", None),
        )


if __name__ == "__main__":
    main()

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import shutil
import re

import yaml
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.hf_utils import get_hf_token, push_model_dir


def _mark_done(run_dir: Path) -> None:
    (run_dir / "_DONE").write_text("ok", encoding="utf-8")


def _is_done(run_dir: Path) -> bool:
    return (run_dir / "_DONE").exists()


@dataclass(frozen=True)
class SFTExample:
    prompt: str
    relation: str


# -------------------------
# Collator: pads inputs AND labels (labels padded with -100)
# -------------------------
@dataclass
class SFTDataCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        labels = [f["labels"] for f in features]
        inputs = [{k: v for k, v in f.items() if k != "labels"} for f in features]

        batch = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
        )

        max_len = int(batch["input_ids"].shape[1])
        padded_labels = torch.full(
            (len(labels), max_len),
            fill_value=self.label_pad_token_id,
            dtype=torch.long,
        )
        for i, lab in enumerate(labels):
            lab_t = torch.tensor(lab, dtype=torch.long)
            padded_labels[i, : lab_t.numel()] = lab_t

        batch["labels"] = padded_labels
        return batch


def _unravel_fewshot_prompt_to_messages(user_prompt: str) -> list[dict]:
    """
    Convert the dataset `prompt` (which may contain few-shot demonstrations) into a list
    of chat messages suitable for tokenizer.apply_chat_template.

    Dataset prompt format assumptions (based on your examples):
      - 0-shot: single user prompt ending with "Answer:" (empty)
      - 2/5-shot: repeated blocks:
          <example text>
          Answer: <label>

          <example text>
          Answer: <label>

          Now you try:
          <final example text>
          Answer:
    We must turn the demonstration blocks into alternating user/assistant messages.

    Output:
      messages = [
        {"role": "user", "content": "<demo1...>\\nAnswer:"},
        {"role": "assistant", "content": "<label1>"},
        ...
        {"role": "user", "content": "<final...>\\nAnswer:"},
      ]
    """
    text = (user_prompt or "").strip()

    # Fast path: if there is no "Now you try:", treat as a single user turn (0-shot).
    # (Even if there are no demos, this still works.)
    if "Now you try:" not in text:
        return [{"role": "user", "content": text}]

    demo_part, final_part = text.split("Now you try:", 1)
    demo_part = demo_part.strip()
    final_part = final_part.strip()

    messages: list[dict] = []

    # Parse demonstrations: repeated "... Answer: <label>" blocks
    # We capture anything up to "Answer:" as the user content, and the remainder of the line
    # as the assistant label. Blocks are separated by blank lines.
    demo_pattern = re.compile(r"(.*?)(?:\r?\n)Answer:\s*(.*?)(?:\r?\n\r?\n|$)", re.DOTALL)
    for m in demo_pattern.finditer(demo_part):
        user_block = (m.group(1) or "").strip()
        label = (m.group(2) or "").strip()

        # Skip malformed demo blocks (should not happen with your data)
        if not user_block or not label:
            continue

        # Ensure the user message ends with "Answer:" to match the prompting style.
        user_content = user_block
        if not user_content.rstrip().endswith("Answer:"):
            user_content = f"{user_content}\nAnswer:"

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": label})

    # Final user query (no label provided in prompt)
    if final_part:
        messages.append({"role": "user", "content": final_part})
    else:
        # If "Now you try:" is present but nothing after it, fall back to the original prompt.
        messages.append({"role": "user", "content": text})

    return messages


def _build_chat_texts(
        *,
        tokenizer,
        system_prompt: str,
        user_prompt: str,
        assistant_answer: str,
) -> Tuple[str, str]:
    """
    Returns (prompt_text, full_text).

    prompt_text:
      - chat formatted text that ends right before assistant content generation
      - for few-shot prompts, includes N (user, assistant) demonstration turns, then the final user turn

    full_text:
      - prompt_text + assistant_answer (as the final assistant message)
    """
    # Use the model's official chat template when available.
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        # Build messages for the prompt (system + unraveled few-shot user/assistant turns)
        messages_prompt = [
            {"role": "system", "content": system_prompt},
            *(_unravel_fewshot_prompt_to_messages(user_prompt)),
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages_prompt,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Build messages for supervised training target (same, plus final assistant answer)
        messages_full = [
            {"role": "system", "content": system_prompt},
            *(_unravel_fewshot_prompt_to_messages(user_prompt)),
            {"role": "assistant", "content": assistant_answer},
        ]

        full_text = tokenizer.apply_chat_template(
            messages_full,
            tokenize=False,
            add_generation_prompt=False,
        )
        return prompt_text, full_text

    # Fallback: generic instruction format (should rarely be used for your listed models).
    prompt_text = f"{system_prompt}\n\n{user_prompt}\n"
    full_text = f"{prompt_text}{assistant_answer}\n"
    return prompt_text, full_text


def _tokenize_sft_row(
        row: Dict[str, Any],
        *,
        tokenizer,
        system_prompt: str,
        max_length: int,
) -> Dict[str, Any]:
    """
    Create input_ids/labels for instruction tuning:
    - input is the (chat-wrapped) prompt
    - target is ONLY the relation label
    - labels mask out prompt tokens with -100
    """
    if "prompt" not in row or "relation" not in row:
        raise ValueError("Expected columns: prompt, relation")

    assistant_answer = str(row["relation"]).strip()

    prompt_text, full_text = _build_chat_texts(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt=str(row["prompt"]),
        assistant_answer=assistant_answer,
    )

    prompt_enc = tokenizer(
        prompt_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    full_enc = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Ensure same length
    labels = labels[: len(input_ids)]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _tokenize_sft_batch(
        batch: Dict[str, Any],
        *,
        tokenizer,
        system_prompt: str,
        max_length: int,
) -> Dict[str, Any]:
    """
    Batched version used for speed.

    Input batch schema (from HF Datasets):
      batch["prompt"]   -> List[str]
      batch["relation"] -> List[str]

    We build two parallel text lists:
      - prompt_texts: chat-formatted prompt with generation prompt
      - full_texts:   chat-formatted prompt + assistant answer

    Then tokenize each list in one tokenizer call (faster than per-row),
    and build labels by masking the prompt tokens with -100.

    Returns:
      {"input_ids": List[List[int]], "attention_mask": List[List[int]], "labels": List[List[int]]}
    """
    prompts = batch.get("prompt", None)
    relations = batch.get("relation", None)
    if prompts is None or relations is None:
        raise ValueError("Expected batched columns: prompt, relation")

    prompt_texts: list[str] = []
    full_texts: list[str] = []

    for p, r in zip(prompts, relations):
        assistant_answer = str(r).strip()
        prompt_text, full_text = _build_chat_texts(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=str(p),
            assistant_answer=assistant_answer,
        )
        prompt_texts.append(prompt_text)
        full_texts.append(full_text)

    prompt_enc = tokenizer(
        prompt_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    full_enc = tokenizer(
        full_texts,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        padding=False,
    )

    input_ids_list = full_enc["input_ids"]
    attention_mask_list = full_enc["attention_mask"]

    labels_list: list[list[int]] = []
    for prompt_ids, full_ids in zip(prompt_enc["input_ids"], input_ids_list):
        prompt_len = len(prompt_ids)
        labels = ([-100] * prompt_len) + full_ids[prompt_len:]
        labels = labels[: len(full_ids)]
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def run_finetune_job(
        *,
        base_model_or_path: str,
        model_id: str,
        dataset_repo: str,
        dataset_name: str,
        shot: int,
        output_dir: str,
        max_seq_length: int,
        learning_rate: float,
        num_train_epochs: int,
        per_device_train_batch_size: int,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        weight_decay: float,
        save_steps: int,
        eval_steps: int,
        logging_steps: int,
        system_prompt: str,
        hf_token: Optional[str] = None,
        hf_token_env: Optional[str] = None,
        hf_org_or_user: str,
        hf_private: bool,
        qlora_enabled: bool = False,
        qlora_config: Optional[Dict[str, Any]] = None,
        preprocessing: Optional[Dict[str, Any]] = None,
        max_train_samples: Optional[int] = None,
        max_eval_samples: Optional[int] = None,
) -> Dict[str, str]:
    """
    Fine-tunes one (model, dataset, shot) configuration using instruction tuning.

    Data format (aggregated datasets):
      - input column:  `prompt`   (already contains N-shot demonstrations)
      - target column: `relation` (label string; can be multi-token)

    Training objective:
      - RE-as-generation where we supervise ONLY the assistant answer tokens
        (prompt tokens are masked out with -100)

    QLoRA behavior (when enabled):
      1) Load base model in 4-bit (NF4) and train LoRA adapters.
      2) Save adapter artifacts to:   {run_dir}/adapter
      3) Optionally merge adapters into full weights and save to: {run_dir}/merged
      4) Push merged (preferred) checkpoint to HF private repo named:
           {model_id}-{dataset_name}-{N}-shot

    TensorBoard:
      - We log to {run_dir}/tb
      - We copy tb logs into the pushed folder under `runs/` so HF can show the TensorBoard tab.

    Preprocessing:
      - HF Datasets `.map()` can be parallelized via `num_proc` and larger `batch_size`.
        This can significantly speed up tokenization for large splits.
    """

    if not hf_private:
        raise ValueError("This project requires pushing to private repos; set hf.private=true in config.")

    run_name = f"{model_id}-{dataset_name}-{shot}-shot"
    run_dir = Path(output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    if _is_done(run_dir):
        return {"local_dir": str(run_dir), "hf_repo": f"{hf_org_or_user}/{run_name}"}

    token = get_hf_token(token=hf_token, token_env=hf_token_env)

    # Tokenizer is shared for both adapter-training and merged saving.
    tokenizer = AutoTokenizer.from_pretrained(base_model_or_path, use_fast=True, token=token)
    # Trainer's default collator pads batches, so we must define one.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"[SFT:{model_id}] Tokenizer has no pad_token and no eos_token; cannot set padding token safely."
            )
        tokenizer.pad_token = tokenizer.eos_token

    adapter_out_dir = run_dir / "adapter"
    merged_out_dir = run_dir / "merged"
    tb_dir = run_dir / "tb"
    adapter_out_dir.mkdir(parents=True, exist_ok=True)
    merged_out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Model load: QLoRA (4-bit + LoRA) or full fine-tuning
    # -------------------------
    if qlora_enabled:
        from transformers import BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if not qlora_config:
            raise ValueError("qlora_enabled=True but qlora_config is missing.")

        import torch

        compute_dtype_str = str(qlora_config.get("bnb_4bit_compute_dtype", "bfloat16")).lower()
        compute_dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(compute_dtype_str)
        if compute_dtype is None:
            raise ValueError(f"Unsupported bnb_4bit_compute_dtype: {compute_dtype_str}")

        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=bool(qlora_config.get("load_in_4bit", True)),
            bnb_4bit_quant_type=str(qlora_config.get("bnb_4bit_quant_type", "nf4")),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=bool(qlora_config.get("bnb_4bit_use_double_quant", True)),
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_or_path,
            token=token,
            quantization_config=bnb_cfg,
            device_map="auto",
            dtype=compute_dtype,
        )

        # Important for QLoRA stability + memory (activations):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Resolve lora_r / lora_alpha:
        # - if per_model[model_id] exists and has the key, use it
        # - else fall back to default qlora_config values
        per_model = qlora_config.get("per_model", {}) if isinstance(qlora_config, dict) else {}
        overrides = per_model.get(model_id, {}) if isinstance(per_model, dict) else {}
        if not isinstance(overrides, dict):
            overrides = {}

        lora_r = int(overrides.get("lora_r", qlora_config.get("lora_r", 64)))
        lora_alpha = int(overrides.get("lora_alpha", qlora_config.get("lora_alpha", 128)))

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=float(qlora_config.get("lora_dropout", 0.05)),
            target_modules=list(qlora_config.get("target_modules", [])),
            bias=str(qlora_config.get("bias", "none")),
            task_type=str(qlora_config.get("task_type", "CAUSAL_LM")),
        )

        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_or_path, token=token)
        model.gradient_checkpointing_enable()

    # Disable KV cache during training (important with checkpointing).
    model.config.use_cache = False

    # -------------------------
    # Dataset loading and tokenization
    # -------------------------
    pre = preprocessing or {}
    num_proc = int(pre.get("num_proc", 1))
    map_bs = int(pre.get("map_batch_size", 512))
    sample_seed = int(pre.get("sample_seed", 42))

    train_split = f"train_{shot}"
    eval_split = f"eval_{shot}"

    # load only 100 random samples from the dataset for training
    train_ds = load_dataset(dataset_repo, split=train_split)
    eval_ds = load_dataset(dataset_repo, split=eval_split)

    if max_train_samples is not None:
        train_ds = train_ds.shuffle(seed=sample_seed).select(
            range(min(int(max_train_samples), len(train_ds)))
        )
    if max_eval_samples is not None:
        eval_ds = eval_ds.shuffle(seed=sample_seed + 1).select(
            range(min(int(max_eval_samples), len(eval_ds)))
        )

    def map_fn(b):
        return _tokenize_sft_batch(
            b,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=max_seq_length,
        )

    train_tok = train_ds.map(
        map_fn,
        remove_columns=train_ds.column_names,
        batched=True,
        batch_size=map_bs,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"[SFT:{model_id}] tokenizing train ({dataset_name}, {shot}-shot)",
    )
    eval_tok = eval_ds.map(
        map_fn,
        remove_columns=eval_ds.column_names,
        batched=True,
        batch_size=map_bs,
        num_proc=num_proc if num_proc > 1 else None,
        desc=f"[SFT:{model_id}] tokenizing eval ({dataset_name}, {shot}-shot)",
    )

    # -------------------------
    # Trainer configuration
    # -------------------------
    extra_args: Dict[str, Any] = {}
    if qlora_enabled and qlora_config:
        extra_args["optim"] = str(qlora_config.get("optim", "paged_adamw_8bit"))

    args = TrainingArguments(
        output_dir=str(run_dir),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        save_steps=save_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        logging_steps=logging_steps,
        save_total_limit=1,
        logging_dir=str(tb_dir),
        report_to=["tensorboard"],
        bf16=False,
        fp16=True,
        **extra_args,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        processing_class=tokenizer,
        data_collator=SFTDataCollator(tokenizer=tokenizer),
    )
    trainer.train()

    # -------------------------
    # Save artifacts locally
    # -------------------------
    if qlora_enabled:
        # Save adapter artifacts
        model.save_pretrained(str(adapter_out_dir))
        tokenizer.save_pretrained(str(adapter_out_dir))

        # Optionally merge to full weights and save
        merge_full = bool(qlora_config.get("merge_full_weights", False))
        if merge_full:
            from peft import PeftModel
            import torch

            merged_dtype_str = str(qlora_config.get("merged_torch_dtype", "bfloat16")).lower()
            merged_dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }.get(merged_dtype_str)
            if merged_dtype is None:
                raise ValueError(f"Unsupported merged_torch_dtype: {merged_dtype_str}")

            # Reload base in bf16 (not 4-bit), attach adapter, merge, and save.
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_or_path,
                token=token,
                torch_dtype=merged_dtype,
                device_map="auto",
            )
            merged = PeftModel.from_pretrained(base_model, str(adapter_out_dir))
            merged = merged.merge_and_unload()

            merged.save_pretrained(str(merged_out_dir))
            tokenizer.save_pretrained(str(merged_out_dir))
    else:
        # Full fine-tune path: merged_out_dir is the actual trained model.
        model.save_pretrained(str(merged_out_dir))
        tokenizer.save_pretrained(str(merged_out_dir))

    # Copy TB logs into the pushed artifact directory so HF can render them.
    # HF typically recognizes `runs/` for TensorBoard event files.
    to_push = merged_out_dir if (merged_out_dir / "config.json").exists() else adapter_out_dir
    hf_runs_dir = to_push / "runs"
    hf_runs_dir.mkdir(parents=True, exist_ok=True)
    if tb_dir.exists():
        # Copy the entire tb directory contents into `runs/`.
        shutil.copytree(tb_dir, hf_runs_dir / "tb", dirs_exist_ok=True)

    # -------------------------
    # Push to Hugging Face (private) using the required naming scheme
    # -------------------------
    hf_repo = f"{hf_org_or_user}/{run_name}"
    push_model_dir(
        local_dir=str(to_push),
        repo_id=hf_repo,
        token=token,
        commit_message=f"SFT: {run_name}",
    )

    _mark_done(run_dir)
    return {"local_dir": str(run_dir), "hf_repo": hf_repo}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--base_model_or_path", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--dataset_repo", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--shot", type=int, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    hf = cfg["hf"]

    out_dir = Path(cfg["paths"]["runs_dir"]) / cfg["finetune"]["output_subdir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    qlora_cfg = dict(cfg.get("finetune", {}).get("qlora", {}))
    qlora_enabled = bool(qlora_cfg.get("enabled", False))

    # Allow per-model QLoRA overrides (e.g., smaller r/alpha for micro models).
    if qlora_enabled:
        per_model = dict(qlora_cfg.get("per_model", {}))
        overrides = per_model.get(args.model_id, None)
        if isinstance(overrides, dict):
            for k in ("lora_r", "lora_alpha"):
                if k in overrides and overrides[k] is not None:
                    qlora_cfg[k] = overrides[k]

    run_finetune_job(
        base_model_or_path=args.base_model_or_path,
        model_id=args.model_id,
        dataset_repo=args.dataset_repo,
        dataset_name=args.dataset_name,
        shot=args.shot,
        output_dir=str(out_dir),
        max_seq_length=int(cfg["finetune"]["max_seq_length"]),
        learning_rate=float(cfg["finetune"]["learning_rate"]),
        num_train_epochs=int(cfg["finetune"]["num_train_epochs"]),
        per_device_train_batch_size=int(cfg["finetune"]["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(cfg["finetune"]["gradient_accumulation_steps"]),
        warmup_ratio=float(cfg["finetune"]["warmup_ratio"]),
        weight_decay=float(cfg["finetune"]["weight_decay"]),
        save_steps=int(cfg["finetune"]["save_steps"]),
        eval_steps=int(cfg["finetune"]["eval_steps"]),
        logging_steps=int(cfg["finetune"]["logging_steps"]),
        system_prompt=str(cfg["prompting"]["system_prompt"]),
        hf_token=hf.get("token", None),
        hf_token_env=hf.get("token_env", None),
        hf_org_or_user=cfg["hf"]["org_or_user"],
        hf_private=bool(cfg["hf"]["private"]),
        qlora_enabled=qlora_enabled,
        qlora_config=qlora_cfg if qlora_enabled else None,
    )


if __name__ == "__main__":
    main()

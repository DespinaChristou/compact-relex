from pathlib import Path
from typing import Dict, List, Tuple

import yaml

from src.dapt import DaptCorpus, run_dapt_job
from src.train import run_finetune_job


def _dapt_output_model_ref(hf_org_or_user: str, model_id: str) -> str:
    """
    After DAPT, the model is pushed to:
      {org}/{model_id}-lit-dapt
    """
    return f"{hf_org_or_user}/{model_id}-lit-dapt"


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    hf = cfg["hf"]
    dapt_cfg = cfg["dapt"]
    ft_cfg = cfg["finetune"]

    # -------------------------
    # 1) DAPT: once per base model
    # -------------------------
    if bool(dapt_cfg.get("enabled", False)):
        corpora = [DaptCorpus(**c) for c in dapt_cfg["corpora"]]
        dapt_out_dir = runs_dir / dapt_cfg["output_subdir"]
        dapt_out_dir.mkdir(parents=True, exist_ok=True)

        for m in cfg["models"]:
            run_dapt_job(
                base_model=m["base_model"],
                model_id=m["id"],
                output_dir=str(dapt_out_dir),
                corpora=corpora,
                max_seq_length=int(dapt_cfg["max_seq_length"]),
                learning_rate=float(dapt_cfg["learning_rate"]),
                num_train_epochs=int(dapt_cfg["num_train_epochs"]),
                per_device_train_batch_size=int(dapt_cfg["per_device_train_batch_size"]),
                gradient_accumulation_steps=int(dapt_cfg["gradient_accumulation_steps"]),
                warmup_ratio=float(dapt_cfg["warmup_ratio"]),
                weight_decay=float(dapt_cfg["weight_decay"]),
                save_steps=int(dapt_cfg["save_steps"]),
                logging_steps=int(dapt_cfg["logging_steps"]),
                hf_token_env=hf["token_env"],
                hf_org_or_user=hf["org_or_user"],
                hf_private=bool(hf["private"]),
            )

    # -------------------------
    # 2) Finetuning: for base + (optionally) dapt models, all datasets, all shots
    # -------------------------
    if not bool(ft_cfg.get("enabled", True)):
        return

    ft_out_dir = runs_dir / ft_cfg["output_subdir"]
    ft_out_dir.mkdir(parents=True, exist_ok=True)

    shots: List[int] = list(cfg["shots"])
    datasets: Dict[str, str] = dict(ft_cfg["datasets"])

    only_models = ft_cfg.get("only_models", None)
    skip_models = ft_cfg.get("skip_models", None)
    include_dapt_models = bool(ft_cfg.get("include_dapt_models", True))

    only_set = set(only_models) if isinstance(only_models, list) else None
    skip_set = set(skip_models) if isinstance(skip_models, list) else set()

    # Build the list of starting points:
    # - base models
    # - dapt models (if enabled and requested)
    starting_points: List[Tuple[str, str]] = []  # (model_ref, model_id_for_naming)

    for m in cfg["models"]:
        base_id = m["id"]

        if only_set is not None and base_id not in only_set:
            continue
        if base_id in skip_set:
            continue

        starting_points.append((m["base_model"], base_id))

        if bool(dapt_cfg.get("enabled", False)) and include_dapt_models:
            starting_points.append((_dapt_output_model_ref(hf["org_or_user"], base_id), f"{base_id}-lit-dapt"))

    for model_ref, model_id in starting_points:
        for dataset_name, dataset_repo in datasets.items():
            for shot in shots:
                run_finetune_job(
                    base_model_or_path=model_ref,
                    model_id=model_id,
                    dataset_repo=dataset_repo,
                    dataset_name=dataset_name,
                    shot=int(shot),
                    output_dir=str(ft_out_dir),
                    max_seq_length=int(ft_cfg["max_seq_length"]),
                    learning_rate=float(ft_cfg["learning_rate"]),
                    num_train_epochs=int(ft_cfg["num_train_epochs"]),
                    per_device_train_batch_size=int(ft_cfg["per_device_train_batch_size"]),
                    gradient_accumulation_steps=int(ft_cfg["gradient_accumulation_steps"]),
                    warmup_ratio=float(ft_cfg["warmup_ratio"]),
                    weight_decay=float(ft_cfg["weight_decay"]),
                    save_steps=int(ft_cfg["save_steps"]),
                    eval_steps=int(ft_cfg["eval_steps"]),
                    logging_steps=int(ft_cfg["logging_steps"]),
                    system_prompt=str(cfg["prompting"]["system_prompt"]),
                    hf_token_env=hf["token_env"],
                    hf_org_or_user=hf["org_or_user"],
                    hf_private=bool(hf["private"]),
                    qlora_enabled=bool(ft_cfg.get("qlora", {}).get("enabled", False)),
                    qlora_config=dict(ft_cfg.get("qlora", {})),
                )


if __name__ == "__main__":
    main()

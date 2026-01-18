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


def _is_assigned(global_index: int, job_index: int, job_count: int) -> bool:
    """
    Deterministic sharding: each worker runs items where (i % job_count == job_index).
    """
    return (global_index % job_count) == job_index


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    # Parallel sharding controls (run 4 copies of this script with job_count=4 and job_index in [0..3])
    parser.add_argument("--job_index", type=int, default=0, help="Shard index in [0..job_count-1]")
    parser.add_argument("--job_count", type=int, default=1, help="Number of shards / parallel workers")
    parser.add_argument(
        "--stage",
        choices=["all", "dapt", "finetune"],
        default="all",
        help="Which stage to run.",
    )
    args = parser.parse_args()

    if args.job_count < 1:
        raise ValueError("--job_count must be >= 1")
    if not (0 <= args.job_index < args.job_count):
        raise ValueError("--job_index must be in [0..job_count-1]")

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    hf = cfg["hf"]
    dapt_cfg = cfg["dapt"]
    ft_cfg = cfg["finetune"]

    # -------------------------
    # 1) DAPT: once per base model
    # -------------------------
    if args.stage in ("all", "dapt") and bool(dapt_cfg.get("enabled", False)):
        if args.job_count == 1 or args.job_index == 0:
            corpora = [DaptCorpus(**c) for c in dapt_cfg["corpora"]]
            dapt_out_dir = runs_dir / dapt_cfg["output_subdir"]
            dapt_out_dir.mkdir(parents=True, exist_ok=True)

            for mi, m in enumerate(cfg["models"]):
                if args.job_count > 1 and not _is_assigned(mi, args.job_index, args.job_count):
                    continue

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
                    hf_token=hf.get("token", None),
                    hf_token_env=hf.get("token_env", None),
                    hf_org_or_user=hf["org_or_user"],
                    hf_private=bool(hf["private"]),
                )

    # -------------------------
    # 2) Finetuning: for base + (optionally) dapt models, all datasets, all shots
    # -------------------------
    if args.stage in ("all", "finetune"):
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

            if include_dapt_models:
                starting_points.append((_dapt_output_model_ref(hf["org_or_user"], base_id), f"{base_id}-lit-dapt"))

        # Expand the full job list deterministically, then shard by index.
        all_jobs: List[Tuple[str, str, str, str, int]] = []
        for model_ref, model_id in starting_points:
            for dataset_name, dataset_repo in datasets.items():
                for shot in shots:
                    all_jobs.append((model_ref, model_id, dataset_name, dataset_repo, int(shot)))

        for i, (model_ref, model_id, dataset_name, dataset_repo, shot) in enumerate(all_jobs):
            if args.job_count > 1 and not _is_assigned(i, args.job_index, args.job_count):
                continue

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
                hf_token=hf.get("token", None),
                hf_token_env=hf.get("token_env", None),
                hf_org_or_user=hf["org_or_user"],
                hf_private=bool(hf["private"]),
                qlora_enabled=bool(ft_cfg.get("qlora", {}).get("enabled", False)),
                qlora_config=dict(ft_cfg.get("qlora", {})),
            )


if __name__ == "__main__":
    # import sys
    # sys.argv = ["run_all", "--config", "../configs/experiments.yaml"]
    main()

#!/usr/bin/env python3
"""
Build GenTune / LitTune / MixTune aggregated datasets from private HF sources.

Inputs (from configs/datasets.yaml):
  - sources: datasets that contain columns prompt_0_shot/prompt_2_shot/prompt_5_shot + relation
  - splits: train/validation/test
  - shot prompts stored as columns (not separate splits)

Outputs:
  - HF datasets with split-per-shot naming:
      train_0, eval_0, test_0
      train_2, eval_2, test_2
      train_5, eval_5, test_5
  - pushed to private repos (e.g., Despina/re_gentune, etc.)

Usage:
  export HF_TOKEN=...
  python src/build_mixtures.py --config configs/datasets.yaml
"""

import argparse
import os
from typing import Dict, Any, List, Optional, Tuple

import yaml
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Please export HF_TOKEN=... (with access to private repos).")
    return token


def _load_split(
        repo: str,
        subset: Optional[str],
        split: str,
        token: str,
) -> Dataset:
    """
    Loads a dataset split from HF.
    subset corresponds to dataset config name; use "default" or None.
    """
    subset_arg = None if (subset is None or subset == "default") else subset
    return load_dataset(repo, subset_arg, split=split, token=token)


def _select_and_reshape(
        ds: Dataset,
        dataset_name: str,
        prompt_col: str,
        label_col: str,
        add_dataset_name: bool,
        dataset_name_field: str,
) -> Dataset:
    """
    Produces a unified dataset with columns:
      - prompt
      - relation
      - dataset (optional)
    """
    required_cols = set([prompt_col, label_col])
    missing = required_cols - set(ds.column_names)
    if missing:
        raise ValueError(f"[{dataset_name}] Missing required columns: {missing}. Available: {ds.column_names}")

    keep_cols = [prompt_col, label_col]
    ds2 = ds.remove_columns([c for c in ds.column_names if c not in keep_cols])

    # Correct text of prompt_0_shot if prompt_col=='prompt_0_shot' add "\nAnswer: "
    if prompt_col.endswith("0_shot"):
        ds2 = ds2.map(lambda x: {prompt_col: x[prompt_col].strip() + "\nAnswer: "})

    # Rename
    ds2 = ds2.rename_column(prompt_col, "prompt")
    if label_col != "relation":
        ds2 = ds2.rename_column(label_col, "relation")

    # Add dataset name
    if add_dataset_name:
        ds2 = ds2.map(lambda _: {dataset_name_field: dataset_name})

    return ds2


def _cap_dataset(ds: Dataset, cap: Optional[int], seed: int) -> Dataset:
    if cap is None:
        return ds
    if len(ds) <= cap:
        return ds
    # Shuffle then select cap
    ds = ds.shuffle(seed=seed)
    return ds.select(range(cap))


def _domain_balanced_mix(general: Dataset, literature: Dataset, seed: int) -> Dataset:
    """
    Equal number of examples from general and literature.
    """
    n = min(len(general), len(literature))
    g = general.shuffle(seed=seed).select(range(n))
    l = literature.shuffle(seed=seed + 1).select(range(n))
    mixed = concatenate_datasets([g, l]).shuffle(seed=seed + 2)
    return mixed


def _dataset_balanced_mix(datasets: List[Dataset], seed: int) -> Dataset:
    """
    Equal number of examples from each component dataset.
    """
    if not datasets:
        raise ValueError("No datasets provided to dataset_balanced_mix.")
    n = min(len(d) for d in datasets)
    parts = []
    for i, d in enumerate(datasets):
        parts.append(d.shuffle(seed=seed + i).select(range(n)))
    return concatenate_datasets(parts).shuffle(seed=seed + 999)


def build_aggregate_for_one_shot(
        cfg: Dict[str, Any],
        token: str,
        aggregate_key: str,
        shot: int,
) -> DatasetDict:
    """
    Returns a DatasetDict containing three splits for a given shot:
      train_{shot}, eval_{shot}, test_{shot}
    """
    sources = cfg["sources"]
    aggregates = cfg["aggregates"]
    shots_cfg = cfg["shots"]
    output_splits = cfg["output_splits"]
    add_meta = cfg.get("add_metadata", {})
    add_dataset_name = bool(add_meta.get("add_dataset_name_field", True))
    dataset_name_field = add_meta.get("dataset_name_field", "dataset")

    split_map = {
        "train": "train",
        "eval": "validation",  # your sources use 'validation'
        "test": "test",
    }

    # which prompt column to use for this shot
    prompt_key = shots_cfg["prompt_column_by_shot"][str(shot)]  # e.g. "prompt_0"
    # source config maps that to actual column name, per dataset
    # We'll look it up from each dataset's `columns` mapping.

    agg = aggregates[aggregate_key]

    def load_group(dataset_keys: List[str], split_alias: str) -> List[Dataset]:
        ds_list: List[Dataset] = []
        for dk in dataset_keys:
            s = sources[dk]
            repo = s["repo"]
            subset = s.get("subset", "default")
            split_name = s["splits"][split_alias]  # expects train/eval/test mapping already in yaml
            cols = s["columns"]
            prompt_col = cols[prompt_key]  # e.g., prompt_0 -> prompt_0_shot
            label_col = cols["label"]

            raw = _load_split(repo=repo, subset=subset, split=split_name, token=token)
            reshaped = _select_and_reshape(
                raw,
                dataset_name=dk,
                prompt_col=prompt_col,
                label_col=label_col,
                add_dataset_name=add_dataset_name,
                dataset_name_field=dataset_name_field,
            )

            cap = agg.get("cap_per_source_dataset", None) or aggregates.get("mixtune", {}).get("cap_per_source_dataset",
                                                                                               None)
            seed = agg.get("shuffle_seed", 42)
            reshaped = _cap_dataset(reshaped, cap=cap, seed=seed)

            ds_list.append(reshaped)
        return ds_list

    def concat_group(ds_list: List[Dataset]) -> Dataset:
        if not ds_list:
            raise ValueError("No datasets to concatenate.")
        if len(ds_list) == 1:
            return ds_list[0]
        return concatenate_datasets(ds_list)

    seed = agg.get("shuffle_seed", 42)

    # Build for each split (train/eval/test)
    out_dd = DatasetDict()
    for split_alias in ["train", "eval", "test"]:
        out_split_name = output_splits[split_alias].format(shot=shot)

        if aggregate_key in ["gentune", "littune"]:
            dataset_keys = agg["includes"]
            ds_list = load_group(dataset_keys, split_alias=split_alias)
            combined = concat_group(ds_list).shuffle(seed=seed)
            out_dd[out_split_name] = combined

        elif aggregate_key == "mixtune":
            mode = agg.get("mode", "domain_balanced")
            gen_keys = agg["includes_general"]
            lit_keys = agg["includes_literature"]

            gen_list = load_group(gen_keys, split_alias=split_alias)
            lit_list = load_group(lit_keys, split_alias=split_alias)
            gen_combined = concat_group(gen_list).shuffle(seed=seed)
            lit_combined = concat_group(lit_list).shuffle(seed=seed + 1)

            if mode == "domain_balanced":
                mixed = _domain_balanced_mix(gen_combined, lit_combined, seed=seed)
            elif mode == "dataset_balanced":
                mixed = _dataset_balanced_mix(gen_list + lit_list, seed=seed)
            else:
                raise ValueError(f"Unknown mixtune mode: {mode}")

            out_dd[out_split_name] = mixed
        else:
            raise ValueError(f"Unknown aggregate key: {aggregate_key}")

    return out_dd


def merge_datasetdicts(dd_list: List[DatasetDict]) -> DatasetDict:
    """
    Merge multiple DatasetDicts with disjoint split names into one DatasetDict.
    """
    out = DatasetDict()
    for dd in dd_list:
        for split_name, ds in dd.items():
            if split_name in out:
                raise ValueError(f"Duplicate split name while merging: {split_name}")
            out[split_name] = ds
    return out


def push_datasetdict(
        dd: DatasetDict,
        repo_id: str,
        private: bool,
        token: str,
) -> None:
    """
    Push to HF. datasets.push_to_hub supports private via token permissions
    and repo settings. If repo doesn't exist, it will be created.
    """
    # datasets uses huggingface_hub under the hood
    dd.push_to_hub(repo_id, private=private, token=token)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to configs/datasets.yaml")
    ap.add_argument("--only", default=None, help="Optional: build only one aggregate: gentune|littune|mixtune")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    token = get_token()

    aggregates_cfg = cfg["aggregates"]
    shot_values = cfg["shots"]["shot_values"]

    aggregate_keys = ["gentune", "littune", "mixtune"]
    if args.only:
        if args.only not in aggregate_keys:
            raise ValueError(f"--only must be one of {aggregate_keys}")
        aggregate_keys = [args.only]

    for agg_key in aggregate_keys:
        out_repo = aggregates_cfg[agg_key]["output_repo"]
        private = bool(aggregates_cfg[agg_key].get("private", cfg["hf"].get("private", True)))

        print(f"\n=== Building aggregate: {agg_key} -> {out_repo} (private={private}) ===")

        dd_parts: List[DatasetDict] = []
        for shot in shot_values:
            print(f"  - Building shot={shot}")
            dd = build_aggregate_for_one_shot(cfg, token=token, aggregate_key=agg_key, shot=int(shot))
            dd_parts.append(dd)

        final_dd = merge_datasetdicts(dd_parts)

        # Quick sanity print
        print("  Splits created:", list(final_dd.keys()))
        for sname in sorted(final_dd.keys()):
            print(f"    {sname}: {len(final_dd[sname])} rows")

        print(f"  Pushing to hub: {out_repo}")
        push_datasetdict(final_dd, repo_id=out_repo, private=private, token=token)
        print(f"  ✅ Done: {out_repo}")


if __name__ == "__main__":
    main()

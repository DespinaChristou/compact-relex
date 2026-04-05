# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for a paper on **relation extraction (RE) with small/micro LLMs** across general and literary domains. Config-driven pipeline that aggregates RE datasets from Hugging Face, optionally runs Domain-Adaptive Pretraining (DAPT), fine-tunes models with QLoRA, and generates/evaluates predictions.

## Key Commands
[generations.csv](runs/generations/gids/generations.csv)
All scripts use `from src...` imports, so **run as modules from the repo root**:

```bash
# Install dependencies
pip install -r requirements.txt

# Build aggregated datasets (one-time)
python src/build_mixtures.py --config configs/datasets.yaml

# Run DAPT, finetuning, or both
python -m src.run_all --config configs/experiments.yaml --stage dapt
python -m src.run_all --config configs/experiments.yaml --stage finetune
python -m src.run_all --config configs/experiments.yaml --stage all

# Multi-GPU: shard by job_index/job_count (one process per GPU)
CUDA_VISIBLE_DEVICES=0 python -m src.run_all --config configs/experiments.yaml --stage finetune --job_count 4 --job_index 0

# Generate predictions from fine-tuned checkpoints
python -m src.run_generations --config configs/generations.yaml

# Merge generation shards into final CSVs
python scripts/merge_generations.py --generations_dir runs/generations --shards 0 1 2 3
```

**Environment requirement:** `HF_TOKEN` must be set with access to private `Despina/*` datasets on Hugging Face.

## Architecture

### Pipeline stages (in order)

1. **Dataset aggregation** (`src/build_mixtures.py`) — Combines source HF datasets into GenTune (general), LitTune (literary), MixTune (balanced mix). One-time step.
2. **DAPT** (`src/dapt.py`) — Optional continued causal LM pretraining on literary corpora (LitBank, BookCorpus). Produces `{model_id}-lit-dapt` checkpoints.
3. **Fine-tuning** (`src/train.py`) — QLoRA instruction tuning for RE. Each run = (model x dataset x shot). Produces `{model_id}-{dataset}-{shot}-shot`.
4. **Generation** (`src/run_generations.py`) — Runs fine-tuned checkpoints on original eval datasets. Produces both "open" (free-form) and "constrained" (label-set restricted) generations.
5. **Merge** (`scripts/merge_generations.py`) — Merges per-worker shard files into final CSV/parquet per eval dataset.

### Orchestration

- `src/run_all.py` — Main entry point for DAPT + finetuning. Reads `configs/experiments.yaml`, builds the full job list (model x dataset x shot), and supports deterministic sharding via `--job_index`/`--job_count` for multi-GPU parallelism.
- `src/run_generations.py` — Same sharding pattern for inference. Reads `configs/generations.yaml`.

### Config files (all YAML)

- `configs/datasets.yaml` — Dataset registry: source HF repos, domain tags, column mappings, aggregation rules.
- `configs/experiments.yaml` — Models list, DAPT hyperparams, finetune hyperparams, QLoRA settings (with per-model overrides), shot values, HF publishing config.
- `configs/generations.yaml` — Eval datasets, generation params, prompt templates, label-set constraints, eval group restrictions (which tuned models eval on which datasets), output format.

### Key design decisions

- **Completeness tracking**: Each run writes a `_DONE` sentinel file; already-completed runs are skipped on re-execution.
- **Few-shot prompt unraveling**: `train.py:_unravel_fewshot_prompt_to_messages()` converts dataset few-shot prompts into alternating user/assistant chat turns for chat-template models.
- **Eval group restrictions**: `generations.yaml` controls which eval datasets each tuned family runs on (e.g., `re_littune` models only eval on literary datasets; `re_mixtune` on all).
- **Two generation types**: "gen_open" uses a generic system prompt; "gen_constrained" injects allowed label sets per eval dataset into the system prompt.
- All checkpoints and datasets are pushed to private HF repos under the `Despina/` namespace.

### Utilities

- `src/hf_utils.py` — HF token resolution (config value or env var) and model directory upload.
- `src/eval.py` — Evaluation metrics (currently a stub/minimal).

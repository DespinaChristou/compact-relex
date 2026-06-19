# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research codebase for a paper on **relation extraction (RE) with small/micro LLMs** across general and literary domains. Config-driven pipeline that aggregates RE datasets from Hugging Face, optionally runs Domain-Adaptive Pretraining (DAPT), fine-tunes models with QLoRA, and generates/evaluates predictions.

## Key Commands

All scripts use `from src...` imports, so **run as modules from the repo root** (`src/` entry points use `python -m`; `scripts/` analysis tools insert the repo root onto `sys.path`, so run them as `python scripts/<name>.py`):

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

# ── Evaluation & analysis (after generations exist) ──
# Compute all metrics → runs/evaluation/per_dataset_metrics.csv + per-class breakdowns
python scripts/run_evaluation.py --config configs/eval.yaml
python scripts/run_evaluation.py --config configs/eval.yaml --datasets conll04 biographical  # subset

# Build paper artifacts from the metrics CSV
python scripts/build_tables.py --config configs/eval.yaml            # Tables 3-5, 9
python scripts/build_figures.py --config configs/eval.yaml           # Figures 1-2
python scripts/build_full_results_table.py --config configs/eval.yaml # appendix 30x9 matrix
python scripts/statistical_tests.py --config configs/eval.yaml       # bootstrap CIs, significance
python scripts/error_analysis.py --config configs/eval.yaml          # qualitative error examples
```

**Environment requirements:**
- `HF_TOKEN` must be set with access to private `Despina/*` datasets on Hugging Face.
- Frontier-model generation (`scripts/run_frontier_generations.py`) requires an OpenRouter API key, passed via `--openrouter_key`.

## Architecture

### Pipeline stages (in order)

1. **Dataset aggregation** (`src/build_mixtures.py`) — Combines source HF datasets into GenTune (general), LitTune (literary), MixTune (balanced mix). One-time step.
2. **DAPT** (`src/dapt.py`) — Optional continued causal LM pretraining on literary corpora (LitBank, BookCorpus). Produces `{model_id}-lit-dapt` checkpoints.
3. **Fine-tuning** (`src/train.py`) — QLoRA instruction tuning for RE. Each run = (model x dataset x shot). Produces `{model_id}-{dataset}-{shot}-shot`.
4. **Generation** (`src/run_generations.py`) — Runs fine-tuned checkpoints on original eval datasets. Produces both "open" (free-form) and "constrained" (label-set restricted) generations.
5. **Merge** (`scripts/merge_generations.py`) — Merges per-worker shard files into final CSV/parquet per eval dataset.
6. **Evaluation** (`scripts/run_evaluation.py` → `src/eval.py`) — Reads merged generation CSVs, computes micro/macro P/R/F1 plus `schema_valid_rate`/`malformed_rate`, and writes `runs/evaluation/per_dataset_metrics.csv` (one row per model-config × dataset × gen_type) and `runs/evaluation/per_class/` breakdowns.
7. **Paper artifacts** (`scripts/build_*.py`, `statistical_tests.py`, `error_analysis.py`) — Consume `per_dataset_metrics.csv` (and, for bootstrap/significance, the raw generation CSVs) to emit the paper's tables (`runs/evaluation/tables/`), figures (`figures/`), and statistical outputs (`runs/evaluation/statistical/`).

The pipeline is staged through files: stages 1-5 produce generation CSVs, then the analysis layer (6-7) is pure post-processing over those CSVs. `src/eval.py` is the single source of truth for metric definitions (label normalization, slice evaluation, per-class metrics) — it is a pure, no-I/O library that every `scripts/` analysis tool imports rather than reimplementing.

### Supplementary experiment tracks

Beyond the main 30-config matrix, several self-contained comparisons each have their own generation config + evaluator and write to a parallel `runs/generations_*/` directory:

- **Frontier baselines** — `scripts/run_frontier_generations.py` (async OpenRouter calls) → `scripts/evaluate_frontier.py` (Table 7).
- **Cross-domain transfer** — `configs/generations_crossdomain.yaml` → `scripts/evaluate_crossdomain.py` (GenTune→literary, LitTune→general degradation).
- **DAPT case study** — `configs/experiments_dapt_casestudy.yaml` / `configs/generations_dapt_casestudy.yaml` → `scripts/evaluate_dapt_casestudy.py` (Llama-3.2-3B with vs. without DAPT).

### Orchestration

- `src/run_all.py` — Main entry point for DAPT + finetuning. Reads `configs/experiments.yaml`, builds the full job list (model x dataset x shot), and supports deterministic sharding via `--job_index`/`--job_count` for multi-GPU parallelism.
- `src/run_generations.py` — Same sharding pattern for inference. Reads `configs/generations.yaml`.

### Config files (all YAML)

- `configs/datasets.yaml` — Dataset registry: source HF repos, domain tags, column mappings, aggregation rules.
- `configs/experiments.yaml` — Models list, DAPT hyperparams, finetune hyperparams, QLoRA settings (with per-model overrides), shot values, HF publishing config.
- `configs/generations.yaml` — Eval datasets, generation params, prompt templates, label-set constraints, eval group restrictions (which tuned models eval on which datasets), output format.
- `configs/eval.yaml` — Evaluation/analysis config: `primary_gen_type` (which gen type populates main tables), `prompt_shot_policy` (`matched` vs `all`), per-dataset `exclude_labels`, `dataset_groups` (general vs literary), `model_metadata` (param counts, scale groups), and CSV streaming settings.
- `configs/generations_crossdomain.yaml`, `configs/generations_dapt_casestudy.yaml`, `configs/experiments_dapt_casestudy.yaml` — Configs for the supplementary tracks above.

### Key design decisions

- **Completeness tracking**: Each run writes a `_DONE` sentinel file; already-completed runs are skipped on re-execution.
- **Few-shot prompt unraveling**: `train.py:_unravel_fewshot_prompt_to_messages()` converts dataset few-shot prompts into alternating user/assistant chat turns for chat-template models.
- **Eval group restrictions**: `generations.yaml` controls which eval datasets each tuned family runs on (e.g., `re_littune` models only eval on literary datasets; `re_mixtune` on all).
- **Two generation types**: "gen_open" uses a generic system prompt; "gen_constrained" injects allowed label sets per eval dataset into the system prompt.
- All checkpoints and datasets are pushed to private HF repos under the `Despina/` namespace.

### Utilities

- `src/hf_utils.py` — HF token resolution (config value or env var) and model directory upload.
- `src/eval.py` — Pure RE metrics library (label normalization, slice/per-class P/R/F1, schema-validity). Imported by every `scripts/` analysis tool; change metric behavior here, not in the scripts.

# Sub-Billion, Super-Frontier: Fine-Tuned Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction

Reproduction code, configurations, and links to released artifacts for the paper.

**Paper:** [arXiv:2606.22606](https://arxiv.org/html/2606.22606v1)

We fine-tune five small / sub-billion language models (360M-3B) for **relation extraction (RE)**
across **general-domain** and **literary** text using QLoRA, under three domain-composition
regimes (GenTune, LitTune, MixTune) and two prompt-conditioned tuning styles (0-shot, 2-shot),
for 30 tuned configurations. We benchmark them against zero-shot frontier LLMs
(GPT-5.4, Claude Sonnet 4.6) and a discriminative RoBERTa baseline, scoring with
**positive-class micro-F1** (the no-relation class excluded). The best sub-billion model
(Qwen2.5-0.5B) reaches 0.83 General Avg, exceeding GPT-5.4 (0.69) and Claude Sonnet 4.6 (0.66)
evaluated zero-shot; the tuned SLMs also lead the frontier on literary RE, and a RoBERTa
baseline tuned in-domain likewise clears both frontier systems.

The pipeline is config-driven: every stage reads a YAML file under `configs/`.

## Released artifacts

| Artifact | Where |
|---|---|
| **Best sub-billion checkpoint** (Qwen2.5-0.5B, GenTune, 2-shot) | Hugging Face: `Despina/Qwen2.5-0.5B-Instruct-re_gentune-2-shot` |
| **Frontier-model outputs** (GPT-5.4 + Claude Sonnet 4.6 generations) | Hugging Face dataset: `Despina/frontier-re-generations` |
| **RE benchmarks** (8 of 9; see licensing) | Hugging Face datasets under the `Despina/` namespace (table below) |
| **TACRED** | **Not redistributed** (LDC license). Rebuild the rows with our scripts. |

### RE benchmark datasets (Hugging Face, `Despina/` namespace)

| Domain | Dataset | Hugging Face repo |
|---|---|---|
| General | SemEval-2010 Task 8 | `Despina/semeval2010_task8` |
| General | CoNLL04 | `Despina/conll04` |
| General | NYT11 | `Despina/nyt_11` |
| General | GIDS | `Despina/gids` |
| General | Re-DocRED | `Despina/re-docred` |
| General | REBEL | `Despina/rebel-dataset` |
| General | TACRED | *(LDC-licensed; not redistributed, rebuild with scripts)* |
| Literature | Biographical | `Despina/biographical` |
| Literature | PG-Fiction | `Despina/project-gutenberg-fiction-relations` |

> Browse a dataset at `https://huggingface.co/datasets/Despina/<repo>` and the checkpoint at
> `https://huggingface.co/Despina/Qwen2.5-0.5B-Instruct-re_gentune-2-shot`.

## Requirements

```bash
pip install -r requirements.txt
```

Set a Hugging Face token (read access to the datasets; write access if you push checkpoints):

```bash
export HF_TOKEN=hf_your_token_here
```

Frontier-model generation additionally needs an OpenRouter API key:

```bash
export OPENROUTER_API_KEY=sk-or-...
```

All scripts use `from src...` imports, so run them from the repo root. `src/` entry points use
`python -m`; `scripts/` analysis tools add the repo root to `sys.path`, so run them as
`python scripts/<name>.py`.

## Reproduce the paper

```bash
# 1. Build the aggregated training mixtures (GenTune / LitTune / MixTune)
python src/build_mixtures.py --config configs/datasets.yaml

# 2. (optional) Domain-adaptive pretraining, then fine-tuning
python -m src.run_all --config configs/experiments.yaml --stage dapt
python -m src.run_all --config configs/experiments.yaml --stage finetune
#    Multi-GPU sharding (one process per GPU):
#    CUDA_VISIBLE_DEVICES=0 python -m src.run_all --config configs/experiments.yaml \
#        --stage finetune --job_count 4 --job_index 0

# 3. Generate predictions from the fine-tuned checkpoints
python -m src.run_generations --config configs/generations.yaml

# 4. Merge the per-worker generation shards into one CSV per dataset
python scripts/merge_generations.py --generations_dir runs/generations --shards 0 1 2 3

# 5. Evaluate -> runs/evaluation/per_dataset_metrics.csv (+ per-class breakdowns)
python scripts/run_evaluation.py --config configs/eval.yaml

# 6. Build the paper artifacts
python scripts/build_tables.py             --config configs/eval.yaml  # main tables
python scripts/build_full_results_table.py --config configs/eval.yaml  # appendix matrix
python scripts/build_figures.py            --config configs/eval.yaml  # figures
python scripts/statistical_tests.py        --config configs/eval.yaml  # bootstrap CIs / significance
```

### Supplementary tracks

```bash
# Frontier baselines (needs OPENROUTER_API_KEY)
python scripts/run_frontier_generations.py --config configs/generations.yaml --openrouter_key "$OPENROUTER_API_KEY"
python scripts/evaluate_frontier.py --config configs/eval.yaml

# Discriminative encoder baseline (RoBERTa)
python scripts/train_encoder_baseline.py --config configs/encoder_baseline.yaml

# Out-of-domain / cross-domain evaluation and the DAPT case study
python -m src.run_generations --config configs/generations_crossdomain.yaml
python scripts/evaluate_crossdomain.py --config configs/eval.yaml
python scripts/evaluate_dapt_casestudy.py --config configs/eval.yaml

# Truncation and scale-vs-family analyses reported in the paper
python scripts/analyze_truncation.py
python scripts/analyze_scale_family.py --config configs/eval.yaml

# Upload the frontier outputs to a private HF dataset (run with your own token)
HF_TOKEN=hf_xxx python scripts/upload_frontier_to_hf.py
```

## Repository layout

```
src/        pipeline: build_mixtures, dapt, train, run_all, run_generations, eval, hf_utils
scripts/    merge, evaluation, table/figure builders, statistical tests, baselines, analyses
configs/    YAML configs (datasets, experiments, generations, eval, baselines, supplementary)
figures/    generated paper figures
runs/       all outputs (checkpoints, generations, evaluation): gitignored, regenerated locally
```

`src/eval.py` is the single source of truth for metric definitions (label normalization,
positive-class P/R/F1, schema validity); every analysis script imports it rather than
reimplementing metrics.

## Data and licensing notes

- **TACRED is not redistributed.** It is distributed under a Linguistic Data Consortium license;
  holders of that license can rebuild the TACRED rows with `src/build_mixtures.py`.
- **PG-Fiction** is annotated by a GPT-4-class model (see the paper); tuned models partly learn
  the annotator's label distribution. The 137-to-48 canonical-ontology mapping is included.
- Frontier outputs are released because the API comparison is not bit-for-bit reproducible.

## Citation

If you use this work, please cite:

> Christou, D., & Tsoumakas, G. (2026). Sub-Billion, Super-Frontier: Small Language Models Rival Zero-Shot Frontier LLMs on General and Literary Relation Extraction. *arXiv preprint arXiv:2606.22606.*

```bibtex
@article{christou2026subbillion,
  title        = {Sub-Billion, Super-Frontier: Small Language Models Rival
                  Zero-Shot Frontier LLMs on General and Literary Relation Extraction},
  author       = {Christou, Despina and Tsoumakas, Grigorios},
  journal      = {arXiv preprint arXiv:2606.22606},
  year         = {2026},
  url          = {https://arxiv.org/abs/2606.22606}
}
```

## License

Code is released under the MIT License (see `LICENSE`). Dataset and model artifacts are subject
to the licenses of their underlying sources.

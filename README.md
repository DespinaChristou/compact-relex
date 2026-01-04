# Paper on RE — Small LLMs for Relation Extraction

This repository contains the code to reproduce the experiments for our paper on **relation extraction (RE) with small
and micro LLMs**, across **general** and **literary** domains.

The project supports:

* aggregating multiple RE datasets hosted on Hugging Face into **GenTune**, **LitTune**, and **MixTune** training sets,
* optional **Domain-Adaptive Pretraining (DAPT)** on literary corpora,
* fine-tuning multiple LLMs under different strategies and few-shot settings,
* saving all outputs locally **and** pushing them to **private Hugging Face repos**.

The setup is intentionally simple and config-driven.

---

## Repository structure (simplified)

```
paper-on-re/
├─ README.md
├─ requirements.txt
├─ configs/
│  ├─ datasets.yaml        # dataset registry + aggregation plan
│  └─ experiments.yaml    # models, strategies, shots, HF publishing
├─ src/
│  ├─ build_mixtures.py   # builds GenTune / LitTune / MixTune datasets
│  ├─ run_all.py          # runs DAPT + finetuning experiments
│  ├─ train.py            # finetuning logic (called by run_all.py)
│  ├─ dapt.py             # DAPT logic (called by run_all.py)
│  ├─ eval.py             # evaluation & metrics
│  └─ hf_utils.py         # Hugging Face upload utilities
└─ runs/                  # (gitignored) local checkpoints and logs
```

---

## Datasets

All source datasets are **private Hugging Face datasets** under the `Despina/*` namespace and already contain:

* splits: `train`, `validation`, `test`
* prompt columns:

    * `prompt_0_shot`
    * `prompt_2_shot`
    * `prompt_5_shot`
* label column: `relation`

### Domains

* **General**: TACRED, SemEval2010 Task 8, CoNLL04, NYT-11, GIDS, Re-DocRED, REBEL
* **Literature**: Biographical, Project Gutenberg Fiction

The file `configs/datasets.yaml` defines:

* which datasets belong to each domain,
* how to aggregate them into:

    * **GenTune** (general only),
    * **LitTune** (literature only),
    * **MixTune** (domain-balanced mix),
* where the aggregated datasets are published on Hugging Face.

---

## Aggregated datasets (output)

Running the dataset builder creates three **private HF datasets**:

* `Despina/re_gentune`
* `Despina/re_littune`
* `Despina/re_mixtune`

Each contains **9 splits**:

```
train_0, eval_0, test_0
train_2, eval_2, test_2
train_5, eval_5, test_5
```

Each example has the unified schema:

```json
{
  "prompt": "...",
  "relation": "...",
  "dataset": "tacred"
}
```

---

## Models

The experiments use **instruction-tuned checkpoints only**.

### Micro models

* SmolLM2-135M
* SmolLM2-360M
* Qwen2.5-0.5B

### Small models

* Gemma 2 2B
* SmolLM3-3B
* Qwen2.5-3B

Each model is fine-tuned under:

* **GenTune**
* **LitTune**
* **Dapt_LitTune**
* **MixTune**

and evaluated under:

* 0-shot
* 2-shot
* 5-shot

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Minimal requirements:

```
datasets
huggingface_hub
transformers
torch
pyyaml
```

### 2. Set Hugging Face token

You must have access to the private datasets and permission to create private repos.

```bash
export HF_TOKEN=your_hf_token_here
```

---

## Step 1 — Build aggregated datasets

This step **only needs to be run once**.

```bash
python src/build_mixtures.py --config configs/datasets.yaml
```

This will:

* load all source datasets from Hugging Face,
* select the correct prompt column for each shot,
* aggregate datasets into GenTune, LitTune, and MixTune,
* push the aggregated datasets to private HF repos.

To build only one aggregate (optional):

```bash
python src/build_mixtures.py --config configs/datasets.yaml --only mixtune
```

---

## Step 2 — Run all experiments

```bash
python src/run_all.py --config configs/experiments.yaml
```

This script:

1. Expands the full experiment grid (models × strategies × shots).
2. Runs **DAPT once per model** (if enabled).
3. Runs RE fine-tuning jobs, respecting dependencies.
4. Saves all outputs locally under `runs/`.
5. Pushes each trained model (or adapter) to a **private Hugging Face repo**.

All runs are resumable: completed jobs are skipped automatically.

---

## Outputs

### Local

All artifacts are stored under:

```
runs/
├─ dapt/
└─ finetune/
```

Each run directory contains:

* model or adapter weights,
* training metadata,
* evaluation outputs,
* a `_DONE` flag for resuming.

### Hugging Face

Each run is pushed to a private repo named:

```
paper-on-re-{job_id}
```

This makes every experiment fully traceable and reproducible.

---

## Reproducibility

* All experiments are config-driven.
* Dataset aggregation is deterministic (fixed seeds).
* Each run logs:

    * model name,
    * tuning strategy,
    * shot setting,
    * dataset mix,
    * hyperparameters.

---

## Notes

* Raw datasets and checkpoints are **not** committed to Git.
* Only aggregated datasets and trained models are stored on Hugging Face.
* The code assumes **RE-as-generation**, with constrained label prediction handled in `train.py`.

---

## Contact

For questions or issues related to the code or experiments, please contact the paper authors.

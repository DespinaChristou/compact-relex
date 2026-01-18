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
 │ ├─ datasets.yaml # dataset registry + aggregation plan 
 │ └─ experiments.yaml # models, strategies, shots, HF publishing 
 ├─ src/ 
 │ ├─ build_mixtures.py # builds GenTune / LitTune / MixTune datasets 
 │ ├─ run_all.py # runs DAPT + finetuning experiments 
 │ ├─ train.py # finetuning logic (called by run_all.py) 
 │ ├─ dapt.py # DAPT logic (called by run_all.py) 
 │ ├─ eval.py # evaluation & metrics 
 │ └─ hf_utils.py # Hugging Face upload utilities 
 └─ runs/ # (gitignored) local checkpoints and logs
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

* SmolLM2-135M-Instruct
* SmolLM2-360M-Instruct
* Qwen2.5-0.5B-Instruct

### Small models

* gemma-2-2b-it
* SmolLM3-3B
* Qwen2.5-3B-Instruct
* Llama-3.2-3B-Instruct

Each model is:

1) optionally DAPT-trained on literary corpora (producing `{model_id}-lit-dapt`),
2) fine-tuned for RE under:
    * `re_gentune`
    * `re_littune`
    * `re_mixtune`

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

## Step 2 — Run all experiments (DAPT → finetune)

```bash
python src/run_all.py --config configs/experiments.yaml
```

This script:

1. Runs **DAPT once per base model** (if enabled).
2. Fine-tunes **all base models and all DAPT models** (i.e., 2× models total) on all datasets × shots.
3. Saves all outputs locally under `runs/`.
4. Pushes each trained model to a **private Hugging Face repo**.

---

## Outputs

### Local

All artifacts are stored under:

```
runs/
├─ dapt/
└─ finetune/
```

DAPT naming:

* `{model_id}-lit-dapt`

Finetuning naming:

* `{model_id}-{dataset_name}-{N}-shot`

### Hugging Face

Each run is pushed to a private repo under your namespace using the **same name** as above.

---

## TensorBoard (Hugging Face)

Training logs are written via the Transformers Trainer integration (`report_to=["tensorboard"]`).

We include TensorBoard event files inside each pushed model repo under a `runs/` directory so that Hugging Face can
render a **TensorBoard** tab.

### Fine-tuning runs

For each fine-tuning run we:

- write logs locally under: `runs/finetune/<run_name>/tb/`
- copy them into the uploaded model artifact under: `runs/` inside the HF model repo

### DAPT runs

For each DAPT run we:

- write logs locally under: `runs/dapt/<model_id>-lit-dapt/tb/`
- copy them into the uploaded model artifact under: `runs/` inside the HF model repo

If you do not see the TensorBoard tab:

- confirm `tensorboard` is installed (`pip install tensorboard`)
- confirm the HF model repo contains event files under a `runs/` directory
- confirm you opened the repo that contains the uploaded checkpoint (DAPT or fine-tuned)

---

## Notes

* Raw datasets and checkpoints are **not** committed to Git.
* Only aggregated datasets and trained models are stored on Hugging Face.
* Fine-tuning is **instruction tuning** using the dataset’s `prompt` (input) and `relation` (target).


#!/usr/bin/env python3
"""
Generate relation predictions from frontier LLMs via OpenRouter API.
Uses async concurrency (50+ parallel requests) for ~50x speedup.

Usage:
    # Full run (all 3 models, all 9 datasets, constrained, 0-shot)
    python scripts/run_frontier_generations.py \
        --config configs/generations.yaml \
        --openrouter_key YOUR_KEY

    # Control concurrency (default: 50 parallel requests)
    python scripts/run_frontier_generations.py \
        --config configs/generations.yaml \
        --openrouter_key YOUR_KEY \
        --concurrency 80

    # Single model
    python scripts/run_frontier_generations.py \
        --config configs/generations.yaml \
        --openrouter_key YOUR_KEY \
        --models gemini-2.5-pro

    # Dry run: estimate cost
    python scripts/run_frontier_generations.py \
        --config configs/generations.yaml --dry_run

    # Merge results after completion
    python scripts/run_frontier_generations.py --merge

Environment:
    pip install aiohttp openai pyyaml datasets tqdm --break-system-packages
"""

import argparse
import asyncio
import csv
import hashlib
import os
import sys
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# OpenRouter model registry
# ---------------------------------------------------------------------------
FRONTIER_MODELS = {
    "gpt-5.4": {
        "openrouter_id": "openai/gpt-5.4",
        "display_name": "GPT-5.4",
        "input_price_per_m": 2.50,
        "output_price_per_m": 20.00,
    },
    "claude-sonnet-4.6": {
        "openrouter_id": "anthropic/claude-sonnet-4.6",
        "display_name": "Claude Sonnet 4.6",
        "input_price_per_m": 3.00,
        "output_price_per_m": 15.00,
    },
    "gemini-2.5-pro": {
        "openrouter_id": "google/gemini-2.5-pro",
        "display_name": "Gemini 2.5 Pro",
        "input_price_per_m": 1.00,
        "output_price_per_m": 10.00,
    },
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_RETRIES = 6
RETRY_BASE_DELAY = 1.5
OUTPUT_DIR = "runs/frontier_generations"
CHARS_PER_TOKEN = 4
DEFAULT_CONCURRENCY = 50        # parallel requests
FLUSH_EVERY = 100               # flush CSV to disk every N rows


def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_hf_token(config: dict) -> str:
    token = config.get("hf", {}).get("token")
    if token:
        return token
    env_var = config.get("hf", {}).get("token_env", "HF_TOKEN")
    token = os.environ.get(env_var)
    if not token:
        raise RuntimeError(f"HF token not found in config or env var {env_var}")
    return token


def load_eval_dataset(ds_config: dict, hf_token: str) -> list[dict]:
    from datasets import load_dataset
    repo = ds_config["repo"]
    subset = ds_config.get("subset")
    split = ds_config.get("split", "test")
    ds = load_dataset(repo, subset, split=split, token=hf_token)
    rows = []
    for row in ds:
        rows.append({
            "prompt_0_shot": row.get("prompt_0_shot", ""),
            "prompt_2_shot": row.get("prompt_2_shot", ""),
            "relation": row.get("relation", ""),
        })
    return rows


def get_label_set(ds_config: dict, config: dict, hf_token: str) -> list[str]:
    from datasets import load_dataset
    repo = ds_config["repo"]
    subset = ds_config.get("subset")
    label_cfg = config.get("label_sets", {}).get("compute_from", {})
    split = label_cfg.get("split", "test")
    rel_col = label_cfg.get("relation_column", "relation")
    ds = load_dataset(repo, subset, split=split, token=hf_token)
    labels = set()
    for row in ds:
        val = row.get(rel_col, None)
        if val is None or (isinstance(val, str) and val.strip() == ""):
            labels.add("none")
        elif isinstance(val, float) and str(val) == "nan":
            labels.add("none")
        else:
            labels.add(str(val).strip())
    return sorted(labels)


def build_system_prompt(gen_type: str, config: dict, label_set: list[str] | None) -> str:
    prompts_cfg = config.get("prompts", {})
    if gen_type == "gen_open":
        return prompts_cfg.get("system_open",
            "You are a relation extraction system. Output ONLY the relation type. No explanation.")
    else:
        template = prompts_cfg.get("system_constrained_template",
            "You are a relation extraction system. Choose exactly one label from: {allowed_labels}")
        labels_str = ", ".join(label_set) if label_set else ""
        return template.replace("{allowed_labels}", labels_str)


def get_done_path(output_dir, dataset_name, model_key, gen_type, prompt_shot):
    return Path(output_dir) / dataset_name / f"_DONE_{model_key}_{gen_type}_ps{prompt_shot}"


def get_output_path(output_dir, dataset_name, model_key, gen_type, prompt_shot):
    return Path(output_dir) / dataset_name / f"generations_{model_key}_{gen_type}_ps{prompt_shot}.csv"


def is_complete(output_dir, dataset_name, model_key, gen_type, prompt_shot):
    return get_done_path(output_dir, dataset_name, model_key, gen_type, prompt_shot).exists()


def load_completed_hashes(out_path: Path) -> set[str]:
    completed = set()
    if out_path.exists():
        with open(out_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = row.get("prompt_hash", "")
                if h:
                    completed.add(h)
    return completed


# ---------------------------------------------------------------------------
# Async API caller
# ---------------------------------------------------------------------------
async def call_openrouter_async(
    session,
    api_key: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 64,
    temperature: float = 0.0,
) -> tuple[str, int, int]:
    """Async OpenRouter call with retry + semaphore-based concurrency control."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/despina-christou/compact-relex",
        "X-Title": "Compact-RelEx Frontier Evaluation",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
    }

    for attempt in range(MAX_RETRIES):
        async with semaphore:
            try:
                async with session.post(
                    OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60
                ) as resp:
                    if resp.status == 429:
                        retry_after = float(resp.headers.get("Retry-After", RETRY_BASE_DELAY * (2 ** attempt)))
                        await asyncio.sleep(min(retry_after, 30))
                        continue
                    if resp.status in (502, 503, 504):
                        await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                        continue

                    data = await resp.json()

                    if "error" in data:
                        err_msg = data["error"].get("message", str(data["error"]))
                        if "rate" in err_msg.lower() or resp.status == 429:
                            await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                            continue
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(RETRY_BASE_DELAY)
                            continue
                        return f"ERROR: {err_msg}", 0, 0

                    choices = data.get("choices", [])
                    content = choices[0]["message"]["content"] if choices else None
                    text = content.strip() if content else ""

                    usage = data.get("usage", {})
                    in_tok = usage.get("prompt_tokens", 0)
                    out_tok = usage.get("completion_tokens", 0)
                    return text, in_tok, out_tok

            except asyncio.TimeoutError:
                await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_BASE_DELAY)
                else:
                    return f"ERROR: {e}", 0, 0

    return "ERROR: max retries exceeded", 0, 0


# ---------------------------------------------------------------------------
# Async generation runner
# ---------------------------------------------------------------------------
async def run_generation_async(
    api_key: str,
    model_key: str,
    model_config: dict,
    dataset_name: str,
    dataset_rows: list[dict],
    gen_type: str,
    prompt_shot: int,
    system_prompt: str,
    output_dir: str,
    concurrency: int,
) -> tuple[int, int]:
    """Run all examples concurrently with a semaphore."""
    import aiohttp

    out_path = get_output_path(output_dir, dataset_name, model_key, gen_type, prompt_shot)
    done_path = get_done_path(output_dir, dataset_name, model_key, gen_type, prompt_shot)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already completed
    completed_hashes = load_completed_hashes(out_path)
    prompt_col = f"prompt_{prompt_shot}_shot"
    openrouter_id = model_config["openrouter_id"]

    # Build pending list
    pending = []
    for row in dataset_rows:
        prompt_text = row.get(prompt_col, "")
        if not prompt_text:
            continue
        prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:12]
        if prompt_hash in completed_hashes:
            continue
        pending.append((row, prompt_text, prompt_hash))

    if not pending:
        print(f"  All {len(dataset_rows)} examples already done for "
              f"{model_key}/{dataset_name}/{gen_type}/ps{prompt_shot}")
        return 0, 0

    total = len(pending)
    print(f"  {model_key}/{dataset_name}/{gen_type}/ps{prompt_shot}: "
          f"{total} pending ({len(completed_hashes)} already done), "
          f"concurrency={concurrency}")

    semaphore = asyncio.Semaphore(concurrency)
    total_input = 0
    total_output = 0
    completed_count = 0
    start_time = time.time()

    # Open CSV for appending
    write_header = not out_path.exists() or out_path.stat().st_size == 0
    csv_file = open(out_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=[
        "eval_dataset_name", "prompt_hash", "relation", "gen_type",
        "model_id", "tuned_dataset_name", "model_shot", "prompt_shot",
        "generated_relation", "input_tokens", "output_tokens",
    ])
    if write_header:
        writer.writeheader()

    async def process_one(row, prompt_text, prompt_hash, session):
        nonlocal total_input, total_output, completed_count
        generated, in_tok, out_tok = await call_openrouter_async(
            session, api_key, openrouter_id, system_prompt, prompt_text,
            semaphore, max_tokens=64, temperature=0.0,
        )
        total_input += in_tok
        total_output += out_tok
        completed_count += 1

        writer.writerow({
            "eval_dataset_name": dataset_name,
            "prompt_hash": prompt_hash,
            "relation": row.get("relation", ""),
            "gen_type": gen_type,
            "model_id": model_config["display_name"],
            "tuned_dataset_name": "frontier",
            "model_shot": "n/a",
            "prompt_shot": prompt_shot,
            "generated_relation": generated,
            "input_tokens": in_tok,
            "output_tokens": out_tok,
        })

        # Periodic flush + progress
        if completed_count % FLUSH_EVERY == 0:
            csv_file.flush()
            elapsed = time.time() - start_time
            rate = completed_count / elapsed if elapsed > 0 else 0
            eta_min = (total - completed_count) / rate / 60 if rate > 0 else 0
            cost = (total_input / 1e6 * model_config["input_price_per_m"]
                    + total_output / 1e6 * model_config["output_price_per_m"])
            print(f"    [{completed_count}/{total}] {rate:.1f} ex/s, "
                  f"ETA {eta_min:.1f}min, cost=${cost:.2f}", flush=True)

    # Fire all requests with controlled concurrency
    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            process_one(row, prompt_text, prompt_hash, session)
            for row, prompt_text, prompt_hash in pending
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # Final flush
    csv_file.flush()
    csv_file.close()

    # Mark done
    done_path.write_text(f"completed {len(dataset_rows)} examples\n")

    elapsed = time.time() - start_time
    cost = (total_input / 1e6 * model_config["input_price_per_m"]
            + total_output / 1e6 * model_config["output_price_per_m"])
    print(f"  DONE {model_key}/{dataset_name}: {completed_count} examples in "
          f"{elapsed:.0f}s ({completed_count/elapsed:.1f} ex/s), cost=${cost:.2f}")

    return total_input, total_output


# ---------------------------------------------------------------------------
# Cost estimation (no API calls)
# ---------------------------------------------------------------------------
def estimate_cost(config, datasets_filter, models_filter, gen_types, prompt_shots, hf_token):
    eval_datasets = config.get("eval_datasets", [])
    print("\n" + "=" * 70)
    print("COST ESTIMATION (dry run)")
    print("=" * 70)

    grand_input = {m: 0 for m in FRONTIER_MODELS}
    grand_output = {m: 0 for m in FRONTIER_MODELS}

    for ds_cfg in eval_datasets:
        ds_name = ds_cfg["name"]
        if datasets_filter and ds_name not in datasets_filter:
            continue
        rows = load_eval_dataset(ds_cfg, hf_token)
        n = len(rows)
        sample = rows[:min(100, n)]
        avg_0 = sum(len(r["prompt_0_shot"]) for r in sample) / len(sample) / CHARS_PER_TOKEN
        avg_2 = sum(len(r["prompt_2_shot"]) for r in sample) / len(sample) / CHARS_PER_TOKEN

        for model_key in FRONTIER_MODELS:
            if models_filter and model_key not in models_filter:
                continue
            for gen_type in gen_types:
                sys_tok = 100 if gen_type == "gen_constrained" else 50
                for ps in prompt_shots:
                    avg_prompt = avg_0 if ps == 0 else avg_2
                    grand_input[model_key] += (avg_prompt + sys_tok) * n
                    grand_output[model_key] += 10 * n

        print(f"  {ds_name}: {n} examples, avg_0shot={avg_0:.0f}tok, avg_2shot={avg_2:.0f}tok")

    print(f"\n{'Model':<25} {'Input Tok':>12} {'Output Tok':>12} {'Cost':>10}")
    print("-" * 65)
    total_cost = 0
    for model_key, mcfg in FRONTIER_MODELS.items():
        if models_filter and model_key not in models_filter:
            continue
        inp, out = grand_input[model_key], grand_output[model_key]
        cost = inp / 1e6 * mcfg["input_price_per_m"] + out / 1e6 * mcfg["output_price_per_m"]
        total_cost += cost
        print(f"  {mcfg['display_name']:<23} {inp:>12,.0f} {out:>12,.0f} ${cost:>8.2f}")
    print("-" * 65)
    print(f"  {'TOTAL':<23} {'':>12} {'':>12} ${total_cost:>8.2f}")
    print()
    return total_cost


# ---------------------------------------------------------------------------
# Merge utility
# ---------------------------------------------------------------------------
def merge_frontier_results(output_dir: str = OUTPUT_DIR):
    output_path = Path(output_dir)
    combined_rows = []
    fieldnames = None
    for ds_dir in sorted(output_path.iterdir()):
        if not ds_dir.is_dir():
            continue
        ds_name = ds_dir.name
        ds_rows = []
        for csv_file in sorted(ds_dir.glob("generations_*.csv")):
            with open(csv_file) as f:
                reader = csv.DictReader(f)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                for row in reader:
                    ds_rows.append(row)
        if ds_rows:
            merged_path = ds_dir / "generations_merged.csv"
            with open(merged_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(ds_rows)
            print(f"  {ds_name}: {len(ds_rows)} rows -> {merged_path}")
            combined_rows.extend(ds_rows)
    if combined_rows and fieldnames:
        combined_path = output_path / "all_frontier_generations.csv"
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        print(f"\n  Combined: {len(combined_rows)} rows -> {combined_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def async_main():
    parser = argparse.ArgumentParser(description="Run frontier LLM evaluations via OpenRouter (async)")
    parser.add_argument("--config", required=True, help="Path to generations.yaml")
    parser.add_argument("--openrouter_key", default=None,
                        help="OpenRouter API key (or set OPENROUTER_API_KEY env var)")
    parser.add_argument("--models", nargs="*", default=None,
                        choices=list(FRONTIER_MODELS.keys()),
                        help="Which frontier models to run (default: all)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Which datasets to run (default: all)")
    parser.add_argument("--gen_types", nargs="*", default=["constrained"],
                        choices=["constrained", "open"],
                        help="Generation types (default: constrained only)")
    parser.add_argument("--prompt_shots", nargs="*", type=int, default=[0],
                        choices=[0, 2],
                        help="Prompt shot settings (default: 0-shot only)")
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--dry_run", action="store_true",
                        help="Only estimate cost, don't make API calls")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap examples per dataset (for testing)")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help=f"Max parallel requests (default: {DEFAULT_CONCURRENCY})")
    args = parser.parse_args()

    config = load_config(args.config)
    hf_token = get_hf_token(config)

    gen_types = [f"gen_{gt}" if not gt.startswith("gen_") else gt for gt in args.gen_types]
    prompt_shots = args.prompt_shots

    if args.dry_run:
        estimate_cost(config, args.datasets, args.models, gen_types, prompt_shots, hf_token)
        return

    api_key = args.openrouter_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("ERROR: No OpenRouter API key. Use --openrouter_key or set OPENROUTER_API_KEY")
        sys.exit(1)

    eval_datasets = config.get("eval_datasets", [])
    total_cost = 0.0
    grand_start = time.time()

    for ds_cfg in eval_datasets:
        ds_name = ds_cfg["name"]
        if args.datasets and ds_name not in args.datasets:
            continue

        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")

        rows = load_eval_dataset(ds_cfg, hf_token)
        if args.max_examples:
            rows = rows[:args.max_examples]
        print(f"  Loaded {len(rows)} test examples")

        label_set = None
        if "gen_constrained" in gen_types:
            label_set = get_label_set(ds_cfg, config, hf_token)
            print(f"  Label set: {len(label_set)} labels")

        for model_key, model_config in FRONTIER_MODELS.items():
            if args.models and model_key not in args.models:
                continue

            for gen_type in gen_types:
                system_prompt = build_system_prompt(gen_type, config, label_set)

                for ps in prompt_shots:
                    if is_complete(args.output_dir, ds_name, model_key, gen_type, ps):
                        print(f"  SKIP (done): {model_key}/{gen_type}/ps{ps}")
                        continue

                    in_tok, out_tok = await run_generation_async(
                        api_key=api_key,
                        model_key=model_key,
                        model_config=model_config,
                        dataset_name=ds_name,
                        dataset_rows=rows,
                        gen_type=gen_type,
                        prompt_shot=ps,
                        system_prompt=system_prompt,
                        output_dir=args.output_dir,
                        concurrency=args.concurrency,
                    )
                    cost = (in_tok / 1e6 * model_config["input_price_per_m"]
                            + out_tok / 1e6 * model_config["output_price_per_m"])
                    total_cost += cost

    elapsed = time.time() - grand_start
    print(f"\n{'='*60}")
    print(f"ALL DONE in {elapsed/60:.1f} minutes, total cost: ${total_cost:.2f}")
    print(f"{'='*60}")


def main():
    if "--merge" in sys.argv:
        output_dir = OUTPUT_DIR
        for i, arg in enumerate(sys.argv):
            if arg == "--output_dir" and i + 1 < len(sys.argv):
                output_dir = sys.argv[i + 1]
        merge_frontier_results(output_dir)
    else:
        asyncio.run(async_main())


if __name__ == "__main__":
    main()

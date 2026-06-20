#!/usr/bin/env python3
"""
Measure the *disaggregated* size / memory footprint AND single-example inference
latency of every tuned SLM, so the paper can stop reporting a single ambiguous
"checkpoint size" column and can replace its "~"-estimated latency figures with
measured ones.

Per (base model, LoRA adapter) pair it measures, with real artifacts:
  1. base_bf16_gb        - original BF16 base checkpoint
  2. nf4_4bit_gb         - NF4 4-bit backbone (QLoRA training / GPU deployment rep)
  3. trainable_params    - exact LoRA trainable parameter count (PEFT)
  4. adapter_disk_mb     - actual saved adapter_model.safetensors size
  5. merged_bf16_gb      - merge_and_unload() + save, measured on disk
  6. peak_infer_vram_gb  - peak CUDA memory for single-example generate (4-bit)
  7. peak_train_vram_gb  - peak CUDA memory for one forward+backward (4-bit + LoRA)
  8. gpu_latency_ms      - single-example (batch 1) latency on the 4-bit model,
                           ~150-token prompt + 5-token greedy completion, warm-up
                           discarded, mean over N timed runs (the paper's protocol)

HARDWARE NOTE: latency is hardware-specific. Run the latency measurement on the
exact GPU the paper claims (desktop RTX 4090) -- NOT on a data-center pod or a
laptop GPU -- or update the claimed hardware to match. Sizes/VRAM (steps 1-7) are
essentially hardware-independent and can run anywhere with the QLoRA stack.

CPU (Q4_K_M GGUF) latency + peak RAM use llama.cpp; the exact commands are printed
at the end (run them on the claimed CPU, e.g. the i7-13700K).

    python scripts/measure_footprints.py                       # full footprint + GPU latency
    python scripts/measure_footprints.py --latency-only        # just GPU latency (fast)
    python scripts/measure_footprints.py --models Llama-3.2-3B-Instruct
    python scripts/measure_footprints.py --latency-runs 100 --no-train-vram

Outputs: runs/footprints/footprints.csv and a paste-ready LaTeX block on stdout.
"""
from __future__ import annotations

import argparse
import gc
import os
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

REPO_ROOT = Path(__file__).resolve().parent.parent

# (id, base_repo, lora_rank, adapter_repo) — adapter size is fixed by rank, so one
# representative adapter repo per model suffices.
MODELS = [
    ("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", 16,
     "Despina/SmolLM2-360M-Instruct-re_gentune-2-shot"),
    ("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 32,
     "Despina/Qwen2.5-0.5B-Instruct-re_gentune-2-shot"),
    ("SmolLM3-3B", "HuggingFaceTB/SmolLM3-3B", 64,
     "Despina/SmolLM3-3B-re_gentune-2-shot"),
    ("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 64,
     "Despina/Qwen2.5-3B-Instruct-re_gentune-2-shot"),
    ("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", 64,
     "Despina/Llama-3.2-3B-Instruct-re_gentune-2-shot"),
]

GB = 1024 ** 3
MB = 1024 ** 2

# Representative ~150-token constrained RE prompt with a short (~5-token)
# relation-label completion, matching the paper's latency protocol.
PROMPT_150 = (
    "You are an expert relation extraction system. Given a sentence and two marked "
    "entities, identify the single relation that holds between them and answer with "
    "only the relation label, exactly as written, with no explanation.\n"
    "Allowed relations: per:employee_of, org:top_members/employees, per:title, "
    "org:subsidiaries, per:cities_of_residence, org:founded_by, per:origin, "
    "org:country_of_headquarters, per:countries_of_residence, no_relation.\n"
    "Sentence: Marillyn Hewson, the chief executive officer of Lockheed Martin, told "
    "reporters in Bethesda that the defense contractor would expand its operations "
    "across several European countries during the coming fiscal year.\n"
    "Entity 1: Marillyn Hewson\nEntity 2: Lockheed Martin\nRelation:")


def _free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _gpu_latency_ms(model, enc, n_warm, n_run):
    """Mean single-example (batch 1) generate latency, warm-up iters discarded."""
    model.eval()
    gen = dict(max_new_tokens=5, do_sample=False, num_beams=1, use_cache=True)
    with torch.inference_mode():
        for _ in range(n_warm):
            model.generate(**enc, **gen)
        _sync()
        t0 = time.perf_counter()
        for _ in range(n_run):
            model.generate(**enc, **gen)
        _sync()
    return 1000.0 * (time.perf_counter() - t0) / n_run


def _adapter_disk_mb(api, adapter_repo):
    info = api.model_info(adapter_repo, files_metadata=True)
    for f in info.siblings:
        if f.rfilename == "adapter_model.safetensors":
            return (f.size or 0) / MB
    return None


def measure(mid, base_repo, rank, adapter_repo, token, args):
    from peft import PeftModel

    api = HfApi(token=token)
    res = {"model": mid, "rank": rank}
    res["adapter_disk_mb"] = _adapter_disk_mb(api, adapter_repo)

    # (1) base BF16 footprint + exact total params (skipped in latency-only mode)
    if not args.latency_only:
        _free()
        base = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=torch.bfloat16, token=token)
        res["total_params_M"] = sum(p.numel() for p in base.parameters()) / 1e6
        res["base_bf16_gb"] = base.get_memory_footprint() / GB
        del base
        _free()

    # (2) NF4 4-bit backbone (paper's QLoRA config)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m4 = AutoModelForCausalLM.from_pretrained(
        base_repo, quantization_config=bnb, device_map={"": 0}, token=token)
    res["nf4_4bit_gb"] = m4.get_memory_footprint() / GB

    # (3) trainable params via the real adapter
    peft_m = PeftModel.from_pretrained(m4, adapter_repo, token=token)
    res["trainable_params_M"] = sum(
        p.numel() for n, p in peft_m.named_parameters() if "lora_" in n) / 1e6

    # (8) single-example GPU latency + (6) peak inference VRAM
    tok = AutoTokenizer.from_pretrained(base_repo, token=token)
    enc = tok(PROMPT_150, return_tensors="pt").to(m4.device)
    res["input_tokens"] = int(enc["input_ids"].shape[1])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    res["gpu_latency_ms"] = _gpu_latency_ms(
        peft_m, enc, args.latency_warmup, args.latency_runs)
    res["peak_infer_vram_gb"] = (
        torch.cuda.max_memory_allocated() / GB if torch.cuda.is_available() else None)

    if args.latency_only:
        del m4, peft_m
        _free()
        return res

    # (7) peak training-step VRAM (one forward+backward through 4-bit + LoRA)
    if not args.no_train_vram:
        from peft import prepare_model_for_kbit_training
        del peft_m
        _free()
        m4b = AutoModelForCausalLM.from_pretrained(
            base_repo, quantization_config=bnb, device_map={"": 0}, token=token)
        m4b = prepare_model_for_kbit_training(m4b)
        tr = PeftModel.from_pretrained(m4b, adapter_repo, token=token, is_trainable=True)
        tr.train()
        batch = tok([PROMPT_150] * 4, return_tensors="pt", padding=True,
                    truncation=True, max_length=1024).to("cuda")
        torch.cuda.reset_peak_memory_stats()
        out = tr(**batch, labels=batch["input_ids"])
        out.loss.backward()
        res["peak_train_vram_gb"] = torch.cuda.max_memory_allocated() / GB
        del m4b, tr, out
    _free()

    # (5) merged BF16 model size on disk
    base_fp = AutoModelForCausalLM.from_pretrained(
        base_repo, torch_dtype=torch.bfloat16, token=token)
    merged = PeftModel.from_pretrained(base_fp, adapter_repo, token=token).merge_and_unload()
    with tempfile.TemporaryDirectory() as td:
        merged.save_pretrained(td, safe_serialization=True)
        nbytes = sum(p.stat().st_size for p in Path(td).glob("*.safetensors"))
    res["merged_bf16_gb"] = nbytes / GB
    del base_fp, merged, m4
    _free()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--latency-only", action="store_true",
                    help="Only measure GPU latency + 4-bit footprint (fast; run on the claimed GPU).")
    ap.add_argument("--no-train-vram", action="store_true",
                    help="Skip the peak training-step VRAM measurement (step 7).")
    ap.add_argument("--latency-runs", type=int, default=50, help="Timed generate iterations.")
    ap.add_argument("--latency-warmup", type=int, default=10, help="Discarded warm-up iterations.")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    todo = MODELS if not args.models else [m for m in MODELS if m[0] in args.models]
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"Measuring {len(todo)} model(s) | device={dev} | "
          f"mode={'latency-only' if args.latency_only else 'full'}\n", flush=True)
    if torch.cuda.is_available() and "4090" not in dev:
        print(f"!! WARNING: latency is being measured on '{dev}', not a desktop RTX 4090.\n"
              f"   Report latency only for the hardware actually used, or re-run on the 4090.\n", flush=True)

    rows = []
    for mid, base_repo, rank, adapter_repo in todo:
        print(f"===== {mid} =====", flush=True)
        try:
            r = measure(mid, base_repo, rank, adapter_repo, token, args)
            rows.append(r)
            print(f"  gpu_latency={r['gpu_latency_ms']:.1f}ms ({r['input_tokens']} in-tok)  "
                  f"nf4={r['nf4_4bit_gb']:.2f}GB  inferVRAM={r['peak_infer_vram_gb']:.2f}GB"
                  + ("" if args.latency_only else
                     f"  base={r['base_bf16_gb']:.2f}GB  trainable={r['trainable_params_M']:.1f}M  "
                     f"adapter={r['adapter_disk_mb']:.0f}MB  merged={r['merged_bf16_gb']:.2f}GB"),
                  flush=True)
        except Exception as e:
            print(f"  {mid} FAILED: {type(e).__name__}: {e}", flush=True)

    if rows:
        out = REPO_ROOT / "runs" / "footprints"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out / "footprints.csv", index=False)
        print(f"\nCSV -> {out/'footprints.csv'}  (device: {dev})\n")
        if args.latency_only:
            print("% paste-ready: Model & GPU latency (ms)")
            for r in rows:
                print(f"{r['model']} & $\\sim${r['gpu_latency_ms']:.0f}\\,ms \\\\")
        else:
            print("% paste-ready: Base BF16 | 4-bit NF4 | trainable | adapter | merged | GPU lat")
            for r in rows:
                print(f"{r['model']} & {r['total_params_M']:.0f}M & {r['base_bf16_gb']:.2f} & "
                      f"{r['nf4_4bit_gb']:.2f} & {r['trainable_params_M']:.1f}M & "
                      f"{r['adapter_disk_mb']:.0f}\\,MB & {r['merged_bf16_gb']:.2f} & "
                      f"$\\sim${r['gpu_latency_ms']:.0f}\\,ms \\\\")

    print("\n# CPU (Q4_K_M GGUF) latency + peak RAM -- run on the claimed CPU via llama.cpp:")
    print("#   python llama.cpp/convert_hf_to_gguf.py <merged_dir> --outfile m.f16.gguf")
    print("#   ./llama.cpp/llama-quantize m.f16.gguf m.Q4_K_M.gguf Q4_K_M       # -> GGUF size")
    print("#   /usr/bin/time -v ./llama.cpp/llama-cli -m m.Q4_K_M.gguf -t 16 -n 5 \\")
    print("#       -p '<the ~150-token prompt>' 2>&1 | grep -E 'prompt eval time|eval time|resident'")
    print("#   CPU latency = (prompt eval time) + (eval time);  peak RAM = 'Maximum resident set size'")


if __name__ == "__main__":
    main()

"""
evaluation/benchmark.py
Benchmark InsureAgent inference across three configurations:
  1. Teacher API   — GPT-4o mini via OpenAI API
  2. Student FP16  — Llama-3.2-1B + LoRA, float16
  3. Student INT8  — Llama-3.2-1B + LoRA, int8 (bitsandbytes)

Run on Colab (T4 GPU recommended):
    python evaluation/benchmark.py --config config/config.yaml

Results saved to evaluation/quantisation_results.json
"""

import argparse
import json
import os
import re
import time
import yaml
import torch

from huggingface_hub import login


# ── CONFIG ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── AUTH ──────────────────────────────────────────────────────────────────────

def hf_login(token: str | None = None) -> str:
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set.")
    login(token=hf_token)
    return hf_token


# ── BENCHMARK CASES (5 representative cases from test set) ───────────────────

BENCHMARK_CASES = [
    {
        "user_id": "P-1021",
        "claim_text": "Strong winds from a storm knocked over my garden fence and damaged my shed roof.",
        "claimed_amount": 2800,
        "expected_verdict": "APPROVED",
    },
    {
        "user_id": "P-1009",
        "claim_text": "Heavy flooding in my area caused water to enter my ground floor rooms.",
        "claimed_amount": 12000,
        "expected_verdict": "APPROVED",
    },
    {
        "user_id": "P-1019",
        "claim_text": "My basement flooded after a heavy rainfall ruined my furniture.",
        "claimed_amount": 6000,
        "expected_verdict": "DENIED",
    },
    {
        "user_id": "P-1030",
        "claim_text": "A major collision on the motorway totalled my car completely.",
        "claimed_amount": 45000,
        "expected_verdict": "APPROVED",
    },
    {
        "user_id": "P-1022",
        "claim_text": "My car was stolen from the street outside my house last night.",
        "claimed_amount": 15000,
        "expected_verdict": "DENIED",
    },
]


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

from agent.prompts import SYSTEM_PROMPT


# ── TOOLS ─────────────────────────────────────────────────────────────────────

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.database import lookup_policy
from tools.rules import check_rules
from tools.calculator import calculate_payout

TOOL_REGISTRY = {
    "lookup_policy":    lookup_policy,
    "check_rules":      check_rules,
    "calculate_payout": calculate_payout,
}


# ── PARSER ────────────────────────────────────────────────────────────────────

def parse_action(text: str):
    match = re.search(r'Action:\s*(\w+)\s*\((.*)\)', text, re.DOTALL)
    if not match:
        return None, None
    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()
    if args_str.startswith('{'):
        for attempt in [args_str, args_str.replace("'", '"')]:
            try:
                args = json.loads(attempt)
                if tool_name == "calculate_payout":
                    args.setdefault("already_claimed_this_year", 0)
                return tool_name, args
            except json.JSONDecodeError:
                pass
    return tool_name, None


# ── EXTRACTION ────────────────────────────────────────────────────────────────

def extract_verdict(trace: list) -> str:
    for msg in reversed(trace):
        if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
            if "APPROVED" in msg["content"]:
                return "APPROVED"
            if "DENIED" in msg["content"]:
                return "DENIED"
    return "NO_VERDICT"


# ── TEACHER BENCHMARK ─────────────────────────────────────────────────────────

def benchmark_teacher(cases: list) -> dict:
    """Benchmark Teacher (GPT-4o mini) — measures API latency."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    latencies = []
    verdicts = []

    print("\n[Teacher] GPT-4o mini via OpenAI API")
    print("-" * 40)

    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Process this claim from policyholder {case['user_id']}: "
                f"{case['claim_text']} "
                f"The policyholder is claiming ${case['claimed_amount']}."
            )}
        ]
        trace = [messages[1]]
        start = time.time()

        for _ in range(8):
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
            )
            output = response.choices[0].message.content
            trace.append({"role": "assistant", "content": output})
            messages.append({"role": "assistant", "content": output})

            if "Verdict:" in output:
                break

            tool_name, args = parse_action(output)
            if tool_name and args and tool_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[tool_name](**args)
                obs = f"Observation: {json.dumps(result)}"
            else:
                obs = "Observation: Error — could not parse action."
            trace.append({"role": "user", "content": obs})
            messages.append({"role": "user", "content": obs})

        latency = (time.time() - start) * 1000
        verdict = extract_verdict(trace)
        latencies.append(latency)
        verdicts.append(verdict == case["expected_verdict"])
        print(f"  {verdict} | {latency:.0f}ms | {'✓' if verdict == case['expected_verdict'] else '✗'}")

    return {
        "config": "teacher_api",
        "model": "gpt-4o-mini",
        "precision": "API",
        "model_size_gb": None,
        "verdict_accuracy": sum(verdicts) / len(verdicts),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1),
            "min":  round(min(latencies), 1),
            "max":  round(max(latencies), 1),
        },
        "memory_gb": None,
        "n_cases": len(cases),
    }


# ── STUDENT BENCHMARK ─────────────────────────────────────────────────────────

def _load_model(model_name: str, adapter_path: str, hf_token: str,
                quantisation: str = "fp16"):
    """
    Load Student model with specified quantisation.

    quantisation:
      "fp16" — standard float16 (default training precision)
      "int8" — 8-bit quantisation via bitsandbytes (~50% memory reduction)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n[Student {quantisation.upper()}] Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    if quantisation == "int8":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            quantization_config=bnb_config,
            device_map="auto",
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)
    model.eval()
    return model, tokenizer


def _get_memory_gb() -> float:
    """Return current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return round(torch.cuda.memory_allocated() / 1e9, 2)
    return 0.0


def _generate(model, tokenizer, messages: list, max_new_tokens: int = 512) -> str:
    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def benchmark_student(cases: list, model_name: str, adapter_path: str,
                       hf_token: str, quantisation: str = "fp16") -> dict:
    """Benchmark Student model at given quantisation level."""

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, tokenizer = _load_model(model_name, adapter_path, hf_token, quantisation)
    memory_after_load = _get_memory_gb()
    print(f"  Memory after load: {memory_after_load:.2f} GB")
    print("-" * 40)

    latencies = []
    verdicts = []

    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Process this claim from policyholder {case['user_id']}: "
                f"{case['claim_text']} "
                f"The policyholder is claiming ${case['claimed_amount']}."
            )}
        ]
        trace = [messages[1]]
        start = time.time()

        for _ in range(8):
            output = _generate(model, tokenizer, messages)
            trace.append({"role": "assistant", "content": output})
            messages.append({"role": "assistant", "content": output})

            if "Verdict:" in output:
                break
            if "Observation:" in output:
                break

            tool_name, args = parse_action(output)
            if tool_name and args and tool_name in TOOL_REGISTRY:
                result = TOOL_REGISTRY[tool_name](**args)
                obs = f"Observation: {json.dumps(result)}"
            else:
                obs = "Observation: Error — could not parse action."
            trace.append({"role": "user", "content": obs})
            messages.append({"role": "user", "content": obs})

        latency = (time.time() - start) * 1000
        verdict = extract_verdict(trace)
        latencies.append(latency)
        verdicts.append(verdict == case["expected_verdict"])
        print(f"  {verdict} | {latency:.0f}ms | {'✓' if verdict == case['expected_verdict'] else '✗'}")

    peak_memory = _get_memory_gb()

    # Clean up
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "config": f"student_{quantisation}",
        "model": model_name,
        "adapter": adapter_path,
        "precision": quantisation.upper(),
        "model_size_gb": memory_after_load,
        "verdict_accuracy": round(sum(verdicts) / len(verdicts), 3),
        "latency_ms": {
            "mean": round(sum(latencies) / len(latencies), 1),
            "min":  round(min(latencies), 1),
            "max":  round(max(latencies), 1),
        },
        "memory_gb": peak_memory,
        "n_cases": len(cases),
    }


# ── REPORT ────────────────────────────────────────────────────────────────────

def print_report(results: list):
    print("\n" + "=" * 60)
    print("QUANTISATION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"{'Config':<20} {'Precision':<10} {'Verdict Acc':<14} {'Mean Latency':<15} {'Memory'}")
    print("-" * 60)
    for r in results:
        memory = f"{r['memory_gb']:.2f} GB" if r["memory_gb"] else "N/A (API)"
        print(
            f"{r['config']:<20} "
            f"{r['precision']:<10} "
            f"{r['verdict_accuracy']*100:.1f}%{'':8} "
            f"{r['latency_ms']['mean']:.0f}ms{'':9} "
            f"{memory}"
        )
    print("=" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsureAgent quantisation benchmark")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--skip-fp16", action="store_true")
    parser.add_argument("--skip-int8", action="store_true")
    parser.add_argument("--output", default="evaluation/quantisation_results.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    hf_token = hf_login(args.hf_token)

    results = []

    if not args.skip_teacher:
        results.append(benchmark_teacher(BENCHMARK_CASES))

    if not args.skip_fp16:
        results.append(benchmark_student(
            BENCHMARK_CASES,
            model_name=cfg["model"]["student_base"],
            adapter_path=cfg["model"]["adapter"],
            hf_token=hf_token,
            quantisation="fp16",
        ))

    if not args.skip_int8:
        results.append(benchmark_student(
            BENCHMARK_CASES,
            model_name=cfg["model"]["student_base"],
            adapter_path=cfg["model"]["adapter"],
            hf_token=hf_token,
            quantisation="int8",
        ))

    print_report(results)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
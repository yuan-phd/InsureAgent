"""
api/inference.py
Model loading and agent execution for InsureAgent FastAPI server.

Supports two modes:
  - teacher:  GPT-4o mini via OpenAI API (default)
  - student:  Llama-3.2-1B + LoRA adapter via HuggingFace

The mode is selected per-request via ClaimRequest.model field.
"""

import json
import os
import re
import time

from dotenv import load_dotenv

load_dotenv()

# ── TOOL REGISTRY ─────────────────────────────────────────────────────────────

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

# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

from agent.prompts import SYSTEM_PROMPT

# ── PARSER: TEACHER (JSON only) ───────────────────────────────────────────────

from agent.parser import parse_action as parse_action_teacher

# ── PARSER: STUDENT (JSON + kwargs + positional) ──────────────────────────────

POSITIONAL_PARAMS = {
    "lookup_policy": ["user_id"],
    "check_rules": ["claim_type", "plan_type", "policy_covers",
                    "policy_status", "claims_this_year"],
    "calculate_payout": ["claimed_amount", "deductible", "max_single_claim",
                         "max_annual_payout", "already_claimed_this_year"],
}


def parse_action_student(text: str):
    """
    Extended parser for Student model output.
    Handles three formats:
      1. JSON:       tool_name({"key": value})
      2. kwargs:     tool_name(key=value, key=value)
      3. positional: tool_name(a, b, c)
    """
    match = re.search(r'Action:\s*(\w+)\s*\((.*)\)', text, re.DOTALL)
    if not match:
        return None, None

    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()

    # Format 1: JSON
    if args_str.startswith('{'):
        for attempt in [args_str, args_str.replace("'", '"')]:
            try:
                args = json.loads(attempt)
                if tool_name == "calculate_payout":
                    args.setdefault("already_claimed_this_year", 0)
                return tool_name, args
            except json.JSONDecodeError:
                pass

    # Format 2: kwargs key=value
    if '=' in args_str:
        try:
            clean = re.sub(r'\s+', ' ', args_str.split('\n')[0].strip())
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?=,\s*\w+\s*=|$)', clean)
            if pairs:
                args = {}
                for k, v in pairs:
                    v = v.strip()
                    try:
                        args[k] = json.loads(v)
                    except json.JSONDecodeError:
                        args[k] = v.strip('"\'')
                if tool_name == "calculate_payout":
                    args.setdefault("already_claimed_this_year", 0)
                if args:
                    return tool_name, args
        except Exception:
            pass

    # Format 3: positional
    params = POSITIONAL_PARAMS.get(tool_name, [])
    if params:
        try:
            raw_vals = [v.strip() for v in args_str.split(',')]
            args = {}
            for i, v in enumerate(raw_vals):
                if i < len(params):
                    try:
                        args[params[i]] = json.loads(v)
                    except json.JSONDecodeError:
                        args[params[i]] = v.strip('"\'')
            if tool_name == "calculate_payout":
                args.setdefault("already_claimed_this_year", 0)
            if args:
                return tool_name, args
        except Exception:
            pass

    return tool_name, None


# ── TEACHER INFERENCE ─────────────────────────────────────────────────────────

def _run_teacher(claim_text: str, user_id: str, claimed_amount: float,
                 max_steps: int = 8) -> list:
    """Run agent loop with GPT-4o mini (Teacher)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Process this claim from policyholder {user_id}: "
            f"{claim_text} "
            f"The policyholder is claiming ${claimed_amount}."
        )}
    ]
    trace = [messages[1]]

    for _ in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        model_output = response.choices[0].message.content
        trace.append({"role": "assistant", "content": model_output})
        messages.append({"role": "assistant", "content": model_output})

        if "Verdict:" in model_output:
            break
        if "Observation:" in model_output:
            trace.append({"role": "system", "content": "[REJECTED: hallucination]"})
            break

        tool_name, args = parse_action_teacher(model_output)
        observation = _execute_tool(tool_name, args)

        trace.append({"role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})

    return trace


# ── STUDENT INFERENCE ─────────────────────────────────────────────────────────

# Module-level cache — model loaded once per server process
_student_model = None
_student_tokenizer = None


def _load_student_model():
    """Load Student model and adapter. Cached after first call."""
    global _student_model, _student_tokenizer

    if _student_model is not None:
        return _student_model, _student_tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    adapter_path = "yuanphd/insureagent-lora-v2"

    print("Loading Student base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)
    model.eval()

    _student_model = model
    _student_tokenizer = tokenizer
    print("Student model ready.")
    return model, tokenizer


def _generate_student(messages: list, max_new_tokens: int = 512) -> str:
    import torch
    model, tokenizer = _load_student_model()
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


def _run_student(claim_text: str, user_id: str, claimed_amount: float,
                 max_steps: int = 8) -> list:
    """Run agent loop with Llama-3.2-1B + LoRA adapter (Student)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Process this claim from policyholder {user_id}: "
            f"{claim_text} "
            f"The policyholder is claiming ${claimed_amount}."
        )}
    ]
    trace = [messages[1]]

    for _ in range(max_steps):
        model_output = _generate_student(messages)
        trace.append({"role": "assistant", "content": model_output})
        messages.append({"role": "assistant", "content": model_output})

        if "Verdict:" in model_output:
            break
        if "Observation:" in model_output:
            trace.append({"role": "system", "content": "[REJECTED: hallucination]"})
            break

        tool_name, args = parse_action_student(model_output)
        observation = _execute_tool(tool_name, args)

        trace.append({"role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})

    return trace


# ── SHARED TOOL EXECUTION ─────────────────────────────────────────────────────

def _execute_tool(tool_name, args) -> str:
    if tool_name is None:
        return "Observation: Error — no valid Action found."
    if tool_name not in TOOL_REGISTRY:
        return f"Observation: Error — unknown tool '{tool_name}'."
    if args is None:
        return f"Observation: Error — could not parse arguments for '{tool_name}'."
    try:
        result = TOOL_REGISTRY[tool_name](**args)
        return f"Observation: {json.dumps(result)}"
    except TypeError as e:
        return f"Observation: Error — wrong arguments for '{tool_name}': {e}"


# ── EXTRACTION HELPERS ────────────────────────────────────────────────────────

def extract_verdict(trace: list) -> str:
    for msg in reversed(trace):
        if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
            if "APPROVED" in msg["content"]:
                return "APPROVED"
            if "DENIED" in msg["content"]:
                return "DENIED"
    return "NO_VERDICT"


def extract_payout(trace: list) -> float:
    for msg in reversed(trace):
        if msg["role"] == "assistant" and "Payout:" in msg["content"]:
            match = re.search(r'Payout:\s*\$?([\d,]+\.?\d*)', msg["content"])
            if match:
                return float(match.group(1).replace(',', ''))
    return 0.0


def extract_reasoning(trace: list) -> str:
    for msg in reversed(trace):
        if msg["role"] == "assistant" and "Reasoning:" in msg["content"]:
            match = re.search(r'Reasoning:\s*(.+)', msg["content"], re.DOTALL)
            if match:
                return match.group(1).strip()
    return ""


# ── PUBLIC API ────────────────────────────────────────────────────────────────

def process_claim(claim_text: str, user_id: str, claimed_amount: float,
                  model: str = "teacher") -> dict:
    """
    Main entry point for FastAPI.
    Runs the full agent loop and returns a structured result dict.

    Args:
        claim_text:     description of the claim
        user_id:        policyholder ID e.g. "P-1001"
        claimed_amount: amount claimed in dollars
        model:          "teacher" or "student"

    Returns:
        dict with verdict, payout, reasoning, trace, latency_ms, model_used
    """
    start = time.time()

    if model == "student":
        trace = _run_student(claim_text, user_id, claimed_amount)
    else:
        trace = _run_teacher(claim_text, user_id, claimed_amount)

    latency_ms = round((time.time() - start) * 1000, 1)

    return {
        "verdict":    extract_verdict(trace),
        "payout":     extract_payout(trace),
        "reasoning":  extract_reasoning(trace),
        "trace":      trace,
        "latency_ms": latency_ms,
        "model_used": model,
    }
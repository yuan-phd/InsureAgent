"""
evaluation/student_eval.py
Evaluate InsureAgent Student model (Llama-3.2-1B + LoRA adapter).

Usage:
    python evaluation/student_eval.py --config config/config.yaml
    python evaluation/student_eval.py --config config/config.yaml --save-results
"""

import argparse
import json
import os
import re
import sqlite3
import yaml
import torch
from collections import defaultdict

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from huggingface_hub import login

from agent.classifier import classify_risk


# ── CONFIG ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── AUTH ──────────────────────────────────────────────────────────────────────

def hf_login(token: str | None = None):
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set.")
    login(token=hf_token)
    return hf_token


# ── SYSTEM PROMPT ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert insurance claims adjuster. You process claims by reasoning step-by-step and calling tools to look up real data.

Available tools:
[
  {
    "name": "lookup_policy",
    "description": "Look up a policyholder's coverage details by their user ID. Always call this first.",
    "parameters": {
      "user_id": {"type": "string", "description": "The policyholder ID, e.g. 'P-1001'"}
    }
  },
  {
    "name": "check_rules",
    "description": "Check if a specific claim type is eligible under the policyholder's plan.",
    "parameters": {
      "claim_type": {"type": "string", "description": "Type of damage: storm, hail, theft, fire, flood, or collision"},
      "plan_type": {"type": "string", "description": "The policyholder's plan: Basic, Standard, or Premium"},
      "policy_covers": {"type": "list", "description": "List of covered claim types from the policy lookup"},
      "policy_status": {"type": "string", "description": "Policy status: active, lapsed, or cancelled"},
      "claims_this_year": {"type": "integer", "description": "Number of claims already filed this year"}
    }
  },
  {
    "name": "calculate_payout",
    "description": "Calculate the payout for an eligible claim. Only call if check_rules returns eligible=true.",
    "parameters": {
      "claimed_amount": {"type": "float", "description": "Amount the policyholder is claiming in dollars"},
      "deductible": {"type": "float", "description": "Policy deductible from the policy lookup"},
      "max_single_claim": {"type": "float", "description": "Maximum allowed for a single claim from check_rules"},
      "max_annual_payout": {"type": "float", "description": "Maximum annual payout from the policy lookup"},
      "already_claimed_this_year": {"type": "float", "description": "Total already claimed this year, use 0 if unknown"}
    }
  }
]

You MUST follow this EXACT format for every step:

Thought: [your reasoning about what to do next]
Action: tool_name({"param": "value"})

After each Action, you will receive an Observation with the real tool result.
Use that result in your next Thought.

When you have enough information to make a final decision, output:

Verdict: APPROVED or DENIED
Payout: $[amount]
Reasoning: [2-3 sentences citing specific data from the tool results]

Rules:
- Always call lookup_policy FIRST with only user_id
- Always call check_rules SECOND
- Only call calculate_payout if check_rules returns eligible=true
- If eligible=false, go directly to Verdict: DENIED, Payout: $0
- Never call a tool more than once
- Never invent or guess tool results
- Never output an Observation yourself
"""


# ── TOOLS ─────────────────────────────────────────────────────────────────────

def make_tools(db_path: str):
    def lookup_policy(user_id: str) -> dict:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT * FROM policies WHERE user_id = ?", (user_id,))
        row = c.fetchone()
        conn.close()
        if not row:
            return {"error": f"No policy found for user_id: {user_id}"}
        return {
            "user_id": row[0], "name": row[1], "plan_type": row[2],
            "covers": json.loads(row[3]), "deductible": row[4],
            "max_annual_payout": row[5], "status": row[6],
            "claims_this_year": row[7],
        }

    def check_rules(claim_type, plan_type, policy_covers,
                    policy_status, claims_this_year) -> dict:
        COVERAGE_RULES = {
            "storm":     {"requires_inspection": False, "max_single_claim": 15000},
            "hail":      {"requires_inspection": False, "max_single_claim": 15000},
            "theft":     {"requires_inspection": True,  "max_single_claim": 25000},
            "fire":      {"requires_inspection": True,  "max_single_claim": 40000},
            "flood":     {"requires_inspection": True,  "max_single_claim": 35000},
            "collision": {"requires_inspection": True,  "max_single_claim": 20000},
        }
        MAX_CLAIMS = {"Basic": 2, "Standard": 4, "Premium": 10}
        if policy_status != "active":
            return {"eligible": False, "reason": f"Policy status is '{policy_status}'."}
        if claim_type not in policy_covers:
            return {"eligible": False, "reason": f"'{claim_type}' not covered."}
        if claims_this_year >= MAX_CLAIMS.get(plan_type, 2):
            return {"eligible": False, "reason": "Annual claim limit reached."}
        rule = COVERAGE_RULES.get(claim_type)
        if not rule:
            return {"eligible": False, "reason": "Unknown claim type."}
        return {"eligible": True, "max_single_claim": rule["max_single_claim"],
                "requires_inspection": rule["requires_inspection"]}

    def calculate_payout(claimed_amount, deductible, max_single_claim,
                         max_annual_payout, already_claimed_this_year=0) -> dict:
        if claimed_amount <= deductible:
            return {"payout": 0, "reason": "Below deductible."}
        after_ded = claimed_amount - deductible
        capped = min(after_ded, max_single_claim)
        remaining = max_annual_payout - already_claimed_this_year
        payout = round(min(capped, remaining), 2)
        return {"payout": payout}

    return {
        "lookup_policy": lookup_policy,
        "check_rules": check_rules,
        "calculate_payout": calculate_payout,
    }


# ── PARSER ────────────────────────────────────────────────────────────────────

POSITIONAL_PARAMS = {
    "lookup_policy": ["user_id"],
    "check_rules": ["claim_type", "plan_type", "policy_covers",
                    "policy_status", "claims_this_year"],
    "calculate_payout": ["claimed_amount", "deductible", "max_single_claim",
                         "max_annual_payout", "already_claimed_this_year"],
}


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


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_student_model(model_name: str, adapter_path: str, hf_token: str):
    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading adapter: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path, token=hf_token)
    model.eval()
    print("Student model ready.")
    return model, tokenizer


# ── INFERENCE ─────────────────────────────────────────────────────────────────

def generate_response(model, tokenizer, messages: list,
                      max_new_tokens: int = 512) -> str:
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


# ── AGENT LOOP ────────────────────────────────────────────────────────────────

def run_student_agent(model, tokenizer, tool_registry: dict,
                      claim_text: str, user_id: str, claimed_amount: float,
                      max_steps: int = 8) -> list:
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
        model_output = generate_response(model, tokenizer, messages)
        trace.append({"role": "assistant", "content": model_output})
        messages.append({"role": "assistant", "content": model_output})

        if "Verdict:" in model_output:
            break
        if "Observation:" in model_output:
            trace.append({"role": "system", "content": "[REJECTED: hallucination]"})
            break

        tool_name, args = parse_action(model_output)

        if tool_name is None:
            observation = "Observation: Error — no valid Action found."
        elif tool_name not in tool_registry:
            observation = f"Observation: Error — unknown tool '{tool_name}'."
        elif args is None:
            observation = "Observation: Error — could not parse arguments."
        else:
            try:
                result = tool_registry[tool_name](**args)
                observation = f"Observation: {json.dumps(result)}"
            except TypeError as e:
                observation = f"Observation: Error — {e}"

        trace.append({"role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})

    return trace


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
    return -1.0


def extract_tools(trace: list) -> list:
    tools = []
    for msg in trace:
        if msg["role"] == "assistant":
            tools.extend(re.findall(r'Action:\s*(\w+)\s*\(', msg["content"]))
    return tools


# ── TEST CASES ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {"user_id": "P-1021", "claim_text": "Strong winds from a storm knocked over my garden fence and damaged my shed roof.", "claimed_amount": 2800, "expected_verdict": "APPROVED", "expected_payout": 2050, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1024", "claim_text": "Hailstorm caused multiple dents on my car bonnet and cracked the windshield.", "claimed_amount": 3200, "expected_verdict": "APPROVED", "expected_payout": 2700, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1030", "claim_text": "An electrical fire in my garage destroyed my car and damaged the walls.", "claimed_amount": 22000, "expected_verdict": "APPROVED", "expected_payout": 21500, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1017", "claim_text": "My bicycle and laptop were stolen from my car while parked downtown.", "claimed_amount": 4500, "expected_verdict": "APPROVED", "expected_payout": 3750, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1029", "claim_text": "Storm damaged my roof and water leaked into the bedroom causing ceiling damage.", "claimed_amount": 5500, "expected_verdict": "APPROVED", "expected_payout": 4750, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1009", "claim_text": "Heavy flooding in my area caused water to enter my ground floor rooms.", "claimed_amount": 12000, "expected_verdict": "APPROVED", "expected_payout": 11500, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1012", "claim_text": "A collision on the motorway caused significant damage to the front of my car.", "claimed_amount": 8000, "expected_verdict": "APPROVED", "expected_payout": 7500, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1019", "claim_text": "My basement flooded after a heavy rainfall ruined my furniture.", "claimed_amount": 6000, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1010", "claim_text": "Storm winds damaged my garden fence and outdoor furniture.", "claimed_amount": 1800, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1025", "claim_text": "A hailstorm dented my car roof and cracked a window.", "claimed_amount": 2200, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1013", "claim_text": "A fire in my kitchen caused damage to the cabinets and appliances.", "claimed_amount": 7000, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1022", "claim_text": "My car was stolen from the street outside my house last night.", "claimed_amount": 15000, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1028", "claim_text": "Storm damage cracked several roof tiles and broke a skylight.", "claimed_amount": 3500, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1011", "claim_text": "Hail damaged my car windshield during a storm yesterday.", "claimed_amount": 1500, "expected_verdict": "DENIED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules"]},
    {"user_id": "P-1021", "claim_text": "A small hail shower left a few minor dents on my car door.", "claimed_amount": 400, "expected_verdict": "APPROVED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1001", "claim_text": "Minor scratches on my bumper from a low speed parking collision.", "claimed_amount": 350, "expected_verdict": "APPROVED", "expected_payout": 0, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1030", "claim_text": "A major collision on the motorway totalled my car completely.", "claimed_amount": 45000, "expected_verdict": "APPROVED", "expected_payout": 19500, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1009", "claim_text": "Severe flooding destroyed all ground floor furniture and flooring.", "claimed_amount": 60000, "expected_verdict": "APPROVED", "expected_payout": 34500, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
    {"user_id": "P-1015", "claim_text": "Storm damaged my conservatory roof and cracked several windows.", "claimed_amount": 4000, "expected_verdict": "APPROVED", "expected_payout": 3250, "expected_tools": ["lookup_policy", "check_rules", "calculate_payout"]},
]


# ── EVALUATION ────────────────────────────────────────────────────────────────

def run_evaluation(model, tokenizer, tool_registry: dict,
                   max_new_tokens: int = 512) -> list:
    results = []
    print("Evaluating Student v2 (Llama-3.2-1B + LoRA v2)")
    print("-" * 50)

    for i, case in enumerate(TEST_CASES):
        print(f"Case {i+1}/{len(TEST_CASES)}: {case['claim_text'][:50]}...")

        # ── Classifier (before agent, logging only) ──
        risk_result = classify_risk(case["claim_text"])
        risk_level      = risk_result["risk_level"]
        risk_confidence = risk_result["confidence"]

        try:
            trace = run_student_agent(
                model, tokenizer, tool_registry,
                claim_text=case["claim_text"],
                user_id=case["user_id"],
                claimed_amount=case["claimed_amount"],
                max_steps=8,
            )
            predicted_verdict = extract_verdict(trace)
            predicted_payout  = extract_payout(trace)
            predicted_tools   = extract_tools(trace)

            verdict_correct = predicted_verdict == case["expected_verdict"]
            payout_correct  = abs(predicted_payout - case["expected_payout"]) < 1.0
            tools_correct   = predicted_tools == case["expected_tools"]

            status = "✓" if verdict_correct else "✗"
            print(f"  {status} {predicted_verdict} | "
                  f"Payout: ${predicted_payout} | "
                  f"Risk: {risk_level} ({risk_confidence})")

            results.append({
                "case_id":          i + 1,
                "user_id":          case["user_id"],
                "claim_text":       case["claim_text"],
                "risk_level":       risk_level,
                "risk_confidence":  risk_confidence,
                "expected_verdict": case["expected_verdict"],
                "predicted_verdict": predicted_verdict,
                "expected_payout":  case["expected_payout"],
                "predicted_payout": predicted_payout,
                "verdict_correct":  verdict_correct,
                "payout_correct":   payout_correct,
                "tools_correct":    tools_correct,
            })

        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "case_id":          i + 1,
                "user_id":          case["user_id"],
                "claim_text":       case["claim_text"],
                "risk_level":       risk_level,
                "risk_confidence":  risk_confidence,
                "expected_verdict": case["expected_verdict"],
                "predicted_verdict": "ERROR",
                "expected_payout":  case["expected_payout"],
                "predicted_payout": -1,
                "verdict_correct":  False,
                "payout_correct":   False,
                "tools_correct":    False,
            })

    return results


# ── SUMMARY ───────────────────────────────────────────────────────────────────

def print_summary(results: list):
    total = len(results)

    print(f"\n{'='*50}")
    print("RESULTS: Student v2 (Llama-3.2-1B + LoRA v2)")
    print(f"{'='*50}")
    print(f"Verdict accuracy:  {sum(r['verdict_correct'] for r in results)/total*100:.1f}%  ({sum(r['verdict_correct'] for r in results)}/{total})")
    print(f"Payout precision:  {sum(r['payout_correct'] for r in results)/total*100:.1f}%  ({sum(r['payout_correct'] for r in results)}/{total})")
    print(f"Tool sequence acc: {sum(r['tools_correct'] for r in results)/total*100:.1f}%  ({sum(r['tools_correct'] for r in results)}/{total})")

    # ── Breakdown by risk level ──
    print(f"\n{'─'*50}")
    print("VERDICT ACCURACY BY RISK LEVEL")
    print(f"{'─'*50}")

    by_risk = defaultdict(list)
    for r in results:
        by_risk[r["risk_level"]].append(r)

    for level in ["high", "medium", "low"]:
        group = by_risk.get(level, [])
        if not group:
            continue
        acc = sum(r["verdict_correct"] for r in group) / len(group) * 100
        avg_conf = sum(r["risk_confidence"] for r in group) / len(group)
        print(f"  {level:6s}  n={len(group):2d}  "
              f"verdict accuracy: {acc:.0f}%  "
              f"avg confidence: {avg_conf:.2f}")

    # ── Interesting failure cases ──
    print(f"\n{'─'*50}")
    print("HIGH-CONFIDENCE FAILURES (most interesting)")
    print(f"{'─'*50}")

    failures = [
        r for r in results
        if not r["verdict_correct"] and r["risk_confidence"] >= 0.80
    ]

    if not failures:
        print("  None — all high-confidence cases were correct.")
    else:
        for r in sorted(failures, key=lambda x: x["risk_confidence"], reverse=True):
            print(f"  Case {r['case_id']:2d} | "
                  f"risk: {r['risk_level']} ({r['risk_confidence']}) | "
                  f"expected: {r['expected_verdict']} | "
                  f"predicted: {r['predicted_verdict']}")
            print(f"         {r['claim_text'][:70]}...")

    print(f"{'='*50}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsureAgent Student evaluation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--db", default="insurance.db")
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--save-results", action="store_true")
    args = parser.parse_args()

    cfg      = load_config(args.config)
    hf_token = hf_login(args.hf_token)

    model, tokenizer = load_student_model(
        model_name=cfg["model"]["student_base"],
        adapter_path=cfg["model"]["adapter"],
        hf_token=hf_token,
    )

    tool_registry = make_tools(args.db)
    results = run_evaluation(model, tokenizer, tool_registry,
                             max_new_tokens=cfg["inference"]["max_new_tokens"])
    print_summary(results)

    if args.save_results:
        out_path = "evaluation/results.json"
        os.makedirs("evaluation", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
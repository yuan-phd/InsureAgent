# calling gpt to generate Teacher baseline

import json
import os
import sys
import re
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from dotenv import load_dotenv

from agent.loop import run_agent
from data.test_cases import TEST_CASES

load_dotenv()

# ── EXTRACTION HELPERS ───────────────────────────────────────

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
    return -1

def extract_tool_calls(trace: list) -> list:
    tools = []
    for msg in trace:
        if msg["role"] == "assistant":
            matches = re.findall(r'Action:\s*(\w+)\s*\(', msg["content"])
            tools.extend(matches)
    return tools

def categorise_error(trace: list, expected_verdict: str,
                     predicted_verdict: str, expected_tools: list,
                     predicted_tools: list) -> str:
    if predicted_verdict == "NO_VERDICT":
        return "no_verdict"
    for msg in trace:
        if msg["role"] == "system" and "REJECTED" in msg.get("content", ""):
            return "hallucination"
    if predicted_tools and predicted_tools[0] != expected_tools[0]:
        return "wrong_tool"
    if set(predicted_tools) != set(expected_tools):
        return "wrong_tool_sequence"
    if predicted_verdict != expected_verdict:
        return "wrong_verdict"
    return "wrong_payout"

# ── EVALUATION RUNNER ────────────────────────────────────────

def evaluate_model(client, model_label: str, delay: float = 0.5) -> list:
    results = []
    print(f"\nEvaluating: {model_label}")
    print("-" * 50)

    for i, case in enumerate(TEST_CASES):
        print(f"  Case {i+1}/{len(TEST_CASES)}: {case['claim_text'][:45]}...")

        try:
            result = run_agent(
                claim_text=case["claim_text"],
                user_id=case["user_id"],
                claimed_amount=case["claimed_amount"],
                client=client
            )
            trace = result["trace"]
        except Exception as e:
            print(f"    ERROR: {e}")
            results.append({
                "case": case,
                "predicted_verdict": "ERROR",
                "predicted_payout": -1,
                "predicted_tools": [],
                "verdict_correct": False,
                "payout_correct": False,
                "tools_correct": False,
                "error_category": "exception"
            })
            continue

        predicted_verdict = extract_verdict(trace)
        predicted_payout  = extract_payout(trace)
        predicted_tools   = extract_tool_calls(trace)

        verdict_correct = predicted_verdict == case["expected_verdict"]
        payout_correct  = abs(predicted_payout - case["expected_payout"]) < 1.0
        tools_correct   = predicted_tools == case["expected_tools"]

        error_category = None
        if not (verdict_correct and payout_correct and tools_correct):
            error_category = categorise_error(
                trace, case["expected_verdict"], predicted_verdict,
                case["expected_tools"], predicted_tools
            )

        status = "✓" if verdict_correct else "✗"
        print(f"    {status} Verdict: {predicted_verdict} "
              f"(expected {case['expected_verdict']}) | "
              f"Payout: ${predicted_payout} (expected ${case['expected_payout']})")

        results.append({
            "case": case,
            "predicted_verdict": predicted_verdict,
            "predicted_payout": predicted_payout,
            "predicted_tools": predicted_tools,
            "verdict_correct": verdict_correct,
            "payout_correct": payout_correct,
            "tools_correct": tools_correct,
            "error_category": error_category
        })

        time.sleep(delay)

    return results

# ── RESULTS SUMMARY ──────────────────────────────────────────

def print_summary(results: list, label: str):
    total = len(results)
    verdict_acc = sum(r["verdict_correct"] for r in results) / total * 100
    payout_acc  = sum(r["payout_correct"] for r in results) / total * 100
    tools_acc   = sum(r["tools_correct"] for r in results) / total * 100

    errors = [r["error_category"] for r in results if r["error_category"]]
    from collections import Counter
    error_counts = Counter(errors)

    print(f"\n{'='*50}")
    print(f"RESULTS: {label}")
    print(f"{'='*50}")
    print(f"Verdict accuracy:  {verdict_acc:.1f}%  ({sum(r['verdict_correct'] for r in results)}/{total})")
    print(f"Payout precision:  {payout_acc:.1f}%  ({sum(r['payout_correct'] for r in results)}/{total})")
    print(f"Tool sequence acc: {tools_acc:.1f}%  ({sum(r['tools_correct'] for r in results)}/{total})")
    if error_counts:
        print(f"Error breakdown:   {dict(error_counts)}")
    print(f"{'='*50}")

    return {
        "label": label,
        "verdict_accuracy": verdict_acc,
        "payout_accuracy": payout_acc,
        "tools_accuracy": tools_acc,
        "error_breakdown": dict(error_counts)
    }

# ── MAIN ─────────────────────────────────────────────────────

if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Evaluate Teacher (GPT-4o mini)
    teacher_results = evaluate_model(client, "Teacher (GPT-4o mini)")
    teacher_summary = print_summary(teacher_results, "Teacher (GPT-4o mini)")

    # Save results
    with open("evaluation/teacher_results.json", "w") as f:
        json.dump(teacher_results, f, indent=2, default=str)

    print("\nTeacher evaluation complete.")
    print("Next: run Student evaluation after loading LoRA adapter.")
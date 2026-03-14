# looping everthing up with following procedure:
# user input a claim request
# send to GPT-4o mini
# model output thought & action
# loop read action
# call real tools (database/rules/calculator) for real output
# send real output as observation to model
# model keep infer loop
# final verdict

import json
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from dotenv import load_dotenv

from tools.database import lookup_policy
from tools.rules import check_rules
from tools.calculator import calculate_payout
from agent.parser import parse_action
from agent.prompts import SYSTEM_PROMPT
from agent.classifier import classify_risk

load_dotenv()

TOOL_REGISTRY = {
    "lookup_policy":   lookup_policy,
    "check_rules":     check_rules,
    "calculate_payout": calculate_payout,
}

def run_agent(claim_text: str, user_id: str, claimed_amount: float,
              client, max_steps: int = 8) -> dict:
    """
    Run the full agentic loop for a single claim.

    Returns a dict with:
      - trace (list): full conversation trace
      - risk_level (str): "high", "medium", or "low"
      - risk_confidence (float): classifier confidence
    """

    # ── Risk classification (before agent loop, logging only) ──
    risk_result = classify_risk(claim_text)
    risk_level      = risk_result["risk_level"]
    risk_confidence = risk_result["confidence"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": (
            f"Process this claim from policyholder {user_id}: "
            f"{claim_text} "
            f"The policyholder is claiming ${claimed_amount}."
        )}
    ]

    trace = [messages[1]]  # start trace with user message only

    for step in range(max_steps):
        # Call the LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.2,
        )
        model_output = response.choices[0].message.content

        trace.append({"role": "assistant", "content": model_output})
        messages.append({"role": "assistant", "content": model_output})

        # Check if model has reached a verdict
        if "Verdict:" in model_output:
            break

        # Detect hallucinated observations — model should never output these
        if "Observation:" in model_output:
            trace.append({
                "role": "system",
                "content": "[REJECTED: model hallucinated Observation]"
            })
            break

        # Parse tool call
        tool_name, args = parse_action(model_output)

        if tool_name is None:
            observation = "Observation: Error — no valid Action found. You must output an Action."
        elif tool_name not in TOOL_REGISTRY:
            observation = f"Observation: Error — unknown tool '{tool_name}'."
        elif args is None:
            observation = f"Observation: Error — could not parse arguments for '{tool_name}'."
        else:
            try:
                result = TOOL_REGISTRY[tool_name](**args)
                observation = f"Observation: {json.dumps(result)}"
            except TypeError as e:
                observation = f"Observation: Error — wrong arguments for '{tool_name}': {e}"

        trace.append({"role": "user", "content": observation})
        messages.append({"role": "user", "content": observation})

    return {
        "trace":            trace,
        "risk_level":       risk_level,
        "risk_confidence":  risk_confidence,
    }


def print_trace(trace: list):
    """Pretty print a trace for debugging."""
    print("\n" + "="*60)
    for msg in trace:
        role = msg["role"].upper()
        content = msg["content"]
        print(f"\n[{role}]\n{content}")
    print("\n" + "="*60)


if __name__ == "__main__":
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Test: valid hail claim
    result = run_agent(
        claim_text="My car windshield was cracked by hail during a storm last Tuesday.",
        user_id="P-1001",
        claimed_amount=1200,
        client=client
    )
    print(f"Risk level: {result['risk_level']} (confidence: {result['risk_confidence']})")
    print_trace(result["trace"])
# let teacher GPT to generate trace (data)
# all enquiry, rule engine, calculator results are seriered into json
# thus student llama learns from trace about the procedure

import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from dotenv import load_dotenv

from agent.loop import run_agent
from data.seeds import SEED_CLAIMS

load_dotenv()

OUTPUT_PATH   = os.path.join(os.path.dirname(__file__), 'train.jsonl')
REJECTED_PATH = os.path.join(os.path.dirname(__file__), 'rejected.jsonl')

# ── VARIATION GENERATION ─────────────────────────────────────

VARIATION_PROMPT = """Generate 15 realistic insurance claim descriptions for a '{claim_type}' claim.
Each should be 1-2 sentences written as a policyholder would describe the incident.
Vary the circumstances, severity, time of day, location, and claimed amounts.
Claimed amounts should vary: some low (near or below deductible), some medium, some high (near or above cap).

Base example: "{claim_text}"
Base claimed amount: ${claimed_amount}

Return ONLY a JSON array of objects. No preamble, no markdown, no explanation.
Format: [{{"claim_text": "...", "claimed_amount": 1234}}, ...]"""

def generate_variations(seed: dict, client: OpenAI) -> list:
    """Generate 15 claim variations for a seed scenario."""
    prompt = VARIATION_PROMPT.format(
        claim_type=seed["claim_type"],
        claim_text=seed["claim_text"],
        claimed_amount=seed["claimed_amount"]
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.8,
    )
    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        variations = json.loads(raw)
        return variations
    except json.JSONDecodeError:
        print(f"  WARNING: Could not parse variations JSON for seed: {seed['claim_text'][:40]}")
        return []

# ── TRACE VALIDATION ─────────────────────────────────────────

def extract_verdict(trace: list) -> str | None:
    """Extract APPROVED or DENIED from the last assistant message."""
    for msg in reversed(trace):
        if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
            if "APPROVED" in msg["content"]:
                return "APPROVED"
            if "DENIED" in msg["content"]:
                return "DENIED"
    return None

def count_tool_calls(trace: list) -> int:
    """Count number of Action: lines in assistant messages."""
    count = 0
    for msg in trace:
        if msg["role"] == "assistant":
            count += msg["content"].count("Action:")
    return count

def has_hallucinated_observation(trace: list) -> bool:
    """Check if model generated its own Observation."""
    for msg in trace:
        if msg["role"] == "system" and "REJECTED" in msg.get("content", ""):
            return True
    return False

def validate_trace(trace: list) -> tuple[bool, str]:
    """
    Returns (is_valid, rejection_reason).
    """
    if has_hallucinated_observation(trace):
        return False, "hallucination"

    tool_calls = count_tool_calls(trace)
    if tool_calls < 2:
        return False, f"too_few_tool_calls ({tool_calls})"

    verdict = extract_verdict(trace)
    if verdict is None:
        return False, "no_verdict"

    return True, "ok"

# ── TRACE FORMATTING ─────────────────────────────────────────

def format_for_training(trace: list, system_prompt: str) -> dict:
    """Format a trace as a multi-turn conversation for training."""
    from agent.prompts import SYSTEM_PROMPT
    conversations = [{"role": "system", "content": SYSTEM_PROMPT}]
    conversations.extend(trace)
    return {"conversations": conversations}

# ── MAIN GENERATION LOOP ─────────────────────────────────────

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    accepted = []
    rejected = []
    rejection_reasons = Counter()

    print(f"Starting data generation for {len(SEED_CLAIMS)} seeds...\n")

    for i, seed in enumerate(SEED_CLAIMS):
        print(f"Seed {i+1}/{len(SEED_CLAIMS)}: {seed['claim_text'][:50]}...")

        # Generate variations
        variations = generate_variations(seed, client)
        if not variations:
            print("  Skipping — no variations generated.")
            continue

        print(f"  Generated {len(variations)} variations. Running agent...")

        for j, var in enumerate(variations):
            claim_text     = var.get("claim_text", "")
            claimed_amount = float(var.get("claimed_amount", seed["claimed_amount"]))

            if not claim_text:
                continue

            try:
                trace = run_agent(
                    claim_text=claim_text,
                    user_id=seed["user_id"],
                    claimed_amount=claimed_amount,
                    client=client
                )
            except Exception as e:
                print(f"  Variation {j+1}: ERROR — {e}")
                rejected.append({
                    "seed_id": i,
                    "variation": var,
                    "reason": f"exception: {e}"
                })
                rejection_reasons["exception"] += 1
                continue

            is_valid, reason = validate_trace(trace)

            if is_valid:
                accepted.append(format_for_training(trace, ""))
            else:
                rejected.append({
                    "seed_id": i,
                    "variation": var,
                    "reason": reason
                })
                rejection_reasons[reason] += 1

            # Small delay to avoid rate limits
            time.sleep(0.3)

        accepted_so_far = len(accepted)
        print(f"  Running total accepted: {accepted_so_far}")

    # ── SAVE RESULTS ─────────────────────────────────────────
    with open(OUTPUT_PATH, 'w') as f:
        for item in accepted:
            f.write(json.dumps(item) + '\n')

    with open(REJECTED_PATH, 'w') as f:
        for item in rejected:
            f.write(json.dumps(item) + '\n')

    print(f"\n{'='*50}")
    print("DONE")
    print(f"Accepted: {len(accepted)}")
    print(f"Rejected: {len(rejected)}")
    print(f"Rejection reasons: {dict(rejection_reasons)}")
    print(f"Saved to: {OUTPUT_PATH}")

    # Distribution check
    verdicts = Counter()
    for item in accepted:
        for msg in reversed(item["conversations"]):
            if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
                if "APPROVED" in msg["content"]:
                    verdicts["APPROVED"] += 1
                elif "DENIED" in msg["content"]:
                    verdicts["DENIED"] += 1
                break
    print(f"Verdict distribution: {dict(verdicts)}")

if __name__ == "__main__":
    main()
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from openai import OpenAI
from dotenv import load_dotenv
from agent.loop import run_agent
from data.generate import (
    generate_variations, validate_trace,
    format_for_training
)

load_dotenv()

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), 'train.jsonl')

# 只用DENIED相关的seed，专门补充这类数据
DENIED_SEEDS = [
    # 不在coverage里
    {
        "user_id": "P-1002",
        "claim_text": "My basement flooded after heavy rainfall and damaged my furniture.",
        "claimed_amount": 5000,
        "expected_verdict": "DENIED",
        "claim_type": "flood"
    },
    {
        "user_id": "P-1007",
        "claim_text": "Storm damage blew tiles off my roof.",
        "claimed_amount": 3200,
        "expected_verdict": "DENIED",
        "claim_type": "storm"
    },
    {
        "user_id": "P-1010",
        "claim_text": "Hail dented my car bonnet during a storm.",
        "claimed_amount": 1800,
        "expected_verdict": "DENIED",
        "claim_type": "hail"
    },
    # Policy lapsed/cancelled
    {
        "user_id": "P-1004",
        "claim_text": "My car was stolen from the driveway overnight.",
        "claimed_amount": 18000,
        "expected_verdict": "DENIED",
        "claim_type": "theft"
    },
    {
        "user_id": "P-1008",
        "claim_text": "A fire in my kitchen caused significant damage.",
        "claimed_amount": 4500,
        "expected_verdict": "DENIED",
        "claim_type": "fire"
    },
    {
        "user_id": "P-1013",
        "claim_text": "Storm winds knocked over my garden fence.",
        "claimed_amount": 2000,
        "expected_verdict": "DENIED",
        "claim_type": "storm"
    },
    # Annual limit reached
    {
        "user_id": "P-1011",
        "claim_text": "Storm damage cracked several roof tiles.",
        "claimed_amount": 2800,
        "expected_verdict": "DENIED",
        "claim_type": "storm"
    },
    {
        "user_id": "P-1007",
        "claim_text": "My car was broken into and my laptop stolen.",
        "claimed_amount": 2200,
        "expected_verdict": "DENIED",
        "claim_type": "theft"
    },
]

def main():
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 读取现有数据
    existing = [json.loads(l) for l in open(OUTPUT_PATH)]
    print(f"Existing traces: {len(existing)}")

    accepted = []
    rejected_count = 0

    print(f"\nGenerating extra DENIED traces from {len(DENIED_SEEDS)} seeds...")

    for i, seed in enumerate(DENIED_SEEDS):
        print(f"Seed {i+1}/{len(DENIED_SEEDS)}: {seed['claim_text'][:50]}...")

        variations = generate_variations(seed, client)
        if not variations:
            continue

        print(f"  Got {len(variations)} variations. Running agent...")

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
            except Exception:
                rejected_count += 1
                continue

            is_valid, reason = validate_trace(trace)
            if is_valid:
                accepted.append(format_for_training(trace, ""))
            else:
                rejected_count += 1

            time.sleep(0.3)

        print(f"  Running total new accepted: {len(accepted)}")

    # 合并新旧数据
    all_data = existing + accepted
    print(f"\nNew traces generated: {len(accepted)}")
    print(f"Rejected: {rejected_count}")
    print(f"Total combined: {len(all_data)}")

    # 检查分布
    verdicts = Counter()
    for item in all_data:
        for msg in reversed(item["conversations"]):
            if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
                if "APPROVED" in msg["content"]: verdicts["APPROVED"] += 1
                elif "DENIED" in msg["content"]: verdicts["DENIED"] += 1
                break
    print(f"Final verdict distribution: {dict(verdicts)}")

    # 保存
    with open(OUTPUT_PATH, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')
    print(f"Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
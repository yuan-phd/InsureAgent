"""
training/train_rl.py
RLVR fine-tuning for InsureAgent using GRPO (Group Relative Policy Optimization).

Starts from the LoRA-fine-tuned student model and further trains it using
verifiable rewards from the rules engine — no human annotation required.

Usage (Colab T4, after supervised fine-tuning):
    python training/train_rl.py --config config/config.yaml

How it works:
    For each claim in the training set, the model generates N candidate traces
    (via temperature sampling). Each trace is scored by the reward function,
    which calls the real tools to verify correctness. GRPO updates the model
    to favour higher-reward traces relative to the group average.

Why GRPO over PPO:
    GRPO does not require a separate value/critic model. It estimates the
    baseline from the group of sampled responses for the same prompt.
    This makes it practical to run on a single T4 GPU.

Requirements:
    pip install transformers peft trl datasets accelerate pyyaml
    TRL >= 0.12.0 (GRPOTrainer)
"""

import argparse
import json
import os
import re
import sys
import yaml

import torch
from datasets import Dataset
from huggingface_hub import login
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tools.calculator import calculate_payout
from tools.database import lookup_policy
from tools.rules import check_rules
from agent.prompts import SYSTEM_PROMPT

TOOL_REGISTRY = {
    "lookup_policy":    lookup_policy,
    "check_rules":      check_rules,
    "calculate_payout": calculate_payout,
}

POSITIONAL_PARAMS = {
    "lookup_policy":    ["user_id"],
    "check_rules":      ["claim_type", "plan_type", "policy_covers",
                         "policy_status", "claims_this_year"],
    "calculate_payout": ["claimed_amount", "deductible", "max_single_claim",
                         "max_annual_payout", "already_claimed_this_year"],
}


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


# ── DATA ──────────────────────────────────────────────────────────────────────

def load_rl_prompts(train_path: str) -> Dataset:
    """
    Load training traces and convert to RL prompts.

    Each example becomes a prompt (system + user turn only).
    The model generates the full agent trajectory as the response.
    Ground truth verdict and payout are kept for reward computation.
    """
    with open(train_path) as f:
        traces = [json.loads(line) for line in f]

    examples = []
    for trace in traces:
        convs = trace["conversations"]

        # Extract system and first user turn as the prompt
        system_msg = next((c for c in convs if c["role"] == "system"), None)
        user_msg = next((c for c in convs if c["role"] == "user"), None)
        if not system_msg or not user_msg:
            continue

        # Extract ground truth from the trace
        true_verdict = None
        true_payout = 0.0
        for msg in reversed(convs):
            if msg["role"] == "assistant" and "Verdict:" in msg["content"]:
                true_verdict = "APPROVED" if "APPROVED" in msg["content"] else "DENIED"
                match = re.search(r'Payout:\s*\$?([\d,]+\.?\d*)', msg["content"])
                if match:
                    true_payout = float(match.group(1).replace(',', ''))
                break

        if true_verdict is None:
            continue

        examples.append({
            "prompt": [
                {"role": "system", "content": system_msg["content"]},
                {"role": "user",   "content": user_msg["content"]},
            ],
            "true_verdict": true_verdict,
            "true_payout":  true_payout,
        })

    print(f"Loaded {len(examples)} RL training prompts.")
    return Dataset.from_list(examples)


# ── PARSER ────────────────────────────────────────────────────────────────────

def parse_action(text: str):
    """Parse tool call from model output. Handles JSON, kwargs, positional."""
    match = re.search(r'Action:\s*(\w+)\s*\((.*)\)', text, re.DOTALL)
    if not match:
        return None, None

    tool_name = match.group(1).strip()
    args_str = match.group(2).strip()

    # JSON
    if args_str.startswith('{'):
        for attempt in [args_str, args_str.replace("'", '"')]:
            try:
                args = json.loads(attempt)
                if tool_name == "calculate_payout":
                    args.setdefault("already_claimed_this_year", 0)
                return tool_name, args
            except json.JSONDecodeError:
                pass

    # kwargs
    if '=' in args_str:
        try:
            pairs = re.findall(r'(\w+)\s*=\s*([^,]+?)(?=,\s*\w+\s*=|$)',
                               args_str.split('\n')[0].strip())
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

    # positional
    params = POSITIONAL_PARAMS.get(tool_name, [])
    if params:
        try:
            args = {}
            for i, v in enumerate([v.strip() for v in args_str.split(',')]):
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


# ── TOOL EXECUTION ────────────────────────────────────────────────────────────

def execute_tool(tool_name, args) -> str:
    if tool_name is None or tool_name not in TOOL_REGISTRY:
        return f"Observation: Error — unknown tool '{tool_name}'."
    if args is None:
        return f"Observation: Error — could not parse arguments for '{tool_name}'."
    try:
        result = TOOL_REGISTRY[tool_name](**args)
        return f"Observation: {json.dumps(result)}"
    except TypeError as e:
        return f"Observation: Error — {e}"


# ── TOOL SEQUENCE VALIDATION ─────────────────────────────────────────────────

def is_valid_tool_sequence(tool_calls: list, observations: list) -> bool:
    """
    Check whether the tool call sequence is logically valid.

    Key rule: calculate_payout must not be called after check_rules
    returned eligible: false.
    """
    check_rules_result = None
    for i, tool_name in enumerate(tool_calls):
        if tool_name == "check_rules" and i < len(observations):
            obs = observations[i]
            try:
                data = json.loads(obs.replace("Observation: ", ""))
                check_rules_result = data.get("eligible", True)
            except Exception:
                pass
        if tool_name == "calculate_payout":
            if check_rules_result is False:
                return False  # called payout on ineligible claim
    return True


# ── REWARD FUNCTION ───────────────────────────────────────────────────────────

def compute_reward(
    response: str,
    true_verdict: str,
    true_payout: float,
) -> float:
    """
    Verifiable reward function. No human annotation required.

    Scoring:
      +1.0  verdict correct
      +0.5  tool sequence logically valid (no payout on ineligible claims)
      +0.3  payout within $100 of ground truth (only if verdict correct)
      -0.5  hallucinated Observation in assistant output

    Max reward: 1.8
    Min reward: -0.5
    """
    reward = 0.0

    # ── extract verdict ──
    predicted_verdict = None
    if "Verdict: APPROVED" in response:
        predicted_verdict = "APPROVED"
    elif "Verdict: DENIED" in response:
        predicted_verdict = "DENIED"

    # ── extract payout ──
    predicted_payout = 0.0
    match = re.search(r'Payout:\s*\$?([\d,]+\.?\d*)', response)
    if match:
        predicted_payout = float(match.group(1).replace(',', ''))

    # ── penalise hallucination ──
    # Assistant generating its own Observation is a hard error
    lines = response.split('\n')
    for line in lines:
        if line.strip().startswith("Observation:"):
            reward -= 0.5
            break

    # ── reward verdict ──
    if predicted_verdict == true_verdict:
        reward += 1.0

        # ── reward payout precision (only if verdict correct) ──
        if abs(predicted_payout - true_payout) < 100:
            reward += 0.3

    # ── reward valid tool sequence ──
    tool_calls = re.findall(r'Action:\s*(\w+)\s*\(', response)
    observations = re.findall(r'(Observation:.*?)(?=\n(?:Thought|Action|Verdict)|$)',
                               response, re.DOTALL)
    if is_valid_tool_sequence(tool_calls, observations):
        reward += 0.5

    return round(reward, 3)


# ── REWARD WRAPPER FOR GRPO ───────────────────────────────────────────────────

def make_reward_fn(dataset: Dataset):
    """
    Returns a reward function compatible with GRPOTrainer.

    GRPOTrainer calls reward_fn(prompts, completions, **kwargs).
    We look up the ground truth from the dataset by matching the prompt.
    """
    # Build lookup: user message content → (true_verdict, true_payout)
    lookup = {}
    for ex in dataset:
        user_content = ex["prompt"][-1]["content"]
        lookup[user_content] = (ex["true_verdict"], ex["true_payout"])

    def reward_fn(prompts, completions, **kwargs):
        rewards = []
        for prompt, completion in zip(prompts, completions):
            user_content = prompt[-1]["content"] if isinstance(prompt, list) else ""
            true_verdict, true_payout = lookup.get(user_content, ("DENIED", 0.0))

            # completion is a list of dicts or a string depending on TRL version
            if isinstance(completion, list):
                response = " ".join(
                    c["content"] for c in completion if isinstance(c, dict)
                )
            else:
                response = str(completion)

            r = compute_reward(response, true_verdict, true_payout)
            rewards.append(r)
        return rewards

    return reward_fn


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_model(cfg: dict, hf_token: str):
    """Load base model + LoRA adapter for GRPO training."""
    model_name   = cfg["model"]["student_base"]
    adapter_path = cfg["model"]["adapter"]

    print(f"Loading base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # required for generation in GRPO

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {adapter_path}")
    model = PeftModel.from_pretrained(
        base_model, adapter_path,
        token=hf_token,
        is_trainable=True,   # adapter weights must be trainable for GRPO
    )
    model.print_trainable_parameters()
    return model, tokenizer


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsureAgent GRPO RL training")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--hf-token",  default=None)
    parser.add_argument("--skip-upload", action="store_true")
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Candidate traces per prompt (GRPO group size)")
    parser.add_argument("--epochs",    type=int, default=1,
                        help="RL training epochs (1-2 is usually enough)")
    parser.add_argument("--output-adapter", default="yuanphd/insureagent-lora-rl-v1")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    hf_token  = hf_login(args.hf_token)
    train_path = cfg["data"]["train_path"]

    # Load data
    dataset = load_rl_prompts(train_path)

    # Load model
    model, tokenizer = load_model(cfg, hf_token)

    # Reward function
    reward_fn = make_reward_fn(dataset)

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir="./insureagent-rl-checkpoints",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,              # lower than SFT — RL is sensitive
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=512,
        temperature=0.8,                 # exploration during generation
        logging_steps=5,
        save_strategy="epoch",
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("\nStarting GRPO training...")
    print(f"  Group size (num_generations): {args.num_generations}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Reward components: verdict (+1.0) · tool sequence (+0.5) · payout (+0.3) · hallucination (-0.5)")
    print(f"  Max reward per trace: 1.8\n")

    trainer.train()
    print("GRPO training complete.")

    # Save adapter
    output_dir = "./insureagent-lora-rl-final"
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"RL adapter saved to {output_dir}")

    # Upload
    if not args.skip_upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(
            repo_id=args.output_adapter,
            repo_type="model",
            private=True,
            token=hf_token,
            exist_ok=True,
        )
        api.upload_folder(
            folder_path=output_dir,
            repo_id=args.output_adapter,
            repo_type="model",
            token=hf_token,
        )
        print(f"Uploaded: https://huggingface.co/{args.output_adapter}")
    else:
        print("Skipping upload (--skip-upload).")


if __name__ == "__main__":
    main()

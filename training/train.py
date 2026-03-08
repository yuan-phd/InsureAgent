"""
training/train.py
InsureAgent LoRA fine-tuning script.

Current usage (Colab / local with pyenv 3.11.9):
    python training/train.py --config config/config.yaml --backend local

Future usage (Databricks cluster):
    python training/train.py --config config/config.yaml --backend databricks

The --backend flag controls where training data is read from:
  local:      data/train.jsonl  (default, works in Colab and locally)
  databricks: Delta table at config.data.delta_path (requires PySpark session)

MLflow tracking:
  local:      logs to local ./mlruns directory
  databricks: logs to Databricks-managed MLflow (set DATABRICKS_HOST + DATABRICKS_TOKEN)

Requirements:
    pip install transformers peft trl datasets accelerate bitsandbytes pyyaml mlflow
"""

import argparse
import os
import yaml
import torch
import mlflow

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from huggingface_hub import login, HfApi

from data_loader import build_dataset


# ── CONFIG ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── AUTH ──────────────────────────────────────────────────────────────────────

def hf_login(token: str | None = None) -> str:
    """Login to HuggingFace. Uses HF_TOKEN env var if token not provided."""
    hf_token = token or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not set. Export it or pass via --hf-token.")
    login(token=hf_token)
    print("Logged in to HuggingFace.")
    return hf_token


def setup_mlflow(backend: str, experiment_name: str):
    """
    Configure MLflow tracking URI.

    local:      logs to ./mlruns (default MLflow behaviour)
    databricks: logs to Databricks-managed MLflow
                requires DATABRICKS_HOST and DATABRICKS_TOKEN env vars
    """
    if backend == "databricks":
        databricks_host = os.environ.get("DATABRICKS_HOST")
        databricks_token = os.environ.get("DATABRICKS_TOKEN")
        if not databricks_host or not databricks_token:
            raise ValueError(
                "DATABRICKS_HOST and DATABRICKS_TOKEN must be set "
                "for --backend databricks MLflow logging."
            )
        mlflow.set_tracking_uri("databricks")
        print(f"[databricks] MLflow tracking: {databricks_host}")
    else:
        # local: mlruns/ in working directory
        print("[local] MLflow tracking: ./mlruns")

    mlflow.set_experiment(experiment_name)


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

def load_base_model(model_name: str, hf_token: str):
    """Load base model and tokenizer."""
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Base model loaded. Parameters: {model.num_parameters():,}")
    return model, tokenizer


# ── LORA ──────────────────────────────────────────────────────────────────────

def apply_lora(model, lora_cfg: dict):
    """Wrap model with LoRA adapter."""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ── TRAINING ──────────────────────────────────────────────────────────────────

def train(model, tokenizer, dataset, train_cfg: dict, output_dir: str):
    """Run SFT training and return the trainer."""
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        logging_steps=10,
        save_strategy="epoch",
        fp16=train_cfg["fp16"],
        dataset_text_field="text",
        report_to="none",          # MLflow logging handled manually below
        gradient_checkpointing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete.")
    return trainer


# ── MLFLOW LOGGING ────────────────────────────────────────────────────────────

def log_to_mlflow(cfg: dict, trainer, backend: str, n_traces: int):
    """Log hyperparameters, loss curve, and run metadata to MLflow."""

    # Build a descriptive run name
    run_name = (
        f"lora-r{cfg['lora']['r']}"
        f"-lr{cfg['training']['learning_rate']}"
        f"-{n_traces}traces"
        f"-{backend}"
    )

    with mlflow.start_run(run_name=run_name):

        # Data provenance
        mlflow.log_param("n_training_traces", n_traces)
        mlflow.log_param("data_backend", backend)
        if backend == "databricks":
            mlflow.log_param("delta_path", cfg["data"]["delta_path"])
        else:
            mlflow.log_param("train_path", cfg["data"]["train_path"])

        # LoRA hyperparameters
        mlflow.log_params({
            "lora_r":           cfg["lora"]["r"],
            "lora_alpha":       cfg["lora"]["lora_alpha"],
            "lora_dropout":     cfg["lora"]["lora_dropout"],
            "target_modules":   str(cfg["lora"]["target_modules"]),
        })

        # Training hyperparameters
        mlflow.log_params({
            "epochs":                       cfg["training"]["epochs"],
            "batch_size":                   cfg["training"]["batch_size"],
            "gradient_accumulation_steps":  cfg["training"]["gradient_accumulation_steps"],
            "learning_rate":                cfg["training"]["learning_rate"],
            "warmup_steps":                 cfg["training"]["warmup_steps"],
            "fp16":                         cfg["training"]["fp16"],
        })

        # Model metadata
        mlflow.log_params({
            "student_base":     cfg["model"]["student_base"],
            "adapter_repo":     cfg["model"]["adapter"],
        })

        # Loss curve — log each step
        if hasattr(trainer.state, "log_history"):
            for entry in trainer.state.log_history:
                if "loss" in entry and "step" in entry:
                    mlflow.log_metric("train_loss", entry["loss"], step=int(entry["step"]))

        # Config file as artifact for full reproducibility
        mlflow.log_artifact("config/config.yaml")

    print(f"MLflow run logged: {run_name}")


# ── SAVE & UPLOAD ─────────────────────────────────────────────────────────────

def save_and_upload(trainer, tokenizer, local_dir: str, repo_id: str, hf_token: str):
    """Save adapter locally and upload to HuggingFace Hub."""
    print(f"Saving adapter to {local_dir}")
    trainer.save_model(local_dir)
    tokenizer.save_pretrained(local_dir)

    print(f"Uploading to HuggingFace: {repo_id}")
    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=True,
        token=hf_token,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type="model",
        token=hf_token,
    )
    print(f"Uploaded: https://huggingface.co/{repo_id}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="InsureAgent LoRA training")
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config.yaml"
    )
    parser.add_argument(
        "--backend", default="local", choices=["local", "databricks"],
        help=(
            "Data backend: 'local' reads data/train.jsonl (Colab/local), "
            "'databricks' reads from Delta table (production cluster)"
        )
    )
    parser.add_argument(
        "--hf-token", default=None,
        help="HuggingFace token (overrides HF_TOKEN env var)"
    )
    parser.add_argument(
        "--skip-upload", action="store_true",
        help="Skip HuggingFace upload (useful for dry runs)"
    )
    parser.add_argument(
        "--skip-mlflow", action="store_true",
        help="Skip MLflow logging"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    print(f"Config loaded from {args.config}")
    print(f"Backend: {args.backend}")

    # Auth
    hf_token = hf_login(args.hf_token)

    # MLflow setup
    if not args.skip_mlflow:
        setup_mlflow(
            backend=args.backend,
            experiment_name="/insureagent/lora-training",
        )

    # Load model + tokenizer
    model, tokenizer = load_base_model(cfg["model"]["student_base"], hf_token)

    # Build dataset (local or Databricks)
    dataset = build_dataset(cfg, tokenizer, backend=args.backend)
    n_traces = len(dataset)

    # Apply LoRA
    model = apply_lora(model, cfg["lora"])

    # Train
    output_dir = "./insureagent-lora-checkpoints"
    trainer = train(model, tokenizer, dataset, cfg["training"], output_dir)

    # Log to MLflow
    if not args.skip_mlflow:
        log_to_mlflow(cfg, trainer, backend=args.backend, n_traces=n_traces)

    # Save adapter locally
    final_dir = "./insureagent-lora-final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to {final_dir}")

    # Upload to HuggingFace Hub
    if not args.skip_upload:
        save_and_upload(
            trainer, tokenizer,
            local_dir=final_dir,
            repo_id=cfg["model"]["adapter"],
            hf_token=hf_token,
        )
    else:
        print("Skipping HuggingFace upload (--skip-upload).")


if __name__ == "__main__":
    main()
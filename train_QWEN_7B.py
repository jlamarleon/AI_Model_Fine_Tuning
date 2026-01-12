"""
train_QWEN_7B.py
========
Fine-tuning Qwen2-7B-Instruct with LoRA for text-only causal language modeling.

Key features:
- HuggingFace Trainer
- PEFT LoRA adapters
- Gradient checkpointing
- BF16 training
- JSON instruction-style dataset

Author: (you)
"""

# ===============================
# Environment configuration
# ===============================
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===============================
# Standard imports
# ===============================
import json
import argparse
import warnings
from typing import List, Dict

import torch
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)

from peft import (
    LoraConfig,
    get_peft_model
)

# ===============================
# Utilities
# ===============================
def get_slurm_job_id() -> str:
    """
    Returns SLURM job ID if running under SLURM,
    otherwise returns a default identifier.
    """
    return os.getenv("SLURM_JOB_ID", "00000")


def create_output_dir(base_dir: str, job_id: str) -> str:
    """
    Creates an output directory using the SLURM job ID.
    """
    path = os.path.join(base_dir, job_id)
    os.makedirs(path, exist_ok=True)
    print(f"[INFO] Output directory: {path}")
    return path


# ===============================
# Dataset preparation
# ===============================
def prepare_sample_batch(
    raw_data: List[Dict],
) -> List[Dict]:
    """
    Converts raw JSON conversation data into a flat
    text-only dataset for causal language modeling.

    Expected JSON structure:
    {
        "messages": [
            {"role": "user", "content": [{"text": "..."}]},
            {"role": "assistant", "content": [{"text": "..."}]}
        ]
    }
    """
    dataset = []

    for idx, item in enumerate(raw_data):
        try:
            messages = item.get("messages", [])
            if len(messages) < 2:
                continue

            question = ""
            for c in messages[0].get("content", []):
                if isinstance(c, dict) and "text" in c:
                    question = c["text"]
                    break

            answer = ""
            for c in messages[1].get("content", []):
                if isinstance(c, dict) and "text" in c:
                    answer = c["text"]
                    break

            if not question or not answer:
                continue

            dataset.append({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            print(f"[WARN] Skipping sample {idx}: {e}")

    print(f"[INFO] Prepared {len(dataset)} samples")
    return dataset


# ===============================
# Data collator
# ===============================
class TextOnlyDataCollator:
    """
    Data collator for causal language modeling.
    - Concatenates question + answer
    - Uses labels=input_ids
    """

    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        texts = [
            f"{ex['question']} {ex['answer']}"
            for ex in examples
        ]

        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Causal LM labels
        batch["labels"] = batch["input_ids"].clone()
        return batch


# ===============================
# Main training logic
# ===============================
def main():
    # ---------------------------
    # CLI arguments
    # ---------------------------
    parser = argparse.ArgumentParser("Qwen2-7B LoRA Fine-tuning")

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-7B-Instruct")
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--eval_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_eval", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-6)
    parser.add_argument("--max_length", type=int, default=1024)

    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--cache_dir", type=str, default="/data/--user--/-----/7B_instruct")


    args = parser.parse_args()

    # ---------------------------
    # Load tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------
    # Load model
    # ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        cache_dir=args.cache_dir
    )

    # ---------------------------
    # LoRA configuration
    # ---------------------------
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"  # to adapt LoRA to specific layers, should be similar to ".*(language_model.*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$)",
    )

    model.gradient_checkpointing_enable()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.train()
    model.config.use_cache = False
    model.enable_input_require_grads()

    # ---------------------------
    # Load datasets
    # ---------------------------
    with open(args.train_json) as f:
        train_raw = json.load(f)

    with open(args.eval_json) as f:
        eval_raw = json.load(f)

    train_dataset = prepare_sample_batch(train_raw)
    eval_dataset = prepare_sample_batch(eval_raw)

    data_collator = TextOnlyDataCollator(
        tokenizer,
        max_length=args.max_length
    )

    # ---------------------------
    # Training arguments
    # ---------------------------
    output_dir = create_output_dir(
        args.output_dir,
        get_slurm_job_id()
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size_eval,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": True},
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=100, # evaluate every 100 steps
        save_steps=100, # save every 100 steps
        logging_steps=100,
        load_best_model_at_end=True,
        remove_unused_columns=False, # important for custom data collator
        report_to="none",
        disable_tqdm=True,
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # ---------------------------
    # Train
    # ---------------------------
    trainer.train()


# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    main()

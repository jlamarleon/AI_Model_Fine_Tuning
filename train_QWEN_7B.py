"""
train_QWEN_7B.py
========
Fine-tuning Qwen2-7B-Instruct with LoRA for text-only causal language modeling.

Key features:
- HuggingFace Trainer: Uses the HuggingFace Trainer API for streamlined training loops.
- PEFT LoRA adapters: Applies Low-Rank Adaptation (LoRA) for efficient fine-tuning.
- Gradient checkpointing: Reduces memory usage by recomputing intermediate activations during backpropagation.
- BF16 training: Uses BFloat16 precision for faster training and reduced memory footprint.
- JSON instruction-style dataset: Expects datasets in a structured JSON format for instruction-based fine-tuning.

Author: (Javier Lamar)
"""

# ===============================
# Environment configuration
# ===============================
import os
# Disable HuggingFace transfer for direct downloads
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# Disable parallelism in tokenizers to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Suppress advisory warnings from the Transformers library
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
# Ignore Python warnings for cleaner output
os.environ["PYTHONWARNINGS"] = "ignore"
# Restrict CUDA to use only the first GPU device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ===============================
# Standard imports
# ===============================
import json  # For handling JSON data files
import argparse  # For parsing command-line arguments
import warnings  # For managing warnings
from typing import List, Dict  # For type hints

import torch  # PyTorch for deep learning
from PIL import Image  # Python Imaging Library (unused in this script, likely for future extensions)

from transformers import (
    AutoModelForCausalLM,  # Auto-loading of causal language models
    AutoTokenizer,  # Auto-loading of tokenizers
    TrainingArguments,  # Configuration for training
    Trainer  # Training loop
)

from peft import (
    LoraConfig,  # Configuration for LoRA adapters
    get_peft_model  # Function to apply LoRA to a model
)

# ===============================
# Utilities
# ===============================
def get_slurm_job_id() -> str:
    """
    Returns SLURM job ID if running under SLURM,
    otherwise returns a default identifier.
    Useful for tracking and organizing output directories in HPC environments.
    """
    return os.getenv("SLURM_JOB_ID", "00000")

def create_output_dir(base_dir: str, job_id: str) -> str:
    """
    Creates an output directory using the SLURM job ID.
    Ensures that each training run has a unique output directory.
    """
    path = os.path.join(base_dir, job_id)
    os.makedirs(path, exist_ok=True)  # Create directory if it doesn't exist
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
    This function extracts the user's question and the assistant's answer for training.
    """
    dataset = []  # Initialize an empty list to store processed samples

    for idx, item in enumerate(raw_data):
        try:
            messages = item.get("messages", [])  # Extract messages from the JSON item
            if len(messages) < 2:  # Skip if there are not enough messages (user + assistant)
                continue

            question = ""
            # Extract the user's question from the first message
            for c in messages[0].get("content", []):
                if isinstance(c, dict) and "text" in c:
                    question = c["text"]
                    break

            answer = ""
            # Extract the assistant's answer from the second message
            for c in messages[1].get("content", []):
                if isinstance(c, dict) and "text" in c:
                    answer = c["text"]
                    break

            if not question or not answer:  # Skip if either question or answer is empty
                continue

            dataset.append({
                "question": question,
                "answer": answer
            })

        except Exception as e:
            print(f"[WARN] Skipping sample {idx}: {e}")  # Log errors and skip problematic samples

    print(f"[INFO] Prepared {len(dataset)} samples")  # Log the number of prepared samples
    return dataset

# ===============================
# Data collator
# ===============================
class TextOnlyDataCollator:
    """
    Data collator for causal language modeling.
    - Concatenates question + answer for input
    - Uses labels=input_ids for causal language modeling
    """
    def __init__(self, tokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer  # Tokenizer for encoding text
        self.max_length = max_length  # Maximum sequence length

    def __call__(self, examples: List[Dict]) -> Dict[str, torch.Tensor]:
        # Concatenate question and answer for each example
        texts = [
            f"{ex['question']} {ex['answer']}"
            for ex in examples
        ]

        # Tokenize the concatenated texts
        batch = self.tokenizer(
            texts,
            padding=True,  # Pad sequences to the same length
            truncation=True,  # Truncate sequences to max_length
            max_length=self.max_length,
            return_tensors="pt"  # Return PyTorch tensors
        )

        # For causal language modeling, labels are the same as input_ids
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
    # Define command-line arguments for model, data, and training parameters
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-7B-Instruct")  # HuggingFace model identifier
    parser.add_argument("--train_json", type=str, required=True)  # Path to training JSON file
    parser.add_argument("--eval_json", type=str, required=True)  # Path to evaluation JSON file
    parser.add_argument("--output_dir", type=str, required=True)  # Directory to save outputs

    parser.add_argument("--epochs", type=int, default=3)  # Number of training epochs
    parser.add_argument("--batch_size", type=int, default=8)  # Batch size for training
    parser.add_argument("--batch_size_eval", type=int, default=16)  # Batch size for evaluation
    parser.add_argument("--lr", type=float, default=3e-6)  # Learning rate
    parser.add_argument("--max_length", type=int, default=1024)  # Maximum sequence length

    parser.add_argument("--lora_r", type=int, default=128)  # LoRA rank
    parser.add_argument("--lora_alpha", type=int, default=32)  # LoRA alpha
    parser.add_argument("--lora_dropout", type=float, default=0.1)  # LoRA dropout
    parser.add_argument("--cache_dir", type=str, default="/data/--user--/-----/7B_instruct")  # Directory for model cache

    args = parser.parse_args()  # Parse command-line arguments

    # ---------------------------
    # Load tokenizer
    # ---------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=True  # Allow loading custom code from the model hub
    )
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to EOS token

    # ---------------------------
    # Load model
    # ---------------------------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,  # Use BFloat16 precision
        device_map="auto",  # Automatically map model to available GPUs
        trust_remote_code=True,  # Allow loading custom code from the model hub
        cache_dir=args.cache_dir  # Directory to cache the model
    )

    # ---------------------------
    # LoRA configuration
    # ---------------------------
    lora_config = LoraConfig(
        r=args.lora_r,  # Rank of the LoRA matrices
        lora_alpha=args.lora_alpha,  # Scaling factor for LoRA
        lora_dropout=args.lora_dropout,  # Dropout rate for LoRA layers
        bias="none",  # Do not apply LoRA to bias terms
        task_type="CAUSAL_LM",  # Task type for causal language modeling
        target_modules="all-linear",  # Apply LoRA to all linear layers
    )

    model.gradient_checkpointing_enable()  # Enable gradient checkpointing to save memory
    model = get_peft_model(model, lora_config)  # Apply LoRA to the model
    model.print_trainable_parameters()  # Print the number of trainable parameters

    model.train()  # Set model to training mode
    model.config.use_cache = False  # Disable caching for gradient checkpointing
    model.enable_input_require_grads()  # Enable gradients for input embeddings

    # ---------------------------
    # Load datasets
    # ---------------------------
    with open(args.train_json) as f:
        train_raw = json.load(f)  # Load training data from JSON file

    with open(args.eval_json) as f:
        eval_raw = json.load(f)  # Load evaluation data from JSON file

    train_dataset = prepare_sample_batch(train_raw)  # Prepare training dataset
    eval_dataset = prepare_sample_batch(eval_raw)  # Prepare evaluation dataset

    data_collator = TextOnlyDataCollator(
        tokenizer,
        max_length=args.max_length  # Use the specified maximum sequence length
    )

    # ---------------------------
    # Training arguments
    # ---------------------------
    output_dir = create_output_dir(
        args.output_dir,
        get_slurm_job_id()  # Create a unique output directory for this job
    )

    training_args = TrainingArguments(
        output_dir=output_dir,  # Directory to save outputs
        num_train_epochs=args.epochs,  # Number of training epochs
        per_device_train_batch_size=args.batch_size,  # Batch size per device for training
        per_device_eval_batch_size=args.batch_size_eval,  # Batch size per device for evaluation
        learning_rate=args.lr,  # Learning rate
        bf16=True,  # Use BFloat16 precision
        gradient_checkpointing=True,  # Enable gradient checkpointing
        gradient_checkpointing_kwargs={"use_reentrant": True},  # Use reentrant gradient checkpointing
        eval_strategy="steps",  # Evaluate every specified number of steps
        save_strategy="steps",  # Save model every specified number of steps
        logging_strategy="steps",  # Log training metrics every specified number of steps
        eval_steps=100,  # Evaluate every 100 steps
        save_steps=100,  # Save model every 100 steps
        logging_steps=100,  # Log metrics every 100 steps
        load_best_model_at_end=True,  # Load the best model at the end of training
        remove_unused_columns=False,  # Do not remove unused columns (important for custom data collator)
        report_to="none",  # Do not report metrics to any external service
        disable_tqdm=True,  # Disable progress bars
    )

    # ---------------------------
    # Trainer
    # ---------------------------
    trainer = Trainer(
        model=model,  # Model to train
        args=training_args,  # Training arguments
        train_dataset=train_dataset,  # Training dataset
        eval_dataset=eval_dataset,  # Evaluation dataset
        data_collator=data_collator,  # Data collator for batching
        tokenizer=tokenizer  # Tokenizer for encoding
    )

    # ---------------------------
    # Train
    # ---------------------------
    trainer.train()  # Start training

# ===============================
# Entry point
# ===============================
if __name__ == "__main__":
    main()  # Call the main function when the script is executed

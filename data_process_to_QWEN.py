# This code is an example to convert from the alpaca JSON dataset (https://huggingface.co/datasets/tatsu-lab/alpaca) to structured data input for Qwen 7B.
# It processes instruction-based JSON data, converts it to a chat format, splits it into train/eval sets,
# and ensures all assistant outputs have at least one token.

import os
import json
import base64
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import random

from transformers import AutoTokenizer, AutoProcessor

def convert_and_split_instruction_json_safe(
    input_json_path,
    processor_or_tokenizer,
    train_output_file='train.json',
    eval_output_file='eval.json',
    train_ratio=0.9,
    seed=42,
    max_samples=None,
    verbose=True
):
    """
    Converts an instruction-json dataset into chat format, splits into train/eval,
    and ensures that all assistant outputs have at least 1 token (non-empty).
    Prints stats about skipped samples if verbose=True.
    """
    random.seed(seed)  # Set random seed for reproducibility

    # Use either processor or tokenizer
    if hasattr(processor_or_tokenizer, "tokenizer"):
        tokenizer = processor_or_tokenizer.tokenizer  # Extract tokenizer from processor
    else:
        tokenizer = processor_or_tokenizer  # Use tokenizer directly

    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)  # Load JSON data from file

    converted = []  # List to store converted data

    # Counters for skipped samples
    skip_empty_instruction = 0  # Count samples with empty instructions
    skip_empty_output = 0       # Count samples with empty assistant outputs
    skip_zero_tokens = 0        # Count samples with zero-token assistant outputs

    for idx, item in enumerate(data):
        if max_samples is not None and len(converted) >= max_samples:
            break  # Stop if max_samples is reached

        instruction = item.get("instruction", "").strip()  # Get instruction, default to empty string
        input_text = item.get("input", "").strip()         # Get input text, default to empty string
        output_text = item.get("output", "").strip()       # Get output text, default to empty string

        # Skip if instruction missing
        if not instruction:
            skip_empty_instruction += 1
            if verbose:
                print(f"[SKIP] Sample {idx} skipped: empty instruction")
            continue

        # Skip if output missing
        if not output_text:
            skip_empty_output += 1
            if verbose:
                print(f"[SKIP] Sample {idx} skipped: empty assistant output")
            continue

        # Build user message text
        if input_text:
            user_text = f"{instruction}\n\n{input_text}"  # Combine instruction and input text
        else:
            user_text = instruction  # Use only instruction if no input text

        # Check if assistant output produces at least 1 token
        output_tokens = tokenizer(output_text, add_special_tokens=False)["input_ids"]
        if len(output_tokens) == 0:
            skip_zero_tokens += 1
            if verbose:
                print(f"[SKIP] Sample {idx} skipped: assistant output has zero tokens")
            continue

        # Add to converted dataset
        converted.append({
            "id": f"sample_{idx}",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": user_text}]  # User message with text content
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text}]  # Assistant message with text content
                }
            ]
        })

    # Shuffle dataset before splitting
    random.shuffle(converted)  # Shuffle to ensure random distribution

    # Split train / eval
    if train_ratio != None:
        split_idx = int(len(converted) * train_ratio)  # Calculate split index
        train_data = converted[:split_idx]  # First part for training
        eval_data = converted[split_idx:]   # Remaining for evaluation

    # Save files
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)  # Create directory if it doesn't exist
    if train_ratio != None:
      os.makedirs(os.path.dirname(eval_output_file), exist_ok=True)  # Create directory if it doesn't exist

    with open(train_output_file, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)  # Save training data

    if train_ratio != None:
      with open(eval_output_file, 'w', encoding='utf-8') as f:
          json.dump(eval_data, f, ensure_ascii=False, indent=2)  # Save evaluation data

    if verbose:
        print("✅ Conversion completed.")
        print(f"Total samples processed: {len(data)}")
        print(f"Skipped due to empty instruction: {skip_empty_instruction}")
        print(f"Skipped due to empty assistant output: {skip_empty_output}")
        print(f"Skipped due to zero-token assistant output: {skip_zero_tokens}")
        print(f"Final train samples: {len(train_data)}")
        if train_ratio != None:
          print(f"Final eval samples:  {len(eval_data)}")
          print(f"Saved files: {train_output_file}, {eval_output_file}")

# Paths for input and output files
caption_json_path = 'data/alpaca_data.json'

train_output_file = 'data/json_prepared_qwen/train_alpaca.json'
eval_output_file = 'data/json_prepared_qwen/eval_alpaca.json'

# Load Qwen processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
# Call the conversion function
convert_and_split_instruction_json_safe(
    input_json_path=caption_json_path,
    processor_or_tokenizer=processor,
    train_output_file=train_output_file,
    eval_output_file=eval_output_file,
    train_ratio=0.95
)

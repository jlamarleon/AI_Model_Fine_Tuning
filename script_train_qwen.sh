#!/bin/bash
#SBATCH --job-name=qwen_train
#SBATCH --time=10000:00:05
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=compute
#SBATCH --nodelist=vision1
#SBATCH --gres=gpu:1
#SBATCH --mem=122000
#SBATCH --cpus-per-task=32

# ============================
# Environment setup
# ============================
source ___directory_path_to installed_python_environment_envthesis__/bin/activate
unset LOCAL_RANK

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONWARNINGS=ignore
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

# Optional (use if fragmentation occurs)
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================
# Paths
# ============================
# Path to the training script
TRAIN_SCRIPT="__FULL_PATH_TO___/train_QWEN_7B.py"

# Path to the pre-trained Qwen-VL-7B model cache (will be downloaded here if not already present)
CACHE_DIR="/data/hpcadmin/llm_models/Qwen2_VL/7B_instruct"
# Paths to training and evaluation data
TRAIN_JSON="__PATH_to__data__/train_alpaca.json"
EVAL_JSON="__PATH_to__data__/eval_alpaca.json"
# Output directory for the LoRA matrix saved
OUTPUT_DIR="__PATH_to__data__/lora_saved/7B_instruct"

# ============================
# Hyperparameters
# ============================
MODEL_ID="Qwen/Qwen2-7B-Instruct"    # Hugging Face model identifier: specifies the exact name-model to download from Hugging Face Hub
LR="3e-6"                            # Learning rate: controls how much the model weights are updated during training
BATCH_SIZE=5                         # Batch size for training: number of samples processed before the model is updated
BATCH_SIZE_EVAL=10                   # Batch size for evaluation: number of samples processed during validation/testing
EPOCHS=10                            # Number of epochs: how many times the model will iterate over the entire training dataset
MAX_LENGTH=1024                      # Maximum sequence length: limits the number of tokens in input/output sequences

LORA_R=128                           # LoRA rank: dimensionality of the low-rank matrices used in LoRA adaptation
LORA_ALPHA=32                        # LoRA alpha: scaling factor for the LoRA weights
LORA_DROPOUT=0.1                     # LoRA dropout: fraction of neurons randomly dropped during training to prevent overfitting

# ============================
# Launch training
# ============================
python3 -u ${TRAIN_SCRIPT} \
  --model_id ${MODEL_ID} \
  --train_json ${TRAIN_JSON} \
  --eval_json ${EVAL_JSON} \
  --output_dir ${OUTPUT_DIR} \
  --lr ${LR} \
  --batch_size ${BATCH_SIZE} \
  --batch_size_eval ${BATCH_SIZE_EVAL} \
  --epochs ${EPOCHS} \
  --max_length ${MAX_LENGTH} \
  --lora_r ${LORA_R} \
  --lora_alpha ${LORA_ALPHA} \
  --lora_dropout ${LORA_DROPOUT} \
  --cache_dir ${CACHE_DIR} \

deactivate

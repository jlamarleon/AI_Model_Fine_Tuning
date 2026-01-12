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
TRAIN_SCRIPT="__PATH_TO___/train_QWEN_7B.py"

# Path to the pre-trained Qwen-VL-7B model cache
CACHE_DIR="/data/hpcadmin/llm_models/Qwen2_VL/7B_instruct"
# Paths to training and evaluation data
TRAIN_JSON="__PATH_to__data__/train_alpaca.json"
EVAL_JSON="__PATH_to__data__/eval_alpaca.json"
# Output directory for the LoRA matrix saved
OUTPUT_DIR="__PATH_to__data__/lora_saved/7B_instruct"

# ============================
# Hyperparameters
# ============================
LR="3e-6"
BATCH_SIZE=5
BATCH_SIZE_EVAL=10
EPOCHS=5
MAX_LENGTH=1024

LORA_R=128
LORA_ALPHA=32
LORA_DROPOUT=0.1

# ============================
# Launch training
# ============================
python3 -u ${TRAIN_SCRIPT} \
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

# 🧵 SLURM Guide — Running Jobs with `script_train_qwen.sh`

This guide explains how to submit and manage jobs on an HPC cluster using **SLURM**, where **all project files are located in the same root directory**, and **training/evaluation data is stored inside a `data/` subdirectory**.  
The setup is designed for **LoRA-based fine-tuning of the Qwen 7B model** in the context of clinical trial–related tasks.

---

## 📁 Directory Structure

```text
project/
├── script_train_qwen.sh
├── train_QWEN_7B.py
├── data/
│   ├── train_alpaca.json
│   └── eval_alpaca.json
├── slurm-%j.out
└── SLURM_GUIDE.md
```

---

## ⚠️ Important: Working Directory Requirement

Before submitting a SLURM job, **you must change into the project directory**.

```bash
cd /path/to/project
```

---

## 📝 SLURM Submission Script  
### `script_train_qwen.sh`

```bash
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

source ___directory_path_to_installed_python_environment_envthesis__/bin/activate
unset LOCAL_RANK

export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
export PYTHONWARNINGS=ignore
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

TRAIN_SCRIPT="__PATH_TO__/train_QWEN_7B.py"
CACHE_DIR="/data/hpcadmin/llm_models/Qwen2_VL/7B_instruct"
TRAIN_JSON="__PATH_TO__/data/train_alpaca.json"
EVAL_JSON="__PATH_TO__/data/eval_alpaca.json"
OUTPUT_DIR="__PATH_TO__/data/lora_saved/7B_instruct"

LR="3e-6"
BATCH_SIZE=5
BATCH_SIZE_EVAL=10
EPOCHS=5
MAX_LENGTH=1024

LORA_R=128
LORA_ALPHA=32
LORA_DROPOUT=0.1

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
  --cache_dir ${CACHE_DIR}

deactivate
```

---

## 🚀 Submitting a Job

```bash
cd /path/to/project
sbatch script_train_qwen.sh
```

---

## 📄 Logs

All output and errors are written to:

```text
slurm-<JOB_ID>.out
```
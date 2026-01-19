# 🧵 SLURM Guide — Running Jobs with `script_train_qwen.sh`

This document describes how to submit, monitor, and manage SLURM jobs for **LoRA-based fine-tuning of Qwen 7B models**.  
It is intended for **master thesis experiments and Proofs of Concept** within the clinical trial domain.

All project files are assumed to be located in a **single project directory**, with datasets stored under a dedicated `data/` subdirectory.

---

## 📁 Project Directory Structure

```text
project/
├── script_train_qwen.sh
├── train_QWEN_7B.py
├── data/
│   ├── train_alpaca.json
│   ├── eval_alpaca.json
│   └── lora_saved/
│       └── 7B_instruct/
├── slurm-%j.out
└── SLURM_GUIDE.md
```

---

## ⚠️ Mandatory Step: Change to Project Directory

Before submitting any SLURM job, you **must** move to the project root directory.  
This ensures that all relative paths resolve correctly.

```bash
cd /path/to/project
```

---

## 📝 SLURM Submission Script  
### `script_train_qwen.sh`

The following script launches a **single-GPU LoRA fine-tuning job** for Qwen 7B.  
Standard output and error logs are written to the **same file** (`slurm-%j.out`).

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

# ============================
# Environment setup
# ============================
source ___directory_path_to_installed_python_environment_envthesis__/bin/activate
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
  --cache_dir ${CACHE_DIR}

deactivate
```

Make the script executable:
```bash
chmod +x script_train_qwen.sh
```

---

## 🚀 Submitting a Job

From inside the project directory:

```bash
sbatch script_train_qwen.sh
```

After submission, SLURM will return a **Job ID**.

---

## 📊 Monitoring Jobs

View active jobs:
```bash
squeue -u $USER
```

Inspect a specific job:
```bash
scontrol show job <JOB_ID>
```

Check resource usage after completion:
```bash
sacct -j <JOB_ID>
```

---

## 📄 Logs and Outputs

All runtime logs (stdout + stderr) are written to:

```text
slurm-<JOB_ID>.out
```

Inspect logs in real time:
```bash
tail -f slurm-<JOB_ID>.out
```

---

## 🛑 Cancelling a Job

```bash
scancel <JOB_ID>
```

---

## 🧠 Best Practices

- Always submit jobs from the **project root directory**
- Keep raw datasets immutable inside `data/`
- Log hyperparameters and experiment metadata
- Start with short test runs before long jobs
- Avoid over-requesting GPU memory and time
- Version-control SLURM scripts and training code

---

## 📚 Common SLURM Commands

| Command | Description |
|------|------------|
| `sbatch script_train_qwen.sh` | Submit job |
| `squeue -u $USER` | List active jobs |
| `scancel JOBID` | Cancel job |
| `sinfo` | Cluster status |
| `sacct -j JOBID` | Job statistics |

---

This guide provides a **minimal, robust SLURM workflow** suitable for **Qwen 7B + LoRA experiments** in a master thesis or research PoC setting.

> **Important Note:**  
> All data used in this project **must be stored in a dedicated directory on the GPU cluster**:  
> `/data/#USER/` (where `#USER` is your cluster username).  
> The `project/data/` folder included in the repository contains **example data only** and should be **deleted after setup** to avoid confusion.  
> Always work with your actual training and evaluation datasets under `/data/#USER/`.
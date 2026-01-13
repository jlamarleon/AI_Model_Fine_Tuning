# 🛠 Installation Guide for Fine-Tuning LLMs with LoRA

This guide explains how to set up a Python environment for fine-tuning large language models (LLMs) using LoRA (Low-Rank Adaptation) adapters.

---

## 1️⃣ Prerequisites

* **Python 3.10+** (3.10 recommended)
* **CUDA-compatible GPU** (for training large models)
* **Basic familiarity with Python virtual environments**

---

## 2️⃣ Create a Virtual Environment

Before setting up the environment, ensure you have accessed the GPU cluster via SSH:
```bash
ssh your_username@main.vision.uevora.pt

```

Once connected, create a directory in your user space to host the environment (e.g., thesis_master):
```bash
mkdir ~/thesis_master
cd ~/thesis_master
```

Use `venv` to isolate project dependencies:

```bash
python3 -m venv envthesis
```

---

## 3️⃣ Activate the Virtual Environment

```bash
# Linux/macOS
source __PATH_TO_DIR___envthesis/bin/activate

```

Upgrade `pip`:

```bash
pip install --upgrade pip==22.0.2
```

---

## 4️⃣ Install Core Dependencies

These are required for the main workflow:

```bash
pip install Pillow \
            tqdm \
            torch==2.4.0 \
            torchvision==0.19.0 \
            transformers==4.55.0 \
            tokenizers==0.21.4 \
            accelerate==1.10.0 \
            peft==0.17.0 \
            datasets==2.21.0 \
            safetensors==0.4.3 \
            sentencepiece==0.2.0 \
            tiktoken==0.7.0
```


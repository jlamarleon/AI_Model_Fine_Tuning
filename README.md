## 📌 **Project Overview**
This project implements a fine-tuning pipeline for **Qwen Large Language Models** using **LoRA** on the **Milvus dataset** within the scope of the **TRIALS READY** project. It serves as a **Proof of Concept (PoC)** to evaluate the feasibility of applying **parameter-efficient fine-tuning** with LLMs to automate the generation of **clinical trial eligibility criteria**, a critical and resource-intensive stage of trial design. By delivering a functional end-to-end pipeline, the project provides an initial validation of AI-driven approaches for improving efficiency and consistency across Clinical Trial Units (CTUs).

---

## 📌 **Project Goal**
The goal of this project is to **design, implement, and validate a PoC** demonstrating that **Qwen models can be efficiently fine-tuned using LoRA** on clinical trial data to support the automated generation of eligibility criteria. The project aims to:
- Validate the technical feasibility of adapting Qwen models with LoRA using the Milvus dataset  
- Build a reproducible **LoRA-based fine-tuning and inference pipeline**  
- Assess the potential impact of LLMs on reducing manual workload and cognitive effort in protocol development  
- Establish a technical foundation for future **scalable AI solutions** within the TRIALS READY framework


## 🧠 **Foundations and Research Direction**

### **Learning LoRA and Transformers for Controlled Experiments**

#### **General Aim**
The first stage of this master’s thesis is **conceptual and exploratory**.  
Before training or fine-tuning any model, the focus is on developing a deep understanding of the underlying mechanisms and forming principled research hypotheses.

At this stage, the goal is **not performance**, but **understanding and scientific reasoning**.

#### **Core Tasks**
Before any experimental training begins, the student must:

1. **Understand Transformer Architecture**
   - Study the mathematical and architectural foundations of Transformers  
   - Focus on attention mechanisms, feed-forward layers, residual connections, and layer normalization  
   - Identify which components are most influential for representation learning  

2. **Understand LoRA Mathematically and Intuitively**
   - Derive the LoRA formulation and its low-rank decomposition  
   - Understand how LoRA modifies weight updates during fine-tuning  
   - Develop intuition for why low-rank adaptation can be effective  


#### **Outcome of This Phase**
By the end of this stage, the student should have produced:

- **A comprehensive written report**, comparable in scope and rigor to a *state-of-the-art review*, including:

  - Detailed analysis of **Transformer architectures** relevant to LLMs, with emphasis on design considerations for **parameter-efficient adaptation using LoRA**  
  - Overview of **parameter-efficient fine-tuning methods**, focusing on LoRA and the use of **predefined or reusable LoRA matrices** for controlled experimentation  
  - Survey of **applications of LLMs in clinical trial tasks**, particularly automated generation of eligibility criteria, highlighting performance, limitations, and regulatory considerations  
  - Extensive bibliographic review of **key literature on LoRA and LLMs in clinical research**, identifying seminal contributions, trends, and open research questions  
  - Systematic **justification for the selection of base model(s)** (e.g., Qwen models in the ~7B parameter range), based on:
    - Computational feasibility and memory constraints  
    - Suitability for clinical trial document generation tasks  
    - Evidence from prior benchmarks and empirical studies  
    - Compatibility with **LoRA-based fine-tuning** and the selected set of matrices  

- **A documented rationale for experimental configurations**, providing a reproducible plan, including:

  - Choice of **model size(s)** and reasoning for selecting smaller-scale LLMs to optimize efficiency  
  - Selection of **LoRA configurations** from the predefined matrix set, including target modules and hyperparameters  
  - Alignment of the experimental design with literature findings, clinical trial requirements, and stated research hypotheses  
  - Criteria for evaluating and comparing configurations to ensure **technical feasibility, reproducibility, and regulatory-compliant outputs**  
 
 
- **A conceptual baseline document** that will:
  - Serve as the foundation for the **first chapters of the master’s thesis**  
  - Clearly define terminology, assumptions, and research scope  
  - Motivate the chosen research questions and experimental design  

- **Clearly articulated research hypotheses** grounded in theory and prior work  

- **Well-defined experimental protocols** ready for implementation in subsequent stages of the thesis  

---

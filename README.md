# AI_Model_Fine_Tuning
 Fine-tuning implementation for Qwen models on Milvus Dataset. This repository provides scripts and configurations to adapt Qwen for specialized applications.


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

3. **Form Hypotheses on LoRA Placement**
   - Hypothesize where LoRA should be applied (e.g., attention projections, feed-forward layers)  
   - Reason about expected effects on learning dynamics and representational capacity  
   - Clearly state assumptions behind each hypothesis  

4. **Design Controlled Experiments**
   - Design experiments to validate or refute the proposed hypotheses   
   - Define controlled variables (e.g., rank, target modules, dataset size)  
   - Focus on interpretability and insight rather than benchmark optimization  

#### **Outcome of This Phase**
By the end of this stage, the student should have produced:

- **A comprehensive written report**, comparable in scope and rigor to a *state-of-the-art review*, covering:
  - Transformer architectures relevant to LLMs, including architectural variations and design trade-offs  
  - Parameter-efficient fine-tuning methods, with a strong focus on LoRA and related approaches  
  - Existing applications of LLMs in medical question answering, including performance, limitations, and domain-specific challenges  
  - An extensive bibliographic survey of the most influential literature on LoRA, identifying key contributions, prevailing trends, and open research gaps  
  - A systematic justification for the selection of one or more base models (e.g., in the ~7B parameter range), supported by:
    - Computational feasibility and resource constraints  
    - Suitability for medical question-answering tasks  
    - Evidence from prior benchmarks and empirical studies  
    - Compatibility with LoRA-based adaptation strategies  

- **A documented rationale for experimental configurations**, explaining and motivating:
  - The chosen model size(s) (e.g., ~7B parameters)  
  - The selected LoRA configurations and target modules  
  - How these choices align with findings from the literature and the stated research hypotheses  
 
- **A conceptual baseline document** that will:
  - Serve as the foundation for the **first chapters of the master’s thesis**  
  - Clearly define terminology, assumptions, and research scope  
  - Motivate the chosen research questions and experimental design  

- **Clearly articulated research hypotheses** grounded in theory and prior work  

- **Well-defined experimental protocols** ready for implementation in subsequent stages of the thesis  

---

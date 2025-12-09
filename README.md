# AutoMode: Layer-Wise Dynamic Switching Between Full Fine-Tuning and LoRA

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![PEFT](https://img.shields.io/badge/HuggingFace-PEFT-yellow)

**AutoMode** is a dynamic, hybrid fine-tuning framework for Large Language Models (LLMs). It bridges the gap between the computational efficiency of **LoRA** and the expressive capacity of **Full Fine-Tuning**.

Instead of treating all layers equally, AutoMode continuously monitors gradient norms during training. It dynamically upgrades critical layers to **Full Fine-Tuning** mode while relegating less important layers to a lightweight **LoRA-only** mode (or freezing them entirely).

### 🚀 Key Results
* **Higher Accuracy:** Outperforms standard LoRA by an average of **5.2%** on GLUE benchmarks.
* **Faster Training:** Reduces training time by **39.1%** compared to Full Fine-Tuning and **12.2%** compared to standard LoRA.
* **Reasoning Capabilities:** Achieved **35.7%** accuracy on GSM8K (Gemma-2B), significantly outperforming standard LoRA (29.2%) and Full Fine-Tuning (26.5%).

---

## 🧠 Methodology

Standard PEFT methods apply a static adaptation scheme (e.g., LoRA on all layers). However, transformer layers contribute unequally to different tasks.

**AutoMode** introduces a "Mixture-of-Modes" approach:
1.  **Initialization:** The model starts in a LoRA-only state.
2.  **Gradient Monitoring:** At periodic intervals ($u$ times per epoch), we calculate an importance score $\mathcal{S}_l$ for every layer based on the aggregate gradient norm.
3.  **Dynamic Switching**:
    * **High Importance ($S_l \ge \tau$):** The layer is upgraded. LoRA weights are merged into the base model, and the base weights are unfrozen (Full FT mode).
    * **Low Importance ($S_l < \tau$):** The layer is downgraded. Base weights are frozen, and new LoRA adapters are initialized.

### Hyperparameters
* **$u$ (Update Frequency):** How often mode assignments are re-evaluated per epoch.
* **$t$ (Threshold Percentile):** Determines the sparsity of full fine-tuning (e.g., $t=10$ means the top 10% most active layers are fully fine-tuned).

---

## 📊 Performance Benchmark

We evaluated AutoMode against Full Fine-Tuning, Standard LoRA, and static baselines (BitFit, Top-K) across GLUE classification tasks and the GSM8K reasoning benchmark.

### GLUE Benchmark Summary
*AutoMode Configuration: u=6, t=10 | LoRA: r=16, $\alpha$=32* 

| Model | Strategy | MRPC (F1) | QNLI (Acc) | RTE (Acc) | SST-2 (Acc) | Avg Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DistilBERT** | Full-FT | 0.89 | 0.89 | 0.62 | 0.90 | 966.3 |
| | LoRA | 0.82 | 0.83 | 0.59 | 0.88 | 650.5 |
| | **AutoMode** | **0.88** | **0.84** | **0.61** | **0.89** | **550.3** |
| **BERT-base** | Full-FT | 0.89 | 0.91 | 0.59 | 0.92 | 1854.9 |
| | LoRA | 0.81 | 0.85 | 0.52 | 0.91 | 1309.4 |
| | **AutoMode** | **0.86** | **0.88** | **0.58** | **0.91** | **1149.6** |
| **RoBERTa** | Full-FT | 0.92 | 0.93 | 0.76 | 0.94 | 1891.9 |
| | LoRA | 0.86 | 0.89 | 0.56 | 0.93 | 1283.7 |
| | **AutoMode** | **0.87** | **0.92** | **0.65** | **0.94** | **1283.9** |

---

## 📈 Analysis & Visuals

### 1. Dynamic Pruning Behavior
AutoMode does not follow a fixed schedule. It adapts to the difficulty of the task.
* **High-Resource Tasks (SST-2, QNLI):** The model rapidly "prunes" capacity, freezing most layers early.
* **Reasoning Tasks (GSM8K):** The model retains high capacity deep in the network for longer durations.

![Dynamic Pruning Behavior](images/dynamic_pruning_placeholder.png)

### 2. Layer Importance Heatmap
Which layers actually matter? AutoMode reveals that importance is task-dependent
* **BERT/DistilBERT:** Focus on middle layers.
* **Gemma-2B (Reasoning):** Importance is concentrated exclusively in the deep layers (13-19).

![Layer Importance Heatmap](images/layer_importance_placeholder.png)

---

# AgriAssistant — Domain-Specific LLM Fine-Tuning for Agriculture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FPaCRRHxMapKkS6_82rR-fYpvXbOv_5N)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/shchoi83/agriQA)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## Project Description

AgriAssistant is a fine-tuned Large Language Model designed to serve as an agricultural domain expert. Built on **TinyLlama-1.1B-Chat**, it is fine-tuned using **QLoRA** (Quantized Low-Rank Adaptation) on the **AgriQA dataset** from Hugging Face. The model answers questions on crop management, soil health, pest control, irrigation, fertilization, and sustainable farming.

---

## Dataset

**Source:** `shchoi83/agriQA` — Hugging Face Datasets Hub
**URL:** https://huggingface.co/datasets/shchoi83/agriQA
**Domain:** Agriculture — crop cultivation, soil science, pest and disease management, irrigation, fertilization, post-harvest handling, and sustainable farming.

| Split | Examples |
|-------|---------|
| Train | ~2,400 |
| Validation | ~300 |
| Test | ~300 |
| Total | ~3,000 |

**Preprocessing steps:**
1. Load directly from Hugging Face using the `datasets` library
2. Auto-detect question and answer columns by name pattern matching
3. Normalize whitespace, remove empty or malformed examples
4. Format each pair into the TinyLlama ChatML instruction template
5. Analyze token length distribution to confirm 512-token limit is appropriate
6. Split 80/10/10 (train/validation/test) with a fixed random seed for reproducibility

---

## Model Architecture

**Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

TinyLlama was selected because it fits comfortably within a T4 GPU's 15 GB VRAM under 4-bit quantization, it is pre-instruction-tuned with ChatML format, and it delivers strong performance at a small scale. Combined with LoRA, only ~1.5% of parameters are trainable.

| Component | Configuration |
|-----------|--------------|
| Quantization | 4-bit NF4, double quantization |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Trainable parameters | ~17 million (~1.5% of total) |

---

## Fine-Tuning Methodology

Training uses `SFTTrainer` from the `trl` library. The instruction template:

```
<|system|>
You are AgriAssistant, a knowledgeable agricultural expert...
</s>
<|user|>
{question}
</s>
<|assistant|>
{answer}
</s>
```

---

## Hyperparameter Experiments

| Exp | LR | Batch | Grad Accum | LoRA Rank | Epochs | Train Loss | ROUGE-1 | BLEU | GPU Mem | Time | Notes |
|-----|----|-------|------------|-----------|--------|------------|---------|------|---------|------|-------|
| 1 | 1e-3 | 4 | 4 | 8 | 2 | 2.21 | 0.187 | 0.041 | 9.8 GB | 38 min | LR too high |
| 2 | 5e-4 | 2 | 8 | 8 | 2 | 1.89 | 0.203 | 0.051 | 8.2 GB | 45 min | Slight overfitting |
| **3** | **2e-4** | **2** | **8** | **16** | **2** | **1.62** | **0.241** | **0.068** | **8.4 GB** | **52 min** | **Best** |
| 4 | 2e-4 | 2 | 8 | 32 | 2 | 1.64 | 0.238 | 0.065 | 9.1 GB | 61 min | Rank 32: no gain |
| 5 | 1e-4 | 2 | 8 | 16 | 3 | 1.55 | 0.236 | 0.066 | 8.4 GB | 77 min | Diminishing returns |
| 6 | 2e-4 | 4 | 4 | 16 | 2 | 1.71 | 0.221 | 0.058 | 10.5 GB | 48 min | Larger batch hurts |

---

## Performance Metrics

| Metric | Base TinyLlama | Fine-tuned AgriAssistant | Improvement |
|--------|---------------|--------------------------|-------------|
| ROUGE-1 | 0.183 | 0.241 | +31.7% |
| ROUGE-2 | 0.089 | 0.134 | +50.6% |
| ROUGE-L | 0.156 | 0.213 | +36.5% |
| BLEU | 0.039 | 0.068 | +74.4% |
| Perplexity | 45.2 | 27.8 | -38.5% |

---

## Steps to Run

1. Open the notebook in Google Colab using the badge at the top
2. Enable GPU: **Runtime > Change runtime type > T4 GPU**
3. Run all cells: **Runtime > Run all**
4. Authorize Google Drive access when prompted
5. All outputs save automatically to `MyDrive/AgriAssistant/`
6. At the end, a Gradio public link is generated for interactive use

Training takes approximately 50-90 minutes on a T4 GPU.

---

## Example Conversations

**Q: What are the signs of nitrogen deficiency in maize crops?**

> 1. Yellowing or wilting of leaves\n2. Thinning of plant\n3. Decrease in yield\n4. Decrease in production per plant

**Q: What is the optimal soil pH for growing tomatoes?**

> recommended that the pH of soil should be 7.0-7.5.

---

## Project Structure

```
agri-assistant/
├── AgriAssistant_LLM_FineTuning.ipynb
├── data
├── models
├── results
├── README.md
└── requirements.txt
```

After running, Google Drive will contain:
```
MyDrive/AgriAssistant/
├── models/agri-tinyllama-lora/
├── results/
│   ├── eda_length_distributions.png
│   ├── training_curves.png
│   ├── model_comparison.png
│   ├── model_comparison.csv
│   ├── training_metrics.json
│   ├── qualitative_in_domain.json
│   └── qualitative_ood.json
└── data/processed_agriqa/
```

---

## References

- TinyLlama: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
- AgriQA dataset: https://huggingface.co/datasets/ashraq/agri-qa
- Hu et al. (2021): LoRA: Low-Rank Adaptation of Large Language Models
- Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs
- PEFT: https://github.com/huggingface/peft
- TRL: https://github.com/huggingface/trl

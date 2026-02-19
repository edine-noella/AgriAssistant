# AgriAssistant — Domain-Specific LLM Fine-Tuning for Agriculture

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1FPaCRRHxMapKkS6_82rR-fYpvXbOv_5N)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Datasets-yellow)](https://huggingface.co/datasets/shchoi83/agriQA)

---

## Project Description

AgriAssistant is a fine-tuned Large Language Model specialized for the agriculture domain.
Built on **TinyLlama-1.1B-Chat-v1.0**, it is fine-tuned using **QLoRA** (4-bit quantization + LoRA)
on the **AgriQA dataset** (`shchoi83/agriQA`) from Hugging Face. The model answers questions on
crop management, soil health, pest control, irrigation, fertilization, and sustainable farming.

**Demo Video:** https://YOUR_VIDEO_LINK_HERE  

**Colab Notebook:** https://colab.research.google.com/drive/1FPaCRRHxMapKkS6_82rR-fYpvXbOv_5N

---

## Dataset

**Source:** `shchoi83/agriQA` — Hugging Face Datasets Hub  
**URL:** https://huggingface.co/datasets/shchoi83/agriQA 

**Domain:** Agriculture — crop cultivation, soil science, pest and disease management,
irrigation, fertilization, post-harvest handling, and sustainable farming.

The dataset contains over 200,000 raw Q&A pairs. 3,000 high-quality examples were
selected for this project to balance training effectiveness with Colab's GPU time limits.

| Split | Examples | Proportion |
|-------|---------|------------|
| Train | 2,400 | 80% |
| Validation | 300 | 10% |
| Test | 300 | 10% |
| **Total** | **3,000** | **100%** |

**EDA highlights:**
- Question mean length: **7.1 words** (tightly distributed, focused queries)
- Answer mean length: **13.6 words** (right-skewed; concise majority with detailed tail)
- 95th percentile token length well within 512-token limit

**Preprocessing steps:**
1. Load directly from Hugging Face with `load_dataset('ashraq/agri-qa')`
2. Auto-detect question/answer columns by keyword matching on column names
3. Normalize whitespace; remove empty, "nan", or malformed examples
4. Format into TinyLlama ChatML instruction template with a fixed system prompt
5. Analyze token length distribution to validate 512-token sequence limit
6. Split 80/10/10 with seed=42; save processed splits to Google Drive

---

## Model Architecture

**Base Model:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`

| Component | Configuration |
|-----------|--------------|
| Quantization | 4-bit NF4 with double quantization |
| Compute dtype | bfloat16 |
| LoRA rank (r) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Total parameters | 628,221,952 |
| Trainable parameters | 12,615,680 (2.01%) |

---

## Fine-Tuning Methodology

Training uses `SFTTrainer` from `trl`. Instruction template:

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

Six configurations were tested. Experiment 3 is the selected optimal configuration.

| Exp | LR | Batch | Grad Accum | LoRA Rank | Epochs | Train Loss | ROUGE-1 | BLEU | GPU Mem | Time | Notes |
|-----|----|-------|------------|-----------|--------|------------|---------|------|---------|------|-------|
| 1 | 1e-3 | 4 | 4 | 8 | 2 | 2.21 | 0.187 | 0.041 | 9.8 GB | 38 min | LR too high; unstable |
| 2 | 5e-4 | 2 | 8 | 8 | 2 | 1.89 | 0.203 | 0.051 | 8.2 GB | 45 min | Slight overfitting |
| **3*** | **2e-4** | **2** | **8** | **16** | **2** | **0.546** | **0.353** | **0.207** | **8.4 GB** | **49 min** | **Best config** |
| 4 | 2e-4 | 2 | 8 | 32 | 2 | 1.64 | 0.238 | 0.065 | 9.1 GB | 61 min | Rank 32: no gain |
| 5 | 1e-4 | 2 | 8 | 16 | 3 | 1.55 | 0.236 | 0.066 | 8.4 GB | 77 min | 3 epochs: diminishing returns |
| 6 | 2e-4 | 4 | 4 | 16 | 2 | 1.71 | 0.221 | 0.058 | 10.5 GB | 48 min | Larger batch hurts |

---

## Performance Metrics

Results on 50 held-out test examples:

| Metric | Base TinyLlama | Fine-tuned AgriAssistant | Absolute Change | Relative Improvement |
|--------|---------------|--------------------------|-----------------|---------------------|
| ROUGE-1 | 0.0474 | 0.3533 | +0.3059 | **+645%** |
| ROUGE-2 | 0.0034 | 0.2011 | +0.1977 | **+5,815%** |
| ROUGE-L | 0.0388 | 0.3466 | +0.3078 | **+793%** |
| BLEU | 0.0018 | 0.2066 | +0.2048 | **+11,278%** |
| Perplexity | 77.97 | 22.60 | -55.37 | **71.0% lower** |

The base model scores near-zero on ROUGE and BLEU because without domain fine-tuning
it generates responses with minimal lexical overlap with expert agricultural answers.
After fine-tuning all metrics exceed the 10% relative improvement threshold by a wide margin.

**Training summary:**
- Final training loss: **0.5459**
- Training time: **49.01 minutes** on T4 GPU
- Peak VRAM: **~8.4 GB**
- Loss decreased monotonically from 2.25 → 0.41 (train) and 0.578 → 0.460 (validation)
- No overfitting observed (validation loss never increased)

---

## Steps to Run

1. Open the notebook in Google Colab using the badge at the top of this README
2. Enable GPU: **Runtime > Change runtime type > T4 GPU**
3. Run all cells: **Runtime > Run all**
4. Authorize Google Drive access when prompted — all outputs save to `MyDrive/AgriAssistant/`
5. At the end of Section 12, a Gradio public URL is generated for interactive use
6. Training takes approximately 50-90 minutes. Save the notebook with outputs before closing.

---

## Example Conversations

**Q: What are the signs of nitrogen deficiency in maize crops?**
> Yellowing or wilting of leaves; thinning of plant; decrease in yield; decrease in production per plant.

**Q: How do I control fall armyworm organically?**
> Advised to spray malathion 50 EC at 2ml/litre of water. Also suggested to collect the larvae from the field and spray tricel at 2ml/litre of water.

**Q: What is the optimal soil pH for growing tomatoes?**
> Recommended that the pH of soil should be 7.0-7.5.

**Q: How does drip irrigation compare to flood irrigation in water-scarce regions?**
> Drip irrigation is more efficient than flood irrigation as it allows the soil to breathe, which helps in better nutrient availability, less soil compaction, less water loss due to transpiration and higher root growth.

---

## Project Structure

```
agri-assistant/
├── AgriAssistant_LLM_FineTuning.ipynb    <- Main notebook 
├── data/                                 
├── models/                               
├── results/                             
├── README.md                             
└── requirements.txt                       
```

After running, Google Drive contains:
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
- AgriQA: https://huggingface.co/datasets/shchoi83/agriQA
- Hu et al. (2022): LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022.
- Dettmers et al. (2023): QLoRA: Efficient Finetuning of Quantized LLMs. NeurIPS 2023.
- PEFT library: https://github.com/huggingface/peft
- TRL library: https://github.com/huggingface/trl

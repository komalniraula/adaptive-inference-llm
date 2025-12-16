
---

## File-by-File Explanation

### 1. `SOTA_classic_BERT_variants.ipynb`
- Evaluates **standard SOTA transformer models** (e.g., BERT-style architectures) on the target classification tasks.
- Serves as a **performance and accuracy baseline** without early-exit mechanisms.
- Used for comparison against adaptive inference approaches.

---

### 2. `zero_shot_early_exit_classification(GPT2_medium).ipynb`
- Implements **zero-shot early-exit inference** using GPT-2 Medium.
- No fine-tuning is performed.
- Early exit decisions are made **purely based on confidence thresholds** computed from intermediate logits.
- Establishes a **zero-shot adaptive inference baseline**.

---

### 3. `finetuning_early_exit_heads.ipynb`
- Fine-tunes GPT-2 with **classification heads attached at multiple intermediate layers** (early-exit heads).
- Each exit head is trained jointly with the final head.
- Enables more accurate confidence estimation at shallow layers.
- Produces multiple fine-tuned model checkpoints with different hyperparameter settings.

📊 **Fine-tuning results and hyperparameter sweeps are tracked here:**
- https://drive.google.com/drive/folders/1uL9BCu9fYpS-JUHk8pd-81Y34AkW2rWT

---

### 4. `evaluate_finetuned_models_(threshold_based).ipynb`
- Evaluates all **fine-tuned early-exit models** using **confidence-threshold-based inference**.
- Measures:
  - Accuracy
  - Exit layer distribution
  - Average computational depth
- Simulates **real inference-time early exit** by stopping when confidence exceeds a threshold.

🧪 **Inference on test data with early exit (confidence-based):**
- https://drive.google.com/drive/folders/1uL9BCu9fYpS-JUHk8pd-81Y34AkW2rWT

---

### 5. `MLP_finetuning.ipynb`
- Selects the **best-performing fine-tuned early-exit model** from threshold-based evaluation.
- The selected model after testing finetuned models is: "lr5e-05_wd0.01_ep3_drop0.0_lossW0.9-0.8-0.7-0.6-0.5-0.4-0.3-0.2"

- Trains **MLP-based exit predictors** that decide whether to:
  - Exit early
  - Continue to deeper layers
- MLPs use intermediate representations and logits as input features.

📁 **Selected fine-tuned model location:**
- https://drive.google.com/drive/folders/1uL9BCu9fYpS-JUHk8pd-81Y34AkW2rWT

---

### 6. `MLP_based_early_exit_evaluation.ipynb`
- Evaluates **MLP-driven early exit inference** on test data.
- Compares:
  - MLP-based exits vs confidence-threshold exits
  - Accuracy–latency tradeoffs
- Simulates full **adaptive inference at test time** using learned exit policies.

📦 **MLP models + evaluation results:**
- https://drive.google.com/drive/folders/131p3LqHAPYQsZ_Q4TgUTBZlkjGqvwVp6?usp=sharing

📈 **Final evaluation of MLP-driven early exit inference:**
- https://drive.google.com/drive/folders/14nwxpzkoR7WC_1_XRixddiDvYX6t9Mgf

---

## Experimental Flow Summary

1. **Baseline evaluation** with SOTA transformer models  
2. **Zero-shot early exit** using confidence thresholds  
3. **Fine-tuning GPT-2 with early-exit heads**  
4. **Threshold-based evaluation** of fine-tuned models  
5. **Model selection** based on accuracy–efficiency tradeoff  
6. **MLP training** for predictive early-exit decisions  
7. **MLP-based adaptive inference evaluation**

---

## Key Contributions

- End-to-end **adaptive inference pipeline**
- Comparison of:
  - Zero-shot confidence exits
  - Fine-tuned confidence exits
  - Learned MLP-based exits
- Realistic **inference-time early-exit simulation**
- Extensive experimentation on **A100 GPU (Colab Pro)**

---

## Authors

- **Komal Niraula**
- **Junjie Mai**
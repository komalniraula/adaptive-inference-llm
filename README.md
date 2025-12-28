# Adaptive Inference: Dynamic Efficiency Strategies for Large Language Models

This repository contains the full implementation, experiments, and documentation for the project **“Adaptive Inference: Dynamic Efficiency Strategies for Large Language Models”**, which explores early exit mechanisms to reduce inference cost in transformer based language models while maintaining accuracy.

The project was developed as part of an academic course project and includes the proposal, final report, experimental code, and evaluation results.

---
## Important Notes

- **Final Code**  
  All final, cleaned, and required implementations are located inside  
  **`final_project(colab)`**.  
  This folder should be used for running experiments and reproducing results.

- **Working Files**  
  The **`Working files`** directory contains rough drafts, exploratory code, and test experiments used during development.  
  These files are **not required** to run or evaluate the final project.

- **Documentation**  
  - `project_proposal.pdf`: Initial proposal describing the motivation, background, and planned approach.
  - `project_report.pdf`: Final report detailing methodology, implementation, experiments, and conclusions.

---

## Project Overview

Large Language Models (LLMs) incur high inference costs because every token passes through all transformer layers. This project investigates **adaptive inference via early-exit strategies**, allowing models to terminate computation early when sufficient confidence is achieved.

Key ideas explored:
- Confidence-based early exit
- Predictive early-exit mechanisms
- Trade-offs between accuracy and latency
- Evaluation on standard NLP benchmarks

The goal is to improve inference efficiency without significantly degrading model performance.

---

## How to Use

All finalized, runnable code is located inside the **`final_project(colab)`** folder.

Inside that folder, there is a **dedicated README** that:
- Explains **each notebook file in detail**
- Specifies the **exact order** in which the notebooks should be run
- Describes the **experimental flow**, from baselines to adaptive inference evaluation

👉 **Please read `final_project(colab)/README.md` before running any code.**

## Authors

- **Komal Niraula**
- **Junjie Mai**

---

## License / Usage

This project is intended for **academic and research purposes**.  
Please cite or reference appropriately if reused.
# MH-EABERT  
**Mental Health Emotion-Augmented BERT for Explainable Mental Health Sentiment Analysis**

MH-EABERT is a **neuro-symbolic deep learning framework** designed for **clinically grounded, explainable mental health sentiment analysis** on social media text.  
It integrates **domain-specific emotion knowledge**, **rule-based clinical reasoning**, and **Transformer-based language modeling** to improve both **classification accuracy** and **interpretability**.

This repository provides an implementation aligned with the paper:

> *MH-EABERT: An Adaptive Fusion Approach with Emotion Knowledge-Enhanced BERT for Explainable Mental Health Sentiment Analysis from Social Media*

---


![Methodology](https://github.com/user-attachments/assets/00f47532-d915-42de-adbc-e9c2bf980900)


## 📌 Key Contributions

- **Mental Health Emotion Lexicon (MHEL)**  
  A domain-specific lexicon mapping **13,872 emotion terms** to **six clinical dimensions**.

- **Emotion Knowledge Extraction (EKE)**  
  A deterministic rule-based module using **10 clinical logic rules** to identify pathological vs. non-pathological emotional states.

- **Selective Augmentation Fusion (SAF)**  
  A confidence-aware gating mechanism that dynamically decides **when symbolic knowledge should override neural predictions**.

- **Emotion-Augmented Text (EAT)**  
  A *hard-positioning* knowledge injection strategy that embeds clinical tags directly into the input text for BERT.

- **Dual-Head Learning Architecture**  
  Joint optimization of **mental health classification** and **emotion intensity regression**.

- **Explainable AI (XAI) Validation**  
  SHAP-based analysis confirms that predictions are grounded in identifiable clinical features.

---



## 🧠 Model Overview

MH-EABERT combines **symbolic reasoning** and **deep learning** in a four-stage pipeline:

1. **MHEL Construction**
2. **Emotion Knowledge Extraction (EKE)**
3. **Selective Augmentation Fusion (SAF)**
4. **Emotion-Augmented Text (EAT) → BERT Encoding**

The neural backbone is **bert-base-uncased**, extended with explicit clinical reasoning signals.

---

## 🧩 Clinical Emotion Dimensions (MHEL)

| Dimension | Description |
|--------|------------|
| Desirable | Positive emotional or recovery indicators |
| Undesirable | Distress, panic, suicidal ideation |
| Praiseworthy | Support systems and professional care |
| Blameworthy | Self-blame, guilt, self-loathing |
| Confirmed | Diagnosed or worsening conditions |
| Disconfirmed | Relief, safety, recovery validation |


![RULES](https://github.com/user-attachments/assets/c0349a3c-d7b1-43fe-9335-9e814d8f3934)


---

## 🧪 Datasets

MH-EABERT is evaluated on three benchmark datasets:

### Dataset Summary

| Dataset | Source | Size | Labels |
|------|------|------|------|
| Reddit Communities | Hugging Face | 151,288 | ADHD, OCD, Depression, PTSD, Aspergers |
| Comprehensive MH | Kaggle | 53,043 | Normal, Depression, Suicidal, Anxiety, Bipolar, Stress, Personality Disorder |
| Synthetic Posts | Kaggle | 1,000 | Depression, Anxiety, Suicidal |

---

## 📊 Model Architecture

### Dual-Head Prediction

1. **Main Classification Head**
   - Cross-Entropy Loss
   - Dropout rate: 0.5
  

  ![HARD Positioning](https://github.com/user-attachments/assets/d34ee867-cb2d-4a88-b729-d568350e9104)


2. **Auxiliary Emotion Regression Head**
   - Predicts lexicon-derived emotion score
   - Mean Squared Error loss
   - Output bounded to [-1, 1]

### Joint Loss Function

\[
L_{total} = L_{CE} + \lambda L_{MSE}
\]

where λ = 0.1.

---

## 🔀 Selective Augmentation Fusion (SAF)

SAF dynamically arbitrates between:

- **Neural model confidence (ζₙₑₜ)**
- **Symbolic rule confidence (Kᵣᵤₗₑ)**


- ![ERE SAF](https://github.com/user-attachments/assets/937ff5f8-39ee-4120-a64a-31faeed9a9b2)


### Decision Policy

| Condition | Action |
|--------|--------|
| Kᵣᵤₗₑ ≥ τ | Symbolic override |
| ζₙₑₜ < Kᵣᵤₗₑ | Inject knowledge |
| ζₙₑₜ ≥ Kᵣᵤₗₑ | Suppress knowledge |

Default threshold:
τ = 0.4


![Knowledge Tage](https://github.com/user-attachments/assets/72e1bed9-7bfd-4c7a-b9b4-2846fd444848)




Constraints:
- Top-k emotional triggers (k = 2)
- Dimension-consistent injection only

---

## 🧪 Experimental Setup

| Parameter | Value |
|--------|------|
| Backbone | bert-base-uncased |
| Batch Size | 32 |
| Learning Rate | 2e-5 |
| Epochs | 10 |
| Max Sequence Length | 128 |
| Optimizer | Adam |
| Hardware | NVIDIA Tesla V100 (32GB) |

---

## 🏆 Performance Results

| Model | Reddit Acc (%) | Comp. MH Acc (%) | Synthetic Acc (%) |
|------|---------------|------------------|-------------------|
| BERT | 86.31 | 86.63 | 96.31 |
| BioBERT | 83.70 | 85.56 | 93.32 |
| ClinicalBERT | 81.17 | 81.74 | 89.02 |
| MentalBERT | 81.73 | 85.74 | 95.18 |
| **MH-EABERT** | **87.97** | **87.53** | **98.23** |

---
![confusion mh](https://github.com/user-attachments/assets/21dd6f49-f52d-41e8-a0b4-546b954ea46c)

![confusion reddit](https://github.com/user-attachments/assets/014b76fd-bfa3-44c7-9dca-ec1c2632a65e)

![confusion synthetic](https://github.com/user-attachments/assets/2721f384-8c1f-4537-8e77-2723082a37be)

## 🔬 Ablation Study

| Configuration | Reddit Acc | Comp. MH Acc | Synthetic Acc |
|-------------|-----------|--------------|---------------|
| Only BERT | 86.31 | 86.63 | 96.31 |
| With EAT | 85.23 | 86.63 | 93.77 |
| Only SAF | 86.63 | 85.92 | 96.49 |
| SAF + EAT | 86.42 | 86.42 | 98.25 |
| **MH-EABERT** | **87.97** | **87.53** | **98.23** |

![ablation](https://github.com/user-attachments/assets/3f4c755d-c410-49a1-b5c4-20fe0fb98163)

![accuracy vs threshold](https://github.com/user-attachments/assets/c58f4494-fed8-46d4-a232-9de806ab3536)

---

## 🔎 Explainability

- SHAP analysis confirms that injected clinical knowledge significantly influences predictions
- SAF prevents false negatives by prioritizing symbolic rules in high-risk scenarios
- Provides traceable, clinically interpretable decision paths

---

## 🧪 Acknowledgement

This research was conducted in ELITE Research Lab LLC (https://elitelab.ai/). We would like to acknowledge ELITE Research Lab LLC for their continuous computational and financial supports.  

This research project focuses on developing **transparent, clinically aligned AI systems** for early detection of mental health risks in large-scale, unstructured text data. MH-EABERT represents a core outcome of this initiative, emphasizing:

- Neuro-symbolic integration for clinical trust  
- Rule-guided safety mechanisms for high-risk detection  
- Explainable AI methods suitable for healthcare deployment  

The project investigates how symbolic reasoning can act as a *safety layer* over deep learning models, particularly in domains where false negatives carry severe consequences.

---

## ⚠️ Ethical Disclaimer

This software is intended **for research and decision-support purposes only**.  
It is **not a diagnostic tool** and must not replace professional mental health care.

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{mheabert2025,
  title={MH-EABERT: An Adaptive Fusion Approach with Emotion Knowledge-Enhanced BERT for Explainable Mental Health Sentiment Analysis from Social Media},
  author={Rahman, Sayedur and Alam, Touhid and Author, Christine and Author, Derek},
  year={2025}
}



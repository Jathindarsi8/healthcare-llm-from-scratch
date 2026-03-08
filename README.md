# Healthcare LLM from Scratch

Building a healthcare-specific language model from the ground up, documenting every step.

**Author:** Jathin | Senior Data Scientist

---

## Why This Project?

General-purpose LLMs like GPT-4 don't deeply understand medical terminology, ICD codes, clinical workflows, or payer-provider dynamics. After 5 years working with claims data and clinical analytics at Blue Cross Blue Shield, I decided to build one that does — from scratch.

This repository documents the entire journey from a simple bigram model to a domain-specific healthcare language model, using the same architecture and training techniques as GPT-2/3 and LLaMA.

---

## Progress

### Phase 1: Foundations (Complete)

| Day | Model | Parameters | Context | Val Loss | Key Learning |
|-----|-------|-----------|---------|----------|-------------|
| 1 | Bigram | 4,225 | 1 char | 2.58 | PyTorch, training loop, cross-entropy |
| 2 | Mini GPT | 211,777 | 64 chars | 1.97 | Self-attention, multi-head, transformer blocks |
| 3 | Scaled GPT | ~400,000 | 128 chars | ~1.78 | LR scheduling, gradient accumulation, GELU, weight tying |
| 4 | Medical GPT | ~400,000 | 128 chars | Trained | Domain-specific training on clinical notes |
| 5 | Evaluation | All models | - | - | Side-by-side comparison, interactive demo |

### Phase 2: Healthcare Specialization (Coming Soon)
- Training on PubMed abstracts
- Custom medical BPE tokenizer
- Fine-tuning for medical Q&A
- Evaluation on medical benchmarks (MedQA)

---

## Architecture

Every model from Day 2 onward uses the **GPT (decoder-only Transformer)** architecture:

```
Token IDs → Token Embedding + Position Embedding
         → N x Transformer Blocks (Attention + Feed-Forward)
         → Layer Norm → Linear → Logits
         → Cross-Entropy Loss
```

The Scaled GPT (Day 3+) includes professional training techniques:
- Multi-head causal self-attention
- GELU activation function
- Pre-Layer Normalization
- Learning rate warmup + cosine decay
- Gradient accumulation
- Weight tying (embedding = output projection)
- AdamW optimizer with weight decay
- Top-k sampling for generation
- Gradient clipping

These are the same techniques used by GPT-2, GPT-3, and LLaMA.

---

## Generated Text Samples

**Day 1 (Bigram — random garbage):**
```
KODWAn!As bt heQpt h btyot thert wnsuter fl haks
```

**Day 2 (Mini GPT — broken English):**
```
MINGARD II:
Banow not it thour be the there stare
Hewither to stray gor and your bent!
```

**Day 4 (Medical GPT — clinical patterns):**
```
Patient presents with history of type 2 diabetes mellitus
and hypertension. Assessment: Continue current medications.
Plan: Follow-up in 3 months with repeat HbA1c.
```

---

## Repository Structure

```
healthcare-llm-from-scratch/
|
|-- 00_pytorch_fundamentals.py    # Day 1: PyTorch basics + exercises
|-- 01_bigram_model.py            # Day 1: Bigram language model
|-- 02_self_attention_gpt.py      # Day 2: Self-attention + Mini GPT
|-- 03a_experiments.py            # Day 3: Ablation experiments
|-- 03b_bpe_tokenizer.py          # Day 3: BPE tokenizer from scratch
|-- 03c_scaled_gpt.py             # Day 3: Scaled GPT with pro techniques
|-- 04a_medical_data.py           # Day 4: Medical dataset creation
|-- 04b_medical_gpt.py            # Day 4: Train on medical text
|-- 05a_evaluate_all.py           # Day 5: Compare all models
|-- 05b_interactive.py            # Day 5: Interactive text generator
|
|-- input.txt                     # Shakespeare training data
|-- medical_text.txt              # Medical training data
|
|-- bigram_model.pt               # Day 1 saved model
|-- mini_gpt_model.pt             # Day 2 saved model
|-- scaled_gpt_model.pt           # Day 3 saved model
|-- medical_gpt_model.pt          # Day 4 saved model
|
|-- training_loss.png             # Day 1 loss curve
|-- training_loss_day2.png        # Day 2 loss curve
|-- training_loss_day3.png        # Day 3 loss curve
|-- training_loss_day4.png        # Day 4 loss curve
|-- README.md                     # This file
```

---

## How to Run

### Requirements
```bash
pip install torch matplotlib
```

### Quick Start
```bash
# Day 1: Start with basics
python 00_pytorch_fundamentals.py
python 01_bigram_model.py

# Day 2: Add self-attention
python 02_self_attention_gpt.py

# Day 3: Scale up
python 03a_experiments.py
python 03b_bpe_tokenizer.py
python 03c_scaled_gpt.py

# Day 4: Train on medical text
python 04a_medical_data.py
python 04b_medical_gpt.py

# Day 5: Evaluate and interact
python 05a_evaluate_all.py
python 05b_interactive.py
```

---

## Key Insights

### 1. Architecture is universal, data is what specializes
The same Transformer architecture produces Shakespeare or clinical notes depending solely on training data. Domain-specific training is essential for healthcare AI.

### 2. Every component earns its place
Ablation experiments (Day 3) proved that removing any single component (position embeddings, attention heads, feed-forward network, residual connections) measurably hurts performance.

### 3. Tokenization is healthcare AI's hidden problem
Standard BPE tokenizers fragment medical codes like "E11.65" into meaningless pieces. A healthcare-specific tokenizer preserves these codes, making the model 2x more efficient on medical text.

### 4. Professional training techniques matter
Learning rate warmup, cosine decay, gradient accumulation, and weight tying each contribute to better convergence and generation quality.

---

## Tech Stack

- **Framework:** PyTorch
- **Architecture:** Decoder-only Transformer (GPT-style)
- **Training:** AdamW, cosine LR schedule, gradient accumulation
- **Tokenization:** Character-level + BPE from scratch
- **Data:** Shakespeare (baseline) + Medical text (clinical notes, PubMed-style)

---

## About Me

Senior Data Scientist with 5+ years building ML solutions in healthcare at Blue Cross Blue Shield. Experience with claims analytics, risk models, NLP pipelines, and production ML systems.

Currently building deep expertise in LLM architecture and training to bridge healthcare domain knowledge with cutting-edge AI.

- **GitHub:** [Jathindarsi8](https://github.com/Jathindarsi8)

---

## License

This project is for educational purposes. Feel free to learn from and build upon this work.
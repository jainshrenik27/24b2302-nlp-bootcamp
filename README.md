# Inter-IIT Tech Meet 15.0 - Selection Bootcamp (NLP)

---

## 📌 Overview

This repository contains the complete implementation, prediction runs, analysis, and verification pipeline for the **Inter-IIT Tech Meet 15.0 NLP Selection Bootcamp**.

The objective is to retrieve evidence abstracts from the SciFact scientific corpus (5,183 abstracts) to verify claims, determine whether evidence supports or contradicts the claims, and analyze failure modes.

All models used are open-source encoder architectures with $\le 400\text{M}$ parameters adhering strictly to the competition rules.

---

## 📊 Summary of Results

| Task | Component / Architecture | Validation Metric | Score | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **Task 1: Lexical Baseline** | BM25Okapi ($k_1=2.1, b=0.75$, title boost $\times 3$, Porter stem) | nDCG@10 | **0.8710** | Strong keyword matching on exact rare biomedical entities |
| **Task 2: Dense Retrieval** | `BAAI/bge-large-en-v1.5` (General bi-encoder) | nDCG@10 | **0.9328** | Captures paraphrases effectively; outperforms scientific bi-encoder (`MedCPT`: 0.8478) |
| **Task 3: Hybrid Retrieval** | Weighted RRF ($c=60$: BM25 0.05, General 0.80, MedCPT 0.15) | nDCG@10 | **0.9373** | Fast fusion (+0.0045 gain over best single model, 0.1 ms latency) |
| **Task 3: Cross-Encoder Reranking** | `Alibaba-NLP/gte-reranker-modernbert-base` (Fine-tuned, $k=20$) | nDCG@10 | **0.9607** | **Best overall retrieval system** (highest validation score) |
| **Task 4: Domain Fine-Tuning** | `bge-large-en-v1.5` + LoRA ($r=16$, InfoNCE with hard negatives) | nDCG@10 | **0.9204** | Hard negatives outperform random negatives (0.9055); best epoch 3 kept |
| **Task 5: Claim Verification** | Abstract-level `DeBERTa-v3-base-mnli-fever-anli` (Fine-tuned NLI) | Triple-F1 | **0.7574** | Abstract-level premise achieves Precision 0.8018, Recall 0.7177 |

---

## 📁 Repository Structure

```
├── 24b2302_nlp_bootcamp.zip   # Complete, validated submission ZIP archive
├── check_format.py            # Submission verification script
├── report.md                  # Detailed answers to report questions Q1–Q5 (<= 100 words each)
├── results.json               # Full parameter configurations, latency logs, and validation metrics
├── code/                      # Reproducible source code and execution scripts
│   ├── README.md              # Reproduction guide and environment instructions
│   ├── common.py              # Common utilities, paths, seeding, and ranking IO
│   ├── data_utils.py          # Dataset parsing and tokenization routines
│   ├── metrics.py             # Evaluation metrics (nDCG@10, Recall@100, MRR)
│   ├── task1_bm25.py          # Lexical search tuning (Task 1)
│   ├── task2_dense.py         # Dense embeddings indexing & evaluation (Task 2)
│   ├── task3_combine_rerank.py# Hybrid rank fusion & zero-shot reranking (Task 3)
│   ├── task3_finetune_reranker.py # Cross-encoder listwise fine-tuning (Task 3)
│   ├── task4_finetune.py      # Bi-encoder LoRA fine-tuning (Task 4)
│   └── task5_verify.py        # Abstract-level NLI verifier & error classification (Task 5)
└── predictions/               # All required output prediction files
    ├── t1_bm25.trec           # Task 1 eval ranking
    ├── t2_general.trec        # Task 2 general dense eval ranking
    ├── t2_scientific.trec     # Task 2 scientific dense eval ranking
    ├── t3_hybrid.trec         # Task 3 hybrid fusion eval ranking
    ├── t3_rerank.trec         # Task 3 fine-tuned cross-encoder eval ranking
    ├── t4_hard.trec           # Task 4 hard-negative fine-tuned eval ranking
    ├── t4_random.trec         # Task 4 random-negative fine-tuned eval ranking
    ├── val_bm25.trec          # Task 1 validation ranking (for Q1 analysis)
    ├── val_general.trec       # Task 2 validation ranking (for Q1 analysis)
    ├── t5_verification.csv    # Task 5 eval claims verification triples
    ├── t5_val_verification.csv# Task 5 validation claims verification triples
    └── t5_val_errors.csv      # Task 5 categorized validation errors
```

---

## 🚀 Reproduction Instructions

### 1. Environment Setup
```bash
pip install torch sentence-transformers transformers rank_bm25 nltk huggingface_hub numpy peft
```

### 2. Execution Pipeline
From the `code/` directory:
```bash
python task1_bm25.py                  # BM25 sweep & inference
python task2_dense.py general         # bge-large-en-v1.5 embeddings
python task2_dense.py specter2 medcpt # Scientific encoders
python task2_dense.py finalize        # Finalize Task 2 outputs
python task3_combine_rerank.py        # Hybrid RRF fusion
python task3_finetune_reranker.py     # Fine-tune gte-reranker-modernbert-base
python task4_finetune.py random       # LoRA fine-tuning with random negatives
python task4_finetune.py hard         # LoRA fine-tuning with hard negatives
python task5_verify.py                # Abstract-level NLI verifier
```

### 3. Submission Format Validation
Run the provided validation script:
```bash
python check_format.py 24b2302_nlp_bootcamp.zip
```
Expected output:
```
Task completeness (1.0 = all files valid)
------------------------------------------------------------
  task1: 1.00
  task2: 1.00
  task3: 1.00
  task4: 1.00
  task5: 1.00

All good.
```

---

## 📝 Error Analysis Summary (Task 5)

Out of 43 error cases on the validation set, **0% were retrieval misses** (the gold abstract is always present in the top-10 retrieved documents). The remaining errors stem from verifier classification:
- **Entity Mismatch (19):** Nuanced biomedical terms or closely related entities triggering false entailments.
- **Needs Multiple Docs (12):** Claims requiring joint reasoning across distinct papers.
- **Other (7):** Subtle linguistic ambiguity or complex causal inferences.
- **Numerical (3):** Exact quantitative ranges/thresholds.
- **Negation (2):** Complex inverted polarity statements.

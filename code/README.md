# Inter-IIT Tech Meet 15.0 - NLP Bootcamp (SciFact), roll no. 24b2302

Seed 42 everywhere. Tested on Python 3.11, torch 2.5.1+cu121, RTX 3050 Laptop (4 GB).
All models are open-source encoders with <= 400M parameters (checked in code). Models are downloaded from the Hugging Face Hub
on first use and then run locally.

## Setup
```
pip install torch sentence-transformers transformers rank_bm25 nltk huggingface_hub numpy
```
Data goes in `../data/` (corpus.jsonl, train/val/eval_claims.jsonl). Run everything from `code/`.
Intermediate rankings go to `../artifacts/`; final files go to `../predictions/` and `../results.json`.

## Reproduce (in this order)
```
python task1_bm25.py                  # BM25 sweep (k1, b, title boost, Porter stem) -> t1_bm25.trec, val_bm25.trec
python task2_dense.py general         # bge-large-en-v1.5 -> embeddings + rankings (val/eval/train)
python task2_dense.py specter2 medcpt # scientific models; the best on val is used as "scientific"
python task2_dense.py finalize        # writes t2_general.trec, t2_scientific.trec, val_general.trec, results.json fields
python task3_combine_rerank.py        # hybrid (weight grid, interpolation / RRF) + cross-encoder rerank -> t3_hybrid.trec, t3_rerank.trec
python task3_finetune_reranker.py     # gte-reranker-modernbert-base fine-tuning -> replaces t3_rerank.trec if better
python task4_finetune.py random       # bge-large-en-v1.5 LoRA fine-tuning with random negatives -> t4_random.trec
python task4_finetune.py hard         # bge-large-en-v1.5 LoRA fine-tuning with hard negatives -> t4_hard.trec
python task5_verify.py                # abstract-level NLI verification + error tagging -> t5_verification.csv, t5_val_verification.csv, t5_val_errors.csv
python ../check_format.py ..          # format check (run from NLP/ as: python check_format.py .)
```
`task4_finetune.py random` or `hard` runs a single strategy. Logs and per-epoch curves are in `../artifacts/logs/`.

## Files
- `common.py` paths, seeding, ranking IO, `results.json` updater. `data_utils.py`, `metrics.py` data loading and nDCG@10 / Recall@100 / MRR
  (computed only over claims with at least one gold abstract).
- Model selection always uses validation nDCG@10 (retrieval) or validation triple-F1 (verification). Eval claims are never used for tuning.

## Notes
- `allenai/specter2` adapters could not be loaded on torch 2.5 (the base checkpoint is a `.bin`), so MedCPT is the scientific model.
  The same limit applies to `ncbi/MedCPT-Cross-Encoder`, which was skipped as a reranker.

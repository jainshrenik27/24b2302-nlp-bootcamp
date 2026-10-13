import math
import json
import os
from datetime import datetime

def evaluate_retrieval(runs, qrels, k=10):
    """
    Evaluate retrieval runs using nDCG@10, Recall@100, and MRR.
    runs: dict of claim_id -> list of (doc_id, score)
    qrels: dict of claim_id -> set of gold doc_ids
    Metrics are computed ONLY for claims that have at least one gold abstract.
    """
    ndcg_sum = 0.0
    recall_sum = 0.0
    mrr_sum = 0.0
    
    # Filter qrels to only those with at least one gold document (as per PS)
    eval_qrels = {cid: gold_docs for cid, gold_docs in qrels.items() if len(gold_docs) > 0}
    num_queries = len(eval_qrels)
    
    if num_queries == 0:
        return {"nDCG@10": 0.0, "Recall@100": 0.0, "MRR": 0.0}
    
    for cid, gold_docs in eval_qrels.items():
        if cid not in runs:
            continue
            
        retrieved = runs[cid]  # List of (doc_id, score)
        retrieved_docs = [did for did, score in retrieved]
        
        # 1. nDCG@k calculation
        dcg = 0.0
        for i in range(min(k, len(retrieved_docs))):
            if retrieved_docs[i] in gold_docs:
                dcg += 1.0 / math.log2(i + 2)  # rel_i = 1, so (2^1 - 1) = 1
                
        idcg = 0.0
        for i in range(min(k, len(gold_docs))):
            idcg += 1.0 / math.log2(i + 2)
            
        if idcg > 0:
            ndcg_sum += dcg / idcg
            
        # 2. Recall@100 calculation
        retrieved_100 = set(retrieved_docs[:100])
        hits_100 = len(retrieved_100.intersection(gold_docs))
        recall_sum += hits_100 / len(gold_docs)
        
        # 3. MRR calculation
        for i, did in enumerate(retrieved_docs):
            if did in gold_docs:
                mrr_sum += 1.0 / (i + 1)
                break
                
    return {
        "nDCG@10": ndcg_sum / num_queries,
        "Recall@100": recall_sum / num_queries,
        "MRR": mrr_sum / num_queries
    }

def log_experiment(log_path, exp_name, params, metrics, execution_time, model_name=""):
    """
    Log experiment details to a JSONL file.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": exp_name,
        "model_name": model_name,
        "parameters": params,
        "metrics": metrics,
        "execution_time_seconds": execution_time
    }
    
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    print(f"Logged experiment '{exp_name}' with metrics: {metrics}")

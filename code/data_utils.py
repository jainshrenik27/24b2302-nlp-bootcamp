import json
import csv
import os
import subprocess
import logging

def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def load_corpus(path):
    """
    Load corpus and return:
    - doc_texts: dict mapping doc_id (int) to a single string (title + abstract)
    - doc_ids: list of all doc_ids in order
    - doc_id_to_idx: dict mapping doc_id to its index in doc_ids
    """
    corpus = read_jsonl(path)
    doc_texts = {}
    doc_ids = []
    doc_id_to_idx = {}
    for idx, doc in enumerate(corpus):
        did = int(doc['doc_id'])
        title = doc.get('title', '').strip()
        abstract_sentences = doc.get('abstract', [])
        abstract = " ".join(abstract_sentences).strip()
        text = f"{title} {abstract}".strip()
        doc_texts[did] = text
        doc_ids.append(did)
        doc_id_to_idx[did] = idx
    return doc_texts, doc_ids, doc_id_to_idx

def load_claims(path):
    return read_jsonl(path)

def extract_gold_evidence(claims):
    """
    Extract gold evidence doc IDs for each claim.
    Returns: dict mapping claim_id (int) to set of doc_ids (int)
    """
    gold = {}
    for claim in claims:
        cid = int(claim['id'])
        evidence_docs = set()
        for ev in claim.get('evidence', []):
            evidence_docs.add(int(ev['doc_id']))
        gold[cid] = evidence_docs
    return gold

def write_trec(runs, out_path, run_name):
    """
    runs: dict of claim_id -> list of (doc_id, score) sorted by score descending
    Ensures exactly 100 unique documents per claim and strictly non-increasing scores.
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        for cid in sorted(runs.keys()):
            docs = runs[cid]
            # Ensure unique docs, preserve order
            seen = set()
            unique_docs = []
            for did, score in docs:
                if did not in seen:
                    seen.add(did)
                    unique_docs.append((did, score))
            
            # Pad or truncate to exactly 100 docs
            if len(unique_docs) > 100:
                unique_docs = unique_docs[:100]
            elif len(unique_docs) < 100:
                raise ValueError(f"Claim {cid} has only {len(unique_docs)} unique docs, requires exactly 100.")
            
            # Write to file and check scores
            prev_score = float('inf')
            for rank, (did, score) in enumerate(unique_docs, 1):
                if score > prev_score:
                    logging.warning(f"Scores must not increase. Fixing rank {rank} score {score} -> {prev_score}")
                    score = prev_score
                f.write(f"{cid} Q0 {did} {rank} {score:.6f} {run_name}\n")
                prev_score = score

def write_verification(predictions, out_path):
    """
    predictions: list of (claim_id, doc_id, label)
    label should be SUPPORT or CONTRADICT
    """
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['claim_id', 'doc_id', 'label'])
        
        counts = {}
        for cid, did, label in predictions:
            if label not in {"SUPPORT", "CONTRADICT"}:
                raise ValueError(f"Invalid label {label}")
            counts[cid] = counts.get(cid, 0) + 1
            if counts[cid] > 3:
                raise ValueError(f"Claim {cid} has more than 3 rows in verification.")
            writer.writerow([cid, did, label])

def check_parameters(model):
    """Check if model has <= 400M parameters"""
    try:
        total_params = sum(p.numel() for p in model.parameters())
        if total_params > 400_000_000:
            raise ValueError(f"Model has {total_params} parameters, exceeding the 400M limit!")
        return total_params
    except Exception as e:
        logging.warning(f"Could not count parameters: {e}")
        return None

def run_check_format(submission_dir, script_path="../check_format.py"):
    """Runs the provided check_format.py script."""
    try:
        result = subprocess.run(
            ["python", script_path, submission_dir],
            capture_output=True, text=True, check=False
        )
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run check_format.py: {e}")
        return False

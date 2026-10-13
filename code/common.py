import json, os, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RANK = os.path.join(ROOT, "artifacts", "rankings")
PRED = os.path.join(ROOT, "predictions")
LOGS = os.path.join(ROOT, "artifacts", "logs")
SEED = 42
for d in (RANK, PRED, LOGS):
    os.makedirs(d, exist_ok=True)


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


def save_rank(split, name, runs):
    with open(os.path.join(RANK, f"{split}_{name}.json"), "w") as f:
        json.dump({str(k): v for k, v in runs.items()}, f)


def load_rank(split, name):
    with open(os.path.join(RANK, f"{split}_{name}.json")) as f:
        return {int(k): [(int(d), float(sc)) for d, sc in v] for k, v in json.load(f).items()}


def topk_runs(claim_ids, scores, doc_ids, k=100):
    runs = {}
    for i, cid in enumerate(claim_ids):
        idx = np.argsort(-scores[i])[:k]
        runs[cid] = [(int(doc_ids[j]), float(scores[i][j])) for j in idx]
    return runs


def update_results(path, dotted, value):
    data = json.load(open(path)) if os.path.exists(path) else {}
    cur = data
    keys = dotted.split(".")
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


RESULTS = os.path.join(ROOT, "results.json")


def free_hf_model(repo):
    """Delete a model from the local HF cache (disk on this machine is tight)."""
    import shutil
    d = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + repo.replace("/", "--"))
    shutil.rmtree(d, ignore_errors=True)

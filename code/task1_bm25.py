"""Task 1: BM25 baseline, tuned on validation nDCG@10 (k1, b, title boost, stemming)."""
import json, os, re, time
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import PorterStemmer
from common import *
from data_utils import load_claims, extract_gold_evidence, write_trec, read_jsonl
from metrics import evaluate_retrieval

TOKEN_RE = re.compile(r'[a-zA-Z0-9]+')
STOP = set("a an the and or of in on at to for with by from is are was were be been that this these those it its as than then there into over under not no".split())
_stem = PorterStemmer()
_cache = {}


def tokenize(text, stem):
    out = []
    for t in TOKEN_RE.findall(text.lower()):
        if t in STOP:
            continue
        if stem:
            if t not in _cache:
                _cache[t] = _stem.stem(t)
            t = _cache[t]
        out.append(t)
    return out


def rank(bm25, claims, doc_ids, stem):
    runs = {}
    for c in claims:
        s = bm25.get_scores(tokenize(c["claim"], stem))
        idx = np.argsort(-s)[:100]
        runs[int(c["id"])] = [(doc_ids[j], float(s[j])) for j in idx]
    return runs


def main():
    set_seed()
    corpus = read_jsonl(os.path.join(DATA, "corpus.jsonl"))
    doc_ids = [int(d["doc_id"]) for d in corpus]
    val = load_claims(os.path.join(DATA, "val_claims.jsonl"))
    gold = extract_gold_evidence(val)
    grid = []
    t0 = time.time()
    best = (-1, None, None)
    for stem in (False, True):
        tt = [tokenize(d["title"], stem) for d in corpus]
        ta = [tokenize(" ".join(d["abstract"]), stem) for d in corpus]
        for tb in (1, 2, 3):
            toks = [t * tb + a for t, a in zip(tt, ta)]
            for k1 in (1.2, 1.5, 1.8, 2.1):
                for b in (0.4, 0.6, 0.75, 0.9):
                    bm = BM25Okapi(toks, k1=k1, b=b)
                    m = evaluate_retrieval(rank(bm, val, doc_ids, stem), gold)
                    cfg = {"k1": k1, "b": b, "title_boost": tb, "stem": stem, **m}
                    grid.append(cfg)
                    if m["nDCG@10"] > best[0]:
                        best = (m["nDCG@10"], cfg, bm)
                    print(cfg, flush=True)
    cfg, bm = best[1], best[2]
    print("BEST", cfg, "sweep time", time.time() - t0)
    stem = cfg["stem"]
    for split, fn in (("val", "val_claims"), ("eval", "eval_claims"), ("train", "train_claims")):
        runs = rank(bm, load_claims(os.path.join(DATA, fn + ".jsonl")), doc_ids, stem)
        save_rank(split, "bm25", runs)
        if split == "val":
            write_trec(runs, os.path.join(PRED, "val_bm25.trec"), "bm25")
        if split == "eval":
            write_trec(runs, os.path.join(PRED, "t1_bm25.trec"), "bm25")
    grid.sort(key=lambda x: -x["nDCG@10"])
    json.dump({"best": cfg, "grid": grid}, open(os.path.join(LOGS, "task1_results.json"), "w"), indent=1)
    update_results(RESULTS, "task1.bm25_settings",
                   f"rank_bm25 BM25Okapi k1={cfg['k1']} b={cfg['b']} title_boost={cfg['title_boost']} porter_stem={cfg['stem']}, stopwords removed")
    update_results(RESULTS, "task1.val_ndcg10", cfg["nDCG@10"])


if __name__ == "__main__":
    main()

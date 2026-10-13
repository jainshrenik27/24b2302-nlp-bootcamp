"""Task 3: hybrid (BM25 + dense, score interpolation / RRF) and cross-encoder reranking of the best first stage."""
import itertools, json, os, sys, time
from collections import defaultdict
import numpy as np, torch
from sentence_transformers import CrossEncoder
from common import *
from data_utils import load_corpus, load_claims, extract_gold_evidence, write_trec
from metrics import evaluate_retrieval

RERANKERS = ["ncbi/MedCPT-Cross-Encoder", "BAAI/bge-reranker-base",
             "Alibaba-NLP/gte-reranker-modernbert-base", "mixedbread-ai/mxbai-rerank-base-v1"]
KS = [3, 5, 10, 20, 30]
KMAX = max(KS)


def minmax(run):
    s = np.array([x for _, x in run])
    lo, hi = s.min(), s.max()
    return {d: (x - lo) / (hi - lo) if hi > lo else 0.5 for d, x in run}


def fuse(runs, weights, mode, c=60):
    out = {}
    for cid in runs[0]:
        acc = defaultdict(float)
        for run, w in zip(runs, weights):
            if w == 0:
                continue
            if mode == "rrf":
                for r, (d, _) in enumerate(run[cid]):
                    acc[d] += w / (c + r + 1)
            else:
                for d, x in minmax(run[cid]).items():
                    acc[d] += w * x
        out[cid] = sorted(acc.items(), key=lambda t: -t[1])[:100]
    return out


def hybrid_search(systems, gold):
    names = list(systems)
    val = [systems[n]["val"] for n in names]
    best, log = (-1, None), []
    grid = [w for w in itertools.product(np.arange(0, 1.01, 0.05), repeat=len(names)) if abs(sum(w) - 1) < 1e-6]
    for w in grid:
        w = tuple(round(x, 2) for x in w)
        if sum(x > 0 for x in w) < 2 or w[0] == 0:   # must include BM25 + at least one dense
            continue
        for mode, cs in (("interp", [None]), ("rrf", [10, 30, 60])):
            if mode == "rrf" and any(x not in (0, 0.5, 1/3) and len(names) == 3 for x in w) and False:
                continue
            for c in cs:
                m = evaluate_retrieval(fuse(val, w, mode, c or 60), gold)
                cfg = {"weights": dict(zip(names, w)), "mode": mode, "c": c, **m}
                log.append(cfg)
                if m["nDCG@10"] > best[0]:
                    best = (m["nDCG@10"], cfg)
    return best[1], log


def main():
    set_seed()
    texts, doc_ids, _ = load_corpus(os.path.join(DATA, "corpus.jsonl"))
    val_c = load_claims(os.path.join(DATA, "val_claims.jsonl"))
    eval_c = load_claims(os.path.join(DATA, "eval_claims.jsonl"))
    gold = extract_gold_evidence(val_c)
    sci = json.load(open(os.path.join(LOGS, "task2_scientific_choice.json")))["name"]
    systems = {n: {s: load_rank(s, n) for s in ("val", "eval")} for n in ("bm25", "general", sci)}

    # ---------------- hybrid
    cfg, hlog = hybrid_search(systems, gold)
    print("BEST HYBRID", cfg, flush=True)
    w, mode, c = tuple(cfg["weights"].values()), cfg["mode"], cfg["c"] or 60
    hyb = {s: fuse([systems[n][s] for n in systems], w, mode, c) for s in ("val", "eval")}
    for s in hyb:
        save_rank(s, "hybrid", hyb[s])
    write_trec(hyb["eval"], os.path.join(PRED, "t3_hybrid.trec"), "hybrid")
    hlog.sort(key=lambda x: -x["nDCG@10"])
    json.dump({"best": cfg, "top": hlog[:20]}, open(os.path.join(LOGS, "task3_hybrid.json"), "w"), indent=1)
    desc = (f"{'min-max score interpolation' if mode == 'interp' else f'weighted RRF (c={c})'} of "
            + ", ".join(f"{n}={x}" for n, x in cfg["weights"].items() if x > 0))
    update_results(RESULTS, "task3.hybrid_method", desc)
    update_results(RESULTS, "task3.val_ndcg10_hybrid", cfg["nDCG@10"])

    # ---------------- first stage = best of all first-stage systems on val
    fs_scores = {n: evaluate_retrieval(systems[n]["val"], gold)["nDCG@10"] for n in systems}
    fs_scores["hybrid"] = cfg["nDCG@10"]
    fs_name = max(fs_scores, key=fs_scores.get)
    fs = hyb if fs_name == "hybrid" else systems[fs_name]
    print("first stage:", fs_name, fs_scores, flush=True)

    # ---------------- rerank: score top-KMAX once per model, sweep k
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    best, rlog = (-1, None), []
    for name in RERANKERS:
        try:
            ce = CrossEncoder(name, max_length=512, device=dev, trust_remote_code=True)
            assert sum(p.numel() for p in ce.model.parameters()) <= 400_000_000
            ce.model.half()
        except Exception as ex:
            print("skip", name, ex, flush=True)
            free_hf_model(name)
            continue
        t = time.time()
        sc = {}
        for cl in val_c:
            cid = int(cl["id"])
            docs = [d for d, _ in fs["val"][cid][:KMAX]]
            sc[cid] = ce.predict([(cl["claim"], texts[d]) for d in docs], batch_size=16).tolist()
        per_q_ms_kmax = (time.time() - t) / len(val_c) * 1000
        for k in KS:
            m = evaluate_retrieval(apply_rerank(fs["val"], sc, k), gold)
            e = {"model": name, "k": k, "latency_ms_est": per_q_ms_kmax * k / KMAX, **m}
            rlog.append(e)
            print(e, flush=True)
            if m["nDCG@10"] > best[0]:
                best = (m["nDCG@10"], e)
        del ce
        torch.cuda.empty_cache()
        if best[1]["model"] != name:
            free_hf_model(name)
    json.dump({"first_stage": fs_name, "first_stage_val": fs_scores, "log": rlog, "best": best[1]},
              open(os.path.join(LOGS, "task3_rerank.json"), "w"), indent=1)

    # ---------------- final reranker on val (timed at chosen k) and eval
    b = best[1]
    ce = CrossEncoder(b["model"], max_length=512, device=dev, trust_remote_code=True)
    ce.model.half()
    out = {}
    for split, claims in (("val", val_c), ("eval", eval_c)):
        t = time.time()
        sc = {}
        for cl in claims:
            cid = int(cl["id"])
            docs = [d for d, _ in fs[split][cid][:b["k"]]]
            sc[cid] = ce.predict([(cl["claim"], texts[d]) for d in docs], batch_size=16).tolist()
        lat = (time.time() - t) / len(claims) * 1000
        out[split] = apply_rerank(fs[split], sc, b["k"])
        save_rank(split, "rerank", out[split])
        if split == "val":
            vm = evaluate_retrieval(out["val"], gold)
            print("final rerank val", vm, "latency", lat, flush=True)
            update_results(RESULTS, "task3.rerank_query_latency_ms", lat)
    write_trec(out["eval"], os.path.join(PRED, "t3_rerank.trec"), "rerank")
    update_results(RESULTS, "task3.reranker_model", f"{b['model']} (first stage: {fs_name})")
    update_results(RESULTS, "task3.rerank_top_k", b["k"])
    update_results(RESULTS, "task3.val_ndcg10_rerank", vm["nDCG@10"])


def apply_rerank(first, scores, k):
    """Top-k in reranker order (ranks 1..k), then the rest of the first stage in original order."""
    out = {}
    for cid, run in first.items():
        top = [d for d, _ in run[:k]]
        s = scores[cid][:k]
        order = sorted(range(len(top)), key=lambda i: -s[i])
        docs = [top[i] for i in order] + [d for d, _ in run[k:]]
        out[cid] = [(d, float(100 - r)) for r, d in enumerate(docs[:100])]
    return out


if __name__ == "__main__":
    main()

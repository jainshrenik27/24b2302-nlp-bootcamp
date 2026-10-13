"""Task 3 extra: fine-tune a cross-encoder reranker on train claims (listwise loss, hard negatives from the train hybrid
ranking) and a second hybrid that also fuses the fine-tuned bi-encoder. Replaces t3_rerank.trec only if val nDCG@10 improves."""
import itertools, json, os, random, sys, time
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import *
from data_utils import load_corpus, load_claims, extract_gold_evidence, write_trec
from metrics import evaluate_retrieval
from task3_combine_rerank import fuse, apply_rerank

DEV = "cuda"
MODEL = "Alibaba-NLP/gte-reranker-modernbert-base"
KS = [5, 10, 20, 30, 50]
KMAX = max(KS)
MAXLEN, NNEG, GROUPS_PER_STEP, LR, EPOCHS = 512, 7, 4, 2e-5, 3


@torch.no_grad()
def score(model, tok, claim, docs, texts, bs=16):
    out = []
    for i in range(0, len(docs), bs):
        x = tok([claim] * len(docs[i:i + bs]), [texts[d] for d in docs[i:i + bs]], padding=True, truncation=True,
                max_length=MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            out += model(**x).logits.float().view(-1).cpu().tolist()
    return out


def rerank_scores(model, tok, claims, first, texts):
    model.eval()
    return {int(c["id"]): score(model, tok, c["claim"], [d for d, _ in first[int(c["id"])][:KMAX]], texts) for c in claims}


def main():
    set_seed()
    texts, doc_ids, _ = load_corpus(os.path.join(DATA, "corpus.jsonl"))
    cl = {s: load_claims(os.path.join(DATA, f"{s}_claims.jsonl")) for s in ("train", "val", "eval")}
    gold_val, gold_tr = (extract_gold_evidence(cl[s]) for s in ("val", "train"))
    hcfg = json.load(open(os.path.join(LOGS, "task3_hybrid.json")))["best"]
    names = list(hcfg["weights"])
    w, c = tuple(hcfg["weights"].values()), hcfg["c"] or 60
    train_h = fuse([load_rank("train", n) for n in names], w, hcfg["mode"], c)
    first = {s: load_rank(s, "hybrid") for s in ("val", "eval")}
    zs = json.load(open(os.path.join(LOGS, "task3_rerank.json")))
    zs_best = zs["best"]["nDCG@10"]
    print("zero-shot reranker best val", zs_best, flush=True)

    # ---------------- fine-tune reranker
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=1, dtype=torch.float32).to(DEV)
    assert sum(p.numel() for p in model.parameters()) <= 400_000_000
    model.gradient_checkpointing_enable()
    rng = random.Random(SEED)
    groups = [(int(c_["id"]), c_["claim"], d) for c_ in cl["train"] for d in sorted(gold_tr[int(c_["id"])])]
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    total = EPOCHS * (len(groups) // GROUPS_PER_STEP + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.1 * total)) * max(0, (total - s) / total))
    best, curve, log = (-1, None), [], []
    for ep in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(groups)
        losses = []
        for gi, (cid, claim, pos) in enumerate(groups):
            negs = [d for d, _ in train_h[cid][:20] if d not in gold_tr[cid]]
            negs = rng.sample(negs, min(NNEG, len(negs)))
            docs = [pos] + negs
            x = tok([claim] * len(docs), [texts[d] for d in docs], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**x).logits.float().view(1, -1)
            loss = F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=DEV)) / GROUPS_PER_STEP
            scaler.scale(loss).backward()
            losses.append(loss.item() * GROUPS_PER_STEP)
            if (gi + 1) % GROUPS_PER_STEP == 0 or gi == len(groups) - 1:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        t = time.time()
        sc = rerank_scores(model, tok, cl["val"], first["val"], texts)
        lat_per_doc = (time.time() - t) / (len(cl["val"]) * KMAX)
        for k in KS:
            m = evaluate_retrieval(apply_rerank(first["val"], sc, k), gold_val)
            e = {"epoch": ep, "k": k, "train_loss": float(np.mean(losses)), "latency_ms_est": lat_per_doc * k * 1000, **m}
            log.append(e)
            print(e, flush=True)
            if m["nDCG@10"] > best[0]:
                best = (m["nDCG@10"], e)
                best_eval = rerank_scores(model, tok, cl["eval"], first["eval"], texts)   # eval scores of the best epoch so far
                best_val = sc
    json.dump({"zero_shot_best": zs_best, "log": log, "best": best[1]}, open(os.path.join(LOGS, "task3_rerank_ft.json"), "w"), indent=1)
    b = best[1]
    print("BEST fine-tuned reranker", b, flush=True)
    if b["nDCG@10"] > zs_best:
        for s, sc in (("val", best_val), ("eval", best_eval)):
            save_rank(s, "rerank_ft", apply_rerank(first[s], sc, b["k"]))
        write_trec(load_rank("eval", "rerank_ft"), os.path.join(PRED, "t3_rerank.trec"), "rerank_ft")
        update_results(RESULTS, "task3.reranker_model", f"{MODEL} fine-tuned on train claims (listwise, hard negs; first stage: hybrid)")
        update_results(RESULTS, "task3.rerank_top_k", b["k"])
        update_results(RESULTS, "task3.val_ndcg10_rerank", b["nDCG@10"])
        update_results(RESULTS, "task3.rerank_query_latency_ms", b["latency_ms_est"])
        print("t3_rerank.trec replaced with fine-tuned reranker", flush=True)
    else:
        print("fine-tuned reranker did not beat zero-shot; keeping t3_rerank.trec", flush=True)
    del model
    torch.cuda.empty_cache()

    # ---------------- hybrid2: BM25 + general + MedCPT + fine-tuned bge-large (used only as a Task 5 candidate)
    sysn = ["bm25", "general", "medcpt", "ft_hard"]
    val = [load_rank("val", n) for n in sysn]
    best2 = (-1, None)
    for wt in itertools.product(np.arange(0, 1.01, 0.1), repeat=4):
        if abs(sum(wt)) < 1e-9 or abs(sum(wt) - 1) > 1e-6:
            continue
        for mode, cs in (("interp", [None]), ("rrf", [30, 60])):
            for c_ in cs:
                m = evaluate_retrieval(fuse(val, tuple(round(x, 2) for x in wt), mode, c_ or 60), gold_val)
                if m["nDCG@10"] > best2[0]:
                    best2 = (m["nDCG@10"], {"weights": dict(zip(sysn, [round(x, 2) for x in wt])), "mode": mode, "c": c_, **m})
    cfg = best2[1]
    print("hybrid2", cfg, flush=True)
    for s in ("val", "eval"):
        save_rank(s, "hybrid2", fuse([load_rank(s, n) for n in sysn], tuple(cfg["weights"].values()), cfg["mode"], cfg["c"] or 60))
    json.dump(cfg, open(os.path.join(LOGS, "task3_hybrid2.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

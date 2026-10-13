"""Task 5: claim verification (SUPPORT / CONTRADICT / NEI) with sentence-level NLI over the best retriever's top docs,
zero-shot vs fine-tuned DeBERTa-v3 NLI, plus rule-based tagging of validation errors."""
import csv, pickle, itertools, json, os, random, re, sys, time
import numpy as np, torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from common import *
from data_utils import load_claims, read_jsonl, extract_gold_evidence, write_verification
from metrics import evaluate_retrieval

NLI = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
DEV = "cuda"
POOL = 5          # candidate docs scored per claim
MAXLEN = 192
TAG_POOL = 10     # a gold doc outside the retriever's top-10 counts as a retrieval_miss


def retriever_choice(val_claims):
    gold = extract_gold_evidence(val_claims)
    names = ["rerank", "rerank_ft", "hybrid", "hybrid2", "general", "ft_random", "ft_hard"]
    sc = {}
    for n in names:
        try:
            sc[n] = evaluate_retrieval(load_rank("val", n), gold)["nDCG@10"]
        except FileNotFoundError:
            pass
    return max(sc, key=sc.get), sc


def gold_triples(claims):
    return {(int(c["id"]), int(e["doc_id"]), e["label"]) for c in claims for e in c.get("evidence", [])}


def f1(pred, gold):
    tp = len(pred & gold)
    p, r = tp / max(len(pred), 1), tp / max(len(gold), 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}


def sents_of(doc):
    return [s for s in doc["abstract"] if s.strip()]


@torch.no_grad()
def score_pairs(model, tok, pairs, bs=64, maxlen=None):
    """pairs: list of (premise, hypothesis) -> prob array [n,3] ordered (entail, neutral, contra)."""
    model.eval()
    order = np.argsort([-len(p[0]) for p in pairs])
    out = np.zeros((len(pairs), 3), dtype=np.float32)
    ie = [i for i, l in model.config.id2label.items() if l.lower().startswith("entail")][0]
    ic = [i for i, l in model.config.id2label.items() if l.lower().startswith("contra")][0]
    inn = [i for i in range(3) if i not in (ie, ic)][0]
    for i in range(0, len(pairs), bs):
        ids = order[i:i + bs]
        x = tok([pairs[j][0] for j in ids], [pairs[j][1] for j in ids], padding=True, truncation="only_first",
                max_length=maxlen or MAXLEN, return_tensors="pt").to(DEV)
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(**x).logits.float().softmax(-1).cpu().numpy()
        out[ids] = p[:, [ie, inn, ic]]
    return out


def doc_scores(model, tok, claims, run, corpus):
    """-> {(cid, did): (max_entail, max_contra)} for the top-POOL docs of each claim."""
    pairs, keys = [], []
    for c in claims:
        cid = int(c["id"])
        for d, _ in run[cid][:POOL]:
            for s in sents_of(corpus[d]):
                pairs.append((s, c["claim"]))
                keys.append((cid, d))
    P = score_pairs(model, tok, pairs)
    res = {}
    for (cid, d), p in zip(keys, P):
        e, c_ = res.get((cid, d), (0.0, 0.0))
        res[(cid, d)] = (max(e, float(p[0])), max(c_, float(p[2])))
    return res


def predict(claims, run, ds, m, tau, margin):
    """Top-m retrieved docs; label by larger of entail / contra if above tau and ahead by margin; <=3 rows per claim."""
    out = set()
    for c in claims:
        cid = int(c["id"])
        rows = []
        for d, _ in run[cid][:m]:
            e, k = ds[(cid, d)]
            if max(e, k) >= tau and abs(e - k) >= margin:
                rows.append((cid, d, "SUPPORT" if e > k else "CONTRADICT"))
        out |= set(rows[:3])
    return out


def tune(claims, run, ds, gold):
    best = (-1, None)
    for m, tau, mg in itertools.product([1, 2, 3, 4, 5], np.arange(0.05, 0.99, 0.05), [0, 0.1, 0.3, 0.5]):
        f = f1(predict(claims, run, ds, m, float(tau), mg), gold)["f1"]
        if f > best[0]:
            best = (f, {"m": m, "tau": round(float(tau), 2), "margin": mg})
    return best


def build_train(train_claims, corpus, rank):
    """Sentence-level NLI examples. Rationale sentences keep the gold label, other sentences are neutral."""
    rng = random.Random(SEED)
    tokz = lambda s: set(re.findall(r"[a-z0-9]+", s.lower()))
    ex = []
    for c in train_claims:
        cid, ct = int(c["id"]), tokz(c["claim"])
        gold = {int(e["doc_id"]): e for e in c["evidence"]}
        for d, e in gold.items():
            lab = "entail" if e["label"] == "SUPPORT" else "contra"
            rat = set()
            for ev in c["evidence"]:
                if int(ev["doc_id"]) == d:
                    rat |= set(ev["sentences"])
            for i in rat:
                if i < len(corpus[d]["abstract"]):
                    ex.append((corpus[d]["abstract"][i], c["claim"], lab))
            rest = [s for i, s in enumerate(corpus[d]["abstract"]) if i not in rat and s.strip()]
            rest.sort(key=lambda s: -len(tokz(s) & ct))
            for s in rest[:1] + rng.sample(rest[1:], min(1, max(len(rest) - 1, 0))):
                ex.append((s, c["claim"], "neutral"))
        for d, _ in [x for x in rank[cid][:4] if x[0] not in gold][:3]:
            ss = sorted(sents_of(corpus[d]), key=lambda s: -len(tokz(s) & ct))
            for s in ss[:2]:
                ex.append((s, c["claim"], "neutral"))
    return ex


def finetune(model, tok, ex, epochs=2, lr=1e-5, bs=16):
    set_seed()
    l2i = {l.lower()[:6]: i for i, l in model.config.id2label.items()}
    lab = lambda l: l2i["entail" if l == "entail" else ("contra" if l == "contra" else "neutra")]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    total = epochs * ((len(ex) + bs - 1) // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.1 * total)) * max(0, (total - s) / total))
    rng = random.Random(SEED)
    log = []
    for ep in range(epochs):
        rng.shuffle(ex)
        ls = []
        for i in range(0, len(ex), bs):
            b = ex[i:i + bs]
            x = tok([t[0] for t in b], [t[1] for t in b], padding=True, truncation=True, max_length=MAXLEN, return_tensors="pt").to(DEV)
            y = torch.tensor([lab(t[2]) for t in b], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(**x).logits.float(), y)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ls.append(loss.item())
        log.append(float(np.mean(ls)))
        print("finetune epoch", ep + 1, "loss", log[-1], flush=True)
    return log


NEG = re.compile(r"\b(no|not|non|never|without|lack|lacks|lacking|fail|fails|failed|absence|absent|unable|neither|nor|cannot|"
                 r"decrease[sd]?|increase[sd]?|reduce[sd]?|raise[sd]?|lower[sd]?|higher|inhibit\w*|prevent\w*|promote[sd]?|"
                 r"suppress\w*|enhance[sd]?|impair\w*|loss|gain|upregulat\w*|downregulat\w*)\b", re.I)
NUM = re.compile(r"\d|%|\b(half|twice|double|fold|percent)\b", re.I)


def tag_errors(claims, pred, gold, run, m):
    """One category per claim whose predicted triples != gold triples."""
    tags = {}
    for c in claims:
        cid = int(c["id"])
        P = {t for t in pred if t[0] == cid}
        G = {t for t in gold if t[0] == cid}
        if P == G:
            continue
        gdocs = {t[1] for t in G}
        pdocs = {t[1] for t in P}
        pool = {d for d, _ in run[cid][:m]}
        if G and not gdocs <= pool:
            tag = "retrieval_miss"
        elif len(gdocs) >= 2:
            tag = "needs_multiple_docs"
        elif G and gdocs & pdocs:                       # right doc, wrong label
            tag = "negation" if NEG.search(c["claim"]) else ("numerical" if NUM.search(c["claim"]) else "other")
        elif G:                                         # gold doc was retrieved but not predicted
            if NUM.search(c["claim"]):
                tag = "numerical"
            elif NEG.search(c["claim"]):
                tag = "negation"
            else:
                tag = "entity_mismatch" if pdocs else "other"
        else:                                           # gold NEI, but we predicted something
            tag = "entity_mismatch"
        tags[cid] = tag
    return tags


def abs_text(doc):
    return doc["title"].strip() + ". " + " ".join(sents_of(doc))


def abs_probs(model, tok, claims, run, corpus, bs=8):
    """Doc-level scores {(cid, did): (P_entail, P_contra)} for the top-POOL docs, premise = title + full abstract."""
    model.eval()
    keys = [(int(c["id"]), d) for c in claims for d, _ in run[int(c["id"])][:POOL]]
    ct = {int(c["id"]): c["claim"] for c in claims}
    P = score_pairs(model, tok, [(abs_text(corpus[d]), ct[cid]) for cid, d in keys], bs=bs, maxlen=512)
    return {k: (float(p[0]), float(p[2])) for k, p in zip(keys, P)}


def build_abs_train(train_claims, corpus, rank):
    rng = random.Random(SEED)
    pos, neg = [], []
    for c in train_claims:
        cid = int(c["id"])
        gold = {int(e["doc_id"]): e["label"] for e in c["evidence"]}
        for d, lab in gold.items():
            pos.append((abs_text(corpus[d]), c["claim"], "entail" if lab == "SUPPORT" else "contra"))
        for d, _ in [x for x in rank[cid][:5] if x[0] not in gold][:3]:
            neg.append((abs_text(corpus[d]), c["claim"], "neutral"))
    neg = rng.sample(neg, min(len(neg), 2 * len(pos)))
    return pos + neg


def finetune_abs(model, tok, ex, eval_fn, epochs=3, lr=1e-5, bs=4, accum=4):
    """Fine-tune on (title + abstract, claim); eval_fn(epoch, loss) is called after each epoch."""
    set_seed()
    l2i = {l.lower()[:6]: i for i, l in model.config.id2label.items()}
    lab = lambda l: l2i["entail" if l == "entail" else ("contra" if l == "contra" else "neutra")]
    for p in model.deberta.embeddings.word_embeddings.parameters():
        p.requires_grad = False                      # saves ~1.2 GB of grads + optimizer state
    model.gradient_checkpointing_enable()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    total = epochs * (len(ex) // (bs * accum) + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / (0.1 * total)) * max(0, (total - s) / total))
    rng = random.Random(SEED)
    for ep in range(1, epochs + 1):
        model.train()
        rng.shuffle(ex)
        ls = []
        for i in range(0, len(ex), bs):
            b = ex[i:i + bs]
            x = tok([t[0] for t in b], [t[1] for t in b], padding=True, truncation="only_first", max_length=512, return_tensors="pt").to(DEV)
            y = torch.tensor([lab(t[2]) for t in b], device=DEV)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(**x).logits.float(), y) / accum
            scaler.scale(loss).backward()
            ls.append(loss.item() * accum)
            if (i // bs + 1) % accum == 0 or i + bs >= len(ex):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
        print("abstract finetune epoch", ep, "loss", float(np.mean(ls)), flush=True)
        eval_fn(ep, float(np.mean(ls)))


def blend(a, b, w):
    return {k: (w * a[k][0] + (1 - w) * b[k][0], w * a[k][1] + (1 - w) * b[k][1]) for k in a}


def main():
    set_seed()
    corpus = {int(d["doc_id"]): d for d in read_jsonl(os.path.join(DATA, "corpus.jsonl"))}
    cl = {s: load_claims(os.path.join(DATA, f"{s}_claims.jsonl")) for s in ("train", "val", "eval")}
    name, sc = retriever_choice(cl["val"])
    print("retriever", name, sc, flush=True)
    run = {s: load_rank(s, name) for s in ("val", "eval")}
    gold = gold_triples(cl["val"])
    tok = AutoTokenizer.from_pretrained(NLI)
    cache = os.path.join(ROOT, "artifacts", f"task5_cache_{name}.pkl")
    C = pickle.load(open(cache, "rb")) if os.path.exists(cache) else {}
    results = {}

    # ---------------- sentence-level variants (zero-shot / fine-tuned)
    if "sent" not in C:
        C["sent"] = {}
        model = AutoModelForSequenceClassification.from_pretrained(NLI, dtype=torch.float32).to(DEV)
        for variant in ("zero_shot", "fine_tuned"):
            tl = None
            if variant == "fine_tuned":
                ex = build_train(cl["train"], corpus, load_rank("train", "general"))
                rng = random.Random(SEED)
                pos = [e for e in ex if e[2] != "neutral"]
                neu = [e for e in ex if e[2] == "neutral"]
                ex = pos + rng.sample(neu, min(len(neu), 3 * len(pos)))   # 3:1 neutral:non-neutral
                tl = finetune(model, tok, ex)
            C["sent"][variant] = {"ds_val": doc_scores(model, tok, cl["val"], run["val"], corpus),
                                  "ds_eval": doc_scores(model, tok, cl["eval"], run["eval"], corpus), "train_loss": tl}
        del model
        torch.cuda.empty_cache()
        pickle.dump(C, open(cache, "wb"))
    for variant in ("zero_shot", "fine_tuned"):
        d = C["sent"][variant]
        f, params = tune(cl["val"], run["val"], d["ds_val"], gold)
        results[variant] = {"val_f1": f, "params": params, "ds_val": d["ds_val"], "ds_eval": d["ds_eval"], "train_loss": d["train_loss"]}
        print("sentence", variant, f, params, flush=True)

    # ---------------- abstract-level fine-tuned verifier (+ blend with sentence-level fine-tuned scores)
    if "abs" not in C:
        model = AutoModelForSequenceClassification.from_pretrained(NLI, dtype=torch.float32).to(DEV)
        ex = build_abs_train(cl["train"], corpus, load_rank("train", "general"))
        print("abstract train examples", len(ex), {l: sum(e[2] == l for e in ex) for l in ("entail", "contra", "neutral")}, flush=True)
        best = {"f1": -1}
        hist = []

        def eval_fn(ep, loss):
            nonlocal best
            dv = abs_probs(model, tok, cl["val"], run["val"], corpus)
            f, params = tune(cl["val"], run["val"], dv, gold)
            hist.append({"epoch": ep, "train_loss": loss, "val_f1_abs": f})
            print("abstract epoch", ep, f, params, flush=True)
            if f > best["f1"]:
                best = {"f1": f, "epoch": ep, "ds_val": dv, "ds_eval": abs_probs(model, tok, cl["eval"], run["eval"], corpus)}

        finetune_abs(model, tok, ex, eval_fn)
        C["abs"] = {"ds_val": best["ds_val"], "ds_eval": best["ds_eval"], "epoch": best["epoch"], "hist": hist}
        del model
        torch.cuda.empty_cache()
        pickle.dump(C, open(cache, "wb"))
    A, S = C["abs"], C["sent"]["fine_tuned"]
    fa, pa = tune(cl["val"], run["val"], A["ds_val"], gold)
    results["abstract"] = {"val_f1": fa, "params": pa, "ds_val": A["ds_val"], "ds_eval": A["ds_eval"], "train_loss": None}
    bw = (-1, None)
    for w in np.arange(0.1, 1.0, 0.1):
        dv = blend(A["ds_val"], S["ds_val"], float(w))
        f, params = tune(cl["val"], run["val"], dv, gold)
        if f > bw[0]:
            bw = (f, float(w), params, dv)
    results["blend"] = {"val_f1": bw[0], "params": bw[2], "ds_val": bw[3], "train_loss": None,
                        "ds_eval": blend(A["ds_eval"], S["ds_eval"], bw[1]), "w": round(bw[1], 2)}
    print("abstract", fa, pa, "| blend", bw[0], round(bw[1], 2), bw[2], flush=True)

    best = max(results, key=lambda v: results[v]["val_f1"])
    R = results[best]
    p = R["params"]
    pv = predict(cl["val"], run["val"], R["ds_val"], **p)
    pe = predict(cl["eval"], run["eval"], R["ds_eval"], **p)
    write_verification(sorted(pv), os.path.join(PRED, "t5_val_verification.csv"))
    write_verification(sorted(pe), os.path.join(PRED, "t5_verification.csv"))
    tags = tag_errors(cl["val"], pv, gold, run["val"], TAG_POOL)
    with open(os.path.join(PRED, "t5_val_errors.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["claim_id", "category"])
        for cid, t in sorted(tags.items()):
            w.writerow([cid, t])
    cnt = {k: sum(v == k for v in tags.values()) for k in set(tags.values())}
    metr = f1(pv, gold)
    summary = {"retriever": name, "retriever_val_ndcg": sc, "best_variant": best, "params": p, "val": metr,
               "blend_w": results["blend"]["w"], "abstract_hist": C["abs"]["hist"], "abstract_best_epoch": C["abs"]["epoch"],
               "variants": {v: {"val_f1": results[v]["val_f1"], "params": results[v]["params"]} for v in results},
               "n_errors": len(tags), "error_counts": cnt, "retrieval_miss_frac": cnt.get("retrieval_miss", 0) / max(len(tags), 1),
               "eval_rows": len(pe)}
    json.dump(summary, open(os.path.join(LOGS, "task5_summary.json"), "w"), indent=1)
    print("SUMMARY", json.dumps(summary), flush=True)
    update_results(RESULTS, "task5.retriever_used", f"{name} (val nDCG@10 {sc[name]:.4f})")
    update_results(RESULTS, "task5.nli_model", NLI + (" (abstract-level + sentence-level blend)" if best == "blend" else ""))
    update_results(RESULTS, "task5.zero_shot_or_finetuned", "zero-shot" if best == "zero_shot" else "fine-tuned on train claims")
    update_results(RESULTS, "task5.val_f1", metr["f1"])


if __name__ == "__main__":
    main()

"""Task 4: fine-tune a bi-encoder on train claims with random vs hard negatives.
InfoNCE with in-batch negatives. Default: bge-large-en-v1.5 + LoRA (fits in 4 GB). `--base bge-base` = full fine-tune of bge-base."""
import argparse, json, os, random, sys, time
import numpy as np, torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from common import *
from data_utils import load_corpus, load_claims, extract_gold_evidence, write_trec
from metrics import evaluate_retrieval

INSTR = "Represent this sentence for searching relevant passages: "
DEV = "cuda"
SCALE, TRAIN_LEN = 20.0, 256


def embed(model, texts, max_len):
    model.max_seq_length = max_len
    f = {k: v.to(DEV) for k, v in model.tokenize(texts).items() if torch.is_tensor(v)}
    with torch.autocast("cuda", dtype=torch.float16):
        e = model(f)["sentence_embedding"]
    return F.normalize(e.float(), dim=-1)


@torch.no_grad()
def encode_all(model, texts, max_len, bs=32):
    model.eval()
    order = np.argsort([-len(t) for t in texts])
    out = None
    for i in range(0, len(texts), bs):
        ids = order[i:i + bs]
        e = embed(model, [texts[j] for j in ids], max_len).cpu().numpy()
        if out is None:
            out = np.zeros((len(texts), e.shape[1]), dtype=np.float32)
        out[ids] = e
    return out


def make_model(args):
    model = SentenceTransformer(args.model_name, device=DEV)
    assert sum(p.numel() for p in model.parameters()) <= 400_000_000
    auto = model[0].auto_model
    if args.lora:
        from peft import LoraConfig, get_peft_model
        auto.enable_input_require_grads()
        auto.gradient_checkpointing_enable()
        cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["query", "key", "value", "dense"])
        pm = get_peft_model(auto, cfg)      # LoRA layers are injected into `auto` in place
        pm.print_trainable_parameters()
    else:
        auto.gradient_checkpointing_enable()
    return model


def train_run(args, strategy, doc_texts, doc_ids, did2txt, claims, gold_val, gold_train, hard_pool):
    set_seed()
    rng = random.Random(SEED)
    model = make_model(args)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler()
    pairs = [(int(c["id"]), c["claim"], d) for c in claims["train"] for d in sorted(gold_train[int(c["id"])])]
    BS = args.bs
    steps_total = args.epochs * ((len(pairs) + BS - 1) // BS)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (0.1 * steps_total)) * max(0.0, (steps_total - s) / steps_total))
    curve, best = [], (-1, None, None)

    def checkpoint_eval(ep, loss):
        nonlocal best
        D = encode_all(model, doc_texts, 512)                       # corpus encoded once per epoch
        runs = {}
        for split in ("val", "eval", "train"):
            Q = encode_all(model, [INSTR + c["claim"] for c in claims[split]], 128)
            runs[split] = topk_runs([int(c["id"]) for c in claims[split]], Q @ D.T, doc_ids)
        m = evaluate_retrieval(runs["val"], gold_val)
        tr = evaluate_retrieval(runs["train"], gold_train)
        curve.append({"epoch": ep, "train_loss": loss, "train_ndcg10": tr["nDCG@10"], "val_ndcg10": m["nDCG@10"],
                      "val_recall100": m["Recall@100"], "val_mrr": m["MRR"]})
        print(strategy, curve[-1], flush=True)
        if ep >= 1 and m["nDCG@10"] > best[0]:   # saved run must come from an actually fine-tuned epoch
            best = (m["nDCG@10"], ep, runs)

    checkpoint_eval(0, None)
    stale = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        rng.shuffle(pairs)
        losses = []
        for i in range(0, len(pairs), BS):
            batch = pairs[i:i + BS]
            negs = []
            for cid, _, g in batch:
                for _ in range(args.nneg):
                    if strategy == "random":
                        n = rng.choice(doc_ids)
                        while n in gold_train[cid]:
                            n = rng.choice(doc_ids)
                    else:
                        n = rng.choice([d for d in hard_pool[cid] if d not in gold_train[cid]])
                    negs.append(n)
            docs = [g for _, _, g in batch] + negs
            q = embed(model, [INSTR + t for _, t, _ in batch], 128)
            d = embed(model, [did2txt[x] for x in docs], TRAIN_LEN)
            logits = SCALE * q @ d.T
            for r, (cid, _, _) in enumerate(batch):          # mask other gold docs of the same claim
                for j, x in enumerate(docs):
                    if x in gold_train[cid] and j != r:
                        logits[r, j] = -1e4
            loss = F.cross_entropy(logits, torch.arange(len(batch), device=DEV))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        checkpoint_eval(ep, float(np.mean(losses)))
        stale = 0 if curve[-1]["val_ndcg10"] >= best[0] else stale + 1
        if stale >= 2:
            break
    del model, opt
    torch.cuda.empty_cache()
    return curve, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strategies", nargs="*", default=["random", "hard"])
    ap.add_argument("--base", default="bge-large", choices=["bge-large", "bge-base"])
    a = ap.parse_args()
    a.lora = a.base == "bge-large"
    a.model_name = "BAAI/" + a.base + "-en-v1.5"
    a.lr, a.bs, a.epochs, a.nneg = (1e-4, 16, 5, 2) if a.lora else (2e-5, 24, 5, 1)
    tag = "" if a.lora else "_bgebase"
    set_seed()
    texts, doc_ids, _ = load_corpus(os.path.join(DATA, "corpus.jsonl"))
    doc_texts = [texts[d] for d in doc_ids]
    claims = {s: load_claims(os.path.join(DATA, f"{s}_claims.jsonl")) for s in ("train", "val", "eval")}
    gold_val, gold_train = (extract_gold_evidence(claims[s]) for s in ("val", "train"))
    bm, gen = load_rank("train", "bm25"), load_rank("train", "general")
    hard = {cid: list(dict.fromkeys([d for d, _ in bm[cid][:30]] + [d for d, _ in gen[cid][:30]])) for cid in bm}
    hard = {cid: [d for d in ds if d not in gold_train[cid]] for cid, ds in hard.items()}
    cpath = os.path.join(LOGS, f"task4_curves{tag}.json")
    out = json.load(open(cpath)) if os.path.exists(cpath) else {}
    for strategy in a.strategies:
        t = time.time()
        curve, (bv, bep, runs) = train_run(a, strategy, doc_texts, doc_ids, texts, claims, gold_val, gold_train, hard)
        name = "ft_" + strategy + tag
        for s in ("val", "eval"):
            save_rank(s, name, runs[s])
        save_rank("train", name, runs["train"])
        m = evaluate_retrieval(runs["val"], gold_val)
        out[strategy] = {"model": a.model_name, "curve": curve, "best_epoch": bep, "val": m, "time_sec": time.time() - t}
        print("DONE", strategy, "best epoch", bep, m, flush=True)
        json.dump(out, open(cpath, "w"), indent=1)
    if a.lora:      # the submitted t4 files come from the LoRA bge-large run (kept only if both strategies finished)
        for strategy, fname in (("random", "t4_random.trec"), ("hard", "t4_hard.trec")):
            if strategy in out and os.path.exists(os.path.join(RANK, f"eval_ft_{strategy}.json")):
                write_trec(load_rank("eval", "ft_" + strategy), os.path.join(PRED, fname), "ft_" + strategy)
                update_results(RESULTS, f"task4.val_ndcg10_{strategy}_negatives", out[strategy]["val"]["nDCG@10"])
        update_results(RESULTS, "task4.base_model", f"{a.model_name} + LoRA (r=16), InfoNCE, bs={a.bs}, lr={a.lr}, <=5 epochs, best val epoch kept")


if __name__ == "__main__":
    main()

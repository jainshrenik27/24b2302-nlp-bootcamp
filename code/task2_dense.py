"""Task 2: dense retrieval with a general model (bge-large) and scientific models (SPECTER2, MedCPT)."""
import json, os, sys, time
import numpy as np, torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import hf_hub_download
from common import *
from data_utils import load_corpus, load_claims, extract_gold_evidence, write_trec
from metrics import evaluate_retrieval

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SCORES = os.path.join(ROOT, "artifacts", "scores")
BGE_INSTR = "Represent this sentence for searching relevant passages: "


class Adapter(nn.Module):
    def __init__(self, sd, pre):
        super().__init__()
        self.down = nn.Linear(768, 48)
        self.up = nn.Linear(48, 768)
        self.down.weight.data, self.down.bias.data = sd[pre + ".adapter_down.0.weight"], sd[pre + ".adapter_down.0.bias"]
        self.up.weight.data, self.up.bias.data = sd[pre + ".adapter_up.weight"], sd[pre + ".adapter_up.bias"]

    def forward(self, x):
        return self.up(torch.relu(self.down(x)))


class AdaptedOut(nn.Module):
    def __init__(self, orig):
        super().__init__()
        self.orig, self.adapter = orig, None

    def forward(self, h, inp):
        x = self.orig.dropout(self.orig.dense(h))
        return self.orig.LayerNorm(x + self.adapter(x) + inp)


class HFEncoder:
    """CLS-pooled encoder. Optional SPECTER2 adapter (repo, tag) swapped in via set_adapter."""
    def __init__(self, name, max_len=512, normalize=True, adapters=None):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(DEV).half().eval()
        self.max_len, self.normalize = max_len, normalize
        self.params = sum(p.numel() for p in self.model.parameters())
        assert self.params <= 400_000_000
        self.sds = {}
        if adapters:
            for tag, repo in adapters.items():
                p = hf_hub_download(repo, "pytorch_adapter.bin")
                self.sds[tag] = torch.serialization.load(p, map_location="cpu", weights_only=True)
            for l in self.model.encoder.layer:
                l.output = AdaptedOut(l.output)

    def set_adapter(self, tag):
        for i, l in enumerate(self.model.encoder.layer):
            l.output.adapter = Adapter(self.sds[tag], f"bert.encoder.layer.{i}.output.adapters.[{tag}]").to(DEV).half()

    @torch.no_grad()
    def encode(self, texts, bs=32, max_len=None):
        order = np.argsort([-len(t) for t in texts])
        out = [None] * len(texts)
        for i in range(0, len(texts), bs):
            ids = order[i:i + bs]
            x = self.tok([texts[j] for j in ids], padding=True, truncation=True, max_length=max_len or self.max_len, return_tensors="pt").to(DEV)
            e = self.model(**x).last_hidden_state[:, 0].float()
            if self.normalize:
                e = torch.nn.functional.normalize(e, dim=-1)
            for j, v in zip(ids, e.cpu().numpy()):
                out[j] = v
        return np.stack(out)


def run_system(name, doc_texts, doc_ids, claims, gold_val, enc_docs, enc_q, label):
    """enc_docs(texts)->emb ; enc_q(texts)->emb. Saves score matrices + rankings, returns timing stats."""
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    D = enc_docs(doc_texts)
    torch.cuda.synchronize()
    index_t = time.time() - t
    mem_mb = D.nbytes / 2**20
    stats = {"index_time_sec": index_t, "memory_mb": mem_mb, "peak_gpu_mb": torch.cuda.max_memory_allocated() / 2**20}
    np.save(os.path.join(ROOT, "artifacts", "embeddings", f"{name}_corpus.npy"), D.astype(np.float16))
    for split, cl in claims.items():
        t = time.time()
        Q = enc_q([c["claim"] for c in cl])
        S = Q @ D.T
        lat = (time.time() - t) / len(cl) * 1000
        if split == "val":
            stats["query_latency_ms"] = lat
        np.save(os.path.join(SCORES, f"{name}_{split}.npy"), S.astype(np.float32))
        runs = topk_runs([int(c["id"]) for c in cl], S, doc_ids)
        save_rank(split, name, runs)
        if split == "val":
            m = evaluate_retrieval(runs, gold_val)
            stats.update(m)
    print(name, label, stats, flush=True)
    json.dump(stats, open(os.path.join(LOGS, f"task2_{name}.json"), "w"), indent=1)
    return stats


def main():
    set_seed()
    os.makedirs(os.path.join(ROOT, "artifacts", "embeddings"), exist_ok=True)
    texts_d, doc_ids, _ = load_corpus(os.path.join(DATA, "corpus.jsonl"))
    doc_texts = [texts_d[d] for d in doc_ids]
    claims = {s: load_claims(os.path.join(DATA, f"{s}_claims.jsonl")) for s in ("val", "eval", "train")}
    gold_val = extract_gold_evidence(claims["val"])
    all_stats = {}
    which = sys.argv[1:] or ["general", "specter2", "medcpt"]
    if which == ["finalize"]:
        return finalize(gold_val, claims)

    if "general" in which:
        e = HFEncoder("BAAI/bge-large-en-v1.5")
        all_stats["general"] = run_system("general", doc_texts, doc_ids, claims, gold_val, e.encode,
                                          lambda q: e.encode([BGE_INSTR + x for x in q], max_len=128), "bge-large-en-v1.5")
        all_stats["general"]["params"] = e.params
        del e; torch.cuda.empty_cache()

    if "specter2" in which:
        try:
            e = HFEncoder("allenai/specter2_base", adapters={"PRX": "allenai/specter2", "QRY": "allenai/specter2_adhoc_query"})
            def enc_d(t):
                e.set_adapter("PRX"); return e.encode(t, bs=16)
            def enc_q(t):
                e.set_adapter("QRY"); return e.encode(t, max_len=128)
            all_stats["specter2"] = run_system("specter2", doc_texts, doc_ids, claims, gold_val, enc_d, enc_q, "specter2+adapters")
            del e; torch.cuda.empty_cache()
        except Exception as ex:
            print('specter2 skipped:', ex)

    if "medcpt" in which:
        try:
            ed = HFEncoder("ncbi/MedCPT-Article-Encoder", normalize=False)
            eq = HFEncoder("ncbi/MedCPT-Query-Encoder", normalize=False)
            all_stats["medcpt"] = run_system("medcpt", doc_texts, doc_ids, claims, gold_val,
                                             lambda t: ed.encode(t, bs=16), lambda q: eq.encode(q, max_len=64), "MedCPT")
            del ed, eq; torch.cuda.empty_cache()
        except Exception as ex:
            print("medcpt skipped:", ex)
    finalize(gold_val, claims)


def finalize(gold_val, claims):
    """Pick the scientific model with best val nDCG@10, write the t2 / val .trec files and results.json."""
    cand = {n: json.load(open(os.path.join(LOGS, f"task2_{n}.json"))) for n in ("specter2", "medcpt")
            if os.path.exists(os.path.join(LOGS, f"task2_{n}.json"))}
    sci = max(cand, key=lambda n: cand[n]["nDCG@10"])
    json.dump({"name": sci, "all": {n: cand[n]["nDCG@10"] for n in cand}}, open(os.path.join(LOGS, "task2_scientific_choice.json"), "w"))
    g = json.load(open(os.path.join(LOGS, "task2_general.json")))
    write_trec(load_rank("eval", "general"), os.path.join(PRED, "t2_general.trec"), "bge_large")
    write_trec(load_rank("val", "general"), os.path.join(PRED, "val_general.trec"), "bge_large")
    write_trec(load_rank("eval", sci), os.path.join(PRED, "t2_scientific.trec"), sci)
    S = cand[sci]
    names = {"specter2": "allenai/specter2_base + specter2 (PRX docs) / adhoc_query (QRY queries) adapters",
             "medcpt": "ncbi/MedCPT Article+Query encoders"}
    update_results(RESULTS, "task2.general_model", "BAAI/bge-large-en-v1.5 (335M) with query instruction")
    update_results(RESULTS, "task2.scientific_model", names[sci])
    update_results(RESULTS, "task2.val_ndcg10_general", g["nDCG@10"])
    update_results(RESULTS, "task2.val_ndcg10_scientific", S["nDCG@10"])
    for key in ("index_time_sec", "query_latency_ms", "memory_mb"):
        update_results(RESULTS, f"task2.{key}", {"general": round(g[key], 2), "scientific": round(S[key], 2)})
    print("scientific choice", sci, {n: cand[n]["nDCG@10"] for n in cand})


if __name__ == "__main__":
    main()

"""Check that your NLP submission has the right files and formats BEFORE you submit.

Usage (run from this folder, so it can find data/):
    python check_format.py path/to/<rollno>_nlp_bootcamp.zip
    python check_format.py path/to/your_submission_folder

It only checks names and formats. It does NOT score your evaluation predictions.
We use the same checks when grading, so a file that fails here gets 0 marks.
"""
import csv
import json
import os
import shutil
import sys
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")

EVAL_RUNS = ["t1_bm25.trec", "t2_general.trec", "t2_scientific.trec", "t3_hybrid.trec",
             "t3_rerank.trec", "t4_random.trec", "t4_hard.trec"]
VAL_RUNS = ["val_bm25.trec", "val_general.trec"]
TASKS = {
    "task1": ["t1_bm25.trec"],
    "task2": ["t2_general.trec", "t2_scientific.trec"],
    "task3": ["t3_hybrid.trec", "t3_rerank.trec"],
    "task4": ["t4_random.trec", "t4_hard.trec"],
    "task5": ["t5_verification.csv", "t5_val_verification.csv", "t5_val_errors.csv"],
}
LABELS = {"SUPPORT", "CONTRADICT"}
CATEGORIES = {"retrieval_miss", "negation", "numerical", "entity_mismatch", "needs_multiple_docs", "other"}
TOP_K = 100


# ---------------------------------------------------------------- helpers
def locate_root(path):
    """Return (root_folder, temp_dir_or_None). root_folder contains predictions/."""
    tmp = None
    if os.path.isfile(path) and path.lower().endswith(".zip"):
        tmp = tempfile.mkdtemp()
        with zipfile.ZipFile(path) as z:
            z.extractall(tmp)
        path = tmp
    for dirpath, dirnames, _ in os.walk(path):
        if "__MACOSX" in dirpath:
            continue
        if "predictions" in dirnames:
            return dirpath, tmp
    return None, tmp


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = [[c.strip() for c in r] for r in csv.reader(f) if any(c.strip() for c in r)]
    if not rows:
        raise ValueError("file is empty")
    return [h.lower() for h in rows[0]], rows[1:]


def load_data_info(data_dir=DATA_DIR):
    return {
        "doc_ids": {int(d["doc_id"]) for d in read_jsonl(os.path.join(data_dir, "corpus.jsonl"))},
        "eval_ids": {int(c["id"]) for c in read_jsonl(os.path.join(data_dir, "eval_claims.jsonl"))},
        "val_claims": read_jsonl(os.path.join(data_dir, "val_claims.jsonl")),
    }


# ---------------------------------------------------------------- per-file checks
def check_trec(path, claim_ids, doc_ids):
    """Returns {claim_id: [doc_id ranked 1..100]}"""
    runs = {}
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            p = line.split()
            if len(p) != 6:
                raise ValueError(f"line {ln}: need 6 fields 'claim_id Q0 doc_id rank score run_name'")
            qid, did, rank = int(p[0]), int(p[2]), int(p[3])
            float(p[4])
            if qid not in claim_ids:
                raise ValueError(f"line {ln}: unknown claim_id {qid}")
            if did not in doc_ids:
                raise ValueError(f"line {ln}: unknown doc_id {did}")
            runs.setdefault(qid, {})
            if rank in runs[qid]:
                raise ValueError(f"line {ln}: claim {qid} has rank {rank} twice")
            runs[qid][rank] = did
    missing = claim_ids - set(runs)
    if missing:
        raise ValueError(f"{len(missing)} claims missing (e.g. {sorted(missing)[:3]})")
    out = {}
    for qid, r in runs.items():
        if sorted(r) != list(range(1, TOP_K + 1)):
            raise ValueError(f"claim {qid}: ranks must be exactly 1..{TOP_K}")
        docs = [r[k] for k in range(1, TOP_K + 1)]
        if len(set(docs)) != TOP_K:
            raise ValueError(f"claim {qid}: doc_ids must be unique")
        out[qid] = docs
    return out


def check_verification(path, claim_ids, doc_ids):
    """Returns set of (claim_id, doc_id, label)"""
    header, rows = read_csv(path)
    if header != ["claim_id", "doc_id", "label"]:
        raise ValueError("header must be 'claim_id,doc_id,label'")
    triples, per_claim, pairs = set(), {}, set()
    for i, r in enumerate(rows, 2):
        cid, did, lab = int(r[0]), int(r[1]), r[2].upper()
        if cid not in claim_ids:
            raise ValueError(f"row {i}: unknown claim_id {cid}")
        if did not in doc_ids:
            raise ValueError(f"row {i}: unknown doc_id {did}")
        if lab not in LABELS:
            raise ValueError(f"row {i}: label must be SUPPORT or CONTRADICT")
        if (cid, did) in pairs:
            raise ValueError(f"row {i}: (claim {cid}, doc {did}) appears twice")
        pairs.add((cid, did))
        per_claim[cid] = per_claim.get(cid, 0) + 1
        if per_claim[cid] > 3:
            raise ValueError(f"claim {cid}: more than 3 rows")
        triples.add((cid, did, lab))
    return triples


def check_errors(path, claim_ids):
    """Returns {claim_id: category}"""
    header, rows = read_csv(path)
    if header != ["claim_id", "category"]:
        raise ValueError("header must be 'claim_id,category'")
    out = {}
    for i, r in enumerate(rows, 2):
        cid, cat = int(r[0]), r[1].lower()
        if cid not in claim_ids:
            raise ValueError(f"row {i}: claim {cid} is not a validation claim")
        if cat not in CATEGORIES:
            raise ValueError(f"row {i}: category must be one of {sorted(CATEGORIES)}")
        if cid in out:
            raise ValueError(f"row {i}: claim {cid} listed twice")
        out[cid] = cat
    return out


# ---------------------------------------------------------------- whole submission
def check_submission(root, info):
    pred_dir = os.path.join(root, "predictions")
    present = set(os.listdir(pred_dir)) if os.path.isdir(pred_dir) else set()
    val_ids = {int(c["id"]) for c in info["val_claims"]}
    status, parsed, general = {}, {}, []

    def run(name, fn):
        if name not in present:
            status[name] = "MISSING"
            return
        try:
            parsed[name] = fn(os.path.join(pred_dir, name))
            status[name] = "OK"
        except Exception as e:  # noqa: BLE001
            status[name] = f"INVALID: {e}"

    for name in EVAL_RUNS:
        run(name, lambda p: check_trec(p, info["eval_ids"], info["doc_ids"]))
    for name in VAL_RUNS:
        run(name, lambda p: check_trec(p, val_ids, info["doc_ids"]))
    run("t5_verification.csv", lambda p: check_verification(p, info["eval_ids"], info["doc_ids"]))
    run("t5_val_verification.csv", lambda p: check_verification(p, val_ids, info["doc_ids"]))
    run("t5_val_errors.csv", lambda p: check_errors(p, val_ids))

    completion = {t: sum(status.get(f) == "OK" for f in fs) / len(fs) for t, fs in TASKS.items()}

    extra = sorted(present - set(status))
    if extra:
        general.append(f"Unexpected files in predictions/ (ignored): {', '.join(extra)}")

    results = None
    rj = os.path.join(root, "results.json")
    if not os.path.isfile(rj):
        general.append("results.json is MISSING")
    else:
        try:
            with open(rj, encoding="utf-8") as f:
                results = json.load(f)
            q1 = results.get("report_q1", {})
            for key in ["bm25_wins", "general_wins"]:
                ids = q1.get(key)
                if not (isinstance(ids, list) and len(ids) == 2 and all(isinstance(x, int) for x in ids)):
                    general.append(f"results.json report_q1.{key} must be a list of 2 validation claim ids")
                elif not set(ids) <= val_ids:
                    general.append(f"results.json report_q1.{key} contains ids that are not validation claims")
        except Exception as e:  # noqa: BLE001
            general.append(f"results.json is not valid JSON: {e}")
    if not os.path.isfile(os.path.join(root, "report.md")):
        general.append("report.md is MISSING")
    if not os.path.isfile(os.path.join(root, "code", "README.md")):
        general.append("code/README.md is MISSING (must say how to reproduce each task)")

    return {"status": status, "parsed": parsed, "completion": completion,
            "general": general, "results": results}


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    root, tmp = locate_root(sys.argv[1])
    try:
        if root is None:
            print("ERROR: could not find a 'predictions/' folder in your submission.")
            sys.exit(1)
        res = check_submission(root, load_data_info())
        print("\nFile check")
        print("-" * 60)
        for name in EVAL_RUNS + VAL_RUNS + ["t5_verification.csv", "t5_val_verification.csv", "t5_val_errors.csv"]:
            print(f"  {name:26s} {res['status'][name]}")
        print("\nTask completeness (1.0 = all files valid)")
        print("-" * 60)
        for t, v in res["completion"].items():
            print(f"  {t}: {v:.2f}")
        if res["general"]:
            print("\nOther issues")
            print("-" * 60)
            for m in res["general"]:
                print("  - " + m)
        ok = all(v == 1.0 for v in res["completion"].values()) and not res["general"] \
            and all(res["status"][f] == "OK" for f in VAL_RUNS)
        print("\n" + ("All good." if ok else "Fix the issues above (or submit what you have)."))
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
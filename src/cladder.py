"""Load CLadder v1 and join questions to their ground-truth causal models."""
from __future__ import annotations
import json, re, random
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "cladder"
DESC = re.compile(r"^(.+?)-(.+?)-(.+?)-model(\d+)-spec(\d+)-q(\d+)$")


def parse_structure(s: str) -> list[tuple[str, str]]:
    """'X->V2,X->Y' -> [('X','V2'), ('X','Y')]"""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split("->")
        out.append((a.strip(), b.strip()))
    return out


def load(questions="cladder-questions.json", models="cladder-meta.json") -> list[dict]:
    """Return question records with the model's DAG attached as 'edges'/'nodes'."""
    qs = json.loads((DATA / questions).read_text(encoding="utf-8"))
    ms = json.loads((DATA / models).read_text(encoding="utf-8"))
    by_id = {m["model_id"]: m for m in ms}

    joined = []
    for q in qs:
        mm = DESC.match(q["desc_id"])
        if not mm:
            continue
        model = by_id.get(int(mm.group(4)))
        if model is None:
            continue
        edges = parse_structure(model["structure"])
        nodes = sorted({n for e in edges for n in e})
        meta = q["meta"]
        joined.append({
            "qid": q["question_id"],
            "desc_id": q["desc_id"],
            "given_info": q["given_info"],
            "question": q["question"],
            "answer": q["answer"],
            "graph_id": meta["graph_id"],
            "query_type": meta["query_type"],
            "rung": meta["rung"],
            "story_id": meta["story_id"],
            "model_id": model["model_id"],
            "structure": model["structure"],
            "edges": edges,
            "nodes": nodes,
            "background": model["background"],
            "variable_mapping": model["variable_mapping"],
        })
    return joined


def stratified_sample(records, n, seed=20260907, by=("graph_id", "rung")):
    """Proportional stratified sample, deterministic. Returns (sample, report)."""
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in records:
        strata[tuple(r[k] for k in by)].append(r)

    total = len(records)
    # largest-remainder allocation so the quotas sum to exactly n
    raw = {k: len(v) * n / total for k, v in strata.items()}
    alloc = {k: int(v) for k, v in raw.items()}
    short = n - sum(alloc.values())
    for k, _ in sorted(raw.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True)[:short]:
        alloc[k] += 1

    out, report = [], []
    for k, want in sorted(alloc.items(), key=lambda kv: str(kv[0])):
        pool = strata[k]
        take = min(want, len(pool))
        out.extend(rng.sample(pool, take))
        report.append({"stratum": k, "pool": len(pool), "want": want, "got": take})
    rng.shuffle(out)
    return out, report


def family_table(records):
    """One row per graph family: counts, node/edge sizes."""
    fam = defaultdict(list)
    for r in records:
        fam[r["graph_id"]].append(r)
    rows = []
    for g, rs in sorted(fam.items()):
        rows.append({
            "graph_id": g,
            "questions": len(rs),
            "models": len({r["model_id"] for r in rs}),
            "nodes": len(rs[0]["nodes"]),
            "edges": len(rs[0]["edges"]),
            "structure": rs[0]["structure"],
            "rungs": dict(sorted(Counter(r["rung"] for r in rs).items())),
        })
    return rows

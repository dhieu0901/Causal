"""Does the model actually USE the causal graph it is handed?

    python scripts/analyze_chains.py
    python scripts/analyze_chains.py --dump 50    # write chains out for hand coding

Every accuracy result in this project is behavioural: the model scores better
with a correct graph and worse with a corrupted one, so something about the
graph reaches the answer. None of it shows the graph reaching the REASONING.
Review round 6 listed this as the one piece of direct mechanism evidence the
project has never collected, and noted the data has been sitting in cache/ all
along, already paid for.

The measurement exploits the fact that DR_k1 reverses exactly one edge. For each
item the true graph contains u -> v, and the DR_k1 prompt instead asserts v -> u.
That gives a clean three-way observable on the same item:

  follows supplied   the chain asserts v -> u, the direction it was handed.
                     The graph reached the reasoning, even when wrong.
  keeps true         the chain asserts u -> v, contradicting the block it was
                     given. The model overrode the prompt with its own prior or
                     with the prose it had already read.
  silent             the chain never commits to a direction for that edge.

Under ORACLE the supplied and true directions coincide, so ORACLE cannot
separate the first two. It is still reported, as the base rate for how often a
chain commits to a direction at all.

Two honest limits, both of which cap what this can claim:

  1. Direction is read off the surface text by pattern, not by understanding it.
     A chain that reasons over the reversed edge without naming it reads as
     `silent`, so `silent` is an upper bound on indifference, not a measure of
     it. Ambiguity is reported as its own category rather than forced into a
     call, and --dump writes the chains out so the automatic coding can be
     checked by hand against a real sample.
  2. This is a correlational reading of text the model produced for another
     purpose. A chain is not a faithful trace of the computation, and a model
     can state one direction while computing with another.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from perturb import ENUMERATORS
from pilot import make_items
from prompts import build, strip_structure, parse_prose_graph

CACHE = ROOT / "cache"
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]

# Vocabulary that only appears when a chain is talking about structure rather
# than only about arithmetic. Deliberately narrow: "effect" alone is excluded
# because CLadder's own question wording uses it.
STRUCT = re.compile(
    r"\b(direct effect|causal (structure|graph|diagram|path|chain)|confound\w*|"
    r"backdoor|back-door|adjust(ment|ing|ed)?\b|mediat\w*|collider|"
    r"d-separat\w*|downstream|upstream|parent node|causal direction)\b",
    re.IGNORECASE)

# A -> B asserted in prose. The arrow forms are listed first so they win when a
# chain draws the graph out.
LINK = (r"(?:\s*(?:-+>|=+>|→)\s*|\s+(?:has a direct effect on|directly affects?|"
        r"affects?|causes?|influences?|leads? to|determines?|drives?)\s+)")


def cache_text(model, prompt, temp=0.0):
    h = hashlib.sha256(f"{model}|{temp}|{prompt}".encode()).hexdigest()[:24]
    f = CACHE / f"{h}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("text", "")
    except json.JSONDecodeError:
        return None


def asserts(text, a, b):
    """Does the text assert a -> b anywhere?"""
    if not text:
        return False
    return bool(re.search(re.escape(a) + LINK + re.escape(b), text, re.IGNORECASE))


def code_direction(text, u, v):
    """u -> v is true; v -> u is what DR_k1 supplied."""
    t = asserts(text, u, v)
    s = asserts(text, v, u)
    if t and s:
        return "both"            # chain states both; cannot be scored
    if s:
        return "followed_graph"       # followed the supplied (reversed) edge
    if t:
        return "kept_true_direction"    # overrode the supplied edge
    return "silent"


def reversed_edge(edges, item_idx, seed):
    """Recover the edge DR_k1 flipped, using build_jobs' own per-item seed."""
    nodes = sorted({n for e in edges for n in e})
    opts = ENUMERATORS["DR"](edges, nodes, 1)
    if not opts:
        return None, None
    bad = random.Random(f"{seed}:{item_idx}:DR:1").choice(opts)
    for u, v in edges:
        if (v, u) in bad and (u, v) not in bad:
            return (u, v), bad
    return None, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--dump", type=int, default=0,
                    help="write N chains to a file for hand coding")
    a = ap.parse_args()
    W = 88

    items = make_items(a.n, a.seed, 1, "full_v1.5_default.csv", drop_nonsense=True)
    cases = []
    for i, r in items.iterrows():
        prompt = r.prompt
        _, removed = strip_structure(prompt)
        edges = parse_prose_graph(removed)
        if not edges:
            continue
        flipped, bad = reversed_edge(edges, i, a.seed)
        if flipped is None:
            continue
        cases.append({"item": i, "u": flipped[0], "v": flipped[1],
                      "oracle": build(prompt, "ORACLE", edges),
                      "dr": build(prompt, "PERTURB", bad)})

    print("=" * W)
    print("1. DO THE REASONING CHAINS USE THE SUPPLIED GRAPH")
    print("=" * W)
    print(f"  {len(cases)} items have exactly one reversed edge under DR_k1.")
    print("  The true graph says u -> v; DR_k1 supplies v -> u. See which direction\n"
          "  the chain states.\n")

    rows, miss, dump = [], 0, []
    for m in TIER:
        rec = {"model": m}
        for cond in ("oracle", "dr"):
            c = {"followed_graph": 0, "kept_true_direction": 0, "silent": 0,
                 "both": 0, "struct": 0, "n": 0}
            for k in cases:
                t = cache_text(m, k[cond])
                if t is None:
                    miss += 1
                    continue
                c["n"] += 1
                c["struct"] += bool(STRUCT.search(t))
                c[code_direction(t, k["u"], k["v"])] += 1
                if a.dump and cond == "dr" and len(dump) < a.dump:
                    dump.append({"model": m, "item": k["item"],
                                 "true_edge": f"{k['u']} -> {k['v']}",
                                 "given_edge": f"{k['v']} -> {k['u']}",
                                 "ma_tu_dong": code_direction(t, k["u"], k["v"]),
                                 "chuoi": t})
            n = max(c["n"], 1)
            tag = "ORACLE" if cond == "oracle" else "DR_k1"
            rec[f"{tag}_n"] = c["n"]
            rec[f"{tag}_ngon_ngu_cau_truc"] = round(100 * c["struct"] / n, 1)
            rec[f"{tag}_theo_do_thi"] = round(100 * c["followed_graph"] / n, 1)
            rec[f"{tag}_giu_chieu_that"] = round(100 * c["kept_true_direction"] / n, 1)
            rec[f"{tag}_khong_noi"] = round(100 * c["silent"] / n, 1)
            rec[f"{tag}_ca_hai"] = round(100 * c["both"] / n, 1)
        rows.append(rec)
    d = pd.DataFrame(rows)

    print("  -- ty le chuoi dung NGON NGU CAU TRUC (%) --")
    print(d[["model", "ORACLE_ngon_ngu_cau_truc", "DR_k1_ngon_ngu_cau_truc",
             "ORACLE_n"]].to_string(index=False))
    print("\n  -- under DR_k1: which direction the chain states for the reversed edge (%) --")
    print(d[["model", "DR_k1_theo_do_thi", "DR_k1_giu_chieu_that",
             "DR_k1_khong_noi", "DR_k1_ca_hai"]].to_string(index=False))
    # Under ORACLE the supplied edge IS the true edge, so the column that means
    # "restated what it was given" is giu_chieu_that, and theo_do_thi means the
    # chain asserted a direction nothing in the prompt supports.
    print("\n  -- under ORACLE: the edge is not reversed, so the right column is 'restates the true direction' --")
    print(d[["model", "ORACLE_giu_chieu_that", "ORACLE_khong_noi",
             "ORACLE_theo_do_thi"]]
          .rename(columns={"ORACLE_giu_chieu_that": "restated_given_direction",
                           "ORACLE_theo_do_thi": "noi_chieu_NGUOC_khong_ai_cap"})
          .to_string(index=False))

    out = ROOT / "results" / "chain_graph_use.csv"
    d.to_csv(out, index=False)
    if miss:
        print(f"\n  ({miss} calls not in the cache, skipped)")

    fol = d.DR_k1_theo_do_thi.mean()
    kep = d.DR_k1_giu_chieu_that.mean()
    sil = d.DR_k1_khong_noi.mean()
    print("\n" + "=" * W)
    print("2. DOC KET QUA")
    print("=" * W)
    print(f"  Trung binh ba model, duoi DR_k1: theo do thi {fol:.1f}%, "
          f"giu chieu that {kep:.1f}%, khong noi {sil:.1f}%.")
    if fol > kep:
        print("  The chains follow the SUPPLIED graph more often than the true direction.")
        print("  The graph reaches the reasoning layer, not just the answer layer - which")
        print("  is the evidence")
        print("  co che truc tiep dau tien cua du an.")
    else:
        print("  The chains keep the true direction more often than they follow the graph.")
        print("  The model overrides the structure block with its own prior, so the")
        print("  performance gap at DR_k1 CANNOT be explained by it obeying a wrong graph.")
    print(f"\n  WARNING: {sil:.1f}% of chains state no direction at all. The 'silent'")
    print("  figure is an UPPER BOUND on indifference, not a measurement of it - a chain")
    print("  can reason over that edge without ever naming it.")

    if a.dump:
        f = ROOT / "results" / "chain_sample_for_hand_coding.md"
        with f.open("w", encoding="utf-8") as fh:
            fh.write("# Sample reasoning chains under DR_k1, for hand coding\n\n")
            fh.write("For each chain: the true graph says `true_edge`, but the prompt "
                     "supplied `given_edge`.\nRead the chain, code it by hand, then "
                     "compare against `auto_code` to check the automatic coder.\n\n")
            for k, c in enumerate(dump, 1):
                fh.write(f"---\n\n## {k}. {c['model']} - item {c['item']}\n\n")
                fh.write(f"- that: `{c['canh_that']}`\n")
                fh.write(f"- prompt cap: `{c['canh_cap']}`\n")
                fh.write(f"- ma tu dong: **{c['ma_tu_dong']}**\n")
                fh.write(f"- ma cua ban: ____________\n\n```\n{c['chuoi']}\n```\n\n")
        print(f"\n  Da ghi {len(dump)} chuoi ra {f.name} de cham tay.")
    print(f"  Da ghi: {out.name}")


if __name__ == "__main__":
    main()

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
from prompts import build, parse_answer, strip_structure, parse_prose_graph

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
                      "gold": str(r.label).strip().lower(),
                      "raw": build(prompt, "RAW"),
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

    out = ROOT / "results" / "cladder" / "chain_graph_use.csv"
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

    # ------------------------------------------------------------------
    # 3. The two caveats REPORT section 9b rests on. Both were computed once,
    # quoted, and never written down: the RAW floor that the 1.9% has to be
    # read against, and whether a chain that states the WRONG direction also
    # gives a different answer. Review item V7-18 asked for the floor column;
    # the prose was toned down but the column never arrived. Correctness is
    # scored here from the chain text and the item's gold label directly, so
    # nothing depends on item ids lining up with another file.
    # ------------------------------------------------------------------
    print("\n" + "=" * W)
    print("3. THE FLOOR, AND WHETHER A STATED DIRECTION REACHES THE ANSWER")
    print("=" * W)

    def first_pos(text, a, b):
        m = re.search(re.escape(a) + LINK + re.escape(b), text or "", re.IGNORECASE)
        return m.start() / max(len(text), 1) if m else None

    det, all_pos = [], []
    for m in TIER:
        raw_true = raw_rev = raw_n = 0
        fol_ok, sil_ok, differ, both_parsed, pos = [], [], 0, 0, []
        sil_both = sil_differ = 0
        for k in cases:
            tr = cache_text(m, k["raw"])
            if tr is not None:
                raw_n += 1
                rc = code_direction(tr, k["u"], k["v"])
                raw_true += rc == "kept_true_direction"
                # The matching floor for the 45% "followed the supplied edge":
                # how often v -> u is stated when nothing supplied it.
                raw_rev += rc == "followed_graph"
            td, to = cache_text(m, k["dr"]), cache_text(m, k["oracle"])
            if td is None:
                continue
            code = code_direction(td, k["u"], k["v"])
            ad = parse_answer(td)
            # Accuracy on parsed answers only, the convention every other
            # accuracy in this project uses.
            if code == "followed_graph":
                if ad is not None:
                    fol_ok.append(ad == k["gold"])
                p = first_pos(td, k["v"], k["u"])
                if p is not None:
                    pos.append(p)
                ao = parse_answer(to) if to is not None else None
                if ad is not None and ao is not None:
                    both_parsed += 1
                    differ += ad != ao
            elif code == "silent" and ad is not None:
                sil_ok.append(ad == k["gold"])
                # Baseline for "the stated direction changes the answer":
                # how often DR_k1 and ORACLE disagree when the chain names no
                # direction at all (review round 10, finding M5).
                ao = parse_answer(to) if to is not None else None
                if ao is not None:
                    sil_both += 1
                    sil_differ += ad != ao
        s_pos = pd.Series(pos, dtype=float)
        all_pos.extend(pos)
        det.append({
            "model": m,
            "RAW_n": raw_n,
            "RAW_states_true_direction_pct": round(100 * raw_true / max(raw_n, 1), 1),
            "RAW_states_reversed_direction_pct": round(100 * raw_rev / max(raw_n, 1), 1),
            "DR_k1_followed_n": len(fol_ok),
            "acc_when_followed_pct": round(100 * sum(fol_ok) / max(len(fol_ok), 1), 1),
            "DR_k1_silent_n": len(sil_ok),
            "acc_when_silent_pct": round(100 * sum(sil_ok) / max(len(sil_ok), 1), 1),
            "followed_both_parsed_n": both_parsed,
            "answer_differs_from_ORACLE_pct": round(100 * differ / max(both_parsed, 1), 1),
            "answer_same_as_ORACLE_pct": round(100 * (both_parsed - differ) / max(both_parsed, 1), 1),
            "silent_differs_from_ORACLE_pct": round(100 * sil_differ / max(sil_both, 1), 1),
            "median_position_pct": round(100 * s_pos.median(), 1) if len(s_pos) else None,
            "in_first_10pct_share": round(100 * (s_pos <= 0.10).mean(), 1) if len(s_pos) else None,
        })
    # Where in the chain the wrong direction is stated, pooled over the three
    # models: early means it is said while setting up, and rarely revisited.
    ap_ = pd.Series(all_pos, dtype=float)
    det.append({"model": "pooled", "DR_k1_followed_n": len(ap_),
                "median_position_pct": round(100 * ap_.median(), 1) if len(ap_) else None,
                "in_first_10pct_share": round(100 * (ap_ <= 0.10).mean(), 1) if len(ap_) else None})
    dd = pd.DataFrame(det)
    for c in [c for c in dd.columns if c.endswith("_n")]:
        dd[c] = dd[c].astype("Int64")      # the pooled row leaves NaN in counts
    print(dd.to_string(index=False))
    per = dd[dd.model != "pooled"]
    fl = per.RAW_states_true_direction_pct
    print(f"\n  RAW floor (states u -> v with no block at all): "
          f"{fl.min():.1f}-{fl.max():.1f}%, mean {fl.mean():.1f}%.")
    print(f"  DR_k1 'keeps the true direction' averages {kep:.1f}% - read it against that floor.")
    same = per.answer_same_as_ORACLE_pct
    print(f"  Chains that state the WRONG direction still give ORACLE's answer "
          f"{same.min():.1f}-{same.max():.1f}% of the time.")
    out2 = ROOT / "results" / "cladder" / "chain_graph_use_detail.csv"
    dd.to_csv(out2, index=False)
    print(f"  Da ghi: {out2.name}")

    if a.dump:
        f = ROOT / "results" / "cladder" / "chain_sample_for_hand_coding.md"
        with f.open("w", encoding="utf-8") as fh:
            fh.write("# Sample reasoning chains under DR_k1, for hand coding\n\n")
            fh.write("For each chain: the true graph says `true_edge`, but the prompt "
                     "supplied `given_edge`.\nRead the chain, code it by hand, then "
                     "compare against `auto_code` to check the automatic coder.\n\n")
            for k, c in enumerate(dump, 1):
                fh.write(f"---\n\n## {k}. {c['model']} - item {c['item']}\n\n")
                # These two keys were renamed to English in the dict above and
                # not here, so every --dump run died with a KeyError.
                fh.write(f"- that: `{c['true_edge']}`\n")
                fh.write(f"- prompt cap: `{c['given_edge']}`\n")
                fh.write(f"- ma tu dong: **{c['ma_tu_dong']}**\n")
                fh.write(f"- ma cua ban: ____________\n\n```\n{c['chuoi']}\n```\n\n")
        print(f"\n  Da ghi {len(dump)} chuoi ra {f.name} de cham tay.")
    print(f"  Da ghi: {out.name}")


if __name__ == "__main__":
    main()

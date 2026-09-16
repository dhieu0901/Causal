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
        return "ca_hai"            # chain states both; cannot be scored
    if s:
        return "theo_do_thi"       # followed the supplied (reversed) edge
    if t:
        return "giu_chieu_that"    # overrode the supplied edge
    return "khong_noi"


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
                    help="ghi N chuoi ra file de cham tay")
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
    print("1. CHUOI SUY LUAN CO DUNG DO THI DUOC CAP KHONG")
    print("=" * W)
    print(f"  {len(cases)} item co dung mot canh bi dao duoi DR_k1.")
    print("  Do thi that noi u -> v; DR_k1 dua vao v -> u. Xem chuoi noi chieu nao.\n")

    rows, miss, dump = [], 0, []
    for m in TIER:
        rec = {"model": m}
        for cond in ("oracle", "dr"):
            c = {"theo_do_thi": 0, "giu_chieu_that": 0, "khong_noi": 0,
                 "ca_hai": 0, "struct": 0, "n": 0}
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
                                 "canh_that": f"{k['u']} -> {k['v']}",
                                 "canh_cap": f"{k['v']} -> {k['u']}",
                                 "ma_tu_dong": code_direction(t, k["u"], k["v"]),
                                 "chuoi": t})
            n = max(c["n"], 1)
            tag = "ORACLE" if cond == "oracle" else "DR_k1"
            rec[f"{tag}_n"] = c["n"]
            rec[f"{tag}_ngon_ngu_cau_truc"] = round(100 * c["struct"] / n, 1)
            rec[f"{tag}_theo_do_thi"] = round(100 * c["theo_do_thi"] / n, 1)
            rec[f"{tag}_giu_chieu_that"] = round(100 * c["giu_chieu_that"] / n, 1)
            rec[f"{tag}_khong_noi"] = round(100 * c["khong_noi"] / n, 1)
            rec[f"{tag}_ca_hai"] = round(100 * c["ca_hai"] / n, 1)
        rows.append(rec)
    d = pd.DataFrame(rows)

    print("  -- ty le chuoi dung NGON NGU CAU TRUC (%) --")
    print(d[["model", "ORACLE_ngon_ngu_cau_truc", "DR_k1_ngon_ngu_cau_truc",
             "ORACLE_n"]].to_string(index=False))
    print("\n  -- duoi DR_k1: chuoi noi chieu nao cho canh bi dao (%) --")
    print(d[["model", "DR_k1_theo_do_thi", "DR_k1_giu_chieu_that",
             "DR_k1_khong_noi", "DR_k1_ca_hai"]].to_string(index=False))
    # Under ORACLE the supplied edge IS the true edge, so the column that means
    # "restated what it was given" is giu_chieu_that, and theo_do_thi means the
    # chain asserted a direction nothing in the prompt supports.
    print("\n  -- duoi ORACLE: canh khong bi dao, nen cot dung la 'noi lai chieu that' --")
    print(d[["model", "ORACLE_giu_chieu_that", "ORACLE_khong_noi",
             "ORACLE_theo_do_thi"]]
          .rename(columns={"ORACLE_giu_chieu_that": "noi_lai_chieu_duoc_cap",
                           "ORACLE_theo_do_thi": "noi_chieu_NGUOC_khong_ai_cap"})
          .to_string(index=False))

    out = ROOT / "results" / "chain_graph_use.csv"
    d.to_csv(out, index=False)
    if miss:
        print(f"\n  ({miss} luot khong co trong cache, da bo qua)")

    fol = d.DR_k1_theo_do_thi.mean()
    kep = d.DR_k1_giu_chieu_that.mean()
    sil = d.DR_k1_khong_noi.mean()
    print("\n" + "=" * W)
    print("2. DOC KET QUA")
    print("=" * W)
    print(f"  Trung binh ba model, duoi DR_k1: theo do thi {fol:.1f}%, "
          f"giu chieu that {kep:.1f}%, khong noi {sil:.1f}%.")
    if fol > kep:
        print("  Chuoi theo do thi duoc cap NHIEU HON theo chieu dung. Do thi vao")
        print("  toi tang suy luan, khong chi tang cau tra loi - day la bang chung")
        print("  co che truc tiep dau tien cua du an.")
    else:
        print("  Chuoi giu chieu that nhieu hon theo do thi duoc cap. Model ghi de")
        print("  khoi cau truc bang prior rieng, va lo hong hieu suat o DR_k1 KHONG")
        print("  the giai thich bang viec no lam theo do thi sai.")
    print(f"\n  CANH BAO: {sil:.1f}% chuoi khong noi chieu nao ca. Con so 'khong noi'")
    print("  la CAN TREN cua su tho o, khong phai phep do su tho o - mot chuoi co")
    print("  the suy luan tren canh do ma khong goi ten no.")

    if a.dump:
        f = ROOT / "results" / "chain_sample_for_hand_coding.md"
        with f.open("w", encoding="utf-8") as fh:
            fh.write("# Mau chuoi suy luan duoi DR_k1, de cham tay\n\n")
            fh.write("Voi moi chuoi: do thi that noi `canh_that`, nhung prompt dua "
                     "vao `canh_cap`.\nDoc chuoi roi tu danh ma, sau do so voi "
                     "`ma_tu_dong` de kiem bo ma may.\n\n")
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

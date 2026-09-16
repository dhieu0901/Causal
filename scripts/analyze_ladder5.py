"""The lexical ladder with the rung that was missing, so rung 1 changes one thing.

    python scripts/analyze_ladder5.py

REPORT.md section 7 reads the ladder as though each rung removes exactly one
thing, and rung 1 does not. Going KEEP -> PERMUTE changes THREE things at once:

    1. the correct prior is removed
    2. a WRONG prior is put in its place
    3. the item becomes strange enough that models say so out loud
       (measured in analyze_anomaly_residue.py: PERMUTE flags contradiction
       language several times more often than any other lexicon)

Because of that, the -7.53 pp charged to rung 1 cannot be attributed to any one
of the three, and the "positive control" argument built on it was retracted in
review round 6.

IRRELEVANT is the missing rung. Real English nouns from a domain with no causal
story of its own - lamp, spoon, curtain, kettle, ladder, basket. It removes the
correct prior WITHOUT supplying a wrong one, and it keeps the words real. That
turns one uninterpretable step into three interpretable ones:

    KEEP -> IRRELEVANT    losing a correct prior, word realness held fixed
    IRRELEVANT -> PERMUTE being handed a WRONG prior, on top of having none
    IRRELEVANT -> SYMBOL  real words versus bare letters, prior absent in both

The second contrast is the one the project's mechanism claim needs. Section 9
already shows wrong priors are more toxic than absent priors for graph
EXTRACTION (5.1-7.7x reversals versus 1.3-2.8x). If IRRELEVANT -> PERMUTE is
also negative for REASONING accuracy, the same asymmetry holds on both tasks and
the mechanism claim carries across them. If it is flat, the extraction result
stands alone and must be stated as being about extraction only.

Bootstrap resamples ITEMS. Equivalence bounds are reported for every contrast,
because several of these are expected to be null and a null needs a bound rather
than a failure to reject.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
LADDER = ["KEEP", "PERMUTE", "IRRELEVANT", "SYMBOL", "PSEUDO"]
ARITH = {"marginal", "correlation"}
IDENT = {"backadj"}

# The three steps IRRELEVANT makes interpretable, plus the two the old ladder
# already had, so the whole thing can be read in one table.
STEPS = [
    ("KEEP -> IRRELEVANT", "KEEP", "IRRELEVANT", "mat prior DUNG, tu van la that"),
    ("IRRELEVANT -> PERMUTE", "IRRELEVANT", "PERMUTE", "bi gan prior SAI"),
    ("IRRELEVANT -> SYMBOL", "IRRELEVANT", "SYMBOL", "tu that -> ky hieu"),
    ("SYMBOL -> PSEUDO", "SYMBOL", "PSEUDO", "ky hieu -> tu gia"),
    ("KEEP -> PERMUTE", "KEEP", "PERMUTE", "bac 1 cu, doi BA thu cung luc"),
]


def load(lex):
    for tag in (f"_lex{lex}", f"_instr{lex}"):
        p = ROOT / "results" / f"pilot_raw{tag}.csv"
        if p.exists():
            return pd.read_csv(p)
    raise SystemExit(
        f"thieu du lieu cho {lex}. Chay:\n"
        f"  python scripts/pilot.py --n 200 --models gpt-4.1-nano,gpt-4.1-mini,"
        f"gpt-4.1 --kmax 1 --types DR --drop-nonsense --lexicon {lex} "
        f"--tag _lex{lex}")


def cell(d, model, cond="RAW", causal_only=True):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)]
    if causal_only:
        s = s[~s.query_type.isin(ARITH | IDENT)]
    return s.set_index("item").correct


def paired(a, b, model, cond, causal_only):
    x, y = cell(a, model, cond, causal_only), cell(b, model, cond, causal_only)
    idx = x.index.intersection(y.index)
    return pd.Series(x[idx].values - y[idx].values, index=idx)


def boot(by_model, seed, n=4000):
    items = sorted(set().union(*[set(s.index) for s in by_model.values()]))
    rng = np.random.default_rng(seed)
    M = np.vstack([by_model[m].reindex(items).values for m in by_model])
    out = np.empty(n)
    for b in range(n):
        pick = rng.integers(0, len(items), len(items))
        out[b] = np.nanmean(M[:, pick])
    return 100 * np.nanmean(M), 100 * out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--cond", default="RAW",
                    help="dieu kien cau truc de so tren do (mac dinh RAW)")
    ap.add_argument("--all-queries", action="store_true",
                    help="dung ca mau thay vi chi nhom cau hoi nhan qua")
    a = ap.parse_args()
    causal = not a.all_queries
    D = {lex: load(lex) for lex in LADDER}
    W = 96

    print("=" * W)
    print("1. THANG NAM BAC - do chinh xac theo tung bo tu vung")
    print("=" * W)
    scope = "chi nhom cau hoi nhan qua that" if causal else "ca mau"
    print(f"  Dieu kien {a.cond}, {scope}.\n")
    rows = []
    for m in TIER:
        r = {"model": m}
        for lex in LADDER:
            c = cell(D[lex], m, a.cond, causal)
            r[lex] = round(100 * c.mean(), 2)
            r[f"n_{lex}"] = len(c)
        rows.append(r)
    acc = pd.DataFrame(rows)
    print(acc[["model"] + LADDER].to_string(index=False))
    acc.to_csv(ROOT / "results" / "ladder5_accuracy.csv", index=False)

    print("\n" + "=" * W)
    print("2. TUNG BAC DOI DUNG MOT THU - day la diem cua IRRELEVANT")
    print("=" * W)
    print(f"  Ghep cap theo item, bootstrap boc lai theo ITEM {a.boot} lan.\n")
    rows = []
    for label, hi, lo, what in STEPS:
        by = {m: paired(D[hi], D[lo], m, a.cond, causal) for m in TIER}
        by = {m: s for m, s in by.items() if len(s)}
        if not by:
            continue
        est, bs = boot(by, a.seed, a.boot)
        clo, chi = np.percentile(bs, [2.5, 97.5])
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        rows.append({"buoc": label, "doi_gi": what, "chenh_pp": round(est, 2),
                     "ci_lo": round(clo, 2), "ci_hi": round(chi, 2),
                     "p": round(p, 4),
                     "xac_lap": "co" if clo > 0 or chi < 0 else "khong",
                     "can_tuong_duong": round(max(abs(clo), abs(chi)), 2)})
    st = pd.DataFrame(rows)
    print(st.to_string(index=False))
    st.to_csv(ROOT / "results" / "ladder5_steps.csv", index=False)
    print("\n  'can_tuong_duong' la cai phai doc khi mot buoc KHONG xac lap: no noi")
    print("  hieu ung that co the lon toi dau, thay vi chi noi 'khong bac bo duoc'.")

    print("\n" + "=" * W)
    print("3. DOC KET QUA")
    print("=" * W)
    g = {r["buoc"]: r for r in rows}
    k_i = g.get("KEEP -> IRRELEVANT")
    i_p = g.get("IRRELEVANT -> PERMUTE")
    i_s = g.get("IRRELEVANT -> SYMBOL")
    k_p = g.get("KEEP -> PERMUTE")
    if k_i and i_p and k_p:
        print(f"  Bac 1 cu (KEEP -> PERMUTE) la {k_p['chenh_pp']:+.2f} pp, va no tach ra:")
        print(f"    mat prior DUNG      {k_i['chenh_pp']:+.2f} pp  "
              f"CI [{k_i['ci_lo']:+.2f}; {k_i['ci_hi']:+.2f}]  {k_i['xac_lap']}")
        print(f"    bi gan prior SAI    {i_p['chenh_pp']:+.2f} pp  "
              f"CI [{i_p['ci_lo']:+.2f}; {i_p['ci_hi']:+.2f}]  {i_p['xac_lap']}")
        if i_p["xac_lap"] == "co" and i_p["chenh_pp"] > 0:
            print("\n  Bi gan prior SAI co chi phi RIENG, ngoai viec mat prior dung.")
            print("  Cung huong voi muc 9 (prior sai doc hon prior vang mat khi TRICH")
            print("  XUAT do thi), nen bat doi xung nay dung tren CA HAI nhiem vu.")
        elif i_p["xac_lap"] == "khong":
            print(f"\n  Bi gan prior SAI KHONG co chi phi rieng do duoc (can tuong duong "
                  f"{i_p['can_tuong_duong']:.2f} pp).")
            print("  Nghia la o nhiem vu SUY LUAN, gan prior sai khong te hon chi bo")
            print("  prior dung. Bat doi xung o muc 9 chi ap dung cho viec TRICH XUAT")
            print("  do thi, va phai phat bieu gioi han o do.")
    if i_s:
        tag = "co" if i_s["xac_lap"] == "co" else "khong"
        print(f"\n  Tu that so voi ky hieu: {i_s['chenh_pp']:+.2f} pp, xac lap: {tag}.")
        if i_s["xac_lap"] == "khong":
            print(f"  Can tuong duong {i_s['can_tuong_duong']:.2f} pp. Khi prior da vang")
            print("  mat o ca hai ben, viec tu co that hay khong KHONG con quan trong -")
            print("  ung ho lap luan rang cai mat di la TRI THUC, khong phai do quen mat.")
    print("\n  Da ghi: results/ladder5_accuracy.csv, results/ladder5_steps.csv")


if __name__ == "__main__":
    main()

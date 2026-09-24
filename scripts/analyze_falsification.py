"""The four competing explanations REPORT section 4.7 rules out.

    python scripts/analyze_falsification.py

That section is the load-bearing half of the argument: it is where the headline
stops being "an effect appeared" and becomes "an effect appeared and here is
what it is not". Until 2026-09-23 none of its four tests had a script, so the
one part of the report designed to survive an attack was the part nobody could
re-run.

  prompt length     A longer prompt could depress accuracy on its own, and the
                    anonymised branches differ in length. Correlate correctness
                    with in_tok inside each (model, condition, lexicon) cell.
                    Pooling the cells would confound the within-cell effect
                    with the between-cell one, which is the thing being tested.

  residue           relabel() leaves some real words behind. If the effect came
                    from residue, items whose anonymisation was CLEAN should
                    show LESS of it. Split and compare.

  a lucky slice     The causal subset is 86 of 174 items and was chosen after
                    seeing the pooled result fail. Draw 86 items at random 2000
                    times and see where the real number falls. This is the only
                    one of the four that can put a number on the post-hoc
                    filtering worry.

  scoring           Unparsed answers are dropped. Score them WRONG instead and
                    recompute; if the effect depends on that choice it is not
                    an effect.

Every one is a test that could fail. The point of writing them down is that
they did not, and a reader can check that for themselves.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_querygroup import ARITH, IDENT, TIER, did_cells, load
from stats import boot_cell_means, boot_interval, boot_p, cluster_boot

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SEED = 20260907
NBOOT = 4000
NPERM = 2000
ANON = ["PERMUTE", "SYMBOL", "PSEUDO"]


def causal_qs(d):
    return set(d["KEEP"].query_type.unique()) - ARITH - IDENT


def boot_ci(W, seed=SEED, n=NBOOT):
    est, lo, hi, p = boot_cell_means(W, seed, n)    # convention C, src/stats.py
    return est, lo, hi, p, len(W.index.unique())


def test_prompt_length(d, rows):
    """Within each cell, does a longer prompt predict a wrong answer?"""
    print("\n1. DO DAI PROMPT")
    print("   Tuong quan correct voi in_tok, TRONG tung o (model x cond x lexicon).")
    rs, n_neg, n_pos = [], 0, 0
    for lex in ["KEEP"] + ANON:
        for m in TIER:
            for c in ["RAW", "ORACLE"]:
                s = d[lex]
                s = s[(s.model == m) & (s.cond == c) & (s.parsed == 1)]
                if len(s) < 30 or s.correct.nunique() < 2:
                    continue
                r = float(np.corrcoef(s.in_tok, s.correct)[0, 1])
                if not np.isfinite(r):
                    continue
                rs.append(r)
                # A crude two-sided test on a correlation, n large enough here.
                z = r * np.sqrt(len(s) - 1)
                if r < 0 and abs(z) > 1.96:
                    n_neg += 1
                elif r > 0 and abs(z) > 1.96:
                    n_pos += 1
    if not rs:
        print("   khong du du lieu")
        return
    print(f"   {len(rs)} o.  r trung binh {np.mean(rs):+.3f}.  "
          f"{n_neg} o am co y nghia, {n_pos} o duong co y nghia.")
    print("   Am nghia la prompt DAI HON thi SAI NHIEU HON - cung chieu voi gia")
    print("   thuyet do dai. Nhung no am o CA HAI nhanh, nen no khong giai thich")
    print("   duoc HIEU giua chung, va do la thu dang do.")
    rows.append({"test": "prompt length", "quantity": "mean r within cell",
                 "value": round(float(np.mean(rs)), 3), "ci_lo": None, "ci_hi": None,
                 "p_boot": None, "n": len(rs),
                 "note": f"{n_neg} negative significant, {n_pos} positive"})


def test_residue(d, rows):
    """Split items by whether anonymisation left real words behind."""
    print("\n2. RESIDUE TU THAT CON SOT")
    flag = ROOT / "results" / "residue_by_item.csv"
    if not flag.exists():
        print("   thieu results/residue_by_item.csv - chay analyze_anomaly_residue.py")
        return
    lab = pd.read_csv(flag).set_index("item")
    qs = causal_qs(d)
    W = did_cells(d, qs)
    if W.empty:
        print("   khong du du lieu")
        return
    # PERMUTE is excluded from the definition on purpose. A derangement REUSES
    # the item's own words by design, so 93.1% of items "keep a real word" under
    # it and "clean across all three lexicons" leaves 2 items - a split that
    # cannot test anything. The residue worry was always about SYMBOL and
    # PSEUDO, the two that claim to remove real words; analyze_anomaly_residue's
    # own docstring says so.
    clean_all = ~(lab[["residue_SYMBOL", "residue_PSEUDO"]].any(axis=1))
    clean = clean_all.reindex(W.index)
    print("   Item SACH = SYMBOL va PSEUDO khong de sot tu that nao.")
    print("   (PERMUTE bi loai khoi dinh nghia: no dung lai chinh tu cua item)")
    n_clean = int(clean.fillna(False).sum())
    print(f"   Trong {len(W)} item nhom nhan qua: {n_clean} sach, "
          f"{len(W) - n_clean} con residue.")

    # The split cannot be run, and the reason is worth more than the test would
    # have been. Residue is almost perfectly confounded with query group.
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv",
                       low_memory=False).set_index("id")
    imap = ROOT / "results" / "_itemmap_lex.csv"
    if imap.exists():
        ids = pd.read_csv(imap).set_index("item").id
        qt = pd.Series(lab.index, index=lab.index).map(ids).map(full.query_type)
        # From the FULL 174-item label set, not the 86 reindexed to the
        # causal group - otherwise backadj, correlation and marginal show
        # count 0, which is the confound this table exists to display.
        tab = pd.DataFrame({"qt": qt, "clean": clean_all.astype(int)}).groupby(
            "qt").clean.agg(["sum", "count"])
        tab["pct_clean"] = (100 * tab["sum"] / tab["count"]).round(1)
        print("\n   Ty le item SACH theo loai truy van, tren ca 174 item:")
        print(tab.to_string())
        for q, r_ in tab.iterrows():
            rows.append({"test": "residue", "quantity": f"pct clean | {q}",
                         "value": float(r_.pct_clean), "ci_lo": None,
                         "ci_hi": None, "p_boot": None, "n": int(r_["count"]),
                         "note": ""})

    print("\n   KHONG CHAY DUOC PHEP TACH, va ly do dang gia hon phep tach.")
    print("   `backadj` va `correlation` SACH 100%, moi loai truy van nhan qua")
    print("   that chi 0-12,5%. Residue gan nhu TRUNG KHIT voi nhom truy van,")
    print("   nen trong nhom nhan qua chi con 5 item sach - khong du de so.")
    print("   Tach theo residue o day se la tach theo loai truy van doi ten.")
    print("\n   He qua doc duoc: gia thuyet residue KHONG bi loai tru bang phep")
    print("   nay. No bi rang buoc boi mot lap luan KHAC - mtc 7 cho thay bac 2")
    print("   va bac 3 deu bang 0 trong khi residue thi khac nhau giua chung.")
    rows.append({"test": "residue", "quantity": "clean items in causal group",
                 "value": n_clean, "ci_lo": None, "ci_hi": None, "p_boot": None,
                 "n": len(W), "note": "confounded with query_type, split not testable"})


def test_lucky_slice(d, rows):
    """Is +14.35 on the causal subset just what any 86-item slice would give?"""
    print("\n3. MOT LAT CAT MAY MAN")
    qs_all = set(d["KEEP"].query_type.unique())
    qs_causal = causal_qs(d)
    W_causal = did_cells(d, qs_causal)
    W_all = did_cells(d, qs_all)
    if W_causal.empty or W_all.empty:
        print("   khong du du lieu")
        return
    real = 100 * np.nanmean(W_causal.mean(axis=0).values)
    k = len(W_causal.index.unique())
    pool = W_all.index.unique()
    rng = np.random.default_rng(SEED)
    draws = np.empty(NPERM)
    for i in range(NPERM):
        s = rng.choice(pool, k, replace=False)
        draws[i] = 100 * np.nanmean(W_all.loc[s].mean(axis=0).values)
    beat = int((draws >= real).sum())
    print(f"   Lat cat that ({k} item, nhom nhan qua): {real:+.2f} pp")
    print(f"   {NPERM} lat cat NGAU NHIEN cung co: trung binh {draws.mean():+.2f}, "
          f"do lech {draws.std(ddof=1):.2f}")
    print(f"   So lan ngau nhien dat >= lat cat that: {beat}/{NPERM} "
          f"(p = {(beat + 1) / (NPERM + 1):.4f})")
    print("   Day la phep kiem hoan vi cho quyet dinh LOC HAU KIEM. No khong")
    print("   bien mot quyet dinh hau kiem thanh tien kiem; no chi noi lat cat")
    print("   do co dac biet hay khong.")
    rows.append({"test": "lucky slice", "quantity": "DiD on causal subset",
                 "value": round(real, 2), "ci_lo": round(float(np.percentile(draws, 2.5)), 2),
                 "ci_hi": round(float(np.percentile(draws, 97.5)), 2),
                 "p_boot": round((beat + 1) / (NPERM + 1), 4), "n": k,
                 "note": f"{beat}/{NPERM} random slices reach it"})
    # REPORT section 4.7 quotes the mean and spread of the random slices.
    for q, v in (("random slices, mean", draws.mean()),
                 ("random slices, sd", draws.std(ddof=1))):
        rows.append({"test": "lucky slice", "quantity": q, "value": round(float(v), 2),
                     "ci_lo": None, "ci_hi": None, "p_boot": None, "n": NPERM, "note": ""})


def test_scoring(d, rows):
    """Score unparsed answers WRONG instead of dropping them."""
    print("\n4. DO NHAY CHAM DIEM")
    qs = causal_qs(d)
    base = did_cells(d, qs)
    est0, lo0, hi0, p0, n0 = boot_ci(base)
    d2 = {k: v.copy() for k, v in d.items()}
    for k in d2:
        d2[k].loc[d2[k].parsed == 0, "correct"] = 0
        d2[k]["parsed"] = 1
    alt = did_cells(d2, qs)
    est1, lo1, hi1, p1, n1 = boot_ci(alt)
    print(f"   Bo cau khong parse duoc : {est0:+6.2f} pp  "
          f"[{lo0:+6.2f} ; {hi0:+6.2f}]  p={p0:.4f}  n={n0}")
    print(f"   Cham chung thanh SAI    : {est1:+6.2f} pp  "
          f"[{lo1:+6.2f} ; {hi1:+6.2f}]  p={p1:.4f}  n={n1}")
    print(f"   Chenh {abs(est1 - est0):.2f} pp.")
    for lab, (e, lo, hi, p, n) in [("drop unparsed", (est0, lo0, hi0, p0, n0)),
                                   ("unparsed = wrong", (est1, lo1, hi1, p1, n1))]:
        rows.append({"test": "scoring sensitivity", "quantity": lab,
                     "value": round(e, 2), "ci_lo": round(lo, 2),
                     "ci_hi": round(hi, 2), "p_boot": round(p, 4), "n": n,
                     "note": ""})


def test_p_floor(d, rows):
    """What p=0.0005 hides. Bootstrap p floors at 2/B, so read two exact tests."""
    print("\n5. SAN P CUA BOOTSTRAP")
    from scipy import stats as st
    W = did_cells(d, causal_qs(d))
    per = W.mean(axis=1).dropna()
    t_p = float(st.ttest_1samp(per, 0).pvalue)
    w_p = float(st.wilcoxon(per).pvalue)
    # REPORT section 4.0 quotes both. They are tests on the per-ITEM mean, so
    # they describe that estimator (+15.13 here), not the per-cell +14.35 -
    # same data, same sign, a different average.
    print(f"   {len(per)} item, trung binh theo item {100 * per.mean():+.2f} pp")
    print(f"   t-test mot mau  p = {t_p:.5f}")
    print(f"   Wilcoxon        p = {w_p:.5f}")
    for lab, p in (("t-test on per-item DiD", t_p), ("Wilcoxon on per-item DiD", w_p)):
        rows.append({"test": "p floor", "quantity": lab,
                     "value": round(100 * float(per.mean()), 2), "ci_lo": None,
                     "ci_hi": None, "p_boot": round(p, 5), "n": len(per),
                     "note": "bootstrap p floors at 2/B = 0.0005"})


def main() -> int:
    d = load()
    print("=" * 84)
    print("4.7 BON GIA THUYET CANH TRANH")
    print("=" * 84)
    rows = []
    test_prompt_length(d, rows)
    test_residue(d, rows)
    test_lucky_slice(d, rows)
    test_scoring(d, rows)
    test_p_floor(d, rows)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "falsification.csv", index=False)
    print("\n  ghi ra results/falsification.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

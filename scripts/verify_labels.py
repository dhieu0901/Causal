"""Nhan yes/no cua CLadder di theo gia tri nao: gia tri CONG BO hay gia tri DUNG.

Boi canh. `verify_groundtruth.py` chung minh truong `meta.groundtruth` cua CLadder
sinh ra tu mot phep tinh sai: no nhan xac suat bien cua cac nut cha nhu the chung
doc lap. Cau hoi tiep theo la sai lech do co cham toi NHAN dung de cham diem khong.

Cach lam, khong lay bat ky con so nao tu REPORT.md:
  1. tinh lai gia tri DUNG bang bo giai SCM trong verify_groundtruth.py
  2. lay gia tri CONG BO tu meta.groundtruth
  3. loc cac cau ma hai gia tri nam HAI PHIA nguong quyet dinh
  4. voi moi cau do, xem nhan yes/no trong du lieu trung voi ben nao

Chay:  python scripts/verify_labels.py
Ket qua: in bang, va ghi results/_label_flips.csv
"""

import csv
import importlib.util
import json
import os
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# chi `marginal` la xac suat nen so voi 0,5; moi dai luong hieu so voi 0
THR = {"marginal": 0.5}

# hai loai nay bo giai chua ho tro, khai ro thay vi im lang bo qua
UNSUPPORTED = ("exp_away", "det-counterfactual")


def _load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", os.path.join(ROOT, "scripts", "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


def threshold(qt):
    return THR.get(qt, 0.0)


def true_value(vg, scm_meta, qt):
    """Tinh lai dai luong bang bo giai SCM. None neu khong ho tro."""
    scm = vg.SCM(scm_meta["params"])
    if qt == "marginal":
        return scm.prob({"Y": 1})
    if qt == "correlation":
        return (scm.prob({"Y": 1}, given={"X": 1})
                - scm.prob({"Y": 1}, given={"X": 0}))
    if qt == "ate":
        return scm.ate()
    if qt == "ett":
        return scm.ett()
    if qt in ("nde", "nie"):
        d = scm.nde_nie()
        # CLadder dung quy uoc both-vs-base cua Pearl, khong phai telescoping
        return d["NDE"] if qt == "nde" else d["NIE_both_vs_base"]
    return None


def label_for(value, thr, polarity):
    """Nhan yes/no suy ra tu gia tri. Quy tac nay da kiem tren toan bo du lieu."""
    return "yes" if ((value > thr) == bool(polarity)) else "no"


def count_label_flips(verbose=True, write_csv=True):
    """Tra ve (so_theo_gia_tri_hong, tong_so_cau_quyet_dinh, bang_theo_query_type)."""
    vg = _load_solver()
    meta = json.loads(open(os.path.join(ROOT, "data", "cladder-meta.json"),
                           encoding="utf-8").read())
    qs = json.loads(open(os.path.join(ROOT, "data", "cladder-questions.json"),
                         encoding="utf-8").read())
    scms = {m["model_id"]: m for m in meta}

    skipped = Counter()
    decisive = []
    for q in qs:
        m = q["meta"]
        qt = m["query_type"]
        pub = m.get("groundtruth")
        if not isinstance(pub, (int, float)):
            skipped["groundtruth khong phai so"] += 1
            continue
        if qt in UNSUPPORTED:
            skipped["bo giai chua ho tro %s" % qt] += 1
            continue
        sm = scms.get(m["model_id"])
        if sm is None:
            skipped["thieu SCM"] += 1
            continue
        try:
            tru = true_value(vg, sm, qt)
        except Exception as e:
            skipped["loi tinh %s" % type(e).__name__] += 1
            continue
        if tru is None:
            skipped["khong ho tro %s" % qt] += 1
            continue
        t = threshold(qt)
        if (float(pub) > t) == (float(tru) > t):
            continue                      # hai gia tri cung phia, nhan khong doi
        decisive.append((q, m, qt, float(pub), float(tru), t))

    per_qt = defaultdict(Counter)
    rows = []
    unresolved = 0
    for q, m, qt, pub, tru, t in decisive:
        pol = m.get("polarity")
        ans = q["answer"]
        if pol is None:
            unresolved += 1
            continue
        lp, lt = label_for(pub, t, pol), label_for(tru, t, pol)
        if lp == lt:
            unresolved += 1
            continue
        if ans == lp:
            side = "hong"
        elif ans == lt:
            side = "dung"
        else:
            unresolved += 1
            continue
        per_qt[qt][side] += 1
        rows.append({"question_id": q.get("question_id"),
                     "graph_id": m["graph_id"], "query_type": qt,
                     "gia_tri_cong_bo": pub, "gia_tri_dung": tru,
                     "nguong": t, "nhan": ans, "nhan_di_theo": side})

    broken = sum(c["hong"] for c in per_qt.values())
    total = broken + sum(c["dung"] for c in per_qt.values())

    if write_csv:
        out = os.path.join(ROOT, "results", "_label_flips.csv")
        with open(out, "w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    if verbose:
        print("KIEM NHAN CHUAN CLADDER")
        print("=" * 62)
        if skipped:
            print("  bo qua:", dict(skipped))
        print("  so cau QUYET DINH (hai gia tri khac phia nguong):", len(decisive))
        if unresolved:
            print("  khong phan dinh duoc:", unresolved)
        print()
        print(f"  {'query_type':14s} {'HONG':>6s} {'DUNG':>6s}")
        print("  " + "-" * 28)
        for qt in sorted(per_qt):
            print(f"  {qt:14s} {per_qt[qt]['hong']:6d} {per_qt[qt]['dung']:6d}")
        print("  " + "-" * 28)
        print(f"  {'TONG':14s} {broken:6d} {total - broken:6d}")
        print()
        print(f"  => {broken}/{total} nhan di theo gia tri HONG "
              f"({100.0 * broken / total:.1f}%)")
        if write_csv:
            print("  da ghi results/_label_flips.csv")

    return broken, total, {k: dict(v) for k, v in per_qt.items()}


if __name__ == "__main__":
    count_label_flips()

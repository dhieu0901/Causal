"""Mot canh dao co phai mot lieu khong, hay lieu la TY LE canh bi hong?

Phan bien nhan duoc ngay 2026-09-20: dao mot canh tren do thi ba canh la hong
33% do thi, con tren do thi nam canh chi la 20%. Vay khi bao "DR_k1" nhu mot
dieu kien duy nhat, ta dang gop bon lieu khac nhau lai lam mot. Phan bien nay
dung, va du lieu quet k san co du de tra loi.

Bay ho trong results/pilot_raw_price400KEEP.csv - 399 item, gap 2,7 lan file
pilot_raw.csv ma ban truoc dung - voi k = 1, 2, 3:

    ho ba canh (confounding, mediation)                33%, 67%, 100%
    ho bon canh (IV, diamond, diamondcut, frontdoor)   25%, 50%,  75%
    ho nam canh (arrowhead)                            20%, 40%,  60%

tuc chin muc lieu tu 20% den 100%. Cau hoi kiem duoc: mat mat di theo SO canh
dao, hay theo TY LE canh dao? Neu theo ty le, cac duong cua cac ho se chap lai
lam mot khi ve theo k/E, chu khong phai khi ve theo k.

Cach do. Voi moi item, so voi chinh no o dieu kien ORACLE (ghep cap, cung item
cung model), nen moi so la mot muc mat mat so voi do thi dung. Bootstrap cum
theo item. Sau do hoi mot cau rat don gian: hoi quy mat mat theo k duoc R2 bao
nhieu, theo k/E duoc bao nhieu.

Chay:  python scripts/analyze_dose.py
Ket qua: in bang, ghi results/dose_response.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

SEED = 20260920
NBOOT = 4000
LOAI = {"DR": "dao chieu canh", "ED": "thieu canh", "FE": "thua canh"}


def canh_moi_ho():
    """So canh cua moi ho, doc thang tu khoa tham so trong cladder-meta.json."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    out = {}
    for m in meta:
        if m["graph_id"] in out:
            continue
        s = vg.SCM(m["params"])
        out[m["graph_id"]] = sum(len(s.parents[n]) for n in s.nodes)
    return out


def mat_mat(d, ho, cond):
    """Mat mat ghep cap so voi ORACLE, tren tung item, gop ba model.

    Tra ve Series chi so la item, gia tri la trung binh tren cac model co du
    ca hai dieu kien. Ghep cap trong item la diem manh cua thiet ke, giu no.
    """
    sub = d[(d.graph_id == ho) & (d.parsed == 1)]
    cols = []
    for m in sorted(sub.model.unique()):
        o = sub[(sub.model == m) & (sub.cond == "ORACLE")].set_index("item").correct
        c = sub[(sub.model == m) & (sub.cond == cond)].set_index("item").correct
        i = o.index.intersection(c.index)
        if len(i) < 5:
            continue
        cols.append(pd.Series((o[i] - c[i]).values, index=i, name=m))
    if not cols:
        return pd.Series(dtype=float)
    return pd.concat(cols, axis=1).mean(axis=1)


def boot(v, seed=SEED, n=NBOOT):
    """Bootstrap cum theo item cho trung binh, tra (uoc luong, lo, hi)."""
    x = v.dropna().values
    if len(x) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for b in range(n):
        out[b] = x[rng.integers(0, len(x), len(x))].mean()
    return 100 * x.mean(), 100 * np.percentile(out, 2.5), 100 * np.percentile(out, 97.5)


def main():
    d = pd.read_csv(ROOT / "results" / "pilot_raw_price400KEEP.csv")
    E = canh_moi_ho()
    hos = sorted(d.graph_id.unique(), key=lambda g: (E[g], g))

    print("=" * 86)
    print("LIEU DUONG: mat mat so voi ORACLE theo SO canh hong va theo TY LE canh hong")
    print("=" * 86)
    print("  ghep cap trong item, gop ba model, bootstrap cum theo item,",
          NBOOT, "lan, seed", SEED)
    print("  so duong la pp mat di so voi chinh item do khi duoc cap do thi DUNG\n")

    rows = []
    for loai in ("DR", "ED", "FE"):
        conds = sorted(c for c in d.cond.unique() if c.startswith(loai + "_k"))
        if not conds:
            continue
        print(f"\n{loai} - {LOAI[loai]}")
        print(f"  {'ho':13s} {'canh':>4s} {'k':>2s} {'ty le hong':>10s} "
              f"{'mat pp':>7s}  {'khoang tin cay':>18s} {'n':>4s}")
        print("  " + "-" * 68)
        for ho in hos:
            for c in conds:
                k = int(c.split("_k")[1])
                if k > E[ho]:
                    continue
                v = mat_mat(d, ho, c)
                if v.empty:
                    continue
                est, lo, hi = boot(v)
                if np.isnan(est):
                    continue
                ty = k / E[ho]
                print(f"  {ho:13s} {E[ho]:4d} {k:2d} {ty:9.0%} "
                      f"{est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {len(v):4d}")
                rows.append({"loai": loai, "ho": ho, "so_canh": E[ho], "k": k,
                             "ty_le_canh_hong": round(ty, 3),
                             "mat_pp": round(est, 2), "ci_lo": round(lo, 2),
                             "ci_hi": round(hi, 2), "n_item": len(v)})

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "dose_response.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 86)
    print("LIEU NAO GIAI THICH TOT HON: so canh k, hay ty le canh k/E?")
    print("=" * 86)
    print("  Hoi quy tuyen tinh mat mat theo tung bien, tren cac o (ho x k).")
    print("  Neu ty le la bien dung, cac ho se chap vao mot duong khi ve theo k/E.\n")
    print("  Cot thu ba la mo hinh chi gom bien gia cho TUNG HO, khong co lieu gi ca.")
    print("  R2 hieu chinh de phat mo hinh nhieu tham so hon.\n")
    print(f"  {'loai':6s} {'so o':>5s} {'R2hc theo k':>12s} {'R2hc theo k/E':>14s} "
          f"{'R2hc theo HO':>13s} {'thang'}")
    print("  " + "-" * 70)

    def r2_hieu_chinh(A, y):
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss = ((y - y.mean()) ** 2).sum()
        if ss <= 0:
            return np.nan
        r2 = 1 - (resid ** 2).sum() / ss
        n, p = len(y), A.shape[1] - 1
        return 1 - (1 - r2) * (n - 1) / (n - p - 1) if n - p - 1 > 0 else np.nan

    for loai in ("DR", "ED", "FE"):
        s = df[df.loai == loai]
        if len(s) < 4 or s.ho.nunique() < 2:
            print(f"  {loai:6s} {len(s):5d}  qua it o de so sanh")
            continue
        y = s.mat_pp.values
        one = np.ones((len(s), 1))
        r2 = {"k": r2_hieu_chinh(np.hstack([s.k.values.astype(float)[:, None], one]), y),
              "k/E": r2_hieu_chinh(np.hstack([s.ty_le_canh_hong.values[:, None], one]), y),
              "ho": r2_hieu_chinh(np.hstack([pd.get_dummies(s.ho, drop_first=True)
                                             .values.astype(float), one]), y)}
        win = max(r2, key=lambda k: -np.inf if np.isnan(r2[k]) else r2[k])
        print(f"  {loai:6s} {len(s):5d} {r2['k']:12.3f} {r2['k/E']:14.3f} "
              f"{r2['ho']:13.3f} {win:>7s}")

    print("\n  Luu y doc bang. Bay ho chi co ba muc so canh (3, 4, 5), va ho nam canh")
    print("  chi co MOT ho la arrowhead - dung ho ma nhan CLadder hong. Nen phep so")
    print("  sanh nay la bang chung dinh huong, chua du de chot bien lieu nao dung.")
    print("  Doc ket qua. MOI R2 o day deu THAP - cao nhat trong hai cot lieu la")
    print("  0,11. Nen cau tra loi cho phan bien khong phai 'k hay k/E moi dung',")
    print("  ma la: LIEU GIAI THICH RAT IT, du do bang cach nao.")
    print("  Voi ED va FE, bien gia theo ho ap dao (0,36 va 0,61): cai quyet dinh la")
    print("  CAU TRUC nao bi hong, khong phai hong bao nhieu. Voi DR thi k/E va ho")
    print("  ngang nhau va deu yeu, nen rieng o DR chua ket luan duoc ben nao.")
    print("\n  DINH CHINH. Ban truoc cua khoi nay chay tren results/pilot_raw.csv")
    print("  (147 item) va bao rang R2 cua CA HAI bien lieu deu AM. Tren mau 399")
    print("  item thi dieu do KHONG con dung voi DR: k/E len 0,11. Ket luan manh")
    print("  'lieu khong phai bien giai thich' chi dung cho ED va FE.")
    print(f"\nda ghi {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

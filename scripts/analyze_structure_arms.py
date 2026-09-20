"""Cac nhanh cau truc ORACLE, PROSE, DR_k1 - tren mau GOP n=490.

Vi sao co file nay, va vi sao no da duoc viet lai. Ban dau REPORT muc 4.3 va 4.4
bao sau gia tri (ORACLE +14,31, PROSE +13,91, DR_k1 +8,64, ...) ma KHONG file
results/*.csv nao chua va khong script nao sinh ra. Ban dau tien cua file nay lap
lai phep tinh do - nhung tren mau kham pha n=86, dung mau ma REPORT muc 4.0 da
RUT vi loi nguyen nguoi thang cuoc.

`pool_samples.py` phuc hoi duoc anh xa item -> id goc CLadder, nen ba mau gop lai
duoc thanh n=490. Ban nay chay tren mau gop do. Con so vi the KHAC han ban truoc,
va do la diem chinh chu khong phai tac dung phu.

Uoc luong, cho mot khoi cau truc C bat ky:
    DiD_i(C) = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | C]
Tach hai ve:
    nang = acc(PSEUDO | C) - acc(PSEUDO | RAW)      nhanh an danh
    keo  = acc(KEEP   | C) - acc(KEEP   | RAW)      nhanh tu goc
    DiD  = nang - keo

PSEUDO la tu vung an danh duy nhat co o ca ba mau. Mau kham pha con co PERMUTE va
SYMBOL, nhung dung chung se khong so sanh duoc giua cac mau.

Chay:  python scripts/analyze_structure_arms.py
Ket qua: in bang, ghi results/structure_arms.csv (make_figures.py doc file nay)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_querygroup import ARITH, IDENT, TIER
from pool_samples import (LEX_GOP, MAU, SEED, boot, cell, did_mau, kiem_anh_xa,
                          nap, _pilot)

ARMS = ["ORACLE", "PROSE", "DR_k1"]


def anh_xa_tat_ca():
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")
    return {tag: kiem_anh_xa(tag, n, kmax, drop, full, pilot)
            for tag, n, kmax, drop in MAU}


def gop_did(amaps, arm):
    """Ma tran DiD (id x o) gop ca ba mau, cho mot khoi cau truc."""
    per = []
    for tag, *_ in MAU:
        W = did_mau(tag, amaps[tag], [LEX_GOP], arm=arm)
        if not W.empty:
            per.append(W.rename(columns=lambda c: f"{tag}|{c}"))
    return pd.concat(per, axis=1) if per else pd.DataFrame()


def gop_hieu(amaps, a, b):
    """Hieu ghep cap tren tung item: DiD(a) tru DiD(b), chi tren o co ca hai."""
    A, B = gop_did(amaps, a), gop_did(amaps, b)
    cols = [c for c in A.columns if c in B.columns]
    i = A.index.intersection(B.index)
    if not cols or len(i) < 10:
        return pd.DataFrame()
    return A.loc[i, cols] - B.loc[i, cols]


def gop_tach(amaps, arm):
    """Tach nang (nhanh an danh) va keo (nhanh KEEP), gop ba mau."""
    up, dn = [], []
    for tag, *_ in MAU:
        amap = amaps[tag]
        keep = nap(tag, "KEEP", amap)
        lex = nap(tag, LEX_GOP, amap)
        qs = set(keep.query_type.unique()) - ARITH - IDENT
        for m in TIER:
            kr, ka = cell(keep, m, "RAW", qs), cell(keep, m, arm, qs)
            j = kr.index.intersection(ka.index)
            if len(j) >= 10:
                dn.append(pd.Series((ka[j] - kr[j]).values, index=j,
                                    name=f"{tag}|{m}"))
            lr, la = cell(lex, m, "RAW", qs), cell(lex, m, arm, qs)
            i = lr.index.intersection(la.index)
            if len(i) >= 10:
                up.append(pd.Series((la[i] - lr[i]).values, index=i,
                                    name=f"{tag}|{m}"))

    def gom(cols):
        if not cols:
            return pd.DataFrame()
        return pd.concat(cols, axis=1).groupby(level=0).mean()

    return gom(up), gom(dn)


def bao(ten, W, rows):
    if W.empty:
        print(f"  {ten:28s} khong du du lieu")
        return
    est, lo, hi, p = boot(W)
    n = len(W.index.unique())
    print(f"  {ten:28s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={n}")
    rows.append({"dai_luong": ten, "uoc_luong_pp": round(est, 2),
                 "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                 "p_boot": round(p, 4), "n_item": n})


def main():
    amaps = anh_xa_tat_ca()

    print("=" * 78)
    print("NHANH CAU TRUC, mau GOP n=490, nhom cau hoi nhan qua that")
    print("=" * 78)
    print("  bootstrap cum theo item, 4000 lan, seed", SEED)
    print(f"  tu vung an danh: {LEX_GOP} (duy nhat co o ca ba mau)\n")

    rows = []
    print("1. DiD theo tung khoi cau truc  (so voi RAW)")
    for arm in ARMS:
        bao(f"DiD | {arm}", gop_did(amaps, arm), rows)

    print("\n2. Hieu ghep cap giua hai khoi")
    bao("ORACLE tru DR_k1", gop_hieu(amaps, "ORACLE", "DR_k1"), rows)
    bao("ORACLE tru PROSE", gop_hieu(amaps, "ORACLE", "PROSE"), rows)

    print("\n3. Tach nang / keo cho khoi ORACLE")
    U, D = gop_tach(amaps, "ORACLE")
    bao("nang nhanh an danh", U, rows)
    bao("keo nhanh KEEP", D, rows)

    if not U.empty and not D.empty:
        up = 100 * np.nanmean(U.values)
        dn = 100 * np.nanmean(D.values)
        tot = up - dn
        print(f"\n  kiem tra cong: nang {up:+.2f} tru keo {dn:+.2f} = {tot:+.2f} pp")
        if abs(dn) > 1e-9 and tot > 0:
            print(f"  phan hieu ung do LAM HAI nhanh KEEP: "
                  f"{100 * abs(min(dn, 0.0)) / tot:.0f}%")

    out = ROOT / "results" / "structure_arms.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nda ghi {out.relative_to(ROOT)}")
    print("\nLUU Y. Ban truoc cua file nay chay tren mau kham pha n=86 va cho")
    print("ORACLE +14,35 / DR_k1 +8,69. Con so o day tren n=490 nen thap hon")
    print("dang ke. Mau kham pha la mau da bi RUT - xem REPORT muc 4.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

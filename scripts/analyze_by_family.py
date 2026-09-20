"""Tach ket qua cau truc theo TUNG HO DO THI, tren mau GOP n=490.

Vi sao co file nay, va vi sao no da duoc viet lai. Ban dau no chay tren mau
kham pha n=86 - dung mau ma REPORT muc 4.0 da RUT vi loi nguyen nguoi thang
cuoc: DiD cua no la +12,71 trong khi uoc luong gop la +5,98. Voi 7 den 10 item
moi ho, khoang tin cay tung ho rong ±20 den ±40 pp, nen bang do chi la manh moi.

`pool_samples.py` phuc hoi duoc anh xa item -> id goc CLadder, nen ba mau gop
lai duoc. Tren mau gop, moi ho co khoang 30 den 70 item thay vi 7 den 10. Day
la phep kiem bien manh moi thanh ket qua - hoac giet no gon gang.

PHEP SO SANH DUNG PSEUDO o ca ba mau, vi hai mau lon chi co tu vung do. Mau
kham pha co ba tu vung an danh nhung dung ca ba se khong so sanh duoc.

LUU Y PHAM VI. Mau `price400` chay voi kmax=3 nen chi phu 7 ho; ba ho hai canh
(chain, collision, fork) chi den tu `lex` va `n600`. So item moi ho vi the khong
deu, va cot n phai doc kem moi o.

Chay:  python scripts/analyze_by_family.py
Ket qua: in bang, ghi results/family_breakdown.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))


import pandas as pd

from pool_samples import LEX_GOP, MAU, SEED, boot, did_mau, kiem_anh_xa, _pilot

# Ho duy nhat ma CLadder tinh sai CA BON dai luong nhan qua, khong chi P(Y=1).
# Xem scripts/audit_cladder_arithmetic.py.
NGHI_NGO = {"arrowhead"}


def cau_truc():
    """(so nut, so canh) moi ho, doc thang tu khoa tham so trong cladder-meta."""
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
        out[m["graph_id"]] = (len(s.nodes), sum(len(s.parents[n]) for n in s.nodes))
    return out


def ma_tran_gop():
    """Ma tran DiD (id x o) gop ca ba mau, va anh xa id -> graph_id."""
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")
    per = {}
    for tag, n, kmax, drop in MAU:
        amap = kiem_anh_xa(tag, n, kmax, drop, full, pilot)
        W = did_mau(tag, amap, [LEX_GOP])
        if not W.empty:
            per[tag] = W.rename(columns=lambda c: f"{tag}|{c}")
    if not per:
        raise SystemExit("khong dung duoc ma tran DiD")
    G = pd.concat(per.values(), axis=1)
    ho = full.set_index("id").graph_id
    return G, ho


def bao(ten, W, extra, rows):
    n = len(W.index.unique())
    if W.empty or n < 5:
        print(f"  {ten:30s} chi {n} item, bo qua")
        return
    est, lo, hi, p = boot(W)
    co = "  <- nhan CLadder hong" if extra.get("nghi_ngo") else ""
    print(f"  {ten:30s} {est:+7.2f}  [{lo:+7.2f} ; {hi:+7.2f}]  p={p:.4f}  n={n}{co}")
    rows.append({"lat_cat": ten, **extra, "uoc_luong_pp": round(est, 2),
                 "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                 "p_boot": round(p, 4), "n_item": n})


def main():
    G, ho = ma_tran_gop()
    st = cau_truc()
    fam = ho.reindex(G.index)
    if fam.isna().any():
        raise SystemExit("co id khong tra duoc graph_id")

    print("=" * 88)
    print("DiD THEO CAU TRUC DO THI - mau GOP ba nguon, bo trung theo id goc")
    print("=" * 88)
    print("  DiD_i = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]")
    print("  nhom cau hoi nhan qua that, bootstrap cum theo item, 4000 lan, seed", SEED)
    print(f"  tong {len(G)} item tren {G.shape[1]} o (mau x model)\n")

    rows = []
    print("1. TUNG HO DO THI  (sap theo so canh, roi ten)")
    print(f"  {'ho':30s} {'DiD':>7s}  {'khoang tin cay 95%':>20s}")
    for g in sorted(st, key=lambda x: (st[x][1], st[x][0], x)):
        nn, ne = st[g]
        idx = fam.index[fam == g]
        bao(f"{g} ({nn} nut, {ne} canh)", G.loc[G.index.isin(idx)],
            {"ho": g, "so_nut": nn, "so_canh": ne, "nhanh": "ORACLE",
             "nghi_ngo": g in NGHI_NGO}, rows)

    print("\n2. GOP THEO SO CANH")
    for ne in sorted({v[1] for v in st.values()}):
        gs = [g for g in st if st[g][1] == ne]
        idx = fam.index[fam.isin(gs)]
        bao(f"{ne} canh: {', '.join(sorted(gs))}"[:30], G.loc[G.index.isin(idx)],
            {"ho": "+".join(sorted(gs)), "so_nut": "", "so_canh": ne,
             "nhanh": "ORACLE", "nghi_ngo": bool(set(gs) & NGHI_NGO)}, rows)

    print("\n3. GOP THEO SO NUT")
    for nn in sorted({v[0] for v in st.values()}):
        gs = [g for g in st if st[g][0] == nn]
        idx = fam.index[fam.isin(gs)]
        bao(f"{nn} nut ({len(gs)} ho)", G.loc[G.index.isin(idx)],
            {"ho": "+".join(sorted(gs)), "so_nut": nn, "so_canh": "",
             "nhanh": "ORACLE", "nghi_ngo": bool(set(gs) & NGHI_NGO)}, rows)

    print("\n4. KIEM DO NHAY: bo ho arrowhead khoi nhom 4 nut")
    gs = [g for g in st if g not in NGHI_NGO and st[g][0] == 4]
    idx = fam.index[fam.isin(gs)]
    bao("4 nut, khong arrowhead", G.loc[G.index.isin(idx)],
        {"ho": "+".join(sorted(gs)), "so_nut": 4, "so_canh": "",
         "nhanh": "ORACLE", "nghi_ngo": False}, rows)

    out = ROOT / "results" / "family_breakdown.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nda ghi {out.relative_to(ROOT)}")
    print("\nDOC BANG. Day la mau GOP n=490, khong phai mau kham pha n=86 ma cac")
    print("ban truoc cua file nay dung. Khoang tin cay hep hon dang ke, nhung 16")
    print("lat cat nay KHONG hieu chinh da so sanh, nen tung o van la tham do.")
    print("O nao co arrowhead phai doc kem luu y ve nhan CLadder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

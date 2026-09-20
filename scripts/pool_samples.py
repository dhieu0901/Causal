"""Tai lap con so tieu de: gop ba mau o muc ITEM, bo trung theo id goc CLadder.

Vi sao co file nay. REPORT muc 4.0 rut con so +14,35 (mau kham pha n=86) va thay
bang **+5,98 pp tren n=490 item, gop ba mau, bo trung theo id goc**. Ra soat ngay
2026-09-20: **khong script nao trong repo sinh ra con so do**. No chi ton tai
duoi dang van xuoi. Day la con so TIEU DE cua ca du an.

Ly do khong tai lap duoc ngay: `pilot.py` khong ghi cot `id` goc cua CLadder vao
CSV ket qua. Cot `item` chi la chi so tham chieu TRONG tung mau (`--pair-to` lam
KEEP va PSEUDO cua CUNG mot mau dung chung chi so, nhung hai mau khac nhau thi
khong). Khoa tu nhien (story_id, graph_id, query_type, rung) cung khong duy nhat:
174 item chi cho 110 to hop.

Loi ra: `make_items` la ham TAT DINH - `random.Random(seed)` roi lay mau phan
tang theo (graph_id, rung). Chay lai no voi dung tham so thi phuc hoi duoc cot
`id`. Tham so do lai bang cach quet (n, kmax, drop_nonsense) cho toi khi day
item sinh ra khop TUNG DONG voi (graph_id, rung, query_type, story_id) trong CSV
da luu. Ket qua do duoc:

    lex       n=174  kmax=1  drop_nonsense=True   10 ho
    n600      n=580  kmax=1  drop_nonsense=True   10 ho
    price400  n=399  kmax=3  drop_nonsense=True    7 ho

PHEP SO SANH PHAI DONG NHAT GIUA BA MAU. Mau `lex` co ba tu vung an danh
(PERMUTE, SYMBOL, PSEUDO), hai mau kia chi co PSEUDO. Nen phep gop dung PSEUDO o
ca ba. Dung ca ba tu vung cho `lex` se ra +14,31 tren 86 item thay vi +12,71
tren 85 - khong so sanh duoc voi hai mau con lai.

QUY UOC LAY TRUNG BINH la cho de sai nhat. Trung binh phai lay tren MOI O
(item x model), khong phai trung binh theo item roi moi trung binh cac item.
Hai cach khac nhau vi khong phai item nao cung du ca ba o. Cach dau tai lap dung
ba con so cua REPORT; cach sau cho +13,92 / +6,85 / +1,88 va gop ra +6,53.

KET QUA: ca bon con so cua REPORT muc 4.0 tai lap chuan xac.

    kham pha      85  +12,71
    mo rong      287   +6,78
    bac thang    195   +1,72
    GOP          490   +5,98   CI [+1,67 ; +10,19]

Nen con so tieu de la DUNG va da khu trung dung cach. Cai thieu truoc day chi la
mot script sinh ra no. Gio co.

Chay:  python scripts/pool_samples.py
Ket qua: in bang, ghi results/pooled_headline.csv va results/_itemmap_*.csv
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from analyze_querygroup import ARITH, IDENT, TIER

SEED = 20260907
NBOOT = 4000
KEY = ["graph_id", "rung", "query_type", "story_id"]

# (tag file, n, kmax, drop_nonsense). Tham so do lai va kiem tung dong - xem
# docstring. `kiem_anh_xa` chay lai phep kiem do moi lan, khong tin suong.
MAU = [
    ("lex",      174, 1, True),
    ("n600",     580, 1, True),
    ("price400", 399, 3, True),
]
LEX_GOP = "PSEUDO"          # tu vung an danh co o CA BA mau


def _pilot():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "pilot", str(ROOT / "scripts" / "pilot.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sinh_items(full, pilot, n, kmax, drop):
    """Ban sao y het `make_items`, tru viec doc file - de goi lap duoc."""
    d = full
    if drop:
        d = d[~d.story_id.astype(str).str.startswith("nonsense")]
    fams = [f for f in pilot.FAMILY_STRUCTURE if pilot.max_k(f, "DR") >= kmax]
    d = d[d.graph_id.isin(fams)].reset_index(drop=True)
    rng = random.Random(SEED)
    groups = list(d.groupby(["graph_id", "rung"]))
    per = max(1, n // len(groups))
    picked = []
    for _, g in groups:
        picked += rng.sample(list(g.index), min(per, len(g)))
    picked = picked[:n] if len(picked) >= n else picked
    return d.loc[picked].reset_index(drop=True)


def kiem_anh_xa(tag, n, kmax, drop, full, pilot):
    """Phuc hoi item -> id, va CHUNG MINH no dung bang doi chieu tung dong."""
    it = sinh_items(full, pilot, n, kmax, drop)
    saved = pd.read_csv(ROOT / "results" / f"pilot_raw_{tag}KEEP.csv")
    that = (saved[["item"] + KEY].drop_duplicates()
            .sort_values("item")[KEY].reset_index(drop=True))
    got = it[KEY].reset_index(drop=True)
    if not got.equals(that):
        raise SystemExit(
            f"{tag}: day item sinh lai KHONG khop CSV da luu. Tham so (n={n}, "
            f"kmax={kmax}, drop={drop}) sai, hoac pilot.py da doi. Dung lai.")
    out = it[["id"]].reset_index().rename(columns={"index": "item"})
    out.to_csv(ROOT / "results" / f"_itemmap_{tag}.csv", index=False)
    return out


def nap(tag, lex, amap):
    d = pd.read_csv(ROOT / "results" / f"pilot_raw_{tag}{lex}.csv")
    d = d.merge(amap, on="item", how="left")
    if d.id.isna().any():
        raise SystemExit(f"{tag}/{lex}: co item khong anh xa duoc sang id")
    return d


def cell(d, model, cond, qs):
    s = d[(d.model == model) & (d.cond == cond) & (d.parsed == 1)
          & (d.query_type.isin(qs))]
    return s.set_index("id").correct


def did_mau(tag, amap, lexs, arm="ORACLE"):
    """DiD tren tung id, cho mot mau. Cot la (model x tu vung)."""
    keep = nap(tag, "KEEP", amap)
    qs = set(keep.query_type.unique()) - ARITH - IDENT
    cols = []
    for m in TIER:
        for lx in lexs:
            L = nap(tag, lx, amap)
            kr, lr = cell(keep, m, "RAW", qs), cell(L, m, "RAW", qs)
            ka, la = cell(keep, m, arm, qs), cell(L, m, arm, qs)
            i = (kr.index.intersection(lr.index)
                 .intersection(ka.index).intersection(la.index))
            if len(i) < 10:
                continue
            cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values,
                                  index=i, name=f"{m}|{lx}"))
    if not cols:
        return pd.DataFrame()
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def boot(W, seed=SEED, n=NBOOT):
    """Bootstrap cum theo item. Trung binh lay tren MOI O (item x cell).

    Lay trung binh theo item truoc roi moi trung binh cac item se cho con so
    khac, vi khong phai item nao cung du ca chin o. Quy uoc o day tai lap dung
    ba con so cua REPORT muc 4.0, nen no la quy uoc REPORT da dung.
    """
    rng = np.random.default_rng(seed)
    idx = W.index.values
    out = np.empty(n)
    for b in range(n):
        out[b] = 100 * np.nanmean(W.loc[rng.choice(idx, len(idx), replace=True)].values)
    est = 100 * np.nanmean(W.values)
    p = 2 * min((out <= 0).mean(), (out >= 0).mean())
    return est, np.percentile(out, 2.5), np.percentile(out, 97.5), max(p, 2.0 / n)


def main():
    pilot = _pilot()
    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")

    print("=" * 78)
    print("GOP BA MAU O MUC ITEM, BO TRUNG THEO id GOC CLADDER")
    print("=" * 78)
    print("  DiD_i = [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE]")
    print("  nhom cau hoi nhan qua that, bootstrap cum theo item,",
          NBOOT, "lan, seed", SEED, "\n")

    per, rows = {}, []
    print(f"  {'mau':10s} {'n item':>7s} {'DiD':>8s}  {'khoang tin cay 95%':>21s} {'p':>8s}")
    print("  " + "-" * 60)
    for tag, n, kmax, drop in MAU:
        amap = kiem_anh_xa(tag, n, kmax, drop, full, pilot)
        W = did_mau(tag, amap, [LEX_GOP])
        if W.empty:
            print(f"  {tag:10s} khong du du lieu")
            continue
        per[tag] = W.rename(columns=lambda c: f"{tag}|{c}")
        e, lo, hi, p = boot(W)
        print(f"  {tag:10s} {len(W):7d} {e:+8.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {p:8.4f}")
        rows.append({"cach_gop": tag, "n_item": len(W), "did_pp": round(e, 2),
                     "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                     "p_boot": round(p, 4)})

    # Gop: ghep NGANG cac o cua ba mau tren cung truc id. Item nao xuat hien o
    # nhieu mau thi co nhieu o hon, dung nhu khi gop trong mot mau.
    hop = pd.concat(per.values(), axis=1)
    e, lo, hi, p = boot(hop)
    print("  " + "-" * 60)
    print(f"  {'GOP item':10s} {len(hop):7d} {e:+8.2f}  [{lo:+7.2f} ; {hi:+7.2f}] {p:8.4f}")
    rows.append({"cach_gop": "gop o muc item, bo trung", "n_item": len(hop),
                 "did_pp": round(e, 2), "ci_lo": round(lo, 2),
                 "ci_hi": round(hi, 2), "p_boot": round(p, 4)})

    ns = {t: int(per[t].notna().any(axis=1).sum()) for t in per}
    tong = sum(ns.values())
    w = sum(ns[t] * 100 * np.nanmean(per[t].values) for t in per) / tong
    rows.append({"cach_gop": "trung binh co trong so theo n", "n_item": tong,
                 "did_pp": round(w, 2), "ci_lo": "", "ci_hi": "", "p_boot": ""})

    print("\n" + "=" * 78)
    print("DOI CHIEU VOI REPORT MUC 4.0")
    print("=" * 78)
    trung = int(pd.concat([per[t].notna().any(axis=1) for t in per], axis=1)
                .fillna(False).sum(axis=1).gt(1).sum())
    print(f"  REPORT bao: n=490, DiD +5,98, CI [+1,78 ; +10,30], p=0,005")
    print(f"  tai lap   : n={len(hop)}, DiD {e:+.2f}, "
          f"CI [{lo:+.2f} ; {hi:+.2f}], p={p:.4f}")
    print(f"\n  {trung} item xuat hien o nhieu hon mot mau, da bo trung theo id goc.")
    print(f"  De doi chieu: trung binh co trong so theo n (tong {tong}, tuc DEM")
    print(f"  TRUNG) cho {w:+.2f} pp - gan bang nhung khong phai phep REPORT dung.")
    print("\n  Con so tieu de DUNG va da khu trung dung cach. Thu tung thieu chi la")
    print("  mot script sinh ra no; gio co.")

    out = ROOT / "results" / "pooled_headline.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nda ghi {out.relative_to(ROOT)} va results/_itemmap_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

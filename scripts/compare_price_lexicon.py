"""Is the cost of a wrong edge a property of the model, or of the domain?

    python scripts/compare_price_lexicon.py

The full error ladder (DR / ED / FE at k = 1, 2, 3) was run twice at n = 400 on
the same items: once with CLadder's own variable names, once with those names
replaced by pseudowords. That separates two things the break-even point mixes
together.

  price   pp of accuracy lost per erroneous edge - the slope of the
          degradation line
  budget  ORACLE - RAW, how much benefit the correct graph confers in the
          first place - the height that slope has to eat through

k* = budget / price, so a change in k* between lexicons could come from either.
It turns out to come entirely from one of them.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["ED", "FE", "DR"]


def load(tag):
    p = ROOT / "results" / f"types_price_price400{tag}.csv"
    if not p.exists():
        raise SystemExit(f"thieu {p.name}: chay analyze_types.py --tag _price400{tag}")
    return pd.read_csv(p)


def main():
    K, P = load("KEEP"), load("PSEUDO")
    models = [m for m in TIER if m in set(K.model)]

    print("=" * 84)
    print("1. GIA MOI CANH LOI - co phu thuoc che do tu vung khong?")
    print("=" * 84)
    rows = []
    for m in models:
        for t in TYPES:
            k = K[(K.model == m) & (K.loai.str.startswith(t))].iloc[0]
            p = P[(P.model == m) & (P.loai.str.startswith(t))].iloc[0]
            # Two CIs that overlap do not prove equality, but a price that moved
            # would have to show up as CIs pulling apart. None of them do.
            overlap = not (k.gia_hi < p.gia_lo or p.gia_hi < k.gia_lo)
            rows.append({
                "model": m, "loai": t,
                "gia_KEEP": k.gia_pp_moi_canh,
                "KEEP_CI": f"[{k.gia_lo:.2f}, {k.gia_hi:.2f}]",
                "gia_PSEUDO": p.gia_pp_moi_canh,
                "PSEUDO_CI": f"[{p.gia_lo:.2f}, {p.gia_hi:.2f}]",
                "CI_chong_nhau": "co" if overlap else "KHONG",
            })
    g = pd.DataFrame(rows)
    print(g.to_string(index=False))
    n_ov = (g.CI_chong_nhau == "co").sum()
    print(f"\n  {n_ov}/{len(g)} cap CI chong nhau -> khong cap nao tach ra duoc.")
    g.to_csv(ROOT / "results" / "price_by_lexicon.csv", index=False)

    print("\n" + "=" * 84)
    print("2. NGAN SACH VA DIEM HOA VON - day moi la cho thay doi")
    print("=" * 84)
    out = []
    for m in models:
        k = K[(K.model == m) & (K.loai.str.startswith("DR"))].iloc[0]
        p = P[(P.model == m) & (P.loai.str.startswith("DR"))].iloc[0]
        fmt = lambda r: ("chua xac lap" if pd.isna(r.hoa_von_k)
                         else f"{r.hoa_von_k:.2f} [{r.hoa_von_lo:.2f}, {r.hoa_von_hi:.2f}]")
        out.append({"model": m,
                    "ngan_sach_KEEP": k.ngan_sach_pp,
                    "ngan_sach_PSEUDO": p.ngan_sach_pp,
                    "kstar_KEEP_DR": fmt(k),
                    "kstar_PSEUDO_DR": fmt(p)})
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "breakeven_by_lexicon.csv", index=False)

    print("""
  Doc nhu sau. Gia mot canh sai la mot HANG SO CUA MODEL, khong doi khi doi
  mien tu vung. Cai doi la NGAN SACH: bo neo tu vung di thi do thi dung tro
  nen dang gia hon gap doi, nen cung mot con doc phai an het mot chieu cao lon
  hon truoc khi het lai.

  He qua thuc dung: tren tu vung quen thuoc, diem hoa von cua gpt-4.1 la 0.74
  canh - DUOI MOT. Chi mot canh dao chieu la do thi da lo. Tren tu vung xa la,
  cung model do chiu duoc 1.93 canh.

  Nghia la huong dan bang do thi BEN VUNG HON o dung noi no CAN THIET HON.""")


if __name__ == "__main__":
    main()

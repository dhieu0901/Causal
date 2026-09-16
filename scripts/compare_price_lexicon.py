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

k* is solved from the fitted line as (intercept - RAW) / price, NOT as
budget / price - the fitted intercept sits 1.8 to 2.2 pp below the measured
ORACLE point, so the two formulas differ by up to 2.7 edges. See
analyze_types.py:175 for the expression actually used.

Round 5 of the review panel rejected the reading that the budget doubles when
the lexical anchor is removed: all three paired difference CIs contain 0 and
gpt-4.1-nano moves the other way. REPORT.md section 8.3 records that
retraction. This script therefore reports the price contrast, which stands,
and prints the budget numbers without a conclusion attached.
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
                    "kstar_PSEUDO_DR": fmt(p),
                    # Carried in the file itself, not only in the printed text:
                    # anyone reading this CSV without the script must see that
                    # the contrast between the two budgets was retracted.
                    "canh_bao": "hieu ngan sach CHUA XAC LAP, ca ba CI chua 0, "
                                "nano di nguoc - xem REPORT.md muc 8.3"})
    o = pd.DataFrame(out)
    print(o.to_string(index=False))
    o.to_csv(ROOT / "results" / "breakeven_by_lexicon.csv", index=False)

    print("""
  CACH DOC, theo REPORT.md muc 8.3.

  VUNG: gia mot canh sai khong doi ro ret theo mien tu vung. 9/9 cap CI chong
  nhau, va CI tren HIEU ghep cap la -0,11 [-1,62 ; 1,42] pp - day moi la co so
  cua ket luan tuong duong, khong phai viec CI chong nhau.

  CHUA XAC LAP: khang dinh "ngan sach tang gap doi khi bo neo tu vung" da bi
  vong phan bien thu 5 bac bo. Ca ba CI tren hieu ngan sach deu chua 0, va
  gpt-4.1-nano di NGUOC huong - ngan sach cua no GIAM. Bang tren in ra de doi
  chieu, khong kem ket luan nao.

  Do do diem hoa von k* cung chua xac lap, vi tu so cua no chua vung. Khong
  duoc doc cap so 0,74 va 1,93 nhu mot quy luat.

  Chay scripts/analyze_budget_paired.py de co CI tren hieu ngan sach va co mau
  can thiet de chot.""")


if __name__ == "__main__":
    main()

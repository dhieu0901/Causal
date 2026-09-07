"""Reproduce every feasibility number in REPORT.md.

    python scripts/feasibility.py

Writes CSVs to results/ and prints a summary. No model calls, no API key needed.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from perturb import FAMILY_STRUCTURE, feasibility, max_k
from stats import mcnemar_power, resolution, bootstrap_break_even

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# Anchors read from NoisyCausal arXiv:2605.04313v1
ORACLE, FULL, NOGRAPH, RANDOM = 85.0, 80.68, 65.32, 60.87
COST = {"ED": 3.3, "FE": 8.2, "DR": 11.8}          # pp per erroneous edge, Table 7


def section(t):
    print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def main():
    # ---- 1. break-even implied by the source paper -----------------------
    section("1. DIEM HOA VON SUY RA TU NoisyCausal (ngoai suy tuyen tinh)")
    budget = ORACLE - NOGRAPH
    be = {t: budget / c for t, c in COST.items()}
    df1 = pd.DataFrame([{"loai": t, "chi_phi_pp_moi_canh": c,
                         "hoa_von_tai_k": round(be[t], 2)} for t, c in COST.items()])
    print(f"ngan sach loi = {ORACLE} - {NOGRAPH} = {budget:.2f} pp")
    print(df1.to_string(index=False))
    df1.to_csv(OUT / "break_even_implied.csv", index=False)

    # ---- 2. is that k reachable on CLadder graphs? -----------------------
    section("2. SO DO THI k-LOI HOP LE TREN CLADDER (liet ke vet can)")
    feas = pd.DataFrame([feasibility(f, kmax=3) for f in FAMILY_STRUCTURE])
    feas = feas.sort_values(["edges", "graph_id"])
    print(feas.to_string(index=False))
    feas.to_csv(OUT / "perturbation_feasibility.csv", index=False)

    mx = pd.DataFrame([{"graph_id": f, **{t: max_k(f, t) for t in COST}}
                       for f in FAMILY_STRUCTURE]).sort_values("graph_id")
    print("\nk toi da moi ho:")
    print(mx.to_string(index=False))
    mx.to_csv(OUT / "max_k_per_family.csv", index=False)

    section("3. DIEM HOA VON CO NAM TRONG TAM VOI KHONG?")
    verdict = []
    for t in COST:
        reach = int(mx[t].max())
        verdict.append({"loai": t, "hoa_von_can_k": round(be[t], 2),
                        "k_toi_da_kha_thi": reach,
                        "ket_luan": "DO TOI" if reach >= be[t] else "KHONG DO TOI"})
    dv = pd.DataFrame(verdict)
    print(dv.to_string(index=False))
    dv.to_csv(OUT / "reachability.csv", index=False)

    # ---- 4. sample size ---------------------------------------------------
    section("4. DO PHAN GIAI k* THEO CO MAU (giai tich)")
    rows = [{"n": n, "loai": t, "phan_giai_+-canh": round(resolution(n, .70, c)["k_resolution"], 2)}
            for n in (200, 400, 800, 1500) for t, c in COST.items()]
    r4 = pd.DataFrame(rows).pivot(index="n", columns="loai", values="phan_giai_+-canh")
    print(r4.to_string())
    r4.to_csv(OUT / "resolution_by_n.csv")

    section("5. POWER McNEMAR (paired, rho=0.6, alpha=0.05)")
    rows = []
    for n in (200, 400, 800):
        r = {"n": n}
        for d in (3, 5, 8, 12):
            r[f"{d}pp"] = round(mcnemar_power(n, .70, .70 - d / 100, rho=.6,
                                              sims=1200, seed=n + d), 3)
        rows.append(r)
    r5 = pd.DataFrame(rows)
    print(r5.to_string(index=False))
    r5.to_csv(OUT / "mcnemar_power.csv", index=False)

    section("6. BOOTSTRAP k* - CO MAU NAO DU DE TACH '1 canh' VOI '2 canh'?")
    truth = {k: (ORACLE - COST["DR"] * k) / 100 for k in range(4)}
    rows = []
    for n in (400, 800, 1200):
        ws = []
        for s in range(4):
            rng = np.random.default_rng(100 + s * 13 + n)
            u = rng.random(n)
            cbk = {k: (u < p).astype(int) for k, p in truth.items()}
            fl = (rng.random(n) < NOGRAPH / 100).astype(int)
            b = bootstrap_break_even(cbk, fl, reps=2000, seed=s)
            ws.append((b["k_star"], b["lo"], b["hi"]))
        lo, hi = np.mean([w[1] for w in ws]), np.mean([w[2] for w in ws])
        rows.append({"n": n, "k*": round(np.mean([w[0] for w in ws]), 2),
                     "CI_lo": round(lo, 2), "CI_hi": round(hi, 2),
                     "rong_CI": round(hi - lo, 2),
                     "tach_duoc": "CO" if hi - lo < 1.0 else "khong"})
    r6 = pd.DataFrame(rows)
    print(r6.to_string(index=False))
    r6.to_csv(OUT / "bootstrap_sample_size.csv", index=False)

    # ---- 7. corpus + cost -------------------------------------------------
    section("7. KHO DU LIEU VA CHI PHI")
    d = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv")
    ok3 = [f for f in FAMILY_STRUCTURE if max_k(f, "DR") >= 3]
    pool = d[d.graph_id.isin(ok3)]
    print(f"CLadder v1.5      : {len(d):,} cau hoi, 10 ho do thi")
    print(f"Pool nhanh DR k<=3: {len(pool):,} cau hoi ({', '.join(sorted(ok3))})")
    print(f"Du cho n=800      : {'CO' if len(pool) >= 800 else 'KHONG'}")

    in_tok = int(d['prompt'].str.len().mean() / 3.8) + 120
    plan = {"RAW": 800, "ORACLE": 800, "DR_k1": 800, "DR_k2": 800, "DR_k3": 800,
            "ED_k1": 400, "ED_k2": 400, "ED_k3": 400, "FE_k1": 400, "FE_k2": 400,
            "INDUCED": 800, "S2_sampling_5x": 4000}
    calls = sum(plan.values())
    tin, tout = calls * in_tok / 1e6, calls * 400 / 1e6
    print(f"\nLuot goi / model  : {calls:,}   ({tin:.2f}M token vao, {tout:.2f}M ra)")
    rows = [{"model": m, "usd_1_model": round(tin * pi + tout * po, 2),
             "usd_3_model_2_nhanh": round((tin * pi + tout * po) * 6, 2)}
            for m, pi, po in [("gpt-4o-mini", .15, .60), ("gpt-4.1-mini", .40, 1.60),
                              ("gpt-4o", 2.50, 10.00)]]
    r7 = pd.DataFrame(rows)
    print(r7.to_string(index=False))
    r7.to_csv(OUT / "cost_estimate.csv", index=False)

    print(f"\n\nDa ghi {len(list(OUT.glob('*.csv')))} file CSV vao {OUT}")


if __name__ == "__main__":
    main()

"""Price each error type, then check whether those prices explain what a real
induced graph costs.

    python scripts/analyze_types.py

The break-even framing assumed all graph errors are alike. The induction run says
they are not: agents almost never reverse an edge (0.01-0.05 per item) but omit
about two. So the question becomes:

  1. What does each error type cost per edge?
  2. Which types do agents actually commit?
  3. Do (1) x (2) predict the observed cost of reasoning over an induced graph?

If the prediction holds, the cost of graph induction is decomposable, and an
agent can be told which of its own mistakes are worth worrying about.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from stats import mcnemar_exact_p

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["ED", "FE", "DR"]
LABEL = {"ED": "thieu canh", "FE": "thua canh", "DR": "dao chieu"}


def paired(df, model):
    return df[df.model == model].pivot_table(
        index="item", columns="cond", values="correct", aggfunc="first")


def fit_of(acc, t, kmax=3):
    """Degradation line for error type `t`, fitted over k = 0..kmax.

    Returns (slope_pp_per_edge, intercept, r2, k_measured, monotone).

    The intercept is fitted rather than pinned to ORACLE, and the break-even is
    then solved against that same fitted line. The earlier version fitted a free
    intercept but computed break-even as (ORACLE - RAW) / slope, which pins the
    line to ORACLE - two different lines in one formula. On nano's DR that alone
    moved k* from 0.96 to 0.51.

    `r2` and `monotone` are reported because the fit is not always meaningful.
    FE has only three points (k = 0, 1, 2) and its accuracy is not monotone in k
    for any model, so its slope is an artefact of which two points dominate.
    """
    ks, ys = [0], [acc["ORACLE"]]
    for k in range(1, kmax + 1):
        c = f"{t}_k{k}"
        if c in acc.index and pd.notna(acc[c]):
            ks.append(k)
            ys.append(acc[c])
    if len(ks) < 2:
        return np.nan, np.nan, np.nan, len(ks) - 1, None
    ks, ys = np.array(ks, float), np.array(ys, float)
    b, a0 = np.polyfit(ks, ys, 1)
    resid = ys - np.polyval([b, a0], ks)
    ss_tot = float(((ys - ys.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else np.nan
    monotone = bool(np.all(np.diff(ys) <= 1e-9))
    return -b, a0, r2, int(ks.max()), monotone


def bootstrap_fit(df_model, t, kmax=3, boot=600, seed=0, base="RAW"):
    """Percentile CIs for the slope and for k*, resampling items.

    Every condition is answered by the same items, so the curve and the RAW
    floor move together under a resample; resampling them independently would
    break that correlation and widen the CI for no reason.

    The slope CI is what decides whether a per-edge price exists at all. Each
    accuracy in the fit carries a standard error near 3.5 pp at this sample
    size, and the fit has three or four points, so a slope can look clean and
    still be indistinguishable from zero. Judging the fit by whether accuracy
    happens to fall monotonically in k is far too strict - a 0.5 pp step up is
    noise, not counter-evidence - and R2 alone says nothing about scale.
    """
    rng = np.random.default_rng(seed)
    wide = df_model.pivot_table(index="item", columns="cond",
                                values="correct", aggfunc="first")
    need = ["ORACLE", base] + [f"{t}_k{k}" for k in range(1, kmax + 1)]
    wide = wide[[c for c in need if c in wide.columns]].dropna()
    if len(wide) < 20 or "ORACLE" not in wide or base not in wide:
        return {}
    idx = np.arange(len(wide))
    slopes, kstars = [], []
    for _ in range(boot):
        s = wide.iloc[rng.choice(idx, len(idx), replace=True)]
        acc_b = s.mean() * 100
        slope, a0, _, _, _ = fit_of(acc_b, t, kmax)
        if slope is None or np.isnan(slope):
            continue
        slopes.append(slope)
        if slope > 0:
            kstars.append((a0 - acc_b[base]) / slope)
    if len(slopes) < boot // 4:
        return {}
    q = lambda v, p: float(np.percentile(v, p))
    out = {"slope_lo": q(slopes, 2.5), "slope_hi": q(slopes, 97.5)}
    # k* is only meaningful while the slope is reliably negative-going; when the
    # slope CI spans zero the break-even runs off to infinity and quoting a
    # percentile of it would invent precision.
    if len(kstars) >= boot // 4 and out["slope_lo"] > 0:
        out |= {"k_lo": q(kstars, 2.5), "k_hi": q(kstars, 97.5)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", default="pilot_raw.csv")
    ap.add_argument("--induction", default="induction_raw.csv")
    ap.add_argument("--tag", default="")
    ap.add_argument("--baseline", default="RAW", choices=["RAW", "RAW_INSTR"],
                    help="nen de tinh ngan sach va diem hoa von. RAW la nen cu; "
                         "RAW_INSTR tru bo phan hieu ung cau lenh ra khoi tu so, "
                         "vi ORACLE cong vao CA khoi do thi LAN cau lenh - xem "
                         "scripts/analyze_instruction.py")
    a = ap.parse_args()
    base = a.baseline

    df = pd.read_csv(ROOT / "results" / a.pilot)
    models = [m for m in TIER if m in set(df.model)] or sorted(df.model.unique())

    # Scoring an unparseable reply as wrong conflates "would not answer" with
    # "answered wrongly", and penalises whichever condition is hardest to answer
    # at all. Parse rates here range 92.5-99.3%, and the gap flipped the sign of
    # nano's Delta_struct (-0.68 all-items vs +2.22 parsed-only). So the parsed
    # subset is primary and the parse rate is reported as its own outcome.
    parsed = df[df.parsed == 1]
    acc = (parsed.groupby(["model", "cond"]).correct.mean().unstack() * 100).round(2)
    acc_all = (df.groupby(["model", "cond"]).correct.mean().unstack() * 100).round(2)
    prate = (df.groupby(["model", "cond"]).parsed.mean().unstack() * 100).round(1)

    print("=" * 80)
    print("1. DO CHINH XAC THEO TUNG LOAI LOI  (chi tren cau parse duoc)")
    print("=" * 80)
    order = ["PROSE", "RAW", "ORACLE"] + [f"{t}_k{k}" for t in TYPES for k in (1, 2, 3)]
    order = [c for c in order if c in acc.columns]
    print(acc.reindex(index=models, columns=order).to_string())
    print("\n-- parse rate (%) --")
    print(prate.reindex(index=models, columns=order).to_string())
    print("\n-- de doi chieu: cham ca cau khong parse duoc thanh sai --")
    print(acc_all.reindex(index=models, columns=order).to_string())
    acc.reindex(index=models, columns=order).to_csv(
        ROOT / "results" / f"types_accuracy{a.tag}.csv")
    prate.reindex(index=models, columns=order).to_csv(
        ROOT / "results" / f"types_parserate{a.tag}.csv")

    print("\n" + "=" * 80)
    print("2. GIA MOI CANH LOI, VA DIEM HOA VON SO VOI DUONG SAN RAW")
    print("=" * 80)
    rows = []
    for m in models:
        r = acc.loc[m]
        budget = r["ORACLE"] - r[base]
        for t in TYPES:
            s, a0, r2, kmax_seen, _ = fit_of(r, t)
            ok = bool(s and s > 0)
            bs = bootstrap_fit(parsed[parsed.model == m], t, base=base)
            # The price stands only if the slope CI clears zero. Everything
            # downstream - the break-even, the additive prediction - is built on
            # that slope, so an unresolved slope has to stop here.
            solid = bool(bs) and bs["slope_lo"] > 0
            rows.append({
                "model": m, "loai": f"{t} ({LABEL[t]})",
                "gia_pp_moi_canh": round(s, 2) if ok else None,
                "gia_lo": round(bs["slope_lo"], 2) if bs else None,
                "gia_hi": round(bs["slope_hi"], 2) if bs else None,
                "r2": round(r2, 2) if pd.notna(r2) else None,
                "k_do_toi": kmax_seen,
                "ngan_sach_pp": round(budget, 2),
                "chan_fit": round(a0, 2) if pd.notna(a0) else None,
                # Solved against the fitted line, not against ORACLE, so the
                # break-even and the curve it comes from are the same line.
                "hoa_von_k": round((a0 - r[base]) / s, 2) if ok and solid else None,
                "hoa_von_lo": round(bs["k_lo"], 2) if solid and "k_lo" in bs else None,
                "hoa_von_hi": round(bs["k_hi"], 2) if solid and "k_hi" in bs else None,
                "canh_bao": "" if solid else "CI do doc chua tach khoi 0",
            })
    pr = pd.DataFrame(rows)
    print(pr.to_string(index=False))
    print("\n  hoa_von_k giai tu chinh duong fit: (chan_fit - RAW) / gia.")
    print("  gia_lo/gia_hi = CI 95% bootstrap 600 lan, boc lai theo item.")
    print("  canh_bao khi CI do doc con chua 0: khong co gia moi canh nao duoc xac lap,")
    print("  nen hoa_von_k de trong thay vi bao mot con so khong co co so.")
    pr.to_csv(ROOT / "results" / f"types_price{a.tag}.csv", index=False)

    print("\n" + "=" * 80)
    print("3. McNEMAR: tung dieu kien so voi RAW")
    print("=" * 80)
    sig = []
    for m in models:
        s = paired(parsed, m)
        for c in order:
            if c in ("PROSE", "RAW") or c not in s:
                continue
            x, y = s[c].dropna(), s["RAW"].dropna()
            i = x.index.intersection(y.index)
            nb = int(((x[i] == 1) & (y[i] == 0)).sum())
            nc = int(((x[i] == 0) & (y[i] == 1)).sum())
            p = mcnemar_exact_p(nb, nc)
            delta = float(acc.loc[m, c]) - float(acc.loc[m, "RAW"])
            sig.append({"model": m, "dieu_kien": c, "n": len(i),
                        "delta_pp": round(delta, 2),
                        "p": round(p, 4), "y_nghia": "*" if p < .05 else ""})
    sg = pd.DataFrame(sig)
    print(sg.to_string(index=False))
    sg.to_csv(ROOT / "results" / f"types_mcnemar{a.tag}.csv", index=False)

    # ---- 4. do the prices explain what induction actually costs? -----------
    ipath = ROOT / "results" / a.induction
    if not ipath.exists():
        print(f"\n(chua co {ipath.name} - bo qua muc 4)")
        return

    ind = pd.read_csv(ipath)

    # The induction file has to come from the same items and the same lexicon as
    # the pilot file, because section 4 subtracts one from the other. The default
    # name does not track --pilot, so pricing a PSEUDO run at n=400 against the
    # default KEEP induction at n=147 joins two unrelated samples and returns a
    # negative "observed loss" that looks like a finding. Item overlap is the
    # cheapest thing that catches it.
    shared = set(ind.item) & set(df.item)
    cover = len(shared) / max(1, len(set(df.item)))
    if cover < 0.9:
        print(f"\n[BO QUA MUC 4] {ipath.name} chi trung {100*cover:.0f}% item voi "
              f"{a.pilot}. Hai file nay khong cung mau, tru nhau se ra so vo nghia.")
        print(f"  Chay induction.py voi cung --n, --lexicon va --tag roi truyen "
              f"--induction cho khop.")
        return
    print("\n" + "=" * 80)
    print("4. GIA x HO SO LOI THUC TE  ->  CO DU DOAN DUOC CHI PHI INDUCTION KHONG?")
    print("=" * 80)
    # A price whose own fit was flagged weak is not a price. Feeding it into the
    # additive prediction would hide the uncertainty inside a number that then
    # gets compared to the observed loss as if both were solid.
    price = {(r.model, str(r.loai).split()[0]): (None if r.canh_bao else r.gia_pp_moi_canh)
             for r in pr.itertuples()}
    out = []
    for m in models:
        sub = ind[ind.model == m]
        if sub.empty:
            continue
        counts = {"ED": sub.n_missing.mean(), "FE": sub.n_spurious.mean(),
                  "DR": sub.n_reversed.mean()}
        # `nan or 0` yields nan, not 0, because nan is truthy - so one unpriced
        # type would silently blank the whole prediction instead of narrowing it.
        def priced(t):
            v = price.get((m, t))
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else v

        unpriced = [t for t in TYPES if priced(t) is None]
        pred_loss = sum((priced(t) or 0.0) * counts[t] for t in TYPES)
        # match the parsed-only basis used for acc, or the comparison is unfair
        sub_ok = sub[sub.parsed_answer == 1] if "parsed_answer" in sub else sub
        obs = acc.loc[m, "ORACLE"] - sub_ok.correct.mean() * 100
        out.append({
            "model": m,
            **{f"so_canh_{t}": round(counts[t], 2) for t in TYPES},
            **{f"gia_{t}": price.get((m, t)) for t in TYPES},
            "du_doan_mat_pp": round(pred_loss, 2),
            "thuc_te_mat_pp": round(obs, 2),
            "lech": round(obs - pred_loss, 2),
            "chua_dinh_gia": ",".join(unpriced) or "-",
        })
    od = pd.DataFrame(out)
    print(od.to_string(index=False))
    od.to_csv(ROOT / "results" / f"types_prediction{a.tag}.csv", index=False)

    print("\n  du_doan_mat_pp = tong(gia moi canh x so canh loi loai do) tren do thi induced")
    print("  thuc_te_mat_pp = ORACLE - INDUCED")
    print("  lech nho  -> chi phi do thi phan ra duoc theo loai loi")
    print("  lech lon  -> con co che khac dang chi phoi")


if __name__ == "__main__":
    main()

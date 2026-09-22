"""Does the per-edge price of a wrong graph change when the variable names go?

    python scripts/analyze_price_paired.py

REPORT.md section 8.1 answers "no" and backs it with nine confidence intervals
on the difference between two slopes. Until 2026-09-23 not one of those nine
existed anywhere but a hardcoded print string in compare_price_lexicon.py: no
function resampled anything, and the section carried its own warning saying so.
The point estimates could be checked by differencing two published slopes; the
nine intervals could not be checked at all. This file computes them.

Why the difference needs its own bootstrap. Section 8.1's first version argued
from "9/9 pairs of CIs overlap". Two overlapping intervals do not show two
quantities are equal - they show the comparison was never made. The quantity in
question is the DIFFERENCE, so the difference is what gets resampled.

Why the resample is paired. The KEEP and PSEUDO ladders were run on the SAME
items, so the two slopes move together: an item that happens to be easy pulls
both branches up. Resampling the branches independently would throw that
correlation away and inflate the interval for no reason. Each draw therefore
picks one item set and scores BOTH branches on it.

What this can and cannot settle. Each accuracy entering a fit carries a standard
error near 3.5 pp, each fit has three or four points, and the difference of two
such slopes is noisier still. A wide interval here is the honest output, not a
failure - but it means "we cannot tell them apart", never "they are the same".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from stats import boot_p

from analyze_types import fit_of

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
TYPES = ["DR", "ED", "FE"]
SEED = 20260907
NBOOT = 4000
KMAX = 3
MIN_ITEMS = 20


def wide_of(df, model, t, kmax=KMAX, base="RAW"):
    """One row per item, one column per condition, only complete rows."""
    d = df[df.model == model]
    w = d.pivot_table(index="item", columns="cond", values="correct",
                      aggfunc="first")
    need = ["ORACLE", base] + [f"{t}_k{k}" for k in range(1, kmax + 1)]
    have = [c for c in need if c in w.columns]
    if "ORACLE" not in have:
        return None
    return w[have].dropna()


def slope_of(w, t, kmax=KMAX):
    """pp of accuracy lost per corrupted edge, positive = costly."""
    s, _, _, _, _ = fit_of(w.mean() * 100, t, kmax)
    return s


def paired_difference(wk, wp, t, seed=SEED, n=NBOOT):
    """Bootstrap (slope_KEEP - slope_PSEUDO) over a shared item resample."""
    idx = wk.index.intersection(wp.index)
    if len(idx) < MIN_ITEMS:
        return None
    wk, wp = wk.loc[idx], wp.loc[idx]
    est = slope_of(wk, t) - slope_of(wp, t)
    if not np.isfinite(est):
        return None

    rng = np.random.default_rng(seed)
    pos = np.arange(len(idx))
    out = []
    for _ in range(n):
        # ONE draw, BOTH branches. This is the whole point of the file.
        take = rng.choice(pos, len(pos), replace=True)
        a = slope_of(wk.iloc[take], t)
        b = slope_of(wp.iloc[take], t)
        if np.isfinite(a) and np.isfinite(b):
            out.append(a - b)
    if len(out) < n // 4:
        return None
    out = np.asarray(out)
    return {"estimate_pp": float(est),
            "ci_lo": float(np.percentile(out, 2.5)),
            "ci_hi": float(np.percentile(out, 97.5)),
            "p_boot": boot_p(out),
            "n_items": int(len(idx)),
            "n_draws_used": int(len(out))}


def main():
    k_raw = ROOT / "results" / "pilot_raw_price400KEEP.csv"
    p_raw = ROOT / "results" / "pilot_raw_price400PSEUDO.csv"
    for f in (k_raw, p_raw):
        if not f.exists():
            raise SystemExit(f"thieu {f.name}")
    K = pd.read_csv(k_raw)
    P = pd.read_csv(p_raw)
    K = K[K.parsed == 1] if "parsed" in K.columns else K
    P = P[P.parsed == 1] if "parsed" in P.columns else P

    models = [m for m in TIER if m in set(K.model) and m in set(P.model)]

    print("=" * 84)
    print("PRICE DIFFERENCE, KEEP MINUS PSEUDO - bootstrapped on a SHARED resample")
    print("=" * 84)
    print(f"\n  {NBOOT} draws, seed {SEED}, items clustered.")
    print("  Positive = the wrong edge costs MORE when the real names are kept.\n")
    print(f"  {'model':14s} {'type':5s} {'hieu':>8s} {'CI 95%':>20s} "
          f"{'p':>8s} {'n':>5s}")
    print("  " + "-" * 68)

    rows = []
    for m in models:
        for t in TYPES:
            wk, wp = wide_of(K, m, t), wide_of(P, m, t)
            if wk is None or wp is None:
                print(f"  {m:14s} {t:5s}  thieu dieu kien")
                continue
            r = paired_difference(wk, wp, t)
            if r is None:
                print(f"  {m:14s} {t:5s}  qua it item de bootstrap")
                continue
            star = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
            print(f"  {m:14s} {t:5s} {r['estimate_pp']:+8.2f} "
                  f"[{r['ci_lo']:+7.2f} ; {r['ci_hi']:+7.2f}] "
                  f"{r['p_boot']:8.4f} {r['n_items']:5d} {star}")
            rows.append({"model": m, "error_type": t,
                         "estimate_pp": round(r["estimate_pp"], 2),
                         "ci_lo": round(r["ci_lo"], 2),
                         "ci_hi": round(r["ci_hi"], 2),
                         "p_boot": round(r["p_boot"], 4),
                         "n_items": r["n_items"],
                         "separates_from_zero": bool(r["ci_lo"] > 0 or r["ci_hi"] < 0)})

    if not rows:
        raise SystemExit("khong tinh duoc o nao")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "price_paired_difference.csv", index=False)

    n_sep = int(out.separates_from_zero.sum())
    widest = out.assign(w=out.ci_hi - out.ci_lo).sort_values("w").iloc[-1]
    tightest = out.assign(w=out.ci_hi - out.ci_lo).sort_values("w").iloc[0]

    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print(f"\n  {n_sep}/{len(out)} o tach khoi 0.")
    print(f"  Hep nhat: {tightest.model} / {tightest.error_type} "
          f"[{tightest.ci_lo:+.2f} ; {tightest.ci_hi:+.2f}]")
    print(f"  Rong nhat: {widest.model} / {widest.error_type} "
          f"[{widest.ci_lo:+.2f} ; {widest.ci_hi:+.2f}]")
    print("\n  An interval that contains 0 says the two prices cannot be told apart")
    print("  at this sample size. It does NOT say they are equal, and the widest")
    print("  interval above is the honest bound on how large a real difference")
    print("  could be hiding here. Quote that bound, not the word 'unchanged'.")
    print("\n  Ghi ra results/price_paired_difference.csv")


if __name__ == "__main__":
    main()

"""Statistics for the break-even study: McNemar power, bootstrap CI for k*."""
from __future__ import annotations
import numpy as np
from scipy import stats


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on the discordant cells."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5)))


def mcnemar_power(n, p_a, p_b, rho=0.6, alpha=0.05, sims=4000, seed=0):
    """Power of the paired test at sample size n.

    rho is the correlation between the two conditions' per-item correctness;
    paired designs on the same items are strongly correlated, which is exactly
    why the paired test buys power over two independent samples.
    """
    rng = np.random.default_rng(seed)
    # Gaussian copula -> correlated Bernoulli pair with the requested marginals
    cov = np.array([[1.0, rho], [rho, 1.0]])
    L = np.linalg.cholesky(cov)
    za, zb = stats.norm.ppf(p_a), stats.norm.ppf(p_b)
    hits = 0
    for _ in range(sims):
        z = rng.standard_normal((n, 2)) @ L.T
        a, b = z[:, 0] < za, z[:, 1] < zb
        n01 = int(np.sum(a & ~b))
        n10 = int(np.sum(~a & b))
        if mcnemar_exact_p(n01, n10) < alpha:
            hits += 1
    return hits / sims


def break_even_k(ks, accs, floor):
    """Linear interpolation of where the accuracy curve crosses the floor."""
    ks, accs = np.asarray(ks, float), np.asarray(accs, float)
    for i in range(len(ks) - 1):
        a0, a1 = accs[i], accs[i + 1]
        if (a0 - floor) * (a1 - floor) <= 0 and a0 != a1:
            return float(ks[i] + (a0 - floor) / (a0 - a1) * (ks[i + 1] - ks[i]))
    if accs[-1] > floor:                      # never crosses inside the measured range
        slope = (accs[0] - accs[-1]) / (ks[-1] - ks[0])
        return float(ks[-1] + (accs[-1] - floor) / slope) if slope > 0 else np.nan
    return float(ks[0])


def bootstrap_break_even(correct_by_k, floor_correct, reps=10000, seed=0):
    """Percentile CI for k*, resampling items (not observations) to keep pairing.

    correct_by_k: dict k -> 0/1 array over the SAME items, in the same order.
    floor_correct: 0/1 array over those items for the no-graph condition.
    """
    rng = np.random.default_rng(seed)
    ks = sorted(correct_by_k)
    mats = np.stack([np.asarray(correct_by_k[k], float) for k in ks])
    floor = np.asarray(floor_correct, float)
    n = mats.shape[1]
    out = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        out[r] = break_even_k(ks, mats[:, idx].mean(axis=1) * 100, floor[idx].mean() * 100)
    out = out[np.isfinite(out)]
    return {
        "k_star": float(break_even_k(ks, mats.mean(axis=1) * 100, floor.mean() * 100)),
        "lo": float(np.percentile(out, 2.5)),
        "hi": float(np.percentile(out, 97.5)),
        "n_valid": int(out.size),
    }


def resolution(n, p=0.70, slope_pp_per_edge=11.8, alpha=0.05):
    """How tightly a sample of n localises k*, given the curve's slope."""
    se = np.sqrt(p * (1 - p) / n) * 100
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {"se_pp": se, "ci_half_pp": half, "k_resolution": half / slope_pp_per_edge}


def boot_p(draws, n_draws=None) -> float:
    """Two-tailed bootstrap p from a set of resampled estimates.

    One definition, used by every bootstrap in this repository, because three
    different conventions were in circulation until 2026-09-23 and each one put
    a different number in a shipped CSV for the same situation:

      clamp   p is 2 * min(left tail, right tail), and a draw landing exactly on
              zero is counted by BOTH tails, so the product can exceed 1 near a
              null effect. results/family_breakdown.csv shipped p = 1.0255.

      floor   with B draws the smallest non-zero two-tailed p that this
              estimator can express is 2/B, so that is the floor. Two scripts
              had no floor at all and published p = 0.0 - see the headline row
              of results/ladder5_steps.csv, +17.67 pp at "p = 0". A bootstrap
              never licenses zero; it licenses "below 2/B". One script floored
              at 1/B instead, so the same situation printed 0.00025 there and
              0.0005 everywhere else.

    Takes the draws rather than a precomputed p so the tail counting cannot
    drift between call sites either. Pass n_draws only when the array has
    already been filtered and the floor should reflect the draws attempted.
    """
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    b = int(len(d)) if n_draws is None else int(n_draws)
    if b < 1 or d.size == 0:
        return float("nan")
    p = 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean()))
    return min(1.0, max(p, 2.0 / b))


def cluster_boot(n_units, stat, seed, n_draws):
    """Resample unit POSITIONS with replacement; apply `stat` to each draw.

    One resampling mechanism for the whole repository. Eleven bootstraps were
    written separately, nine drawing with `rng.integers(0, n, n)` and eight with
    `rng.choice(arr, n, replace=True)`. Those two are not obviously the same
    call, so the divergence looked like it might carry a cost to fix.

    It does not. Given one freshly seeded Generator the two consume the stream
    identically - verified on arange, on a non-contiguous id array and on a
    pandas Index - so routing every caller through this function leaves every
    published number exactly where it was. That is why this is a cleanup and
    not a reanalysis.

    What is deliberately NOT unified is the STATISTIC. pool_samples averages
    over every (item, cell) pair while analyze_vs_raw averages per item first;
    those are different estimators on purpose, and its docstring says so.
    Collapsing them would silently change results under the banner of tidying
    up. `stat` therefore stays with the caller: it receives an array of
    positions into that caller's own units and returns one number.
    """
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws, dtype=float)
    for b in range(n_draws):
        out[b] = stat(rng.integers(0, n_units, n_units))
    return out


def boot_interval(point, draws, n_draws=None):
    """(estimate, lo, hi, p) from a point estimate and its bootstrap replicates."""
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return point, float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(d, [2.5, 97.5])
    return point, float(lo), float(hi), boot_p(d, n_draws or len(d))

"""Does a second model family show what GPT-4.1 shows? B1 and B2 of docs/plan.md.

    python scripts/analyze_second_family.py

Llama 3.3 70B Instruct (scripts/run_llama70b.sh) was sent the very prompts
GPT-4.1 was sent, on all three samples, so every contrast here is computed the
way the GPT-4.1 number was, on the same items:

  1. HEADLINE   the pooled DiD of pool_samples.py, [(KEEP - PSEUDO) | RAW] -
                [(KEEP - PSEUDO) | ORACLE], causal group, de-duplicated by
                CLadder id, convention B (src/stats.py); then Llama minus the
                GPT-4.1 family, paired by item. The GPT-4.1 rows must reproduce
                results/cladder/pooled_headline.csv exactly, or the script stops
  2. ARMS       ORACLE, DR_k1, PROSE minus RAW, and ORACLE minus DR_k1, per
                lexicon, pooled over the three samples
  3. R1         DeepSeek-R1 (scripts/run_r1.sh) on the lex causal items: does a
                model that reasons at length still gain from being handed the
                graph? Against GPT-4.1 and Llama on the same items
  4. LADDER     the five-rung lexical ladder of analyze_ladder5.py on n600, RAW,
                causal items, for Llama: is what anonymisation takes away
                mostly the correct prior in this family too?
  5. RUNS       parse rate, answers cut off by the token cap, answers read from
                the reasoning field, billed cost - all from the records

CUT-OFFS. An answer the token cap cut off is unparsed and drops out of every
pair. Llama was capped at 700 tokens like GPT-4.1, and 4 to 7% of its answers
ran longer, more under RAW than under PROSE; R1 was capped at 8,000 and was
cut off only when handed a graph. Every cut-off was asked again with a larger
cap (pilot.py --recap: 1,500 for Llama, 16,000 for R1) and each headline
contrast is reported both ways.

Writes: results/cladder/second_family.csv, results/cladder/second_family_runs.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from stats import boot_cells, boot_items
from analyze_querygroup import ARITH, IDENT, TIER
from pool_samples import attach_id, cell

RES = ROOT / "results" / "cladder"
SEED, NBOOT = 20260907, 4000
LLAMA = "meta-llama/llama-3.3-70b-instruct"
R1 = "deepseek/deepseek-r1"
SAMPLES = ("lex", "n600", "price400")
# results/cladder/raw/pilot_raw_{prefix}{lexicon}.csv
PREFIX = {"gpt": {"lex": "lex", "n600": "n600", "price400": "price400"},
          "llama": {"lex": "llama70b", "n600": "llama70b_n600",
                    "price400": "llama70b_price400"},
          "r1": {"lex": "r1"},
          "llama ladder": {"n600": "llama70b_n600ladder"}}
MODELS = {"gpt": TIER, "llama": [LLAMA], "r1": [R1]}
RECAP = {"llama": 1500, "r1": 16000, "llama ladder": 1500}   # pilot.py --recap


def exists(fam, sample, lex, recap=False):
    f = f"pilot_raw_{PREFIX[fam][sample]}{lex}"
    return ((RES / "raw" / f"{f}.csv").exists()
            and (not recap or (RES / "raw" / f"{f}_recap{RECAP[fam]}.csv").exists()))


def load(fam, sample, lex, recap=False):
    """One results file; with `recap`, every cut-off row replaced by its re-ask."""
    imap = pd.read_csv(RES / f"_itemmap_{sample}.csv")     # written by pool_samples.py
    f = f"pilot_raw_{PREFIX[fam][sample]}{lex}"
    d = pd.read_csv(RES / "raw" / f"{f}.csv")
    if recap:
        r = pd.read_csv(RES / "raw" / f"{f}_recap{RECAP[fam]}.csv")
        key = ["model", "item", "cond"]
        if len(r):
            cut = set(map(tuple, d.loc[d.finish == "length", key].values))
            if set(map(tuple, r[key].values)) != cut:
                raise SystemExit(f"{f}: the re-asked rows are not exactly the cut-offs")
            d = pd.concat([d[~d.set_index(key).index.isin(list(cut))], r],
                          ignore_index=True)
    return attach_id(d, imap, f"{fam}/{sample}/{lex}")


def causal(d):
    return set(d.query_type.unique()) - ARITH - IDENT


def row(section, model, sample, lexicon, quantity, r, n):
    return {"section": section, "model": model, "sample": sample, "lexicon": lexicon,
            "quantity": quantity, "estimate_pp": round(r[0], 2), "ci_lo": round(r[1], 2),
            "ci_hi": round(r[2], 2), "p_boot": round(r[3], 4), "n_items": n}


def did_W(fam, sample, models, recap=False):
    """Per-id DiD, one column per model: pool_samples.did_sample for any family."""
    K, P = load(fam, sample, "KEEP", recap), load(fam, sample, "PSEUDO", recap)
    qs = causal(K)
    cols = []
    for m in models:
        kr, lr = cell(K, m, "RAW", qs), cell(P, m, "RAW", qs)
        ka, la = cell(K, m, "ORACLE", qs), cell(P, m, "ORACLE", qs)
        i = kr.index.intersection(lr.index).intersection(ka.index).intersection(la.index)
        if len(i) >= 10:
            cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values, index=i,
                                  name=f"{sample}|{m}"))
    # groupby sorts by id, exactly as pool_samples.did_sample does: the bootstrap
    # resamples positions, so the same items in another order give another
    # interval from the same seed (the GPT-4.1 check below caught it).
    return pd.concat(cols, axis=1).groupby(level=0).mean() if cols else pd.DataFrame()


def headline(rows):
    print("=" * 78)
    print("1. THE HEADLINE DiD, GPT-4.1 AGAINST LLAMA 3.3 70B, SAME ITEMS")
    print("=" * 78)
    print("  [(KEEP - PSEUDO) | RAW] - [(KEEP - PSEUDO) | ORACLE], causal group\n")
    ref = pd.read_csv(RES / "pooled_headline.csv").set_index("pooling")
    per = {f: {} for f in ("gpt", "gpt-4.1", "llama", "llama recap")}
    for s in SAMPLES:
        if not (exists("llama", s, "KEEP") and exists("llama", s, "PSEUDO")):
            print(f"  {s}: no Llama run yet")
            continue
        per["gpt"][s] = did_W("gpt", s, TIER)
        per["gpt-4.1"][s] = did_W("gpt", s, ["gpt-4.1"])
        per["llama"][s] = did_W("llama", s, [LLAMA])
        if exists("llama", s, "KEEP", True) and exists("llama", s, "PSEUDO", True):
            per["llama recap"][s] = did_W("llama", s, [LLAMA], recap=True)
    for fam, label in (("gpt", "GPT-4.1 family"), ("gpt-4.1", "gpt-4.1 alone"),
                       ("llama", "Llama 3.3 70B"),
                       ("llama recap", "Llama, cut-offs re-asked")):
        for s, W in per[fam].items():
            r = boot_cells(W, SEED, NBOOT)
            if fam == "gpt":
                want = (ref.loc[s, "did_pp"], ref.loc[s, "ci_lo"], ref.loc[s, "ci_hi"])
                if tuple(round(x, 2) for x in r[:3]) != tuple(want):
                    raise SystemExit(f"GPT-4.1 {s}: {r[:3]} does not reproduce "
                                     f"pooled_headline.csv {want}; stopping")
            print(f"  {label:15s} {s:9s} {r[0]:+7.2f} [{r[1]:+7.2f} ; {r[2]:+7.2f}]  "
                  f"p={r[3]:.4f}  n={len(W)}")
            rows.append(row("1 headline", label, s, "PSEUDO", "DiD", r, len(W)))
        if len(per[fam]) == len(SAMPLES):
            W = pd.concat(per[fam].values(), axis=1)
            r = boot_cells(W, SEED, NBOOT)
            if fam == "gpt":
                g = ref.loc["item level, de-duplicated"]
                if tuple(round(x, 2) for x in r[:3]) != (g.did_pp, g.ci_lo, g.ci_hi):
                    raise SystemExit(f"GPT-4.1 pooled: {r[:3]} does not reproduce "
                                     f"pooled_headline.csv; stopping")
            print(f"  {label:15s} {'POOLED':9s} {r[0]:+7.2f} [{r[1]:+7.2f} ; {r[2]:+7.2f}]  "
                  f"p={r[3]:.4f}  n={len(W)}")
            rows.append(row("1 headline", label, "pooled", "PSEUDO", "DiD", r, len(W)))
        print()
    if len(per["llama"]) == len(SAMPLES):
        # Paired by item: Llama's DiD minus the mean of the GPT-4.1 cells.
        cols = []
        for s in SAMPLES:
            g, l_ = per["gpt"][s].mean(axis=1), per["llama"][s].iloc[:, 0]
            i = g.index.intersection(l_.index)
            cols.append((l_[i] - g[i]).rename(s))
        W = pd.concat(cols, axis=1)
        r = boot_cells(W, SEED, NBOOT)
        print(f"  Llama minus GPT-4.1 family, paired by item, pooled: {r[0]:+.2f} "
              f"[{r[1]:+.2f} ; {r[2]:+.2f}]  p={r[3]:.4f}  n={len(W)}")
        rows.append(row("1 headline", "Llama minus GPT-4.1 family", "pooled", "PSEUDO",
                        "DiD difference, paired", r, len(W)))


ARMS = [("ORACLE", "RAW"), ("DR_k1", "RAW"), ("PROSE", "RAW"), ("ORACLE", "DR_k1")]


def arms(rows):
    print("\n" + "=" * 78)
    print("2. EACH ARM AGAINST RAW, CAUSAL GROUP, THREE SAMPLES POOLED BY ID")
    print("=" * 78 + "\n")
    if not all(exists("llama", s, x) for s in SAMPLES for x in ("KEEP", "PSEUDO")):
        print("  waiting for every Llama run")
        return
    for lex in ("KEEP", "PSEUDO"):
        for a, b in ARMS:
            for fam, models, label in (("gpt", TIER, "GPT-4.1 family"),
                                       ("gpt", ["gpt-4.1"], "gpt-4.1 alone"),
                                       ("llama", [LLAMA], "Llama 3.3 70B")):
                by_s = {}
                for s in SAMPLES:
                    D = load(fam, s, lex)
                    qs = causal(D)
                    for m in models:
                        x, y = cell(D, m, a, qs), cell(D, m, b, qs)
                        i = x.index.intersection(y.index)
                        by_s.setdefault(s, []).append((x[i] - y[i]).rename(f"{s}|{m}"))
                # The bootstrap resamples positions, so the row order is part of
                # the interval. analyze_structure_arms.py pools two ways: its lift
                # and drag (ORACLE minus RAW) sort all cells by id at once, its
                # direct contrasts sort within each sample and then join. Each
                # arm here is built the way its published counterpart was, and
                # held to it below: one quantity must not carry two intervals.
                if (a, b) == ("ORACLE", "RAW"):
                    W = pd.concat([c for v in by_s.values() for c in v], axis=1)
                    W = W.groupby(level=0).mean()
                else:
                    W = pd.concat([pd.concat(v, axis=1).groupby(level=0).mean()
                                   for v in by_s.values()], axis=1)
                r = boot_cells(W, SEED, NBOOT)
                published = {("PSEUDO", "ORACLE", "RAW"): "lift, anonymised branch",
                             ("KEEP", "ORACLE", "RAW"): "drag, KEEP branch",
                             ("PSEUDO", "ORACLE", "DR_k1"): "PSEUDO branch, ORACLE minus DR_k1"}
                if fam == "gpt" and models == TIER and (lex, a, b) in published:
                    q = published[(lex, a, b)]
                    g = pd.read_csv(RES / "structure_arms.csv").set_index("quantity").loc[q]
                    if tuple(round(x, 2) for x in r[:3]) != (g.estimate_pp, g.ci_lo, g.ci_hi):
                        raise SystemExit(f"GPT-4.1 {q}: {r[:3]} does not reproduce "
                                         f"structure_arms.csv; stopping")
                print(f"  {lex:6s} {a + ' - ' + b:16s} {label:15s} {r[0]:+7.2f} "
                      f"[{r[1]:+7.2f} ; {r[2]:+7.2f}]  p={r[3]:.4f}  n={len(W)}")
                rows.append(row("2 arms", label, "pooled", lex, f"{a} minus {b}", r, len(W)))
        print()


def reasoning_model(rows):
    print("\n" + "=" * 78)
    print("3. DEEPSEEK-R1: DOES A MODEL THAT REASONS AT LENGTH STILL NEED THE GRAPH?")
    print("=" * 78 + "\n")
    lexes = [x for x in ("KEEP", "PSEUDO") if exists("r1", "lex", x)]
    if not lexes:
        print("  no R1 run yet")
        return
    variants = [(lx, False) for lx in lexes] + [(lx, True) for lx in lexes
                                                if exists("r1", "lex", lx, True)]
    for lex, recap in variants:
        tag = f"{lex}, cut-offs re-asked at {RECAP['r1']} tokens" if recap else lex
        R = load("r1", "lex", lex, recap)
        G = load("gpt", "lex", lex)
        L = (load("llama", "lex", lex, recap and exists("llama", "lex", lex, True))
             if exists("llama", "lex", lex) else None)
        qs = causal(R)
        conds = [c for c in ("PROSE", "RAW", "ORACLE", "DR_k1") if c in set(R.cond)]
        srcs = [(R1, R)] + ([(LLAMA, L)] if L is not None else []) + [(m, G) for m in TIER[::-1]]
        # Accuracy per condition, on the items R1 answered in every condition.
        ids = None
        for c in conds:
            s_ = cell(R, R1, c, qs).index
            ids = s_ if ids is None else ids.intersection(s_)
        print(f"  {tag}: accuracy on the {len(ids)} causal items R1 answered in every "
              f"condition ({', '.join(conds)})")
        for m, D in srcs:
            acc = [100 * cell(D, m, c, qs).reindex(ids).mean() for c in conds]
            print(f"    {m:34s} " + "  ".join(f"{c} {v:5.1f}" for c, v in zip(conds, acc)))
            nan = float("nan")
            for c, v in zip(conds, acc):
                rows.append(row("3 reasoning model", m, "lex causal", tag, f"accuracy {c}",
                                (v, nan, nan, nan), len(ids)))
        # With no graph at all, does reasoning at length buy R1 anything over a
        # model that does not? Paired by item, RAW only (review round 12, K2).
        rr = cell(R, R1, "RAW", qs)
        for m, D in srcs[1:]:
            g = cell(D, m, "RAW", qs)
            i = rr.index.intersection(g.index)
            r = boot_items((rr[i] - g[i]).values, SEED, NBOOT)
            print(f"    R1 minus {m:25s} at RAW {r[0]:+7.2f} [{r[1]:+7.2f} ; {r[2]:+7.2f}]  "
                  f"p={r[3]:.4f}  n={len(i)}")
            rows.append(row("3 reasoning model", f"R1 minus {m}", "lex causal", tag,
                            "accuracy at RAW", r, len(i)))
        print()
        for a, b in [p for p in ARMS if p[0] in conds and p[1] in conds]:
            x, y = cell(R, R1, a, qs), cell(R, R1, b, qs)
            i = x.index.intersection(y.index)
            for m, D in srcs:
                xm, ym = cell(D, m, a, qs).reindex(i), cell(D, m, b, qs).reindex(i)
                ok = xm.notna() & ym.notna()
                r = boot_items((xm - ym)[ok].values, SEED, NBOOT)
                print(f"    {a + ' - ' + b:16s} {m:34s} {r[0]:+7.2f} [{r[1]:+7.2f} ; "
                      f"{r[2]:+7.2f}]  p={r[3]:.4f}  n={int(ok.sum())}")
                rows.append(row("3 reasoning model", m, "lex causal", tag,
                                f"{a} minus {b}", r, int(ok.sum())))
            # R1 minus gpt-4.1, the same contrast, paired by item
            g = (cell(G, "gpt-4.1", a, qs) - cell(G, "gpt-4.1", b, qs)).reindex(i)
            dd = ((x[i] - y[i]) - g).dropna()
            r = boot_items(dd.values, SEED, NBOOT)
            print(f"    {'':16s} {'R1 minus gpt-4.1, paired':34s} {r[0]:+7.2f} [{r[1]:+7.2f} ; "
                  f"{r[2]:+7.2f}]  p={r[3]:.4f}  n={len(dd)}")
            rows.append(row("3 reasoning model", "R1 minus gpt-4.1", "lex causal", tag,
                            f"{a} minus {b}", r, len(dd)))
            # An answer the token cap cut off is unparsed and drops out above;
            # scored as wrong instead, it stays in.
            full = R[(R.model == R1) & R.query_type.isin(qs)]
            piv = full.pivot_table(index="id", columns="cond", values="correct")
            dd = (piv[a] - piv[b]).dropna()
            r = boot_items(dd.values, SEED, NBOOT)
            print(f"    {'':16s} {'R1, unparsed scored as wrong':34s} {r[0]:+7.2f} [{r[1]:+7.2f} ; "
                  f"{r[2]:+7.2f}]  p={r[3]:.4f}  n={len(dd)}")
            rows.append(row("3 reasoning model", "R1, unparsed scored as wrong", "lex causal",
                            tag, f"{a} minus {b}", r, len(dd)))
            # A third of R1's answers came back in the reasoning field, not the
            # answer (pilot.py reads the strict ANSWER line at its very end), and
            # that share differs by condition (RAW 38%, ORACLE 26%). The same
            # contrast on the items whose two answers both came back normally:
            xc = full[(full.cond == a) & (full.parsed == 1)].set_index("id")
            yc = full[(full.cond == b) & (full.parsed == 1)].set_index("id")
            j = xc.index.intersection(yc.index)
            j = j[(xc.answer_from[j] == "content").values & (yc.answer_from[j] == "content").values]
            r = boot_items((xc.correct[j] - yc.correct[j]).values, SEED, NBOOT)
            print(f"    {'':16s} {'R1, answer-field answers only':34s} {r[0]:+7.2f} [{r[1]:+7.2f} ; "
                  f"{r[2]:+7.2f}]  p={r[3]:.4f}  n={len(j)}")
            rows.append(row("3 reasoning model", "R1, answer-field answers only", "lex causal",
                            tag, f"{a} minus {b}", r, len(j)))
            print()


STEPS = [("KEEP", "IRRELEVANT"), ("IRRELEVANT", "PERMUTE"), ("IRRELEVANT", "SYMBOL"),
         ("SYMBOL", "PSEUDO"), ("KEEP", "SYMBOL"), ("KEEP", "PSEUDO")]


def ladder(rows):
    print("\n" + "=" * 78)
    print("4. THE LEXICAL LADDER ON n600, RAW, CAUSAL ITEMS: LLAMA AGAINST GPT-4.1")
    print("=" * 78 + "\n")
    ref = pd.read_csv(RES / "ladder5_steps_n600.csv").set_index("step")
    for recap in (False, True):
        ladder_one(rows, ref, recap)


def ladder_one(rows, ref, recap):
    rungs = {}
    for lex in ("KEEP", "PSEUDO"):
        if exists("llama", "n600", lex, recap):
            rungs[lex] = load("llama", "n600", lex, recap)
    for lex in ("PERMUTE", "IRRELEVANT", "SYMBOL"):
        if exists("llama ladder", "n600", lex, recap):
            rungs[lex] = load("llama ladder", "n600", lex, recap)
    if len(rungs) < 5:
        print("  waiting for the Llama ladder runs")
        return
    tag = f"RAW, cut-offs re-asked at {RECAP['llama']} tokens" if recap else "RAW"
    print(f"  {tag}")
    qs = causal(rungs["KEEP"])
    acc = {k: cell(v, LLAMA, "RAW", qs) for k, v in rungs.items()}
    print("  Llama accuracy at RAW: " + "  ".join(f"{k} {100 * v.mean():.1f}"
                                                  for k, v in acc.items()))
    for hi, lo in STEPS:
        i = acc[hi].index.intersection(acc[lo].index)
        r = boot_items((acc[hi][i] - acc[lo][i]).values, SEED, NBOOT)
        g = ref.loc[f"{hi} -> {lo}"] if f"{hi} -> {lo}" in ref.index else None
        gtxt = (f"   GPT-4.1 family {g.delta_pp:+6.2f} [{g.ci_lo:+.2f} ; {g.ci_hi:+.2f}]"
                if g is not None else "")
        print(f"  {hi + ' -> ' + lo:22s} Llama {r[0]:+6.2f} [{r[1]:+6.2f} ; {r[2]:+6.2f}]  "
              f"p={r[3]:.4f}  n={len(i)}{gtxt}")
        rows.append(row("4 ladder", "Llama 3.3 70B", "n600 causal", tag,
                        f"{hi} -> {lo}", r, len(i)))
    print()


def runs():
    print("\n" + "=" * 78)
    print("5. THE RUNS: PARSING, CUT-OFFS, COST (from the records)")
    print("=" * 78 + "\n")
    out = []
    for fam in ("llama", "r1", "llama ladder"):
        for s in PREFIX[fam]:
            for lex in ("KEEP", "PSEUDO", "PERMUTE", "IRRELEVANT", "SYMBOL"):
                if not exists(fam, s, lex):
                    continue
                d = pd.read_csv(RES / "raw" / f"pilot_raw_{PREFIX[fam][s]}{lex}.csv")
                rf = RES / "raw" / f"pilot_raw_{PREFIX[fam][s]}{lex}_recap{RECAP.get(fam, 0)}.csv"
                r = pd.read_csv(rf) if rf.exists() else pd.DataFrame()
                rec = {"model": d.model.iloc[0], "sample": s, "lexicon": lex,
                       "rows": len(d), "parsed_pct": round(100 * d.parsed.mean(), 2),
                       "cut_off_by_cap": int((d.finish == "length").sum()),
                       "answer_from_reasoning": int((d.answer_from == "reasoning").sum()),
                       "mean_out_tok": round(d.out_tok.mean(), 1),
                       "mean_reason_tok": round(d.reason_tok.mean(), 1),
                       "providers": ",".join(sorted(d.provider.astype(str).unique())),
                       "billed_usd": round(d.usd.sum(), 4),
                       "reasked": len(r),
                       "reasked_still_cut": int((r.finish == "length").sum()) if len(r) else 0,
                       "reasked_usd": round(r.usd.sum(), 4) if len(r) else 0.0}
                out.append(rec)
                print("  " + "  ".join(f"{k}={v}" for k, v in rec.items()))
    if out:
        t = pd.DataFrame(out)
        print(f"\n  billed in total: {t.billed_usd.sum() + t.reasked_usd.sum():.4f} USD over "
              f"{t.rows.sum() + t.reasked.sum()} records")
        t.to_csv(RES / "second_family_runs.csv", index=False)


def main():
    rows = []
    headline(rows)
    arms(rows)
    reasoning_model(rows)
    ladder(rows)
    runs()
    pd.DataFrame(rows).to_csv(RES / "second_family.csv", index=False)
    print("\n  wrote results/cladder/second_family.csv, results/cladder/second_family_runs.csv")


if __name__ == "__main__":
    main()

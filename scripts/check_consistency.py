"""Do the numbers agree with themselves? Cross-checks no single script can make.

    python scripts/check_consistency.py

Why this file exists. check_numbers.py proves that every number in the text is
in some results file, and verify_determinism.py proves that re-running a script
gives the same file. Neither proves that two scripts computing the SAME quantity
agree, that a condition answered from the cache really carries the answer it
should, or that a verdict does not hang on the bootstrap seed. On 2026-09-24 a
replay in classify_perturbations.py was found to have named the wrong
perturbation for half its items since it was written, and every gate passed
throughout. This file is the missing layer.

  1. AGREEMENT     one quantity, computed by different scripts, must be equal to
                   the last digit
  2. RECOMPUTED    a few headline numbers re-derived from the raw per-response
                   records with fresh code that calls no project function
  3. INTEGRITY     no duplicate rows; items of one sample agree across files;
                   an identical prompt carries an identical answer everywhere
                   (RAW_CLEAN = RAW off the latent families, SCRAMBLE = DR_k3 on
                   the complete three-node families, one CLadder id asked in two
                   samples). Where two answers to one prompt differ, the cache
                   must show why: the prompt was sent twice and the later answer
                   overwrote the earlier (scripts/audit_cache_agreement.py found
                   494 such cells, 0.54%). An unexplained difference fails.
  4. SEEDS         the key and the borderline quantities re-bootstrapped under
                   five seeds; a verdict (interval excludes zero) that flips with
                   the seed is reported as BORDERLINE, to be quoted with its p
                   range and never as either side of 0.05. The first seed is
                   the project seed, and there the result must equal the
                   published CSV row exactly: the bootstrap resamples positions,
                   so a script that orders the same items differently gets a
                   different interval from the same seed (found 2026-09-24)

Exit 1 on any failed check. Borderline verdicts are listed, not failed.
Writes: results/seed_stability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from stats import boot_items, boot_p, boot_two_sample, cluster_boot

RES = ROOT / "results"
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
NONCAUSAL = {"marginal", "correlation", "backadj"}
LATENT_FAMILIES = {"IV", "arrowhead", "frontdoor"}
SEVEN = {"IV", "arrowhead", "confounding", "diamond", "diamondcut", "frontdoor", "mediation"}
SEEDS = [20260907, 1, 2, 3, 4]
NBOOT = 4000
FAILS: list[str] = []
BORDERLINE: list[str] = []


def check(label, ok, detail=""):
    print(f"  {'OK  ' if ok else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS.append(label)


def row(file, **sel):
    d = pd.read_csv(RES / file)
    for k, v in sel.items():
        d = d[d[k] == v]
    if len(d) != 1:
        raise SystemExit(f"{file} {sel}: {len(d)} rows, expected 1")
    return d.iloc[0]


def triple(r, est):
    return (round(float(r[est]), 2), round(float(r["ci_lo"]), 2), round(float(r["ci_hi"]), 2))


def raw(tag):
    return pd.read_csv(RES / "raw" / f"pilot_raw_{tag}.csv")


def itemmap(sample):
    return pd.read_csv(RES / f"_itemmap_{sample}.csv").set_index("item").id


# --------------------------------------------------------------- 1. agreement
def agreement():
    print("=" * 78)
    print("1. ONE QUANTITY, SEVERAL SCRIPTS")
    print("=" * 78)
    head = [("pool_samples", triple(row("pooled_headline.csv", pooling="item level, de-duplicated"), "did_pp")),
            ("analyze_structure_arms", triple(row("structure_arms.csv", quantity="DiD | ORACLE"), "estimate_pp")),
            ("measure_raw_leak s2", triple(row("raw_leak_sensitivity.csv", subset="all items (headline)"), "did_pp")),
            ("measure_raw_leak s3", triple(row("raw_clean.csv", section="3a headline DiD",
                                               quantity="RAW (the published headline)"), "estimate_pp"))]
    check("headline DiD in four scripts", len({t for _, t in head}) == 1,
          "; ".join(f"{n} {t}" for n, t in head))

    vs = pd.read_csv(RES / "vs_raw.csv")
    vs = vs[vs.query_group == "causal"]
    split = pd.read_csv(RES / "perturbation_split.csv")
    for lex in ("KEEP", "PSEUDO"):
        for cond in ("DR_k1", "DR_k2", "DR_k3", "ED_k1", "ED_k2", "FE_k1"):
            a = triple(vs[(vs["sample"] == "price400") & (vs.lexicon == lex) & (vs.cond == cond)].iloc[0], "delta_pp")
            b = triple(split[(split.lexicon == lex) & (split.cond == cond) & (split.subset == "all")].iloc[0], "delta_pp")
            check(f"price400 {lex} {cond} minus RAW: analyze_vs_raw = classify_perturbations", a == b, f"{a} {b}")
    s600 = pd.read_csv(RES / "perturbation_split_n600.csv")
    rc = pd.read_csv(RES / "raw_clean.csv")
    for lex in ("KEEP", "PSEUDO"):
        a = triple(vs[(vs["sample"] == "n600") & (vs.lexicon == lex) & (vs.cond == "DR_k1")].iloc[0], "delta_pp")
        b = triple(s600[(s600.lexicon == lex) & (s600.families == "all 10") & (s600.cond == "DR_k1")
                        & (s600.subset == "all")].iloc[0], "delta_pp")
        check(f"n600 {lex} DR_k1 minus RAW: analyze_vs_raw = classify_perturbations s3", a == b, f"{a} {b}")
        for t in ("lex", "n600", "price400"):
            for arm in ("ORACLE", "DR_k1"):
                a = triple(vs[(vs["sample"] == t) & (vs.lexicon == lex) & (vs.cond == arm)].iloc[0], "delta_pp")
                b = triple(rc[(rc.section == "3c direct contrast") & (rc["sample"] == t) & (rc.lexicon == lex)
                              & (rc.quantity == f"{arm} minus RAW")].iloc[0], "estimate_pp")
                check(f"{t} {lex} {arm} minus RAW: analyze_vs_raw = measure_raw_leak s3", a == b, f"{a} {b}")
    tr = pd.read_csv(RES / "vs_raw_trend.csv")
    cs = pd.read_csv(RES / "perturbation_conditional_slope.csv")
    for lex in ("KEEP", "PSEUDO"):
        a = triple(tr[(tr.arm == "DR") & (tr.lexicon == lex)].iloc[0], "estimate")
        b = triple(cs[(cs.lexicon == lex) & (cs["sample"] == "price400")
                      & (cs.quantity == "unconditional slope")].iloc[0], "delta_pp")
        check(f"price400 {lex} unconditional DR slope: analyze_vs_raw = classify_perturbations s4",
              a == b, f"{a} {b}")
        a = triple(s600[(s600.lexicon == lex) & (s600.cond == "DR slope")].iloc[0], "delta_pp")
        b = triple(cs[(cs.lexicon == lex) & (cs["sample"] == "n600")
                      & (cs.quantity == "unconditional slope")].iloc[0], "delta_pp")
        check(f"n600 {lex} unconditional DR slope: classify s3 = classify s4", a == b, f"{a} {b}")
    a = triple(row("structure_arms.csv", quantity="PSEUDO branch, ORACLE minus DR_k1 | n600"), "estimate_pp")
    b = triple(row("structure_arms.csv", quantity="PSEUDO | ORACLE minus DR_k1 | n600"), "estimate_pp")
    check("n600 PSEUDO ORACLE minus DR_k1: structure_arms s4b = s6", a == b, f"{a} {b}")


# -------------------------------------------------------------- 2. recomputed
def causal_cells(d, cond, key="item"):
    """{model: Series key -> correct}, parsed answers on the causal group only."""
    d = d[(d.cond == cond) & (d.parsed == 1) & ~d.query_type.isin(NONCAUSAL)]
    return {m: g.set_index(key).correct for m, g in d.groupby("model")}


def recomputed():
    print("\n" + "=" * 78)
    print("2. RE-DERIVED FROM THE RAW RECORDS WITH FRESH CODE")
    print("=" * 78)
    # (a) n600 KEEP DR_k1 minus RAW: per item, mean over models; then mean over items
    d = raw("n600KEEP")
    r, c = causal_cells(d, "RAW"), causal_cells(d, "DR_k1")
    per = pd.concat([(c[m] - r[m]).dropna() for m in TIER], axis=1).mean(axis=1)
    want = row("vs_raw.csv", query_group="causal", sample="n600", lexicon="KEEP", cond="DR_k1").delta_pp
    check("n600 KEEP DR_k1 minus RAW", round(100 * per.mean(), 2) == want, f"{100 * per.mean():.2f} vs {want}")

    # (b) the headline: every (sample x model) cell of the DiD, keyed by CLadder id
    cols = []
    for s in ("lex", "n600", "price400"):
        im = itemmap(s)
        K, P = raw(f"{s}KEEP"), raw(f"{s}PSEUDO")
        K, P = K.assign(id=K.item.map(im)), P.assign(id=P.item.map(im))
        kr, pr, ko, po = (causal_cells(K, "RAW", "id"), causal_cells(P, "RAW", "id"),
                          causal_cells(K, "ORACLE", "id"), causal_cells(P, "ORACLE", "id"))
        for m in TIER:
            x = ((kr[m] - pr[m]) - (ko[m] - po[m])).dropna()
            cols.append(x.rename(f"{s}|{m}"))
    W = pd.concat(cols, axis=1).groupby(level=0).mean()
    got = 100 * np.nanmean(W.values)
    want = row("pooled_headline.csv", pooling="item level, de-duplicated").did_pp
    check("pooled headline DiD, n=490", round(got, 2) == want and len(W) == 490,
          f"{got:.2f} on {len(W)} ids vs {want}")

    # (c) the ladder's first rung on n600: every (model x item) cell
    K, I = raw("n600KEEP"), raw("n600ladderIRRELEVANT")
    kc, ic = causal_cells(K, "RAW"), causal_cells(I, "RAW")
    vals = np.concatenate([(kc[m] - ic[m]).dropna().values for m in TIER])
    want = row("ladder5_steps_n600.csv", step="KEEP -> IRRELEVANT").delta_pp
    check("n600 ladder KEEP -> IRRELEVANT", round(100 * vals.mean(), 2) == want,
          f"{100 * vals.mean():.2f} vs {want}")

    # (d) SCRAMBLE minus NAMES_ONLY, n600 KEEP, cells keyed by id
    A = raw("n600armsKEEP")
    s_, n_ = causal_cells(A, "SCRAMBLE"), causal_cells(A, "NAMES_ONLY")
    W = pd.concat([(s_[m] - n_[m]).dropna() for m in TIER], axis=1)
    got = 100 * np.nanmean(W.values)
    want = row("structure_arms.csv", quantity="KEEP | SCRAMBLE minus NAMES_ONLY | n600").estimate_pp
    check("n600 KEEP SCRAMBLE minus NAMES_ONLY", round(got, 2) == want, f"{got:.2f} vs {want}")

    # (e) drift, all cells of the n600 re-ask: per item mean, then mean
    R = pd.read_csv(RES / "drift_check_raw.csv")
    per = (R.new_correct - R.old_correct).groupby(R.item).mean()
    want = row("drift_check.csv", lexicon="all", cond="all", model="pooled").drift_pp
    check("n600 drift, pooled", round(100 * per.mean(), 2) == want, f"{100 * per.mean():.2f} vs {want}")


# --------------------------------------------------------------- 3. integrity
SAMPLE_FILES = {
    "lex": ["lexKEEP", "lexPERMUTE", "lexIRRELEVANT", "lexSYMBOL", "lexPSEUDO", "edfeKEEP",
            "edfePERMUTE", "edfeSYMBOL", "edfePSEUDO", "instrKEEP", "instrPSEUDO",
            "cleanrawlexKEEP", "cleanrawlexPSEUDO"],
    "n600": ["n600KEEP", "n600PSEUDO", "n600armsKEEP", "n600armsPSEUDO", "n600ladderPERMUTE",
             "n600ladderIRRELEVANT", "n600ladderSYMBOL", "cleanrawn600KEEP", "cleanrawn600PSEUDO"],
    "price400": ["price400KEEP", "price400PSEUDO", "cleanrawprice400KEEP", "cleanrawprice400PSEUDO"],
}
META = ["graph_id", "gold", "query_type", "story_id"]
FULL = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv", low_memory=False).set_index("id")


def pred(d):
    return d.set_index(["model", "item"]).pred.fillna("")


_PROMPTS: dict = {}
SAMPLE_ARGS = {"lex": (200, 1, ("DR",)), "n600": (600, 1, ("DR",)),
               "price400": (400, 3, ("DR", "ED", "FE"))}


def cache_answer(sample, lexicon, item, cond, model):
    """The answer the API cache holds for this cell's prompt; None if no cache."""
    import json
    import pilot
    from prompts import parse_answer
    from runner import _key
    if (sample, lexicon) not in _PROMPTS:
        n, k, t = SAMPLE_ARGS[sample]
        items = pilot.make_items(n, 20260907, k, "full_v1.5_default.csv", None, True)
        _PROMPTS[(sample, lexicon)] = {(j["item"], j["cond"]): j["prompt"] for j in
                                       pilot.build_jobs(items, k, 20260907, t, lexicon)}
    f = _key(model, 0.0, _PROMPTS[(sample, lexicon)][(item, cond)])
    if not f.exists():
        return None
    try:
        a = parse_answer(json.loads(f.read_text(encoding="utf-8"))["text"])
    except json.JSONDecodeError:
        return None
    return "" if a is None else a


def explain(label, mismatches):
    """Mismatch between two answers to ONE prompt. Explained when the side read
    back from the cache equals the cache and the CSV side does not: the prompt
    was sent twice and the second answer overwrote the first in the cache
    (scripts/audit_cache_agreement.py). Anything else is a real inconsistency."""
    if not mismatches:
        check(label, True, "no mismatch")
        return
    have_cache = any((ROOT / "cache").glob("*.json"))
    if not have_cache:
        print(f"  WARN {label}   {len(mismatches)} mismatches; cache absent, cannot explain")
        return
    bad = [m for m in mismatches if not m()]
    check(f"{label}: {len(mismatches)} mismatches, all a prompt sent twice",
          not bad, f"{len(bad)} unexplained")


def integrity():
    print("\n" + "=" * 78)
    print("3. INTEGRITY OF THE PER-RESPONSE RECORDS")
    print("=" * 78)
    for s, tags in SAMPLE_FILES.items():
        base = None
        for t in tags:
            d = raw(t)
            dup = int(d.duplicated(["model", "item", "cond"]).sum())
            err = int(d.error.fillna("").astype(str).str.len().gt(0).sum()) if "error" in d else 0
            meta = d.drop_duplicates("item").set_index("item")[META]
            if base is None:
                base = meta
            i = base.index.intersection(meta.index)
            same = bool((base.loc[i].astype(str) == meta.loc[i].astype(str)).all().all())
            check(f"{t}: no duplicate rows, no API errors, items agree with {tags[0]}",
                  dup == 0 and err == 0 and same and len(i) == len(meta),
                  f"dup={dup} err={err} items={len(meta)}")
            # The record's own CLadder id (scripts/backfill_ids.py) must agree with
            # the verified item map AND with CLadder's row for that id.
            if "id" in d.columns:
                ok_map = bool((d.id == d.item.map(itemmap(s))).all())
                row = FULL.loc[d.id]
                ok_rows = all((d[a].astype(str).values == row[b].astype(str).values).all()
                              for a, b in (("graph_id", "graph_id"), ("gold", "label"),
                                           ("query_type", "query_type"), ("story_id", "story_id")))
                check(f"{t}: id column agrees with the item map and with CLadder",
                      ok_map and ok_rows, f"map={ok_map} rows={ok_rows}")
            else:
                check(f"{t}: carries the CLadder id column", False)

    print()
    for s in ("lex", "n600", "price400"):
        for lex in ("KEEP", "PSEUDO"):
            d = raw(f"{s}{lex}")
            c = raw(f"cleanraw{s}{lex}")
            off = c[~c.graph_id.isin(LATENT_FAMILIES)]
            a = pred(d[d.cond == "RAW"]).reindex(pred(off).index)
            b = pred(off)
            mis = [(lambda m=m, i=i, x=b[(m, i)], y=a[(m, i)], s=s, lex=lex:
                    cache_answer(s, lex, i, "RAW", m) == x != y)
                   for (m, i) in b.index[(a != b).values]]
            print(f"  {s} {lex}: RAW_CLEAN = RAW off the latent families in "
                  f"{len(b) - len(mis)}/{len(b)} cells")
            explain(f"{s} {lex}: RAW_CLEAN vs RAW off the latent families", mis)
    for lex in ("KEEP", "PSEUDO"):
        a = raw(f"n600arms{lex}")
        a = a[a.graph_id.isin({"confounding", "mediation"})]
        x, y = pred(a[a.cond == "SCRAMBLE"]), pred(a[a.cond == "DR_k3"])
        i = x.index.intersection(y.index)
        check(f"n600 {lex}: SCRAMBLE answer = DR_k3 answer on the complete 3-node families",
              int((x[i] == y[i]).sum()) == len(i) and len(i) > 0, f"{int((x[i] == y[i]).sum())}/{len(i)}")

    print()
    pairs = [("n600", "price400"), ("n600", "lex"), ("lex", "price400")]
    for lex in ("KEEP", "PSEUDO"):
        for cond in ("RAW", "PROSE", "ORACLE"):
            for s1, s2 in pairs:
                a, b = raw(f"{s1}{lex}"), raw(f"{s2}{lex}")
                a = a[a.cond == cond].assign(id=lambda z: z.item.map(itemmap(s1)))
                b = b[b.cond == cond].assign(id=lambda z: z.item.map(itemmap(s2)))
                j = a.merge(b, on=["model", "id"], suffixes=("_a", "_b"))
                j = j.assign(pa=j.pred_a.fillna(""), pb=j.pred_b.fillna(""))
                bad = j[j.pa != j.pb]
                mis = [(lambda r=r, s1=s1, s2=s2, lex=lex, cond=cond:
                        (lambda ca, cb: (ca == r.pa) != (cb == r.pb))(
                            cache_answer(s1, lex, r.item_a, cond, r.model),
                            cache_answer(s2, lex, r.item_b, cond, r.model)))
                       for r in bad.itertuples()]
                print(f"  {lex} {cond}: id shared by {s1} and {s2}, one answer in "
                      f"{len(j) - len(bad)}/{len(j)} cells")
                explain(f"{lex} {cond}: ids shared by {s1} and {s2}", mis)
    a = raw("n600ladderIRRELEVANT").assign(id=lambda z: z.item.map(itemmap("n600")))
    b = raw("lexIRRELEVANT")
    b = b[b.cond == "RAW"].assign(id=lambda z: z.item.map(itemmap("lex")))
    j = a.merge(b, on=["model", "id"], suffixes=("_a", "_b"))
    n_same = int((j.pred_a.fillna("") == j.pred_b.fillna("")).sum())
    check("IRRELEVANT RAW: n600 ladder and lex sample agree on shared ids",
          n_same == len(j) and len(j) > 0, f"{n_same}/{len(j)}")


# ------------------------------------------------------------------- 4. seeds


def seeds():
    print("\n" + "=" * 78)
    print("4. DO THE VERDICTS DEPEND ON THE BOOTSTRAP SEED?")
    print("=" * 78)
    from pool_samples import boot as pool_boot, did_sample
    from measure_raw_leak import did_base
    from analyze_ladder5 import load as lload, paired as lpaired, boot as lboot
    from analyze_structure_arms import load_n600
    from pool_samples import cell

    q = {}   # label -> function(seed) -> (est, lo, hi, p)
    ims = {t: pd.read_csv(RES / f"_itemmap_{t}.csv") for t in ("lex", "n600", "price400")}
    W = pd.concat([did_sample(t, ims[t], ["PSEUDO"]).rename(columns=lambda c, t=t: f"{t}|{c}")
                   for t in ims], axis=1)
    q["headline DiD (RAW)"] = lambda s, W=W: pool_boot(W, seed=s)
    C = pd.concat([did_base(t, ims[t], "RAW_CLEAN") for t in ims], axis=1)
    detcf = set()
    for t in ims:
        for lex in ("KEEP",):
            d = raw(f"{t}{lex}")
            detcf |= set(d[d.query_type == "det-counterfactual"].item.map(itemmap(t)))
    Cc = C.loc[[i for i in C.index if i not in detcf]]
    q["headline DiD, RAW_CLEAN, no det-cf"] = lambda s, W=Cc: pool_boot(W, seed=s)

    L = {x: lload(x, "n600") for x in ("KEEP", "IRRELEVANT", "PERMUTE", "SYMBOL")}
    for hi_, lo_ in (("KEEP", "IRRELEVANT"), ("IRRELEVANT", "PERMUTE"), ("IRRELEVANT", "SYMBOL")):
        by = {m: lpaired(L[hi_], L[lo_], m, "RAW", True) for m in TIER}
        q[f"n600 ladder {hi_} -> {lo_}"] = lambda s, by=by: (lambda e, b: (
            e, *np.percentile(b, [2.5, 97.5]), boot_p(b, NBOOT)))(*lboot(by, s, NBOOT))

    im6 = ims["n600"]
    D = {x: load_n600(x, im6) for x in ("KEEP", "PSEUDO")}
    qs = set(D["KEEP"].query_type.unique()) - NONCAUSAL
    for lex, a_, b_ in (("PSEUDO", "ORACLE", "NAMES_ONLY"), ("PSEUDO", "NAMES_ONLY", "RAW"),
                        ("KEEP", "SCRAMBLE", "NAMES_ONLY"), ("KEEP", "DR_k3", "DR_k2")):
        cols = []
        for m in TIER:
            x, y = cell(D[lex], m, a_, qs), cell(D[lex], m, b_, qs)
            i = x.index.intersection(y.index)
            cols.append(pd.Series((x[i] - y[i]).values, index=i, name=m))
        M = pd.concat(cols, axis=1).groupby(level=0).mean()
        q[f"n600 {lex} {a_} minus {b_}"] = lambda s, M=M: pool_boot(M, seed=s)

    # the items-fixed conditional slope, pooled over price400 and n600
    for lex in ("KEEP", "PSEUDO"):
        parts = []
        for tag, cls, files in (("price400", "perturbation_classes.csv", [f"price400{lex}"]),
                                ("n600", "perturbation_classes_n600.csv", [f"n600{lex}", f"n600arms{lex}"])):
            d = pd.concat([raw(f) for f in files], ignore_index=True)
            d = d[d.graph_id.isin(SEVEN)]
            v = {}
            for k in (1, 2, 3):
                r_, c_ = causal_cells(d, "RAW"), causal_cells(d, f"DR_k{k}")
                v[k] = pd.concat([(c_[m] - r_[m]).dropna() for m in TIER], axis=1).mean(axis=1)
            M = pd.concat(v, axis=1).dropna()
            f = pd.read_csv(RES / cls)
            f = f[(f.cond == "DR_k1") & (f.lexicon == lex)].set_index("item").unchanged_qt
            M = M[f.reindex(M.index).eq(False).values]
            parts.append(pd.Series(((M[3] - M[1]) / 2).values, index=M.index.map(itemmap(tag))))
        x = pd.concat(parts).groupby(level=0).mean().values
        q[f"conditional slope, pooled, {lex}"] = lambda s, x=x: boot_items(x, s, NBOOT)
        # parts[0] is price400, parts[1] n600: the two-sample difference of section 4
        a_, b_ = parts[1].sort_index().values, parts[0].sort_index().values
        q[f"conditional slope, n600 minus price400, {lex}"] = (
            lambda s, a_=a_, b_=b_: boot_two_sample(a_, b_, s, NBOOT))

    # The second model family and the reasoning model, rebuilt here from the
    # records with the item map, not imported from analyze_second_family.py.
    LL = {"lex": "llama70b", "n600": "llama70b_n600", "price400": "llama70b_price400"}
    if all((RES / "raw" / f"pilot_raw_{LL[t]}{x}.csv").exists() for t in LL for x in ("KEEP", "PSEUDO")):
        cols = []
        for t, pre in LL.items():
            ck = causal_cells(raw(f"{pre}KEEP"), "RAW"), causal_cells(raw(f"{pre}KEEP"), "ORACLE")
            cp = causal_cells(raw(f"{pre}PSEUDO"), "RAW"), causal_cells(raw(f"{pre}PSEUDO"), "ORACLE")
            m = "meta-llama/llama-3.3-70b-instruct"
            kr, ka, pr, pa = ck[0][m], ck[1][m], cp[0][m], cp[1][m]
            i = kr.index.intersection(ka.index).intersection(pr.index).intersection(pa.index)
            v = (kr[i] - pr[i]) - (ka[i] - pa[i])
            cols.append(pd.Series(v.values, index=i.map(itemmap(t)), name=t).sort_index())
        Wl = pd.concat(cols, axis=1)
        q["Llama pooled DiD"] = lambda s, W=Wl: pool_boot(W, seed=s)
        # its two halves, lift and drag: all cells sorted by id at once, as
        # analyze_structure_arms.py builds them for GPT-4.1 (review round 12, M12-1)
        for lex, half in (("PSEUDO", "lift"), ("KEEP", "drag")):
            cols = []
            for t, pre in LL.items():
                cc = causal_cells(raw(f"{pre}{lex}"), "RAW"), causal_cells(raw(f"{pre}{lex}"), "ORACLE")
                r_, o_ = cc[0]["meta-llama/llama-3.3-70b-instruct"], cc[1]["meta-llama/llama-3.3-70b-instruct"]
                i = r_.index.intersection(o_.index)
                cols.append(pd.Series((o_[i] - r_[i]).values, index=i.map(itemmap(t)), name=t))
            Wh = pd.concat(cols, axis=1).groupby(level=0).mean()
            q[f"Llama {half}, {lex} ORACLE minus RAW"] = lambda s, W=Wh: pool_boot(W, seed=s)
    if (RES / "raw" / "pilot_raw_r1PSEUDO.csv").exists():
        R = raw("r1PSEUDO")
        # the 12 answers cut off at 8,000 tokens replaced by their re-asks
        rr = raw("r1PSEUDO_recap16000")
        cut = set(zip(R.loc[R.finish == "length", "item"], R.loc[R.finish == "length", "cond"]))
        Rr = pd.concat([R[[(i, c) not in cut for i, c in zip(R.item, R.cond)]], rr],
                       ignore_index=True)
        for tag, D in (("", R), (" re-asked", Rr)):
            for a_, b_ in (("ORACLE", "RAW"), ("ORACLE", "DR_k1")):
                x = causal_cells(D, a_)["deepseek/deepseek-r1"]
                y = causal_cells(D, b_)["deepseek/deepseek-r1"]
                i = x.index.intersection(y.index)
                q[f"R1 PSEUDO{tag} {a_} minus {b_}"] = (
                    lambda s, v=(x[i] - y[i]).values: boot_items(v, s, NBOOT))

    # the four-edge slice against the rest (analyze_by_family.py section 7)
    from analyze_by_family import pooled_matrix, structure
    Gm, fam_of = pooled_matrix()
    st = structure()
    xm = Gm.mean(axis=1)
    ed = fam_of.reindex(Gm.index).map(lambda g: st[g][1]).values
    a4, b4 = xm[ed == 4].sort_index().values, xm[ed != 4].sort_index().values
    q["four edges minus the rest"] = lambda s, a=a4, b=b4: boot_two_sample(a, b, s, NBOOT)

    published = {
        "four edges minus the rest": ("family_breakdown.csv", "estimate_pp",
                                      {"slice": "4 edges minus the rest"}),
        "headline DiD (RAW)": ("pooled_headline.csv", "did_pp", {"pooling": "item level, de-duplicated"}),
        "Llama pooled DiD": ("second_family.csv", "estimate_pp",
                             {"section": "1 headline", "model": "Llama 3.3 70B",
                              "sample": "pooled"}),
        "Llama lift, PSEUDO ORACLE minus RAW": ("second_family.csv", "estimate_pp",
                                                {"section": "2 arms", "model": "Llama 3.3 70B",
                                                 "lexicon": "PSEUDO", "quantity": "ORACLE minus RAW"}),
        "Llama drag, KEEP ORACLE minus RAW": ("second_family.csv", "estimate_pp",
                                              {"section": "2 arms", "model": "Llama 3.3 70B",
                                               "lexicon": "KEEP", "quantity": "ORACLE minus RAW"}),
        "R1 PSEUDO ORACLE minus RAW": ("second_family.csv", "estimate_pp",
                                       {"model": "deepseek/deepseek-r1", "lexicon": "PSEUDO",
                                        "quantity": "ORACLE minus RAW"}),
        "R1 PSEUDO ORACLE minus DR_k1": ("second_family.csv", "estimate_pp",
                                         {"model": "deepseek/deepseek-r1", "lexicon": "PSEUDO",
                                          "quantity": "ORACLE minus DR_k1"}),
        "R1 PSEUDO re-asked ORACLE minus RAW": (
            "second_family.csv", "estimate_pp",
            {"model": "deepseek/deepseek-r1", "quantity": "ORACLE minus RAW",
             "lexicon": "PSEUDO, cut-offs re-asked at 16000 tokens"}),
        "R1 PSEUDO re-asked ORACLE minus DR_k1": (
            "second_family.csv", "estimate_pp",
            {"model": "deepseek/deepseek-r1", "quantity": "ORACLE minus DR_k1",
             "lexicon": "PSEUDO, cut-offs re-asked at 16000 tokens"}),
        "headline DiD, RAW_CLEAN, no det-cf": ("raw_clean.csv", "estimate_pp",
                                               {"quantity": "RAW_CLEAN, drop det-counterfactual"}),
        "n600 PSEUDO ORACLE minus NAMES_ONLY": ("structure_arms.csv", "estimate_pp",
                                                {"quantity": "PSEUDO | ORACLE minus NAMES_ONLY | n600"}),
        "n600 PSEUDO NAMES_ONLY minus RAW": ("structure_arms.csv", "estimate_pp",
                                             {"quantity": "PSEUDO | NAMES_ONLY minus RAW | n600"}),
        "n600 KEEP SCRAMBLE minus NAMES_ONLY": ("structure_arms.csv", "estimate_pp",
                                                {"quantity": "KEEP | SCRAMBLE minus NAMES_ONLY | n600"}),
        "n600 KEEP DR_k3 minus DR_k2": ("structure_arms.csv", "estimate_pp",
                                        {"quantity": "KEEP | DR_k3 minus DR_k2 | n600"}),
    }
    for lex in ("KEEP", "PSEUDO"):
        published[f"conditional slope, pooled, {lex}"] = (
            "perturbation_conditional_slope.csv", "delta_pp",
            {"lexicon": lex, "sample": "pooled", "quantity": "slope per reversed edge"})
        published[f"conditional slope, n600 minus price400, {lex}"] = (
            "perturbation_conditional_slope.csv", "delta_pp",
            {"lexicon": lex, "sample": "n600 minus price400"})
    for hi_, lo_ in (("KEEP", "IRRELEVANT"), ("IRRELEVANT", "PERMUTE"), ("IRRELEVANT", "SYMBOL")):
        published[f"n600 ladder {hi_} -> {lo_}"] = ("ladder5_steps_n600.csv", "delta_pp",
                                                    {"step": f"{hi_} -> {lo_}"})

    rows = []
    for label, fn in q.items():
        res = [fn(s) for s in SEEDS]
        if label in published:
            f, col, sel = published[label]
            got = (round(float(res[0][0]), 2), round(float(res[0][1]), 2), round(float(res[0][2]), 2))
            check(f"{label}: at the project seed it equals {f}", got == triple(row(f, **sel), col),
                  f"{got} {triple(row(f, **sel), col)}")
        est = {round(r[0], 2) for r in res}
        los, his, ps = [r[1] for r in res], [r[2] for r in res], [r[3] for r in res]
        excl = {(lo > 0 or hi < 0) for lo, hi in zip(los, his)}
        sig = {p < 0.05 for p in ps}
        stable = len(excl) == 1 and len(sig) == 1 and len(est) == 1
        rows.append(dict(quantity=label, estimate_pp=res[0][0], ci_lo_min=round(min(los), 2),
                         ci_lo_max=round(max(los), 2), ci_hi_min=round(min(his), 2),
                         ci_hi_max=round(max(his), 2), p_min=round(min(ps), 4),
                         p_max=round(max(ps), 4), verdict_stable=stable))
        text = (f"{label}: {res[0][0]:+.2f}, lo {min(los):+.2f}..{max(los):+.2f}, "
                f"hi {min(his):+.2f}..{max(his):+.2f}, p {min(ps):.4f}..{max(ps):.4f}")
        # A verdict that flips with the seed is not a computing error; it is a
        # quantity sitting on the 0.05 line. It must be reported as borderline,
        # with its p range, and never as either side of the line.
        print(f"  {'OK        ' if stable else 'BORDERLINE'} {text}")
        if not stable:
            BORDERLINE.append(text)
    pd.DataFrame(rows).to_csv(RES / "seed_stability.csv", index=False)
    print(f"\n  {len(SEEDS)} seeds x {NBOOT} draws each; wrote results/seed_stability.csv")


def main() -> int:
    agreement()
    recomputed()
    integrity()
    seeds()
    print("\n" + "=" * 78)
    if BORDERLINE:
        print(f"  {len(BORDERLINE)} verdict(s) on the 0.05 line - quote the p range, not a side:")
        for b in BORDERLINE:
            print(f"    - {b}")
    if FAILS:
        print(f"  {len(FAILS)} FAILED:")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  Every cross-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

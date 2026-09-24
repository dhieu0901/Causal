"""How much causal structure survives in RAW, the supposedly graph-free arm?

    python scripts/measure_raw_leak.py

RAW is built by `strip_structure()`, which deletes every sentence matching
"<A> has a direct effect on <B>". That pattern covers 100% of CLadder prompts,
so the DAG paragraph is always removed. The question this file answers is what
is left behind, because REPORT.md section 4 leaned on the phrase "RAW is not
'no graph'" without a corpus-wide number attached to it.

Two different things survive, and lumping them together overstates the problem
by a factor of four. Both are measured separately here.

  edge stated in prose   "We know that alarm set by husband causes alarm not
                         set by wife." This IS an edge, written in ordinary
                         language, and strip_structure never sees it.

  latent disclosed       "Unobserved confounders is unobserved." This does not
                         name an edge; it discloses that a latent confounder
                         exists, which is weaker but still structural.

A third phrase looked like leakage and is NOT: "To understand how husband
affects alarm clock, is it more correct to use Method 1 than Method 2?" That is
the backadj QUESTION stem. It names the pair being asked about, every arm needs
it, and counting it would have turned a 12% leak into a 52% one. It is excluded
deliberately, and this comment is here so nobody re-adds it.

What this does NOT do: widen the regex. Changing strip_structure() changes the
prompts, which means re-running the experiment, which costs money. The finding
is a limitation to report, not a patch to apply - and because every arm is built
from the same stripped body, the leak sits in RAW and in every graph condition
alike, so it does not bias a within-item paired contrast.

An earlier version of this docstring added that the leak makes the measured
benefit of a graph a LOWER bound - so dropping leaky items should raise it.
Section 2 below tests that and the data do not support it: dropping the items
that state an edge in prose moves the headline DiD DOWN, 5.98 to 5.07 pp
(p=0.046), within noise but the wrong way for a lower bound. The latent split
cannot test it at all - "is unobserved" occurs in 100% of IV, arrowhead and
frontdoor items and 0% elsewhere, so dropping it drops three graph families.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from pool_samples import POOL_LEXICON, attach_id, boot, cell, did_sample, load
from prompts import strip_structure
from analyze_querygroup import ARITH, IDENT, TIER
from analyze_vs_raw import boot as vs_boot

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLES = ["lex", "n600", "price400"]
# The families whose items carry "X is unobserved." (section 1 measures it).
LATENT_FAMILIES = {"IV", "arrowhead", "frontdoor"}
EDGE = re.compile(r"\bcauses\b", re.IGNORECASE)
LATENT = re.compile(r"is unobserved", re.IGNORECASE)


def main() -> int:
    src = ROOT / "data" / "full_v1.5_default.csv"
    if not src.exists():
        raise SystemExit("thieu data/full_v1.5_default.csv")
    full = pd.read_csv(src, low_memory=False).set_index("id")

    print("=" * 78)
    print("CAU TRUC CON SOT TRONG RAW")
    print("=" * 78)
    print("\n  strip_structure() xoa moi cau 'has a direct effect on'.")
    print("  Mau do khop 100% prompt CLadder, nen doan DAG luon bi xoa het.\n")

    rows = []
    edge_ids, latent_ids = set(), set()
    for tag in SAMPLES:
        im = ROOT / "results" / f"_itemmap_{tag}.csv"
        if not im.exists():
            print(f"  {tag}: thieu _itemmap_{tag}.csv, bo qua")
            continue
        ids = [i for i in pd.read_csv(im).id if i in full.index]
        bodies = [strip_structure(str(full.loc[i, "prompt"]))[0] for i in ids]
        n = len(bodies)
        if not n:
            continue
        e = [bool(EDGE.search(b)) for b in bodies]
        l = [bool(LATENT.search(b)) for b in bodies]
        edge_ids |= {i for i, x in zip(ids, e) if x}
        latent_ids |= {i for i, x in zip(ids, l) if x}
        rows.append({"sample": tag, "n_items": n,
                     "n_edge_stated": sum(e),
                     "edge_stated_pct": round(100 * sum(e) / n, 1),
                     "n_latent_disclosed": sum(l),
                     "latent_disclosed_pct": round(100 * sum(l) / n, 1),
                     "n_either": sum(x or y for x, y in zip(e, l)),
                     "either_pct": round(100 * sum(x or y for x, y in zip(e, l)) / n, 1),
                     "n_both": sum(x and y for x, y in zip(e, l))})

    if not rows:
        raise SystemExit("khong co mau nao de do")
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    out.to_csv(ROOT / "results" / "raw_structure_leak.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Does the headline survive without the leaky items? Review item V7-16
    # asked for this sensitivity; the regex fix it also asked for costs money
    # (see the docstring), this does not. The pooled DiD is rebuilt with the
    # SAME did_sample() and boot() that produce the +5.98 headline, so the
    # first row must reproduce it exactly - if it does not, the rows below it
    # are not comparable and mean nothing.
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("2. HEADLINE DiD WITHOUT THE ITEMS WHOSE RAW STILL CARRIES STRUCTURE")
    print("=" * 78)
    per = []
    for tag in SAMPLES:
        im = ROOT / "results" / f"_itemmap_{tag}.csv"
        if not im.exists():
            continue
        W = did_sample(tag, pd.read_csv(im), [POOL_LEXICON], arm="ORACLE")
        if not W.empty:
            per.append(W.rename(columns=lambda c, t=tag: f"{t}|{c}"))
    sens = []
    if per:
        P = pd.concat(per, axis=1)
        for label, drop in (("all items (headline)", set()),
                            ("drop edge stated in prose", edge_ids),
                            ("drop latent disclosed", latent_ids),
                            ("drop either", edge_ids | latent_ids)):
            M = P.loc[[i for i in P.index if i not in drop]]
            est, lo, hi, p = boot(M)
            print(f"  {label:28s} n={len(M):4d}  DiD {est:+6.2f} pp  "
                  f"[{lo:+6.2f} ; {hi:+6.2f}]  p={p:.4f}")
            sens.append({"subset": label, "n_items": len(M), "did_pp": round(est, 2),
                         "ci_lo": round(lo, 2), "ci_hi": round(hi, 2),
                         "p_boot": round(p, 4)})
        pd.DataFrame(sens).to_csv(ROOT / "results" / "raw_leak_sensitivity.csv",
                                  index=False)

    clean_baseline(edge_ids, latent_ids)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("\n  'edge_stated' la ro that: mot canh viet bang loi thuong, con nguyen")
    print("  trong RAW. Khoang 12% item o ca ba mau.")
    print("\n  'latent_disclosed' yeu hon: no khong neu ten canh nao, chi cho biet")
    print("  co mot bien gay nhieu khong quan sat duoc. 31% toi 43%.")
    print("\n  KHONG cong them 'affects': cum do nam trong CAU HOI backadj")
    print("  ('To understand how X affects Y, ...'), moi nhanh deu can no, va dem")
    print("  no vao se bien mot mac ro 12% thanh 52%.")
    print("\n  HE QUA. Ro nay nam trong RAW va trong MOI dieu kien do thi nhu nhau,")
    print("  vi tat ca deu dung tren cung mot than bai da boc. No khong lam lech")
    print("  hieu ghep cap trong tung item. Nhung goi RAW la 'khong do thi' thi sai.")
    print("\n  KHONG doc no la 'loi ich do duoc la CAN DUOI'. Muc 2 kiem dieu do:")
    print("  bo item noi canh bang loi, DiD GIAM chu khong tang. Va phep tach theo")
    print("  'is unobserved' khong kiem duoc gi - no trung khit voi ba ho do thi.")
    print("\n  ghi ra results/raw_structure_leak.csv")
    return 0


def with_clean(tag, lex, imap):
    """A sample's answers plus its RAW_CLEAN run (scripts/run_clean_raw.sh)."""
    d = load(tag, lex, imap)
    f = ROOT / "results" / f"pilot_raw_cleanraw{tag}{lex}.csv"
    if not f.exists():
        return None
    e = attach_id(pd.read_csv(f), imap, f"cleanraw{tag}{lex}")
    # Off the latent families RAW_CLEAN IS RAW, prompt for prompt, and its answer
    # was read back from the cache. For some n600 cells the cache holds a SECOND
    # answer to that prompt, not the one this sample's RAW row carries: the prompt
    # was sent twice during the 2026-09-16 run and the later answer overwrote the
    # earlier in the cache (scripts/check_consistency.py, section 3). Taking RAW's
    # own row there keeps RAW_CLEAN minus RAW exactly zero wherever the prompts
    # are identical, so the contrast measures the latent sentence and nothing else.
    off = ~e.graph_id.isin(LATENT_FAMILIES)
    r = d[d.cond == "RAW"].set_index(["model", "item"])
    k = pd.MultiIndex.from_arrays([e.model[off], e.item[off]])
    for col in ("pred", "correct", "parsed"):
        e.loc[off, col] = r[col].reindex(k).values
    return pd.concat([d, e], ignore_index=True)


def did_base(tag, imap, base):
    """did_sample() with the no-graph baseline as a parameter."""
    K, L = with_clean(tag, "KEEP", imap), with_clean(tag, POOL_LEXICON, imap)
    qs = set(K.query_type.unique()) - ARITH - IDENT
    cols = []
    for m in TIER:
        kr, lr = cell(K, m, base, qs), cell(L, m, base, qs)
        ka, la = cell(K, m, "ORACLE", qs), cell(L, m, "ORACLE", qs)
        i = kr.index.intersection(lr.index).intersection(ka.index).intersection(la.index)
        if len(i) < 10:
            continue
        cols.append(pd.Series(((kr[i] - lr[i]) - (ka[i] - la[i])).values, index=i,
                              name=f"{tag}|{m}|{POOL_LEXICON}"))
    return pd.concat(cols, axis=1).groupby(level=0).mean()


def clean_baseline(edge_ids, latent_ids):
    """Section 3: the headline and the direct contrasts on a latent-free RAW.

    RAW_CLEAN is RAW without "X is unobserved." (src/prompts.py); on the seven
    families with no latent it IS RAW, prompt for prompt. The one leak left is
    det-counterfactual's structural equations, which cannot be stripped, so the
    fully clean headline also drops that query type. Every table's first row
    uses RAW and must reproduce the number already published from RAW; if it
    does not, the rows beneath it are not comparable, and the script stops.
    """
    print("\n" + "=" * 78)
    print("3. A LATENT-FREE BASELINE: RAW_CLEAN")
    print("=" * 78)
    imaps = {t: pd.read_csv(ROOT / "results" / f"_itemmap_{t}.csv") for t in SAMPLES}
    if not all((ROOT / "results" / f"pilot_raw_cleanraw{t}KEEP.csv").exists()
               for t in SAMPLES):
        print("  SKIPPED: run scripts/run_clean_raw.sh first.")
        return
    rows = []

    def add(section, sample, lexicon, quantity, est, lo, hi, p, n):
        rows.append(dict(section=section, sample=sample, lexicon=lexicon,
                         quantity=quantity, estimate_pp=round(est, 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2), p_boot=round(p, 4),
                         n_items=n))
        print(f"  {sample:9s} {lexicon:7s} {quantity:44s} {est:+6.2f} "
              f"[{lo:+6.2f} ; {hi:+6.2f}]  p={p:.4f}  n={n}")

    # 3a. The headline DiD, pooled over the three samples by CLadder id.
    print("\n  3a. Headline DiD, causal group, pooled n=490 convention\n")
    per = {b: pd.concat([did_base(t, imaps[t], b) for t in SAMPLES], axis=1)
           for b in ("RAW", "RAW_CLEAN")}
    ref = pd.read_csv(ROOT / "results" / "raw_leak_sensitivity.csv").set_index("subset")
    for b, label, drop in (("RAW", "RAW (the published headline)", set()),
                           ("RAW_CLEAN", "RAW_CLEAN", set()),
                           ("RAW", "RAW, drop det-counterfactual", edge_ids),
                           ("RAW_CLEAN", "RAW_CLEAN, drop det-counterfactual", edge_ids)):
        M = per[b].loc[[i for i in per[b].index if i not in drop]]
        est, lo, hi, p = boot(M)
        if label.startswith("RAW (the"):
            want = ref.loc["all items (headline)", "did_pp"]
            if round(est, 2) != want:
                raise SystemExit(f"RAW row gives {est:.2f}, headline is {want}: not comparable")
        add("3a headline DiD", "pooled", POOL_LEXICON, label, est, lo, hi, p, len(M))

    # 3b. What the latent sentence itself was worth, on the items that had it.
    print("\n  3b. RAW_CLEAN minus RAW on the latent families, causal group\n")
    for lex in ("KEEP", POOL_LEXICON):
        old, now = [], []
        for t in SAMPLES:
            D = with_clean(t, lex, imaps[t])
            qs = set(D.query_type.unique()) - ARITH - IDENT
            sfx = "" if t == "n600" else f"_{t}"
            f = ROOT / "results" / f"drift_check_raw{sfx}.csv"
            R = pd.read_csv(f) if f.exists() else None
            for m in TIER:
                c, r = cell(D, m, "RAW_CLEAN", qs), cell(D, m, "RAW", qs)
                i = [x for x in c.index.intersection(r.index) if x in latent_ids]
                old.append(pd.Series((c[i] - r[i]).values, index=i, name=f"{t}|{m}"))
                if R is not None:
                    q = R[(R.model == m) & (R.lexicon == lex) & (R.cond == "RAW")
                          & R.new_pred.notna()]
                    q = attach_id(q, imaps[t], f"drift {t}").set_index("id").new_correct
                    j = [x for x in c.index.intersection(q.index) if x in latent_ids]
                    now.append(pd.Series((c[j] - q[j]).values, index=j, name=f"{t}|{m}"))
        for label, cols in (("RAW_CLEAN minus RAW, RAW as first answered", old),
                            ("RAW_CLEAN minus RAW, RAW asked again today", now)):
            if not cols:
                continue
            W = pd.concat(cols, axis=1).groupby(level=0).mean()
            est, lo, hi, p = boot(W)
            add("3b latent sentence", "pooled", lex, label, est, lo, hi, p, len(W))

    # 3c. The direct contrasts of analyze_vs_raw.py, per sample, on either baseline.
    print("\n  3c. Direct contrasts, causal group, per sample (analyze_vs_raw convention)\n")
    vs = pd.read_csv(ROOT / "results" / "vs_raw.csv")
    for t in SAMPLES:
        for lex in ("KEEP", POOL_LEXICON):
            D = with_clean(t, lex, imaps[t])
            D = D[(D.parsed == 1) & ~D.query_type.isin(ARITH | IDENT)]
            for arm in ("ORACLE", "DR_k1"):
                for b in ("RAW", "RAW_CLEAN"):
                    cols = []
                    for m in sorted(D.model.unique()):
                        r = D[(D.model == m) & (D.cond == b)].set_index("item").correct
                        c = D[(D.model == m) & (D.cond == arm)].set_index("item").correct
                        i = r.index.intersection(c.index)
                        if len(i) >= 5:
                            cols.append(pd.Series((c[i] - r[i]).values, index=i))
                    v = pd.concat(cols, axis=1).mean(axis=1).dropna()
                    est, lo, hi, p = vs_boot(v)
                    if b == "RAW":
                        want = vs[(vs.query_group == "causal") & (vs["sample"] == t)
                                  & (vs.lexicon == lex) & (vs.cond == arm)].delta_pp.iloc[0]
                        if round(est, 2) != want:
                            raise SystemExit(f"{t} {lex} {arm} minus RAW gives {est:.2f}, "
                                             f"vs_raw.csv has {want}: not comparable")
                    add("3c direct contrast", t, lex, f"{arm} minus {b}", est, lo, hi, p, len(v))

    pd.DataFrame(rows).to_csv(ROOT / "results" / "raw_clean.csv", index=False)
    print("\n  wrote results/raw_clean.csv")


if __name__ == "__main__":
    raise SystemExit(main())

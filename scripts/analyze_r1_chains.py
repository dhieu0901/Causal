"""What DeepSeek-R1 writes when one edge of its graph is reversed.

    python scripts/analyze_r1_chains.py

R1 (scripts/run_r1.sh) answered the 86 causal items of the lex sample, names
anonymised, under RAW, ORACLE and DR_k1. It was right 84.9% of the time with
the correct graph and 61.6% with one edge reversed. Its reasoning, 3,000 to
8,000 words a call, sits in cache/ with every answer, already paid for. This
file reads it. No call is sent.

What is counted, per item, on the full chain (reasoning field plus answer; the
12 calls cut off at 8,000 tokens are read from their 16,000-token re-ask).
DR_k1 shows v -> u where the true graph has u -> v.

  restates shown    the chain asserts v -> u
  states true       the chain asserts u -> v. Under ORACLE that is simply the
                    edge it was given; under DR_k1 R1 had to get it from
                    somewhere else. For det-counterfactual items the prompt's
                    own equations state it, so those are counted apart.
  conflict words    contradict*, inconsisten*, conflict*, mismatch*, "does not
                    match / make sense / align", typo, reversed, backwards,
                    "opposite direction" anywhere in the chain
  conflict near     a conflict word within 400 characters of both names of the
     the edge       reversed pair: the chain is at least talking about that
                    edge when it says something does not fit
  no-effect         "no (causal / directed / direct) path / effect", "does not
     phrases        cause / affect / influence", "not a cause", "no way for":
                    how often the chain argues that the treatment cannot move
                    the outcome, counted per chain

"asserts" is analyze_chains.asserts: one name, a causal verb or an arrow, the
other name, in that order. The pseudowords are distinctive tokens, which makes
the name match cleaner than on real names, but a chain that abbreviates the
names ("let J = jyka") and then writes "J -> Q" is missed. Every rate is
therefore a lower bound on how often R1 says the thing, and each DR_k1 rate is
set against the same items under ORACLE and RAW, where the same misses occur.
Conflict words are a coarse net: R1 also uses them for arithmetic ("0.06 <
a, contradiction"), for awkward question wording, and for real words that the
anonymisation left in the prompt (REPORT section 6: residue). The ORACLE rate
is the floor for all of that.

For the ate and ett items, each reversal is also placed in the groups of
analyze_answer_change.py (does the shown graph change the answer, and does it
cut every X -> Y path), with R1's accuracy in each.

Exploratory; a reading of generated text, not of the computation behind it.

Writes: results/cladder/r1_chains.csv          one row per item
        results/cladder/r1_chains_summary.csv  the counted rates
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from analyze_answer_change import Matcher, draws_for
from analyze_chains import asserts, reversed_edge
from analyze_querygroup import ARITH, IDENT
from analyze_second_family import R1, load
from lexical import relabel_item, residue
from pilot import build_jobs, make_items
from prompts import parse_prose_graph, strip_structure
from runner import read_cached

RESULTS = ROOT / "results" / "cladder"
SEED = 20260907
TEMP, CAP, RECAP = 0.6, 8000, 16000          # scripts/run_r1.sh
LEX = dict(n_items=200, sample_kmax=1, kmax=1, seed=SEED, exclude_ids=None)
CONF = re.compile(r"(contradict\w*|inconsisten\w*|conflict\w*|mismatch\w*|"
                  r"does(?:n.t| not) (?:match|make sense|align)|typo|reversed|backwards|"
                  r"opposite direction)", re.IGNORECASE)
NEAR = 400
NOEFFECT = re.compile(r"(no (?:directed |causal |direct )?(?:path|effect|causal (?:effect|link|"
                      r"relationship))|(?:does not|doesn.t|cannot|can.t) (?:cause|affect|"
                      r"influence)|not a cause|no way for)", re.IGNORECASE)


def chain(prompt):
    rec = read_cached(R1, TEMP, prompt, max_tokens=CAP)
    if rec is None:
        return None
    if rec.get("finish") == "length":
        re_ask = read_cached(R1, TEMP, prompt, max_tokens=RECAP)
        rec = re_ask or rec
    return (rec.get("reasoning") or "") + "\n" + (rec.get("text") or "")


def conflict_near(text, u, v):
    for m in CONF.finditer(text):
        w = text[max(0, m.start() - NEAR):m.end() + NEAR].lower()
        if u.lower() in w and v.lower() in w:
            return True
    return False


def main() -> int:
    items = make_items(LEX["n_items"], SEED, 1, "full_v1.5_default.csv", None, True)
    jobs = [j for j in build_jobs(items, 1, SEED, ("DR",), "PSEUDO")
            if j["cond"] in ("RAW", "ORACLE", "DR_k1")
            and j["query_type"] not in ARITH | IDENT]
    by = {(j["item"], j["cond"]): j for j in jobs}

    A = load("r1", "lex", "PSEUDO", recap=True)
    A = A.set_index(["item", "cond"])

    rows, missing = [], 0
    for i in sorted({j["item"] for j in jobs}):
        if (i, "DR_k1") not in by:
            continue
        r = items.loc[i]
        prompt, _ = relabel_item(r.prompt, "PSEUDO", seed=str(r.id))
        edges = parse_prose_graph(strip_structure(prompt)[1])
        (u, v), _ = reversed_edge(edges, i, SEED)
        ch = {c: chain(by[(i, c)]["prompt"]) for c in ("RAW", "ORACLE", "DR_k1")}
        if any(t is None for t in ch.values()):
            missing += 1
            continue
        row = dict(item=i, id=int(r.id), family=r.graph_id, query_type=r.query_type,
                   true_edge=f"{u} -> {v}", shown_edge=f"{v} -> {u}",
                   residue=bool(residue(r.prompt, prompt, "PSEUDO")),
                   prompt_states_true=r.query_type == "det-counterfactual")
        for c, t in ch.items():
            tag = c.lower()
            row[f"{tag}_states_shown"] = asserts(t, v, u)
            row[f"{tag}_states_true"] = asserts(t, u, v)
            row[f"{tag}_conflict_words"] = bool(CONF.search(t))
            row[f"{tag}_conflict_near_edge"] = conflict_near(t, u, v)
            row[f"{tag}_no_effect_phrases"] = len(NOEFFECT.findall(t))
            row[f"{tag}_chars"] = len(t)
            a = A.loc[(i, c)] if (i, c) in A.index else None
            row[f"{tag}_pred"] = a.pred if a is not None and a.parsed == 1 else ""
            row[f"{tag}_correct"] = int(a.correct) if a is not None and a.parsed == 1 else None
        rows.append(row)
    if missing:
        raise SystemExit(f"{missing} items have a chain missing from cache/")
    T = pd.DataFrame(rows)

    # the groups of analyze_answer_change.py, for the ate/ett items
    G = draws_for("lex", Matcher(), kw=LEX, lexicons=("PSEUDO",))
    G = G[G.cond == "DR_k1"].set_index("item")
    T["group"] = T.item.map(G.group)
    T["route"] = T.item.map(G.route)
    T.to_csv(RESULTS / "r1_chains.csv", index=False)

    S = []

    def rate(what, mask, col, cond):
        x = T[mask]
        n = int(x[col].notna().sum()) if x[col].dtype == object else len(x)
        k = int(x[col].fillna(False).astype(bool).sum())
        S.append(dict(what=what, cond=cond, count=k, n=n, pct=round(100 * k / n, 1) if n else None))

    alli = T.item.notna()
    other = ~T.prompt_states_true
    for c in ("DR_k1", "ORACLE", "RAW"):
        t = c.lower()
        rate("restates the shown reversed edge v -> u", alli, f"{t}_states_shown", c)
        rate("states the true edge u -> v, prompt does not", other, f"{t}_states_true", c)
        rate("states the true edge u -> v, det-counterfactual", ~other, f"{t}_states_true", c)
        rate("conflict words anywhere", alli, f"{t}_conflict_words", c)
        rate("conflict words near both names of the pair", alli, f"{t}_conflict_near_edge", c)
        rate("conflict words near the pair, no residue", ~T.residue, f"{t}_conflict_near_edge", c)
    both = T.dr_k1_pred.ne("") & T.oracle_pred.ne("")
    S.append(dict(what="DR_k1 answer equals the ORACLE answer", cond="DR_k1 vs ORACLE",
                  count=int((T[both].dr_k1_pred == T[both].oracle_pred).sum()), n=int(both.sum()),
                  pct=round(100 * (T[both].dr_k1_pred == T[both].oracle_pred).mean(), 1)))
    for c in ("DR_k1", "ORACLE", "RAW"):
        x = T[f"{c.lower()}_correct"].dropna()
        S.append(dict(what="accuracy", cond=c, count=int(x.sum()), n=len(x),
                      pct=round(100 * x.mean(), 1)))
    for g in ("estimand kept", "estimand changed, answer kept", "answer changed"):
        for route in ("all", "no path", "backdoor"):
            x = T[T.group.eq(g) & (T.route.eq(route) if route != "all" else True)]
            for c in ("DR_k1", "ORACLE"):
                y = x[f"{c.lower()}_correct"].dropna()
                if len(y):
                    S.append(dict(what=f"accuracy, ate/ett, {g}, route {route}", cond=c,
                                  count=int(y.sum()), n=len(y), pct=round(100 * y.mean(), 1)))
    # Where the reversal cuts every X -> Y path on a Yes item: does R1 turn
    # to No, and does its chain then argue "no effect" more than under ORACLE?
    cut = T[T.group.eq("answer changed") & T.route.eq("no path")]
    flip = cut[cut.dr_k1_pred.eq("no") & cut.oracle_pred.eq("yes")]
    S.append(dict(what="path cut on a Yes item: DR_k1 says No where ORACLE said Yes",
                  cond="DR_k1 vs ORACLE", count=len(flip), n=len(cut),
                  pct=round(100 * len(flip) / len(cut), 1) if len(cut) else None))
    more = flip.dr_k1_no_effect_phrases > flip.oracle_no_effect_phrases
    S.append(dict(what="  of those, the DR_k1 chain has more no-effect phrases",
                  cond="DR_k1 vs ORACLE", count=int(more.sum()), n=len(flip),
                  pct=round(100 * more.mean(), 1) if len(flip) else None))
    S = pd.DataFrame(S)
    S.to_csv(RESULTS / "r1_chains_summary.csv", index=False)

    print("=" * 92)
    print("DEEPSEEK-R1 UNDER ONE REVERSED EDGE: WHAT THE CHAINS SAY (exploratory)")
    print("=" * 92)
    print(f"  {len(T)} items, {int(T.residue.sum())} with real-word residue, "
          f"{int(T.prompt_states_true.sum())} det-counterfactual\n")
    for r in S.itertuples():
        print(f"  {r.what:52s} {r.cond:16s} {r.count:3d}/{r.n:3d}  {r.pct:5.1f}%")
    print("\n  wrote results/cladder/r1_chains.csv, r1_chains_summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

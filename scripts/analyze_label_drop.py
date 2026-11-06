"""Do the registered verdicts survive dropping the questions whose label is in doubt?

    python scripts/analyze_label_drop.py

verify_sample_labels.py checks the label of every question this project sampled
and lists, per sample, the questions whose label a published CLadder error
decides and those whose label the question's own stated numbers cannot decide
(results/cladder/sample_labels.csv). Every registered test scores against the
published labels. This script reruns, with every such question of the
registered samples removed, each registered test that a CLadder label enters:

  Study 1, H1 to H4, three families      analyze_confirmatory.confirm
  Study 3, M1 to M3, R1, R2, two families analyze_b6.confirm
  S1 to S3, three families                analyze_new_stories.run_tests
  P1, P2, N1, N2, Llama and gpt-5.6-luna  analyze_names_premise.run_tests

The test functions are imported unchanged. The only change is that every read
of a record file (results/cladder/raw/pilot_raw_*.csv) drops the rows of those
questions, so ladder cells, new-story cells and Study 3's draws lose the same
questions as the main cells. The GPT-4.1 family's P1 to N2 have no verdict (its
run stopped before N2's cell) and are not rerun. CaLM's answers were recomputed
from each question's graph (analyze_calm), and the path probes are scored
against the graph, not a CLadder label, so neither enters here. Each row carries
the published primary estimate and verdict beside the new ones. Exploratory, not
registered.

Writes results/cladder/label_drop.csv.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = ROOT / "results" / "cladder"
RAW = RESULTS / "raw"
LABEL = "flagged labels dropped"
# the record files of the registered samples: Study 1 (conf...), its follow-ups
# (premise...), Study 3 (b6...)
REGISTERED = re.compile(r"^(conf|llamaconf|lunaconf|premise|llamapremise|lunapremise|b6|b6llama)"
                        r"(KEEP|PSEUDO|ladder|NEWSTORY)")
ID_COLUMNS = ("ids_decided", "ids_not_decidable", "ids_label_wrong_otherwise")


def flagged() -> dict[str, set[int]]:
    """sample tag -> ids of its questions whose label is in doubt."""
    L = pd.read_csv(RESULTS / "sample_labels.csv")
    L = L[L["sample"].astype(str).str.match(REGISTERED)]
    out = {}
    for _, r in L.iterrows():
        ids = set()
        for c in ID_COLUMNS:
            if pd.notna(r[c]):
                ids |= {int(x) for x in str(r[c]).split(",") if x.strip()}
        out[r["sample"]] = ids
    return out


def without(drop: set[int]):
    """pd.read_csv, with the rows of `drop` removed from every record file."""
    read_csv, raw = pd.read_csv, os.path.normcase(str(RAW.resolve()))

    def read(path, *a, **k):
        d = read_csv(path, *a, **k)
        if isinstance(path, (str, Path)):
            p = Path(path)
            if (os.path.normcase(str(p.resolve().parent)) == raw and p.name.startswith("pilot_raw_")
                    and "id" in d.columns):
                d = d[~d["id"].isin(drop)].reset_index(drop=True)
        return d
    return read


def published() -> pd.DataFrame:
    P = pd.concat([
        pd.read_csv(RESULTS / "confirmatory.csv").assign(study="Study 1"),
        pd.read_csv(RESULTS / "b6.csv").assign(study="Study 3", family="gpt"),
        pd.read_csv(RESULTS / "b6_llama.csv").assign(study="Study 3", family="llama"),
        pd.read_csv(RESULTS / "new_stories.csv").assign(study="New stories"),
        pd.read_csv(RESULTS / "names_premise.csv").assign(study="Premise and name roles"),
    ], ignore_index=True)
    P = P[P.analysis == "primary"]
    if P.duplicated(["study", "family", "test"]).any():
        raise SystemExit("a published primary test appears twice")
    return P.set_index(["study", "family", "test"])


def main() -> int:
    F = flagged()
    drop = set().union(*F.values())
    if not drop:
        raise SystemExit("no flagged question in the registered samples: nothing to drop")
    for tag in sorted(F):
        print(f"  {tag:28s} {sorted(F[tag])}")
    print(f"\n  dropped from every record file: {sorted(drop)}")
    pub = published()                              # read before the filter goes in

    pd.read_csv = without(drop)
    import analyze_b6 as B6
    import analyze_confirmatory as AC
    import analyze_names_premise as NP
    import analyze_new_stories as NS
    from pilot import read_ids

    rows = []
    print("\n  Study 1: H1 to H4")
    cls = AC.classes(AC.CONF["n"], AC.CONF["seed"], read_ids(AC.CONF["exclude"]))
    for family, (models, prefix, _) in AC.FAMILIES.items():
        S = AC.conf_data(prefix, None)
        if S is not None:
            print(f"  {family}")
            rows += [dict(study="Study 1", **r) for r in AC.confirm(family, S, models, cls, LABEL)]

    print("\n  Study 3: M1 to M3, R1, R2")
    for family, prefix in (("gpt", "b6"), ("llama", "b6llama")):
        X = B6.b6_draws(prefix=prefix)
        if X is not None:
            print(f"  {family}: {len(X)} draws on {X.item.nunique()} items")
            rows += [dict(study="Study 3", family=family, **r) for r in B6.confirm(X, LABEL)]

    print("\n  new stories: S1 to S3")
    for family, (models, prefix, recap) in NS.FAMILIES.items():
        cells = NS.load(prefix, models, "primary", recap)
        if cells is not None:
            rows += [dict(study="New stories", **r)
                     for r in NS.rows_of(family, LABEL, NS.run_tests(cells))]

    print("\n  premise and name roles: P1, P2, N1, N2")
    NP.SHOWN.update(NP.third_shown())
    for family in ("llama", "luna"):
        cells = NP.load(family, "primary")
        if cells is not None:
            first = pd.concat(cells.values())
            story_of = first.groupby(NP.item_of(first.index)).story_id.first()
            rows += [dict(study="Premise and name roles", **r)
                     for r in NP.rows_of(family, LABEL, NP.run_tests(cells, story_of))]

    R = pd.DataFrame(rows)
    key = list(zip(R.study, R.family, R.test))
    for col in ("estimate_pp", "ci_lo", "ci_hi", "verdict"):
        R[f"{col}_published"] = [pub[col].get(k) for k in key]
    R["n_items_published"] = [pub["n_items"].get(k) for k in key]
    if "n_draws" in pub:
        R["n_draws_published"] = [pub["n_draws"].get(k) for k in key]
    if R.verdict_published.isna().any():
        raise SystemExit("a rerun test has no published primary row")
    R["same_verdict"] = R.verdict == R.verdict_published
    R["shift_pp"] = (R.estimate_pp - R.estimate_pp_published).round(2)
    R.to_csv(RESULTS / "label_drop.csv", index=False)

    with pd.option_context("display.width", 230, "display.max_rows", 100):
        print("\n" + R[["study", "family", "test", "estimate_pp", "ci_lo", "ci_hi",
                        "estimate_pp_published", "shift_pp", "verdict",
                        "verdict_published"]].to_string(index=False))
    same = int(R.same_verdict.sum())
    print(f"\n  {same} of {len(R)} registered verdicts unchanged; largest shift "
          f"{R.shift_pp.abs().max():.2f} points")
    print("  wrote results/cladder/label_drop.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

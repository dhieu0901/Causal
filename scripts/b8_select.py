"""Which B8 calls are sent. Writes prereg/b8_selection.csv; sends nothing.

    python scripts/b8_select.py

prereg/B8.md registers a targeted design: the five tests use only some draws,
so only those are sent. The registered classifier (analyze_answer_change.
draws_for) gives every draw of the B8 sample its group before any call:

  reversals   DR_k_NI and DR_k_U for the draws in "answer changed, path cut"
              (the shown graph leaves no directed X -> Y path and implies an
              answer other than the label): T0, T3, T4
  deletions   ED_k_NI for the draws in that group: T1
  spurious    FE_k_NI for every draw: T2
  references  ORACLE_NI for every item with a sent draw; ORACLE_U and RAW for
              every item with a sent reversal (RAW is descriptive)

Real names (KEEP) only. Nothing has been sent when this runs, so the replay
check is skipped here; analyze_b8.py runs it on exactly the listed draws.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from analyze_answer_change import Matcher, draws_for
from analyze_b8 import B8, LEXICON, SELECTION, cut
from pilot import read_ids


def main() -> int:
    kw = B8 | {"exclude_ids": read_ids("prereg/excluded_ids_b8.txt")}
    G = draws_for("b8", Matcher(), kw=kw, lexicons=(LEXICON,), arms=("DR", "ED", "FE"),
                  skip_replay=True)
    G = G[G.lexicon == LEXICON]
    arm = G.cond.str[:2]
    dr, ed, fe = cut(G[arm == "DR"]), cut(G[arm == "ED"]), G[arm == "FE"]
    ids = dict(zip(G.item, G.id))
    rows = ([(r.item, f"{r.cond}_NI") for r in dr.itertuples()]
            + [(r.item, f"{r.cond}_U") for r in dr.itertuples()]
            + [(r.item, f"{r.cond}_NI") for r in ed.itertuples()]
            + [(r.item, f"{r.cond}_NI") for r in fe.itertuples()]
            + [(i, "ORACLE_NI") for i in set(dr.item) | set(ed.item) | set(fe.item)]
            + [(i, c) for i in set(dr.item) for c in ("ORACLE_U", "RAW")])
    S = pd.DataFrame(rows, columns=["item", "cond"])
    if S.duplicated().any():
        raise SystemExit("a job is listed twice")
    S = (S.assign(lexicon=LEXICON, id=S.item.map(ids))[["lexicon", "item", "id", "cond"]]
         .sort_values(["item", "cond"]).reset_index(drop=True))
    S.to_csv(SELECTION, index=False, lineterminator="\n")
    print(S.cond.str.replace(r"_k\d", "_k", regex=True).value_counts().sort_index().to_string())
    print(f"{len(S)} calls per model on {S.item.nunique()} items")
    print(f"wrote {SELECTION.relative_to(ROOT).as_posix()}  "
          f"sha256 {hashlib.sha256(SELECTION.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Where is the interaction concentrated - by graph size, and by query type?

    python scripts/analyze_moderators.py

REPORT.md section 4.6 carries two tables that decide how the headline may be
worded: the DiD split by node count, and the DiD split by query_type. They are
the evidence for the sentence "this is mainly an ATE result", which is the most
deflationary sentence in the report. Until 2026-09-23 no script produced either.

It does NOT define its own DiD. `pool_samples.did_sample` is the project's one
definition, and rebuilding it here is how a fifth convention gets born - the
bootstrap p had three in circulation for exactly that reason.

A note on the n. REPORT quoted these splits against a total of n=86, DiD +14.35.
That total is itself historical: no script in the repo computes it. The only
place it survives is a printed sentence in analyze_structure_arms.py describing
an earlier run. The current pipeline gives n=85 and +13.92 on the same sample,
and the per-cell splits below move with it - two of them change verdict. Prefer
this file's output; the older figures have no source to check them against.

Node counts come from the graph family, via src/perturb.py's FAMILY_STRUCTURE,
not from a table typed in here.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from perturb import FAMILY_STRUCTURE, to_edges
from pool_samples import NBOOT, SEED, did_sample
from stats import boot_p

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLE = "lex"          # section 4.6 is measured on the 174-item lexical sample
MIN_ITEMS = 8


def nodes_of_family(fam: str) -> int:
    """Number of distinct nodes in a family's edge list."""
    spec = FAMILY_STRUCTURE.get(fam)
    if spec is None:
        return 0
    return len({n for e in to_edges(spec) for n in e})


def boot_rows(v: pd.Series, seed=SEED, n=NBOOT):
    x = v.dropna().values
    if len(x) < MIN_ITEMS:
        return None
    rng = np.random.default_rng(seed)
    out = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    return (100 * x.mean(), 100 * np.percentile(out, 2.5),
            100 * np.percentile(out, 97.5), boot_p(out, n), len(x))


def main() -> int:
    # The saved map is written and row-by-row verified by pool_samples.py, so
    # it is read rather than regenerated - regenerating it here would be a
    # second place for the item ordering to drift.
    imap_path = ROOT / "results" / f"_itemmap_{SAMPLE}.csv"
    if not imap_path.exists():
        raise SystemExit(f"thieu {imap_path.name}: chay pool_samples.py truoc")
    imap = pd.read_csv(imap_path)
    W = did_sample(SAMPLE, imap, ["PSEUDO"], arm="ORACLE")
    if W.empty:
        raise SystemExit("khong dung duoc ma tran DiD")
    per_item = W.mean(axis=1)

    full = pd.read_csv(ROOT / "data" / "full_v1.5_default.csv",
                       low_memory=False).set_index("id")
    fam = per_item.index.to_series().map(full.graph_id)
    qty = per_item.index.to_series().map(full.query_type)
    nodes = fam.map(nodes_of_family)

    print("=" * 84)
    print("4.6 HIEU UNG TAP TRUNG O DAU")
    print("=" * 84)
    tot = boot_rows(per_item)
    print(f"\n  Toan nhom causal, mau {SAMPLE}: DiD {tot[0]:+.2f} pp "
          f"[{tot[1]:+.2f} ; {tot[2]:+.2f}] p={tot[3]:.4f} n={tot[4]}\n")

    rows = []
    print("  --- theo SO NUT cua do thi ---")
    for label, mask in [("3 nut", nodes == 3), (">= 4 nut", nodes >= 4)]:
        r = boot_rows(per_item[mask.values])
        if r is None:
            print(f"  {label:26s} qua it item")
            continue
        print(f"  {label:26s} n={r[4]:3d}  {r[0]:+7.2f}  "
              f"[{r[1]:+7.2f} ; {r[2]:+7.2f}]  p={r[3]:.4f}")
        rows.append({"moderator": "n_nodes", "level": label, "n_items": r[4],
                     "did_pp": round(r[0], 2), "ci_lo": round(r[1], 2),
                     "ci_hi": round(r[2], 2), "p_boot": round(r[3], 4)})

    print("\n  --- theo QUERY_TYPE ---")
    for q in sorted(qty.dropna().unique()):
        r = boot_rows(per_item[(qty == q).values])
        if r is None:
            continue
        print(f"  {q:26s} n={r[4]:3d}  {r[0]:+7.2f}  "
              f"[{r[1]:+7.2f} ; {r[2]:+7.2f}]  p={r[3]:.4f}")
        rows.append({"moderator": "query_type", "level": q, "n_items": r[4],
                     "did_pp": round(r[0], 2), "ci_lo": round(r[1], 2),
                     "ci_hi": round(r[2], 2), "p_boot": round(r[3], 4)})

    # The deflationary test: does the effect survive without ate?
    r = boot_rows(per_item[(qty != "ate").values])
    if r:
        print(f"\n  --- BO 'ate' RA ---")
        print(f"  {'con lai':26s} n={r[4]:3d}  {r[0]:+7.2f}  "
              f"[{r[1]:+7.2f} ; {r[2]:+7.2f}]  p={r[3]:.4f}")
        rows.append({"moderator": "drop_ate", "level": "without ate",
                     "n_items": r[4], "did_pp": round(r[0], 2),
                     "ci_lo": round(r[1], 2), "ci_hi": round(r[2], 2),
                     "p_boot": round(r[3], 4)})

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "moderators.csv", index=False)

    print("\n" + "=" * 84)
    print("HOW TO READ THIS")
    print("=" * 84)
    print("\n  Cac o nay nho - vai chuc item moi o - nen CI rong va khong o nao")
    print("  chiu duoc mot phep hieu chinh boi. Doc chung nhu HUONG, khong phai")
    print("  nhu ket qua rieng le. Dieu chung noi la hieu ung KHONG deu: no tap")
    print("  trung o do thi lon hon va o truy van ate.")
    print("\n  ghi ra results/moderators.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

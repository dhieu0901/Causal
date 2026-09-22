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
alike, so it does not bias a within-item paired contrast. It does mean the
measured benefit of supplying a graph is a LOWER bound.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from prompts import strip_structure

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SAMPLES = ["lex", "n600", "price400"]
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
    print("  hieu ghep cap trong tung item. No co nghia la loi ich do duoc cua viec")
    print("  cap do thi la mot CAN DUOI - nen goi RAW la 'khong do thi' thi sai.")
    print("\n  ghi ra results/raw_structure_leak.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

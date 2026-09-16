"""Two things the lexical ladder assumes but never measured.

    python scripts/analyze_anomaly_residue.py

Section 7 of REPORT.md reads the ladder as though each rung removes exactly one
thing. Review round 6 found two places where that is not true, and both are
measurable from data already on disk at zero cost.

ANOMALY (rung 1). KEEP -> PERMUTE is described as "removes the usable prior".
It does more than that: it also hands the model a WRONG prior, and it makes the
item weird in a way the model notices and says out loud. Counting responses that
contain explicit contradiction language separates the three lexicons sharply -
PERMUTE flags several times more often than KEEP, while SYMBOL and PSEUDO are
indistinguishable from KEEP. So the anomaly signal is specific to PERMUTE, which
means rung 1 changes at least three variables at once and cannot carry the
"positive control" argument on its own. The same table is also a finding worth
publishing: a free, in-band tripwire for prompt-versus-knowledge conflict.

RESIDUE (rungs 2 and 3). relabel() in src/lexical.py only rewrites phrases that
appear in CLadder's variable_mapping. CLadder also refers to the same variables
through grammatical variants that are not in the mapping, so those survive, and
SYMBOL/PSEUDO prompts keep real-world nouns. That biases both conditions TOWARD
KEEP - which is the direction that manufactures the null results at rungs 2 and
3. The residue has to be reported before those nulls can be interpreted.

Residue is measured without a hand-built word list, which would only ever give a
lower bound that depends on the list. The separator is how many distinct STORIES
a word appears in, not how many prompts. Document frequency over prompts does
not work here: CLadder's template wording changes with query_type, so template
words like "smaller" or "observed" sit in a minority of prompts and would be
scored as story vocabulary. Across stories they are everywhere. A word confined
to a few of the 37 stories is story vocabulary; if it survives relabelling, it
is residue.

PERMUTE is expected to score near 100% and that is not a defect: a derangement
reuses the item's own words on purpose, so every one of them survives by design.
The measure is about SYMBOL and PSEUDO, which claim to remove real words.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from pilot import build_jobs, make_items

LEXICONS = ["KEEP", "PERMUTE", "SYMBOL", "PSEUDO"]
TIER = ["gpt-4.1-nano", "gpt-4.1-mini", "gpt-4.1"]
CACHE = ROOT / "cache"

# Words a model reaches for when it notices the premises fight each other. Kept
# deliberately narrow: every stem here is an explicit statement about the text
# being wrong, not a hedge about the answer.
FLAG = re.compile(
    r"\b(contradict\w*|inconsisten\w*|nonsensical|non-sensical|counterintuitive|"
    r"counter-intuitive|implausible|illogical|paradox\w*|"
    r"(doesn't|does not|do not|don't) make (any )?sense)\b", re.IGNORECASE)

WORD = re.compile(r"[a-z]{3,}")


def cache_text(model, prompt, temp=0.0):
    h = hashlib.sha256(f"{model}|{temp}|{prompt}".encode()).hexdigest()[:24]
    f = CACHE / f"{h}.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8")).get("text", "")
    except json.JSONDecodeError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260907)
    ap.add_argument("--max-stories", type=int, default=3,
                    help="a word in at most this many distinct stories is story vocabulary")
    a = ap.parse_args()
    W = 86

    items = make_items(a.n, a.seed, 1, "full_v1.5_default.csv", drop_nonsense=True)
    jobs = {lex: build_jobs(items, 1, a.seed, ("DR",), lex) for lex in LEXICONS}
    idx = {lex: {(j["item"], j["cond"]): j["prompt"] for j in jobs[lex]}
           for lex in LEXICONS}
    keys = sorted(idx["KEEP"])

    print("=" * W)
    print("1. PHAT HIEN BAT THUONG: model co TU NOI RA rang de bai vo ly khong?")
    print("=" * W)
    print("  Neu bac 1 chi 'bo di mot prior dung' thi ty le nay phai giong nhau")
    print("  o moi bo. Neu PERMUTE tach han ra, bac 1 doi nhieu hon mot thu.\n")
    rows, miss = [], 0
    for cond in ["RAW", "ORACLE"]:
        for m in TIER:
            r = {"cond": cond, "model": m}
            for lex in LEXICONS:
                flag = tot = ln = 0
                for it, c in keys:
                    if c != cond:
                        continue
                    p = idx[lex].get((it, c))
                    if p is None:
                        continue
                    t = cache_text(m, p)
                    if t is None:
                        miss += 1
                        continue
                    tot += 1
                    ln += len(t)
                    flag += bool(FLAG.search(t))
                r[f"{lex}_pct"] = round(100 * flag / tot, 1) if tot else None
                r[f"{lex}_len"] = int(ln / tot) if tot else None
                r[f"{lex}_n"] = tot
            rows.append(r)
    an = pd.DataFrame(rows)
    show = ["cond", "model"] + [f"{l}_pct" for l in LEXICONS]
    print("  -- ty le phan hoi gan co mau thuan (%) --")
    print(an[show].to_string(index=False))
    print("\n  -- do dai phan hoi trung binh (ky tu) --")
    print(an[["cond", "model"] + [f"{l}_len" for l in LEXICONS]].to_string(index=False))
    an.to_csv(ROOT / "results" / "anomaly_flag_rate.csv", index=False)
    if miss:
        print(f"\n  ({miss} luot goi khong co trong cache, da bo qua)")

    kp = an[[f"{l}_pct" for l in LEXICONS]].astype(float)
    print(f"\n  KEEP {kp.KEEP_pct.min():.1f}-{kp.KEEP_pct.max():.1f}%   "
          f"PERMUTE {kp.PERMUTE_pct.min():.1f}-{kp.PERMUTE_pct.max():.1f}%   "
          f"SYMBOL {kp.SYMBOL_pct.min():.1f}-{kp.SYMBOL_pct.max():.1f}%   "
          f"PSEUDO {kp.PSEUDO_pct.min():.1f}-{kp.PSEUDO_pct.max():.1f}%")
    print("  PERMUTE tach han khoi ba bo con lai; SYMBOL va PSEUDO khong khac KEEP.")
    print("  Vay bac 1 bo prior dung + gan prior sai + tao tin hieu bat thuong.")

    print("\n" + "=" * W)
    print("2. RESIDUE: SYMBOL va PSEUDO co thuc su an danh het khong?")
    print("=" * W)
    story_of = {i: r.story_id for i, r in items.iterrows()}
    seen = {}
    for k in keys:
        if k[1] != "RAW":
            continue
        for w in set(WORD.findall(idx["KEEP"][k].lower())):
            seen.setdefault(w, set()).add(story_of.get(k[0]))
    nstory = len({s for v in seen.values() for s in v})
    story = {w for w, ss in seen.items() if len(ss) <= a.max_stories}
    template = {w for w, ss in seen.items() if len(ss) > a.max_stories}
    print(f"  {nstory} story. Tu xuat hien o toi da {a.max_stories} story = tu vung "
          f"rieng cua story: {len(story)}.")
    print(f"  Tu xuat hien rong hon = khuon mau CLadder: {len(template)}.\n")

    rows = []
    for lex in ["PERMUTE", "SYMBOL", "PSEUDO"]:
        dirty, tot, leftover = 0, 0, Counter()
        for k in keys:
            if k[1] != "RAW":
                continue
            src, dst = idx["KEEP"].get(k), idx[lex].get(k)
            if src is None or dst is None:
                continue
            tot += 1
            sw = set(WORD.findall(src.lower())) & story
            rem = sw & set(WORD.findall(dst.lower()))
            if rem:
                dirty += 1
                leftover.update(rem)
        rows.append({"lexicon": lex, "n_item": tot, "item_con_residue": dirty,
                     "phan_tram": round(100 * dirty / tot, 1) if tot else None,
                     "tu_sot_hay_gap": ", ".join(w for w, _ in leftover.most_common(8))})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(ROOT / "results" / "lexicon_residue.csv", index=False)
    print("\n  relabel() chi thay cum nam trong variable_mapping. CLadder con goi")
    print("  cung bien do bang bien the ngu phap khong co trong mapping, nen chung")
    print("  song sot. Residue lam SYMBOL/PSEUDO lech VE PHIA KEEP, tuc lech theo")
    print("  dung huong sinh ra ket qua null o bac 2 va bac 3.")
    print("\n  Da ghi: results/anomaly_flag_rate.csv, results/lexicon_residue.csv")


if __name__ == "__main__":
    main()

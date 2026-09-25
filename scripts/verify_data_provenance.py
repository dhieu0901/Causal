"""Check that every file in data/ is bit-identical to its upstream release.

Nothing else in this repo proves where data/ came from. The project renames two
of the files on download, the two upstream channels carry DIFFERENT files, and
data/ is gitignored, so a corrupted or mismatched copy would sit there silently
and every number downstream would be wrong in a way no analysis script could see.

The hashes below are git blob SHA-1, i.e. sha1(b"blob <len>\\0" + content). Both
GitHub's contents API and HuggingFace's tree API report exactly that value, so
the expected column is copied verbatim from upstream rather than computed here.

Read before editing: the two JSON files exist ONLY on GitHub and the CSV files
exist ONLY on HuggingFace. "Download from GitHub or HuggingFace" is wrong advice
- you need both, and README said "or" until 2026-09-22.

Exit code 1 if any file is missing or mismatched, so it can gate a commit.
"""
import hashlib
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cladder"

GH = "github causalNLP/cladder@main:data"
HF = "huggingface causal-nlp/CLadder@main:data"

# local name -> (upstream name, upstream channel, git blob sha1)
EXPECTED = {
    "cladder-meta.json": (
        "cladder-v1-meta-models.json", GH, "1d0b84bfe4b13f43ec8d8c92616c1d0d4e1d1df1"),
    "cladder-questions.json": (
        "cladder-v1-questions.json", GH, "3c8bd92e478be700cc3b932ec02cefb1951d3330"),
    "full_v1.5_default.csv": (
        "full_v1.5_default.csv", HF, "d22fdfca196a29a77db90e682e226f1ce6a3ed6e"),
    "test-anticommonsense-v1.5.csv": (
        "test-anticommonsense-v1.5.csv", HF, "e8a9d34834687907a0f95af56e700bdf6b402493"),
    "test-balanced-v1.5.csv": (
        "test-balanced-v1.5.csv", HF, "f55945096c12f74762fd2823906ef1ad33806e74"),
    "test-commonsense-v1.5.csv": (
        "test-commonsense-v1.5.csv", HF, "45cc0a3a63e87f1a00f5a55c4daf54f1fe2b9133"),
    "test-easy-v1.5.csv": (
        "test-easy-v1.5.csv", HF, "9925483a2790a9758986e71e5bc035d85ce4388f"),
    "test-hard-v1.5.csv": (
        "test-hard-v1.5.csv", HF, "45917b92043dfbcf0bbef015d97c2ecbffac541f"),
    "test-noncommonsense-v1.5.csv": (
        "test-noncommonsense-v1.5.csv", HF, "da418d6c8d88086ffe71714dd9efeebbf1686ddf"),
}

# Only full_v1.5_default.csv is usable. Every test-* split ships the world
# description with the question stripped off, so scoring one yields about 50%
# and that looks exactly like a finding. verify() re-measures this rather than
# trusting the claim, because the claim said "three files" until it was checked
# and the real number is six.
REQUIRED_QUESTION_RATE = {"full_v1.5_default.csv": 100.0}


def blob_sha1(path):
    b = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


def main():
    bad = 0
    print("=" * 78)
    print("  XUAT XU DU LIEU - doi chieu voi ban phat hanh goc")
    print("=" * 78)
    print(f"\n  {'file':34s} {'kenh':12s} ket qua")
    print("  " + "-" * 74)
    for name, (upstream, channel, want) in EXPECTED.items():
        path = DATA / name
        tag = channel.split()[0]
        if not path.exists():
            print(f"  {name:34s} {tag:12s} THIEU")
            bad += 1
            continue
        got = blob_sha1(path)
        if got != want:
            print(f"  {name:34s} {tag:12s} LECH")
            print(f"  {'':34s} {'':12s}   local {got}")
            print(f"  {'':34s} {'':12s}   goc   {want}")
            bad += 1
        else:
            rename = "" if upstream == name else f"  (goc: {upstream})"
            print(f"  {name:34s} {tag:12s} khop{rename}")

    print("\n  Hai file JSON chi co tren GitHub. Cac file CSV chi co tren HuggingFace.")
    print("  Phai tai tu CA HAI noi, khong phai mot trong hai.\n")

    # Re-measure the emptiness of the test splits instead of citing it.
    try:
        import pandas as pd
    except ImportError:
        print("  (bo qua phan do dau '?': khong co pandas)")
        pd = None

    if pd is not None:
        print("  Ty le prompt co dau '?' - do lai, khong trich:")
        print(f"\n  {'file':34s} {'n':>8s} {'co ?':>8s} {'cot':>5s}")
        print("  " + "-" * 58)
        empty = []
        for name in EXPECTED:
            if not name.endswith(".csv"):
                continue
            path = DATA / name
            if not path.exists():
                continue
            df = pd.read_csv(path, low_memory=False)
            pct = 100.0 * df["prompt"].astype(str).str.contains(r"\?").mean()
            print(f"  {name:34s} {len(df):>8,d} {pct:>7.2f}% {len(df.columns):>5d}")
            want_rate = REQUIRED_QUESTION_RATE.get(name)
            if want_rate is not None and abs(pct - want_rate) > 0.005:
                print(f"  {'':34s} -> cho doi {want_rate:.2f}%, KHONG KHOP")
                bad += 1
            if want_rate is None and pct > 0.005:
                print(f"  {'':34s} -> cho doi 0%, KHONG KHOP")
                bad += 1
            if want_rate is None:
                empty.append(name)
        print(f"\n  {len(empty)}/{len(empty)} file test-* khong chua mot dau hoi nao.")
        print("  Cham diem tren bat ky file nao trong so do se ra khoang 50% vi mo hinh")
        print("  buoc phai doan - con so do trong y het mot phat hien ve nhan thuc.")
        print("  Chi full_v1.5_default.csv dung duoc, va no la file duy nhat con cot")
        print("  question_property de gan nhan loai tu vung.")

    print("\n" + "=" * 78)
    if bad:
        print(f"  {bad} van de. Tai lai tu dung kenh truoc khi chay bat cu phan tich nao.")
        return 1
    print("  Toan bo data/ khop tuyet doi voi ban phat hanh goc.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

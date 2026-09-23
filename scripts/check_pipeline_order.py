"""Does any script read a results file that a LATER script writes?

    python scripts/check_pipeline_order.py

Why this exists. Running the analyses in alphabetical order put analyze_types.py
before induction.py, which WRITES the file analyze_types.py reads. The whole
pipeline still exited 0; it just silently scored the new analysis against the
previous run's data, and the reproducibility check then reported a difference
that looked like non-determinism and was not.

Why it parses instead of greps. The first version of this check was a regular
expression over the source, and on 2026-09-23 it reported "0 violations" for an
edge that had just been introduced - a read written as

    ref = ROOT / "results" / "structure_arms.csv"
    ...
    pd.read_csv(ref)

because the filename and the read were on different lines. A grep sees text; a
dependency is a value flowing through a variable. So this walks the AST and
propagates simple local assignments.

What it deliberately does NOT do: follow a filename through a function argument,
a dict, or a loop variable. Those exist in this repo (the f-string tags like
f"pilot_raw_{tag}.csv"), and they are reported separately as DYNAMIC rather than
guessed at. A checker that quietly guesses is the thing this file replaces.

Exit code 1 if an ordering violation is found, so it can gate a commit.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

# The canonical dependency order. Kept here rather than in a shell script so the
# check travels with the repository.
ORDER = [
    "verify_data_provenance", "verify_groundtruth", "verify_labels",
    "verify_counterfactual", "audit_cladder_arithmetic", "verify_explanations",
    "pool_samples",
    "induction", "induction_baselines",
    "analyze_lexical", "analyze_types",
    "analyze_vs_raw", "classify_perturbations", "analyze_structure_arms",
    "analyze_by_family", "analyze_querygroup", "analyze_dose", "analyze_ladder5",
    "analyze_pilot", "analyze_instruction", "analyze_prior_strength",
    "analyze_chains", "analyze_errortypes_lexical", "analyze_anomaly_residue",
    "analyze_budget_paired", "analyze_moderators", "analyze_falsification",
    "compare_price_lexicon", "analyze_price_paired",
    "feasibility", "make_figures",
    "measure_raw_leak",
    "check_numbers", "check_pipeline_order", "verify_determinism",
]

READERS = {"read_csv", "read_json"}
WRITERS = {"to_csv", "to_json"}


class Flow(ast.NodeVisitor):
    """Collects results/*.csv reads and writes, following local assignments."""

    def __init__(self):
        self.env: dict[str, str] = {}      # varname -> filename
        self.reads: set[str] = set()
        self.writes: set[str] = set()
        # f-string filenames keep their read/write side. Lumping them into one
        # "dynamic" bucket hid most of this pipeline: nearly every file here is
        # tagged (pilot_raw_{tag}.csv), so a single bucket meant the check
        # covered the handful of untagged files and quietly skipped the rest.
        self.dyn_reads: set[str] = set()
        self.dyn_writes: set[str] = set()

    # -- resolving an expression to a results/ filename --------------------

    def _name(self, node) -> str | None:
        """A results/<file> path, or None. Handles ROOT / "results" / "x.csv"."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value if node.value.endswith((".csv", ".json")) else None
        if isinstance(node, ast.JoinedStr):
            parts = ["{}" if isinstance(v, ast.FormattedValue)
                     else getattr(v, "value", "") for v in node.values]
            s = "".join(parts)
            return s if s.endswith((".csv", ".json")) else None
        if isinstance(node, ast.Name):
            return self.env.get(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            # ROOT / "results" / "x.csv"  ->  take the rightmost operand
            return self._name(node.right)
        return None

    # -- visiting ----------------------------------------------------------

    def visit_Assign(self, node):
        val = self._name(node.value)
        if val:
            for t in node.targets:
                if isinstance(t, ast.Name):
                    self.env[t.id] = val
        self.generic_visit(node)

    def visit_Call(self, node):
        fn = node.func
        attr = fn.attr if isinstance(fn, ast.Attribute) else None
        if attr in READERS and node.args:
            self._record(self._name(node.args[0]), self.reads, self.dyn_reads)
        elif attr in WRITERS and node.args:
            self._record(self._name(node.args[0]), self.writes, self.dyn_writes)
        self.generic_visit(node)

    def _record(self, name, exact, dynamic):
        if not name:
            return
        (dynamic if "{}" in name else exact).add(name)


def scan(path: Path) -> Flow:
    f = Flow()
    # Two passes: the first fills env with assignments that appear after a use,
    # which happens when a module-level constant is defined below a helper.
    tree = ast.parse(path.read_text(encoding="utf-8"))
    f.visit(tree)
    f.visit(tree)
    return f


def main() -> int:
    flows = {}
    for p in sorted(SCRIPTS.glob("*.py")):
        try:
            flows[p.stem] = scan(p)
        except SyntaxError as e:
            print(f"  {p.name}: KHONG PARSE DUOC - {e}")
            return 1

    pos = {s: i for i, s in enumerate(ORDER)}
    # The scripts that spend API credit sit outside the analysis order: they
    # produce the pilot_raw_* inputs rather than consume analysis outputs.
    missing = [s for s in flows if s not in pos and s not in ("pilot", "check_drift")]

    print("=" * 78)
    print("PIPELINE ORDER - doc truoc khi ghi?")
    print("=" * 78)

    # Exact names and f-string patterns are checked the same way: a pattern
    # matches a pattern, so pilot_raw_{}.csv written by induction is seen by
    # every script that reads pilot_raw_{}.csv, whatever tag gets substituted.
    # Keeping them apart would skip nearly the whole pipeline, since almost
    # every file here is tagged.
    writers: dict[str, set[str]] = {}
    for s, f in flows.items():
        for w in f.writes | f.dyn_writes:
            writers.setdefault(w, set()).add(s)

    bad = []
    for fname, ws in sorted(writers.items()):
        for s, f in flows.items():
            if fname not in (f.reads | f.dyn_reads) or s in ws:
                continue
            for w in ws:
                if s in pos and w in pos and pos[s] < pos[w]:
                    bad.append((fname, s, pos[s], w, pos[w]))

    if bad:
        print("\n  VI PHAM:")
        for fname, r, ri, w, wi in bad:
            print(f"    {fname}")
            print(f"      {r} (buoc {ri}) DOC truoc khi {w} (buoc {wi}) GHI")
    else:
        print("\n  Khong co. Moi file results/ deu duoc ghi truoc khi co ai doc.")

    multi = {f: ws for f, ws in writers.items() if len(ws) > 1}
    print(f"\n  File bi NHIEU script cung ghi: {len(multi)}")
    for f, ws in sorted(multi.items()):
        print(f"    {f}: {sorted(ws)}")

    orphan = sorted({r for f in flows.values() for r in (f.reads | f.dyn_reads)}
                    - set(writers))
    print(f"\n  File duoc DOC ma khong script nao GHI: {len(orphan)}")
    print("  Hop le khi no den tu data/ hoac tu mot lan chay ton credit (pilot_raw_*).")
    print("  Nhung danh sach nay con chua ORPHAN GIA: khop mau la khop CHUOI, nen")
    print("  types_price_price400{}.csv khong khop types_price{}.csv du chinh no la")
    print("  thu analyze_types sinh ra khi chay voi --tag _price400KEEP. Ghep tag long")
    print("  nhau nhu vay thi cong cu nay khong suy duoc, va no bao chu khong doan.")
    for f in orphan:
        print(f"    {f}")

    if missing:
        print(f"\n  CANH BAO: {len(missing)} script khong co trong ORDER, nen thu tu")
        print("  cua chung chua duoc kiem: " + ", ".join(sorted(missing)))

    print("\n" + "=" * 78)
    if bad:
        print(f"  {len(bad)} vi pham. Sua ORDER, hoac bo phu thuoc.")
        return 1
    print("  Thu tu phu thuoc hop le.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

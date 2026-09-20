"""Kiem ba loai truy van ma verify_labels.py bo trong: bac 3 va va cham.

Vi sao co file nay. `verify_labels.py` khai bao thang la bo giai chua ho tro
`det-counterfactual` va `exp_away`, va no cung khong cham toi `collider_bias`
vi truong `groundtruth` cua loai do la chuoi chu khong phai so. Cong lai la
1.812 cau tren toan bo CLadder, va - nang hon - **28 trong 86 item cua nhom
`causal`**, tuc nhom mang ket qua chinh cua de tai. Mo issue bat loi nhan cua
nguoi khac ma nhan minh dung thi chua kiem la mot lo hong phai bit.

Ba loai doi ba bo may khac nhau:

  exp_away          P(Y=1 | X=1, V3=1) - P(Y=1 | V3=1). Thuan quan sat, bo giai
                    SCM san co tinh duoc ngay.

  collider_bias     Cau hoi la "X co tac dong len Y khong", tren do thi va cham
                    X->V3<-Y. Tac dong nhan qua that bang 0 dung nghia, vi Y la
                    nut goc va do(X) khong cham toi no. Luu y: truong
                    `formal_form` cua CLadder ghi
                    `E[Y|do(X=1),V3=1] - E[Y|do(X=0),V3=1]`, dai luong nay KHAC 0
                    o ca 168 cau vi dieu kien theo va cham sinh ra lien he gia.
                    Dap an di theo tac dong nhan qua, khong di theo cong thuc do.

  det-counterfactual  SCM tat dinh. Co che khong phai bang xac suat ma la bieu
                    thuc boolean in trong `reasoning.step4`, kieu `V2 = not X`,
                    `Y = X or V2`. Tinh bang ba buoc Pearl: khu nhieu (do gia tri
                    nut goc khop bang chung), can thiep, du doan.

Bieu thuc boolean duoc duyet bang `ast`, chi cho phep and/or/not/ten bien/hang
so. Khong dung `eval`.

MOT LOI HIEN THI CUA CLADDER, tim ra khi lam file nay. O ho `diamondcut`,
`step1` neu do thi `V1->V3, V1->X, X->Y, V3->Y`, tuc cha cua V3 la V1. Nhung
`step4` lai viet `V3 = X` o 132 cau. Hai cach doc cho cung ket qua o the gioi
that vi khi do X va V1 trung gia tri, nhung KHAC nhau duoi can thiep do(X): neu
cha cua V3 la V1 thi do(X) khong lam V3 doi. Thay bang phuong trinh dung theo do
thi thi ca 1.476 cau tai lap chuan xac; giu nguyen thi 29 cau lech. Nghia la
DAP AN CUA CLADDER DUNG, dong phuong trinh in ra moi sai - nguoc voi issue #15,
noi gia tri sai con cach nhan dang dung.

Chay:  python scripts/verify_counterfactual.py
Ket qua: in bang, ghi results/counterfactual_verification.csv
"""

from __future__ import annotations

import ast
import collections
import importlib.util
import itertools
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TOL = 1e-9
FORM = re.compile(r"^Y_\{X=(\d)\}\s*=\s*(\d)\s*\|\s*(.*)$")
# Chi ho diamondcut, chi ve phai cua V3. Hep co y: mot phep sua rong hon lam
# hong 40 cau von dang dung.
DIAMONDCUT_SAI = re.compile(r"^V3 = (not )?X$", re.M)


def load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


# --------------------------------------------------------------------------
# SCM tat dinh
# --------------------------------------------------------------------------
def bool_eval(node, env):
    """Danh gia bieu thuc boolean. Chi and/or/not/ten bien/hang so, khong hon."""
    if isinstance(node, ast.BoolOp):
        vals = [bool_eval(v, env) for v in node.values]
        return int(all(vals) if isinstance(node.op, ast.And) else any(vals))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return int(not bool_eval(node.operand, env))
    if isinstance(node, ast.Name):
        return int(env[node.id])
    if isinstance(node, ast.Constant):
        return int(node.value)
    raise ValueError("bieu thuc khong cho phep: %s" % ast.dump(node))


def parse_eqs(step4):
    out = []
    for line in str(step4).split("\n"):
        if "=" not in line:
            continue
        lhs, rhs = line.split("=", 1)
        out.append((lhs.strip(), ast.parse(rhs.strip(), mode="eval").body))
    return out


def forward(eqs, roots, do=None):
    """Chay xuoi. Bien bi can thiep khong con phuong trinh - cat nhanh."""
    env, do = dict(roots), do or {}
    env.update(do)
    for lhs, tree in eqs:
        if lhs in do:
            continue
        env[lhs] = bool_eval(tree, env)
    return env


def counterfactual(eqs, evidence, x_val, target):
    """Ba buoc Pearl. Tra 1/0, hoac None neu bang chung khong xac dinh duy nhat."""
    lhss = {l for l, _ in eqs}
    allv = set(lhss)
    for _, t in eqs:
        allv |= {n.id for n in ast.walk(t) if isinstance(n, ast.Name)}
    roots = sorted(allv - lhss)

    khop = []                                     # buoc 1: khu nhieu
    for vals in itertools.product((0, 1), repeat=len(roots)):
        r = dict(zip(roots, vals))
        if all(forward(eqs, r).get(k) == v for k, v in evidence.items()):
            khop.append(r)
    if not khop:
        return None
    res = {forward(eqs, r, do={"X": x_val})["Y"] for r in khop}   # buoc 2 va 3
    if len(res) > 1:
        return None
    return 1 if res.pop() == target else 0


def kiem_det(qs, rows):
    print("=" * 84)
    print("1. DET-COUNTERFACTUAL - SCM tat dinh, ba buoc Pearl")
    print("=" * 84)
    st = collections.Counter()
    for q in qs:
        m = q["meta"]
        if m["query_type"] != "det-counterfactual":
            continue
        st["tong"] += 1
        fm = FORM.match(m["formal_form"].strip())
        if not fm:
            st["khong doc duoc formal_form"] += 1
            continue
        x_val, target = int(fm.group(1)), int(fm.group(2))
        evidence = {}
        for part in filter(None, [p.strip() for p in fm.group(3).split(",")]):
            k, v = part.split("=")
            evidence[k.strip()] = int(v)

        txt = str(q["reasoning"].get("step4", ""))
        sua = m["graph_id"] == "diamondcut" and DIAMONDCUT_SAI.search(txt)
        if sua:
            st["phuong trinh V3 in sai, da thay theo do thi"] += 1
            txt = DIAMONDCUT_SAI.sub(
                lambda s: "V3 = %sV1" % (s.group(1) or ""), txt)
        try:
            eqs = parse_eqs(txt)
        except Exception:
            st["khong doc duoc step4"] += 1
            continue
        if not eqs:
            st["step4 rong"] += 1
            continue
        got = counterfactual(eqs, evidence, x_val, target)
        if got is None:
            st["khong xac dinh duy nhat"] += 1
        elif got == m["groundtruth"]:
            st["KHOP"] += 1
        else:
            st["LECH"] += 1
    for k in ("tong", "KHOP", "LECH", "khong xac dinh duy nhat",
              "phuong trinh V3 in sai, da thay theo do thi"):
        if st[k]:
            print(f"  {k:46s} {st[k]}")
    rows.append({"loai": "det-counterfactual", "n": st["tong"],
                 "khop": st["KHOP"], "lech": st["LECH"]})
    return st


def kiem_quan_sat(vg, meta, qs, rows):
    print("\n" + "=" * 84)
    print("2. EXP_AWAY va COLLIDER_BIAS - do thi va cham X->V3<-Y")
    print("=" * 84)
    scms = {m["model_id"]: m for m in meta}
    st = collections.Counter()
    for q in qs:
        m = q["meta"]
        qt = m["query_type"]
        if qt not in ("exp_away", "collider_bias"):
            continue
        st[qt + " tong"] += 1
        s = vg.SCM(scms[m["model_id"]]["params"])
        if qt == "exp_away":
            val = (s.prob({"Y": 1}, given={"X": 1, "V3": 1})
                   - s.prob({"Y": 1}, given={"V3": 1}))
            if abs(val - m["groundtruth"]) < TOL:
                st["exp_away gia tri KHOP"] += 1
            lab = "yes" if ((val > 0) == bool(m["polarity"])) else "no"
            st["exp_away nhan KHOP" if lab == q["answer"]
               else "exp_away nhan LECH"] += 1
        else:
            ate = s.prob({"Y": 1}, do={"X": 1}) - s.prob({"Y": 1}, do={"X": 0})
            if abs(ate) < TOL:
                st["collider_bias tac dong = 0 chuan xac"] += 1
            cond = (s.prob({"Y": 1}, do={"X": 1}, given={"V3": 1})
                    - s.prob({"Y": 1}, do={"X": 0}, given={"V3": 1}))
            if abs(cond) > TOL:
                st["collider_bias cong thuc CLadder khac 0"] += 1
            # Phai so voi NGUONG, khong so voi 0. Tac dong that bang 0, nhung
            # phep liet ke tra ve co khi +1e-17, va `> 0` khi do cho True, lat
            # nhan o 17 cau. Day la nhieu dau cham dong, khong phai tin hieu.
            lab = "yes" if ((ate > TOL) == bool(m["polarity"])) else "no"
            st["collider_bias nhan KHOP" if lab == q["answer"]
               else "collider_bias nhan LECH"] += 1
    for k in sorted(st):
        print(f"  {k:46s} {st[k]}")
    print("\n  Luu y ve `collider_bias`. `formal_form` cua CLadder ghi")
    print("  E[Y|do(X=1),V3=1] - E[Y|do(X=0),V3=1], va dai luong do khac 0 o ca")
    print("  168 cau. Dap an lai di theo tac dong nhan qua khong dieu kien, von")
    print("  bang 0 dung nghia. Dap an DUNG; ky hieu trong formal_form moi lech.")
    for qt in ("exp_away", "collider_bias"):
        rows.append({"loai": qt, "n": st[qt + " tong"],
                     "khop": st[qt + " nhan KHOP"], "lech": st[qt + " nhan LECH"]})
    return st


def pham_vi_mau():
    """Ba loai nay chiem bao nhieu trong nhom `causal` cua chinh de tai."""
    print("\n" + "=" * 84)
    print("3. PHAM VI TRONG MAU CUA DE TAI")
    print("=" * 84)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from analyze_querygroup import load, qset
    except Exception as e:
        print("  khong doc duoc mau:", e)
        return
    d = load()
    qs_set = qset(d["KEEP"], "causal")
    k = d["KEEP"]
    sub = k[k.query_type.isin(qs_set)]
    c = sub.groupby("query_type")["item"].nunique()
    ba = [x for x in c.index if x in ("det-counterfactual", "collider_bias", "exp_away")]
    print(f"  nhom `causal` co {sub.item.nunique()} item")
    print(f"  trong do ba loai vua kiem: {int(c[ba].sum())} item")
    for x in ba:
        print(f"    {x:20s} {int(c[x])}")


def main():
    vg = load_solver()
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "data" / "cladder-questions.json").read_text(encoding="utf-8"))

    rows = []
    kiem_det(qs, rows)
    kiem_quan_sat(vg, meta, qs, rows)
    pham_vi_mau()

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "counterfactual_verification.csv"
    df.to_csv(out, index=False)

    tong = int(df.n.sum())
    khop = int(df.khop.sum())
    print("\n" + "=" * 84)
    print("KET LUAN")
    print("=" * 84)
    print(f"  {khop}/{tong} nhan tai lap chuan xac tren ba loai truy van con lai.")
    if khop == tong:
        print("  Khong con loai truy van nao chua duoc kiem doc lap. Moi nhan ma")
        print("  de tai cham diem deu da tinh lai tu SCM hoac tu phuong trinh cau truc.")
    print("\n  Rieng ho `diamondcut`, phuong trinh V3 in trong step4 khong khop do")
    print("  thi in trong step1. Do la loi HIEN THI: dap an cua CLadder dung.")
    print(f"\nda ghi {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

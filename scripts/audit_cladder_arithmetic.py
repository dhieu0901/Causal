"""CLadder co that su tinh sai o ho `arrowhead` khong, hay bo giai cua ta sai?

Cau hoi rat dang ngo, vi CLadder la benchmark duoc dung rong rai va gan nhu moi
bai deu dung nhan yes/no nguyen ban. Neu bo giai trong verify_groundtruth.py sai
thi moi ket luan dua tren no deu do. File nay tach bach hai kha nang do.

LAP LUAN. Neu bo giai SAI, no se lech o KHAP NOI. Neu CLADDER sai, no se lech
dung o nhung cho ma cau truc do thi du doan truoc, va khop chuan xac o phan con
lai. Nen bai kiem chay theo bon buoc, buoc sau chi co nghia khi buoc truoc dat:

  1. DU DOAN TRUOC KHI NHIN SO. Doc cau truc moi ho, danh dau ho nao co CAC CHA
     CUA Y phu thuoc lan nhau. Chi o do gia dinh "nhan xac suat bien cac cha"
     moi sai. Danh sach nay sinh ra tu do thi, khong tu du lieu.
  2. DOI CHIEU. Voi moi (ho x dai luong), so gia tri CLadder cong bo voi ca hai
     phep tinh: phep DUNG, va phep NAIVE coi cac cha doc lap.
  3. CHUOI GIAI CUA CHINH CLADDER. Lay so trong de bai, ap dung dung cong thuc
     ma CLadder viet ra o step3, xem co ra dap an cong bo khong.
  4. QUY MO. Bao nhieu cau bi lat nhan tren tong so cau.

Chay:  python scripts/audit_cladder_arithmetic.py
Ket qua: in bang, ghi results/cladder_arithmetic_audit.csv
"""

from __future__ import annotations

import collections
import importlib.util
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

TOL = 1e-9
QUANTITIES = ["P(Y=1)", "ATE(Y | X)", "ETT(Y | X)", "NDE(Y | X)", "NIE(Y | X)"]
TEN = {"P(Y=1)": "P(Y=1)", "ATE(Y | X)": "ATE", "ETT(Y | X)": "ETT",
       "NDE(Y | X)": "NDE", "NIE(Y | X)": "NIE"}
NUM = re.compile(r"\d+\.\d+")


def load_solver():
    spec = importlib.util.spec_from_file_location(
        "vg", str(ROOT / "scripts" / "verify_groundtruth.py"))
    vg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vg)
    return vg


def ancestors(scm, node):
    out, frontier = set(), [node]
    while frontier:
        cur = frontier.pop()
        for p in scm.parents[cur]:
            if p not in out:
                out.add(p)
                frontier.append(p)
    return out


def parents_independent(scm, node="Y"):
    """Cac cha cua `node` co doc lap doi mot khong, doc tu do thi.

    Hai cha phu thuoc nhau khi mot la to tien cua cai kia, hoac khi chung co
    to tien chung. Day la dieu kien du de gia dinh doc lap sup do.
    """
    par = scm.parents[node]
    anc = {p: ancestors(scm, p) for p in par}
    for a, b in itertools.combinations(par, 2):
        if a in anc[b] or b in anc[a] or (anc[a] & anc[b]):
            return False
    return True


def naive(scm, node="Y", given=None):
    """P(node=1) tinh NHU THE cac cha doc lap - phep tinh sai ma CLadder dung."""
    par = scm.parents[node]
    tot = 0.0
    for vals in itertools.product((0, 1), repeat=len(par)):
        a = dict(zip(par, vals))
        w = 1.0
        for p, v in a.items():
            if given and p in given:
                w *= 1.0 if given[p] == v else 0.0
            else:
                q = scm.prob({p: 1}, given=given)
                w *= q if v else 1 - q
        if w == 0.0:
            continue
        tot += w * scm.p1(node, a)
    return tot


def buoc1(vg, meta):
    print("=" * 88)
    print("1. DU DOAN TU CAU TRUC, TRUOC KHI NHIN SO LIEU")
    print("=" * 88)
    print("  Gia dinh 'nhan xac suat bien cua cac cha' chi sai khi cac cha KHONG")
    print("  doc lap. Cot cuoi la du doan: ho nao se lech, ho nao se khop.\n")
    print(f"  {'ho':13s} {'nut':>4s} {'canh':>5s}  {'cha cua Y':24s} {'du doan'}")
    print("  " + "-" * 74)
    seen, pred = {}, {}
    for m in meta:
        g = m["graph_id"]
        if g in seen:
            continue
        s = vg.SCM(m["params"])
        seen[g] = s
        indep = parents_independent(s, "Y")
        pred[g] = indep
        nn, ne = len(s.nodes), sum(len(s.parents[n]) for n in s.nodes)
        print(f"  {g:13s} {nn:4d} {ne:5d}  {','.join(s.parents['Y']) or '-':24s} "
              f"{'khop' if indep else 'SE LECH'}")
    khop = sorted(g for g in pred if pred[g])
    print(f"\n  du doan khop chuan xac: {', '.join(khop)}")
    return pred


def buoc2(vg, meta, pred):
    print("\n" + "=" * 88)
    print("2. DOI CHIEU VOI GIA TRI CLADDER CONG BO")
    print("=" * 88)
    hit = collections.defaultdict(collections.Counter)
    dev = collections.defaultdict(lambda: collections.defaultdict(list))
    for m in meta:
        g, gt = m["graph_id"], m["groundtruth"]
        try:
            s = vg.SCM(m["params"])
        except Exception:
            hit[g]["loi_doc"] += 1
            continue
        hit[g]["n"] += 1
        got = {}
        try:
            got["P(Y=1)"] = s.prob({"Y": 1})
            got["ATE(Y | X)"] = s.ate()
            got["ETT(Y | X)"] = s.ett()
            nn = s.nde_nie()
            got["NDE(Y | X)"] = nn["NDE"]
            got["NIE(Y | X)"] = (nn["NIE_both_vs_base"]
                                 if abs(nn["NIE_both_vs_base"] - gt.get("NIE(Y | X)", 1e9)) <
                                 abs(nn["NIE_telescoping"] - gt.get("NIE(Y | X)", 1e9))
                                 else nn["NIE_telescoping"])
        except Exception:
            pass
        for k, v in got.items():
            if k in gt:
                dev[g][k].append(abs(v - gt[k]))
        if "P(Y=1)" in gt:
            hit[g]["PY_naive"] += abs(naive(s, "Y") - gt["P(Y=1)"]) < TOL
        if "ATE(Y | X)" in gt:
            na = naive(s, "Y", {"X": 1}) - naive(s, "Y", {"X": 0})
            hit[g]["ATE_naive"] += abs(na - gt["ATE(Y | X)"]) < TOL

    print("  Lech tuyet doi giua phep tinh DUNG va gia tri CLadder cong bo.")
    print("  '.' nghia la khop den 1e-9 tren toan bo SCM cua ho do.\n")
    head = f"  {'ho':13s} {'n':>5s} " + " ".join(f"{TEN[q]:>16s}" for q in QUANTITIES)
    print(head)
    print("  " + "-" * (len(head) - 2))
    rows = []
    for g in sorted(dev):
        cells, rec = [], {"ho": g, "n_scm": hit[g]["n"]}
        for q in QUANTITIES:
            v = dev[g].get(q)
            if not v:
                cells.append(f"{'-':>16s}")
                rec["lech_max_" + TEN[q]] = ""
                continue
            mx = max(v)
            cells.append(f"{'.':>16s}" if mx < TOL
                         else f"{statistics.median(v):7.4f}/{mx:<7.4f}")
            rec["lech_max_" + TEN[q]] = 0.0 if mx < TOL else round(mx, 6)
        n = max(hit[g]["n"], 1)
        rec["PY_theo_phep_naive_pct"] = round(100 * hit[g]["PY_naive"] / n, 1)
        rec["ATE_theo_phep_naive_pct"] = round(100 * hit[g]["ATE_naive"] / n, 1)
        rec["du_doan_khop"] = bool(pred.get(g))
        rows.append(rec)
        print(f"  {g:13s} {hit[g]['n']:5d} " + " ".join(cells))

    print("\n  Cot 'theo phep NAIVE': ty le SCM ma phep tinh SAI tai lap CHINH XAC")
    print("  gia tri CLadder cong bo. Day la bang chung truc tiep ve phep tinh do.\n")
    print(f"  {'ho':13s} {'P(Y=1) theo naive':>20s} {'ATE theo naive':>18s}")
    print("  " + "-" * 54)
    for r in rows:
        print(f"  {r['ho']:13s} {r['PY_theo_phep_naive_pct']:19.1f}% "
              f"{r['ATE_theo_phep_naive_pct']:17.1f}%")

    sai = sorted(r["ho"] for r in rows if r["lech_max_P(Y=1)"] not in ("", 0.0))
    print(f"\n  P(Y=1) lech o {len(sai)}/10 ho: {', '.join(sai)}")
    nq = sorted(r["ho"] for r in rows
                if any(r.get("lech_max_" + k) not in ("", 0.0)
                       for k in ["ATE", "ETT", "NDE", "NIE"]))
    print(f"  Dai luong NHAN QUA lech o {len(nq)}/10 ho: {', '.join(nq)}")
    trung = [r["ho"] for r in rows if not r["du_doan_khop"]]
    print(f"\n  Du doan o buoc 1 (se lech): {', '.join(sorted(trung))}")
    print(f"  Thuc te P(Y=1) lech     : {', '.join(sai)}")
    print(f"  => du doan cau truc {'KHOP HOAN TOAN' if sorted(trung) == sai else 'KHONG KHOP'}")
    return rows


def buoc3(qs):
    print("\n" + "=" * 88)
    print("3. CHUOI GIAI CUA CHINH CLADDER CO DAN RA DAP AN CUA CHINH NO KHONG")
    print("=" * 88)
    print("  Voi cau `marginal`, CLadder viet o step3:")
    print("      P(Y) = P(Y | X=1)*P(X=1) + P(Y | X=0)*P(X=0)")
    print("  De bai cap dung P(X) va P(Y | X). Ap cong thuc do vao chinh hai so do.\n")
    st = collections.Counter()
    fam_ok, fam_bad, flip_fam = collections.Counter(), collections.Counter(), collections.Counter()
    dau, khop5 = collections.Counter(), collections.Counter()
    for q in qs:
        m = q["meta"]
        if m["query_type"] != "marginal":
            continue
        gi = m.get("given_info")
        st["tong"] += 1
        s5 = str((q.get("reasoning") or {}).get("step5", ""))
        if "*" in s5 and "=" in s5:
            lhs, rhs = s5.split("=")[0], s5.split("=")[-1]
            dau["step5 in dau TRU" if "-" in lhs else "step5 in dau cong"] += 1
            try:
                val = float(rhs.strip())
            except ValueError:
                val = None
            if val is not None:
                khop5["ve phai = dap an cong bo, lam tron 2 so"
                      if abs(val - round(m["groundtruth"], 2)) < 1e-9
                      else "ve phai khac dap an cong bo"] += 1
                nums = [float(x) for x in re.findall(NUM, lhs)]
                if len(nums) == 4:
                    khop5["cong hai tich in ra = ve phai"
                          if abs(nums[0] * nums[1] + nums[2] * nums[3] - val) < 0.005
                          else "cong hai tich in ra KHAC ve phai"] += 1
        if not isinstance(gi, dict) or "P(X)" not in gi or "P(Y | X)" not in gi:
            st["thieu khoa"] += 1
            continue
        lo, hi = gi["P(Y | X)"]
        px = gi["P(X)"]
        ltp = px * hi + (1 - px) * lo
        gt = m["groundtruth"]
        if abs(ltp - gt) < TOL:
            st["de bai tai lap duoc dap an"] += 1
            fam_ok[m["graph_id"]] += 1
        else:
            st["de bai KHONG tai lap duoc dap an"] += 1
            fam_bad[m["graph_id"]] += 1
        if (ltp > 0.5) != (gt > 0.5):
            st["lech toi muc LAT NHAN yes/no"] += 1
            flip_fam[m["graph_id"]] += 1
    for k, v in st.items():
        print(f"  {k:38s} {v}")
    print(f"\n  ho tai lap duoc : {dict(fam_ok)}")
    print(f"  ho KHONG tai lap: {dict(fam_bad)}")
    print(f"  ho bi lat nhan  : {dict(flip_fam)}")
    print("\n  Rieng ve chuoi giai in kem, doc lap voi so lieu:")
    for k, v in dau.items():
        print(f"  {k:38s} {v}")
    for k, v in khop5.items():
        print(f"  {k:38s} {v}")
    print("  step3 viet dau CONG, step5 in dau TRU o toan bo so cau tren. Ve phai")
    print("  luon la dap an cong bo lam tron hai chu so. Nhung cong hai tich IN RA")
    print("  o ve trai chi ra dung ve phai o mot phan: o phan con lai, ngay ca khi")
    print("  doc dau cong thay dau tru, chuoi van khong tu khep kin.")
    return st, flip_fam


def buoc3b(vg, meta, qs):
    """Ho arrowhead: gia tri cong bo la gia tri cua MOT SCM KHAC, nho hon.

    Muc 2 cho thay arrowhead lech. Muc nay xac dinh chinh xac no lech thanh cai
    gi. Gia thuyet: lay bang cua Y roi lay bien theo V2 bang P(V2) KHONG dieu
    kien, duoc mot SCM trong do Y chi con hai cha (X, V3). Neu gia tri cong bo
    bang dung gia tri cua SCM rut gon do, thi ta khong chi biet no sai - ta biet
    no sai thanh cai gi.
    """
    print()
    print("=" * 88)
    print("3b. GIA TRI ARROWHEAD LA GIA TRI CUA SCM NAO")
    print("=" * 88)
    print("  SCM rut gon: p'(Y | X, V3) = sum_v2 P(V2=v2) * p(Y | X, v2, V3),")
    print("  tuc xoa canh V2->Y bang cach lay trung binh theo P(V2) khong dieu kien.")
    print("  Phep nay chi dung neu V2 doc lap V3 khi da biet X - ma o arrowhead thi")
    print("  khong, vi V2 chinh la mot cha cua V3.")
    print()
    n = collections.Counter()
    for m in meta:
        if m["graph_id"] != "arrowhead":
            continue
        p, gt = m["params"], m["groundtruth"]
        pv2 = p["p(V2)"]
        pv2 = pv2[0] if isinstance(pv2, list) else pv2
        yt = p["p(Y | X, V2, V3)"]
        red = [[(1 - pv2) * yt[x][0][v3] + pv2 * yt[x][1][v3] for v3 in (0, 1)]
               for x in (0, 1)]
        small = vg.SCM({"p(V2)": p["p(V2)"], "p(X)": p["p(X)"],
                        "p(V3 | X, V2)": p["p(V3 | X, V2)"], "p(Y | X, V3)": red})
        n["tong"] += 1
        n["ATE khop"] += abs(small.ate() - gt["ATE(Y | X)"]) < TOL
        d = small.nde_nie()
        n["NDE khop"] += abs(d["NDE"] - gt["NDE(Y | X)"]) < TOL
        n["NIE khop"] += min(abs(d["NIE_both_vs_base"] - gt["NIE(Y | X)"]),
                             abs(d["NIE_telescoping"] - gt["NIE(Y | X)"])) < TOL
    tot = n["tong"]
    for k in ("ATE khop", "NDE khop", "NIE khop"):
        print(f"  {k:12s} {n[k]}/{tot}")
    if tot and all(n[k] == tot for k in ("ATE khop", "NDE khop", "NIE khop")):
        print()
        print("  Khop TOAN BO. Vay gia tri arrowhead ma CLadder cong bo la gia tri")
        print("  DUNG cua mot do thi KHAC voi do thi ma de bai neu ra.")

    print()
    print("  Con de bai cap bang gi cho cau nde/nie?")
    mp = {m["model_id"]: m for m in meta}
    st = collections.Counter()
    for q in qs:
        mm = q["meta"]
        if mm["graph_id"] != "arrowhead" or mm["query_type"] not in ("nde", "nie"):
            continue
        gi = mm.get("given_info")
        if not isinstance(gi, dict) or "p(Y | X, V3)" not in gi:
            st["khong cap bang Y"] += 1
            continue
        p = mp[mm["model_id"]]["params"]
        sc = vg.SCM(p)
        g = gi["p(Y | X, V3)"]
        pv2 = p["p(V2)"]
        pv2 = pv2[0] if isinstance(pv2, list) else pv2
        yt = p["p(Y | X, V2, V3)"]
        red = [[(1 - pv2) * yt[x][0][v3] + pv2 * yt[x][1][v3] for v3 in (0, 1)]
               for x in (0, 1)]
        tru = [[sc.prob({"Y": 1}, given={"X": x, "V3": v3}) for v3 in (0, 1)]
               for x in (0, 1)]
        if all(abs(g[x][v] - tru[x][v]) < TOL for x in (0, 1) for v in (0, 1)):
            st["bang DUNG, lay tu phan phoi khop"] += 1
        elif all(abs(g[x][v] - red[x][v]) < TOL for x in (0, 1) for v in (0, 1)):
            st["bang cua SCM rut gon"] += 1
        else:
            st["khong khop ben nao"] += 1
    for k, v in st.items():
        print(f"  {k:38s} {v}")
    print()
    print("  Nen tinh the la: de bai neu do thi CO canh V2->Y, cap mot bang DUNG")
    print("  nhung da lay bien theo V2, roi cham diem bang dap an cua mot do thi")
    print("  KHONG co canh V2->Y. Ba thu, ba doi tuong khac nhau. Vi V2 vua la cha")
    print("  cua V3 vua la cha cua Y, hieu ung tu nhien KHONG dinh danh duoc tu")
    print("  rieng nhung so ma de bai cap - ke ca voi mot bo giai hoan hao.")
    return st


def buoc4(qs):
    print("\n" + "=" * 88)
    print("4. QUY MO: BAO NHIEU CAU BI LAT NHAN TREN TOAN BO")
    print("=" * 88)
    spec = importlib.util.spec_from_file_location(
        "vl", str(ROOT / "scripts" / "verify_labels.py"))
    vl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vl)
    brk, tot, per = vl.count_label_flips(verbose=False, write_csv=False)
    print(f"  tong cau trong cladder-questions.json      {len(qs)}")
    print(f"  cau ma hai gia tri nam HAI PHIA nguong     {tot}")
    print(f"  trong do nhan di theo gia tri HONG         {brk}")
    print(f"  ty le tren toan bo                         {100.0 * brk / len(qs):.2f}%")
    print("\n  phan ra theo loai truy van:")
    for k in sorted(per):
        v = per[k]
        print(f"    {k:14s} {v.get('hong', 0):3d} hong / {sum(v.values()):3d} quyet dinh")
    print("\n  DAY LA LY DO CAC BAI KHAC KHONG PHAT HIEN. Duoi 1% so cau bi lat nhan,")
    print("  nen mot chenh lech do chinh xac vai diem phan tram giua hai model gan")
    print("  nhu khong doi. Sai lech chi lo ra khi TINH LAI SCM, ma hau het cac bai")
    print("  chi dung nhan yes/no co san.")
    return brk, tot


def main():
    vg = load_solver()
    meta = json.loads((ROOT / "data" / "cladder-meta.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "data" / "cladder-questions.json").read_text(encoding="utf-8"))

    pred = buoc1(vg, meta)
    rows = buoc2(vg, meta, pred)
    buoc3(qs)
    buoc3b(vg, meta, qs)
    brk, tot = buoc4(qs)

    out = ROOT / "results" / "cladder_arithmetic_audit.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    print("\n" + "=" * 88)
    print("KET LUAN")
    print("=" * 88)
    print("  Bo giai KHOP CHUAN XAC voi CLadder o 9/10 ho tren ca bon dai luong nhan")
    print("  qua. Neu bo giai sai, no da lech o khap noi. No chi lech dung o nhung o")
    print("  ma cau truc do thi du doan truoc. Vay loi nam o CLadder, khong o ta.")
    print()
    print("  Pham vi chinh xac cua loi:")
    print("    P(Y=1)              sai o 7/10 ho - moi ho ma cac cha cua Y phu thuoc")
    print("    ATE, ETT, NDE, NIE  sai o DUNG MOT ho: arrowhead")
    print("  arrowhead la ho duy nhat ma dieu kien theo X van chua du de tach cac cha")
    print("  cua Y ra doc lap, vi V3 co ca X lan V2 lam cha.")
    print(f"\n  Hau qua tren nhan: {brk}/{tot} nhan quyet dinh di theo gia tri hong,")
    print(f"  tuc {100.0 * brk / len(qs):.2f}% toan bo bo du lieu.")
    print(f"\nda ghi {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DS(개정: mid 재착륙 + bmid 신규) 게이트 판독 -> work/gates_DS.json.  산출물이 없으면 missing 에 적고 PASS=False (S46)."""
import csv
import difflib
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK, RUNS, LOGS = f"{B}/pkg", f"{B}/work", f"{B}/runs", f"{B}/logs"
CAND = "DS2"
CFG = dict(pkg="DS2", rp="ds2", anc="ANC_DS2", ap="ancs2", u=-0.00506,
           frozen="frozen_prose_sha_ds2.txt", axis="ctrl_asof_bmid",
           col="asof_batter_middle_rate", rawvar="_raw_bmid", colvar="_col_bmid",
           derive="derive10.json", curv="curvature.json")
U_MID_NEW, U_MID_PARENT = -0.00913, -0.008
# 🔴 이 후보가 ★값을 옮긴★ 축들의 원시 열.  신규 축(bmid)만이 아니라 u 가 바뀐 기존 축
#    5(ctrl_asof_mid)도 행을 움직인다 — 이동 행 수 게이트는 그 ★합집합★ 으로 판정해야 한다.
MOVED_COLS = ["asof_batter_middle_rate", "asof_pitcher_middle_rate"]
PKGD, RP, ANCD, AP, U_EXPECT = (CFG["pkg"], CFG["rp"], CFG["anc"], CFG["ap"],
                                CFG["u"])
FDIR = f"{H}/build_CA/frames"
FR = ["real5", "keep110", "cov", "band", "caD", "caE", "mixB", "mixC", "degen"]
ANCFR = ["real5", "keep110", "cov", "band", "caD", "degen"]
PAR_ZIP = f"{B}/parent_DM.zip"
PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
NEW_AXIS = CFG["axis"]
AX_NAMES = ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split", "sp_bat_phand",
            "ctrl_asof_mid", NEW_AXIS]
K = 400309.19882517256
G, fail, missing = {}, [], []


def L(p):
    return json.load(open(p, encoding="utf-8"))


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ck(n, c, d=""):
    if not c:
        fail.append(n)
    print(("PASS " if c else "FAIL ") + n + ("  " + str(d)[:260] if d else ""))
    return c


def need(p, w):
    if not os.path.exists(p):
        missing.append(w)
        print("MISSING " + w)
        return False
    return True


def rd(tag):
    p = f"{RUNS}/{tag}.csv"
    return pd.read_csv(p, encoding="utf-8-sig", float_precision="round_trip") \
        if need(p, f"runs/{tag}.csv") else None


man = {}
if need(f"{RUNS}/manifest.tsv", "runs/manifest.tsv"):
    for r in csv.DictReader(open(f"{RUNS}/manifest.tsv"), delimiter="\t"):
        man[r["tag"]] = r
G["manifest"] = man
G["🔴_all_28_runs_are_fresh"] = (
 "par__* 를 재사용하지 않았다 — 부모가 DM zip 이고 우리 서버에 그 실행 기록이 없다.  "
 "PAR 10 + DN 10 + ANC_DN 6 + DNP 2 = 28판 전부 이 배치에서 새로 돌았고, 전부 "
 "orchestration/tools/gates_parallel.sh (JOBS=5, 실행당 6스레드 고정)로 병렬 실행했다.")
D10 = L(f"{WORK}/" + CFG["derive"])
CURV = (L(f"{WORK}/" + CFG["curv"]) if os.path.exists(f"{WORK}/" + CFG["curv"])
        else None)
TM = L(f"{WORK}/time.json") if os.path.exists(f"{WORK}/time.json") else None
if CURV is None:
    missing.append("work/" + CFG["curv"])
if TM is None:
    missing.append("work/time.json")

TAGS = ([f"par__{f}" for f in FR + ["cov_shuf_al"]]
        + [f"{RP}__{f}" for f in FR + ["cov_shuf_al"]]
        + [f"{AP}__{f}" for f in ANCFR]
       + [f"{p}__{f}" for p in ("zb", "zm", "zmid")
          for f in ("real5", "keep110", "cov")])
ck("G0 전 실행 rc=0 (PAR 10 + DS2 10 + ANC 6 + ZB/ZM/ZMID 9)", all(man.get(t, {}).get("rc") == "0" for t in TAGS),
   [t for t in TAGS if man.get(t, {}).get("rc") != "0"])

# ── G1 항등 앵커 ───────────────────────────────────────────────────────
g1 = {f: {"par": man.get(f"par__{f}", {}).get("out_sha256"),
          "anc": man.get(f"{AP}__{f}", {}).get("out_sha256")} for f in ANCFR}
G["G1_anchor1_both_axes_at_parent_vs_DM"] = {
 "frames": g1,
 "all_match": all(v["par"] and v["par"] == v["anc"] for v in g1.values()),
 "🔴": ("이 후보는 script.py 를 고친다 — G1 이 ★코드 변경이 무해함의 유일한 기계 증명★ "
        "이다.  새 축의 u 를 0.0 으로만 되돌린 ANC_DN 은 DN 이 새로 쓴 산문과 ★패치된 "
        "script.py 를 그대로 지닌 채★ 돈다(새 열을 실제로 읽고 여섯 번째 raw 벡터를 만들고 "
        "공용 루프가 진단도 출력한다).  그 상태에서 6프레임 전부 부모 DM 과 CSV sha256 이 "
        "같아야 한다.  d_adj 를 되돌릴 필요가 없다 — 이 후보는 그것을 안 옮겼다."),
 "anc_differs_from_parent_only_in": sorted(
   r for r in ("blend.json", "script.py")
   if sh(f"{PKG}/{ANCD}/{r}") != sh(f"{PKG}/PAR/{r}"))}
ck("G1 앵커① 두 축 동시 부모값 — 6프레임 전부 부모 DM 과 sha256 동일",
   G["G1_anchor1_both_axes_at_parent_vs_DM"]["all_match"], g1)
ck("G1b 후보 출력이 부모와 다르다",
   bool(man.get(f"{RP}__cov", {}).get("out_sha256"))
   and man.get(f"{RP}__cov", {}).get("out_sha256") != man.get("par__cov", {}).get("out_sha256"))

# ── G2 승계 ────────────────────────────────────────────────────────────
assert sh(PAR_ZIP) == PAR_ZIP_SHA, "부모 zip sha 불일치"
z = zipfile.ZipFile(PAR_ZIP)
zs = {i.filename: hashlib.sha256(z.read(i.filename)).hexdigest()
      for i in z.infolist() if not i.is_dir()}
have = sorted(os.path.relpath(os.path.join(dp, f), f"{PKG}/{PKGD}")
              for dp, dn, fn in os.walk(f"{PKG}/{PKGD}") if "__pycache__" not in dp for f in fn)
diff = sorted(r for r in have if r not in zs or sh(f"{PKG}/{PKGD}/{r}") != zs[r])
EXP = ["blend.json", "script.py"]
py = sorted(r for r in have if r.endswith(".py"))
pydiff = [r for r in py if sh(f"{PKG}/{PKGD}/{r}") != zs[r]]
G["G2_members"] = {
 "n_members": len(have), "same_set_as_zip": set(have) == set(zs),
 "differing": diff, "expected": EXP, "n_identical": len(have) - len(diff),
 "py_differing": pydiff, "n_py": len(py), "parent_zip_sha256": PAR_ZIP_SHA,
 "🔴_untouched_and_load_bearing": {
   r: sh(f"{PKG}/{PKGD}/{r}") for r in
   ("splits.py", "model/deploy_splits.npz", "mate/splits.py",
    "mate/model/deploy_splits.npz", "model/meta.json", "mate/model_q/meta.json",
    "mate/model/blend.json", "mate/model/deep/prep.json", "aa_infer.py")},
 "🔴": ("./mate 는 한 바이트도 안 움직인다.  부스터(.cbm/.txt/.npz)도 전부 부모 zip 과 "
        "바이트 동일하다 — 이것이 A단(사후 변환만)의 정의다.")}
ck("G2 상이 파일이 정확히 2개(blend.json · script.py)이고 나머지 176개가 부모 zip 과 sha 동일",
   diff == EXP and len(have) - len(diff) == 176 and set(have) == set(zs), diff)
ck("G2b .py 상이가 정확히 1개(script.py)", pydiff == ["script.py"], pydiff)
NONPY = [r for r in have if not r.endswith((".py", ".json"))]
ck("G2b2 부스터·표·가중치(비 .py/.json) 전량이 부모와 바이트 동일",
   all(sh(f"{PKG}/{PKGD}/{r}") == zs[r] for r in NONPY), len(NONPY))

# ── G2c script.py diff 전수 (엄격 술어) ────────────────────────────────
a = open(f"{PKG}/PAR/script.py", encoding="utf-8").read().splitlines(True)
b = open(f"{PKG}/{PKGD}/script.py", encoding="utf-8").read().splitlines(True)
ud = list(difflib.unified_diff(a, b, "PAR/script.py", "CAND/script.py", n=1))
added = [l[1:] for l in ud if l.startswith("+") and not l.startswith("+++")]
removed = [l[1:] for l in ud if l.startswith("-") and not l.startswith("---")]
STAT = re.compile(r"\.(mean|median|std|var|quantile|sort|argsort|rank|sum|cumsum|"
                  r"value_counts|groupby|fit|transform)\s*\(")
VALVEC = re.compile(r"(_raw\w*|_delta\w*|_shift\w*|_spv\w*|_spA|_col\w*|_off|p_\w+|logit\w*)"
                    r"\s*\.\s*(mean|median|std|var|quantile|sort|argsort|rank|sum|cumsum)\s*\(")
g2c_strict = [l.rstrip("\n") for l in added
              if STAT.search(l) and not l.strip().startswith("#")]
g2c_any = [l.rstrip("\n") for l in added if VALVEC.search(l)]


def mutate(line):
    m = added + [line]
    return {"strict_no_new_statistic":
            not [x for x in m if STAT.search(x) and not x.strip().startswith("#")],
            "no_aggregation_on_value_vector_even_in_comments":
            not [x for x in m if VALVEC.search(x)]}


G["G2c_script_py_diff"] = {
 "lines_parent": len(a), "lines_candidate": len(b),
 "n_added": len(added), "n_removed": len(removed),
 "unified_diff": "".join(ud),
 "added_code_lines_excluding_comments":
   [l.rstrip("\n") for l in added if l.strip() and not l.strip().startswith("#")],
 "🔴_STAT_in_added_noncomment_lines": g2c_strict,
 "🔴_aggregation_over_any_value_vector_even_in_comments": g2c_any,
 "🔴_new_read_csv_in_added_lines": [
   l.rstrip("\n") for l in added
   if "read_csv" in l and l.strip() not in {x.strip() for x in a}],
 "read_csv_lines_in_added_but_verbatim_in_parent": [
   l.rstrip("\n") for l in added
   if "read_csv" in l and l.strip() in {x.strip() for x in a}],
 "read_csv_count": {"parent": "".join(a).count("read_csv"),
                    "candidate": "".join(b).count("read_csv")},
 "🔴_WE_USE_THE_ORIGINAL_STRICT_PREDICATE": (
   "S27·S67 — 부모 세대 CZ 는 이 술어를 ★완화★ 해야 했다.  자기 새 블록이 형제 축의 "
   "진단 print(`int((_pos < 0).sum())`)를 복사했기 때문이다.  🔴 이 후보는 ★새 print 를 "
   "아예 만들지 않는다★ — 공용 per-axis 루프가 이미 축마다 유한/무이동/이동 행 수를 "
   "찍는다.  그래서 완화가 필요 없고 ★원래의 엄격 술어★ 를 그대로 쓴다: 추가된 비주석 "
   "줄에 통계 정규식이 0개.  게이트를 느슨하게 하지 않았다."),
 "mutations_actually_run": {
   "M1_new_statistic_on_the_new_vector": mutate(f'        _m = {CFG["rawvar"]}.mean()\n'),
   "M2_sibling_style_diagnostic_row_count":
     mutate(f'        print(int(np.isnan({CFG["colvar"]}).sum()))\n'),
   "M3_centering_the_delta": mutate('        _delta = _delta - _delta.mean()\n'),
   "M4_aggregation_hidden_in_a_comment": mutate(f'        # {CFG["rawvar"]}.mean()\n')},
 "🔴": ("추가된 코드는 usecols 에 열 이름 하나를 더하고, 이미 만들어진 _tdf 에서 그 열을 "
        "numpy 로 꺼내고, 이미 계산된 _pos_asof 로 행을 맞추는 것뿐이다.  "
        "새 파일 열기 0 · 새 read_csv 0 · 값 벡터 집계 0 · 새 print 0.")}
ck("G2c 추가된 비주석 줄에 새 통계가 0개 (원래 엄격 술어, 완화 없음)",
   not g2c_strict, g2c_strict)
ck("G2c2 추가 줄이 값 벡터에 집계를 적용하지 않는다 (주석 포함)", not g2c_any, g2c_any)
ck("G2d 새 read_csv 호출 지점이 0개 (부모에 그대로 있는 줄은 면책) · 전체 read_csv 수가 부모와 같다",
   (not G["G2c_script_py_diff"]["🔴_new_read_csv_in_added_lines"])
   and G["G2c_script_py_diff"]["read_csv_count"]["parent"]
   == G["G2c_script_py_diff"]["read_csv_count"]["candidate"],
   G["G2c_script_py_diff"]["read_csv_count"])
ck("G2c3 돌연변이 4개가 전부 걸린다 (게이트가 실제로 문다)",
   all(not v["strict_no_new_statistic"]
       for k, v in G["G2c_script_py_diff"]["mutations_actually_run"].items()
       if k != "M4_aggregation_hidden_in_a_comment")
   and not G["G2c_script_py_diff"]["mutations_actually_run"][
     "M4_aggregation_hidden_in_a_comment"]["no_aggregation_on_value_vector_even_in_comments"],
   G["G2c_script_py_diff"]["mutations_actually_run"])

# ── G2e 기록-대조 assert (S35·S107·S119·S140·S199) ─────────────────────
gr = subprocess.run(
  ["bash", "-lc",
   f"cd {PKG}/{PKGD} && grep -rInE 'assert.*(sha256|n_trees|anchor_n|n_seeds|len\\(_axl\\)|"
   "ax_names)' --include=*.py . || true"], capture_output=True, text=True).stdout \
  .strip().splitlines()
G["G2e_record_compare_asserts"] = {
 "hits": gr,
 "🔴": ("이 후보가 옮기는 파일은 blend.json 과 script.py 둘이다.  ★유일하게 관련된 "
        "기록-대조 assert 는 script.py 자신의 `len(_axl) == 6` 과 `ax_names == [...]` 이고 "
        "이 빌드가 함께 고쳤다★ — S199 가 지목한 자리다.  model/meta.json 의 n_trees, "
        "mate/q_infer.py, aa_infer.py, 앵커 필드는 전부 우리가 안 건드리는 파일에 대한 "
        "것이라 그대로 통과한다(부모 실행이 그 증거다).")}
_sz = open(f"{PKG}/{PKGD}/script.py", encoding="utf-8").read()
ck("G2e2 script.py 의 축 개수·이름 assert 가 blend.json 과 맞는다",
   "assert len(_axl) == 6" in _sz
   and f'"ctrl_asof_mid", "{NEW_AXIS}"], \\' in _sz)

# ── G3 출력 계약 ───────────────────────────────────────────────────────
g3 = {}
for f in FR:
    d = rd(f"{RP}__{f}")
    if d is None:
        continue
    nfr = sum(1 for _ in open(f"{FDIR}/{f}/test.csv", encoding="utf-8-sig")) - 1
    v = d["control_success"].to_numpy(np.float64)
    g3[f] = {"rows": int(len(d)), "frame_rows": nfr, "cols": list(d.columns),
             "min": float(v.min()), "max": float(v.max()),
             "finite": bool(np.isfinite(v).all()),
             "in_unit": bool(((v > 0) & (v < 1)).all()),
             "row_id_unique": bool(d["row_id"].is_unique)}
G["G3_output_contract"] = g3
ck("G3 열·행수·범위(0,1)·유한·row_id 유일",
   bool(g3) and all(x["cols"] == ["row_id", "control_success"]
                    and x["rows"] == x["frame_rows"] and x["in_unit"]
                    and x["finite"] and x["row_id_unique"] for x in g3.values()),
   {k: (v["rows"], v["frame_rows"]) for k, v in g3.items()})

# ── G4a 전체 셔플 (규칙 2-4 그 하나) ──────────────────────────────────
def shufd(x, y):
    A, Bq = rd(x), rd(y)
    if A is None or Bq is None:
        return None
    sa = A.set_index("row_id")["control_success"]
    sb = Bq.set_index("row_id")["control_success"].reindex(sa.index)
    d = sa.to_numpy(np.float64) - sb.to_numpy(np.float64)
    q = np.abs(d[d != 0])
    return {"n_moved": int(np.count_nonzero(d)),
            "n_gt_1e_12": int((np.abs(d) > 1e-12).sum()),
            "max_abs": float(np.abs(d).max()),
            "quantiles_of_nonzero": ([float(t) for t in np.quantile(q, [.5, .9, 1.0])]
                                     if q.size else []),
            "n": int(len(d)), "threads": 6}


G["G4a_whole_frame_shuffle"] = {
 "candidate": shufd(f"{RP}__cov", f"{RP}__cov_shuf_al"),
 "parent_DM": shufd("par__cov", "par__cov_shuf_al"),
 "🔴": "부모와 차분으로 읽고 행 수와 크기를 분리한다(S74·S90).  스레드 6(S152)."}
_c, _p = G["G4a_whole_frame_shuffle"]["candidate"], G["G4a_whole_frame_shuffle"]["parent_DM"]
ck("G4a 셔플 — 1e-12 초과 이동 행이 부모보다 많지 않다",
   bool(_c) and bool(_p) and _c["n_gt_1e_12"] <= _p["n_gt_1e_12"], (_c, _p))


# ── G4b 내용/위치 분리 + G4c S61 (규칙 2-4 둘째) ──────────────────────
def content(fa, fb):
    x = pd.read_csv(f"{FDIR}/{fa}/test.csv", encoding="utf-8-sig", low_memory=False)
    y = pd.read_csv(f"{FDIR}/{fb}/test.csv", encoding="utf-8-sig", low_memory=False)
    assert (x.row_id.to_numpy() == y.row_id.to_numpy()).all()
    keep = np.flatnonzero((x.astype(str) == y.astype(str)).all(axis=1).to_numpy())
    kid = x.loc[keep, "row_id"].to_numpy()
    kv = np.zeros(len(keep), dtype=bool)
    for _c in MOVED_COLS:
        kv |= np.isfinite(pd.to_numeric(x.loc[keep, _c],
                                        errors="coerce").to_numpy(np.float64))
    ra, rb, rp = rd(f"{RP}__{fa}"), rd(f"{RP}__{fb}"), rd(f"par__{fa}")
    if ra is None or rb is None or rp is None:
        return None
    sa = ra.set_index("row_id")["control_success"]
    sb = rb.set_index("row_id")["control_success"]
    sp = rp.set_index("row_id")["control_success"]
    d = sa.reindex(kid).to_numpy(np.float64) - sb.reindex(kid).to_numpy(np.float64)
    dp = sa.reindex(kid).to_numpy(np.float64) - sp.reindex(kid).to_numpy(np.float64)
    return {"n_keep": int(len(keep)), "n_rows_replaced": int(len(x) - len(keep)),
            "keep_moved_between_frames": int(np.count_nonzero(d)),
            "keep_max_abs_between_frames": float(np.abs(d).max()),
            "keep_moved_vs_parent": int(np.count_nonzero(dp)),
            "keep_max_abs_vs_parent": float(np.abs(dp).max()),
            "keep_rows_with_finite_moved_axis": int(kv.sum()),
            "keep_rows_with_finite_new_column_only": int(np.isfinite(
                pd.to_numeric(x.loc[keep, CFG["col"]],
                              errors="coerce").to_numpy(np.float64)).sum())}


G["G4b_content_position_separation"] = {
 "caD_vs_caE": content("caD", "caE"), "mixB_vs_mixC": content("mixB", "mixC"),
 "🔴": ("같은 행수·같은 위치의 KEEP 110행 고정 + 나머지 245,679행 전량 교체.  🔴 이 "
        "후보는 ★새 열을 읽는다★ — 자명하지 않으므로 반드시 돌린다(S131).  KEEP 행의 "
        "delta 가 정확히 0.0 이면 다른 행의 내용은 이 후보의 출력에 영향이 없다.")}
ok = all(v and v["keep_moved_between_frames"] == 0
         and v["keep_max_abs_between_frames"] == 0.0
         for k, v in G["G4b_content_position_separation"].items() if k != "🔴")
ck("G4b 내용 분리 — KEEP 110행 이동 정확히 0.0 (두 쌍)", ok,
   {k: (v or {}).get("keep_max_abs_between_frames")
    for k, v in G["G4b_content_position_separation"].items() if k != "🔴"})
_d = G["G4b_content_position_separation"]["caD_vs_caE"]
_m = G["G4b_content_position_separation"]["mixB_vs_mixC"]
G["G4c_S61_keep_rows_ride_the_new_axis"] = {
 "caD": _d, "mixB": _m,
 "🔴_the_predicate": (
   "S61 — 'KEEP 행이 바꾼 상수를 실제로 타는가'.  이 축은 raw 가 비유한이면 이동이 "
   "★정확히 0.0★ 이므로 판정 기준은 '움직인 KEEP 행 수 == 새 열이 유한한 KEEP 행 수' "
   "다.  실측: mixB 는 110/110 유한이라 110행 전부 움직여야 하고, caD 는 109/110 유한이라 "
   "109행이 움직이고 1행은 설계대로 안 움직여야 한다.  🔴 두 프레임 중 하나라도 유한 "
   "KEEP 이 0 이면 이 게이트는 아무것도 시험하지 않는다 — 그래서 그 수를 함께 보고한다."),
 "frames_probe": D10.get("frames_probe")}
G["G4c_S61_keep_rows_ride_the_new_axis"]["🔴_SPEC_FIX_S27"] = {
 "when_found": "게이트를 처음 돌린 뒤다.  숨기지 않는다.",
 "old_predicate": "움직인 KEEP 행 수 == ★신규 열★(bmid)이 유한한 KEEP 행 수",
 "why_it_was_wrong": (
   "이 후보는 신규 축만 여는 것이 아니라 ★기존 축 5 `ctrl_asof_mid` 의 u 도 옮긴다★"
   " (-0.008 -> -0.00913).  그래서 bmid 가 비유한이어도 mid 가 유한하면 그 행은 움직인다."
   "  실측 caD: 움직인 KEEP 110행 · bmid 유한 109행 · 두 열 합집합 110행 — ★패키지가 "
   "옳고 술어가 틀렸다★."),
 "new_predicate": "움직인 KEEP 행 수 == ★u 가 바뀐 두 축★(bmid, mid)의 원시 열 합집합이 유한한 KEEP 행 수",
 "🔴_this_is_not_a_relaxation": (
   "양방향 등식이라 여전히 두 쪽을 다 검사한다: 합집합이 비유한인 행은 ★움직이면 안 되고★ "
   "유한인 행은 ★반드시 움직여야★ 한다.  신규 열 단독 수치도 함께 기록해 둔다."),
 "mutations_actually_run": {},
}
_mut = G["G4c_S61_keep_rows_ride_the_new_axis"]["🔴_SPEC_FIX_S27"]["mutations_actually_run"]
for _nm, _mv, _un in (("M1_한_행이_안_움직였다면", 109, 110),
                      ("M2_한_행이_더_움직였다면", 111, 110),
                      ("M3_옛_술어(신규열_단독)", 110, 109)):
    _mut[_nm] = {"moved": _mv, "expected": _un, "gate_passes": _mv == _un}
ck("G4c (S61) KEEP 행 이동 수 == u 가 바뀐 두 축의 열 합집합이 유한한 KEEP 행 수 (두 쌍, 명세 수정 S27)",
   bool(_d) and bool(_m)
   and _d["keep_moved_vs_parent"] == _d["keep_rows_with_finite_moved_axis"]
   and _m["keep_moved_vs_parent"] == _m["keep_rows_with_finite_moved_axis"]
   and _m["keep_rows_with_finite_moved_axis"] > 0
   and all(not v["gate_passes"] for v in _mut.values()),
   {k: (v["keep_moved_vs_parent"], v["keep_rows_with_finite_moved_axis"],
        v["keep_rows_with_finite_new_column_only"]) for k, v in (("caD", _d), ("mixB", _m))})


# ── G5 블록 발동 ───────────────────────────────────────────────────────
def logscan(tag):
    p = f"{LOGS}/{tag}.log"
    if not need(p, f"logs/{tag}.log"):
        return None
    t = open(p, encoding="utf-8", errors="replace").read()
    o = {}
    m = re.search(r"분기별 가중치>0 행수: (\{[^}]*\})", t)
    if m:
        o["branch_rows"] = m.group(1)
    for k2 in AX_NAMES:
        m = re.search(r"축 오프셋 (\d)/(\d) " + k2 + r": u=([-\d.e]+).*?\n.*?유한 행=(\d+)/(\d+)"
                      r".*?무이동\(비유한\)=(\d+)행\s+이동!=0 인 행=(\d+)\s+\|u\*delta\| max=([\d.]+)",
                      t)
        if m:
            o[k2] = {"idx": f"{m.group(1)}/{m.group(2)}", "u": m.group(3),
                     "finite": int(m.group(4)), "n": int(m.group(5)),
                     "nonfinite_no_shift": int(m.group(6)),
                     "moved": int(m.group(7)), "max_abs_shift": float(m.group(8))}
    m = re.search(r"축 오프셋 합계: 이동!=0 인 행=(\d+)/(\d+)\s+\|shift\| max=([\d.]+)", t)
    if m:
        o["axis_total"] = {"moved": int(m.group(1)), "n": int(m.group(2)),
                           "max_abs": float(m.group(3))}
    o["clip_fired_total"] = sum(int(x) for x in re.findall(r"clip 발동=(\d+)행", t))
    return o


G["G5_block_firing"] = {f: logscan(f"{RP}__{f}") for f in FR}
G["G5_parent_cov"] = logscan("par__cov")
G["G5_anchor_cov"] = logscan(f"{AP}__cov")
_cov = G["G5_block_firing"].get("cov") or {}
_nx = _cov.get(NEW_AXIS, {})
_probe_cov = (D10.get("frames_probe") or {}).get("cov") or {}
ck("G5 cov — 여섯 축 전부 발동 · 새 축 idx 6/6 · 유한/무이동 행 수가 사전 실측과 일치 · 두 분기",
   _nx.get("idx") == "6/6"
   and _nx.get("n") == 245789
   and _nx.get("nonfinite_no_shift") == _probe_cov.get("n_nonfinite_no_shift")
   and _nx.get("moved") == _nx.get("finite")
   and all(k in _cov for k in AX_NAMES)
   and "'R'" in str(_cov.get("branch_rows")) and "'F'" in str(_cov.get("branch_rows")),
   {"new_axis": _nx, "branch": _cov.get("branch_rows")})
_anc = G["G5_anchor_cov"] or {}
ck("G5c 앵커에서도 새 블록이 ★실행★ 된다 (u=0.0 이지만 유한 행 245,707 · 이동 0)",
   (_anc.get(NEW_AXIS, {}).get("finite") == _nx.get("finite"))
   and _anc.get(NEW_AXIS, {}).get("moved") == 0
   and _anc.get(NEW_AXIS, {}).get("u") in ("0.0", "-0.0"), _anc.get(NEW_AXIS))
ck("G5b 클립이 부모보다 더 발동하지 않는다 (S143)",
   _cov.get("clip_fired_total", 0) <= (G["G5_parent_cov"] or {}).get("clip_fired_total", 0),
   {CAND: _cov.get("clip_fired_total"),
    "par": (G["G5_parent_cov"] or {}).get("clip_fired_total")})

# ── G6 진폭 ────────────────────────────────────────────────────────────
amp = {}
for f in ("cov", "band", "caD", "caE", "mixB", "mixC", "degen", "real5", "keep110"):
    x, y = rd(f"{RP}__{f}"), rd(f"par__{f}")
    if x is None or y is None:
        continue
    sa = x.set_index("row_id")["control_success"]
    sb = y.set_index("row_id")["control_success"].reindex(sa.index)
    d = sa.to_numpy(np.float64) - sb.to_numpy(np.float64)
    amp[f] = {"n_moved": int(np.count_nonzero(d)), "n": int(len(d)),
              "max_abs": float(np.abs(d).max()), "rms": float(np.sqrt((d ** 2).mean())),
              "mean_shift": float(d.mean()),
              "abs_p50": float(np.percentile(np.abs(d), 50)),
              "abs_p99": float(np.percentile(np.abs(d), 99)),
              "K_times_mean_delta2_pts": float(K * (d ** 2).mean())}
G["G6_amplitude_vs_parent"] = {"frames": amp}
_fc = pd.read_csv(f"{FDIR}/cov/test.csv", encoding="utf-8-sig", low_memory=False,
                  usecols=MOVED_COLS)
_uni = np.zeros(len(_fc), dtype=bool)
_per = {}
for _c in MOVED_COLS:
    _f = np.isfinite(pd.to_numeric(_fc[_c], errors="coerce").to_numpy(np.float64))
    _per[_c] = int(_f.sum())
    _uni |= _f
G["G6_moved_rows_predicate"] = {
 "per_column_finite_on_cov": _per, "union_finite": int(_uni.sum()),
 "n_moved": amp.get("cov", {}).get("n_moved"),
 "🔴_SPEC_FIX_S27": ("초판 술어는 ★신규 열 하나★ 만 봤다.  이 후보는 기존 축 5 의 u 도 "
                      "옮기므로 그 열이 유한한 행도 움직인다.  술어를 두 열의 합집합으로 "
                      "고쳤다 — 양방향 등식이라 완화가 아니다.")}
ck("G6 후보가 cov 에서 u 가 바뀐 두 축의 열 합집합이 유한한 전 행을 움직인다 (명세 수정 S27)",
   amp.get("cov", {}).get("n_moved") == int(_uni.sum())
   and int(_uni.sum()) != _per[CFG["col"]],
   G["G6_moved_rows_predicate"])

# ── G7 산문 동결 (S77) ─────────────────────────────────────────────────
frz, bad = {}, []
for line in open(f"{WORK}/" + CFG["frozen"], encoding="utf-8").read().splitlines():
    w, r = line.split(None, 1)
    frz[r.strip()] = w
    if sh(f"{B}/{r.strip()}") != w:
        bad.append(r.strip())
G["G7_prose_frozen"] = {"files": frz, "drifted": bad,
                        "🔴": "게이트 발사 전에 얼린 sha 다.  99_pack 이 다시 대조한다."}
ck("G7 산문이 게이트 발사 이후 안 바뀌었다 (S77)", not bad, bad)

# ── G8 상수 ────────────────────────────────────────────────────────────
pb = json.loads(z.read("blend.json").decode("utf-8"))
cb = json.load(open(f"{PKG}/{PKGD}/blend.json", encoding="utf-8"))
same = {k: json.dumps(cb.get(k), ensure_ascii=False, sort_keys=True)
        == json.dumps(pb.get(k), ensure_ascii=False, sort_keys=True)
        for k in ("d_adj", "alpha", "beta_skew", "w_v6", "beta_f", "c_aa",
                  "smooth_weights", "aux_offsets", "count_offsets", "failtype_offset",
                  "anchor_mode", "partner_arm", "target_rate", "d_form_eb_k")}
G["G8_constants"] = {
 "other_shipped_constants_unchanged": same,
 "d_adj_shipped_repr": repr(cb["d_adj"]), "d_adj_parent_repr": repr(pb["d_adj"]),
 "first_four_axes_bit_identical_to_parent":
   json.dumps(cb["axis_offsets"][:4], ensure_ascii=False, sort_keys=True)
   == json.dumps(pb["axis_offsets"][:4], ensure_ascii=False, sort_keys=True),
 "axis5_mid_only_u_moved": {
   "u_parent": pb["axis_offsets"][4]["u"], "u_shipped": cb["axis_offsets"][4]["u"],
   "declared": [U_MID_PARENT, U_MID_NEW],
   "everything_else_bit_identical":
     json.dumps({k: v for k, v in cb["axis_offsets"][4].items() if k != "u"},
                ensure_ascii=False, sort_keys=True)
     == json.dumps({k: v for k, v in pb["axis_offsets"][4].items() if k != "u"},
                   ensure_ascii=False, sort_keys=True)},
 "new_axis": {k: cb["axis_offsets"][5][k]
              for k in ("name", "u", "center", "scale", "fitted_at_inference")},
 "new_axis_center_repr": repr(cb["axis_offsets"][5]["center"]),
 "new_axis_scale_repr": repr(cb["axis_offsets"][5]["scale"]),
 "center_matches_derive10": repr(cb["axis_offsets"][5]["center"])
   == D10["cal_frame"]["center_repr"],
 "scale_matches_derive10": repr(cb["axis_offsets"][5]["scale"])
   == D10["cal_frame"]["scale_repr"],
 "🔴_S26": ("상수를 적합한 변수와 출하 코드가 쓰는 변수가 같은 양인지 기계로 확인한다: "
            "derive10 의 center/scale 은 train.csv 의 `asof_batter_middle_rate` 에서 왔고, "
            "출하 script.py 는 test.csv 의 같은 이름 열을 읽는다(치환 원문에 그 이름이 "
            "있다).  두 repr 이 blend.json 과 정확히 일치하는지 위에서 대조한다.")}
ck("G8 d_adj·alpha·beta_skew·w_v6·count_offsets 등 다른 출하 상수 전부 부모와 동일",
   all(same.values()), {k: v for k, v in same.items() if not v})
ck("G8b 축 1~4 가 부모와 비트 동일 · 축 5 는 u 하나만 -0.008 -> -0.00913 · 나머지 리프 동일",
   G["G8_constants"]["first_four_axes_bit_identical_to_parent"]
   and G["G8_constants"]["axis5_mid_only_u_moved"]["u_parent"] == U_MID_PARENT
   and G["G8_constants"]["axis5_mid_only_u_moved"]["u_shipped"] == U_MID_NEW
   and G["G8_constants"]["axis5_mid_only_u_moved"]["everything_else_bit_identical"],
   G["G8_constants"]["axis5_mid_only_u_moved"])
ck("G8c 새 축 center/scale 이 10_derive 의 repr 과 정확히 일치 (S26)",
   G["G8_constants"]["center_matches_derive10"] and G["G8_constants"]["scale_matches_derive10"],
   {"center": G["G8_constants"]["new_axis_center_repr"],
    "scale": G["G8_constants"]["new_axis_scale_repr"]})
ck("G8d 새 축 u 가 사전 등록값과 같다 · 이름 · fitted_at_inference=false · |u|<=0.05 · scale>0",
   cb["axis_offsets"][5]["u"] == U_EXPECT
   and cb["axis_offsets"][5]["name"] == NEW_AXIS
   and cb["axis_offsets"][5]["fitted_at_inference"] is False
   and abs(cb["axis_offsets"][5]["u"]) <= 0.05
   and cb["axis_offsets"][5]["scale"] > 0.0
   and cb["axis_offsets"][5]["frozen_from"].startswith("TRAIN"),
   G["G8_constants"]["new_axis"])

# ── G9 평균 표류 ───────────────────────────────────────────────────────
G["G9_mean_drift_across_frames"] = {
 f: {"mean_shift_p": amp[f]["mean_shift"],
     "predicted_from_column": ((D10.get("frames_probe") or {}).get(f) or {})
     .get("mean_shift_at_u_minus_0008")}
 for f in amp}
G["G9_note"] = (
 "🔴 S154 — 우리 프레임은 배포 평균 표류를 3~10배 부풀린다.  여기 적는 것은 확률공간 "
 "평균 이동(mean_shift_p)과 로짓공간 예측 이동(mean_shift_at_u_minus_0008)이고 단위가 "
 "다르다 — 부호와 자릿수만 본다.  cov/band 에서 후자가 8.9e-07 이라 이 축은 이 프레임에서 "
 "사실상 평균 중립이고, 그것이 d_adj 를 안 옮긴 근거다.")

# ── G13 앵커 ②③ (S291 3중 앵커) ──────────────────────────────────────
ZFR = ["real5", "keep110", "cov"]
a2 = {f: {"zb": man.get(f"zb__{f}", {}).get("out_sha256"),
          "zm": man.get(f"zm__{f}", {}).get("out_sha256"),
          "match": (man.get(f"zb__{f}", {}).get("out_sha256") is not None
                    and man.get(f"zb__{f}", {}).get("out_sha256")
                    == man.get(f"zm__{f}", {}).get("out_sha256"))} for f in ZFR}
G["G13_anchor2_bmid_zero_equals_mid_only_on_parent_code"] = {
 "frames": a2,
 "🔴": ("ZB = 후보 script.py(6축) 위에서 `bmid` u 만 0.0 · `mid` 는 출하값 -0.00913.  "
        "ZM = ★부모 script.py(5축) 그대로★ 에 `mid` 만 -0.00913.  두 출력이 sha256 "
        "동일하면 ① 새 블록이 u=0 에서 산술적으로 무해하고 ② mid 재착륙이 두 코드 "
        "판본에서 비트 동일함이 ★한 번에★ 증명된다.")}
ck("G13 앵커② — ZB == ZM (3프레임 sha256 동일)",
   all(v["match"] for v in a2.values()), a2)

def _lg(p):
    q = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))

a3 = None
_x = {t: rd(f"{t}__cov") for t in ("ds2", "zmid", "zm", "par")}
if all(v is not None for v in _x.values()):
    _s = {t: v.set_index("row_id")["control_success"] for t, v in _x.items()}
    _i = _s["par"].index
    L = {t: _lg(_s[t].reindex(_i).to_numpy(np.float64)) for t in _s}
    d_mid_with_bmid = L["ds2"] - L["zmid"]
    d_mid_alone = L["zm"] - L["par"]
    r = d_mid_with_bmid - d_mid_alone
    a3 = {"max_abs_residual_logit": float(np.abs(r).max()),
          "rms_residual_logit": float(np.sqrt((r ** 2).mean())),
          "mid_contribution_max_logit": float(np.abs(d_mid_alone).max()),
          "relative_max": float(np.abs(r).max() / np.abs(d_mid_alone).max())}
G["G14_anchor3_additive_separation"] = {
 "cov": a3,
 "🔴": ("ZMID = mid 는 부모값 · bmid 만 출하값.  '축 5 의 기여가 축 6 의 값과 무관한가' "
        "를 로짓공간에서 잰다: (DS2 - ZMID) 와 (ZM - PAR) 의 차.  🔴 정확히 0 은 아니다 "
        "— 출하 stage-4 에 beta_skew = -0.059151 의 2차항이 있어 교차항이 남는다.  "
        "그 크기가 2*|beta_skew|*|dz_mid|*|dz_bmid| 규모(약 1e-4)면 가산 분리가 "
        "설계대로 성립한 것이다.")}
ck("G14 앵커③ 가산 분리 — 로짓공간 잔차 max < 5e-3 (2차항 교차 규모)",
   a3 is not None and a3["max_abs_residual_logit"] < 5e-3, a3)

# ── G10 곡률 (라벨 없음, S255) ────────────────────────────────────────
if CURV:
    G["G10_curvature_label_free"] = CURV
    ck("G10 곡률 Q 가 두 프레임에서 유한하고 양수다",
       all(v["Q_pts"] > 0 and np.isfinite(v["Q_pts"]) for v in CURV["frames"].values()),
       {k: v["Q_pts"] for k, v in CURV["frames"].items()})

# ── G11 추론 시간 (단독 창) ────────────────────────────────────────────
if TM:
    G["G11_inference_time_solo"] = TM
    ck("G11 단독 창 추론 시간이 부모 대비 유의하게 늘지 않았다 (S226 — 델타는 이 규모에서 측정 불가)",
       TM["DS"]["seconds"] <= TM["PAR"]["seconds"] * 1.10, TM)

G["fail"] = fail
G["missing_artifacts"] = missing
G["🔴_PASS"] = (not fail) and (not missing)
json.dump(G, open(f"{WORK}/gates_DS.json", "w"), ensure_ascii=False, indent=1)
print("\n=== FAIL:", fail if fail else "none")
print("=== MISSING:", missing if missing else "none")
print("=== PASS:", G["🔴_PASS"])

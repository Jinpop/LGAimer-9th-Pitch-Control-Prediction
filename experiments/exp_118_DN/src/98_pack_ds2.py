#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""zip.  🔴 S45 — 게이트 JSON 이 PASS 가 아니면 모듈 최상단 assert 에서 죽어 zip 을 안 쓴다."""
import ast
import hashlib
import json
import os
import sys
import zipfile

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"
PAR_ZIP = f"{B}/parent_DM.zip"
CAND = "DS"
CFG = dict(pkg="DS2", name="submit_DS_midland_bmid.zip", u=-0.00506,
           axis_key="DS_axes", sweep_key="DS_S73_prose_sweep",
           inh_key="DS_inherited_derived_from",
           df_marker="CANDIDATE DS DOES NOT RE-SOLVE d_adj",
           frozen="frozen_prose_sha_ds2.txt",
           axis="ctrl_asof_bmid", col="asof_batter_middle_rate")
U_MID_NEW, U_MID_PARENT = -0.00913, -0.008
PKGD, NAME = CFG["pkg"], CFG["name"]
DIFF2 = ["blend.json", "script.py"]

GP = f"{WORK}/gates_DS.json"
G = json.load(open(GP, encoding="utf-8"))
assert G["🔴_PASS"] and not G["fail"] and not G["missing_artifacts"], \
    "게이트 실패 — zip 을 만들지 않는다: " + str(G["fail"] + G["missing_artifacts"])
assert G["G1_anchor1_both_axes_at_parent_vs_DM"]["all_match"], "앵커① 실패"
assert all(v["match"] for v in
           G["G13_anchor2_bmid_zero_equals_mid_only_on_parent_code"]["frames"].values()), \
    "앵커② 실패"


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


frozen = {}
for line in open(f"{WORK}/" + CFG["frozen"], encoding="utf-8").read().splitlines():
    w, r = line.split(None, 1)
    frozen[r.strip()] = w
for rel, want in frozen.items():
    got = sh(f"{B}/{rel}")
    assert got == want, f"🔴 산문이 게이트 발사 뒤에 바뀌었다: {rel} {want} -> {got}"
print(f"S77 산문 동결 대조 OK ({len(frozen)}파일)")

src = zipfile.ZipFile(PAR_ZIP)
par_sha = {i.filename: hashlib.sha256(src.read(i.filename)).hexdigest()
           for i in src.infolist() if not i.is_dir()}
have = {os.path.relpath(os.path.join(dp, f), f"{PKG}/{PKGD}")
        for dp, dn, fn in os.walk(f"{PKG}/{PKGD}") if "__pycache__" not in dp for f in fn}
assert have == set(par_sha), sorted(have ^ set(par_sha))[:8]

out = f"{B}/{NAME}"
n_dir = n_file = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
    for i in src.infolist():
        zi = zipfile.ZipInfo(i.filename, date_time=i.date_time)
        zi.compress_type = i.compress_type
        zi.external_attr = i.external_attr
        zi.create_system = i.create_system
        if i.is_dir():
            zo.writestr(zi, b"")
            n_dir += 1
            continue
        zo.writestr(zi, open(os.path.join(PKG, PKGD, i.filename), "rb").read())
        n_file += 1

o = zipfile.ZipFile(out)
names = [i.filename for i in o.infolist()]
fails = []


def cka(nm, cond, det=""):
    if not cond:
        fails.append(nm)
    print(("PASS " if cond else "FAIL ") + f"[{CAND}] " + nm
          + ("  " + str(det)[:200] if det else ""))


cka("멤버 목록·순서가 부모 zip 과 완전히 동일", names == [i.filename for i in src.infolist()])
cka("__pycache__ 없음", not [n for n in names if "__pycache__" in n])
cka("data/ · output/ 멤버 없음",
    not [n for n in names if n.startswith(("data/", "output/"))
         or "/data/" in n or "/output/" in n])
cka("basename 30자 이하", not [n for n in names if len(os.path.basename(n)) > 30],
    max(len(os.path.basename(n)) for n in names))
cka(f"zip 파일 이름 {len(NAME)}자 (30자 이하)", len(NAME) <= 30)
for must in ("script.py", "blend.json", "aa_infer.py", "splits.py",
             "model/deploy_splits.npz", "mate/splits.py", "mate/model/deploy_splits.npz",
             "mate/script.py", "mate/q_infer.py", "mate/deep_infer.py",
             "mate/model_q/meta.json", "model/meta.json", "mate/model/blend.json"):
    cka(f"멤버 존재: {must}", must in names)
cka("학습 전용 자산 없음",
    not [n for n in names if n.endswith((".parquet", ".ipynb"))
         or os.path.basename(n) in ("teacher.npy", "calls.parquet", "qlab.parquet")])
cka("추출 충실도: 모든 멤버가 pkg 파일과 sha 동일",
    all(hashlib.sha256(o.read(n)).hexdigest() == sh(os.path.join(PKG, PKGD, n))
        for n in names if not n.endswith("/")))
d = sorted(n for n in names if not n.endswith("/")
           and (n not in par_sha or par_sha[n] != hashlib.sha256(o.read(n)).hexdigest()))
cka("부모 DM zip 대비 다른 파일이 정확히 2개(blend.json · script.py)", d == DIFF2, d)
_py = [n for n in names if n.endswith(".py")]
_pyd = [n for n in _py if hashlib.sha256(o.read(n)).hexdigest() != par_sha[n]]
cka(f"zip 안 .py {len(_py)}개 중 상이가 정확히 1개(script.py)", _pyd == ["script.py"], _pyd)
for n in _py:
    ast.parse(o.read(n).decode("utf-8"))
cka("모든 .py 가 zip 안에서 파싱된다", True)
_boost = [n for n in names if n.endswith((".cbm", ".txt", ".npz", ".pkl", ".csv"))]
cka(f"🔴 A단 — 부스터·표·가중치 {len(_boost)}개 전량이 부모와 바이트 동일",
    all(hashlib.sha256(o.read(n)).hexdigest() == par_sha[n] for n in _boost))

blz = json.loads(o.read("blend.json").decode("utf-8"))
cbz = json.loads(src.read("blend.json").decode("utf-8"))
AXN = ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split", "sp_bat_phand",
       "ctrl_asof_mid", CFG["axis"]]
cka(f"zip 안 candidate 가 {CAND} 이다 (부모는 자기를 CZ 라 잘못 적었다 — S91)",
    blz["candidate"] == CAND and cbz["candidate"] == "CZ")
cka("🔴 zip 안 axis_offsets 가 6축이고 이름 순서가 script.py assert 와 맞는다",
    [a["name"] for a in blz["axis_offsets"]] == AXN)
cka("🔴 zip 안 축 1~4 가 부모와 비트 동일",
    json.dumps(blz["axis_offsets"][:4], sort_keys=True, ensure_ascii=False)
    == json.dumps(cbz["axis_offsets"][:4], sort_keys=True, ensure_ascii=False))
cka("🔴 zip 안 축 5 `ctrl_asof_mid` 는 u 하나만 -0.008 -> -0.00913 (나머지 리프 비트 동일)",
    blz["axis_offsets"][4]["name"] == "ctrl_asof_mid"
    and cbz["axis_offsets"][4]["u"] == U_MID_PARENT
    and blz["axis_offsets"][4]["u"] == U_MID_NEW
    and json.dumps({k: v for k, v in blz["axis_offsets"][4].items() if k != "u"},
                   sort_keys=True, ensure_ascii=False)
    == json.dumps({k: v for k, v in cbz["axis_offsets"][4].items() if k != "u"},
                  sort_keys=True, ensure_ascii=False))
cka(f"🔴 zip 안 새 축 u = {CFG['u']!r} (OOT 정점 -0.00907 x 전이율 0.617)",
    blz["axis_offsets"][5]["u"] == CFG["u"])
cka("🔴 zip 안 총적재가 -0.009426 (mid + bmid*rho) 로 LB 정점에 앉는다",
    abs((blz["axis_offsets"][4]["u"] + blz["axis_offsets"][5]["u"] * 0.058425)
        - (-0.009426)) < 1e-6)
cka("zip 안 새 축 fitted_at_inference=false · frozen_from 이 TRAIN 으로 시작",
    blz["axis_offsets"][5]["fitted_at_inference"] is False
    and blz["axis_offsets"][5]["frozen_from"].startswith("TRAIN"))
cka("zip 안 새 축 |u| <= 0.05 · scale > 0 (script.py assert 조건)",
    abs(blz["axis_offsets"][5]["u"]) <= 0.05 and blz["axis_offsets"][5]["scale"] > 0.0)
cka("🔴 zip 안 d_adj 가 부모와 한 비트도 다르지 않다 (이 후보는 재유도하지 않았다)",
    repr(blz["d_adj"]) == repr(cbz["d_adj"]))
cka("🔴 zip 안 count_offsets.theta 12칸이 한 비트도 안 움직였다",
    blz["count_offsets"]["theta"] == cbz["count_offsets"]["theta"]
    and len(blz["count_offsets"]["theta"]) == 12)
cka("🔴 zip 안 failtype_offset 이 부모와 완전 동일",
    json.dumps(blz["failtype_offset"], sort_keys=True)
    == json.dumps(cbz["failtype_offset"], sort_keys=True))
for k in ("alpha", "beta_skew", "w_v6", "beta_f", "target_rate", "c_aa",
          "smooth_weights", "aux_offsets", "anchor_mode", "partner_arm"):
    cka(f"zip 안 {k} 가 부모와 동일",
        json.dumps(blz[k], sort_keys=True) == json.dumps(cbz[k], sort_keys=True))
cka("zip 안 derived_from 이 TRAIN 으로 시작 (script.py assert)",
    blz["derived_from"].startswith("TRAIN"))
cka("🔴 zip 안 derived_from 이 DN 세대로 다시 쓰였다 (S240)",
    blz["derived_from"] != cbz["derived_from"]
    and CFG["df_marker"] in blz["derived_from"]
    and repr(blz["d_adj"]) in blz["derived_from"]
    and "CANDIDATE CZ AIMS" not in blz["derived_from"])
cka("🔴 부모 zip 에는 그 문장이 실제로 CZ 세대였다 (수정이 무의미하지 않다)",
    "CANDIDATE CZ AIMS" in cbz["derived_from"])
cka("🔴 물려받은 derived_from 원문이 지워지지 않고 별도 키에 남았다",
    blz[CFG["inh_key"]] == cbz["derived_from"])
cka("계보 키 parent 가 DM 을 지목하고 sha 가 맞는다 (S91)",
    blz["parent"]["🔴_CORRECTED_FOR_THIS_CANDIDATE"]["candidate"] == "DM"
    and blz["parent"]["🔴_CORRECTED_FOR_THIS_CANDIDATE"]["zip_sha256"] == sh(PAR_ZIP))
cka("계보 키 parent 가 부모 zip 의 자기소개 오류를 명시한다 (S91)",
    "거짓" in blz["parent"]["🔴_CORRECTED_FOR_THIS_CANDIDATE"][
        "🔴_the_parent_zip_mislabels_itself"])
cka("identity_anchor 가 이 후보 것으로 다시 쓰였다 (S91)",
    "THE TARGET IS DM" in blz["identity_anchor"]["🔴_CORRECTED_FOR_THIS_CANDIDATE"][
        "🔴_how_to_reproduce_the_parent_from_this_package"]
    and blz["identity_anchor"]["🔴_CORRECTED_FOR_THIS_CANDIDATE"]["files_to_touch"]
        == ["blend.json"])
cka("물려받은 계보 이력이 지워지지 않았다",
    blz["parent"]["inherited_history_verbatim_from_the_parent_zip"] == cbz["parent"]
    and blz["identity_anchor"]["inherited_history_verbatim_from_the_parent_zip"]
        == cbz["identity_anchor"])
cka("🔴 전이율 tau 로 축소했다는 서술이 zip 안에 있다",
    "전이율" in blz["axis_offsets"][5]["derivation"]["🔴_진폭은_전이율_tau_로_축소한_값이다"]
    and "0.617" in blz["axis_offsets"][5]["derivation"]["🔴_진폭은_전이율_tau_로_축소한_값이다"])
cka("🔴 mid 재착륙이 '새 적합이 아님' 으로 서술돼 있다",
    "새로 LB 로 적합한 양은 ★0개★"
    in blz["DS_axes"]["axis_5_ctrl_asof_mid"]["🔴_this_is_not_a_new_fit"])
cka("🔴 strike_rate 를 뺀 이유가 zip 안에 적혀 있다",
    "72%" in blz["DS_axes"]["🔴_WHY_asof_pitcher_strike_rate_IS_NOT_HERE"])
cka("🔴 강건성 표(모든 tau 에서 현 부모를 넘는다)가 zip 안에 있다",
    "tau_0.45" in blz["axis_offsets"][5]["derivation"]["🔴_강건성_모든_tau_에서_현_부모를_넘는다"])
cka("🔴 기반을 넘어 계수를 빌리지 않았음이 zip 안에 적혀 있다 (S262)",
    "다른 후보·다른 기반의 계수를 옮겨 쓰지 않았다"
    in blz["axis_offsets"][5]["derivation"]["🔴_어떤_계수도_기반을_넘어_빌려오지_않았다"])
cka("🔴 LB 결과별 다음 수가 측정 전에 zip 안에 적혀 있다 (S8)",
    set(blz[CFG["axis_key"]]["🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT"])
    >= {"if_gain_is_positive", "if_gain_is_negative", "if_abs_gain_is_under_1.4",
        "🔴_attribution_caveat"})
cka("면책 키 개수 문장이 기계 계수와 일치하는 자리에 있다",
    blz[CFG["sweep_key"]][f"top_level_keys_describing_{CAND}"]["total"]
    == len(blz[CFG["sweep_key"]][f"top_level_keys_describing_{CAND}"]["new"])
    + len(blz[CFG["sweep_key"]][f"top_level_keys_describing_{CAND}"]["changed"]))
cka("자리표시자 잔존 없음", "@@" not in json.dumps(blz, ensure_ascii=False))
cka("changed_vs_DM 이 두 파일이라 적는다",
    "EXACTLY TWO FILES" in blz["changed_vs_DM"]["🔴_WHAT_CHANGED"]
    and blz["changed_vs_DM"]["files_that_differ_from_the_parent_zip"] == DIFF2)
for rel in ("splits.py", "model/deploy_splits.npz", "mate/splits.py",
            "mate/model/deploy_splits.npz", "mate/model_q/meta.json", "model/meta.json",
            "mate/script.py", "mate/q_infer.py", "mate/v6_infer.py", "aa_infer.py",
            "inseason.py", "mate/model/blend.json", "mate/model/deep/prep.json"):
    cka(f"zip 안 {rel} 이 부모 DM 과 바이트 동일",
        hashlib.sha256(o.read(rel)).hexdigest() == par_sha[rel])
sz = o.read("script.py").decode("utf-8")
psz = src.read("script.py").decode("utf-8")
cka("zip 안 script.py 에 assert len(_axl) == 6 이 있다", "assert len(_axl) == 6" in sz)
cka(f"zip 안 script.py 의 축 이름 목록에 {CFG['axis']} 가 있다",
    f'"ctrl_asof_mid", "{CFG["axis"]}"], \\' in sz)
cka("zip 안 script.py 의 assert 메시지가 자기 후보(DS)를 가리킨다 (S40)",
    'assert len(_axl) == 6, "DS는' in sz)
cka("zip 안 script.py 가 test.csv 를 부모보다 더 읽지 않는다",
    sz.count("read_csv") == psz.count("read_csv"))
cka(f"zip 안 script.py 의 코드에서 {CFG['col']} 가 정확히 2번 나온다"
    " (usecols 1 + _tdf 조회 1)",
    sz.count('"' + CFG["col"] + '"') == 2
    and psz.count('"' + CFG["col"] + '"') == 0)

h = hashlib.sha256()
with open(out, "rb") as f:
    for b in iter(lambda: f.read(1 << 20), b""):
        h.update(b)
G["G_ZIP"] = {"zip": NAME, "sha256": h.hexdigest(), "bytes": os.path.getsize(out),
              "members": len(names), "dirs": n_dir, "files": n_file,
              "template_zip": os.path.basename(PAR_ZIP),
              "differing_vs_parent_zip": d,
              "member_sha256": {rel: hashlib.sha256(o.read(rel)).hexdigest()
                                for rel in DIFF2},
              "fail": fails}
G["🔴_PASS"] = G["🔴_PASS"] and not fails
json.dump(G, open(GP, "w"), ensure_ascii=False, indent=1)
print(f"\n[{CAND}] {NAME}  sha256 {h.hexdigest()}\n"
      f"       bytes {os.path.getsize(out)}  members {len(names)} "
      f"(dirs {n_dir} + files {n_file})")
print("ZIP FAILED:", fails if fails else "none")
sys.exit(1 if fails else 0)

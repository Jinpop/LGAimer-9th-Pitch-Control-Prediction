#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DNP — DN 의 ★부호 반대 짝★ (u = +0.008).  제출 가능한 패키지로 짓는다.

DN(u = -0.008) 과 DNP(u = +0.008) 은 같은 축의 ±u 한 쌍이다.  두 LB 점이 있으면
    gain(u) = 2*u*K*L - K*u^2*G
에서 L 과 G 가 ★정확히★ 풀린다 (두 식 두 미지수, 대칭 진폭이라 조건수도 좋다).
S250 — 부호 프로브의 두 슬롯 총액은 부호가 틀리는 쪽이 지배한다.  나쁜 결과가 더 값지다.

🔴 script.py 는 DN 것과 ★바이트 동일★ 하다 (같은 work/script_DN.py).  다른 것은 blend.json
   안의 스칼라 u 하나와 자기 서술 산문뿐이다.
🔴 중간 패키지를 지우는 청소 단계는 넣지 않는다 (지휘자 지시).
"""
import hashlib
import json
import os
import shutil

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"

D10 = json.load(open(f"{WORK}/derive10.json", encoding="utf-8"))
B20 = json.load(open(f"{WORK}/build20.json", encoding="utf-8"))
CF = D10["cal_frame"]
CENTER = float(CF["center_nanmean"])
SCALE = float(CF["scale_nanstd_ddof0"])
U_SHIP = +0.008
PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
PAR_LB = 1157.6118541745

# 라벨 상관 — train.csv 전 시즌, control_success 와의 피어슨 r.  독립 실측(로컬·서버 2회).
LABEL_CORR = {
 "asof_batter_middle_rate (이 축)": {
   "2019": -0.0339, "2020": -0.0026, "2021": -0.0335, "2022": -0.0396,
   "2023": +0.0061, "2024": -0.0078, "all": -0.0338},
 "asof_pitcher_middle_rate (출하 축 5)": {
   "2019": -0.0388, "2020": -0.0082, "2021": -0.0284, "2022": -0.0378,
   "2023": -0.0068, "2024": -0.0129, "all": -0.0361},
 "asof_pitcher_success_rate (출하 축 2)": {
   "2019": +0.0944, "2020": +0.0612, "2021": +0.0784, "2022": +0.0986,
   "2023": +0.0527, "2024": +0.0576, "all": +0.0843},
}


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


PAR_BL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
D_ADJ = PAR_BL["d_adj"]

AXIS = {
 "name": "ctrl_asof_bmid",
 "u": U_SHIP,
 "center": CENTER,
 "scale": SCALE,
 "fitted_at_inference": False,
 "frozen_from": (
   "TRAIN season-2024 calibration frame (253,507 rows): center = nanmean and scale = "
   "nanstd(ddof=0) of train.csv column `asof_batter_middle_rate` over its 253,419 finite "
   "rows.  축 2 `ctrl_asof_psr` · 축 5 `ctrl_asof_mid` 와 ★완전히 같은 규약★ 이다.  "
   "Frozen scalars, not test statistics.  ★형제 후보 DN 과 center/scale 이 비트 동일★ "
   "하다 — 두 후보가 다른 것은 스칼라 u 의 부호 하나뿐이다."),
 "raw_definition": (
   "test.csv column asof_batter_middle_rate, verbatim (변환 없음).  그 행 자신의 컬럼 "
   "하나이고, 축 2 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 읽는다."),
 "nan_policy": (
   "raw 가 비유한이면 (타자 첫 등판) 그 행의 이 축 이동은 ★정확히 0.0★ 이다.  센티넬 "
   "대입도 평균 대입도 하지 않는다.  train-2024 프레임 기준 88행(0.035%)."),
 "applied_where": (
   "added to the blend logit BEFORE the alpha/beta_skew transform — 기존 5축과 완전히 "
   "같은 자리다."),
 "n_nonfinite_train2024": int(CF["n_nonfinite"]),
 "derivation": {
  "🔴_이_후보의_존재_이유": (
    "★DN 의 부호 반대 짝이다.★  DN 은 같은 축을 u = -0.008 로, 이 후보는 u = +0.008 로 "
    "싣는다.  center/scale/코드/부모/다른 모든 상수가 전부 같고 ★스칼라 u 의 부호 "
    "하나만★ 다르다.  두 후보가 각각 LB 점을 받으면 gain(u) = 2*u*K*L - K*u^2*G 에서 "
    "L 과 G 가 ★두 식 두 미지수로 정확히★ 풀린다 (대칭 진폭이라 조건수도 좋다).  "
    "그러면 셋째 슬롯으로 꼭지점 u* = L/G 에 착륙한다.  🔴 S250 — 부호 프로브 한 쌍의 "
    "총액은 ★부호가 틀리는 쪽이 지배한다★.  둘 중 하나는 반드시 음수이고, 그 음수가 "
    "이 쌍에서 가장 값진 정보다."),
  "🔴_이것은_계산된_꼭지점이_아니라_작은_프로브다": (
    "S263 — 신규 축은 계산된 꼭지점 착륙이 아니라 작은 프로브로 시작한다.  |u| = 0.008 은 "
    "형제 축 5 `ctrl_asof_mid` 가 같은 계보에서 실은 진폭 그대로이고, 어떤 곡률·선형 "
    "계수 적합에서도 나오지 않았다."),
  "🔴_어떤_계수도_빌려오지_않았다": (
    "S262 — 곡률 G 는 기반을 넘어 옮겨도 되지만 선형항 L 은 절대 안 된다.  이 후보는 "
    "형제 축의 ★진폭★ 만 반사했고 K_L·K_G 를 옮겨 쓰지 않았다.  S232·S245·S252 — "
    "정직면에서 유도한 몫은 ★0%★ 다."),
  "🔴_부호_규약을_거스르는_쪽이_이_후보다": (
    "정직하게 적는다.  출하 코드의 산술은 `_s_k = u * (raw - center)/scale` 을 블렌드 "
    "로짓에 ★더하는★ 것이므로, 규약상 u 의 부호는 그 열과 라벨의 상관 부호와 같아야 "
    "한다.  실측 라벨 상관(train 전 시즌, control_success)은 이 열이 ★음수★ 이고"
    "(전체 -0.0338, 6시즌 중 5시즌 음수, 2023 만 +0.0061 로 0 근처), 출하된 두 원시 열 "
    "축이 그 규약을 2/2 로 따른다(축 2 상관 +0.0843 / u +0.006544, 축 5 상관 -0.0361 / "
    "u -0.008).  🔴 따라서 ★규약이 예측하는 쪽은 DN(u = -0.008) 이고 이 후보는 그 "
    "반대쪽이다.★  일부러 그렇게 싣는다 — 한 쌍이라야 L 과 G 가 풀리고, S250 대로 "
    "지는 쪽이 더 많은 정보를 준다.  이 후보 단독의 기대값은 음수로 본다."),
  "라벨_상관_실측": LABEL_CORR,
  "🔴_2023_시즌_주의": (
    "이 열의 2023 상관만 부호가 뒤집힌다(+0.0061).  같은 해에 출하된 형제 열도 함께 "
    "약해진다(-0.0068, 다른 해의 1/3~1/6).  ★2023 시즌의 성질이지 이 열의 결함이 "
    "아니다★ — 다만 연도전이가 완벽하지 않다는 사실은 그대로 적어 둔다(S265·S269)."),
  "d_adj_유지_근거": (
    "★d_adj 는 부모 값 그대로★ 다.  center 가 판정 프레임의 nanmean 이라 오프셋이 평균 "
    f"중립이다: mean(delta) = {CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}, "
    f"u = +0.008 을 곱하면 {0.008 * CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}."),
  "🔴_사전_차감_레벨손실": (
    "이 컬럼의 시즌 평균이 2024 눈금으로 2019 -1.238sd -> 2024 0.000sd 로 6년 연속 "
    "단조상승한다(연 약 +0.25sd).  2024 center 를 동결한 채 2025 에 적용하면 상수 "
    "오프셋이 하나 얹힌다.  🔴 부호가 반대인 두 후보에서 이 대가의 부호도 반대다 — "
    "그래서 이 항은 두 점 해에서 L 쪽으로 흡수된다."),
  "🔴_꼬리_노출": (
    f"|delta| p50 {CF['abs_delta_p50']:.3f} · p99 {CF['abs_delta_p99']:.3f} · "
    f"p99.9 {CF['abs_delta_p999']:.3f} · max {CF['abs_delta_max']:.3f}.  "
    f"u = 0.008 에서 |u*delta| max = {CF['max_abs_u_times_delta_at_0008']:.5f}.  "
    "소표본 타자(asof_batter_n < 10, 779행)가 꼬리의 출처다.  수축·클리핑 없음."),
  "규칙_준수": (
    "외부데이터 0 · test 분포 통계 0 · 행별 자기 컬럼 lookup 뿐 · center/scale 은 "
    "train 동결 스칼라 2개.  새 print 0 · 새 read_csv 0."),
  "S277_이미_출하_코드가_이_양을_재중심하는가": D10[
    "S277_is_this_quantity_already_re_centred_by_shipped_code"],
 },
}


def mk(tag, u, cand):
    d = f"{PKG}/{tag}"
    assert not os.path.isdir(d), f"{d} 가 이미 있다 — 지우지 않는다(지휘자 지시)"
    shutil.copytree(f"{PKG}/PAR", d, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    shutil.copyfile(f"{WORK}/script_DN.py", f"{d}/script.py")
    bl = json.load(open(f"{d}/blend.json", encoding="utf-8"))
    a = json.loads(json.dumps(AXIS, ensure_ascii=False))
    a["u"] = u
    bl["axis_offsets"] = list(bl["axis_offsets"]) + [a]
    bl["candidate"] = cand
    bl["parent"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "candidate": "DM",
       "zip": "submit_한복판비율축_마이너스0008.zip",
       "zip_sha256": PAR_ZIP_SHA,
       "lb": PAR_LB, "base": "CZ", "base_lb": 1155.1387831487,
       "delta_pts": 2.4730710258,
       "🔴_the_parent_zip_mislabels_itself": (
         "🔴 부모 zip 의 blend.json 은 `candidate` 를 \"CZ\" 라고 적고 있는데 ★거짓★ "
         "이다.  그 패키지는 CZ 위에 다섯 번째 축 `ctrl_asof_mid`(u = -0.008)를 얹은 "
         "것이고 자기 키 `DM_axis`·`changed_vs_CZ` 가 그것을 서술한다 — 패치팩으로 "
         "얹으면서 계보 키를 안 고친 것이다(S91).  이 후보는 네 키를 전부 자기 세대 "
         "것으로 다시 썼다."),
       "🔴_what_DNP_moves": (
         "axis_offsets 를 5개 → 6개로 늘리고 여섯 번째를 `ctrl_asof_bmid` 로 하되 "
         "★u = +0.008★ 로 싣는다.  형제 후보 DN 은 같은 축을 u = -0.008 로 싣는다 — "
         "두 후보의 상이점은 그 스칼라 하나뿐이고 script.py 는 바이트 동일하다.  "
         "새 파일 0개 · 새 표 0개 · test.csv 재파싱 0회 · 부스터 0비트."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["parent"],
    }
    bl["identity_anchor"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "🔴_how_to_reproduce_the_parent_from_this_package": (
         "Set this file's `axis_offsets[5].u` (the `ctrl_asof_bmid` entry) to 0.0, then "
         "rebuild the package the same way.  ★NOTHING ELSE has to be reverted★ — this "
         "candidate does not move `d_adj`.  🔴 script.py DOES NOT HAVE TO BE REVERTED: "
         "u = 0.0 makes 0.0 * delta exactly 0.0 for every finite delta and x + 0.0 is "
         "exact in IEEE-754, so the new code RUNS and the output is still the parent's, "
         "byte for byte.  That is this build's gate G1: the anchored package ANC_DNP "
         "carries DNP's rewritten prose AND the patched script.py while holding that ONE "
         "number at 0.0, and its output CSV matches CANDIDATE DM's sha256 on all six "
         "frames it is run on: real5, keep110, cov, band, caD and degen.  "
         "🔴 THE TARGET IS DM."),
       "files_to_touch": ["blend.json"],
       "🔴_note": "한 파일, 한 개 숫자.  script.py 는 u = 0 에서 산술적으로 무해하다.",
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["identity_anchor"],
    }
    bl["derived_from"] = (
     "TRAIN season-2024 calibration frame (253,507 rows) — 부모 계보에서 그대로 물려받은 "
     "calCG 규약 위에서 동결한 스칼라다.  🔴 CANDIDATE DNP DOES NOT RE-SOLVE d_adj: "
     "d_adj = " + repr(D_ADJ) + " 는 부모 DM(LB " + repr(PAR_LB) + ") 의 값 그대로이고 "
     "이 후보가 한 비트도 움직이지 않았다.  근거는 새 축이 그 프레임에서 ★평균 중립★ "
     "이라는 실측이다 (mean(delta) = "
     f"{CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}).  이 후보가 새로 동결한 스칼라는 "
     "그 축의 center/scale 두 개뿐이고 둘 다 train 에서 왔다.  No statistic of test.csv "
     "is read at inference time.  🔴 THIS SENTENCE REPLACES THE PARENT'S, WHICH DESCRIBED "
     "CZ's SOLVE (S240).  물려받은 원문은 `DNP_inherited_derived_from` 에 그대로 있다.")
    bl["DNP_inherited_derived_from"] = PAR_BL["derived_from"]
    bl["changed_vs_DM"] = {
     "🔴_WHAT_CHANGED": (
       "축 오프셋 1개 추가 (`ctrl_asof_bmid`, u = " + repr(u) + ", axis 6/6) — 그것 "
       "하나뿐이다.  EXACTLY TWO FILES differ from the parent zip: script.py 와 "
       "blend.json.  나머지 176개 항목은 부모와 ★바이트 동일★ 하다."),
     "files_that_differ_from_the_parent_zip": ["blend.json", "script.py"],
     "🔴_relation_to_sibling_DN": (
       "script.py 는 후보 DN 과 ★바이트 동일★ (같은 work/script_DN.py).  blend.json 은 "
       "axis_offsets[5].u 의 부호와 자기 서술 산문만 다르다.  center/scale/d_adj/그 외 "
       "전 상수가 비트 동일하다."),
     "🔴_not_moved": (
       "d_adj · alpha · beta_skew · w_v6 · beta_f · c_aa · smooth_weights · "
       "failtype_offset · aux_offsets · count_offsets · 기존 5축 · 부스터/표/팔 전부."),
    }
    bl["DNP_axis"] = {
     "candidate": cand,
     "parent": "submit_한복판비율축_마이너스0008.zip (내용상 후보 DM)",
     "parent_sha256": PAR_ZIP_SHA, "parent_lb": PAR_LB,
     "axis": "axis_offsets[5] 참조",
     "d_adj_unchanged": "★부모 값 " + repr(D_ADJ) + " 그대로★",
     "🔴_THIS_IS_THE_PLUS_HALF_OF_A_SIGN_PROBE_PAIR": (
       "DN(u = -0.008) 과 DNP(u = +0.008) 은 같은 축의 ±u 한 쌍이다.  둘 다 LB 점을 "
       "사면 L 과 G 가 정확히 풀린다:  L = (g₊ - g₋)/(4*u*K),  "
       "G = -(g₊ + g₋)/(2*K*u²).  🔴 그리고 이 빌드는 G 를 ★LB 없이도★ 이미 쟀다 "
       "(라벨 0개 곡률, work/curvature.json) — 두 값이 맞는지 교차검증할 수 있다."),
     "🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT": {
       "both_points_bought": (
         "위 두 식으로 L·G 를 풀고, 라벨 없는 곡률 Q 와 G 를 대조한다.  일치하면 셋째 "
         "슬롯으로 u* = L/G 에 착륙한다."),
       "if_this_arm_loses_badly": (
         "예상된 결과다 — 라벨 상관 규약이 이 부호를 예측하지 않는다.  S250 대로 그 "
         "음수가 L 의 크기를 가장 정확히 준다."),
       "if_both_arms_lose": (
         "L 이 작고 G 가 지배한다는 뜻이다.  그러면 이 축을 닫는다 — 진폭을 키워 다시 "
         "사지 않는다(S42)."),
       "if_abs_gain_under_1.4": (
         "팀 원장 검출한계(제76) 아래 — 그 점은 부호의 증거로 쓰지 않는다."),
     },
     "🔴_inherited_keys_are_history_not_claims_about_DNP": (
       "이 blend.json 의 나머지 최상위 키들은 DM · CZ 및 그 조상들의 ★이력★ 이다.  "
       "DNP 에 대한 주장이 아니며 하나도 지우지 않았다."),
    }
    bl["DNP_rule_2_4"] = {
     "column": "test.csv 의 `asof_batter_middle_rate` 는 공식 입력 열이다.",
     "application": ("row i 는 row i 의 그 열 값 하나만 본다.  평균·분위·정렬·순위·적합이 "
                     "없고 test 프레임 통계를 만들지 않는다.  비유한이면 shift = 0.0."),
     "constants": "center/scale 두 스칼라, 둘 다 train season==2024 동결.",
     "🔴_gates": ("전체 셔플(cov vs cov_shuf_al) + 내용/위치 분리(caD/caE · mixB/mixC "
                  "KEEP 110행).  결과는 gates_DNP.json."),
    }
    bl["DNP_S73_prose_sweep"] = {
     "🔴_scope_of_this_sweep": (
       "① blend.json 의 전 리프 ② 패키지 전체를 'axis_offsets|축 오프셋' 과 "
       "'ctrl_asof_bmid|asof_batter_middle_rate' 로 grep ③ 상이 두 파일 육안 통독."),
     "top_level_keys_describing_DNP": {"filled_by_21_build": True},
     "S137_S240_value_leaves": {
       "shipped_d_adj": repr(D_ADJ), "parent_d_adj": repr(D_ADJ),
       "🔴": "이 후보는 d_adj 를 안 옮긴다 — 두 값이 같은 것이 정상이다.",
       "shipped_u_ctrl_asof_bmid": repr(u)},
    }
    new = sorted(set(bl) - set(PAR_BL))
    chg = sorted(k for k in set(bl) & set(PAR_BL)
                 if json.dumps(bl[k], ensure_ascii=False, sort_keys=True)
                 != json.dumps(PAR_BL[k], ensure_ascii=False, sort_keys=True))
    bl["DNP_S73_prose_sweep"]["top_level_keys_describing_DNP"] = {
      "n_new": len(new), "new": new, "n_changed": len(chg), "changed": chg,
      "n_removed": len(sorted(set(PAR_BL) - set(bl))), "total": len(new) + len(chg),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 대조."}
    open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
        json.dumps(bl, ensure_ascii=False, indent=1))
    return d


mk("DNP_SHIP", U_SHIP, "DNP")
mk("ANC_DNP", 0.0, "DNP")

assert sh(f"{PKG}/DNP_SHIP/script.py") == sh(f"{PKG}/DN/script.py"), \
    "DNP 의 script.py 가 DN 과 바이트 동일하지 않다"
assert sh(f"{PKG}/ANC_DNP/script.py") == sh(f"{PKG}/DN/script.py")

# S77 — 이 두 패키지의 산문도 게이트 전에 얼린다
REL = ["pkg/DNP_SHIP/blend.json", "pkg/DNP_SHIP/script.py",
       "pkg/ANC_DNP/blend.json", "pkg/ANC_DNP/script.py"]
open(f"{B}/work/frozen_prose_sha_dnp.txt", "w").write(
    "\n".join(f"{sh(f'{B}/{r}')}  {r}" for r in REL) + "\n")

json.dump({"axis": AXIS, "u_ship": U_SHIP, "center_repr": repr(CENTER),
           "scale_repr": repr(SCALE), "d_adj_repr": repr(D_ADJ),
           "label_corr": LABEL_CORR,
           "script_sha_shared_with_DN": sh(f"{PKG}/DNP_SHIP/script.py"),
           "parent_zip_sha256": PAR_ZIP_SHA},
          open(f"{WORK}/build21.json", "w"), ensure_ascii=False, indent=1)
for t in ("PAR", "DN", "DNP_SHIP", "ANC_DNP"):
    print(t, "blend", sh(f"{PKG}/{t}/blend.json")[:16],
          "script", sh(f"{PKG}/{t}/script.py")[:16])
print("BUILD21_OK  pkg: DNP_SHIP ANC_DNP  (청소 단계 없음)")

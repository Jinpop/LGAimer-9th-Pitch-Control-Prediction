#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DS (개정 사양) — 기존 축 5 를 정점에 재착륙시키고 새 축 하나를 얹는다.

    축 5  ctrl_asof_mid   u : -0.008  ->  -0.00913   (기존 축, 정점 재착륙 — 새 적합 아님)
    축 6  ctrl_asof_bmid  u : 0       ->  -0.00506   (유일한 신규 축)

🔴 `asof_pitcher_strike_rate` 는 싣지 않는다 — OOT K*L 이 가장 컸으나 투수블록 널을
   못 넘는다는 보고(ratio_true 0.615)가 있고, 결정 행렬상 그 축이 진짜일 확률이 72% 를
   넘어야 싣는 값이 있다.  근거를 받기 전에는 싣지 않는다.

🔴 script.py 는 후보 DN 의 6축 판과 ★자기 서술 문장 두 줄만★ 다르다 (assert 메시지와
   축 6 머리표).  신규 축이 `bmid` 하나뿐이라 코드 구조가 같다.
🔴 청소 단계 없음.
"""
import hashlib
import json
import os
import shutil

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"
D10 = json.load(open(f"{WORK}/derive10.json", encoding="utf-8"))
CF = D10["cal_frame"]
U_MID_NEW = -0.00913          # 기존 축 5 정점 재착륙 (이미 치른 LB 점에서 유도)
U_MID_PARENT = -0.008
U_BMID = -0.00506             # 신규 축, OOT 정점 -0.00907 x 전이율 tau 0.617
TAU = 0.617
PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
PAR_LB = 1157.6118541745
RHO_MID_BMID = 0.058425
TOTAL_LOAD = U_MID_NEW + U_BMID * RHO_MID_BMID


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── script.py — DN 의 6축 판에서 자기 서술 두 줄만 고친다 ───────────────
SRC = open(f"{WORK}/script_DN.py", encoding="utf-8").read()
REP = [
 ('    assert len(_axl) == 6, "DN은 축 오프셋을 정확히 6개 쓴다 (ctrl_asof_bmid 추가)"\n',
  '    assert len(_axl) == 6, "DS는 축 오프셋을 정확히 6개 쓴다 (ctrl_asof_bmid 추가)"\n'),
 ('        # --- axis 6/6 `ctrl_asof_bmid` -- 후보 DN -------------------------\n',
  '        # --- axis 6/6 `ctrl_asof_bmid` -- 후보 DS -------------------------\n'),
]
NEW = SRC
for i, (a, b) in enumerate(REP):
    assert NEW.count(a) == 1, f"치환 {i}: 원문 {NEW.count(a)}회"
    NEW = NEW.replace(a, b)
    assert b in NEW, f"치환 {i} 미적용"
PARSRC = open(f"{PKG}/PAR/script.py", encoding="utf-8").read()
assert NEW.count("read_csv") == PARSRC.count("read_csv"), "test.csv 재파싱 횟수가 늘었다"
open(f"{WORK}/script_DS2.py", "w", encoding="utf-8", newline="").write(NEW)
print(f"script_DS2.py: DN 판 대비 치환 {len(REP)}건 (자기 서술만)")

PAR_BL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
D_ADJ = PAR_BL["d_adj"]
MID_PARENT = json.loads(json.dumps(PAR_BL["axis_offsets"][4], ensure_ascii=False))
assert MID_PARENT["name"] == "ctrl_asof_mid" and MID_PARENT["u"] == U_MID_PARENT

ROBUST = {"tau_0.45": {"현_부모": 2.473, "이_배치": 2.823},
          "tau_0.617": {"현_부모": 2.473, "이_배치": 3.260},
          "tau_0.8": {"현_부모": 2.473, "이_배치": 3.739},
          "tau_1.0": {"현_부모": 2.473, "이_배치": 4.262},
          "tau_1.5": {"현_부모": 2.473, "이_배치": 5.571},
          "최악": {"현_부모": 2.473, "이_배치": 2.823},
          "🔴": ("모든 tau 에서 현 부모를 넘는 유일한 배치다.  프로브 진폭 0.008 판은 "
                 "tau=0.45 에서 +1.501 로 현 부모 아래로 떨어진다.")}

AXIS6 = {
 "name": "ctrl_asof_bmid", "u": U_BMID,
 "center": float(CF["center_nanmean"]), "scale": float(CF["scale_nanstd_ddof0"]),
 "fitted_at_inference": False,
 "frozen_from": ("TRAIN season-2024 calibration frame (253,507 rows): center = nanmean, "
                 "scale = nanstd(ddof=0) of train.csv column `asof_batter_middle_rate` "
                 "over its 253,419 finite rows.  축 2·5 와 완전히 같은 규약."),
 "raw_definition": ("test.csv column asof_batter_middle_rate, verbatim (변환 없음).  "
                    "축 2 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 "
                    "읽는다 — 재파싱 0회 증가(빌드가 read_csv 호출 수를 assert 로 대조)."),
 "nan_policy": ("비유한이면 이 축 이동은 ★정확히 0.0★ 이다.  센티넬도 평균 대입도 "
                "하지 않는다.  train-2024 기준 88행(0.035%)."),
 "applied_where": "added to the blend logit BEFORE the alpha/beta_skew transform.",
 "n_nonfinite_train2024": int(CF["n_nonfinite"]),
 "derivation": {
  "🔴_부호는_표본_밖_잔차가_정했다": (
    "이 축의 u 부호는 ★out-of-time 정직면(<=2023 학습 팔 -> 2024 행 253,507개)에서 잰 "
    "잔차화 선형항★ 이 정했다: 출하 5축으로 G-계량 잔차화한 K*L = ★-258.6★ (원 -249.5, "
    "유지율 103.7%) — 음수이므로 u 는 음수다.  🔴 그 측정은 출하 축 둘로 ★양성 대조 "
    "2/2★ 를 통과했다: `asof_pitcher_middle_rate` K*L -406.3 -> 출하 u -0.008 (부호 일치) · "
    "`asof_pitcher_success_rate` K*L +74.8 -> 출하 u +0.006544 (부호 일치).  "
    "🔴 같은 대조를 표본 안 프레임(cov·band)에 걸었더니 ★2/2 로 부호가 반대★ 였고"
    "(+561.7 · -863.8) 그래서 표본 안 면을 기각했다.  원인은 배포 팔이 <=2024 학습인데 "
    "그 프레임이 2024 행이라 잔차가 정직하지 않은 것이다(S210).  "
    "라벨 상관(전체 -0.0338, 6시즌 중 5시즌 음수)도 같은 부호다 — 이 축은 두 근거가 "
    "★충돌하지 않는다★."),
  "🔴_진폭은_전이율_tau_로_축소한_값이다": (
    f"OOT 면의 정점은 u* = -0.00907 이다.  거기에 ★전이율 tau = {TAU}★ 를 곱해 "
    f"u = {U_BMID} 를 실었다.  tau 의 출처는 ★출하 축 `ctrl_asof_mid` 의 LB 실측 한 "
    "점★ 이다: 우리 양성 대조가 그 축에서 OOT/실제 = 1.62 를 줬고 그 역수 0.617 이 "
    "팀원의 독립 경로(f24 정점 -0.0154 vs LB 정점 -0.0094)가 준 0.610 과 일치한다.  "
    "🔴 tau = 1 가정은 우리 원장이 4/4 로 반증했다(S232·S245·S252 — 정직면 유래 몫 "
    "100% 축 합계 -38.6).  그래서 정직면 정점에 그대로 착륙하지 않고 축소했다."),
  "🔴_강건성_모든_tau_에서_현_부모를_넘는다": ROBUST,
  "🔴_어떤_계수도_기반을_넘어_빌려오지_않았다": (
    "S262 — 곡률 G 는 기반을 넘어 옮겨도 되고 선형항 L 은 안 된다.  이 배치의 "
    "L 은 ★이 기반에서 산 LB 점★(축 5 의 +2.4730710258)과 ★같은 면에서 잰 OOT 잔차★ "
    "에서만 왔다.  다른 후보·다른 기반의 계수를 옮겨 쓰지 않았다."),
  "라벨_상관_실측": {"2019": -0.0339, "2020": -0.0026, "2021": -0.0335, "2022": -0.0396,
                     "2023": 0.0061, "2024": -0.0078, "all": -0.0338},
  "🔴_사전_차감_레벨손실": (
    "이 컬럼의 시즌 평균이 2024 눈금으로 2019 -1.238sd -> 2024 0.000sd 로 6년 연속 "
    "단조상승한다(연 약 +0.25sd).  2024 center 를 동결한 채 2025 에 적용하면 상수 "
    "오프셋이 하나 얹힌다.  추정이지 측정이 아니다."),
  "🔴_꼬리_노출": (
    f"|delta| p50 {CF['abs_delta_p50']:.3f} · p99 {CF['abs_delta_p99']:.3f} · "
    f"p99.9 {CF['abs_delta_p999']:.3f} · max {CF['abs_delta_max']:.3f}.  "
    f"u = {U_BMID} 에서 |u*delta| max = {abs(U_BMID) * CF['abs_delta_max']:.5f}.  "
    "소표본 타자가 꼬리의 출처.  수축·클리핑 없음."),
  "곡률_실측": (
    "이 축의 자기곡률을 ★라벨 0개·LB 슬롯 0개★ 로 쟀다(S255): u=±0.008 두 판본에서 "
    "Q = (K/2)[mean((p+-p0)^2)+mean((p--p0)^2)] = ★1.8242 pts (cov) · 1.8616 (band)★, "
    "K*G = 28,503 / 29,088.  두 반쪽 비 0.996~0.998 로 2차식 모형이 잘 맞는다.  "
    "🔴 프레임은 대회 test 가 아니라 게이트 대역 프레임이다(S209)."),
  "규칙_준수": (
    "외부데이터 0 · test 분포 통계 0 · ★행별 자기 컬럼 lookup 뿐★ · center/scale 은 "
    "train 동결 스칼라 2개.  새 print 0 · 새 read_csv 0."),
  "S277_이미_출하_코드가_이_양을_재중심하는가": D10[
    "S277_is_this_quantity_already_re_centred_by_shipped_code"],
 },
}


def mk(tag, u_mid, u_bmid, cand, script, note=None):
    d = f"{PKG}/{tag}"
    assert not os.path.isdir(d), f"{d} 가 이미 있다 — 지우지 않는다(지휘자 지시)"
    shutil.copytree(f"{PKG}/PAR", d, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    if script:
        shutil.copyfile(script, f"{d}/script.py")
    bl = json.load(open(f"{d}/blend.json", encoding="utf-8"))
    bl["axis_offsets"][4]["u"] = u_mid
    if u_bmid is not None:
        a6 = json.loads(json.dumps(AXIS6, ensure_ascii=False))
        a6["u"] = u_bmid
        bl["axis_offsets"] = list(bl["axis_offsets"]) + [a6]
    bl["candidate"] = cand
    if note:
        bl["🔴_THIS_PACKAGE_IS_NEVER_SHIPPED"] = note
        open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
            json.dumps(bl, ensure_ascii=False, indent=1))
        return d
    bl["parent"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "candidate": "DM", "zip": "submit_한복판비율축_마이너스0008.zip",
       "zip_sha256": PAR_ZIP_SHA, "lb": PAR_LB, "base": "CZ",
       "base_lb": 1155.1387831487, "delta_pts": 2.4730710258,
       "🔴_the_parent_zip_mislabels_itself": (
         "🔴 부모 zip 의 blend.json 은 `candidate` 를 \"CZ\" 라고 적고 있는데 ★거짓★ "
         "이다.  그 패키지는 CZ 위에 다섯 번째 축 `ctrl_asof_mid`(u=-0.008)를 얹은 "
         "것이고 자기 키 `DM_axis`·`changed_vs_CZ` 가 그것을 서술한다(S91).  이 후보는 "
         "네 계보 키를 전부 자기 세대 것으로 다시 썼다."),
       "🔴_what_DS_moves": (
         "★두 개다.★  ① 기존 축 5 `ctrl_asof_mid` 의 u 를 " + repr(U_MID_PARENT)
         + " -> " + repr(U_MID_NEW) + " 로 옮긴다 (정점 재착륙 — 새 적합이 아니라 "
         "이미 치른 LB 점에서 유도).  ② 여섯째 축 `ctrl_asof_bmid`(u=" + repr(U_BMID)
         + ") 를 새로 연다.  ★새 파일 0개 · 새 표 0개 · test.csv 재파싱 0회 · 부스터 "
         "0비트★.  d_adj·alpha·beta_skew·w_v6·failtype_offset·aux_offsets·count_offsets·"
         "축 1~4 는 전부 부모 값 그대로다."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["parent"],
    }
    bl["identity_anchor"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "🔴_how_to_reproduce_the_parent_from_this_package": (
         "Set `axis_offsets[5].u` (`ctrl_asof_bmid`) to 0.0 AND set `axis_offsets[4].u` "
         "(`ctrl_asof_mid`) back to " + repr(U_MID_PARENT) + ", then rebuild the package "
         "the same way.  ★NOTHING ELSE has to be reverted★ — this candidate does not move "
         "`d_adj` (both axes are mean-neutral on the calibration frame: the `mid` axis has "
         "mean(delta) = -1.82e-15 so the u change moves the mean by 2.1e-18, and the new "
         "axis has mean(delta) = 1.5e-15).  🔴 script.py DOES NOT HAVE TO BE REVERTED: "
         "u = 0.0 makes 0.0 * delta exactly 0.0 and x + 0.0 is exact in IEEE-754, so the "
         "new code RUNS and the output is still the parent's, byte for byte.  "
         "🔴 THE TARGET IS DM.\n"
         "🔴 이 빌드는 ★3중 앵커★ 를 돌린다(S291): ① 두 축을 동시에 부모값으로 되돌린 "
         "ANC_DS2 가 부모 sha 와 일치 ② `bmid` 만 0.0 으로 둔 ZB 가 ★부모 script.py "
         "(5축) 위에서 mid 만 -0.00913 으로 옮긴 ZM★ 과 일치 — 이것이 새 블록이 u=0 에서 "
         "무해함과 mid 재착륙이 두 코드 판본에서 동일함을 동시에 증명한다 ③ mid 만 "
         "부모값으로 되돌린 ZMID 로 두 축의 기여가 가산 분리됨을 확인한다."),
       "files_to_touch": ["blend.json"],
       "🔴_note": "한 파일, 두 개 숫자.  script.py 는 두 u 가 부모값일 때 산술적으로 무해하다.",
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["identity_anchor"],
    }
    bl["derived_from"] = (
     "TRAIN season-2024 calibration frame (253,507 rows) — 부모 계보에서 그대로 물려받은 "
     "calCG 규약 위에서 동결한 스칼라다.  🔴 CANDIDATE DS DOES NOT RE-SOLVE d_adj: "
     "d_adj = " + repr(D_ADJ) + " 는 부모 DM(LB " + repr(PAR_LB) + ") 의 값 그대로이고 "
     "이 후보가 한 비트도 움직이지 않았다.  근거는 두 축이 그 프레임에서 ★평균 중립★ "
     "이라는 실측이다: 축 5 는 mean(delta) = -1.8225e-15 라 u 를 " + repr(U_MID_PARENT)
     + " -> " + repr(U_MID_NEW) + " 로 옮겨도 평균이 2.06e-18 만 움직이고, 축 6 은 "
     f"mean(delta) = {CF['mean_delta_all_rows_nonfinite_as_zero']:.4e} 다.  "
     "이 후보가 새로 동결한 스칼라는 축 6 의 center/scale 두 개뿐이고 둘 다 train 에서 "
     "왔다.  No statistic of test.csv is read at inference time.  🔴 THIS SENTENCE "
     "REPLACES THE PARENT'S, WHICH DESCRIBED CZ's SOLVE (S240).  물려받은 원문은 "
     "`DS_inherited_derived_from` 에 그대로 있다.")
    bl["DS_inherited_derived_from"] = PAR_BL["derived_from"]
    bl["changed_vs_DM"] = {
     "🔴_WHAT_CHANGED": (
       "EXACTLY TWO FILES differ from the parent zip: script.py 와 blend.json.  "
       "옮긴 숫자는 둘이다 — 축 5 `ctrl_asof_mid` 의 u (" + repr(U_MID_PARENT) + " -> "
       + repr(U_MID_NEW) + ") 와 새로 연 축 6 `ctrl_asof_bmid` 의 u (" + repr(U_BMID)
       + ").  나머지 176개 항목은 부모와 ★바이트 동일★ 하다."),
     "files_that_differ_from_the_parent_zip": ["blend.json", "script.py"],
     "script_py": ("len(_axl) assert 5->6 · ax_names 에 여섯째 이름 추가 · 축 2 의 "
                   "test.csv read 에 usecols 로 asof_batter_middle_rate 추가 · 축 6 raw "
                   "벡터 2줄 · _rawmap 항목 1개 · 머리 주석 FIVE->SIX 와 축 번호표.  "
                   "★새 read_csv 0 · 새 집계 0 · 새 print 0★."),
     "🔴_not_moved": ("d_adj · alpha · beta_skew · w_v6 · beta_f · c_aa · smooth_weights · "
                      "failtype_offset · aux_offsets · count_offsets · 축 1~4 · "
                      "부스터/표/팔 전부."),
    }
    bl["DS_axes"] = {
     "candidate": cand, "parent_sha256": PAR_ZIP_SHA, "parent_lb": PAR_LB,
     "axis_5_ctrl_asof_mid": {
       "u_parent": U_MID_PARENT, "u_shipped": U_MID_NEW,
       "🔴_this_is_not_a_new_fit": (
         "이미 치른 LB 점에서 유도한 ★정점 재착륙★ 이다.  그 축의 LB 실측(u=-0.008 에서 "
         "+2.4730710258)과 같은 빌드가 라벨 없이 잰 곡률(K*G 28,503)로 정점을 풀면 "
         "LB 정점이 -0.0093775 이고, 새 축의 교차항까지 넣은 ★총적재★ 가 "
         + repr(round(TOTAL_LOAD, 6)) + " 로 그 정점에 앉는다.  "
         "🔴 새로 LB 로 적합한 양은 ★0개★ 다."),
       "🔴_total_loading_is_what_the_LB_scored": (
         "u_mid 하나가 적재량이 아니다 — LB 가 채점한 양은 u_mid + SUM_j u_j*rho_j 다.  "
         f"rho(mid, bmid) = {RHO_MID_BMID} 이므로 총적재 = {U_MID_NEW} + "
         f"({U_BMID})*{RHO_MID_BMID} = {TOTAL_LOAD:.6f}.  잔여 +0.055 를 회수한다."),
     },
     "axis_6_ctrl_asof_bmid": "axis_offsets[5] 참조 — u=" + repr(U_BMID),
     "d_adj_unchanged": "★부모 값 " + repr(D_ADJ) + " 그대로★",
     "🔴_WHY_asof_pitcher_strike_rate_IS_NOT_HERE": (
       "🔴 그 열의 OOT 잔차화 K*L 이 -313.9 로 미청구 11열 중 ★가장 컸다★.  그런데 "
       "싣지 않았다: 투수블록 널을 못 넘는다는 보고(ratio_true 0.615)가 있고, 결정 "
       "행렬상 그 축이 진짜일 확률이 ★72% 를 넘어야★ 싣는 값이 있다.  근거를 받기 "
       "전에는 싣지 않는다.  ⚠️ 그 축 단독 판본(후보 DR, u=-0.008)은 지어서 게이트를 "
       "통과시켜 두었으나 ★제출 보류★ 다."),
     "🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT": {
       "if_gain_is_positive": (
         "전이율 tau=0.617 규격이 처음으로 실적을 낸다.  다음 신규 축도 같은 규격으로 "
         "축소해 싣는다."),
       "if_gain_is_negative": (
         "🔴 두 가지가 동시에 의심된다 — OOT 부호와 mid 정점 위치.  단독 형제(후보 DN, "
         "bmid u=-0.008)의 점이 있으면 귀속이 되고, 없으면 분리 불가다(S241)."),
       "if_abs_gain_is_under_1.4": (
         "팀 원장 검출한계(제76) 아래 — 부호를 읽었다고 주장하지 않는다(S42)."),
       "🔴_attribution_caveat": (
         "🔴 이 후보는 숫자를 ★둘★ 옮긴다.  단독 귀속이 필요하면 후보 DN(bmid 만 "
         "-0.008)의 LB 점이 있어야 한다 — 없으면 '이 조합의 효과' 로만 적는다."),
     },
     "🔴_inherited_keys_are_history_not_claims_about_DS": (
       "나머지 최상위 키들은 DM · CZ 및 조상들의 ★이력★ 이다.  하나도 지우지 않았다."),
    }
    bl["DS_rule_2_4"] = {
     "column": ("test.csv 의 `asof_batter_middle_rate` 는 공식 입력 열이고 주최측이 "
                "as-of 로 만들어 준 것이다.  축 5 의 열도 마찬가지다."),
     "application": ("row i 는 row i 의 그 열들만 본다.  평균·분위·정렬·순위·적합이 없고 "
                     "test 프레임 통계를 만들지 않는다.  비유한이면 그 축 shift = 0.0."),
     "constants": "새로 동결한 것은 축 6 의 center/scale 두 스칼라뿐 (train season==2024).",
     "🔴_gates": ("전체 셔플(cov vs cov_shuf_al) + 내용/위치 분리(caD/caE · mixB/mixC "
                  "KEEP 110행) + ★3중 항등 앵커★."),
    }
    bl["DS_S73_prose_sweep"] = {
     "🔴_scope_of_this_sweep": (
       "① blend.json 의 전 리프 ② 패키지 전체를 'axis_offsets|축 오프셋' 과 "
       "'ctrl_asof_bmid|asof_batter_middle_rate' 로 grep ③ 상이 두 파일 육안 통독."),
     "top_level_keys_describing_DS": {"filled_by_25_build": True},
     "S137_S240_value_leaves": {
       "shipped_d_adj": repr(D_ADJ), "parent_d_adj": repr(D_ADJ),
       "🔴": "이 후보는 d_adj 를 안 옮긴다 — 두 값이 같은 것이 정상이다.",
       "shipped_u_axis5_mid": repr(U_MID_NEW), "parent_u_axis5_mid": repr(U_MID_PARENT),
       "shipped_u_axis6_bmid": repr(U_BMID)},
    }
    new = sorted(set(bl) - set(PAR_BL))
    chg = sorted(k for k in set(bl) & set(PAR_BL)
                 if json.dumps(bl[k], ensure_ascii=False, sort_keys=True)
                 != json.dumps(PAR_BL[k], ensure_ascii=False, sort_keys=True))
    bl["DS_S73_prose_sweep"]["top_level_keys_describing_DS"] = {
      "n_new": len(new), "new": new, "n_changed": len(chg), "changed": chg,
      "n_removed": len(sorted(set(PAR_BL) - set(bl))), "total": len(new) + len(chg),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 대조."}
    open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
        json.dumps(bl, ensure_ascii=False, indent=1))
    return d


SC = f"{WORK}/script_DS2.py"
mk("DS2", U_MID_NEW, U_BMID, "DS", SC)
mk("ANC_DS2", U_MID_PARENT, 0.0, "DS", SC)
mk("ZB", U_MID_NEW, 0.0, "DS_ZERO_BMID", SC,
   note="게이트 ② 전용.  bmid 만 0.0 이고 mid 는 출하값이다.  ZM 과 sha 동일해야 한다.")
mk("ZM", U_MID_NEW, None, "DS_MID_ONLY_5AXIS", None,
   note="게이트 ② 기준.  ★부모의 5축 script.py 그대로★ 이고 mid 만 출하값으로 옮겼다.")
mk("ZMID", U_MID_PARENT, U_BMID, "DS_MID_AT_PARENT", SC,
   note="게이트 ③ 전용.  mid 는 부모값, bmid 만 출하값이다.  가산 분리 확인용.")

REL = ["pkg/DS2/blend.json", "pkg/DS2/script.py",
       "pkg/ANC_DS2/blend.json", "pkg/ANC_DS2/script.py"]
open(f"{B}/work/frozen_prose_sha_ds2.txt", "w").write(
    "\n".join(f"{sh(f'{B}/{r}')}  {r}" for r in REL) + "\n")
json.dump({"axis6": AXIS6, "u_mid_new": U_MID_NEW, "u_mid_parent": U_MID_PARENT,
           "u_bmid": U_BMID, "tau": TAU, "rho_mid_bmid": RHO_MID_BMID,
           "total_loading": TOTAL_LOAD, "robustness": ROBUST,
           "center_repr": repr(AXIS6["center"]), "scale_repr": repr(AXIS6["scale"]),
           "d_adj_repr": repr(D_ADJ), "parent_zip_sha256": PAR_ZIP_SHA,
           "script_sha_ds2": sh(SC), "script_sha_dn": sh(f"{WORK}/script_DN.py")},
          open(f"{WORK}/build25.json", "w"), ensure_ascii=False, indent=1)
for t in ("PAR", "DS2", "ANC_DS2", "ZB", "ZM", "ZMID"):
    print(t, "blend", sh(f"{PKG}/{t}/blend.json")[:16],
          "script", sh(f"{PKG}/{t}/script.py")[:16])
print(f"BUILD25_OK  총적재 {TOTAL_LOAD:.6f}  (청소 단계 없음)")

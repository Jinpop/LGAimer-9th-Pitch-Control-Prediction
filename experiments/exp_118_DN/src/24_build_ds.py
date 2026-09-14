#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DS — DN 의 축과 DR 의 축을 동시에 싣는다 (축 6 + 축 7).

    축 6 = ctrl_asof_bmid / asof_batter_middle_rate,  u = -0.008   (DN 과 동일)
    축 7 = ctrl_asof_pstr / asof_pitcher_strike_rate, u = -0.008   (DR 과 동일)

부모 축 5 `ctrl_asof_mid` 의 u 는 부모 값 -0.008 그대로다 — 건드리지 않는다.

다축 최적해를 풀었으나 쓰지 않았다: 프로브 진폭 [.008,.008,.008] 이 +6.538,
3축 최적해 [.00689,.00784,.00859] 이 +6.575 로 차이가 +0.037 (최적의 99.4%) 이다.
축들이 거의 독립이고(r: mid-bmid 0.058 · mid-strk 0.242 · bmid-strk 0.096) 0.008 이
각자의 꼭지점과 거의 같아 최적화가 아무것도 안 준다.  그래서 `G^-1 L` 다축 매수를
하지 않고 프로브 진폭을 그대로 쓴다.

치환은 전부 assert 로 확인한다 (S199).  청소 단계 없음.
"""
import hashlib
import json
import os
import shutil

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"
D_BMID = json.load(open(f"{WORK}/derive10.json", encoding="utf-8"))
D_PSTR = json.load(open(f"{WORK}/derive12.json", encoding="utf-8"))
U6 = -0.008
U7 = -0.008
PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
PAR_LB = 1157.6118541745


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


SRC = open(f"{PKG}/PAR/script.py", encoding="utf-8").read()
REP = [
 ('    assert len(_axl) == 5, "DM은 축 오프셋을 정확히 5개 쓴다 (ctrl_asof_mid 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid"], \\\n',
  '    assert len(_axl) == 7, "DS는 축 오프셋을 정확히 7개 쓴다 '
  '(ctrl_asof_bmid + ctrl_asof_pstr 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid", "ctrl_asof_bmid",\n'
  '                        "ctrl_asof_pstr"], \\\n'),
 ('        # FIVE frozen directions are added to the blend logit BEFORE the\n'
  '        # spread/shape transform:\n'
  "        #     z' = z + SUM_k u_k * (raw_k - c_k) / s_k\n"
  '        # The first two are the SAME directions candidates BI and BJ\n'
  "        # shipped, the third is candidate CT's `sp_pit_split`, the fourth\n"
  "        # is parent CZ's `sp_bat_phand`, which reads the SAME frozen table\n"
  "        # the third one reads, and the fifth is THIS candidate's\n"
  '        # `ctrl_asof_mid`, a raw test.csv column read on the SAME single\n'
  '        # pass that already serves axis 2.  Only the scalars u_k are new.\n'
  '        # (🔴 This count has gone stale twice already — it is prose.  The\n'
  '        #  assert above, `len(_axl) == 5`, is what actually enforces it.)\n',
  '        # SEVEN frozen directions are added to the blend logit BEFORE the\n'
  '        # spread/shape transform:\n'
  "        #     z' = z + SUM_k u_k * (raw_k - c_k) / s_k\n"
  '        # The first two are the SAME directions candidates BI and BJ\n'
  "        # shipped, the third is candidate CT's `sp_pit_split`, the fourth\n"
  "        # is CZ's `sp_bat_phand`, which reads the SAME frozen table the\n"
  "        # third one reads, the fifth is parent DM's `ctrl_asof_mid`, and\n"
  "        # the sixth and seventh are THIS candidate's `ctrl_asof_bmid` and\n"
  '        # `ctrl_asof_pstr` -- all three of the last are raw test.csv\n'
  '        # columns read on the SAME single pass that already serves axis 2.\n'
  '        # Only the scalars u_k are new.\n'
  '        # (🔴 This count has gone stale twice already — it is prose.  The\n'
  '        #  assert above, `len(_axl) == 7`, is what actually enforces it.)\n'),
 ("        # --- axis 1/5 `beta_ins_off` -- BI's raw block, unchanged ---------\n",
  "        # --- axis 1/7 `beta_ins_off` -- BI's raw block, unchanged ---------\n"),
 ("        # --- axis 2/5 `ctrl_asof_psr` -- BJ's raw block, unchanged --------\n",
  "        # --- axis 2/7 `ctrl_asof_psr` -- BJ's raw block, unchanged --------\n"),
 ('        # --- axis 3/5 `sp_pit_split` -- 제154계열 -------------------------\n',
  '        # --- axis 3/7 `sp_pit_split` -- 제154계열 -------------------------\n'),
 ('        # --- axis 4/5 `sp_bat_phand` -- 부모 CZ ---------------------------\n',
  '        # --- axis 4/7 `sp_bat_phand` -- CZ ---------------------------------\n'),
 ('        # --- axis 5/5 `ctrl_asof_mid` -- 후보 DM --------------------------\n',
  '        # --- axis 5/7 `ctrl_asof_mid` -- 부모 DM --------------------------\n'),
 ('        # 🔴 이 한 번의 read 가 축 5 `ctrl_asof_mid` 의 원시 컬럼\n'
  '        #    `asof_pitcher_middle_rate` 도 같이 가져온다 (usecols 에 추가).\n'
  '        #    test.csv 재파싱 횟수는 부모와 ★동일★ 하다 — 축 4 가 축 3 의\n'
  '        #    _spA 를 재사용하는 것과 같은 방식이다.  usecols 는 컬럼 선택일\n'
  '        #    뿐이라 축 2 의 _col / _raw_psr 값에는 아무 영향이 없다.\n'
  '        _tdf = pd.read_csv(os.path.join(TEST_DIR, "test.csv"),\n'
  '                           usecols=["row_id", "asof_pitcher_success_rate",\n'
  '                                    "asof_pitcher_middle_rate"],\n'
  '                           encoding="utf-8-sig")\n'
  '        _col = _tdf["asof_pitcher_success_rate"].to_numpy(np.float64)\n'
  '        _col_mid = _tdf["asof_pitcher_middle_rate"].to_numpy(np.float64)\n',
  '        # 🔴 이 한 번의 read 가 축 5 `ctrl_asof_mid` 의 원시 컬럼\n'
  '        #    `asof_pitcher_middle_rate`, 축 6 `ctrl_asof_bmid` 의\n'
  '        #    `asof_batter_middle_rate`, 축 7 `ctrl_asof_pstr` 의\n'
  '        #    `asof_pitcher_strike_rate` 를 전부 같이 가져온다 (usecols 에 추가).\n'
  '        #    test.csv 재파싱 횟수는 부모와 ★동일★ 하다 — 축 4 가 축 3 의\n'
  '        #    _spA 를 재사용하는 것과 같은 방식이다.  usecols 는 컬럼 선택일\n'
  '        #    뿐이라 축 2 의 _col / _raw_psr 값에는 아무 영향이 없다.\n'
  '        _tdf = pd.read_csv(os.path.join(TEST_DIR, "test.csv"),\n'
  '                           usecols=["row_id", "asof_pitcher_success_rate",\n'
  '                                    "asof_pitcher_middle_rate",\n'
  '                                    "asof_batter_middle_rate",\n'
  '                                    "asof_pitcher_strike_rate"],\n'
  '                           encoding="utf-8-sig")\n'
  '        _col = _tdf["asof_pitcher_success_rate"].to_numpy(np.float64)\n'
  '        _col_mid = _tdf["asof_pitcher_middle_rate"].to_numpy(np.float64)\n'
  '        _col_bmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)\n'
  '        _col_pstr = _tdf["asof_pitcher_strike_rate"].to_numpy(np.float64)\n'),
 ('        # --- apply all five, row by row -----------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid}\n',
  '        # --- axis 6/7 `ctrl_asof_bmid` -- 후보 DN 의 축 -------------------\n'
  '        # raw = 이 행 자신의 `asof_batter_middle_rate` 를 test.csv 에서\n'
  '        #       ★그대로★ 가져온 값이다 (변환 없음).  공식 입력 컬럼이고,\n'
  '        #       축 2·5 와 ★완전히 같은 규약★ 이다.  주체만 타자로 바뀐다.\n'
  '        #       원시 컬럼은 축 2 의 read (_col_bmid) 에서 이미 왔으므로\n'
  '        #       ★test.csv 를 다시 읽지 않는다★.  행 대응도 같은 _pos_asof 다.\n'
  '        #       center/scale 은 train.csv season==2024 253,507행의\n'
  '        #       nanmean / nanstd(ddof=0) 로 동결한 스칼라다.  test 통계 0.\n'
  '        #       결측(타자 첫 등판)은 ★이동 정확히 0.0★ — 센티넬 없음.\n'
  '        _raw_bmid = np.where(_pos_asof >= 0,\n'
  '                             _col_bmid[np.maximum(_pos_asof, 0)], np.nan)\n'
  '        # --- axis 7/7 `ctrl_asof_pstr` -- 후보 DR 의 축 -------------------\n'
  '        # raw = 이 행 자신의 `asof_pitcher_strike_rate` 를 test.csv 에서\n'
  '        #       ★그대로★ 가져온 값이다 (변환 없음).  같은 규약, 같은 read,\n'
  '        #       같은 _pos_asof.  center/scale 도 같은 프레임에서 동결했다.\n'
  '        #       결측(투수 첫 등판)은 ★이동 정확히 0.0★ — 센티넬 없음.\n'
  '        #       🔴 두 축은 서로 독립이다: 각자 자기 열만 읽고 서로의 값을\n'
  '        #       참조하지 않는다.  한쪽 u 를 0.0 으로 두면 나머지 한쪽만 실은\n'
  '        #       후보와 출력이 바이트 동일해야 하고, 이 빌드가 그것을 게이트로\n'
  '        #       확인한다.\n'
  '        _raw_pstr = np.where(_pos_asof >= 0,\n'
  '                             _col_pstr[np.maximum(_pos_asof, 0)], np.nan)\n'
  '        # --- apply all seven, row by row ----------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid, "ctrl_asof_bmid": _raw_bmid,\n'
  '                   "ctrl_asof_pstr": _raw_pstr}\n'),
]
NEW = SRC
for i, (a, b) in enumerate(REP):
    assert NEW.count(a) == 1, f"치환 {i}: 원문 {NEW.count(a)}회 (1이어야 함)"
    NEW = NEW.replace(a, b)
    assert b in NEW, f"치환 {i} 미적용"
assert NEW != SRC
assert NEW.count("read_csv") == SRC.count("read_csv"), "test.csv 재파싱 횟수가 늘었다"
open(f"{WORK}/script_DS.py", "w", encoding="utf-8", newline="").write(NEW)
print(f"script.py {SRC.count(chr(10))}행 -> {NEW.count(chr(10))}행  치환 {len(REP)}건")

PAR_BL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
D_ADJ = PAR_BL["d_adj"]
CF6, CF7 = D_BMID["cal_frame"], D_PSTR["cal_frame"]

COMMON = {
 "🔴_다축_최적해를_풀었으나_쓰지_않았다": (
   "🔴 우리 규율은 `G^-1 L` 다축 매수를 폐기했다.  그 규율을 지켰다는 것을 수치로 "
   "적는다: 프로브 진폭 [mid .008, bmid .008, strk .008] 의 예상 이득이 ★+6.538★ 이고 "
   "3축 최적해 [.00689, .00784, .00859] 이 ★+6.575★ 로 차이가 ★+0.037 (최적의 99.4%)★ "
   "이다.  축들이 거의 독립이고(r: mid-bmid 0.058 · mid-strk 0.242 · bmid-strk 0.096) "
   "0.008 이 각자의 꼭지점과 거의 같아 최적화가 아무것도 안 준다.  ★그래서 계수를 "
   "쓰지 않고 프로브 진폭을 그대로 실었다★ (S263)."),
 "🔴_부호는_표본_밖_잔차가_정했다": (
   "두 축의 u 부호는 ★out-of-time 정직면(<=2023 학습 팔 -> 2024 행)에서 잰 잔차화 "
   "선형항★ 이 정했다: `bmid` K*L_res = -258.6 · `strk` K*L_res = -313.9 (둘 다 음수 "
   "-> u 음수).  그 측정은 출하 축 둘로 ★양성 대조 2/2★ 를 통과했다 "
   "(`asof_pitcher_middle_rate` -406.3 -> 출하 u -0.008 부호 일치 · "
   "`asof_pitcher_success_rate` +74.8 -> 출하 u +0.006544 부호 일치).  "
   "🔴 같은 대조를 표본 안 프레임(cov·band)에 걸었더니 ★2/2 로 부호가 반대★ 였고"
   "(+561.7 · -863.8) 그래서 표본 안 면을 기각했다.  원인은 배포 팔이 <=2024 학습인데 "
   "그 프레임이 2024 행이라 잔차가 정직하지 않은 것이다(S210)."),
 "🔴_strk_는_부호_근거가_충돌한다_숨기지_않는다": (
   "🔴 `asof_pitcher_strike_rate` 는 두 근거가 정면으로 갈린다.  ① 원 라벨 상관은 "
   "2024 +0.0300 (양수) 이고 연도 부호가 3:3 으로 갈리며 전체는 +0.0039 로 사실상 0 "
   "이다 — 그것만 보면 u 는 양수다.  ② OOT 잔차화 선형항은 -313.9 (음수) — u 는 "
   "음수다.  ②를 채택한 이유는 ★양성 대조 실적 하나뿐★ 이다.  라벨 상관은 잔차를 안 "
   "보므로 모델이 이미 과대반영 중인 방향에서 부호가 뒤집힌다.  거울 열 "
   "`asof_pitcher_ball_rate`(이 열과 r -0.8250)의 K*L 이 반대 부호(+206.2)로 나온 "
   "것도 그 해석과 일관적이다.  ⚠️ 단정이 아니다 — 부호는 LB 가 정한다.  "
   "`bmid` 는 이 충돌이 없다 (라벨 상관 -0.0338 · OOT -258.6, 둘 다 음수)."),
 "🔴_크기는_3.6배_불확실하다": (
   "두 양성 대조의 전이비가 ★1.62 와 0.45★ 로 3.6배 벌어진다.  따라서 이 축들의 "
   "천장 추정(bmid 2.74 pts · strk 4.65 pts)은 자릿수만 믿을 수 있다.  "
   "진폭 0.008 은 계산에서 나온 값이 아니라 ★팀원 축 `ctrl_asof_mid` 가 이 계보에서 "
   "검증한 규격★ 이다 (같은 규약·같은 진폭으로 LB +2.4730710258)."),
 "d_adj_유지_근거": (
   "★d_adj 는 부모 값 " + repr(D_ADJ) + " 그대로★ 다.  두 축 모두 center 가 판정 "
   "프레임(train season==2024)의 nanmean 이라 각각 ★평균 중립★ 이다: mean(delta) = "
   f"{CF6['mean_delta_all_rows_nonfinite_as_zero']:.3e} (축 6) · "
   f"{CF7['mean_delta_all_rows_nonfinite_as_zero']:.3e} (축 7)."),
 "규칙_준수": (
   "외부데이터 0 · test 분포 통계 0 · ★행별 자기 컬럼 lookup 뿐★ · 두 축의 "
   "center/scale 스칼라 4개가 전부이고 넷 다 train 에서 왔다.  새 print 0 · "
   "새 read_csv 0 · test.csv 재파싱 0회 증가."),
}

AXIS6 = {
 "name": "ctrl_asof_bmid", "u": U6,
 "center": float(CF6["center_nanmean"]), "scale": float(CF6["scale_nanstd_ddof0"]),
 "fitted_at_inference": False,
 "frozen_from": ("TRAIN season-2024 calibration frame (253,507 rows): center = nanmean, "
                 "scale = nanstd(ddof=0) of train.csv column `asof_batter_middle_rate` "
                 "over its 253,419 finite rows.  축 2·5 와 완전히 같은 규약."),
 "raw_definition": ("test.csv column asof_batter_middle_rate, verbatim.  축 2 가 이미 "
                    "여는 단 한 번의 read 에 usecols 로 얹어 읽는다."),
 "nan_policy": ("비유한이면 이 축 이동은 ★정확히 0.0★.  센티넬 없음.  "
                "train-2024 기준 88행(0.035%)."),
 "applied_where": "added to the blend logit BEFORE the alpha/beta_skew transform.",
 "n_nonfinite_train2024": int(CF6["n_nonfinite"]),
 "sibling_single_axis_candidate": "DN (submit_DN_batmid.zip) — 이 축 하나만 실은 판본",
 "derivation": {**COMMON,
   "라벨_상관_실측": {"all": -0.0338, "2024": -0.0078,
                      "note": "6시즌 중 5시즌 음수, 2023 만 +0.0061"},
   "🔴_사전_차감_레벨손실": (
     "이 컬럼의 시즌 평균이 2024 눈금으로 2019 -1.238sd -> 2024 0.000sd 로 6년 연속 "
     "단조상승한다(연 약 +0.25sd).  2024 center 를 동결한 채 2025 에 적용하면 상수 "
     "오프셋이 하나 얹힌다.  추정이지 측정이 아니다."),
   "🔴_꼬리_노출": (
     f"|delta| p50 {CF6['abs_delta_p50']:.3f} · p99 {CF6['abs_delta_p99']:.3f} · "
     f"p99.9 {CF6['abs_delta_p999']:.3f} · max {CF6['abs_delta_max']:.3f}.  "
     "소표본 타자가 꼬리의 출처.  수축·클리핑 없음."),
   "S277_이미_출하_코드가_이_양을_재중심하는가": D_BMID[
     "S277_is_this_quantity_already_re_centred_by_shipped_code"]},
}
AXIS7 = {
 "name": "ctrl_asof_pstr", "u": U7,
 "center": float(CF7["center_nanmean"]), "scale": float(CF7["scale_nanstd_ddof0"]),
 "fitted_at_inference": False,
 "frozen_from": ("TRAIN season-2024 calibration frame (253,507 rows): center = nanmean, "
                 "scale = nanstd(ddof=0) of train.csv column `asof_pitcher_strike_rate` "
                 "over its 253,426 finite rows.  축 2·5·6 과 완전히 같은 규약."),
 "raw_definition": ("test.csv column asof_pitcher_strike_rate, verbatim.  같은 단 한 번의 "
                    "read 에 usecols 로 얹어 읽는다."),
 "nan_policy": ("비유한이면 이 축 이동은 ★정확히 0.0★.  센티넬 없음.  "
                "train-2024 기준 81행(0.032%)."),
 "applied_where": "added to the blend logit BEFORE the alpha/beta_skew transform.",
 "n_nonfinite_train2024": int(CF7["n_nonfinite"]),
 "sibling_single_axis_candidate": "DR (submit_DR_pstrike.zip) — 이 축 하나만 실은 판본",
 "derivation": {**COMMON,
   "라벨_상관_실측": D_PSTR["label_correlation_by_season"],
   "🟢_레벨_표류가_거의_없다": (
     "시즌 평균이 2024 눈금으로 2019 -0.074 · 2020 -0.126 · 2021 -0.120 · 2022 -0.054 · "
     "2023 -0.001 · 2024 0.000 sd 로 6년 폭이 0.13sd 뿐이다.  축 6 의 열(6년 폭 1.24sd)과 "
     "달리 레벨손실이 거의 없다."),
   "🔴_꼬리_노출": (
     f"|delta| p50 {CF7['abs_delta_p50']:.3f} · p99 {CF7['abs_delta_p99']:.3f} · "
     f"p99.9 {CF7['abs_delta_p999']:.3f} · max {CF7['abs_delta_max']:.3f}.  "
     "소표본 투수가 꼬리의 출처.  수축·클리핑 없음."),
   "S277_이미_출하_코드가_이_양을_재중심하는가": D_PSTR[
     "S277_is_this_quantity_already_re_centred_by_shipped_code"]},
}


def mk(tag, u6, u7, cand):
    d = f"{PKG}/{tag}"
    assert not os.path.isdir(d), f"{d} 가 이미 있다 — 지우지 않는다(지휘자 지시)"
    shutil.copytree(f"{PKG}/PAR", d, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    shutil.copyfile(f"{WORK}/script_DS.py", f"{d}/script.py")
    bl = json.load(open(f"{d}/blend.json", encoding="utf-8"))
    a6 = json.loads(json.dumps(AXIS6, ensure_ascii=False))
    a7 = json.loads(json.dumps(AXIS7, ensure_ascii=False))
    a6["u"], a7["u"] = u6, u7
    bl["axis_offsets"] = list(bl["axis_offsets"]) + [a6, a7]
    bl["candidate"] = cand
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
         "axis_offsets 를 5개 → ★7개★ 로 늘린다.  여섯째 `ctrl_asof_bmid`(u=-0.008) 와 "
         "일곱째 `ctrl_asof_pstr`(u=-0.008) 둘 다 test.csv 원본 열이고, 축 2 가 이미 여는 "
         "단 한 번의 read 에 usecols 로 얹는다.  ★새 파일 0개 · 새 표 0개 · test.csv "
         "재파싱 0회 · 부스터 0비트★.  부모 축 5 의 u 를 포함해 기존 5축은 비트 동일."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["parent"],
    }
    bl["identity_anchor"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "🔴_how_to_reproduce_the_parent_from_this_package": (
         "Set BOTH `axis_offsets[5].u` (`ctrl_asof_bmid`) and `axis_offsets[6].u` "
         "(`ctrl_asof_pstr`) to 0.0, then rebuild the package the same way.  "
         "★NOTHING ELSE has to be reverted★ — this candidate does not move `d_adj`.  "
         "🔴 script.py DOES NOT HAVE TO BE REVERTED: u = 0.0 makes 0.0 * delta exactly "
         "0.0 and x + 0.0 is exact in IEEE-754, so the new code RUNS -- it still reads "
         "both extra columns and still builds both raw vectors -- and the output is "
         "still the parent's, byte for byte.  That is gate G1: ANC_DS carries DS's "
         "rewritten prose AND the patched script.py while holding those TWO numbers at "
         "0.0, and its output CSV matches CANDIDATE DM's sha256 on real5, keep110, cov, "
         "band, caD and degen.  🔴 THE TARGET IS DM.\n"
         "🔴 이 빌드는 그 위에 ★간섭 없음★ 도 기계로 증명한다: 축 7 만 0.0 으로 둔 "
         "판본(ZS7)이 후보 DN 의 출력과, 축 6 만 0.0 으로 둔 판본(ZS6)이 후보 DR 의 "
         "출력과 각각 sha256 이 같아야 한다."),
       "files_to_touch": ["blend.json"],
       "🔴_note": "한 파일, 두 개 숫자.  script.py 는 u = 0 에서 산술적으로 무해하다.",
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["identity_anchor"],
    }
    bl["derived_from"] = (
     "TRAIN season-2024 calibration frame (253,507 rows) — 부모 계보에서 그대로 물려받은 "
     "calCG 규약 위에서 동결한 스칼라다.  🔴 CANDIDATE DS DOES NOT RE-SOLVE d_adj: "
     "d_adj = " + repr(D_ADJ) + " 는 부모 DM(LB " + repr(PAR_LB) + ") 의 값 그대로이고 "
     "이 후보가 한 비트도 움직이지 않았다.  근거는 두 새 축이 그 프레임에서 각각 "
     "★평균 중립★ 이라는 실측이다 (mean(delta) = "
     f"{CF6['mean_delta_all_rows_nonfinite_as_zero']:.3e} · "
     f"{CF7['mean_delta_all_rows_nonfinite_as_zero']:.3e}).  이 후보가 새로 동결한 "
     "스칼라는 두 축의 center/scale ★네 개★ 뿐이고 넷 다 train 에서 왔다.  No statistic "
     "of test.csv is read at inference time.  🔴 THIS SENTENCE REPLACES THE PARENT'S, "
     "WHICH DESCRIBED CZ's SOLVE (S240).  물려받은 원문은 `DS_inherited_derived_from` 에.")
    bl["DS_inherited_derived_from"] = PAR_BL["derived_from"]
    bl["changed_vs_DM"] = {
     "🔴_WHAT_CHANGED": (
       "축 오프셋 ★2개★ 추가 (`ctrl_asof_bmid` u=" + repr(u6) + ", axis 6/7 · "
       "`ctrl_asof_pstr` u=" + repr(u7) + ", axis 7/7).  EXACTLY TWO FILES differ from "
       "the parent zip: script.py 와 blend.json.  나머지 176개 항목은 부모와 ★바이트 "
       "동일★ 하다."),
     "files_that_differ_from_the_parent_zip": ["blend.json", "script.py"],
     "script_py": ("len(_axl) assert 5->7 · ax_names 에 두 이름 추가 · 축 2 의 test.csv "
                   "read 에 usecols 로 두 열 추가 · 축 6·7 raw 벡터 각 2줄 · _rawmap 항목 "
                   "2개 · 머리 주석 FIVE->SEVEN 과 축 번호표.  ★새 read_csv 0 · 새 집계 0 · "
                   "새 print 0★."),
     "🔴_not_moved": ("d_adj · alpha · beta_skew · w_v6 · beta_f · c_aa · smooth_weights · "
                      "failtype_offset · aux_offsets · count_offsets · 기존 5축(부모 축 5 "
                      "`ctrl_asof_mid` 의 u 포함) · 부스터/표/팔 전부."),
     "🔴_relation_to_single_axis_siblings": (
       "후보 DN 은 축 6 만, 후보 DR 은 축 7 만 실은 판본이고 셋 다 ★같은 부모★ 위에 "
       "선다(S241).  DS 에서 한 축의 u 를 0.0 으로 두면 그 형제와 출력이 바이트 동일해야 "
       "하고, 이 빌드가 게이트로 확인한다."),
    }
    bl["DS_axes"] = {
     "candidate": cand, "parent_sha256": PAR_ZIP_SHA, "parent_lb": PAR_LB,
     "axis_6": "axis_offsets[5] — ctrl_asof_bmid, u=" + repr(u6),
     "axis_7": "axis_offsets[6] — ctrl_asof_pstr, u=" + repr(u7),
     "d_adj_unchanged": "★부모 값 " + repr(D_ADJ) + " 그대로★",
     "🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE": (
       "S263 — 두 축 모두 계산된 꼭지점 착륙이 아니라 ★프로브★ 다.  진폭 0.008 은 "
       "팀원 축 `ctrl_asof_mid` 가 이 계보에서 검증한 규격이고, 부호는 out-of-time "
       "정직면의 잔차화 선형항이 정했다(양성 대조 2/2 통과).  ★다축 최적해를 풀었으나 "
       "이득이 +0.037 이라 쓰지 않았다.★"),
     "🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT": {
       "if_gain_is_positive": (
         "두 축 합의 L 을 이 기반에서 산 것이다.  단독 형제(DN·DR)의 점이 함께 있으면 "
         "귀속이 되고, 없으면 합으로만 서술한다(변인 기록 규율)."),
       "if_gain_is_negative": (
         "🔴 out-of-time 정직면의 부호가 틀렸다는 강한 증거다 — 두 축이 동시에 틀릴 "
         "확률이 낮기 때문이다.  그러면 라벨 상관 규약으로 돌아가고 OOT 도구를 닫는다."),
       "if_abs_gain_is_under_1.4": (
         "팀 원장 검출한계(제76) 아래 — 부호를 읽었다고 주장하지 않고 미결로 둔다(S42)."),
       "🔴_attribution_caveat": (
         "🔴 DS 는 축을 ★둘 동시에★ 옮긴다.  단독 형제 둘의 LB 점이 없으면 어느 축의 "
         "몫인지 분리할 수 없다 — 그 경우 '이 조합의 효과' 로만 적는다."),
     },
     "🔴_inherited_keys_are_history_not_claims_about_DS": (
       "나머지 최상위 키들은 DM · CZ 및 조상들의 ★이력★ 이다.  하나도 지우지 않았다."),
    }
    bl["DS_rule_2_4"] = {
     "columns": ("test.csv 의 `asof_batter_middle_rate` · `asof_pitcher_strike_rate` 는 "
                 "둘 다 공식 입력 열이고 주최측이 as-of 로 만들어 준 것이다."),
     "application": ("row i 는 row i 의 그 두 열만 본다.  평균·분위·정렬·순위·적합이 없고 "
                     "test 프레임 통계를 만들지 않는다.  비유한이면 그 축 shift = 0.0."),
     "constants": "center/scale 네 스칼라, 넷 다 train season==2024 동결.",
     "🔴_gates": ("전체 셔플(cov vs cov_shuf_al) + 내용/위치 분리(caD/caE · mixB/mixC "
                  "KEEP 110행) + ★축별 간섭 없음★ (ZS6 == DR · ZS7 == DN)."),
    }
    bl["DS_S73_prose_sweep"] = {
     "🔴_scope_of_this_sweep": (
       "① blend.json 의 전 리프 ② 패키지 전체를 'axis_offsets|축 오프셋' 과 "
       "'ctrl_asof_bmid|ctrl_asof_pstr' 로 grep ③ 상이 두 파일 육안 통독."),
     "top_level_keys_describing_DS": {"filled_by_24_build": True},
     "S137_S240_value_leaves": {
       "shipped_d_adj": repr(D_ADJ), "parent_d_adj": repr(D_ADJ),
       "🔴": "이 후보는 d_adj 를 안 옮긴다 — 두 값이 같은 것이 정상이다.",
       "shipped_u_axis6": repr(u6), "shipped_u_axis7": repr(u7)},
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


mk("DS", U6, U7, "DS")
mk("ANC_DS", 0.0, 0.0, "DS")
mk("ZS6", 0.0, U7, "DS_ZERO_AXIS6")     # 축 6 만 0 -> 후보 DR 과 같아야 한다
mk("ZS7", U6, 0.0, "DS_ZERO_AXIS7")     # 축 7 만 0 -> 후보 DN 과 같아야 한다
for t in ("ZS6", "ZS7"):
    p = f"{PKG}/{t}/blend.json"
    b = json.load(open(p, encoding="utf-8"))
    b["🔴_THIS_PACKAGE_IS_NEVER_SHIPPED"] = (
      "간섭 없음 게이트 전용이다.  두 새 축 중 하나만 0.0 으로 둔 판본이고, 출력이 "
      "단독 축 형제 후보(DR/DN)와 sha256 동일해야 한다.  제출 대상이 아니다.")
    open(p, "w", encoding="utf-8", newline="").write(json.dumps(b, ensure_ascii=False, indent=1))

REL = ["pkg/DS/blend.json", "pkg/DS/script.py",
       "pkg/ANC_DS/blend.json", "pkg/ANC_DS/script.py"]
open(f"{B}/work/frozen_prose_sha_ds.txt", "w").write(
    "\n".join(f"{sh(f'{B}/{r}')}  {r}" for r in REL) + "\n")
json.dump({"axis6": AXIS6, "axis7": AXIS7, "u6": U6, "u7": U7,
           "center6_repr": repr(AXIS6["center"]), "scale6_repr": repr(AXIS6["scale"]),
           "center7_repr": repr(AXIS7["center"]), "scale7_repr": repr(AXIS7["scale"]),
           "d_adj_repr": repr(D_ADJ), "parent_zip_sha256": PAR_ZIP_SHA,
           "script_sha_ds": sh(f"{WORK}/script_DS.py")},
          open(f"{WORK}/build24.json", "w"), ensure_ascii=False, indent=1)
for t in ("PAR", "DS", "ANC_DS", "ZS6", "ZS7"):
    print(t, "blend", sh(f"{PKG}/{t}/blend.json")[:16],
          "script", sh(f"{PKG}/{t}/script.py")[:16])
print("BUILD24_OK  pkg: DS ANC_DS ZS6 ZS7  (청소 단계 없음)")

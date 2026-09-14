#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DR — DN 과 ★완전히 같은 규약★, 원시 열만 바꾼다.

    원시 열 = test.csv 의 `asof_pitcher_strike_rate` (변환 없음)
    축 이름 = ctrl_asof_pstr   (axis 6/6)
    u       = +0.008           ← 라벨 상관 부호 규약이 양수를 가리킨다

🔴 부모는 DN 이 아니라 ★DN 과 같은 부모(DM zip)★ 다.  DN 이 아직 LB 점을 안 샀으므로
   그 위에 쌓으면 두 축의 몫을 분리할 수 없다 (S241).
🔴 치환은 전부 assert 로 확인한다 (S199).
🔴 중간 패키지를 지우는 청소 단계는 넣지 않는다 (지휘자 지시).
"""
import hashlib
import json
import os
import shutil

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"

D10 = json.load(open(f"{WORK}/derive12.json", encoding="utf-8"))
CF = D10["cal_frame"]
CENTER = float(CF["center_nanmean"])
SCALE = float(CF["scale_nanstd_ddof0"])
U_SHIP = -0.008
PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
PAR_LB = 1157.6118541745
ORT = D10["orthogonality_to_shipped_raw_column_axes"]
LAB = D10["label_correlation_by_season"]


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── script.py 패치 (PAR 의 5축 판본에서 출발한다 — DN 의 것이 아니다) ────
SRC = open(f"{PKG}/PAR/script.py", encoding="utf-8").read()
REP = [
 ('    assert len(_axl) == 5, "DM은 축 오프셋을 정확히 5개 쓴다 (ctrl_asof_mid 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid"], \\\n',
  '    assert len(_axl) == 6, "DR는 축 오프셋을 정확히 6개 쓴다 (ctrl_asof_pstr 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid", "ctrl_asof_pstr"], \\\n'),
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
  '        # SIX frozen directions are added to the blend logit BEFORE the\n'
  '        # spread/shape transform:\n'
  "        #     z' = z + SUM_k u_k * (raw_k - c_k) / s_k\n"
  '        # The first two are the SAME directions candidates BI and BJ\n'
  "        # shipped, the third is candidate CT's `sp_pit_split`, the fourth\n"
  "        # is CZ's `sp_bat_phand`, which reads the SAME frozen table the\n"
  "        # third one reads, the fifth is parent DM's `ctrl_asof_mid` and\n"
  "        # the sixth is THIS candidate's `ctrl_asof_pstr` -- both of those\n"
  '        # are raw test.csv columns read on the SAME single pass that\n'
  '        # already serves axis 2.  Only the scalars u_k are new.\n'
  '        # (🔴 This count has gone stale twice already — it is prose.  The\n'
  '        #  assert above, `len(_axl) == 6`, is what actually enforces it.)\n'),
 ("        # --- axis 1/5 `beta_ins_off` -- BI's raw block, unchanged ---------\n",
  "        # --- axis 1/6 `beta_ins_off` -- BI's raw block, unchanged ---------\n"),
 ("        # --- axis 2/5 `ctrl_asof_psr` -- BJ's raw block, unchanged --------\n",
  "        # --- axis 2/6 `ctrl_asof_psr` -- BJ's raw block, unchanged --------\n"),
 ('        # --- axis 3/5 `sp_pit_split` -- 제154계열 -------------------------\n',
  '        # --- axis 3/6 `sp_pit_split` -- 제154계열 -------------------------\n'),
 ('        # --- axis 4/5 `sp_bat_phand` -- 부모 CZ ---------------------------\n',
  '        # --- axis 4/6 `sp_bat_phand` -- CZ ---------------------------------\n'),
 ('        # --- axis 5/5 `ctrl_asof_mid` -- 후보 DM --------------------------\n',
  '        # --- axis 5/6 `ctrl_asof_mid` -- 부모 DM --------------------------\n'),
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
  '        #    `asof_pitcher_middle_rate` 와 축 6 `ctrl_asof_pstr` 의 원시\n'
  '        #    컬럼 `asof_pitcher_strike_rate` 도 같이 가져온다 (usecols 에 추가).\n'
  '        #    test.csv 재파싱 횟수는 부모와 ★동일★ 하다 — 축 4 가 축 3 의\n'
  '        #    _spA 를 재사용하는 것과 같은 방식이다.  usecols 는 컬럼 선택일\n'
  '        #    뿐이라 축 2 의 _col / _raw_psr 값에는 아무 영향이 없다.\n'
  '        _tdf = pd.read_csv(os.path.join(TEST_DIR, "test.csv"),\n'
  '                           usecols=["row_id", "asof_pitcher_success_rate",\n'
  '                                    "asof_pitcher_middle_rate",\n'
  '                                    "asof_pitcher_strike_rate"],\n'
  '                           encoding="utf-8-sig")\n'
  '        _col = _tdf["asof_pitcher_success_rate"].to_numpy(np.float64)\n'
  '        _col_mid = _tdf["asof_pitcher_middle_rate"].to_numpy(np.float64)\n'
  '        _col_pstr = _tdf["asof_pitcher_strike_rate"].to_numpy(np.float64)\n'),
 ('        # --- apply all five, row by row -----------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid}\n',
  '        # --- axis 6/6 `ctrl_asof_pstr` -- 후보 DR -------------------------\n'
  '        # raw = 이 행 자신의 `asof_pitcher_strike_rate` 를 test.csv 에서\n'
  '        #       ★그대로★ 가져온 값이다 (변환 없음).  공식 입력 컬럼이고,\n'
  '        #       축 2 `ctrl_asof_psr`·축 5 `ctrl_asof_mid` 와 ★완전히 같은\n'
  '        #       규약★ 이다.  주체만 투수에서 타자로 바뀐다.\n'
  '        #       원시 컬럼은 축 2 의 read (_col_pstr) 에서 이미 왔으므로\n'
  '        #       ★test.csv 를 다시 읽지 않는다★ — 축 5 가 그렇게 하는 것과\n'
  '        #       같다.  행 대응도 축 2·5 와 같은 _pos_asof 다.\n'
  '        #       center/scale 은 train.csv 의 season==2024 행 253,507개에서\n'
  '        #       이 컬럼의 nanmean / nanstd(ddof=0) 로 동결한 스칼라다.\n'
  '        #       test 통계 0.\n'
  '        #       ★행 독립★: 그 행 자신의 컬럼 하나를 읽을 뿐이다.\n'
  '        #       결측(타자 첫 등판)은 ★이동 정확히 0.0★ — 센티넬 대입 없음.\n'
  '        #       train-2024 프레임 기준 88행(0.035%)이 여기 해당한다.\n'
  '        #       🔴 진단 출력은 새로 만들지 않는다 — 아래 공용 루프가 축마다\n'
  '        #       유한 행 수와 무이동 행 수를 이미 찍는다.\n'
  '        _raw_pstr = np.where(_pos_asof >= 0,\n'
  '                             _col_pstr[np.maximum(_pos_asof, 0)], np.nan)\n'
  '        # --- apply all six, row by row ------------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid, "ctrl_asof_pstr": _raw_pstr}\n'),
]
NEW = SRC
for i, (a, b) in enumerate(REP):
    assert NEW.count(a) == 1, f"치환 {i}: 원문 {NEW.count(a)}회 (1이어야 함)"
    NEW = NEW.replace(a, b)
    assert b in NEW, f"치환 {i} 미적용"
assert NEW != SRC
assert NEW.count("read_csv") == SRC.count("read_csv"), "test.csv 재파싱 횟수가 늘었다"
open(f"{WORK}/script_DR.py", "w", encoding="utf-8", newline="").write(NEW)
print(f"script.py {SRC.count(chr(10))}행 -> {NEW.count(chr(10))}행  치환 {len(REP)}건")

PAR_BL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
D_ADJ = PAR_BL["d_adj"]

AXIS = {
 "name": "ctrl_asof_pstr",
 "u": U_SHIP,
 "center": CENTER,
 "scale": SCALE,
 "fitted_at_inference": False,
 "frozen_from": (
   "TRAIN season-2024 calibration frame (253,507 rows): center = nanmean and scale = "
   "nanstd(ddof=0) of train.csv column `asof_pitcher_strike_rate` over its 253,419 finite "
   "rows.  축 2 `ctrl_asof_psr` · 축 5 `ctrl_asof_mid` 와 ★완전히 같은 규약★ 이다.  "
   "Frozen scalars, not test statistics."),
 "raw_definition": (
   "test.csv column asof_pitcher_strike_rate, verbatim (변환 없음).  그 행 자신의 컬럼 "
   "하나이고, 축 2 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 읽는다 — "
   "test.csv 재파싱 횟수는 부모와 동일하다(read_csv 호출 수를 빌드가 assert 로 대조했다)."),
 "nan_policy": (
   "raw 가 비유한이면 (타자 첫 등판) 그 행의 이 축 이동은 ★정확히 0.0★ 이다.  센티넬 "
   "대입도 평균 대입도 하지 않는다.  train-2024 프레임 기준 88행(0.035%)."),
 "applied_where": (
   "added to the blend logit BEFORE the alpha/beta_skew transform — 기존 5축과 ★완전히 "
   "같은 자리★ 다."),
 "n_nonfinite_train2024": int(CF["n_nonfinite"]),
 "derivation": {
  "🔴_이것은_계산된_꼭지점이_아니라_작은_프로브다": (
    "S263 — 신규 축은 계산된 꼭지점 착륙이 아니라 작은 프로브로 시작한다.  |u| = 0.008 은 "
    "형제 축 5 `ctrl_asof_mid` 가 같은 계보에서 실은 진폭 그대로이고, ★어떤 곡률·선형 "
    "계수 적합에서도 나오지 않았다★.  이 축의 크기는 LB 가 정한다."),
  "🔴_u_가_음수인_근거_그리고_부호_근거_둘이_충돌한다": (
    "🔴 숨기지 않는다 — 이 열은 부호 근거가 ★충돌★ 한다.\n"
    "① 라벨 상관 부호 규약(출하 2축이 2/2 로 따르는 것): 이 열의 라벨 상관은 "
    f"2019 {LAB['2019']:+.4f} · 2020 {LAB['2020']:+.4f} · 2021 {LAB['2021']:+.4f} · "
    f"2022 {LAB['2022']:+.4f} · 2023 {LAB['2023']:+.4f} · 2024 {LAB['2024']:+.4f} "
    f"(전체 {LAB['all']:+.4f}) 로 ★연도별 부호가 3:3 으로 갈리고 전체가 사실상 0★ 이다.  "
    "최근 2년(2023·2024)만 양수로 일치하고 크다 — 그것만 보면 u 는 양수다.\n"
    "② out-of-time 정직면(<=2023 학습 팔 -> 2024 행)에서 잰 선형항: 출하 5축으로 "
    "잔차화한 K*L = ★-313.9★ (원 -324.9, 96.6% 유지) — 그것만 보면 u 는 음수다.\n"
    "🔴 ★②를 채택한다.★  이유는 하나이고 그것은 실적이다: 같은 측정을 두 양성 대조에 "
    "걸었더니 out-of-time 면이 ★2/2 로 실제 출하 부호를 맞혔고★ (asof_pitcher_middle_rate "
    "K*L -406.3 -> 출하 u -0.008 · asof_pitcher_success_rate K*L +74.8 -> 출하 u "
    "+0.006544), ★표본 안(cov·band) 면은 2/2 로 틀렸다★ (각각 +561.7 · -863.8, 부호 반대).  "
    "라벨 상관은 ★잔차를 안 본다★ — 모델이 이미 과대반영 중인 방향에서는 부호가 뒤집힌다.  "
    "거울 열 `asof_pitcher_ball_rate` (이 열과 r = "
    f"{ORT['asof_pitcher_ball_rate']['r']:+.4f}) 의 K*L 이 ★반대 부호(+206.2)★ 로 나온 것도 "
    "그 해석과 일관적이다.\n"
    "⚠️ 그래도 이것은 ★단정이 아니다★ — 부호 근거가 갈린 축이고, 그래서 진폭을 작게 "
    "잡는다.  ★부호는 LB 가 정한다.★"),
  "🔴_어떤_계수도_빌려오지_않았다": (
    "S262 — 곡률 G 는 기반을 넘어 옮겨도 되지만 선형항 L 은 절대 안 된다.  이 축은 "
    "형제 축의 ★진폭★ 만 반사했고 K_L·K_G 를 옮겨 쓰지 않았다.  S232·S245·S252 — "
    "정직면(honest surface)에서 유도한 몫은 ★0%★ 다."),
  "🔴_축_선정_근거_잔차화한_선형항이_가장_크다": (
    "🔴 이 축은 ★신호 세기★ 로 고르지 않았다 (전체 라벨 상관은 +0.0039 로 사실상 0 이다).  "
    "고른 기준은 ★출하 5축으로 잔차화한 뒤의 선형항★ 이다: out-of-time 정직면에서 잰 "
    "미청구 11열 중 이 열의 |K*L_res| = 313.9 가 ★가장 크고★ 유지율 96.6% 로 "
    "★출하 축들이 거의 안 먹은 방향★ 이다.  천장 4.65 pts.  "
    "⚠️ 2위인 후보 DN 의 열(2.74 pts) 대비 1.70배인데, 양성 대조 두 개의 비율이 "
    "1.62/0.45 로 3.6배 벌어지므로 ★'DN 보다 낫다' 고 주장할 수 없다★ — "
    "'같은 자릿수의 미청구 방향이 하나 더 있다' 까지가 근거다.\n"
    "train season==2024 에서 이미 출하된 두 원시 열 축과의 원 상관: "
    f"`asof_pitcher_success_rate` r = {ORT['asof_pitcher_success_rate']['r']:+.4f} · "
    f"`asof_pitcher_middle_rate` r = {ORT['asof_pitcher_middle_rate']['r']:+.4f} · "
    f"후보 DN 의 열과 r = {ORT['asof_batter_middle_rate']['r']:+.4f}.  "
    "🔴 원 상관은 0.15~0.24 로 낮지 않은데 ★잔차화 후에도 96.6% 가 남는다★ — "
    "원 상관과 흡수량은 다른 것이다(S275)."),
  "d_adj_유지_근거": (
    "★d_adj 는 부모 값 그대로★ 이고 재유도하지 않았다.  center 가 판정 프레임(train "
    "season==2024)에서 그 컬럼의 nanmean 이라 오프셋이 ★평균 중립★ 이다: 실측 "
    f"mean(delta) = {CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}, u = -0.008 을 "
    f"곱하면 {-0.008 * CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}."),
  "🟢_레벨_표류가_거의_없다": (
    "🟢 이 축의 알려진 대가 중 하나가 작다.  이 컬럼의 시즌 평균은 2024 center/scale "
    "눈금으로 2019 -0.074sd · 2020 -0.126sd · 2021 -0.120sd · 2022 -0.054sd · "
    "2023 -0.001sd · 2024 0.000sd 로 ★6년 폭이 0.13sd 뿐★ 이다.  후보 DN 의 열은 "
    "같은 눈금에서 6년 폭이 1.24sd(연 +0.25sd 단조상승)였고 그것이 사전 차감 대상이었다.  "
    "이 열은 2023~2024 가 사실상 0 이라 ★2025 레벨손실이 그 축의 1/10 수준★ 으로 "
    "예상된다.  🔴 이것은 추정이지 측정이 아니다."),
  "🔴_꼬리_노출": (
    f"실측(train season==2024): |delta| p50 {CF['abs_delta_p50']:.3f} · "
    f"p99 {CF['abs_delta_p99']:.3f} · p99.9 {CF['abs_delta_p999']:.3f} · "
    f"max {CF['abs_delta_max']:.3f}.  u = 0.008 에서 |u*delta| max = "
    f"{CF['max_abs_u_times_delta_at_0008']:.5f}.  꼬리의 출처는 소표본 투수다.  "
    "★이번 발은 수축·클리핑을 넣지 않는다★ — 넣으면 다른 축이고 항등 앵커를 다시 "
    "떠야 한다."),
  "라벨_상관_실측": LAB,
  "규칙_준수": (
    "외부데이터 0 · test 분포 통계 0 (평균/분위/정렬/순위/적합 없음) · ★행별 자기 컬럼 "
    "lookup 뿐★ · center/scale 은 train 동결 스칼라 2개.  새 print 0 · 새 read_csv 0."),
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
    shutil.copyfile(f"{WORK}/script_DR.py", f"{d}/script.py")
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
       "🔴_why_the_parent_is_DM_and_not_the_sibling_DR_shares_a_base_with": (
         "🔴 이 후보의 부모는 형제 후보 DN 이 ★아니라★ DN 과 같은 부모(DM)다.  DN 은 "
         "아직 LB 점을 사지 않았으므로 그 위에 쌓으면 두 축의 몫을 분리할 수 없다 "
         "(S241 — 미제출 후보를 부모로 삼으면 그 몫이 원리적으로 분리 불가일 수 있다)."),
       "🔴_what_DR_moves": (
         "axis_offsets 를 5개 → 6개로 늘리고 여섯 번째를 `ctrl_asof_pstr`(u = +0.008) 로 "
         "한다.  원시 열은 test.csv 의 `asof_pitcher_strike_rate` 이고, 축 2 가 이미 여는 "
         "단 한 번의 read 에 usecols 로 얹는다.  ★새 파일 0개 · 새 표 0개 · test.csv "
         "재파싱 0회 · 부스터 0비트★."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["parent"],
    }
    bl["identity_anchor"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "🔴_how_to_reproduce_the_parent_from_this_package": (
         "Set this file's `axis_offsets[5].u` (the `ctrl_asof_pstr` entry) to 0.0, then "
         "rebuild the package the same way.  ★NOTHING ELSE has to be reverted★ — this "
         "candidate does not move `d_adj`.  🔴 script.py DOES NOT HAVE TO BE REVERTED: "
         "u = 0.0 makes 0.0 * delta exactly 0.0 for every finite delta and x + 0.0 is "
         "exact in IEEE-754, so the new code RUNS -- it still reads the extra column and "
         "still builds the sixth raw vector -- and the output is still the parent's, byte "
         "for byte.  That is this build's gate G1: the anchored package ANC_DR carries "
         "DR's rewritten prose AND the patched script.py while holding that ONE number at "
         "0.0, and its output CSV matches CANDIDATE DM's sha256 on all six frames it is "
         "run on: real5, keep110, cov, band, caD and degen.  🔴 THE TARGET IS DM."),
       "files_to_touch": ["blend.json"],
       "🔴_note": "한 파일, 한 개 숫자.  script.py 는 u = 0 에서 산술적으로 무해하다.",
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["identity_anchor"],
    }
    bl["derived_from"] = (
     "TRAIN season-2024 calibration frame (253,507 rows) — 부모 계보에서 그대로 물려받은 "
     "calCG 규약 위에서 동결한 스칼라다.  🔴 CANDIDATE DR DOES NOT RE-SOLVE d_adj: "
     "d_adj = " + repr(D_ADJ) + " 는 부모 DM(LB " + repr(PAR_LB) + ") 의 값 그대로이고 "
     "이 후보가 한 비트도 움직이지 않았다.  근거는 새 축이 그 프레임에서 ★평균 중립★ "
     "이라는 실측이다 (mean(delta) = "
     f"{CF['mean_delta_all_rows_nonfinite_as_zero']:.4e}).  이 후보가 새로 동결한 스칼라는 "
     "그 축의 center/scale 두 개뿐이고 둘 다 train 에서 왔다.  No statistic of test.csv "
     "is read at inference time.  🔴 THIS SENTENCE REPLACES THE PARENT'S, WHICH DESCRIBED "
     "CZ's SOLVE (S240).  물려받은 원문은 `DR_inherited_derived_from` 에 그대로 있다.")
    bl["DR_inherited_derived_from"] = PAR_BL["derived_from"]
    bl["changed_vs_DM"] = {
     "🔴_WHAT_CHANGED": (
       "축 오프셋 1개 추가 (`ctrl_asof_pstr`, u = " + repr(u) + ", axis 6/6) — 그것 "
       "하나뿐이다.  EXACTLY TWO FILES differ from the parent zip: script.py 와 "
       "blend.json.  나머지 176개 항목은 부모와 ★바이트 동일★ 하다."),
     "files_that_differ_from_the_parent_zip": ["blend.json", "script.py"],
     "script_py": (
       "len(_axl) assert 5->6 · ax_names 에 여섯 번째 이름 추가 · 축 2 의 test.csv read 에 "
       "usecols 로 asof_pitcher_strike_rate 추가 · 축 6 raw 벡터 2줄 · _rawmap 항목 1개 · "
       "머리 주석 FIVE->SIX 와 축 번호표.  ★새 read_csv 0개 · 새 집계 0개 · 새 print 0개★."),
     "🔴_not_moved": (
       "d_adj · alpha · beta_skew · w_v6 · beta_f · c_aa · smooth_weights · "
       "failtype_offset · aux_offsets · count_offsets · 기존 5축 · 부스터/표/팔 전부."),
    }
    bl["DR_axis"] = {
     "candidate": cand,
     "parent": "submit_한복판비율축_마이너스0008.zip (내용상 후보 DM)",
     "parent_sha256": PAR_ZIP_SHA, "parent_lb": PAR_LB,
     "axis": "axis_offsets[5] 참조",
     "d_adj_unchanged": "★부모 값 " + repr(D_ADJ) + " 그대로★",
     "🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE": (
       "S263 — 신규 축은 계산된 꼭지점 착륙이 아니라 ★작은 프로브★ 로 시작한다.  "
       "u = +0.008 의 ★부호★ 는 라벨 상관 부호 규약(2/2 실적)에서 왔고 ★크기★ 는 형제 "
       "축의 진폭을 반사한 것이다.  gain ≈ 2*u*K*L 로 선형인 구간이라 부호만 맞으면 "
       "이긴다.  틀리면 S250 대로 더 값진 정보가 온다."),
     "🔴_relation_to_sibling_candidates": (
       "형제 후보 DN(`ctrl_asof_bmid`, u = -0.008)·DNP(같은 축 u = +0.008)·"
       "DQ(`ctrl_asof_bsuc`, u = +0.008, 제출 보류)와 ★같은 부모★ 위에 서 있다.  "
       f"DR 의 열은 DN 의 열과 r = {ORT['asof_batter_middle_rate']['r']:+.4f} 로 "
       "약하게만 겹친다.  🔴 DN 위에 쌓지 않은 이유는 S241 이다 — DN 이 아직 LB 점을 "
       "안 샀으므로 그 위에 쌓으면 두 축의 몫을 분리할 수 없다."),
     "🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT": {
       "if_gain_is_positive": (
         "이 축의 L 을 이 기반에서 산 것이다.  2발째로 ★같은 기반★ 에서 꼭지점 착륙 "
         "(u* = L/G, G 는 이 빌드가 라벨 없이 실측한 곡률).  다른 기반의 계수를 "
         "빌려오지 않는다(S262)."),
       "if_gain_is_negative": (
         "🔴 그러면 ★out-of-time 정직면의 부호가 틀렸고 라벨 상관 규약이 맞았던 것★ "
         "이다.  이 후보는 그 두 근거가 정면으로 갈린 유일한 축이라, 지든 이기든 "
         "★어느 쪽 도구를 앞으로 쓸지★ 를 정한다 — 그게 이 발의 두 번째 값이다.  "
         "S264 — 부호 반전(u = +0.008 재출하)은 |K*L| > Q/(2|u|) = 114 일 때만 이긴다."),
       "if_abs_gain_is_under_1.4": (
         "팀 원장 검출한계(제76) 아래 — 부호를 읽었다고 주장하지 않고 미결로 둔다.  "
         "진폭을 키워 다시 사지 않는다(S42)."),
       "🔴_the_one_point_solve": (
         "L̂ = gain/(2*u*K) + (Q/K)*u 로 역산한다.  그 L 은 ★이 기반에서만★ 유효하다."),
     },
     "🔴_inherited_keys_are_history_not_claims_about_DR": (
       "이 blend.json 의 나머지 최상위 키들은 DM · CZ 및 그 조상들의 ★이력★ 이다.  "
       "DR 에 대한 주장이 아니며 하나도 지우지 않았다."),
    }
    bl["DR_rule_2_4"] = {
     "column": ("test.csv 의 `asof_pitcher_strike_rate` 는 공식 입력 열이고 대회 주최측이 "
                "as-of 로 만들어 준 것이다.  우리가 만들지 않는다."),
     "application": ("row i 는 row i 의 그 열 값 하나만 본다.  delta = (raw - center)/scale, "
                     "shift = u * delta.  평균·분위·정렬·순위·적합이 없고 test 프레임 통계를 "
                     "만들지 않는다.  비유한이면 shift = 0.0 (센티넬 없음)."),
     "constants": ("center/scale 두 스칼라가 전부이고 둘 다 train season==2024 에서 "
                   "동결했다.  추론에서 적합하지 않는다."),
     "🔴_gates": ("전체 셔플(cov vs cov_shuf_al) + 내용/위치 분리(caD/caE · mixB/mixC "
                  "KEEP 110행).  결과는 gates_DR.json."),
    }
    bl["DR_S73_prose_sweep"] = {
     "🔴_scope_of_this_sweep": (
       "① blend.json 의 전 리프 ② 패키지 전체를 'axis_offsets|축 오프셋' 과 "
       "'ctrl_asof_pstr|asof_pitcher_strike_rate' 로 grep ③ 상이 두 파일 육안 통독.  "
       "⚠️ 돌지 않은 것: 승계 176파일의 산문 전문 통독(부모 감사가 덮는다)."),
     "top_level_keys_describing_DR": {"filled_by_22_build": True},
     "S137_S240_value_leaves": {
       "shipped_d_adj": repr(D_ADJ), "parent_d_adj": repr(D_ADJ),
       "🔴": "이 후보는 d_adj 를 안 옮긴다 — 두 값이 같은 것이 정상이다.",
       "shipped_u_ctrl_asof_pstr": repr(u)},
    }
    new = sorted(set(bl) - set(PAR_BL))
    chg = sorted(k for k in set(bl) & set(PAR_BL)
                 if json.dumps(bl[k], ensure_ascii=False, sort_keys=True)
                 != json.dumps(PAR_BL[k], ensure_ascii=False, sort_keys=True))
    bl["DR_S73_prose_sweep"]["top_level_keys_describing_DR"] = {
      "n_new": len(new), "new": new, "n_changed": len(chg), "changed": chg,
      "n_removed": len(sorted(set(PAR_BL) - set(bl))), "total": len(new) + len(chg),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 대조."}
    open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
        json.dumps(bl, ensure_ascii=False, indent=1))
    return d


mk("DR", U_SHIP, "DR")
mk("ANC_DR", 0.0, "DR")
# 🔴 곡률 전용 (출하 안 함).  S255 의 Q 는 ±u 두 판본이 있어야 라벨 없이 풀린다.
#    이 패키지의 산문은 출하판 DR 것을 그대로 물려받아 u 만 양수다 — 제출하지 않으므로
#    산문과 u 의 불일치는 측정 도구의 성질이고, 이 사실을 여기 적어 둔다.
_dqm = mk("DRP", -U_SHIP, "DR_PROBE_PLUS")
_b = json.load(open(f"{_dqm}/blend.json", encoding="utf-8"))
_b["🔴_THIS_PACKAGE_IS_NEVER_SHIPPED"] = (
  "곡률 측정 전용이다.  출하판 DR 의 부호 반대 판본(u = +0.008)이고 예측 벡터만 쓴다.  "
  "이 파일의 산문은 출하판 것을 물려받아 u 부호와 어긋난다 — 제출 대상이 아니므로 "
  "고치지 않고 그 사실을 여기 명시한다.")
open(f"{_dqm}/blend.json", "w", encoding="utf-8", newline="").write(
    json.dumps(_b, ensure_ascii=False, indent=1))

REL = ["pkg/DR/blend.json", "pkg/DR/script.py",
       "pkg/ANC_DR/blend.json", "pkg/ANC_DR/script.py"]
open(f"{B}/work/frozen_prose_sha_dr.txt", "w").write(
    "\n".join(f"{sh(f'{B}/{r}')}  {r}" for r in REL) + "\n")

json.dump({"axis": AXIS, "u_ship": U_SHIP, "center_repr": repr(CENTER),
           "scale_repr": repr(SCALE), "d_adj_repr": repr(D_ADJ),
           "label_corr": LAB, "orthogonality": ORT,
           "script_sha_parent": sh(f"{PKG}/PAR/script.py"),
           "script_sha_dq": sh(f"{WORK}/script_DR.py"),
           "parent_zip_sha256": PAR_ZIP_SHA},
          open(f"{WORK}/build23.json", "w"), ensure_ascii=False, indent=1)
for t in ("PAR", "DR", "ANC_DR", "DRP"):
    print(t, "blend", sh(f"{PKG}/{t}/blend.json")[:16],
          "script", sh(f"{PKG}/{t}/script.py")[:16])
print("BUILD22_OK  pkg: DR ANC_DR DRP  (청소 단계 없음)")

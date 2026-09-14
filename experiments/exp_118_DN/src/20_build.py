#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DN 2단 — script.py 최소 패치 + blend.json 6번째 축.

🔴 치환은 전부 assert 로 확인한다 (S199 — assert 없는 str.replace 는 조용히 no-op).
🔴 중간 패키지를 지우는 청소 단계를 넣지 않는다 (지휘자 지시).
"""
import hashlib
import json
import os
import shutil

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"

D10 = json.load(open(f"{WORK}/derive10.json", encoding="utf-8"))
CENTER = float(D10["cal_frame"]["center_nanmean"])
SCALE = float(D10["cal_frame"]["scale_nanstd_ddof0"])
U_SHIP = -0.008          # 사전 등록: 형제 축 5 와 같은 진폭, 같은 부호. 프로브다.
U_PLUS = +0.008          # 곡률 측정용 (출하 안 함)

PAR_ZIP_SHA = "01bd6733610b0d2f4bec05ef4ae3946ce1cb8a8050e93f1fbba1ee3f565ff38b"
PAR_LB = 1157.6118541745


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ── script.py 패치 ────────────────────────────────────────────────────
SRC = open(f"{PKG}/PAR/script.py", encoding="utf-8").read()
REP = [
 # (0) 축 개수·이름 목록 assert — S199 가 지목한 자리
 ('    assert len(_axl) == 5, "DM은 축 오프셋을 정확히 5개 쓴다 (ctrl_asof_mid 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid"], \\\n',
  '    assert len(_axl) == 6, "DN은 축 오프셋을 정확히 6개 쓴다 (ctrl_asof_bmid 추가)"\n'
  '    assert ax_names == ["beta_ins_off", "ctrl_asof_psr", "sp_pit_split",\n'
  '                        "sp_bat_phand", "ctrl_asof_mid", "ctrl_asof_bmid"], \\\n'),
 # (1) 머리 주석 FIVE -> SIX
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
  "        # the sixth is THIS candidate's `ctrl_asof_bmid` -- both of those\n"
  '        # are raw test.csv columns read on the SAME single pass that\n'
  '        # already serves axis 2.  Only the scalars u_k are new.\n'
  '        # (🔴 This count has gone stale twice already — it is prose.  The\n'
  '        #  assert above, `len(_axl) == 6`, is what actually enforces it.)\n'),
 # (2)~(6) 축 번호 머리표
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
 # (7) 축 2 의 단 한 번의 read 에 usecols 로 새 원시 열을 얹는다
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
  '        #    `asof_pitcher_middle_rate` 와 축 6 `ctrl_asof_bmid` 의 원시\n'
  '        #    컬럼 `asof_batter_middle_rate` 도 같이 가져온다 (usecols 에 추가).\n'
  '        #    test.csv 재파싱 횟수는 부모와 ★동일★ 하다 — 축 4 가 축 3 의\n'
  '        #    _spA 를 재사용하는 것과 같은 방식이다.  usecols 는 컬럼 선택일\n'
  '        #    뿐이라 축 2 의 _col / _raw_psr 값에는 아무 영향이 없다.\n'
  '        _tdf = pd.read_csv(os.path.join(TEST_DIR, "test.csv"),\n'
  '                           usecols=["row_id", "asof_pitcher_success_rate",\n'
  '                                    "asof_pitcher_middle_rate",\n'
  '                                    "asof_batter_middle_rate"],\n'
  '                           encoding="utf-8-sig")\n'
  '        _col = _tdf["asof_pitcher_success_rate"].to_numpy(np.float64)\n'
  '        _col_mid = _tdf["asof_pitcher_middle_rate"].to_numpy(np.float64)\n'
  '        _col_bmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)\n'),
 # (8) 축 6 블록 + rawmap
 ('        # --- apply all five, row by row -----------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid}\n',
  '        # --- axis 6/6 `ctrl_asof_bmid` -- 후보 DN -------------------------\n'
  '        # raw = 이 행 자신의 `asof_batter_middle_rate` 를 test.csv 에서\n'
  '        #       ★그대로★ 가져온 값이다 (변환 없음).  공식 입력 컬럼이고,\n'
  '        #       축 2 `ctrl_asof_psr`·축 5 `ctrl_asof_mid` 와 ★완전히 같은\n'
  '        #       규약★ 이다.  주체만 투수에서 타자로 바뀐다.\n'
  '        #       원시 컬럼은 축 2 의 read (_col_bmid) 에서 이미 왔으므로\n'
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
  '        _raw_bmid = np.where(_pos_asof >= 0,\n'
  '                             _col_bmid[np.maximum(_pos_asof, 0)], np.nan)\n'
  '        # --- apply all six, row by row ------------------------------------\n'
  '        _rawmap = {"beta_ins_off": _raw_ins, "ctrl_asof_psr": _raw_psr,\n'
  '                   "sp_pit_split": _raw_sp, "sp_bat_phand": _raw_spb,\n'
  '                   "ctrl_asof_mid": _raw_mid, "ctrl_asof_bmid": _raw_bmid}\n'),
]
NEW = SRC
for i, (a, b) in enumerate(REP):
    assert NEW.count(a) == 1, f"치환 {i}: 원문 {NEW.count(a)}회 (1이어야 함)"
    NEW = NEW.replace(a, b)
    assert b in NEW, f"치환 {i} 미적용"
assert NEW != SRC
assert NEW.count("read_csv") == SRC.count("read_csv"), "test.csv 재파싱 횟수가 늘었다"
open(f"{WORK}/script_DN.py", "w", encoding="utf-8", newline="").write(NEW)
print(f"script.py {SRC.count(chr(10))}행 -> {NEW.count(chr(10))}행  치환 {len(REP)}건")

# ── 새 축 ─────────────────────────────────────────────────────────────
CF = D10["cal_frame"]
ORT = D10["orthogonality_to_shipped_raw_column_axes"]

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
   "Frozen scalars, not test statistics."),
 "raw_definition": (
   "test.csv column asof_batter_middle_rate, verbatim (변환 없음).  그 행 자신의 컬럼 "
   "하나이고, 축 2 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 읽는다 — "
   "test.csv 재파싱 횟수는 부모와 동일하다(read_csv 호출 수를 빌드가 assert 로 대조했다)."),
 "nan_policy": (
   "raw 가 비유한이면 (타자 첫 등판) 그 행의 이 축 이동은 ★정확히 0.0★ 이다.  센티넬 "
   "대입도 평균 대입도 하지 않는다.  train-2024 프레임 기준 88행(0.035%)이 여기 해당한다."),
 "applied_where": (
   "added to the blend logit BEFORE the alpha/beta_skew transform — 기존 5축과 ★완전히 "
   "같은 자리★ 다: script.py 의 `_shift` 누적에 6번째로 들어가고 `logit_mix = logit_mix "
   "+ _shift` 로 더해진다."),
 "n_nonfinite_train2024": int(CF["n_nonfinite"]),
 "derivation": {
  "🔴_이것은_계산된_꼭지점이_아니라_작은_프로브다": (
    "S263 — 2026-08-28 4발이 이것 하나로 갈렸다.  계수를 쓴 셋(CW·FB·FC)이 3.95~20배 "
    "틀렸고, 계수를 안 쓰고 작은 프로브를 놓은 하나(CZ)가 이겼다.  이 축은 ★그 규칙을 "
    "따른다★: u = -0.008 은 어떤 곡률·선형 계수 적합에서도 나오지 않았다.  이 축의 "
    "부호와 크기는 ★LB 가 정한다★."),
  "u_선정_근거": (
    "u = -0.008 은 ★형제 축 5 `ctrl_asof_mid` 가 같은 부모 계보에서 실은 값 그대로★ 다.  "
    "그 축은 같은 종류의 양(한복판 비율)을 같은 규약(원시 test.csv 열, train-2024 "
    "nanmean/nanstd 로 표준화)으로 실어 LB +2.4730710258 을 냈다.  주체만 투수에서 "
    "타자로 바뀐다.  ★새로 적합한 양은 0 개다★ — 형제 축의 걸음을 그대로 반사했다."),
  "🔴_어떤_계수도_빌려오지_않았다": (
    "S262 — 곡률 G 는 기반을 넘어 옮겨도 되지만 선형항 L 은 절대 안 된다.  이 축의 "
    "출하 진폭은 형제 축의 ★진폭★ 하나만 반사한 것이고, 형제 축의 K_L·K_G(부모 DM 이 "
    "자기 axis_offsets[4].derivation 에 적어 둔 값)를 이 축에 옮겨 쓰지 않았다.  "
    "S232·S245·S252 — 정직면(honest surface)에서 이 축의 걸음을 유도하지 않았다.  "
    "이 축의 정직면 유래 몫은 ★0%★ 다."),
  "축_선정_근거_로컬": (
    "🔴 LB 밖에서 왔다(LB 적합 규율 3).  train season==2024 253,507행에서 이미 출하된 "
    "두 ★원시 열★ 축과의 상관: `asof_pitcher_success_rate` r = "
    f"{ORT['asof_pitcher_success_rate']['r']:+.6f} · `asof_pitcher_middle_rate` r = "
    f"{ORT['asof_pitcher_middle_rate']['r']:+.6f}.  즉 우리 장부에서 가장 직교한 미청구 "
    "방향이다.  결측은 88행(0.035%)뿐이라 커버리지 손실도 없다."),
  "곡률_라벨없이_실측": (
    "🔴 S255 — 걸음을 정하기 전에 곡률을 실측한다.  이 빌드가 u = ±0.008 두 판본을 "
    "만들어 부모 예측 p0 대비 Q = (K/2)*[mean((p+ - p0)^2) + mean((p- - p0)^2)] 를 "
    "★라벨 0개·LB 슬롯 0개★ 로 쟀다.  결과는 work/curvature.json 과 gates_DN.json 의 "
    "`G10_curvature_label_free` 에 있다.  🔴 이 값은 ★크기의 상한★ 을 주는 것이지 "
    "부호를 주지 않는다."),
  "d_adj_유지_근거": (
    "★d_adj 는 부모 값 그대로★ 이고 재유도하지 않았다.  center 가 판정 프레임(train "
    "season==2024, 253,507행)에서 그 컬럼의 nanmean 이라 오프셋이 ★평균 중립★ 이다: "
    f"실측 mean(delta) = {CF['mean_delta_all_rows_nonfinite_as_zero']:.4e} (비유한 88행은 "
    f"delta 0.0 으로 두고 전 행 평균), u = -0.008 을 곱하면 "
    f"{CF['mean_shift_at_u_minus_0008']:.4e} 다.  형제 축 5 가 같은 근거로 d_adj 를 "
    "안 옮겼고, 이 후보도 안 옮긴다 — 조준점은 부모다."),
  "🔴_사전_차감_레벨손실": (
    "이 축의 알려진 대가를 미리 적어 둔다(S8 — 결과를 보고 쓰는 것이 아니다).  이 "
    "컬럼의 시즌 평균은 2024 center/scale 눈금으로 2019 -1.238sd · 2020 -0.679sd · "
    "2021 -0.489sd · 2022 -0.389sd · 2023 -0.272sd · 2024 0.000sd 로 ★6년 연속 "
    "단조상승★ 한다(연평균 약 +0.25sd).  2024 center 를 동결한 채 2025 test 에 "
    "적용하면 2025 의 레벨이 center 보다 위에 있어 축이 상수 오프셋을 하나 얹는 셈이 "
    "되고, 그만큼을 잃는다.  형제 축 5 도 같은 성질이었고 부모 DM 은 그것을 "
    "-0.6 ~ -1.0 pts 로 적었다 — 이 축의 연 표류가 형제보다 크므로(0.25 vs 0.21 sd/년) "
    "적어도 그만큼은 든다.  🔴 이것은 추정이지 측정이 아니다."),
  "🔴_꼬리_노출_G5문서화": (
    "절차 규약은 축 2·5 와 같지만 ★분포는 같지 않다★.  실측(train season==2024, "
    f"253,507행): |delta| p50 {CF['abs_delta_p50']:.3f} · p99 {CF['abs_delta_p99']:.3f} · "
    f"p99.9 {CF['abs_delta_p999']:.3f} · max {CF['abs_delta_max']:.3f} — 축 5 는 "
    "0.542 / 3.026 / 5.055 / 27.411, 축 2 는 0.555 / 3.008 / 6.777 / 9.403 이다.  "
    f"u = 0.008 에서 |u*delta| max = {CF['max_abs_u_times_delta_at_0008']:.5f}.  꼬리의 "
    "출처는 소표본 타자다 — asof_batter_n < 10 인 779행이 mean|delta| 6.691 · "
    "max 37.229 를 갖고, n >= 200 인 241,828행은 mean|delta| 0.525 다.  ★이번 발은 "
    "이것을 그대로 둔다★ — 수축이나 클리핑을 넣으면 다른 축이 되고 항등 앵커를 다시 "
    "떠야 한다."),
  "규칙_준수": (
    "외부데이터 0 · test 분포 통계 0 (평균/분위/정렬/순위/적합 없음) · ★행별 자기 컬럼 "
    "lookup 뿐★ (row i 는 row i 의 asof_batter_middle_rate 만 본다) · center/scale 은 "
    "train 에서 동결한 스칼라 2개이고 추론에서 적합하지 않는다.  🔴 이 축은 진단 print "
    "를 새로 만들지 않는다 — 공용 루프의 축별 유한/무이동 행 수만 나온다."),
  "S277_이미_출하_코드가_이_양을_재중심하는가": D10[
    "S277_is_this_quantity_already_re_centred_by_shipped_code"],
 },
}


def mk(tag, mut):
    d = f"{PKG}/{tag}"
    assert not os.path.isdir(d), f"{d} 가 이미 있다 — 지우지 않는다(지휘자 지시)"
    shutil.copytree(f"{PKG}/PAR", d, symlinks=False,
                    ignore=shutil.ignore_patterns("__pycache__", "data", "output",
                                                  "_module_origins.json",
                                                  "_module_origins_root.json"))
    bl = json.load(open(f"{d}/blend.json", encoding="utf-8"))
    mut(d, bl)
    open(f"{d}/blend.json", "w", encoding="utf-8", newline="").write(
        json.dumps(bl, ensure_ascii=False, indent=1))
    return d


PAR_BL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
D_ADJ = PAR_BL["d_adj"]


def lineage(bl, cand):
    """S91·S240 — 계보 키 4종을 이 세대 것으로 다시 쓴다."""
    bl["candidate"] = cand
    bl["parent"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "candidate": "DM",
       "zip": "submit_한복판비율축_마이너스0008.zip",
       "zip_sha256": PAR_ZIP_SHA,
       "lb": PAR_LB,
       "base": "CZ",
       "base_lb": 1155.1387831487,
       "delta_pts": 2.4730710258,
       "🔴_the_parent_zip_mislabels_itself": (
         "🔴 부모 zip 의 blend.json 은 `candidate` 를 \"CZ\" 라고 적고 있는데 ★거짓★ 이다.  "
         "그 패키지는 CZ 위에 다섯 번째 축 `ctrl_asof_mid`(u = -0.008)를 얹은 것이고 자기 "
         "키 `DM_axis`·`changed_vs_CZ` 가 그것을 서술한다 — 패치팩으로 얹으면서 계보 키를 "
         "안 고친 것이다(S91).  이 후보는 그 네 키(candidate·parent·identity_anchor·"
         "derived_from)를 ★전부 자기 세대 것으로 다시 썼다★."),
       "🔴_what_DN_moves": (
         "axis_offsets 를 5개 → 6개로 늘리고 여섯 번째를 `ctrl_asof_bmid` 로 한다.  원시 "
         "열은 test.csv 의 `asof_batter_middle_rate` 이고, 축 2 가 이미 여는 단 한 번의 "
         "read 에 usecols 로 얹는다.  ★새 파일 0개 · 새 표 0개 · test.csv 재파싱 0회 · "
         "부스터 0비트★.  d_adj·alpha·beta_skew·w_v6·failtype_offset·aux_offsets·"
         "count_offsets·기존 5축의 u/center/scale/sd1/sd2 전부 부모 값 그대로다."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["parent"],
    }
    bl["identity_anchor"] = {
     "🔴_CORRECTED_FOR_THIS_CANDIDATE": {
       "🔴_how_to_reproduce_the_parent_from_this_package": (
         "Set this file's `axis_offsets[5].u` (the `ctrl_asof_bmid` entry) to 0.0, then "
         "rebuild the package the same way.  ★NOTHING ELSE has to be reverted★ — this "
         "candidate does not move `d_adj` (the new axis is mean-neutral on the "
         "calibration frame by construction; see axis_offsets[5].derivation."
         "d_adj_유지_근거).  🔴 script.py DOES NOT HAVE TO BE REVERTED: u = 0.0 makes "
         "0.0 * delta exactly 0.0 for every finite delta, 0.0 + 0.0 is exactly 0.0 and "
         "x + 0.0 is exact in IEEE-754, so the new code RUNS -- it still reads the extra "
         "column, still builds the sixth raw vector and still reports it through the "
         "shared per-axis diagnostic loop -- and the output is still the parent's, byte "
         "for byte.  That is this build's gate G1: the anchored package ANC_DN carries "
         "DN's rewritten prose AND the patched script.py while holding that ONE number "
         "at 0.0, and its output CSV matches CANDIDATE DM's sha256 on all six frames it "
         "is run on: real5, keep110, cov, band, caD and degen.  🔴 THE TARGET IS DM."),
       "files_to_touch": ["blend.json"],
       "🔴_note": (
         "한 파일, ★한 개 숫자★ — 부모 zip 과 다른 파일은 둘(blend.json · script.py)이지만 "
         "script.py 는 u = 0 에서 산술적으로 무해하다.  이것이 코드 변경이 무해함의 유일한 "
         "기계 증명이다."),
     },
     "inherited_history_verbatim_from_the_parent_zip": PAR_BL["identity_anchor"],
    }
    bl["derived_from"] = (
     "TRAIN season-2024 calibration frame (253,507 rows) — 부모 계보에서 그대로 물려받은 "
     "calCG 규약(두 in-season 앵커를 anchor_lt2024 로 교체, offset.lookup_d / lookup_n 을 "
     "<=2023 키 집합 1,202 로 제한) 위에서 동결한 스칼라다.  🔴 CANDIDATE DN DOES NOT "
     "RE-SOLVE d_adj: d_adj = " + repr(D_ADJ) + " 는 부모 DM(LB " + repr(PAR_LB) + ") 의 "
     "값 그대로이고 이 후보가 한 비트도 움직이지 않았다.  근거는 새 축이 그 프레임에서 "
     "★평균 중립★ 이라는 실측이다 — center 가 그 프레임에서 원시 열 "
     "`asof_batter_middle_rate` 의 nanmean 이라 mean(delta) = "
     f"{CF['mean_delta_all_rows_nonfinite_as_zero']:.4e} 이고 u = -0.008 을 곱하면 "
     f"{CF['mean_shift_at_u_minus_0008']:.4e} 다.  이 후보가 새로 동결한 스칼라는 그 축의 "
     "center/scale 두 개뿐이고 둘 다 train 에서 왔다.  No statistic of test.csv is read at "
     "inference time.  🔴 THIS SENTENCE REPLACES THE PARENT'S, WHICH DESCRIBED CZ's "
     "SOLVE (S240 — `derived_from` 은 면책 대상이 아니라 매 세대 다시 쓰는 키다).  "
     "물려받은 원문은 `DN_inherited_derived_from` 에 그대로 있다.")
    bl["DN_inherited_derived_from"] = PAR_BL["derived_from"]


def add_axis(u, cand):
    def f(d, bl):
        shutil.copyfile(f"{WORK}/script_DN.py", f"{d}/script.py")
        a = dict(AXIS)
        a["u"] = u
        a["derivation"] = json.loads(json.dumps(AXIS["derivation"], ensure_ascii=False))
        bl["axis_offsets"] = list(bl["axis_offsets"]) + [a]
        lineage(bl, cand)
        bl["changed_vs_DM"] = {
         "🔴_WHAT_CHANGED": (
           "축 오프셋 1개 추가 (`ctrl_asof_bmid`, u = " + repr(u) + ", axis 6/6) — "
           "그것 하나뿐이다.  EXACTLY TWO FILES differ from the parent zip: script.py 와 "
           "blend.json.  나머지 176개 항목은 부모와 ★바이트 동일★ 하다."),
         "files_that_differ_from_the_parent_zip": ["blend.json", "script.py"],
         "script_py": (
           "len(_axl) assert 5->6 · ax_names 에 여섯 번째 이름 추가 · 축 2 의 test.csv "
           "read 에 usecols 로 asof_batter_middle_rate 추가 · 축 6 raw 벡터 2줄 · "
           "_rawmap 항목 1개 · 머리 주석 FIVE->SIX 와 축 번호표.  ★새 read_csv 0개 · "
           "새 집계 0개 · 새 print 0개★."),
         "blend_json": (
           "axis_offsets 6번째 원소 + DN_axis + DN_rule_2_4 + DN_probe_declaration + "
           "DN_S73_prose_sweep + changed_vs_DM + DN_inherited_derived_from, 그리고 계보 "
           "키 4종(candidate·parent·identity_anchor·derived_from) 재작성."),
         "🔴_not_moved": (
           "d_adj · alpha · beta_skew · w_v6 · beta_f · c_aa · smooth_weights · "
           "failtype_offset · aux_offsets · count_offsets(theta 12칸) · 기존 5축 전부 · "
           "부스터/표/팔 구성 전부."),
        }
        bl["DN_axis"] = {
         "candidate": cand,
         "parent": "submit_한복판비율축_마이너스0008.zip (내용상 후보 DM)",
         "parent_sha256": PAR_ZIP_SHA,
         "parent_lb": PAR_LB,
         "axis": "axis_offsets[5] 참조 — u/center/scale/nan_policy/derivation 전부 거기에 있다",
         "d_adj_unchanged": (
           "★d_adj 는 부모 값 " + repr(D_ADJ) + " 를 그대로 승계한다★ — 재유도하지 않았다."),
         "🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE": (
           "S263 — 신규 축은 계산된 꼭지점 착륙이 아니라 ★작은 프로브★ 로 시작한다.  "
           "u = -0.008 은 형제 축 5 의 걸음을 그대로 반사한 값이고 어떤 계수 적합에서도 "
           "나오지 않았다.  gain ≈ 2*u*K*L 로 선형인 구간이라 ★부호만 맞으면 이긴다★.  "
           "틀리면 S250 대로 더 값진 정보(부호)가 온다."),
         "🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT": {
           "if_gain_is_positive": (
             "이 축의 L 을 이 기반에서 산 것이다.  2발째로 ★같은 기반★ 에서 꼭지점 "
             "착륙(u* = L/G, G 는 이 빌드가 라벨 없이 실측한 곡률)을 한다.  🔴 다른 "
             "기반의 곡률·선형항을 빌려오지 않는다(S262)."),
           "if_gain_is_negative": (
             "부호가 반대다.  S264 — 부호 반전은 |L| > Q 일 때만 이긴다.  이 빌드가 잰 "
             "Q(라벨 없는 자기곡률)와 실측 |gain| 을 비교해 판정하고, 조건을 못 넘으면 "
             "이 축을 닫는다."),
           "if_abs_gain_is_under_1.4": (
             "팀 원장의 검출한계(제76, LB +1.4) 아래다 — 부호를 읽었다고 주장하지 않고 "
             "이 축을 미결로 둔다.  진폭을 키워 다시 사지 않는다(S42)."),
           "🔴_the_one_point_solve": (
             "이 한 점이 이 축·이 기반의 L 을 준다.  L̂ = gain/(2*u*K) + (Q/K)*u 로 "
             "역산하고, 그 L 은 ★이 기반에서만★ 유효하다."),
         },
         "🔴_inherited_keys_are_history_not_claims_about_DN": (
           "이 blend.json 의 나머지 최상위 키들은 DM · CZ 및 그 조상들의 ★이력★ 이다.  "
           "DN 에 대한 주장이 아니며 하나도 지우지 않았다."),
        }
        bl["DN_rule_2_4"] = {
         "column": ("test.csv 의 `asof_batter_middle_rate` 는 공식 입력 열이고 대회 "
                    "주최측이 as-of 로 만들어 준 것이다.  우리가 만들지 않는다."),
         "application": (
           "row i 는 row i 의 그 열 값 하나만 본다.  delta = (raw - center)/scale, "
           "shift = u * delta.  평균·분위·정렬·순위·적합이 없고 test 프레임 통계를 "
           "만들지 않는다.  비유한이면 shift = 0.0 (센티넬 없음)."),
         "constants": ("center/scale 두 스칼라가 전부이고 둘 다 train season==2024 에서 "
                       "동결했다.  추론에서 적합하지 않는다(fitted_at_inference: false)."),
         "🔴_gates": ("전체 셔플(cov vs cov_shuf_al) + 내용/위치 분리(caD/caE · mixB/mixC "
                      "KEEP 110행) 두 가지로 기계 판정한다.  결과는 gates_DN.json."),
        }
        bl["DN_S73_prose_sweep"] = {
         "🔴_scope_of_this_sweep": (
           "① blend.json 의 전 리프(이력 표시 리프 포함) ② 패키지 전체의 .json/.py/.txt 를 "
           "'axis_offsets|축 오프셋' 과 'ctrl_asof_bmid|asof_batter_middle_rate' 로 grep "
           "③ 상이 두 파일의 자기 서술 문장 육안 통독.  ⚠️ 돌지 않은 것: 승계 176파일의 "
           "산문 전문 통독(부모 감사가 덮는다)."),
         "top_level_keys_describing_DN": {"filled_by_20_build": True},
         "S137_S240_value_leaves": {
           "shipped_d_adj": repr(D_ADJ),
           "parent_d_adj": repr(D_ADJ),
           "🔴": "이 후보는 d_adj 를 안 옮긴다 — 두 값이 같은 것이 정상이다.",
           "shipped_u_ctrl_asof_bmid": repr(u),
         },
        }
    return f


os.makedirs(PKG, exist_ok=True)
mk("DN", add_axis(U_SHIP, "DN"))
mk("ANC_DN", add_axis(0.0, "DN"))            # 항등 앵커 (산문은 DN 것)
mk("DNP", add_axis(U_PLUS, "DN_PROBE_PLUS"))  # 곡률 전용, 출하 안 함

# ── 산문 스윕 계수는 기계로 센다 ────────────────────────────────────────
for tag in ("DN", "ANC_DN", "DNP"):
    p = f"{PKG}/{tag}/blend.json"
    bl = json.load(open(p, encoding="utf-8"))
    new = sorted(set(bl) - set(PAR_BL))
    chg = sorted(k for k in set(bl) & set(PAR_BL)
                 if json.dumps(bl[k], ensure_ascii=False, sort_keys=True)
                 != json.dumps(PAR_BL[k], ensure_ascii=False, sort_keys=True))
    bl["DN_S73_prose_sweep"]["top_level_keys_describing_DN"] = {
      "n_new": len(new), "new": new, "n_changed": len(chg), "changed": chg,
      "n_removed": len(sorted(set(PAR_BL) - set(bl))),
      "total": len(new) + len(chg),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 JSON 직렬화로 대조했다.",
    }
    open(p, "w", encoding="utf-8", newline="").write(
        json.dumps(bl, ensure_ascii=False, indent=1))

json.dump({"axis": AXIS, "u_ship": U_SHIP, "u_plus": U_PLUS,
           "center_repr": repr(CENTER), "scale_repr": repr(SCALE),
           "d_adj_repr": repr(D_ADJ),
           "script_sha_parent": sh(f"{PKG}/PAR/script.py"),
           "script_sha_dn": sh(f"{WORK}/script_DN.py"),
           "parent_zip_sha256": PAR_ZIP_SHA},
          open(f"{WORK}/build20.json", "w"), ensure_ascii=False, indent=1)
for t in ("PAR", "DN", "ANC_DN", "DNP"):
    print(t, "blend", sh(f"{PKG}/{t}/blend.json")[:16],
          "script", sh(f"{PKG}/{t}/script.py")[:16])
print("BUILD20_OK  pkg: DN ANC_DN DNP  (청소 단계 없음 — 중간 산출물 전부 보존)")

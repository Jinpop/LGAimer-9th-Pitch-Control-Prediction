"""exp_073 stage 0 — 출하 실현 가능성.  🔴 스크린보다 먼저 돈다.

BM 이 이 검사를 건너뛰고 32분을 쓰고 zip 없이 끝났다.  BI·BJ·BK 와 출하 비용이
다르다 — 그 셋은 이미 출하된 값으로 방향을 만들어 blend.json 한 줄이면 됐지만
TrackMan 축은 **새 표 파일**이 패키지에 들어간다.

Q1 표 1개(CSV/npy)를 패키지에 넣고 script.py 에서 행별 방향을 만들 수 있는가.  상이·추가 파일 몇 개인가.
Q2 2025 test 행에 붙일 키가 pitcher_id 하나로 충분한가 (추론 시점에 존재하는가).
Q3 커버리지 — 표에 없는 투수 행 비율.  그 행은 오프셋 0 (sentinel 금지).
Q4 추론 시간 증가분.  현행 146 초 대비.

여기서 재는 것만 측정값이고 나머지는 코드 대조로 답한다.
"""
import json
import os
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp073_tm"
PKG = f"{H}/build_BN/pkg_BN"
MAP571 = f"{H}/p6_verify/src/pitcher_id_map.csv"
MAP685 = f"{H}/experiments/exp_002/out/pitcher_id_map.csv"
COV = f"{H}/build_AL/frames/cov/test.csv"
REAL5 = f"{H}/build_AL/frames/real5/test.csv"
TR2024 = f"{H}/build_AC/work/tr2024/test.csv"

t_all = time.time()
R = {}

# ── Q1 ────────────────────────────────────────────────────────────────
# 구조 답: 이 패키지는 **이미** pitcher_id 로 키를 잡는 동결 표를 싣고 있다.
#   model_form/pmap_pitcher_ids.npy + n_end_2024ext.npy + s_end_2024ext.npy
# form_offset.load_frozen() 이 그 표를 읽고 `.get(pid, unseen_ix)` 로 행별 조회한다.
# 축 오프셋 기구(axis_offsets)는 blend.json 의 name -> script.py 의 raw 블록 대응이다.
src = open(f"{PKG}/script.py", encoding="utf-8").read()
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
ax_names = [a["name"] for a in bl["axis_offsets"]]
R["Q1_shipping_vehicle"] = {
    "existing_axis_names": ax_names,
    "axis_name_assert_in_script": ('assert ax_names == ["beta_ins_off", '
                                   '"ctrl_asof_psr"]' in src.replace("'", '"')),
    "hardcoded_axis_count_assert": 'assert len(_axl) == 2' in src,
    "precedent_per_pitcher_table_in_pkg": sorted(os.listdir(f"{PKG}/model_form")),
    "precedent_lookup": ("form_offset.load_frozen(): pmap = {int(v): i for i, v "
                         "in enumerate(pmap_ids)} 후 행별 .get(pid, unseen_ix) — "
                         "프레임 전체를 읽지 않는 행별 조회"),
    "files_added_or_changed_for_a_trackman_axis": {
        "added": ["model_tm/<표>.npz  (pitcher_id -> tm_* 값 · train 유래 동결)"],
        "changed": ["blend.json  (axis_offsets 에 항목 1개 · u/center/scale)",
                    "script.py   (raw 블록 1개 + ax_names/len(_axl) assert 2줄)"],
        "n_added": 1, "n_changed": 2, "total_differing_vs_parent": 3},
    "answer": ("예 — 가능하다.  다만 A단(사후 변환만·상이 파일 1개)이 아니라 "
               "**C단(새 피처 소스)** 이다: 새 원료 파일(trackman)에서 만든 새 표가 "
               "패키지에 들어가고 script.py 가 바뀐다.  packager 전면 검증 대상."),
    "grade": "C단"}

# ── Q2 ────────────────────────────────────────────────────────────────
cols_cov = pd.read_csv(COV, encoding="utf-8-sig", nrows=0).columns.tolist()
cols_r5 = pd.read_csv(REAL5, encoding="utf-8-sig", nrows=0).columns.tolist()
R["Q2_key"] = {
    "pitcher_id_in_cov_frame": "pitcher_id" in cols_cov,
    "pitcher_id_in_real5_frame": "pitcher_id" in cols_r5,
    "trackman_side_key": "pitcher_trackman_id",
    "answer": ("예 — pitcher_id 는 대회 test.csv 의 공식 입력 열이다.  "
               "트랙맨 쪽 키 pitcher_trackman_id 는 표를 만들 때(train 시점)만 "
               "쓰이고 추론에는 등장하지 않는다.")}

# ── Q3 커버리지 ───────────────────────────────────────────────────────
m571 = pd.read_csv(MAP571)
m685 = pd.read_csv(MAP685)
ids571 = set(m571["pitcher_id"].astype(int))
ids685 = set(m685["pitcher_id"].astype(int))

cov = {}
for nm, path in (("tr2024_frame", TR2024), ("cov_frame", COV),
                 ("real5_frame", REAL5)):
    pid = pd.read_csv(path, encoding="utf-8-sig",
                      usecols=["pitcher_id"])["pitcher_id"].astype(int)
    cov[nm] = {
        "n_rows": int(len(pid)), "n_pitchers": int(pid.nunique()),
        "row_cov_571": float(pid.isin(ids571).mean()),
        "row_cov_685": float(pid.isin(ids685).mean()),
        "pitcher_cov_571": float(pd.Series(sorted(pid.unique()))
                                 .isin(ids571).mean()),
        "pitcher_cov_685": float(pd.Series(sorted(pid.unique()))
                                 .isin(ids685).mean())}
R["Q3_coverage_map_only"] = cov
R["Q3_note"] = ("이건 **매핑** 커버리지다.  표 값이 유한한지(그 투수의 season<Y "
                "트랙맨 행이 실제로 있는지)는 s1 이 낸다.  cov 프레임은 2024 행을 "
                "2025 로 덮은 대역이라 2025 신인은 들어 있지 않다 — 진짜 2025 "
                "커버리지의 상한이다.  tr2024 프레임에 Y=2024 표(season<=2023)를 "
                "붙인 수치가 '한 시즌 앞' 의 정직한 대리값이다(s1).")
R["Q3_missing_policy"] = ("표에 없는 투수 행은 raw = NaN → script.py 의 기존 "
                          "축 블록이 `_delta[_fin]` 로 0.0 을 남긴다.  sentinel "
                          "충전 없음 — 부모가 둔 자리에 그대로 둔다.")

# ── Q4 추론 시간 증가분 (단독 실행 · 경합 없음) ───────────────────────
# 한 축이 추론에 더하는 것: test.csv 한 번 더 읽기(2열) + 사전 조회 + 정규화.
tbl = {int(p): float(v) for p, v in
       zip(sorted(ids571), np.linspace(0.05, 0.30, len(ids571)))}
reps, ts = 5, []
for _ in range(reps):
    t0 = time.perf_counter()
    tdf = pd.read_csv(COV, encoding="utf-8-sig",
                      usecols=["row_id", "pitcher_id"])
    pidv = tdf["pitcher_id"].to_numpy()
    raw = np.array([tbl.get(int(p), np.nan) for p in pidv], dtype=np.float64)
    fin = np.isfinite(raw)
    delta = np.zeros(len(raw))
    delta[fin] = (raw[fin] - 0.1055) / 0.03
    _ = 0.006 * delta
    ts.append(time.perf_counter() - t0)
R["Q4_inference_cost"] = {
    "frame_rows": int(len(pidv)), "reps": reps,
    "seconds_per_axis_block": {"min": round(min(ts), 3),
                               "median": round(float(np.median(ts)), 3),
                               "max": round(max(ts), 3)},
    "baseline_G6_seconds_BN": 146,
    "estimated_new_total_seconds": 146 + round(float(np.median(ts)), 1),
    "answer": ("무시할 수 있는 수준.  표 로드는 npz 수백 KB, 조회는 행별 dict "
               "이고 test.csv 재독은 기존 축 블록 둘이 이미 하고 있다.")}

os.makedirs(OUT, exist_ok=True)
R["wall_s"] = round(time.time() - t_all, 1)
json.dump(R, open(f"{OUT}/stage0.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))

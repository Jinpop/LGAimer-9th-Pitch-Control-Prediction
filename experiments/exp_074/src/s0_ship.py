"""exp_074 stage 0 — 배포 가능성.  🔴 커버리지는 표가 나온 뒤라야 진짜 값이 나온다.

Q1 표 1개(pitcher_id 키)로 전 축을 실을 수 있는가.  상이·추가 파일 수는.
Q2 커버리지 — 표에 없거나 값이 결측인 투수 비율.  결측은 오프셋 0 (sentinel 금지).
Q3 🔴 real5(진짜 2025 행 5개) 커버리지도 함께 재라.
   exp_073 에서 대역 96.49% vs real5 20% 였다 — **대역은 상한이다.**
Q4 추론 시간 증가분 (경합 없는 단독 실행).

🔴 설계 단계에서 이미 절반을 잘라냈다: 경기 내 폼 · 최근 N투구 창 ·
   test 안으로 넘어가는 창은 전부 배제.  2025 test 행의 최근 창을 계산하려면
   다른 test 행을 읽어야 하므로 규칙 2-4 위반이다.  살아남는 형태는
   **투수별로 <=2024 말에 얼린 스칼라 하나** 뿐이고 그건 ins_*·d_form_eb 와
   정확히 같은 배포 형태다.
"""
import json
import os
import time

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp074_fm"
PKG = f"{H}/build_BN/pkg_BN"
COV = f"{H}/build_AL/frames/cov/test.csv"
REAL5 = f"{H}/build_AL/frames/real5/test.csv"
TR2024 = f"{H}/build_AC/work/tr2024/test.csv"
AXES = ["fm_rev_g5", "fm_ball_g5", "fm_strike_g5", "fm_suc_sd5",
        "fm_suc_sd10", "fm_suc_trend"]

t_all = time.time()
R = {}

src = open(f"{PKG}/script.py", encoding="utf-8").read()
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
ax_names = [a["name"] for a in bl["axis_offsets"]]

# ── Q1 출하 수단 ─────────────────────────────────────────────────────
R["Q1_shipping_vehicle"] = {
    "existing_axis_names": ax_names,
    "precedent_per_pitcher_table_in_pkg": sorted(os.listdir(f"{PKG}/model_form")),
    "precedent_lookup": ("form_offset.load_frozen(): pmap = {int(v): i ...} 후 "
                         "행별 .get(pid, unseen_ix) — 프레임 전체를 읽지 않는 "
                         "행별 조회.  ins 계열은 inseason.apply(df, anchor) 로 "
                         "행 자기 값 + 동결 앵커만 쓴다."),
    "as_offset_axis": {
        "added": ["model_fm/fm_table.npz  (pitcher_id -> fm_* · train 유래 동결)"],
        "changed": ["blend.json (axis_offsets 항목 +1)",
                    "script.py  (raw 블록 1개 + ax_names/len(_axl) assert 2줄)"],
        "n_added": 1, "n_changed": 2, "total_differing_vs_parent": 3,
        "grade": "C단 (새 피처 소스·새 표) — packager 전면 검증 대상"},
    "as_stacker_column": {
        "note": ("🔴 config 가 지정한 다음 단계는 오프셋이 아니라 **스태커 열** "
                 "이다 (배포 meta 의 'the offset was the WEAK form of the same "
                 "signal').  그 경로는 AA 팔 재학습이 필요하고 model/ 부스터가 "
                 "통째로 바뀐다 — 상이 파일이 훨씬 많고 meta.json 의 트리 수 "
                 "기록도 함께 고쳐야 한다."),
        "grade": "C단 + 새 부스터"},
    "answer": ("예 — 표 1개(pitcher_id 키)로 6축 전부를 실을 수 있다. "
               "이 패키지는 이미 같은 형태의 동결 표(model_form/)를 싣고 있다.")}

# ── Q2/Q3 커버리지 — 매핑이 아니라 **표 값이 유한한 행** 기준 ─────────
tb24 = pd.read_csv(f"{OUT}/fm_table_Y2024.csv", index_col="pitcher_id")
tb25 = pd.read_csv(f"{OUT}/fm_table_Y2025.csv", index_col="pitcher_id")
COVR = {}
for fname, path, tb, tag in (
        ("tr2024_frame", TR2024, tb24, "Y=2024 표(season<=2023) — 스크린 프레임"),
        ("cov_frame_standin2025", COV, tb25, "Y=2025 표(season<=2024) — 🔴 대역"),
        ("real5_frame", REAL5, tb25, "Y=2025 표 — 🔴 진짜 2025 행 5개")):
    pid = pd.read_csv(path, encoding="utf-8-sig",
                      usecols=["pitcher_id"])["pitcher_id"].astype(int)
    e = {"table": tag, "n_rows": int(len(pid)),
         "n_pitchers": int(pid.nunique()),
         "row_in_table": float(pid.isin(tb.index).mean()),
         "pitcher_in_table": float(pd.Series(sorted(pid.unique()))
                                   .isin(tb.index).mean())}
    for c in AXES:
        d = {int(k): float(v) for k, v in tb[c].items()}
        v = np.array([d.get(int(p), np.nan) for p in pid], dtype=np.float64)
        e[f"row_finite_{c}"] = float(np.isfinite(v).mean())
    COVR[fname] = e
R["Q2Q3_coverage"] = COVR
R["Q3_note"] = ("cov 프레임은 train[season==2024] 의 season 을 2025 로 덮은 "
                "**대역**이라 2025 신인이 들어 있지 않다 — 진짜 커버리지의 "
                "**상한**이다 (S: 대역 프레임 함정). real5 는 대회 test.csv 의 "
                "실제 5행이라 표본이 작지만 대역이 아니다. 두 값이 갈리면 "
                "real5 쪽이 배포 현실에 가깝다.")
R["Q2_missing_policy"] = ("표에 없거나 값이 결측인 투수 행은 raw = NaN → "
                          "script.py 의 기존 축 블록이 _delta[_fin] 로 0.0 을 "
                          "남긴다.  sentinel 충전 없음.")

# ── Q4 추론 시간 증가분 (단독 실행) ──────────────────────────────────
tbl = {int(p): float(v) for p, v in
       zip(tb25.index.astype(int), np.linspace(0.05, 0.30, len(tb25)))}
reps, ts = 5, []
for _ in range(reps):
    t0 = time.perf_counter()
    tdf = pd.read_csv(COV, encoding="utf-8-sig", usecols=["row_id", "pitcher_id"])
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
    "answer": "무시할 수 있는 수준 — 표는 수백 KB, 조회는 행별 dict."}

os.makedirs(OUT, exist_ok=True)
R["wall_s"] = round(time.time() - t_all, 1)
json.dump(R, open(f"{OUT}/stage0.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))

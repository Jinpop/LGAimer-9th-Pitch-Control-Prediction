"""exp_075 stage 0 — 배포 가능성 · 커버리지 · 🔴 AN 함정 점검.

Q1 표 1개(batter_id 키)로 표 축 전부를 실을 수 있는가.  상이·추가 파일 수는.
   `script.py` 의 ax_names·len(_axl) assert 동반 수정 필요 여부도 확인한다.
Q2 커버리지 — 🔴 **대역 프레임과 real5 양쪽**.  exp_073 96.49% vs 20%,
   exp_074 98.76% vs 20% 로 두 번 갈렸다.  **대역은 상한이다.**
Q3 결측은 오프셋 0 (sentinel 금지).
Q4 추론 시간 증가분 (경합 없는 단독 실행).

🔴 AN 함정 — 팀원 Q·v6 가 이미 가진 정보인가.
   AN 은 팔 눈금 +41.13 을 냈는데 배포는 −1.23 이었다.  원인은 팀원 Q 가 그
   정보를 이미 갖고 있어 블렌드에 새것이 없었기 때문이다.
   여기서는 세 층으로 본다:
     (a) 이름 — v6 features.txt · Q meta.json features 에 타자 계열이 무엇이 있나
     (b) 구조 — 이번 스크린의 잔차 r0 는 **BN 블렌드 전체**(v6 를 w_v6=0.70 으로
         포함) 에 대한 것이다.  AN 때의 팔 단독 잔차와 다르다.  v6 가 이미
         흡수했다면 여기서는 t 가 죽는 형태로 이미 드러난다.
     (c) 실측 — corr(축 u, logit(v6 예측)) 로 v6 출력이 축을 얼마나 따라가는지
"""
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp075_bt"
PKG = f"{H}/build_BN/pkg_BN"
COV = f"{H}/build_AL/frames/cov/test.csv"
REAL5 = f"{H}/build_AL/frames/real5/test.csv"
TR2024 = f"{H}/build_AC/work/tr2024/test.csv"
AXES_TABLE = ["bt_rev", "bt_ball", "bt_strike", "bt_suc_sd", "bt_hand_dev"]
AXES_ROW = ["bt_x_psr", "bt_d_psr"]
EPS = 1e-6

t_all = time.time()
R = {}

src = open(f"{PKG}/script.py", encoding="utf-8").read()
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
ax_names = [a["name"] for a in bl["axis_offsets"]]

# ── Q1 출하 수단 ─────────────────────────────────────────────────────
R["Q1_shipping_vehicle"] = {
    "existing_axis_names": ax_names,
    "precedent_per_key_table_in_pkg": sorted(os.listdir(f"{PKG}/model_form")),
    "precedent_lookup": ("form_offset.load_frozen(): pmap = {int(v): i ...} 후 "
                         "행별 .get(id, unseen_ix) — 프레임 전체를 읽지 않는 "
                         "행별 조회.  batter_id 키도 같은 구조로 그대로 된다."),
    "script_py_assert_lines": [l.strip() for l in src.splitlines()
                               if "ax_names" in l or "_axl" in l],
    "table_axes_as_offset": {
        "added": ["model_bt/bt_table.npz  (batter_id -> bt_* · train 유래 동결)"],
        "changed": ["blend.json (axis_offsets 항목 +1)",
                    "script.py  (raw 블록 1개 + ax_names/len(_axl) assert 2줄)"],
        "n_added": 1, "n_changed": 2, "total_differing_vs_parent": 3,
        "grade": "C단 (새 피처 소스·새 표) — packager 전면 검증 대상"},
    "row_axes_as_offset": {
        "added": [],
        "changed": ["blend.json", "script.py (산술 1줄 — 주최측 열 둘의 곱/차)"],
        "n_added": 0, "n_changed": 2, "total_differing_vs_parent": 2,
        "grade": ("A/B단 수준 — 새 표 파일이 없다.  bt_x_psr·bt_d_psr 은 "
                  "test.csv 가 이미 주는 두 열의 산술이라 동결 표가 필요 없다."),
        "note": "🔴 배포 난이도가 표 축보다 확실히 낮다."},
    "as_stacker_column": {
        "note": ("🔴 config 가 지정한 다음 단계는 오프셋이 아니라 **스태커 열** "
                 "이다.  그 경로는 AA 팔 재학습이 필요하고 model/ 부스터가 통째로 "
                 "바뀐다 — meta.json 의 트리 수 기록도 함께 고쳐야 한다 "
                 "(안 고치면 채점 서버에서 AssertionError)."),
        "grade": "C단 + 새 부스터"},
    "answer": ("예. 표 축 5개는 batter_id 키 표 1개로, 행 축 2개는 표 없이 "
               "산술로 실을 수 있다. batter_id 는 대회 test.csv 의 공식 입력 열이다.")}

# ── Q2/Q3 커버리지 ───────────────────────────────────────────────────
tb24 = pd.read_csv(f"{OUT}/bt_table_Y2024.csv", index_col="batter_id")
tb25 = pd.read_csv(f"{OUT}/bt_table_Y2025.csv", index_col="batter_id")
COVR = {}
for fname, path, tb, tag in (
        ("tr2024_frame", TR2024, tb24, "Y=2024 표(season<=2023) — 스크린 프레임"),
        ("cov_frame_standin2025", COV, tb25, "Y=2025 표(season<=2024) — 🔴 대역"),
        ("real5_frame", REAL5, tb25, "Y=2025 표 — 🔴 진짜 2025 행 5개")):
    d = pd.read_csv(path, encoding="utf-8-sig",
                    usecols=["batter_id", "asof_batter_success_rate",
                             "asof_pitcher_success_rate"])
    bid = d["batter_id"].astype(int)
    e = {"table": tag, "n_rows": int(len(bid)),
         "n_batters": int(bid.nunique()),
         "row_in_table": float(bid.isin(tb.index).mean()),
         "batter_in_table": float(pd.Series(sorted(bid.unique()))
                                  .isin(tb.index).mean())}
    for c in AXES_TABLE:
        m = {int(k): float(v) for k, v in tb[c].items()}
        v = np.array([m.get(int(p), np.nan) for p in bid], dtype=np.float64)
        e[f"row_finite_{c}"] = float(np.isfinite(v).mean())
    # 행 축은 표가 필요 없다 — 주최측 열의 유한성만 본다
    xp = (d["asof_pitcher_success_rate"].to_numpy(np.float64)
          * d["asof_batter_success_rate"].to_numpy(np.float64))
    e["row_finite_bt_x_psr"] = float(np.isfinite(xp).mean())
    e["row_finite_bt_d_psr"] = float(np.isfinite(
        d["asof_pitcher_success_rate"].to_numpy(np.float64)
        - d["asof_batter_success_rate"].to_numpy(np.float64)).mean())
    COVR[fname] = e
R["Q2Q3_coverage"] = COVR
R["Q2_note"] = ("cov 프레임은 train[season==2024] 의 season 을 2025 로 덮은 "
                "**대역**이라 2025 신인이 들어 있지 않다 — 진짜 커버리지의 "
                "**상한**이다.  real5 는 대회 test.csv 의 실제 5행이라 표본이 "
                "작지만 대역이 아니다.  두 값이 갈리면 real5 쪽이 배포 현실에 "
                "가깝다 (exp_073·exp_074 에서 두 번 갈렸다).")
R["Q3_missing_policy"] = ("표에 없거나 값이 결측인 타자 행은 raw = NaN → "
                          "script.py 의 기존 축 블록이 _delta[_fin] 로 0.0 을 "
                          "남긴다.  sentinel 충전 없음.")

# ── Q4 추론 시간 ─────────────────────────────────────────────────────
tbl = {int(p): float(v) for p, v in
       zip(tb25.index.astype(int), np.linspace(0.05, 0.30, len(tb25)))}
reps, ts = 5, []
for _ in range(reps):
    t0 = time.perf_counter()
    tdf = pd.read_csv(COV, encoding="utf-8-sig", usecols=["row_id", "batter_id"])
    bv = tdf["batter_id"].to_numpy()
    raw = np.array([tbl.get(int(p), np.nan) for p in bv], dtype=np.float64)
    fin = np.isfinite(raw)
    delta = np.zeros(len(raw))
    delta[fin] = (raw[fin] - 0.1055) / 0.03
    _ = 0.006 * delta
    ts.append(time.perf_counter() - t0)
R["Q4_inference_cost"] = {
    "frame_rows": int(len(bv)), "reps": reps,
    "seconds_per_axis_block": {"min": round(min(ts), 3),
                               "median": round(float(np.median(ts)), 3),
                               "max": round(max(ts), 3)},
    "baseline_G6_seconds_BN": 146,
    "answer": "무시할 수 있는 수준 — 표는 수백 KB, 조회는 행별 dict."}

# ── 🔴 AN 함정 점검 ──────────────────────────────────────────────────
v6_feats = [l.strip() for l in
            open(f"{PKG}/model_v6/features.txt", encoding="utf-8")
            if l.strip()]
qm = json.load(open(f"{PKG}/mate/model_q/meta.json", encoding="utf-8"))
q_feats = list(qm.get("features") or [])
q_base = list(qm.get("base_features") or [])
q_stack = list(qm.get("stack_features") or [])
aa = json.load(open(f"{PKG}/model/meta.json", encoding="utf-8"))


def bat(lst):
    return sorted([f for f in lst if "bat" in f.lower()])


AN = {
    "v6_n_features": len(v6_feats),
    "v6_batter_related": bat(v6_feats),
    "q_n_features": len(q_feats), "q_n_base_features": len(q_base),
    "q_batter_related": bat(q_feats),
    "q_base_batter_related": bat(q_base),
    "q_stack_features": q_stack,
    "our_new_axis_names": AXES_TABLE + AXES_ROW,
    "name_level_collision": sorted(
        set(AXES_TABLE + AXES_ROW) & (set(v6_feats) | set(q_feats) | set(q_base))),
    "batter_career_kinds_available_to_anyone": [
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate"],
    "batter_kinds_nobody_has": ["reverse", "ball", "strike", "게임간 산포(기복)",
                                "플래툰 스플릿(좌/우 상대 분리)"],
    "structural_note": (
        "🔴 이번 스크린의 잔차 r0 = y - p_BN 은 **BN 블렌드 전체**(v6 를 "
        "w_v6=0.70 으로, Q 계열을 S1/S2/T70 으로 포함) 에 대한 것이다. "
        "AN 때는 팔 단독 잔차에서 +41.13 이 나왔고 블렌드에 새것이 없어 진 "
        "것이었다. 여기서는 v6·Q 가 이미 가진 정보면 t 가 애초에 죽는 형태로 "
        "드러난다 — AN 과 같은 방식으로 속을 자리가 구조적으로 줄어 있다. "
        "다만 이것이 '전이비가 양수다' 를 보장하지는 않는다(S2)."),
}
# (c) 실측 — v6 출력이 축을 따라가는가
fr = pd.read_csv(TR2024, encoding="utf-8-sig",
                 usecols=["batter_id", "asof_batter_success_rate",
                          "asof_pitcher_success_rate"])
bid_row = fr["batter_id"].to_numpy(np.int64)
lp_v6 = logit(np.clip(np.load(f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"),
                      EPS, 1 - EPS))
lp_aa = logit(np.clip(np.load(f"{H}/build_AY/work/pcal_aa.npy"), EPS, 1 - EPS))
V = {}
for c in AXES_TABLE:
    m = {int(k): float(v) for k, v in tb24[c].items()}
    V[c] = np.array([m.get(int(b), np.nan) for b in bid_row], dtype=np.float64)
V["bt_x_psr"] = (fr["asof_pitcher_success_rate"].to_numpy(np.float64)
                 * fr["asof_batter_success_rate"].to_numpy(np.float64))
V["bt_d_psr"] = (fr["asof_pitcher_success_rate"].to_numpy(np.float64)
                 - fr["asof_batter_success_rate"].to_numpy(np.float64))
AN["measured_corr_axis_vs_arm_logit"] = {}
for c in AXES_TABLE + AXES_ROW:
    f = np.isfinite(V[c])
    AN["measured_corr_axis_vs_arm_logit"][c] = {
        "corr_vs_logit_v6": round(float(np.corrcoef(V[c][f], lp_v6[f])[0, 1]), 4),
        "corr_vs_logit_aa": round(float(np.corrcoef(V[c][f], lp_aa[f])[0, 1]), 4),
        "n_finite_rows": int(f.sum())}
R["AN_trap_check"] = AN

os.makedirs(OUT, exist_ok=True)
R["wall_s_s0"] = round(time.time() - t_all, 1)
json.dump(R, open(f"{OUT}/stage0.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))

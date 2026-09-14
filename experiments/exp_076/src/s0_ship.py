"""exp_076 stage 0 — 배포 수단 · 🔴 커버리지 · 🔴 AN 함정 점검.

Q1 이 축들을 무엇으로 실어야 하나.  🔴 rg_trend·rg_mix_ent 는 **표가 아예
   필요 없다** — 주최측이 test 행마다 주는 열의 산술뿐이다.
Q2 🔴 커버리지 — **대역·real5 양쪽**.  exp_073·074·075 에서 세 번 갈렸다.
   여기에 더해 **갈린 이유를 기전으로 규명**한다: prev 열은 시즌 내 경기
   집계라 시즌 초반 행에서 NaN 이 몰린다.  real5 의 row_id 가 전부 시즌
   초반(TEST_000001 ~ TEST_035185)이라면 60% NaN 은 커버리지 문제가 아니라
   **표본 위치 문제**다.  train 2024 의 같은 위치에서 같은 NaN 율이 나오는지
   본다.
Q3 결측 정책 — 오프셋 0 (sentinel 금지).
Q4 추론 시간 증가분.

🔴 AN 함정 — 팀원 Q(81피처)·v6(76피처) 가 이미 가진 정보인가.
   config 경고: **구종믹스 4열은 v6 가 쓴다.**
"""
import json
import os
import re
import time

import numpy as np
import pandas as pd
from scipy.special import logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp076_rg"
PKG = f"{H}/build_BN/pkg_BN"
COV = f"{H}/build_AL/frames/cov/test.csv"
REAL5 = f"{H}/build_AL/frames/real5/test.csv"
TR2024 = f"{H}/build_AC/work/tr2024/test.csv"
TRAIN = f"{H}/data/train.csv"
EPS = 1e-6

PREV_S = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
PREV_M = [f"asof_pitcher_prev{k}_game_middle_rate" for k in (1, 3, 5)]
MIXC = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate"]
AX_A = ["rg_prev1_suc_dev", "rg_prev3_suc_dev", "rg_prev5_suc_dev",
        "rg_prev3_mid_dev", "rg_trend"]
AX_B = ["rg_fb_dev", "rg_br_dev", "rg_off_dev", "rg_mix_ent"]
AXES = AX_A + AX_B

t_all = time.time()
R = {}

src = open(f"{PKG}/script.py", encoding="utf-8").read()
bl = json.load(open(f"{PKG}/blend.json", encoding="utf-8"))
ax_names = [a["name"] for a in bl["axis_offsets"]]

# ── Q1 출하 수단 ─────────────────────────────────────────────────────
R["Q1_shipping_vehicle"] = {
    "existing_axis_names": ax_names,
    "script_py_assert_lines": [l.strip() for l in src.splitlines()
                               if "ax_names" in l or "_axl" in l],
    "row_only_axes_no_table_needed": {
        "axes": ["rg_trend", "rg_mix_ent"],
        "added_files": [], "changed_files": ["blend.json", "script.py"],
        "total_differing_vs_parent": 2,
        "grade": ("A/B단 수준 — 새 표 파일이 0개.  주최측이 test 행마다 주는 "
                  "열의 산술 한 줄이다.  동결 포장 자체가 없다."),
        "arithmetic": {"rg_trend": "prev1_game_success_rate - prev5_game_success_rate",
                       "rg_mix_ent": "-(f*ln f + b*ln b + o*ln o), 0*ln0=0"}},
    "anchor_axes_need_table": {
        "axes": [a for a in AXES if a not in ("rg_trend", "rg_mix_ent")],
        "added_files": ["model_rg/anchor.npz (pitcher_id -> anc_* · train 유래 동결)"],
        "changed_files": ["blend.json", "script.py"],
        "total_differing_vs_parent": 3,
        "grade": "C단 (새 표 파일) — packager 전면 검증 대상",
        "precedent": ("build_AI 의 ins 앵커(anchor_lt2024.npz)가 정확히 같은 "
                      "구조다 — 투수별 동결 스칼라 + 행별 조회.  이미 4번 출하됐다.")},
    "as_stacker_column": {
        "note": ("🔴 config 가 지정한 통과 후 경로는 오프셋이 아니라 **스태커 "
                 "열**이다.  AA 팔 재학습 → model/ 부스터 교체 → meta.json 의 "
                 "트리 수 기록도 함께 고쳐야 한다(안 고치면 채점 서버 크래시)."),
        "grade": "C단 + 새 부스터"}}

# ── Q2 커버리지 ──────────────────────────────────────────────────────
a24 = pd.read_csv(f"{OUT}/anchor_Y2024.csv", index_col="pitcher_id")
a25 = pd.read_csv(f"{OUT}/anchor_Y2025.csv", index_col="pitcher_id")
ANC_COLS = ["anc_suc", "anc_mid", "anc_fb", "anc_br", "anc_off"]
NEED = ["pitcher_id"] + PREV_S + PREV_M + MIXC


def entropy3(a, b, c):
    P = np.stack([a, b, c], 1)
    bad = ~np.isfinite(P).all(1)
    Q = np.where(np.isfinite(P) & (P > 0), P, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(np.isfinite(Q), -Q * np.log(Q), 0.0)
    out = term.sum(1)
    out[bad] = np.nan
    return out


def anchor_cols_for(pid, tb):
    """🔴 dict 는 열마다 **한 번만** 만든다 (행마다 만들면 245,789x1,500 이다)."""
    out = {}
    for c in ANC_COLS:
        d_ = {int(k): float(v) for k, v in tb[c].items()}
        out[c] = np.array([d_.get(int(p), np.nan) for p in pid], np.float64)
    return out


def axis_finite(d, tb):
    pid = d["pitcher_id"].astype(int).to_numpy()
    A = anchor_cols_for(pid, tb)
    g = lambda c: d[c].to_numpy(np.float64)
    V = {"rg_prev1_suc_dev": g(PREV_S[0]) - A["anc_suc"],
         "rg_prev3_suc_dev": g(PREV_S[1]) - A["anc_suc"],
         "rg_prev5_suc_dev": g(PREV_S[2]) - A["anc_suc"],
         "rg_prev3_mid_dev": g(PREV_M[1]) - A["anc_mid"],
         "rg_trend": g(PREV_S[0]) - g(PREV_S[2]),
         "rg_fb_dev": g(MIXC[0]) - A["anc_fb"],
         "rg_br_dev": g(MIXC[1]) - A["anc_br"],
         "rg_off_dev": g(MIXC[2]) - A["anc_off"],
         "rg_mix_ent": entropy3(g(MIXC[0]), g(MIXC[1]), g(MIXC[2]))}
    e = {f"row_finite_{k}": float(np.isfinite(v).mean()) for k, v in V.items()}
    e["row_finite_raw_prev3_suc"] = float(np.isfinite(g(PREV_S[1])).mean())
    e["row_finite_raw_fastball"] = float(np.isfinite(g(MIXC[0])).mean())
    e["pitcher_in_anchor_table"] = float(
        pd.Series(sorted(set(pid))).isin(tb.index).mean())
    e["row_pitcher_in_anchor_table"] = float(pd.Series(pid).isin(tb.index).mean())
    return e


COVR = {}
for fname, path, tb, tag in (
        ("tr2024_frame", TR2024, a24, "Y=2024 앵커(season<=2023) — 스크린 프레임"),
        ("cov_frame_standin2025", COV, a25, "Y=2025 앵커(season<=2024) — 🔴 대역"),
        ("real5_frame", REAL5, a25, "Y=2025 앵커 — 🔴 진짜 2025 행 5개")):
    d = pd.read_csv(path, encoding="utf-8-sig", usecols=NEED)
    e = axis_finite(d, tb)
    e.update({"table": tag, "n_rows": int(len(d)),
              "n_pitchers": int(d["pitcher_id"].nunique())})
    COVR[fname] = e
R["Q2_coverage"] = COVR
print("=== 커버리지 ===")
for k, e in COVR.items():
    print(f"  {k:<24} 행 {e['n_rows']:>7,}  "
          f"prev3_dev {e['row_finite_rg_prev3_suc_dev']:.4f}  "
          f"trend {e['row_finite_rg_trend']:.4f}  "
          f"fb_dev {e['row_finite_rg_fb_dev']:.4f}  "
          f"mix_ent {e['row_finite_rg_mix_ent']:.4f}")

# ── 🔴 Q2b 대역/real5 가 갈리는 기전 — 시즌 내 위치 ──────────────────
d5 = pd.read_csv(REAL5, encoding="utf-8-sig",
                 usecols=["row_id", "asof_pitcher_n"] + PREV_S)
rid = [int(re.sub(r"\D", "", s)) for s in d5["row_id"].astype(str)]
tr = pd.read_csv(TRAIN, encoding="utf-8-sig",
                 usecols=["season", "asof_pitcher_n", PREV_S[0]])
m24 = (tr["season"].to_numpy() == 2024)
p1_24 = tr.loc[m24, PREV_S[0]].to_numpy(np.float64)
rank24 = np.arange(m24.sum())          # 파일 순서 = 시즌 내 시간 순서
MECH = {
    "real5_row_ids": rid,
    "real5_prev1_nan": [bool(np.isnan(v)) for v in
                        d5[PREV_S[0]].to_numpy(np.float64)],
    "real5_asof_pitcher_n": [float(v) for v in
                             d5["asof_pitcher_n"].to_numpy(np.float64)],
    "real5_max_row_id": max(rid),
    "train2024_n_rows": int(m24.sum()),
    "train2024_prev1_nan_rate_full_season": float(np.isnan(p1_24).mean()),
    "train2024_prev1_nan_rate_first_K_rows": {
        str(K): float(np.isnan(p1_24[rank24 < K]).mean())
        for K in (250, 1000, 5000, 35185, 50000)},
    "train2024_prev1_nan_at_same_ranks": {
        str(k): (bool(np.isnan(p1_24[k - 1])) if k - 1 < m24.sum() else None)
        for k in rid},
    "interpretation": (
        "prev 열은 시즌 내 경기 집계라 시즌 초반 행에서 NaN 이 몰린다. "
        "real5 의 row_id 는 전부 시즌 초반이다.  train 2024 의 같은 위치 "
        "구간에서 비슷한 NaN 율이 나오면 real5 의 60% 는 커버리지 결함이 "
        "아니라 **표본 위치**다.  5행은 이항 표준오차 ±22%p 라 어차피 "
        "정밀도가 없다.")}
R["Q2b_missing_mechanism"] = MECH
print(f"\n=== 🔴 prev 결측 기전 ===")
print(f"  real5 row_id {rid} · prev1 NaN {MECH['real5_prev1_nan']}")
print(f"  train2024 전 시즌 prev1 NaN {MECH['train2024_prev1_nan_rate_full_season']:.4f}")
print(f"  train2024 앞 K행 NaN 율 {MECH['train2024_prev1_nan_rate_first_K_rows']}")

R["Q3_missing_policy"] = ("결측 행은 raw = NaN → script.py 의 기존 축 블록이 "
                          "_delta[_fin] 로 0.0 을 남긴다.  sentinel 충전 없음.")

# ── Q4 추론 시간 ─────────────────────────────────────────────────────
tblq = {int(p): float(v) for p, v in
        zip(a25.index.astype(int), np.linspace(0.05, 0.30, len(a25)))}
reps, ts_tab, ts_row = 5, [], []
for _ in range(reps):
    t1 = time.perf_counter()
    td = pd.read_csv(COV, encoding="utf-8-sig",
                     usecols=["pitcher_id", PREV_S[1]])
    pv = td["pitcher_id"].to_numpy()
    raw = (td[PREV_S[1]].to_numpy(np.float64)
           - np.array([tblq.get(int(p), np.nan) for p in pv], np.float64))
    fin = np.isfinite(raw)
    dl = np.zeros(len(raw)); dl[fin] = (raw[fin] - 0.01) / 0.09
    _ = 0.006 * dl
    ts_tab.append(time.perf_counter() - t1)
    t1 = time.perf_counter()
    td = pd.read_csv(COV, encoding="utf-8-sig",
                     usecols=[PREV_S[0], PREV_S[2]])
    raw = td[PREV_S[0]].to_numpy(np.float64) - td[PREV_S[2]].to_numpy(np.float64)
    fin = np.isfinite(raw)
    dl = np.zeros(len(raw)); dl[fin] = (raw[fin] - 0.0) / 0.1
    _ = 0.006 * dl
    ts_row.append(time.perf_counter() - t1)
R["Q4_inference_cost"] = {
    "frame_rows": int(len(pv)), "reps": reps,
    "seconds_anchor_axis_block": {"min": round(min(ts_tab), 3),
                                  "median": round(float(np.median(ts_tab)), 3)},
    "seconds_row_only_axis_block": {"min": round(min(ts_row), 3),
                                    "median": round(float(np.median(ts_row)), 3)},
    "baseline_G6_seconds_BN": 146,
    "answer": "무시할 수 있는 수준."}

# ── 🔴 AN 함정 점검 ──────────────────────────────────────────────────
v6_feats = [l.strip() for l in
            open(f"{PKG}/model_v6/features.txt", encoding="utf-8") if l.strip()]
qm = json.load(open(f"{PKG}/mate/model_q/meta.json", encoding="utf-8"))
q_feats = list(qm.get("features") or [])
q_base = list(qm.get("base_features") or [])
RAWCOLS = PREV_S + PREV_M + MIXC + ["asof_pitcher_pitchmix_n"]
AN = {
    "v6_n_features": len(v6_feats), "q_n_features": len(q_feats),
    "q_n_base_features": len(q_base),
    "raw_columns_our_axes_are_built_from": RAWCOLS,
    "raw_column_in_v6": {c: bool(c in v6_feats) for c in RAWCOLS},
    "raw_column_in_q": {c: bool(c in q_feats) for c in RAWCOLS},
    "ins_columns_in_q": sorted([f for f in q_feats if f.startswith("ins_")]),
    "ins_columns_in_v6": sorted([f for f in v6_feats if f.startswith("ins_")]),
    "our_new_axis_names": AXES,
    "name_level_collision": sorted(set(AXES) & (set(v6_feats) | set(q_feats))),
    "verdict_raw_layer": None,
    "what_is_actually_new": [
        "동결 앵커(직전 시즌 종료 커리어률) 와의 **차분** — 두 팔 어디에도 "
        "차분 형태로는 없다.  다만 Q 는 ins_p_r 와 ins_p_r_dev 를 둘 다 갖고 "
        "있어 prior = ins_p_r - ins_p_r_dev 로 **앵커를 선형으로 복원할 수 "
        "있다**(구종믹스는 제외 — ins 는 구종을 안 담는다).",
        "구종 3비율의 **엔트로피** — 비선형이고 어느 팔에도 없다.",
        "prev1 - prev5 **차분** — v6(신경망)는 선형결합을 만들 수 있고 "
        "Q(CatBoost)는 축 평행 분기라 한 분기로 못 만든다."],
    "structural_note": (
        "🔴 이번 스크린의 잔차 r0 = y - p_BN 은 **BN 블렌드 전체**(v6 를 "
        "w_v6=0.70, Q 계열을 S1/S2/T70 으로 포함) 에 대한 것이다. AN 때는 팔 "
        "단독 잔차에서 +41.13 이 나왔고 블렌드에 새것이 없어 진 것이었다. "
        "여기서는 v6·Q 가 이미 흡수한 정보면 t 가 애초에 죽는 형태로 드러난다. "
        "다만 이것이 '전이비가 양수다' 를 보장하지는 않는다(S2).")}
AN["verdict_raw_layer"] = (
    "🔴 raw 열은 두 팔이 전부 갖고 있다 — prev 6열은 v6·Q 둘 다, 구종 3비율도 "
    "v6·Q 둘 다, pitchmix_n 은 Q 가 갖고 있다. 따라서 이 실험이 물을 수 있는 "
    "것은 'raw 열이 새로운가' 가 아니라 '**차분·엔트로피라는 형태**가 새로운가' "
    "뿐이다."
    if all(AN["raw_column_in_v6"].get(c) or AN["raw_column_in_q"].get(c)
           for c in RAWCOLS)
    else "일부 raw 열은 두 팔 어디에도 없다 — 목록 확인 필요")

fr = pd.read_csv(TR2024, encoding="utf-8-sig", usecols=NEED)
e_tr = axis_finite(fr, a24)
lp_v6 = logit(np.clip(np.load(f"{H}/build_BN/work/tr2024_p_v19_BN_wv6_070.npy"),
                      EPS, 1 - EPS))
lp_aa = logit(np.clip(np.load(f"{H}/build_AY/work/pcal_aa.npy"), EPS, 1 - EPS))
pid = fr["pitcher_id"].astype(int).to_numpy()
A = anchor_cols_for(pid, a24)
g = lambda c: fr[c].to_numpy(np.float64)
VV = {"rg_prev1_suc_dev": g(PREV_S[0]) - A["anc_suc"],
      "rg_prev3_suc_dev": g(PREV_S[1]) - A["anc_suc"],
      "rg_prev5_suc_dev": g(PREV_S[2]) - A["anc_suc"],
      "rg_prev3_mid_dev": g(PREV_M[1]) - A["anc_mid"],
      "rg_trend": g(PREV_S[0]) - g(PREV_S[2]),
      "rg_fb_dev": g(MIXC[0]) - A["anc_fb"],
      "rg_br_dev": g(MIXC[1]) - A["anc_br"],
      "rg_off_dev": g(MIXC[2]) - A["anc_off"],
      "rg_mix_ent": entropy3(g(MIXC[0]), g(MIXC[1]), g(MIXC[2]))}
AN["measured_corr_axis_vs_arm_logit"] = {}
for c in AXES:
    f = np.isfinite(VV[c])
    AN["measured_corr_axis_vs_arm_logit"][c] = {
        "corr_vs_logit_v19_mate": round(float(np.corrcoef(VV[c][f], lp_v6[f])[0, 1]), 4),
        "corr_vs_logit_aa": round(float(np.corrcoef(VV[c][f], lp_aa[f])[0, 1]), 4),
        "n_finite_rows": int(f.sum())}
R["AN_trap_check"] = AN
print(f"\n=== 🔴 AN 함정 ===\n  {AN['verdict_raw_layer']}")
print(f"  v6 {len(v6_feats)}피처 · Q {len(q_feats)}피처 · Q ins 열 "
      f"{len(AN['ins_columns_in_q'])}개")
print("  축-팔 로짓 상관:")
for c in AXES:
    z_ = AN["measured_corr_axis_vs_arm_logit"][c]
    print(f"    {c:<20} v19 {z_['corr_vs_logit_v19_mate']:+.4f}  "
          f"aa {z_['corr_vs_logit_aa']:+.4f}")

os.makedirs(OUT, exist_ok=True)
R["wall_s_s0"] = round(time.time() - t_all, 1)
json.dump(R, open(f"{OUT}/stage0.json", "w"), indent=1, ensure_ascii=False)
print(f"\ns0 완료 {R['wall_s_s0']}s")

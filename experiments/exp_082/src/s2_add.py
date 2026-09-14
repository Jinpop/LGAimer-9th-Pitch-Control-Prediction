"""exp_082 보유 — S21 점수 영향 정량 · 정직면 전이비 보정 · 판정."""
import os, json, sys
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np, pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"; OUT = f"{H}/exp082"
K = 400309.19882517256; REF = 0.2498069
MU_MR, MU_WO = -0.45376277509804325, -2.0056142032773767
B, D_BO, D_NEW = -0.05, 0.02875748759679648, 0.028693042338963914
M = json.load(open(f"{OUT}/out/metrics.json", encoding="utf-8"))

# ── 1. S21 — 파서 선택이 최종 예측·점수에 주는 영향 (프레임별) ───────────
S21 = {}
FR = {"tr2024_cal": (f"{OUT}/runs/ft_matecal.csv", f"{H}/build_AC/work/tr2024/test.csv"),
      "cov": (f"{OUT}/runs/ft_bonly__cov.csv", f"{H}/build_BT/frames/cov/test.csv"),
      "band": (f"{OUT}/runs/ft_bonly__band.csv", f"{H}/build_BT/frames/band/test.csv"),
      "real5": (f"{OUT}/runs/ft_bonly__real5.csv", f"{H}/build_BT/frames/real5/test.csv")}
for fr, (ftp, tp) in FR.items():
    a = pd.read_csv(ftp, encoding="utf-8-sig")
    b = pd.read_csv(ftp, encoding="utf-8-sig", float_precision="round_trip")
    e = {}
    for col in ("lg_mr", "lg_wo"):
        d = np.abs(a[col].to_numpy(np.float64) - b[col].to_numpy(np.float64))
        e[col] = {"n_rows_differ": int(np.count_nonzero(d)),
                  "share": float(np.count_nonzero(d) / len(d)),
                  "max_abs_diff": float(d.max())}
    # 최종 예측에 미치는 상한: |b|*Δ(lg_mr) * max p(1-p)
    dmr = np.abs(a["lg_mr"].to_numpy(np.float64) - b["lg_mr"].to_numpy(np.float64))
    e["max_abs_dp_at_b_-0.05"] = float(abs(B) * dmr.max() * 0.25)
    e["max_abs_pts_bound"] = float(K * 2.0 * e["max_abs_dp_at_b_-0.05"] * 1.0)
    S21[fr] = e
M["S21_float_parser_per_frame"] = S21
M["S21_verdict"] = (
    f"팀원 script.py:291 은 pd.read_csv 기본 파서로 failtype.csv 를 읽는다(쓸 때는 %.17g). "
    f"round_trip 으로 다시 읽으면 lg_mr 이 tr2024 {S21['tr2024_cal']['lg_mr']['share']*100:.1f}% 행, "
    f"lg_wo 가 {S21['tr2024_cal']['lg_wo']['share']*100:.1f}% 행에서 1 ulp 어긋난다. "
    f"b=-0.05 에서 최종 예측에 주는 상한은 |Δp| ≤ {S21['tr2024_cal']['max_abs_dp_at_b_-0.05']:.3e} "
    f"(점수 상한 {S21['tr2024_cal']['max_abs_pts_bound']:.3e}점) — 규칙 문제도 점수 문제도 아니다. "
    "다만 **'바이트 동일'을 주장하는 재현 대조는 같은 파서를 태워야 성립한다**(S21). "
    "이번 게이트는 양쪽 다 round_trip 으로 읽어 max|Δlogit| 8.9e-16 을 냈다.")

# ── 2. 정직면 전이비 보정 (S25·S33 의 이 축 판) ───────────────────────
HS = M["stage2_honest_surface"]
g_b_honest_at_ship = HS["b_lg_mr"]["gain_at_shipped_b_-0.05"]
g_b_lb = 2.6766924977
ratio = g_b_lb / g_b_honest_at_ship
M["honest_surface_calibration"] = {
    "b_honest_gain_at_shipped_-0.05_pts": g_b_honest_at_ship,
    "b_LB_gain_at_same_point_pts": g_b_lb,
    "transfer_ratio_b": ratio,
    "c_honest_raw_gain_pts": HS["c_lg_wo"]["raw_gain_pts"],
    "c_honest_sd_matched_gain_pts": HS["c_lg_wo"]["sd_matched_own_gain_pts"],
    "c_implied_LB_gain_raw_pts": HS["c_lg_wo"]["raw_gain_pts"] * ratio,
    "c_implied_LB_gain_sdm_pts": HS["c_lg_wo"]["sd_matched_own_gain_pts"] * ratio,
    "🔴": ("정직면은 이 축 계열에 **오염돼 있다** — lg_mr·lg_wo 는 ≤2024 학습 Q 셀에서 "
           "나오는데 정직면 행이 바로 그 2024 행이다.  실패유형 셀은 라벨 정의에 직접 "
           f"닿는 양이라 b 의 정직면 이득이 같은 점에서 **{g_b_honest_at_ship:+.2f}점**인데 "
           f"실제 LB 는 **{g_b_lb:+.4f}점**이다 — **{1/ratio:.1f}배 부풀어 있다**.  "
           "따라서 c 의 정직면 원값·sd맞춤값은 그대로 읽으면 안 되고, 같은 면 위에서 "
           "LB 로 눈금이 잡힌 b 의 전이비로 환산해야 한다.  "
           "sd 맞춤은 이 오염을 **걷어내지 못한다**(척도가 아니라 정보 누출이다)."),
    "S25_S33_note": (f"c 의 sd 맞춤 이득 {HS['c_lg_wo']['sd_matched_own_gain_pts']:.5f} 은 "
                     f"원값 {HS['c_lg_wo']['raw_gain_pts']:.5f} 보다 오히려 크다"
                     f"(정보몫 {HS['c_lg_wo']['information_share']:.3f}) — 이 축의 이득은 "
                     "척도가 아니다(sd비 0.99989).  alpha 통로와 겹치지 않는다는 뜻이며, "
                     "크기 자체가 작다는 결론은 그대로다.")}

# ── 3. b 착륙표 ────────────────────────────────────────────────────────
LAND = {}
for fr in ("tr2024_cal", "cov", "band", "real5"):
    r = M["stage1_b_vertex"][f"{fr}/secant_G"]
    LAND[fr] = {"b_now": B, "b_star": r["b_star"], "delta_b": r["b_star"] - B,
                "remaining_pts": r["remaining_headroom_pts"],
                "predicted_LB_at_vertex": r["predicted_LB_at_vertex"],
                "K_G_bb_pts_at_u1": M["stage1_frames"][fr]["K_G_bb_pts_at_u1"]}
M["stage1_b_landing_table"] = LAND
M["stage1_primary_frame"] = (
    "tr2024_cal — core078 의 K·Q 가 장부값(BA·BC·BH·BI·BJ·BK·BN 7건)을 그대로 재현하는 "
    "유일한 프레임이라 역사 눈금과 같은 자다.  cov·band 는 배포형 대역이지만 ins_* 가 "
    "축퇴해 있고(S11), real5 는 실제 대회 행이지만 n=5 다.")

# ── 4. c 의 b-직교 잔여 곡률 ──────────────────────────────────────────
for fr in ("tr2024_cal", "cov", "band", "real5"):
    g = M["stage1_frames"][fr]
    M["stage2_c"][fr]["K_G_cc_after_orthogonalising_b_pts"] = float(
        K * (g["G_cc"] - g["G_bc"] ** 2 / g["G_bb"]))

# ── 5. 사전 등록 판정 ─────────────────────────────────────────────────
b_rem_max = M["stage1_frame_spread"]["remaining_max_pts"]
c_rho = M["stage2_rho"]["c_max_abs_rho"]
c_sdm = HS["c_lg_wo"]["sd_matched_own_gain_pts"]
A_b = bool(b_rem_max >= 0.5)
A_c = bool(c_rho <= 0.5 and c_sdm >= 2.0)
verdict = "A" if (A_b and A_c) else ("B" if (A_b or A_c) else "C")
M["preregistered_verdict"] = {
    "criterion_b_remaining_ge_0.5": {"value_max_over_frames": b_rem_max,
                                     "value_primary_frame": LAND["tr2024_cal"]["remaining_pts"],
                                     "threshold": 0.5, "PASS": A_b},
    "criterion_c_new_direction_and_honest_ge_2.0": {
        "max_abs_rho": c_rho, "rho_threshold": 0.5,
        "rho_partner_axis": M["stage2_rho"]["c_max_abs_rho_axis"],
        "sd_matched_honest_pts": c_sdm, "honest_threshold": 2.0, "PASS": A_c},
    "VERDICT": verdict,
    "S8": "문턱은 config 에 사전 등록된 값이며 결과를 보고 바꾸지 않았다."}
M["q_league_delta_NaN_note"] = (
    "rho 표의 q_league_* 4축이 NaN 인 것은 그 δ 가 정확히 0 벡터이기 때문이다 — "
    "S32 (대역 프레임에서 리그 백오프가 한 행도 발동하지 않는다) 의 재확인이고 이번 실험의 결함이 아니다.")
json.dump(M, open(f"{OUT}/out/metrics.json", "w"), indent=1, ensure_ascii=False)
print("판정", verdict, "| b 남은몫 max", round(b_rem_max, 5), "| c |ρ|", round(c_rho, 4),
      "| c 정직면 sd맞춤", round(c_sdm, 5), "| 전이비", ratio)
print("c 환산 LB 기대치 raw", M["honest_surface_calibration"]["c_implied_LB_gain_raw_pts"],
      "sdm", M["honest_surface_calibration"]["c_implied_LB_gain_sdm_pts"])
print(json.dumps(S21["tr2024_cal"], ensure_ascii=False))

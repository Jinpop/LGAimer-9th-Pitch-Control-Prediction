#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_084 metrics.json 조립 — 세 stage 산출물을 표준 형식으로 합친다."""
import json, time
H = "/home/<user>/SungjinPark"; OUT = f"{H}/exp084/out"
S1 = json.load(open(f"{OUT}/s1_missrate.json", encoding="utf-8"))
S2 = json.load(open(f"{OUT}/s2_curv.json", encoding="utf-8"))
S3 = json.load(open(f"{OUT}/s3_solve.json", encoding="utf-8"))
PS = S1["per_season"]

exl = S3["extrapolation_2025"]["fits"]
half = (exl["miss_frac_all"]["pi95"][1] - exl["miss_frac_all"]["pi95"][0]) / 2

M = {
 "exp_id": "exp_084",
 "date": time.strftime("%Y-%m-%d %H:%M:%S"),
 "title": "q_league 2|2 축의 곡률 불확실성 — 미적중률 추세 · G 법칙 · 이동폭 재선택",
 "학습_건수": 0, "제출물": "없음", "train_time_min": 0.0,
 "wall_sec_total": S1["wall_sec"] + S2["wall_sec"] + S3["wall_sec"],
 "K_pts_per_brier": 400309.19882517256,
 "🔴_판정": "B",
 "판정_근거": {
   "사전등록": {"A": "2025 외삽 미적중률 폭 ±5%p 안 → 권고 Δ 확정·BW 를 그 값으로",
              "B": "±5~15%p → Δ 0.04 유지", "C": "그보다 크거나 추세 불성립 → Δ 0.04 유지"},
   "측정된_폭_95%예측구간_반폭_pp": 100 * half,
   "band": "±5~15%p → B",
   "기울기_유의성": {"miss_all_r2": exl["miss_frac_all"]["r2"],
                "miss_all_t": exl["miss_frac_all"]["t_slope"],
                "miss_all_p": exl["miss_frac_all"]["p_slope_two_sided"],
                "f22_t": exl["f22_fire_frac"]["t_slope"],
                "f22_p": exl["f22_fire_frac"]["p_slope_two_sided"],
                "note": ("기울기가 5% 에서 유의하지 않다 — 이것만 보면 C 다. "
                         "B 와 C 의 지시 행동이 동일(Δ 0.04 유지)하므로 판정 모호가 "
                         "행동에 닿지 않는다.  S8 — 결과를 보고 문턱을 바꾸지 않았다.")},
   "행동": "BW 이동폭 Δ = 0.04 유지"},

 "stage1_미적중률": {
   "표_생성_규약": ("출하 mate/model_q/meta.json offset.key_cols=[pitcher_id,batter_hand] · "
                "lookup_d 생성 = build_BR/src/solve_BR.py build_tables · "
                "exp_004 마스크 season<=max & ~(season<=2022 & game_type=='F') & label notna"),
   "대조_le2023표_2024행": S1["control_le2023_to_2024"],
   "연도별": {s: {"table": f"<={e['table_max_season']}", "n_table_keys": e["n_table_keys"],
                "n_rows": e["n_rows"], "miss_frac": e["miss_frac"],
                "cell_22_share": e["per_cell"]["2|2"]["cell_share_of_frame"],
                "cell_22_miss_within": e["per_cell"]["2|2"]["miss_frac_within_cell"],
                "f22_fire_frac_of_frame": e["per_cell"]["2|2"]["fire_frac_of_frame"]}
             for s, e in PS.items()},
   "추세_2025": {"miss_all": exl["miss_frac_all"], "f22_fire": exl["f22_fire_frac"],
               "cell_share_22": exl["cell_share_22"], "miss_within_22": exl["miss_within_22"],
               "f22_via_product": exl["f22_via_product"]},
   "real5": S3["real5"]},

 "stage2_곡률법칙": {
   "기준면": S2["frame"],
   "자체증명": {"build_q_reproduces_p_BO_max_abs": S2["G0a_build_q_reproduces_p_BO"],
             "d_m_bitwise": S2["G0b_d_m_bitwise"], "lgq_bitwise": S2["G0c_lgq_bitwise"]},
   "대조_diag_qleague": S2["control_vs_diag_qleague"],
   "실측_le2023프레임": {c: S2["frame_le2023_actual"]["per_cell"][c]
                    for c in ["1|1", "1|2", "2|1", "2|2"]},
   "반대칭_le2023프레임": S2["frame_le2023_actual"]["antisym_ph2"],
   "합성프레임": [{"target_m": e["target_m"], "miss_frac": e["miss_frac"],
               "f22_fire": e["per_cell"]["2|2"]["fire_frac_of_frame"],
               "K_G_22": e["per_cell"]["2|2"]["K_G_jj"],
               "K_G_21": e["per_cell"]["2|1"]["K_G_jj"],
               "K_G_antisym": e["antisym_ph2"]["K_G_jj"]} for e in S2["synthetic_frames"]],
   "🔑_법칙": {"form": "K·G_jj = A · f · (1 − f),  f = 그 셀의 발동 행 비중(프레임 대비)",
            "왜_(1−f)": ("d_adj 평균 못박힘이 Σ Δp = 0 을 강제하므로 δ = s·(1_S − f) 가 되고 "
                        "E[δ²] = s²·f(1−f).  CV_LB G11 의 ρ 닫힌식과 같은 기전."),
            "A": {c: S3["curvature_law"][c]["A_f_times_1_minus_f"]
                  for c in ["1|1", "1|2", "2|1", "2|2"]},
            "A_antisym": S3["curvature_law"]["antisym_ph2"]["A_phi"],
            "max_rel_resid": {c: S3["curvature_law"][c]["max_rel_resid"]
                              for c in ["1|1", "1|2", "2|1", "2|2"]},
            "max_rel_resid_antisym": S3["curvature_law"]["antisym_ph2"]["max_rel_resid"],
            "순수비례_max_rel_resid": {c: S3["curvature_law"][c]["pure_proportional_max_rel_resid"]
                                  for c in ["1|1", "1|2", "2|1", "2|2"]},
            "판독": ("f 1.58%~36.45% 를 훑어도 최대 상대잔차 0.63%(2|2).  순수 비례(A·f)는 "
                    "28% 어긋난다.  → 외삽 f 하나에서 G 를 바로 계산할 수 있다."),
            "antisym_phi": S3["curvature_law"]["antisym_ph2"]["phi_form"]}},

 "stage3_이동폭": {
   "LB_두_점": S3["LB_points"],
   "꼭지점_기계": "gain(u) = 2K·u·L − K·u²·G,  Q0 = K·u0²·G,  L = (g0+Q0)/(2K u0),  u* = L/G",
   "곡률_시나리오": S3["curvature_scenarios"],
   "이동폭별_예상이득": S3["delta_table"],
   "미니맥스": S3["minimax_over_scenarios"],
   "비선형": S3["nonlinearity"],
   "🔴_권고": {
     "권고_Δ": 0.05,
     "권고_근거": ("증거가 지지하는 c(=G/G_frame) 범위 {추세 0.6445, 프레임 0.9937, "
                "real5 2.5256} 에서 최악이 가장 높은 Δ = 0.0530(최악 +4.954).  "
                "0.05 의 최악은 +4.937 로 사실상 같다."),
     "🔴_행동": ("사전 등록 판정이 B 이므로 **BW 이동폭은 0.04 를 유지한다**.  "
                "0.04 의 최악은 +4.653 으로 0.05 대비 −0.284 점.  이 0.284 를 "
                "얻으려고 사전 등록 문턱을 옮기지 않는다 (S8)."),
     "config_가정_대비": ("계획서는 c ∈ {1,2,4} 를 가정해 Δ=0.04 의 이득을 "
                      "+6.03/+5.13/+3.33 로 봤다.  측정된 범위는 c ∈ [0.64, 2.53] 이고 "
                      "같은 Δ 의 이득은 +6.35/+6.04/+4.65 다.  **폭이 1.81배에서 "
                      "1.37배로 줄고 바닥이 +3.33 에서 +4.65 로 올라갔다.**"),
     "최선_점추정": ("최근 3년 평균 f22 = 0.066487 → c = 0.9797.  즉 **우리가 이미 쓰던 "
                 "프레임 G 가 최선의 점추정**이고, 추세는 아래로 real5 는 위로 민다.")}},

 "stage4_반대칭축": {
   "구조": ("league_dev 는 ph 안에서 표본수 가중 합이 0 이다 → ph 마다 자유도 1.  "
          "반대칭 축: ΔL(2|2) = +u, ΔL(2|1) = −u·n22/n21"),
   "w_n22_over_n21": S2["frame_le2023_actual"]["antisym_ph2"]["weight_ratio_n22_over_n21"],
   "결과": {k: v for k, v in S3["stage4_antisym"].items() if not k.startswith("🔴")},
   "🔴_zero_sum_산문": S3["stage4_antisym"]["🔴_zero_sum_prose"],
   "판정": ("**반대칭 축은 2|2 단독보다 나쁘다.**  최대이득이 세 시나리오 전부에서 낮고"
          "(프레임 6.27 vs 11.52 · 추세 9.93 vs 17.40 · real5 2.11 vs 4.95), "
          "Δ=0.04 에서도 2/3 에서 진다.  L 은 11.9% 크지만 G 가 2.30배라 "
          "u* 가 절반으로 잘린다.  지휘자 어림 +1.44 vs +1.32 는 **BS4 진폭(0.0076)에서만** "
          "맞고(재현 +1.427 vs +1.318), 그 진폭에서는 2차항이 아직 안 무는 구간이다.  "
          "교차점 Δ = 0.0236(프레임) 위로는 단독이 이긴다.")},

 "stage5_교차항": dict(S2["stage5_cross"], 판정=(
   "|ρ| 전부 0.3 아래 — b 축 −0.0140, c 축 +0.1580, 반대칭 대비 −0.0215.  "
   "**팀원 실패유형 축 위에 얹을 때 단순 합산이 안전하다** (S16 문턱 미달).  "
   "2축 동시 해 불필요.")),

 "특이사항": [
   "🔴 real5 의 개인 키는 5개가 아니라 **4개**다 — TEST_005332 와 TEST_035185 가 같은 "
   "`24713|2`.  독립 표본은 4이고 그중 3이 미적중이다.  P(≥4/5 | p=0.215) = 0.88%, "
   "P(≥3/4 | p=0.215) = 3.33% — 어느 쪽이든 5% 아래라 프레임과 real5 는 통계적으로 "
   "부딪힌다.  다만 real5 의 `2|2` 발동은 **단 1행**이라 c=2.53 시나리오 전체가 그 한 "
   "행에 얹혀 있다.",
   "🔴 2025 의 f22 상한 시나리오조차 c = 2.53 이고 계획서가 상정한 4배는 **전 행 발동"
   "(f22 = 0.3645)** 이라야 나온다 — 개인 표가 1,392키인데 2025 에 그럴 수 없다.",
   "⚠️ BS3 의 LB 델타를 소수점 2자리(−0.19)로만 받았다.  반대칭 축의 L_anti 는 "
   "±0.02 밴드에서 ±0.35% 만 움직여 판정을 안 뒤집는다(민감도 기록됨).",
   "⚠️ 미적중률 시계열은 단조가 아니다 — 2023 16.49% 에서 2024 21.50% 로 되튀었다. "
   "표가 커지는데도 되튄 것이라 '표 누적 = 미적중 감소' 이야기가 완전하지 않다.",
   "⚠️ 합성 프레임에서 미적중률을 낮출 때, 미적중 → 적중 전환 행의 d_s 를 0 으로 뒀다"
   "(빈 셀의 축소 목표값).  실제 저미적중 세계에서는 작은 셀의 d_s 가 붙었을 것이다. "
   "f ≤ 0.215 구간의 법칙 적합에만 영향이고 잔차 0.63% 안에 들어 있다.",
   "🟢 numpy corrcoef RuntimeWarning 1건 — m=0 합성 프레임에서 δ ≡ 0 이라 상관이 정의되지 "
   "않은 것.  그 프레임은 법칙 적합에서 제외된다(f>0 조건)."],

 "산출물": {
   "server": ["~/SungjinPark/exp084/src/{s1_missrate,s2_curv,s3_solve}.py",
              "~/SungjinPark/exp084/out/{s1_missrate,s2_curv,s3_solve}.json",
              "~/SungjinPark/exp084/out/delta_{q22,q21,b}_le2023.npy",
              "~/SungjinPark/exp084/logs/{s1,s2}.log"],
   "note": "학습 0건 · 모델 가중치 없음 · 어떤 기존 파일도 지우거나 덮어쓰지 않았다"},
}
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print("판정", M["🔴_판정"], "· 폭 ±%.2f%%p" % M["판정_근거"]["측정된_폭_95%예측구간_반폭_pp"])
print("METRICS_OK")

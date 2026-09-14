"""exp_083 최종 metrics.json 조립 — 갈래별 소명 + 사전 등록 판정."""
import json, os
H = "/home/<user>/SungjinPark"
O = f"{H}/exp083"
A = json.load(open(f"{O}/out/analyze_cal.json", encoding="utf-8"))
C = json.load(open(f"{O}/out/ctrl_cal.json", encoding="utf-8"))
ADD = json.load(open(f"{O}/out/add.json", encoding="utf-8"))
XFER = 0.05153
RHO_GATE, PTS_GATE = 0.5, 1.0

BRANCH = {
 "A_우리_AA팔_셀": {
  "이미_계산되는가": True,
  "줄번호": ["aa_infer.py:653-654  for name, model in zip(stack_feats, cells): "
             "F[name] = predict_prob(model, F_base, cat_idx_base)",
             "aa_infer.py:894-895  stack_features / cell_model_order",
             "model/meta.json branches.R.cell_model_order 및 branches.F.cell_model_order"],
  "🔑_파트너_Q_셀과_같은_4종인가": True,
  "근거": ("우리 meta 의 cell_model_order 가 R·F 두 분기 모두 "
           "['is_mid_only','is_rev_only','is_both','is_wayoff_true'] 로 파트너 Q 와 "
           "★이름·개수·순서까지 동일★.  다른 모델(CatBoost 67열 스태커, R 18시드 / F 3시드)로 "
           "추정한 같은 신호다."),
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": ("3파일 — aa_infer.py(predict_one_seed 반환에 셀 dict 추가 / "
                        "run_branch 에서 시드별 수집 / run 반환에 2열 추가, 약 25줄), "
                        "script.py(로짓 2개 조립 후 최종 블렌드 로짓에 가산, 약 20줄), "
                        "blend.json(상수 블록).  팀원 판례가 4파일인 것은 파트너 팔이 "
                        "자식 인터프리터라 CSV 를 경유해야 하기 때문이고, AA 는 루트 프로세스 "
                        "안이라 그 한 파일이 필요 없다."),
  "규칙2_4": "통과 — 시드 방향 합만 있고 행 방향 집계가 없다 (average_over_seeds 와 같은 논증)",
  "판정": "기각 — 새 방향이 아니다",
  "기각_이유": ("δ 상관이 팀원이 ★이미 산★ 축과 거의 1 이다: "
              "ρ(A_aa_lg_mr, teammate_b) = 0.9549 · ρ(A_aa_lg_wo, teammate_c) = 0.9786. "
              "사전 등록 문턱 |ρ| ≤ 0.5 를 크게 넘는다.  "
              "직교 잔여를 따로 만든 불일치 축(A_aa_minus_q_mr·_wo)은 ρ 를 통과하지만(0.3785·0.1454) "
              "정직면 이득이 +0.7182·+0.7821 로 순열 귀무의 최대값(+0.9076, 30회) ★안★ 이다 "
              "→ 두 모델이 어긋나는 성분에도 신호가 없다.  이것이 1순위 질문에 대한 답이다."),
  "🔴_정직면_원값을_그대로_읽으면_안_되는_이유": (
      "A_aa_lg_mr 의 정직면 원값은 +179.62 로 팀원 b(+51.94)의 3.46배다.  그러나 대조 실측: "
      "우리 AA 팔의 ★자기 stage-2 로짓★(배포 18시드)을 같은 축으로 넣으면 ★+403.43★, 정직면이 c_aa 자리에 "
      "실제로 쓰는 벡터(exp_049 A1 3시드)를 넣으면 +3.14 다.  c_aa 항을 배포 18시드로 갈아끼우는 "
      "것만으로 +281.98 이 나온다.  즉 surface079 의 AA 항이 배포보다 "
      "훨씬 약한 대역이고, AA 계열 방향이면 무엇이든 크게 벌어 보인다.  "
      "전이비 0.05153 은 ★같은 면에서★ LB 로 눈금이 잡힌 값이므로 이 결함까지 흡수한다 "
      "(S54).  그래서 판정은 환산값으로 한다."),
 },
 "B_EBM_피처별_기여": {
  "이미_계산되는가": True,
  "줄번호": ["mate/ebm_core.py:110  for feats, ts in spec['terms']:",
             "mate/ebm_core.py:114  z += ts[np.clip(bi, 0, ts.shape[0]-1)]   (1차항)",
             "mate/ebm_core.py:118  z += ts[b0, b1]                          (2차 교호항)"],
  "항_수": 59, "시드_수": 3,
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": ("4파일 — ebm_core.py(score_numpy 가 항별 배열도 반환, 약 6줄), "
                        "infer_ebm6.py(전달), mate/script.py(CSV 로 내보내기), "
                        "script.py + blend.json(소비).  가산 모델이라 분해가 ★정확★하다."),
  "규칙2_4": "통과 — 행별 bin 조회, 다른 행 미참조",
  "판정": "기각",
  "기각_이유": ("상위 6항 중 ρ 문턱을 통과한 것은 3개고(ins_p_n 0.2177 · ins_b_success 0.2111 · "
              "ins_p_reverse 0.6665 과 game_type 0.5169 은 초과), "
              "통과분의 환산 이득이 +0.0005 · +0.1625 로 문턱 +1.0 은 물론 잡음바닥급이다. "
              "이득이 큰 항(ins_p_success_dev +4.33)은 ρ 0.9197 로 beta_ins 축 그 자체다 — "
              "EBM 의 ins 항 기여는 우리가 이미 beta_ins 로 사고 있는 방향이다."),
 },
 "C_v6_Leaf_경로": {
  "이미_계산되는가": True,
  "줄번호": ["mate/nn_infer.py:72  L = booster.predict(X.iloc[s:e], pred_leaf=True)",
             "mate/nn_infer.py:77  ok = col >= 0            (잎 어휘 미등재 마스크)",
             "mate/nn_infer.py:78  z[ok] += coef[offs[j] + col[ok]]  (LR 부분합)"],
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": "3파일 (nn_infer.py 반환 + v6_infer.py 전달 + mate/script.py 내보내기)",
  "규칙2_4": "통과",
  "판정": "기각",
  "기각_이유": ("① 잎 인덱스의 OOV(어휘 미등재) 신호는 ★존재하지 않는다★ — 나무가 50그루뿐이고 "
              "253,507행 전부 모든 그루에서 학습 때 본 잎에 떨어진다(leaf_oov 최대 0.0, sd 0). "
              "δ 가 항등적으로 0 이라 측정 자체가 불가능하다. "
              "② LR 부분합(pos/neg/absmass)은 잎 팔의 로짓을 분해한 것이라 기존 축과 겹친다 — "
              "최대 |ρ| pos 0.6315(alpha) · neg 0.7938(w_out) · absmass 0.5059(off_BK) — 전부 문턱 초과. "
              "환산 이득은 neg +1.6852 · pos +1.2554 로 점수 문턱은 넘지만 ρ 에서 떨어진다 — "
              "★사고 싶어도 이미 산 방향★ 이다."),
 },
 "D_v6_NN_은닉": {
  "이미_계산되는가": True,
  "줄번호": ["mate/nn_infer.py:49  z = lin(z, 8); z = _relu(z)   ← 마지막 은닉 64차원",
             "mate/nn_infer.py:50  z = lin(z, 10)                (이 뒤로 은닉은 버려진다)"],
  "차원": 64, "시드_수": 4,
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": ("주성분을 쓰면 4파일 + ★train 에서 동결한 회전 행렬 파일 1개★. "
                        "활성 개수(nact)를 쓰면 회전 행렬이 필요 없어 3파일이고 한 줄이다 "
                        "((z > 0).sum(1))."),
  "규칙2_4": ("주성분: ⚠️ ★조건부★ — 이번 측정의 회전은 이 프레임에서 SVD 로 잡았고 그것은 "
             "행 방향 집계다.  출하하려면 회전 행렬을 train 에서 동결해 패키지에 실어야 한다. "
             "nact: 통과 — 행 i 의 자기 64값에 대한 행 내 축약이라 회전이 없다."),
  "판정": "★생존 — D_nn_hidden_nact★ (이 실험의 유일한 생존 축)",
  "🔑_nact_정확한_소명": ("은닉 64차원 벡터는 nn_infer.py:49 에서 ★이미 계산된다★. "
     "활성 개수는 그 위의 새 축약 한 줄((z > 0).sum(1))이고 부스터 호출 증가분은 0 이다. "
     "행 i 의 값이 행 i 자신의 64개 은닉값만으로 정해지므로 규칙 2-4 를 만족한다. "
     "산 축 대비 max|ρ| 0.3853(c_S1) · 형제 축 대비 0.4611(pc3) · "
     "정직면 +27.0519 는 순열 귀무(60회 평균 +0.3061 sd 0.4921 max +2.4199) 대비 z = 54.3 · p < 1/60. "
     "환산 +1.3940 으로 두 문턱을 모두 통과한 유일한 축이다."),
  "비고": ("pc1 은 분산의 66.5% 를 먹고 사실상 NN 로짓 방향이라 이득이 작다(+2.17). "
          "pc2 는 이득이 크지만(+89.14) ρ 0.8713 로 w_out 축이다. "
          "pc3 는 ρ 0.4801 통과이나 환산 +0.3987 로 문턱 미달."),
 },
 "E_시드_산포": {
  "이미_계산되는가": True,
  "줄번호": ["aa_infer.py:956,969  per_seed 리스트 축적",
             "aa_infer.py:971      p = average_over_seeds(per_seed)   ← 여기서 산포가 버려진다",
             "mate/v6_infer.py:138(lgb) :147-152(cat) :156-159(nn) :163-167(leaf)",
             "mate/q_infer.py      per_seed (팀원이 이미 셀을 꺼낸 바로 그 리스트)"],
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": "3파일 · 약 8줄 (리스트를 np.stack 해 std, 반환에 1열 추가)",
  "규칙2_4": ("통과 — 행 i 의 sd 는 행 i 의 18개 값만 쓴다(average_over_seeds 와 정확히 같은 논증). "
             "🔴 단 aa_infer.py:977-979 가 ★시드 산포 진단을 일부러 인쇄하지 않는다★고 명시하고 "
             "그 위에 규칙 2-4 논증을 세워 뒀다.  값을 행별로 쓰는 것은 무방하나 "
             "★어떤 통계도 인쇄하면 안 된다★ — 행 수만 찍을 것."),
  "판정": "차점 — ρ 만 통과, 점수 문턱 미달 (B 티어)",
  "비고": ("E_aa_seed_sd18(18시드 AA stage-2 로짓의 행별 sd)이 이 계열의 최대다. "
          "산 축 대비 max|ρ| ★0.2916★(c_T70) 로 이번 조사에서 가장 직교하는 방향이고, "
          "정직면 원값 +16.3613 은 순열 귀무(60회 평균 +0.3597 sd 0.4572 max +2.6039) 대비 "
          "z = 35.0 · p < 1/60 로 확실한 신호다.  그러나 환산값 +0.8431 이 문턱 +1.0 에 미달한다. "
          "🔴 앵커 수정 전에는 +1.06 이라 문턱을 넘었다 — 배포 앵커로 잘못 잡은 1차 캡처에서 "
          "ins 16열이 축퇴해 시드 산포가 과대평가된 것이었다. "
          "파트너 v6 쪽 시드 산포는 "
          "시드가 3~4개뿐이라 추정이 거칠고 전부 환산 +0.2 미만."),
 },
 "F_클립_카운터": {
  "이미_계산되는가": True,
  "줄번호": ["aa_infer.py:1140-1142  d_p, hit = lookup_table_values(...) (개인 셀 적중 마스크)",
             "aa_infer.py:1153,1162  L, has_L = lookup_table_values(...) (리그 백오프 마스크)",
             "aa_infer.py:1181-1182  raw_off / n_clipped 마스크",
             "script.py:257-258      n_clip_p · n_clip_aa",
             "script.py:421          n_moved"],
  "부스터_호출_증가분": 0,
  "꺼내는_코드_변경_규모": "1~2파일 · 각 3줄 미만 (가장 싸다)",
  "규칙2_4": "통과 (행별 조회 결과)",
  "판정": "기각 — 이 프레임에서 측정 불가, 그리고 새 방향일 가망도 낮다",
  "기각_이유": ("① tr2024 보정 프레임에서 세 마스크가 ★전부 상수 1★ 이다(개인 셀 적중률 "
              "100.00% = 253,507/253,507 — S32/S47 이 말한 바로 그 축퇴).  clip 마스크도 0행. "
              "leaf OOV 도 0.  δ 가 항등적으로 0 이라 rms(δ)·K·G·ρ 를 낼 수 없다. "
              "② 재측정하려면 ≤2023 표를 2024 행에 적용한 프레임이 필요한데(S32, miss 21.50%), "
              "그 마스크는 정확히 q_league 백오프 4축의 받침이다 — 이미 산 축이라 "
              "새 자유도가 아닐 공산이 크다(S37)."),
 },
}

AXES = A["axes"]; HS = A["honest_surface"]
ROWS = {}
for k in AXES:
    raw = HS[k]["raw_gain_pts"]
    ROWS[k] = {
        "rms_delta": AXES[k]["rms_delta"], "K_G_jj_pts_at_u1": AXES[k]["K_G_jj_pts_at_u1"],
        "amp_KQ1": AXES[k]["amp_KQ1"], "amp_KQ1_times_0p2_S42": AXES[k]["amp_KQ1_times_0p2"],
        "K_Q_at_0p2x_pts": AXES[k]["K_Q_at_0p2x_pts"],
        "max_abs_rho_existing": AXES[k]["max_abs_rho"],
        "max_abs_rho_axis": AXES[k]["max_abs_rho_axis"],
        "max_abs_rho_sibling_exp083": AXES[k].get("max_abs_rho_sibling_exp083"),
        "max_abs_rho_sibling_axis": AXES[k].get("max_abs_rho_sibling_axis"),
        "rho_top": AXES[k]["rho_top5"],
        "honest_raw_pts": raw,
        "honest_xfer_0p05153_pts": raw * XFER,
        "rho_gate_pass": bool(AXES[k]["max_abs_rho"] <= RHO_GATE),
        "pts_gate_pass": bool(raw * XFER >= PTS_GATE),
        "SURVIVES": bool(AXES[k]["max_abs_rho"] <= RHO_GATE and raw * XFER >= PTS_GATE)}
SURV = {k: v for k, v in ROWS.items() if v["SURVIVES"]}
VERD = "A" if SURV else ("B" if any(r["rho_gate_pass"] and r["honest_xfer_0p05153_pts"] >= 0.4
                                    for r in ROWS.values()) else "C")

M = {
 "exp_id": "exp_083",
 "title": "이미 계산되고 버려지는 중간값 전수 조사 — 팀원 +2.68 의 일반형",
 "date": "2026-08-22", "학습_건수": 0, "제출물": 0,
 "사전_등록_판정": VERD,
 "사전_등록_문턱": {"rho_max": RHO_GATE, "xfer_pts_min": PTS_GATE,
                 "🔴": "결과를 보고 바꾸지 않았다 (S8).  두 문턱 모두 config.yaml 사전 등록값."},
 "생존_축": {k: {"max_abs_rho": v["max_abs_rho_existing"],
               "honest_raw_pts": v["honest_raw_pts"],
               "honest_xfer_pts": v["honest_xfer_0p05153_pts"],
               "amp_KQ1": v["amp_KQ1"], "amp_KQ1_times_0p2": v["amp_KQ1_times_0p2_S42"]}
             for k, v in SURV.items()},
 "갈래별_소명": BRANCH,
 "축별_측정": ROWS,
 "축퇴로_측정_불가": A["axes_dropped_degenerate"],
 "정직면": {"base_pts": A["honest_surface_base_pts"],
           "module": "exp_079/src/surface079.py (≤2023 학습 팔 → 2024 253,507행)",
           "전이비_S54": XFER,
           "🔴": ("정직면 원값은 이 계열에서 LB 로 읽으면 안 된다.  팀원 b 축 실측 "
                 "정직면 +51.94414 → LB +2.6766924977 (전이비 0.05153).  "
                 "여기 모든 판정은 환산값으로 했다.")},
 "귀무_순열": C["permutation_null"],
 "생존_후보_귀무_60회_및_산축단독_rho": ADD,
 "🔴_rho_기준_정의": ("사전 등록된 '기존 축'은 exp_078 15축 + q_beta + q_league 4 + 셀 8 + "
                  "팀원 b·c 다.  exp_083 이 이번에 새로 만든 형제 축은 그 목록에 없으므로 "
                  "게이트에 넣지 않는다(문턱 0.5 는 그대로 — S8).  다만 형제 축과의 ρ 도 "
                  "max_abs_rho_sibling_exp083 로 전부 기록했다: 두 축 이상을 함께 실을 때 "
                  "naive 합산이 위험하기 때문이다(S16)."),
 "귀속_대조": C["attribution_controls"],
 "정직면_AA팔_교체_대조": C["surface_with_deployed_AA_arm"],
 "게이트": {
   "AA_캡처_항등앵커": {
     "what": "monkeypatch 캡처 실행의 stage-3 출력이 저장된 pcal_aa 와 같은가",
     "p_off_vs_build_AY/work/pcal_aa.npy": {"ndiff": 0, "max_abs": 0.0, "PASS": True},
     "smooth_S2/S1/T70_vs_pcal_*": {"ndiff": 0, "max_abs": 0.0, "PASS": True},
     "의미": "감싼 함수가 원본 반환값을 손대지 않았음이 비트 단위로 증명된다"},
   "v6_캡처_재현": {
     "what": "복제한 4팔 블렌드가 exp_078 저장본 tr2024_v6_pv.npy 와 같은가",
     "ndiff": A["capture_gates"]["v6_replication_ndiff"],
     "max_abs": A["capture_gates"]["v6_replication_max_abs"],
     "PASS": A["capture_gates"]["v6_replication_ndiff"] == 0},
   "행_순서": C["row_order_gate"],
   "AA_부스터_호출_수": A["capture_gates"]["n_booster_calls_aa"]},
 "🔴_앵커_사고_기록": {
   "무슨_일": ("1차 캡처는 AA 모델의 ins 앵커를 ★배포 앵커(7b96e3de…)★ 로 두고 2024 행에 "
             "적용했다.  그러면 ins_p_n = asof_n − 2024종료누적 ≤ 0 이 되어 ins 16열이 "
             "전부 축퇴한다(BO blend.json 이 이미 경고한 그것).  저장된 pcal_aa 와 "
             "253,338행이 어긋나 발견했다."),
   "고침": ("model 을 build_AY/work/aa_AY/model_cal 로 바꿔 재캡처 "
           "(ins 앵커 0e4b586d… = lt2024 보정 앵커).  그러자 p_off 가 pcal_aa 와 "
           "★비트 동일★ 이 됐다."),
   "교훈": ("‘저장된 벡터와 대조되지 않는 캡처는 캡처가 아니다.’  "
           "1차 결과는 폐기하지 않고 out/cap_aa.npz 로 남겨 뒀다(파일 삭제 없음).")},
 "프레임_한계": [
   "S49/S52 — K·G_jj 와 K·Q=1 진폭은 프레임 조건부다.  여기 수치는 tr2024 보정 프레임 단독.",
   ("S56 — 생존 두 축(AA 시드 sd · NN 활성 개수)은 개인 셀 표·백오프를 타지 않으므로 "
    "프레임 폭이 q_league 계열(15배)보다 훨씬 작을 것으로 본다.  다만 ★측정하지 않았다★ — "
    "빌드 전에 cov/band/real5 에서 재측정할 것."),
   "S16 — 두 생존 축을 함께 실으면 프로브 u 를 naive 로 더하지 말고 G⁻¹L 로 재적합할 것.",
   "S42 — 매수 진폭은 K·Q=1 의 0.2배."],
 "runs": {"cap_aa(배포앵커·폐기)": "out/cap_aa.npz",
          "cap_aa(보정앵커·채택)": "out/cap_aa_cal.npz",
          "cap_mate": "out/cap_mate.npz",
          "analyze": "out/analyze_cal.json", "ctrl": "out/ctrl_cal.json"},
 "wall_sec": {"analyze": A["wall_sec"], "ctrl": C["wall_sec"]},
 "K_pts_per_brier": A["K_pts_per_brier"], "ref_brier": A["ref_brier"],
 "n_rows": A["n_rows"], "threads": 6,
}
json.dump(M, open(f"{O}/metrics.json", "w"), indent=1, ensure_ascii=False)
print("판정", VERD, "· 생존", list(SURV))
for k, v in sorted(ROWS.items(), key=lambda kv: -kv[1]["honest_xfer_0p05153_pts"]):
    print(f"{'✔' if v['SURVIVES'] else ' '} {k:<32} |ρ| {v['max_abs_rho_existing']:.4f} "
          f"raw {v['honest_raw_pts']:+9.4f}  환산 {v['honest_xfer_0p05153_pts']:+8.4f}  "
          f"K·G {v['K_G_jj_pts_at_u1']:9.2f}  KQ1 {v['amp_KQ1']:.5g}")
print("->", f"{O}/metrics.json")

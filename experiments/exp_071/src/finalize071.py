"""exp_071 산출물 조립 — metrics.json (+ report 키 + 벽시계)."""
import json, os, time
H = "/home/<user>/SungjinPark"; E = f"{H}/exp071"; OUT = f"{E}/out"
S01 = json.load(open(f"{OUT}/stage01.json"))
TR = json.load(open(f"{OUT}/train071.json"))
S3 = json.load(open(f"{OUT}/stage3.json"))
A3, A4 = S3["점추정"]["3종판"], S3["점추정"]["4팔완성판"]
BS = S3["부트스트랩"]
M = {
 "exp_id": "exp_071",
 "type": "팀원 배포 NN(159차원) 구조 재현 → p_nn_oot2024 생성 → p_v6 4팔 완성 → w_v6 1축 곡선 재실행",
 "제출물": "없음 (판정·채택·제출 결정하지 않음)",
 "stage0_게이트": {
   "G0a_numpy독립순전파_vs_배포추론기_maxabs": S01["G0a_numpy_vs_deploy_maxabs"],
   "G0a": S01["G0a_verdict"],
   "G0b_torch_DeployNN_vs_배포추론기_maxabs": S01["G0b_torch_vs_deploy_maxabs"],
   "G0b": S01["G0b_verdict"],
   "G0c_export_npz_왕복_maxabs": S01["G0c_export_roundtrip_maxabs"],
   "G0c": S01["G0c_verdict"],
   "해설": ("G0a 는 사양서 verify_spec 과 동형 — 비트 일치(0.000e+00)로 팀원 실측을 재현. "
            "G0b 는 우리 **학습 모듈**이 같은 구조임을 증명한다(float32 누산 순서 차이만 남는다 — "
            "팀원 nn_infer.py 자체 주석도 torch 대비 5.96e-08). "
            "G0c 는 우리가 만든 npz 를 배포 추론기가 그대로 읽음을 증명한다."),
   "함정_확인": S01["함정_확인"]},
 "stage1_프레임": {
   "기반": "exp064/cache/frame99_ship.parquet (읽기 전용)",
   "frame_sha256": S01["frame_sha256"],
   "붙인_열": ["pitcher_id", "batter_id"],
   "파생": "cnt_state = balls_before*3 + strikes_before",
   "밀집열_순서_assert": S01["dense_order_assert"],
   "행수_assert": S01["row_count_assert"],
   "n_train": S01["n_train_rows"], "n_2024": S01["n_2024_rows"],
   "row_id_2024_OOT벡터와_동일": S01["rowid_matches_oot_vector"],
   "🔴": "프레임 검사는 ρ 가 아니라 학습행 수로 했다(사양서 경고)"},
 "stage2_학습": {
   "구조": "159 → 256(BN) → 128(BN) → 64 → 1 · edims [28,28,6,6,6,6,6] · BCEWithLogits",
   "🔴_우리가_새로_정한_변인": TR["레시피_우리변인"],
   "levels_우리(≤2023 학습행)": TR["levels_n"],
   "levels_배포(≤2024)": TR["levels_n_배포"],
   "2024_미지범주_셀수": TR["unknown_cells_2024"],
   "2024_미지범주_행수_any": TR["unknown_rows_2024_any"],
   "mu_sd_출처": TR["mu_sd_출처"],
   "mu_최대차_vs배포": TR["mu_max_abs_diff_vs_배포"],
   "sd_최대차_vs배포": TR["sd_max_abs_diff_vs_배포"],
   "시드별_best_epoch": TR["best_epoch_per_seed"],
   "k_고정": TR["k_고정"], "k_유도": TR["k_유도"],
   "앙상블_원시Brier": TR["앙상블_원시"],
   "k_강건성_원시Brier": TR["k_강건성_원시Brier"],
   "출하왕복_nn_infer_vs_torch_maxabs": TR["출하왕복_max_abs_vs_torch"],
   "epoch_curves": TR["curves"]},
 "stage3_p_v6완성_그리고_곡선": {
   "🔴_항등대조": S3["항등대조_3종판_재정규화"],
   "팔_단독_재센터링점수": S3["팔_단독_재센터링점수"],
   "상관_로짓": S3["상관_로짓"],
   "exp_070_재현대조": S3["exp_070_재현대조"],
   "부트_exp070_재현대조": S3["부트_exp070_재현대조"],
   "★w_v6_단독_0.50→0.70": {
     "3종판(exp_070)": {"Δ_pts": A3["Δ_pts"], "조건부_argmax": A3["조건부_argmax"],
                        "조건부_argmax_Δ_pts": A3["조건부_argmax_Δ_pts"],
                        "부트_CI95": BS["3종판"]["Δ(0.70)_부트_CI95"],
                        "부트_양수비율": BS["3종판"]["Δ(0.70)_부트_양수비율"],
                        "조건부_argmax_부트_CI95": BS["3종판"]["조건부_argmax_부트_CI95"]},
     "4팔완성판": {"Δ_pts": A4["Δ_pts"], "조건부_argmax": A4["조건부_argmax"],
                   "조건부_argmax_Δ_pts": A4["조건부_argmax_Δ_pts"],
                   "부트_CI95": BS["4팔완성판"]["Δ(0.70)_부트_CI95"],
                   "부트_양수비율": BS["4팔완성판"]["Δ(0.70)_부트_양수비율"],
                   "조건부_argmax_부트_CI95": BS["4팔완성판"]["조건부_argmax_부트_CI95"]},
     "짝지은_차이": BS["짝지은_차이_4팔−3종판"],
     "형태": "w_v6 만 이동, 나머지 5개(w_ebm −0.10 · β 0.03 · w_out 0.7 · c_aa 0.25244 · c_t70 0.04756) 배포 고정"},
   "격자_3종판": A3["0.00~1.00_격자"], "격자_4팔": A4["0.00~1.00_격자"],
   "k_민감도": S3["k_민감도"],
   "부트스트랩_설정": {"R": BS["R_BOOT"], "seed": BS["SEED_BOOT"], "방식": BS["방식"]}},
 "fold_2024": {
   "눈금": "OOT — train ≤2023 마스크(1,116,277) → valid 2024 전체(253,507)",
   "p_nn_ours_단독_재센터링_점수": S3["팔_단독_재센터링점수"]["p_nn_ours(k고정)"],
   "p_v6_4arm_단독_재센터링_점수": S3["팔_단독_재센터링점수"]["p_v6_4arm"],
   "p_v6_3arm_단독_재센터링_점수": S3["팔_단독_재센터링점수"]["p_v6_3arm"],
   "🔴_S1": "로컬 OOT 눈금이다. LB 예측치로 옮기지 마라."},
 "산출물_서버경로": {
   "p_nn_oot2024": f"{OUT}/p_nn_oot2024.npy",
   "p_nn_oot2024_bestepoch": f"{OUT}/p_nn_oot2024_bestepoch.npy",
   "p_nn_oot2024_last25": f"{OUT}/p_nn_oot2024_last25.npy",
   "p_v6_4arm_oot2024": f"{OUT}/p_v6_4arm_oot2024.npy",
   "npz_4시드": f"{E}/model/nn_ours_oot_a{{42,777,2024,101}}.npz (+ _meta.json)",
   "에폭별_2024예측": f"{E}/cache/p24_epochs_seed{{seed}}.npy (25×253507 float32)",
   "프레임": f"{E}/cache/frame071.parquet"},
 "wall_clock_min": {
   "stage1_프레임": S01["stage1_min"], "stage0_게이트": S01["stage0_min"],
   "stage2_학습": TR["train_time_min"],
   "stage3_점추정": S3["wall"]["점추정_min"], "stage3_부트": S3["wall"]["부트_min"],
   "stage3_총": S3["wall"]["stage3_총_min"]},
 "env": {"threads": "OMP/OPENBLAS/MKL/NUMEXPR/VECLIB = 6",
         "python": "3.11.15 venv · torch 2.7.1+cu128 · numpy 1.26.4 · pandas 2.0.3",
         "gpu": "RTX A5000 (1장 사용)"},
 "score_conversion": "점수 = 100000 × (1 − Brier/0.2498069). 6자리 Brier 환산은 ±0.20 불확실(S3).",
}
M["train_time_min"] = TR["train_time_min"]
M["inference_time_sec"] = None
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print("METRICS_WRITTEN", f"{OUT}/metrics.json")

# ── 부록·진단·배포NN 배율 + report 키 붙이기 ─────────────────────
AD = json.load(open(f"{OUT}/addendum.json"))
DG = json.load(open(f"{OUT}/diag_scale.json"))
DN = json.load(open(f"{OUT}/depnn_scale.json"))
M = json.load(open(f"{OUT}/metrics.json"))
M["stage3_p_v6완성_그리고_곡선"]["k_민감도_절대점수"] = AD
M["stage3_p_v6완성_그리고_곡선"]["배율_대_정보_분해"] = DG
M["stage3_p_v6완성_그리고_곡선"]["배포NN_로짓배율_대조"] = DN
A3 = M["stage3_p_v6완성_그리고_곡선"]["★w_v6_단독_0.50→0.70"]["3종판(exp_070)"]
A4 = M["stage3_p_v6완성_그리고_곡선"]["★w_v6_단독_0.50→0.70"]["4팔완성판"]
PR = M["stage3_p_v6완성_그리고_곡선"]["★w_v6_단독_0.50→0.70"]["짝지은_차이"]
M["report"] = {
 "가설": ("exp_070 곡면의 p_v6 는 NN 지분 0.15 가 빈 3종판이라 v6 팔이 약하고, 그래서 "
          "w_v6 argmax 가 아래로 눌려 있다 — NN 을 채우면 argmax 와 Δ 가 **올라간다**."),
 "결과_한줄": ("🔴 반대로 나왔다. 우리 NN 을 채우자 Δ(0.70) 가 +4.61814 → "
               f"{A4['Δ_pts']:+.5f} 로 내려가고 조건부 argmax 도 0.8581 → "
               f"{A4['조건부_argmax']:.4f} 로 내려갔다. 부트 CI 는 0 을 품는다."),
 "🔴그러나_결론은_이_방향으로_확정되지_않는다": (
     "그 답이 **우리가 새로 정한 학습 레시피의 에폭 규칙 하나에 뒤집힌다.** "
     "k=2~8 에서는 Δ 가 +3.30~+3.87 로 3종판(+4.61814)보다 낮고, k≥10 에서는 "
     "+4.36~+5.01 로 **3종판을 넘는다.** argmax 도 0.778 ↔ 0.853 으로 갈린다. "
     "즉 이 실험이 잰 것은 'NN 을 채우면 어떻게 되는가'가 아니라 "
     "'우리 레시피의 어떤 NN 을 채우는가'다."),
 "배율_아님_확인": (
     "3종판 로짓을 같은 표준편차로 늘리기만 한 대조군을 만들어 뺐다. "
     "NN 의 **정보 순증**은 k≤8 에서 음수(−2.6 ~ −6.3점), k≥10 에서 양수(+1.1 ~ +3.8점). "
     "블렌드 로짓 sd 는 어느 k 에서도 3종판(0.183)을 넘지 않으므로 로짓 배율 효과가 아니다."),
 "왜_그런가": (
     "표본 하나짜리 Brier 를 최소화하는 에폭(우리 규칙)은 기저율 쪽으로 수축된 NN 을 고른다. "
     "그 NN 은 GBDT 가 이미 가진 신호의 단조 변환에 가까워 블렌드에 새 정보를 못 넣는다. "
     "더 길게 학습한 NN 은 단독 점수가 나쁘지만(k=25 단독 198.67) 탈상관돼 있어 "
     "블렌드에는 더 보탠다. 블렌드가 원하는 것과 단독 Brier 가 원하는 것이 갈린다."),
 "변경_변인_전량": {
   "우리가_도입": ["dropout 0.30", "AdamW wd 1e-4", "batch 4096 drop_last",
                   "OneCycle max_lr 3e-3 / pct_start 0.3 / div 25 / final_div 1e4",
                   "25에폭 스케줄 · k=4 에서 꺼냄(스케줄 길이는 자르지 않음)",
                   "levels·mu/sd 를 우리 학습행(≤2023)에서 재계산",
                   "🔴 k 를 2024 valid 의 시드별 best epoch 중앙값 내림으로 골랐다 — "
                   "고른 것은 스칼라 하나지만 그 2024 가 바로 이 OOT 벡터가 예측하는 행이다"],
   "사양에서_고정": ["159차원 입력·연결 순서", "edims [28,28,6,6,6,6,6]",
                     "층 [256,128,64] · BN 은 앞 두 블록만", "임베딩 행 = 레벨+1, 0=미지",
                     "표준화 후 nan_to_num", "BCEWithLogits", "4시드 로짓평균"],
   "대조군에서_승계": ["p_lgb·p_cat·p_leaf·p_q·p_ebm·off 벡터", "DEP 6상수",
                      "재센터링 산술", "투수 블록 부트스트랩 RNG(seed 70, R=1000)"]},
 "🔴산출물은_그들의_NN이_아니다": (
     "Dropout·학습률·에폭·옵티마이저가 배포 아티팩트에 남지 않아 우리가 새로 정했다. "
     "이것은 **그 구조의 NN** 이지 **그들의 NN** 이 아니다. "
     f"참고: 배포 NN 4시드 로짓평균의 2024 sd 는 {DN['배포NN_4시드로짓평균_sd_2024']:.4f} 인데 "
     "이는 표본내(2024 가 학습 범위 안)라 부풀려진 상한이다. 우리 k=4 는 0.1580, k=8 은 0.1771. "
     f"ρ(배포NN, 우리 k=4) = {DN['rho(배포NN, 우리NN k4)']:.4f} 로 k=25 의 "
     f"{DN['rho(배포NN, 우리NN k25)']:.4f} 보다 높다 — 내용으로는 이른 정지 쪽이 배포본에 가깝다."),
 "🔴편향_정정_기록": (
     "exp_070 이 짚은 편향은 둘이다. 이 실험은 **첫째(p_v6 3종판)만** 다뤘다. "
     "**둘째(p_q 1시드)는 그대로 남아 있고 w_v6 를 반대 방향으로 민다.** "
     "따라서 이 결과를 '이제 정확하다' 로 읽으면 안 된다."),
 "재현_증거": (
     "같은 코드로 3종판을 돌려 exp_070 의 Δ(0.70) = 4.618141264278734 를 "
     f"abs_diff {M['stage3_p_v6완성_그리고_곡선']['exp_070_재현대조']['abs_diff']:.3e} 로, "
     "부트 CI [0.28391923018694304, 8.87803124864863] 를 자리까지 그대로 재현했다. "
     "파이프라인이 exp_070 과 같은 것임이 기계적으로 증명된다."),
 "판정": "🔴 채택·기각 판정 없음 (config 지시). 다음 후보의 설계 변인으로 넘긴다.",
 "다음_제안": [
   "① 에폭 규칙을 **블렌드 기준**으로 다시 정의할 것인지 결정 — 지금 규칙(단독 Brier)은 "
   "블렌드 목적과 어긋난다는 증거가 나왔다. 다만 그것은 S8(사전 등록) 대상이므로 "
   "**결과를 보고 고치는 형태가 되지 않게** 별도 사전 등록이 필요하다.",
   "② p_q 를 3시드로 채워 둘째 편향을 제거하기 전에는 w_v6 축의 부호를 신뢰하지 말 것.",
   "③ w_v6 를 후보로 사겠다면 **로컬 곡면이 아니라 LB 로 사서 읽는 축**으로 다루는 편이 "
   "정직하다 — 이 실험이 로컬 곡면의 이 축이 우리 임의 선택 하나에 ±1.7점 흔들림을 보였다."],
 "★BN용_숫자": {
   "3종판(exp_070 기준값)": {"Δ_pts": A3["Δ_pts"], "CI": A3["부트_CI95"],
                             "양수비율": A3["부트_양수비율"], "argmax": A3["조건부_argmax"]},
   "4팔완성판(k=4 정본)": {"Δ_pts": A4["Δ_pts"], "CI": A4["부트_CI95"],
                           "양수비율": A4["부트_양수비율"], "argmax": A4["조건부_argmax"]},
   "짝지은_차이": {"평균": PR["Δ차_부트_평균"], "CI": PR["Δ차_부트_CI95"],
                   "양수비율": PR["Δ차_양수비율"]},
   "k_에_따른_Δ_범위": {k_: v["Δ_0.70"] for k_, v in AD.items() if isinstance(v, dict)},
   "🔴": "S1 — fold_2024 OOT 로컬 눈금. LB 예측치로 옮기지 마라."},
 "ETA_대비": "ETA 30~40분 · 실측 약 52분(게이트 진단 2건 추가로 +8분, 부트 3.57분)",
}
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print("METRICS_FINAL", f"{OUT}/metrics.json")

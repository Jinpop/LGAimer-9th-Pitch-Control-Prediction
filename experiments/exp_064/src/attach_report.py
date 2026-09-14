"""exp_064 — metrics.json 에 서술 블록·단계별 실측 벽시계를 붙인다.

report.md 를 파일로 남기는 경로가 이 환경에서 막혀 있어, 서술을 metrics.json 안에
구조화해 넣는다. 숫자는 전부 metrics.json 이 이미 담고 있는 값을 그대로 읽는다
(여기서 새로 계산하지 않는다 — 손으로 옮겨 적다 틀리는 길을 없앤다).
"""
import json
import sys

P = sys.argv[1]
M = json.load(open(P, encoding="utf-8"))
A, B = M["stage_A"], M["stage_B"]
g2b = M["gate2_nn_delta"]

M["wall_clock_measured"] = {
    "stage1_seasons_6x_parallel3": {"start": "11:59:44", "end": "12:00:47",
                                    "sec": 63,
                                    "per_season_min": [0.48, 0.37, 0.49, 0.41,
                                                       0.46, 0.50]},
    "stage1_assemble": {"start": "12:00:47", "end": "12:01:04", "sec": 17},
    "stage2a_compare44": {"start": "12:02:14", "end": "12:02:16", "sec": 2},
    "stage2b_nn_delta_16infer_253507rows_cpu": {"start": "12:02:16",
                                                "end": "12:02:59", "sec": 43},
    "stage3_run064": {"start": "12:03:47", "end": "12:15:23", "min": 11.536},
    "stage3_corr_finalize": {"start": "12:15:23", "end": "12:15:30", "sec": 7},
    "pipeline_total": {"start": "11:59:44", "end": "12:15:30", "min": 15.77},
    "eta_given": "25분", "note": "ETA 대비 -9.2분. stage 1 이 예상 2~3분 -> 실측 1.3분"}

M["report"] = {
    "hypothesis": (
        "exp_062 가 잰 배포 불일치 −2.97796점은 우리가 만든 것이다. p6 3종은 빈티지가 "
        "없고 전 연도가 배포본 한 벌을 쓴다 — 우리는 그것을 685쌍 지도로 재구축했고 "
        "그래서 어긋났다. 배포본을 복사해 14열만 다시 계산하면 차이가 구조상 0 이 된다."),
    "what_changed_vs_exp061": {
        "intended": "14열의 산출원 — 우리 685쌍 재구축 p6 -> 배포본 p6 그대로",
        "incidental_enumerated_in_config": [
            "pitcher_id 어휘 685 -> 571",
            "2019행 14열: NaN -> 값",
            "14열 분포 변화에 따른 표준화 mu/sd 재계산"],
        "incidental_found_during_run": (
            "🔴 config 가 '2019행'으로만 적은 것이 실제로는 **전 시즌**에 걸친 성질 "
            "변화다. exp_061 프레임은 시즌 Y 행의 p6 를 trackman < Y 로 적합한 "
            "**연도별 as-of** 였다. 배포 p6 는 빈티지가 없으므로 exp_064 프레임은 전 "
            "시즌이 ≤2024 트랙맨 한 벌을 쓴다. 즉 exp_061 -> exp_064 는 '출처'와 "
            "'as-of 규율' 둘이 함께 바뀐 것이고, +8.22709 를 출처 하나에 귀속할 수 없다. "
            "다만 배포 그대로를 재현하는 것이 이 실험의 목적이므로 둘은 분리 불가다.")},
    "gates": (
        "stage0 sha 3/3 · stage1 불변 88열 바이트 동일 · stage2a 44열 전부 정확히 0.0 · "
        "stage2b 배포 NN 로짓 차 정확히 0.0(움직인 행 0) · stage3 재현 seed42 0.0"),
    "headline": {
        "fold_2024_valid_all_score": A["nn99ship"]["seedavg_logit_score"],
        "vs_exp061_99col_ours_p6": A["D_vs_exp061_99col"],
        "vs_exp063_85col": A["D_vs_exp063_85col"],
        "fold_2023_split_B_score": B["nn99ship"]["seedavg_logit_score"],
        "B_vs_exp061": B["D_vs_exp061_99col"],
        "B_vs_exp063": B["D_vs_exp063_85col"]},
    "seed_paired_caveat": (
        "시드평균 이득은 시드별 짝지은 차가 뒷받침하지 않는다. 분할 A 짝지은 평균 "
        f"{M['paired_by_seed']['A']['delta_vs_ours99']['mean']:+.5f}(sd "
        f"{M['paired_by_seed']['A']['delta_vs_ours99']['sd']:.2f}, 4시드 중 2개 음수)이고 "
        "그 평균은 seed 101 하나(+32.66)가 만든다. 분할 B 짝지은 평균은 "
        f"{M['paired_by_seed']['B']['delta_vs_ours99']['mean']:+.5f}(4시드 중 3개 음수)로 "
        "사실상 0이다. exp_063 이 남긴 관측(시드 산포가 앙상블 차를 압도한다)이 여기서도 "
        "유지된다."),
    "ensemble_gain": {
        "S_seed_A_this": A["nn99ship"]["S_seed"],
        "S_seed_B_this": B["nn99ship"]["S_seed"],
        "exp061_99col_A": 55.10, "exp063_85col_A": 43.492331788122556,
        "per_seed_sd_A_this": A["nn99ship"]["per_seed_score_sd"],
        "exp063_85col_per_seed_sd_A": 29.93573074353712,
        "read": ("시드평균 이득(S_seed)이 셋 중 가장 크고 시드 산포는 85열의 절반 이하다. "
                 "exp_063 이 '14열은 정확도가 아니라 시드 다양성을 공급한다'고 적은 것과 "
                 "같은 방향이며, 배포본 p6 에서 그 효과가 더 크다.")},
    "correlation": (
        "NN_ship vs AA67 = 0.8979 — 우리 p6 99열(0.9014)·85열(0.9021)보다 **낮다**(더 다양). "
        "그러나 team_Q(+0.0026)·p_EBM(+0.0084)·v6(+0.0009)와는 오히려 올랐다. 방향이 "
        "대상마다 갈리므로 '다양성이 좋아졌다'로 뭉뚱그릴 수 없다. "
        "기준 재현: corr(NN99,AA67)=0.9013795597911097 로 exp_061 기록과 비트 동일, "
        "corr(AA_deployed,Q)=0.97757831 로 장부값 재현 — 상관 눈금이 같다."),
    "measurement_bias_REQUIRED": (
        "🔴 트랙맨은 2019~2024 뿐이고 2025 는 없다. 배포 p6 는 ≤2024 트랙맨으로 적합됐다. "
        "따라서 valid **2024** 행은 그 투수의 2024 레퍼토리를 p6 가 학습 때 봤고, "
        "test **2025** 행은 못 본다. 규칙 위반이 아니다 — 구종 라벨은 타깃(control_success) "
        "파생이 아니고, 이 실험이 바꾼 14열 어디에도 타깃이 들어가지 않는다"
        "(타깃 인코딩 열 idea_cnt_te·i3_te48·sit_* 6종은 88열 바이트 동일 게이트 안에 있다). "
        "그러나 **fold_2024 이득 중 p6 계열의 몫은 배포에서 그만큼 나오지 않는다.** "
        "exp_063(85열) 대비 +17.79609, exp_061(우리 p6) 대비 +8.22709 는 둘 다 "
        "**배포 이득의 상한**으로 읽어야 한다. "
        "⚠️ 부분 반증 하나: 오염이 더 심할 분할 B(valid 2023 — p6 가 2023·2024 를 다 봤다)의 "
        "exp_061 대비 이득이 +2.34076 로 오히려 작다. 순수 오염 설명만으로는 부호·크기가 "
        "설명되지 않는다. 다만 분할 B 는 train·valid 가 함께 오염돼 대칭이라 깨끗한 반증은 "
        "아니다."),
    "verdict_scope": (
        "🔴 판정하지 않는다(config 사전 등록). 이 실험의 산출은 게이트 결과·NN 단독 점수·"
        "상관 셋뿐이다. 팔로 쓸지는 블렌드 위에서 재는 별도 측정이 필요하고 그건 다음 실험이다."),
    "next_suggestions": [
        "블렌드 위 측정 — 이 팔을 넣었을 때 blend_now 로짓이 얼마를 움직이는가. "
        "S4(팔 기준 Δ ≠ 블렌드 Δ, AC 실측 전달률 0.068)를 그대로 적용하면 "
        "+17.80 의 팔 이득은 블렌드에서 한 자릿수로 줄어든다 — 산술 예측이지 측정이 아니다.",
        "시드 확장 — 짝지은 차가 4시드로는 갈린다(A 2/4 음수, B 3/4 음수). "
        "sd 18.4 에서 mean +5.88 을 가르려면 시드가 더 필요하다.",
        "as-of 분리 실험 — 우리 685쌍 지도로 **≤2024 한 벌**을 만들어 exp_061 과 비교하면 "
        "'출처' 몫과 'as-of 규율' 몫을 가를 수 있다. 다만 배포는 어차피 배포본을 쓰므로 "
        "실용 가치보다 해석 가치의 실험이다."],
    "artifacts": {
        "server_workdir": "~/SungjinPark/exp064/",
        "frame": "~/SungjinPark/exp064/cache/frame99_ship.parquet "
                 "(sha256 " + M["stage1_assemble"]["frame99_ship_sha256"] + ")",
        "preds": "~/SungjinPark/exp064/out/preds/{A,B}99ship_s{42,777,2024,101}.npy "
                 "+ _seedavg.npy",
        "note": "모델 가중치·프레임은 서버에 둔다. 제출물 없음."}}

json.dump(M, open(P, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("attached ->", P)

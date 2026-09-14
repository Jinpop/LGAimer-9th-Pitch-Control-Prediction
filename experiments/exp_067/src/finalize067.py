"""exp_067 마무리 — 산출물 취합 + 에폭 곡선 분석 + metrics.json 작성.

에폭 곡선 분석은 **이미 계산된 숫자만** 쓴다(추가 학습 없음).
fold_2024 에는 정직한 out-of-time valid 가 있으므로, "고정 에폭 k 규칙"이
25에폭 완주 대비 어떤가를 각 k 에 대해 그대로 읽을 수 있다.
🔴 이건 자료지 결정이 아니다. 배포 에폭 규칙은 사용자 판단 사항이다.
"""
import json
import os

import numpy as np

B = os.path.expanduser("~/SungjinPark/exp067")
OUT = f"{B}/out"
SEEDS = [42, 777, 2024, 101]
REF = 0.2498069


def conv(b):
    return 100000.0 * (1.0 - b / REF)


R = json.load(open(f"{OUT}/run067.json"))
G0 = json.load(open(f"{OUT}/g0_frame.json"))
G0B = json.load(open(f"{OUT}/g0_real5.json"))
G3 = json.load(open(f"{OUT}/g3_ship.json"))

# ── fold_2024 에폭별 per-seed 평균 점수 (곡선에서 그대로) ────────────────────
curves = {s: json.load(open(f"{OUT}/curves/curve_A_s{s}.json")) for s in SEEDS}
n_ep = len(curves[SEEDS[0]])
by_ep = []
for k in range(n_ep):
    br = [curves[s][k]["valid_brier"] for s in SEEDS]
    by_ep.append({"epoch": k + 1,
                  "per_seed_brier": br,
                  "mean_brier": float(np.mean(br)),
                  "mean_score": float(np.mean([conv(b) for b in br])),
                  "worst_seed_score": float(min(conv(b) for b in br))})
best_fixed = max(by_ep, key=lambda d: d["mean_score"])
ep25 = by_ep[-1]
EPOCH_ANALYSIS = {
    "🔴": "fold_2024(<=2023 학습, 2024 valid) 곡선에서 읽은 **시드 평균 per-seed 점수**다. "
          "로짓 4시드 평균 점수가 아니다(그건 예측 벡터가 있어야 계산된다). "
          "고정 에폭 규칙끼리의 상대 비교용.",
    "prereg_epoch25_last": ep25,
    "best_fixed_epoch_on_fold2024": best_fixed,
    "gap_points_best_fixed_minus_25": best_fixed["mean_score"] - ep25["mean_score"],
    "per_seed_argmin_epoch": {str(s): int(np.argmin(
        [c["valid_brier"] for c in curves[s]]) + 1) for s in SEEDS},
    "curve_mean_score_by_epoch": [d["mean_score"] for d in by_ep],
    "해석_주의": "여기서 좋은 에폭을 골라 배포에 쓰면 그것은 fold_2024 위의 선택이다. "
                 "허용되는 설계이지만 **사전 등록 규칙을 결과를 보고 바꾸는 것은 아니어야 한다**(S8). "
                 "이 실험의 배포본은 사전 등록대로 25에폭 완주(last)로 만들었다.",
    # ── 학습 행수가 늘면 과적합 시작점이 뒤로 밀리는가 (기존 자료만 사용) ──
    "argmin_epoch_vs_train_size": {
        "관측점": [
            {"split": "exp_064 B (2020~2022 -> 2023R)", "n_train": 659125,
             "steps_per_epoch": 659125 // 4096,
             "best_epochs": [9, 3, 4, 5], "mean": 5.25, "median": 4.5},
            {"split": "exp_064/067 A (<=2023 -> 2024)", "n_train": 1116277,
             "steps_per_epoch": 1116277 // 4096,
             "best_epochs": [5, 9, 8, 12], "mean": 8.5, "median": 8.5},
            {"split": "exp_067 배포 (<=2024)", "n_train": 1369784,
             "steps_per_epoch": 1369784 // 4096,
             "best_epochs": "🔴 잴 수 없다 — out-of-time valid 가 없다"}],
        "선형_외삽": {"slope_epochs_per_row": (8.5 - 5.25) / (1116277 - 659125),
                      "extrapolated_argmin_at_1369784": 8.5 + (8.5 - 5.25)
                      / (1116277 - 659125) * (1369784 - 1116277)},
        "🔴 신뢰도": "관측점이 둘, 각 4시드이고 시드 산포가 크다(3~12). 추세는 "
                     "**시사적일 뿐 확립된 것이 아니다**. 그리고 '에폭'은 행수에 따라 "
                     "스텝 수가 달라지는 단위라(160 -> 272 -> 334 스텝) 눈금 자체가 "
                     "행수와 얽혀 있다. 외삽값을 규칙으로 쓰지 마라."},
}

s2a = R["stage2a_fold2024"]
s2b = R["stage2b_deploy"]


# ── 3자 대조 — 변인을 하나씩 떼어 본다 (전부 2024 행 253,507 위) ────────────
def _lg(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


PA = {"exp064_le2023_best": np.load(f"{OUT}/preds/A_best_seedavg.npy"),
      "exp067_le2023_last": np.load(f"{OUT}/preds/A_last_seedavg.npy"),
      "exp067_le2024_last_DEPLOY": np.load(f"{OUT}/preds/D_last_seedavg.npy")}
pairs = {}
ks = list(PA)
for i in range(len(ks)):
    for j in range(i + 1, len(ks)):
        a, b = PA[ks[i]], PA[ks[j]]
        pairs[f"{ks[i]} vs {ks[j]}"] = {
            "pearson_prob": float(np.corrcoef(a, b)[0, 1]),
            "pearson_logit": float(np.corrcoef(_lg(a), _lg(b))[0, 1]),
            "mean_abs_prob_diff": float(np.mean(np.abs(a - b))),
            "max_abs_prob_diff": float(np.max(np.abs(a - b)))}
THREE_WAY = {
    "🔴": "세 벡터 전부 2024 행 위다. 앞 둘은 out-of-time(2024 미학습), "
          "셋째는 **표본내**다(2024 가 학습 안에 있다). 상관·분포는 서술이고 "
          "셋째의 Brier 를 성능으로 읽으면 안 된다(S1).",
    "변인_분해": "① exp064_le2023_best -> exp067_le2023_last = 에폭 규칙 단독 변인 "
                 "② exp067_le2023_last -> exp067_le2024_last = 학습 범위 단독 변인",
    "moments": {k: {"mean": float(v.mean()), "sd": float(v.std()),
                    "logit_sd": float(_lg(v).std())} for k, v in PA.items()},
    "pairs": pairs}
M = {
    "exp_id": "exp_067",
    "date": "2026-08-21",
    "type": "배포용(<=2024) NN 재학습 + 출하 실현 가능성 게이트. 제출물 없음.",
    "prereg": R["prereg"],
    "env": R["env"],

    # ── 표준 지표 자리 ──────────────────────────────────────────────────
    "fold_2022": None,
    "fold_2023": None,
    "fold_2024": {
        "🔴 무엇을_잰_것인가": "이 실험의 **배포본은 fold_2024 로 평가할 수 없다**"
                              "(2024 가 학습 안에 있다, S1). 아래 fold_2024 숫자는 "
                              "배포본이 아니라 **에폭 선택 규칙 변인을 재기 위해 같은 코드로 "
                              "돌린 <=2023 학습 판**이다.",
        "best_epoch_arm": {
            "brier": s2a["best"]["seedavg_logit_brier"],
            "bss": 1.0 - s2a["best"]["seedavg_logit_brier"] / REF,
            "score": s2a["best"]["seedavg_logit_score"],
            "per_seed_score": s2a["best"]["per_seed_score"],
            "sel_epochs": s2a["best"]["sel_epochs"],
            "seg": s2a["best"]["seg"],
            "==exp_064": abs(s2a["best"]["seedavg_logit_score"]
                             - R["prereg"]["control_exp064_fold2024_bestepoch"]["score"]) < 1e-9},
        "last_epoch_arm_사전등록_배포규칙": {
            "brier": s2a["last"]["seedavg_logit_brier"],
            "bss": 1.0 - s2a["last"]["seedavg_logit_brier"] / REF,
            "score": s2a["last"]["seedavg_logit_score"],
            "per_seed_score": s2a["last"]["per_seed_score"],
            "seg": s2a["last"]["seg"]},
        "D_last_minus_best": s2a["D_last_minus_best"],
        "n_train": s2a["n_train"], "n_valid": s2a["n_valid"]},

    "stage0_gate": {
        "Q1_99열이_추론시점_test행에서_만들어지는가": {
            "answer": "예",
            "band_frame_245789행": G0["Q1_all_99_present"],
            "실제_대회표본_5행": G0B["Q1_all_99_present_on_real_rows"],
            "attrib_tally": G0["attrib_tally"],
            "🔴 cnt_state": "features_sp.txt 99열에는 없다. NN 입력은 99 + cnt_state = "
                            "100열이고 cnt_state 는 v6_infer 가 자기 줄에서 만든다. "
                            f"실측 존재: {G0B['cnt_state_present']}"},
        "Q2_빠진_열": {"n": G0["Q2_n_missing"], "cols": G0["Q2_missing"],
                       "새_표·새_조인·새_상수_필요": "없음"},
        "Q3_출하_상이·추가_파일": G3["Q3_new_files"],
        "Q4_추론_시간": G3["S4_inference"],
        "Q5_R/F_분기_규칙": {
            "answer": "NN 에는 R/F 분기가 없다 — 그리고 그것이 이미 정의된 규칙이다(새 자유도 아님)",
            "근거": [
                "NN 은 game_type 을 범주 입력으로 받는 단일 모델이다. 분기 코드가 없다.",
                "학습 마스크의 F 제외는 season<=2022 에 고정된 데이터 품질 규칙이고 "
                "배포 팔(T70)이 이미 쓰는 바로 그 규칙이다(stage 1 에서 행수로 확인).",
                "블렌드의 팔 가중은 전역 스칼라뿐이다(w_v6·c_aa·c_S2·c_S1·c_T70). "
                "세그먼트별 팔 가중은 존재하지 않는다. beta_f 는 현재 0.0.",
                "따라서 여섯째 팔을 더하는 것은 전역 스칼라 c_nn 하나를 더하는 일이고, "
                "그건 이미 감사된 축의 확장이지 새 종류의 자유도가 아니다."],
            "🔴 남는_것": "c_nn 값 자체는 이 실험이 정하지 않는다(판정 금지). 빌드 계획서 소관.",
            "🔴 빌드가_고쳐야_할_것": "script.py 의 "
                "`assert abs((c_aa+c_s2+c_s1+c_t70)-(1-w_v6))<1e-12` 는 5항 합만 검사한다. "
                "c_nn 을 더하면 이 assert 를 함께 고치지 않는 한 채점 서버에서 터진다."},
        "🔴 대역_프레임_경고": {
            "관찰": f"패키지 안 test.csv(row_id 가 TRAIN_1221586..)는 대역 프레임이다. "
                    f"거기서 ins_* 16열이 {G0['coverage']['ins_p_n_zero_rows']}/"
                    f"{G0['n_test']} 행 = 100% 축퇴한다(ins_p_n 전부 0).",
            "실제_표본_대조": {"ins_p_n": G0B["ins_p_n"],
                              "발동_행수": G0B["has_ins_form"],
                              "n": G0B["n_real_rows"]},
            "함의": "커버리지·발동 판정을 대역 단독으로 하면 아무것도 시험하지 않은 것이 된다"
                    "(CLAUDE.md 의 AF~AI 사고와 같은 자리). 이 실험은 실제 표본을 병행했다."},
        "wall_s": G0["t"]["total_s"] + G0B["build_s"]},

    "stage1_mask": R["stage1_mask"],
    "stage2b_deploy_배포본": s2b,
    "deploy_contrast_2024rows": R["deploy_contrast_2024rows"],
    "three_way_contrast": THREE_WAY,
    "repro_gate": R["repro_gate"],
    "ship_format_gates": {
        "S1_npz_roundtrip": G3["S1_npz_roundtrip"],
        "S1_seedavg_max_abs_diff": G3["S1_seedavg_max_abs_diff"],
        "S2_cat_vocab_pass": G3["S2_pass"],
        "S2_cat_vocab": G3["S2_cat_vocab"],
        "S3_real5": G3["S3_real5"],
        "meta_shape": G3["meta_shape"],
        "source_sha": G3["source_sha"]},
    "epoch_rule_analysis": EPOCH_ANALYSIS,
    "train_time_min": R["train_time_min"],
    "inference_time_sec": G3["S4_inference"]["nn_forward_total_s"],
}
json.dump(M, open(f"{OUT}/metrics_core.json", "w"), indent=1, ensure_ascii=False)
print("epoch25 mean_score", ep25["mean_score"])
print("best fixed epoch", best_fixed["epoch"], best_fixed["mean_score"])
print("gap", EPOCH_ANALYSIS["gap_points_best_fixed_minus_25"])
print("fold2024 best seedavg", s2a["best"]["seedavg_logit_score"])
print("fold2024 last seedavg", s2a["last"]["seedavg_logit_score"])
print("D", s2a["D_last_minus_best"]["seedavg_score"])

"""exp_064 — 단계별 산출을 표준 metrics.json 하나로 합친다.

🔴 판정하지 않는다(config `사전_등록_판정_범위`). 여기서는 게이트 통과 여부·점수·상관만
기계적으로 모으고, 채택/기각 문장을 만들지 않는다.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

OUT = f"{L.B}/out"
M = json.load(open(f"{OUT}/metrics_partial.json"))


def load(name):
    p = f"{OUT}/{name}"
    return json.load(open(p)) if os.path.exists(p) else None


M["stage0_tables"] = load("stage0_tables.json")
M["stage1_assemble"] = load("stage1_assemble.json")
M["gate2_compare44"] = load("gate2_compare44.json")
M["gate2_nn_delta"] = load("gate2_nn_delta.json")
cor = load("corr.json")
if cor:
    M["correlations"] = cor

g2a = M["gate2_compare44"]["prereg_check"] if M["gate2_compare44"] else None
g2b = M["gate2_nn_delta"]["prereg_check"] if M["gate2_nn_delta"] else None
M["gates"] = {
    "stage0_ship_sha": all(v["pass"] for v in
                           M["stage0_tables"]["ship_p6_sha"].values()),
    "stage1_invariance_85col_byte_identical":
        M["stage1_assemble"]["invariance_gate"]["byte_identical"],
    "stage2a_14col_max_abs": g2a["cols14_max_abs"] if g2a else None,
    "stage2a_pass": g2a["cond_14col_max_le_1e-6"] if g2a else None,
    "stage2a_exactly_zero": g2a["exactly_zero_14"] if g2a else None,
    "stage2b_score_diff": g2b["score_diff"] if g2b else None,
    "stage2b_pass": g2b["cond_abs_score_diff_lt_1point"] if g2b else None,
    "stage3_repro_seed42_max_abs_diff": M["repro_gate"]["max_abs_diff"],
    "stage3_repro_pass": M["repro_gate"]["pass"]}

A = M["stage_A"]["nn99ship"]
B = M["stage_B"]["nn99ship"]
ms = A["metrics_seedavg"]
M["fold_2024"] = {
    "brier": ms["brier"], "bss": ms["brier"] and (1.0 - ms["brier"] / L.REF),
    "score": ms["score"], "auc": ms["auc"], "logloss": ms["logloss"],
    "valid_R": ms.get("valid_R"), "valid_F": ms.get("valid_F"),
    "n": M["split_A"]["n_valid"],
    "desc": "train<=2023 -> valid 2024 valid_all, 4시드 로짓평균 (주지표)"}
M["fold_2023"] = {
    "brier": B["seedavg_logit_brier"],
    "bss": 1.0 - B["seedavg_logit_brier"] / L.REF,
    "score": B["seedavg_logit_score"], "n": M["split_B"]["n_valid"],
    "desc": ("train 2020~2022 R -> valid 2023 R. 환산 ref 는 2024 눈금이라 "
             "수준은 분할 간 비교 불가 — 분할 안의 차만 읽는다")}
M["fold_2022"] = None
M["train_time_min"] = M.get("total_wall_min")
M["inference_time_sec"] = None
M["notes"] = {
    "inference_time": "제출물을 만들지 않는 실험 — 추론 시간 측정 대상 없음",
    "measurement_bias": (
        "트랙맨은 2019~2024 뿐이고 2025 는 없다. 배포 p6 는 ≤2024 트랙맨으로 적합됐다 — "
        "valid 2024 행의 투수 레퍼토리를 p6 가 학습 때 봤고, test 2025 행은 못 본다. "
        "규칙 위반은 아니지만(구종은 타깃 파생이 아니다) fold_2024 이득 중 p6 계열 몫은 "
        "배포에서 그만큼 나오지 않는다 — 이 실험의 이득은 **배포 이득의 상한**이다."),
    "verdict_scope": "🔴 팔 채택을 판정하지 않는다. 게이트·점수·상관만 낸다(config S8)."}

json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"gates": M["gates"], "fold_2024": M["fold_2024"],
                  "fold_2023": M["fold_2023"],
                  "D_vs_exp061_99col_A": M["stage_A"]["D_vs_exp061_99col"],
                  "D_vs_exp063_85col_A": M["stage_A"]["D_vs_exp063_85col"],
                  "D_vs_exp061_99col_B": M["stage_B"]["D_vs_exp061_99col"],
                  "D_vs_exp063_85col_B": M["stage_B"]["D_vs_exp063_85col"],
                  "train_time_min": M["train_time_min"]},
                 indent=1, ensure_ascii=False))
print("FINALIZE_OK", flush=True)

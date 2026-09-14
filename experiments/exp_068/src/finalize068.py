"""exp_068 취합 — run068.json + ship068.json -> experiments/exp_068/metrics.json.

🔴 이 환경은 서브에이전트의 .md 작성을 차단하므로 서술은 metrics.json 의
   `report` 키에 넣는다.
🔴 판정하지 않는다. 채택·제출 결정은 지휘자·사용자 소관이다.
"""
import json
import os
import subprocess
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

B68 = os.path.expanduser("~/SungjinPark/exp068")
OUT = f"{B68}/out"
E67 = os.path.expanduser("~/SungjinPark/exp067/out")
REF = 0.2498069
SEEDS = [42, 777, 2024, 101]
K = 8


def conv(b):
    return 100000.0 * (1.0 - b / REF)


R = json.load(open(f"{OUT}/run068.json"))
S = json.load(open(f"{OUT}/ship068.json"))

# ── AUC (fold_2024 out-of-time) ─────────────────────────────────────────────
fr = pd.read_parquet(os.path.expanduser(
    "~/SungjinPark/exp064/cache/frame99_ship.parquet"),
    columns=["season", "game_type", "control_success"])
m24 = fr["season"].to_numpy() == 2024
y = fr.loc[m24, "control_success"].to_numpy(np.float64)
g = fr.loc[m24, "game_type"].to_numpy()
del fr
p_k8 = np.load(f"{OUT}/preds/A_k{K}_ens.npy")
assert len(p_k8) == len(y) == 253507
auc_k8 = float(roc_auc_score(y, p_k8))

s1 = R["stage1_fold2024"]
b_k8 = s1["headline_k8_ensemble"]["brier"]

M = {
    "exp_id": "exp_068",
    "date": R["date"],
    "type": R["type"],
    "prereg": R["prereg"],
    "env": R["env"],
    "fold_2022": None,
    "fold_2023": None,
    "fold_2024": {
        "🔴 무엇을_잰_것인가": (
            "이 실험의 **배포본(stage 2)은 fold_2024 로 평가할 수 없다**(2024 가 학습 "
            "안에 있다, S1). 아래 fold_2024 숫자는 배포본이 아니라 **에폭 규칙을 재기 "
            "위해 같은 코드로 돌린 <=2023 학습 판**이다."),
        "brier": b_k8,
        "bss": 1.0 - b_k8 / REF,
        "score": conv(b_k8),
        "auc": auc_k8,
        "arm": f"k={K} 고정 4시드 로짓평균 앙상블 (이 실험의 주지표)",
        "per_seed_score": s1["headline_k8_ensemble"]["per_seed_score"],
        "seg": s1["headline_k8_ensemble"]["seg"],
        "controls": {
            "best_epoch_ens_상한_exp064": s1["control_best_epoch_ensemble"],
            "last_epoch_ens_반증된규칙_exp067": s1["control_last_epoch_ensemble"]},
        "vs_controls": s1["vs_controls"],
        "n_train": s1["n_train"], "n_valid": s1["n_valid"]},
    "masks": R["masks"],
    "repro_gate": R["repro_gate"],
    "stage1_fold2024_full": s1,
    "stage2_deploy": R["stage2_deploy"],
    "stage3_robustness": R["stage3_robustness"],
    "deploy_contrast_2024rows": R["deploy_contrast_2024rows"],
    "ship_files": R["ship_files"],
    "ship_format_gates": {k: S[k] for k in
                          ("S1_npz_roundtrip", "S1_ens_max_abs_diff",
                           "S2_cat_vocab", "S2_pass", "S3_real5",
                           "meta_shape", "meta_CATS_matches",
                           "band_contrast_vs_exp067", "source_sha")},
    "inference_time_sec": S["S4_inference"]["nn_forward_total_s"],
    "inference_detail": S["S4_inference"],
    "train_time_min": R["train_time_min"],
}

# ── 강건성 표 (읽기 쉬운 형태) ───────────────────────────────────────────────
rob = R["stage3_robustness"]["ensemble_by_k"]
M["stage3_robustness"]["표"] = {
    f"k={k}": f"{rob[f'k{k}']['score']:.5f}점 (brier {rob[f'k{k}']['brier']:.10f})"
    for k in (4, 6, 8, 10, 12)}

# ── C단 인수인계 갱신 ────────────────────────────────────────────────────────
M["handoff_C단"] = {
    "🔴 exp_067_대비_바뀐_것": (
        f"배포 npz 4벌의 내용이 '25에폭 완주 가중치' 에서 '25에폭 OneCycle 스케줄 위 "
        f"{K}에폭 시점 가중치' 로 바뀌었다. 파일 이름·레이아웃·meta 는 그대로다 "
        "(meta sha 는 exp_067 과 동일 — 마스크·프레임·전처리가 같기 때문)."),
    "🔴 왜_바꿨나": (
        f"exp_067 이 잰 결과 25에폭 완주 규칙의 대가가 fold_2024 에서 "
        f"{-R['prereg']['control']['exp064_best_epoch_ens'] + R['prereg']['control']['exp067_last_epoch_ens']:.5f}점이었다. "
        "config exp_067 이 '둘 다 만들어 차이를 재고 기록하라' 고 명시했으므로 "
        "재는 것이 계획이었고, 반증된 구성을 그대로 출하하는 것이 오히려 규율 위반이다."),
    "k_유도_소명용": R["prereg"]["k_유도"],
    "리크가_아닌_이유": (
        "2024 는 배포 모델의 학습 범위 안이다(배포는 <=2024 전량 적합). "
        "'검증으로 하이퍼파라미터를 고르고 전체 데이터로 재적합' 은 표준 절차이며 "
        "test(2025) 는 이 경로 어디에도 들어오지 않는다. 고른 것은 에폭 하나(스칼라)다."),
    "넣을_것": (
        f"새 디렉터리 `model_nn_sp/` 에 npz 4 + meta 1~4 "
        f"(총 {R['ship_files']['total_bytes']/1e6:.2f} MB). "
        "🔴 `mate/` 안의 파일은 한 바이트도 바뀌지 않는다."),
    "서버_경로": f"{B68}/out/ship/nn_sp{{42,777,2024,101}}.npz (+ _meta.json)",
    "고칠_것": [
        "🔴 top-level `script.py` 의 `assert abs((c_aa+c_s2+c_s1+c_t70)-(1.0-w_v6))<1e-12` "
        "— 5항 합만 검사한다. `c_nn` 을 더하면서 이 assert 를 같이 고치지 않으면 "
        "채점 서버에서 터진다.",
        "`blend.json` 에 `c_nn` 과 그 유래 기록.",
        "BOOT 문자열에 `nn_infer.predict_nn` 래퍼 추가 (초안: exp_067/src/nn_arm_draft.py).",
        f"🔴 NN 가중치 출처를 exp_067 이 아니라 **exp_068(k={K})** 로 적을 것."],
    "안_고쳐도_되는_것": (
        "`model/meta.json` 의 부스터 트리 수 대조 — NN 팔은 부스터를 건드리지 않는다. "
        "그래도 빌드 때 확인은 할 것."),
    "추론_시간": (
        f"NN 팔 추가비용 = **{S['S4_inference']['nn_forward_total_s']}초** "
        f"(4시드 x {S['S4_inference']['n_rows']:,}행, numpy, 스레드 6, 단독 실행). "
        f"exp_067 실측 {S['S4_inference']['exp067_reference_s']}초와 같은 자리 — "
        "가중치 값만 바뀌었고 연산량은 동일하다."),
    "게이트_설계": [
        "항등 앵커: `c_nn = 0.0` 으로 빌드해 출력 CSV sha 가 부모와 같은지.",
        "규칙 2-4 는 내용/위치 분리 실험으로. 우리 NN 도 팀원 v6 NN 과 같은 numpy "
        "축약이라 SIMD 블록 경계 잡음(~1e-9)이 나올 수 있다.",
        "🔴 커버리지 게이트를 대역 프레임 단독으로 돌리지 말 것 — 대역 프레임은 배포 "
        "앵커보다 앞이라 시즌-내 폼 계열이 축퇴한다. 실제 대회 표본 5행 병행."],
    "판정하지_않은_것": "`c_nn` 값. 이 팔을 실제로 쓸지. 전부 사용자·빌드 계획서 소관이다."}

M["artifacts"] = {
    "서버_배포_가중치": f"{B68}/out/ship/nn_sp{{42,777,2024,101}}.npz + _meta.json",
    "예측_벡터": (f"{B68}/out/preds/ — A_k{K}_s*.npy·A_k{K}_ens.npy (2024 out-of-time), "
                  f"D_k{K}_s*.npy·D_k{K}_ens.npy (🔴 표본내), "
                  f"BAND_k{K}_ens.npy (대역 프레임 245,789행)"),
    "게이트_원자료": f"{B68}/out/run068.json · {B68}/out/ship068.json · stage1_partial.json",
    "에폭_곡선": f"{B68}/out/curves/curve_{{A,D}}_s{{seed}}.json",
    "로컬": "experiments/exp_068/ (metrics.json + src/)",
    "🔴 exp_067_산출물": "그대로 보존했다. 삭제·수정 없음."}

json.dump(M, open(f"{OUT}/metrics_core.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"k8_score": M["fold_2024"]["score"],
                  "k8_brier": M["fold_2024"]["brier"],
                  "k8_auc": M["fold_2024"]["auc"],
                  "robust": M["stage3_robustness"]["표"],
                  "vs_controls": M["fold_2024"]["vs_controls"],
                  "repro_pass": M["repro_gate"]["all_seeds_pass"],
                  "ship_pass": M["ship_format_gates"]["S1_npz_roundtrip"]["pass"],
                  "inference_s": M["inference_time_sec"]},
                 indent=1, ensure_ascii=False))

"""exp_063 — 85열 NN 예측과 기존 팔들의 상관. fold_2024 valid_all 253,507행.

exp_061 src/corr.py 승계. 추가한 것 하나: **99열 NN(exp_061 25ep 파일럿 4시드
로짓평균)** 을 계열에 넣어 `corr(NN85, NN99)` 를 잰다 — 열 14개 제거가 팔을
얼마나 움직였는가.

⚠️ 눈금 주의 (이 파일이 단정하는 것은 자기가 계산한 상관·Brier 뿐):
  · exp_048/exp_049 A1@1400 = **out-of-time**(<=2023 학습) — NN 과 같은 눈금.
  · ~/SungjinPark/arms/* (X_arms 5팔) = mate 패키지 배포본 계측 = **tr2024 표본 내**.
    상관 자체는 읽을 수 있으나 그 팔들의 점수는 out-of-time 이 아니다.
  · 행 순서는 arms/ids.npy 와 frame99 season==2024 의 row_id 가 일치함을 assert.
  · S14 상시 단서: `arms/p_ebm.npy` 는 이름과 달리 로짓이 아니라 확률이 아닐 수
    있으므로 X_arms 의 저장 형태를 그대로 쓴다(exp_061 corr.py 와 동일 취급).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

H = os.path.expanduser("~")
OUT = f"{L.B}/out"
E61 = f"{H}/SungjinPark/exp061_nn/out2"

NN85 = np.load(f"{OUT}/preds/A85_seedavg.npy")
NN99 = np.load(f"{E61}/preds/A_pilot_seedavg.npy")

df = pd.read_parquet(f"{L.EXP60}/cache/frame99.parquet",
                     columns=["season", "game_type", "control_success", "row_id"])
v = df[df.season == 2024]
y = v.control_success.to_numpy(np.float64)
assert len(v) == len(NN85) == len(NN99) == 253507

ids = np.load(f"{H}/SungjinPark/arms/ids.npy", allow_pickle=True)
assert np.array_equal(np.asarray(ids).astype(str), v.row_id.to_numpy().astype(str)), \
    "arms/ids.npy 행 순서가 frame99 2024 와 다르다"

X = np.load(f"{H}/SungjinPark/arms/X_arms.npy")     # [lg_v6, lg_Q, p_EBM, lg_AA, off]
coef = np.load(f"{H}/SungjinPark/arms/coef_now.npy")
assert X.shape == (5, 253507)


def mean3(exp, arm):
    return np.mean([np.load(f"{H}/SungjinPark/experiments/{exp}/pred/"
                            f"pred_{arm}_u2024_seed{s}_at1400.npy")
                    for s in (42, 7, 2024)], axis=0)


AA67 = mean3("exp_048", "A1")
AA81 = mean3("exp_049", "A1")

series = {
    "NN85_seedavg(exp063, out-of-time)": L.lg(NN85),
    "NN99_seedavg(exp061 25ep pilot, out-of-time)": L.lg(NN99),
    "AA_67col_A1@1400_3seed(exp048, out-of-time)": L.lg(AA67),
    "AA_81col_A1@1400_3seed(exp049, out-of-time)": L.lg(AA81),
    "lg_p_v6(arms, tr2024 표본내)": X[0],
    "lg_p_Q(arms, tr2024 표본내)": X[1],
    "p_EBM(arms, tr2024 표본내)": X[2],
    "lg_p_AA_deployed(arms, tr2024 표본내)": X[3],
    "off(arms)": X[4],
    "blend_now_logit_ex_d(arms, 합성 = coef_now·X)": coef @ X,
}
names = list(series)
Mtx = np.corrcoef(np.vstack([series[n] for n in names]))
res = {"n_rows": int(len(v)),
       "space": "로짓 (블렌드가 로짓에서 더해지므로 이 공간에서 잰다)",
       "coef_now": coef.tolist(),
       "corr_matrix": {a: {b: float(Mtx[i, j]) for j, b in enumerate(names)}
                       for i, a in enumerate(names)},
       "NN85_vs": {n: float(Mtx[0, i]) for i, n in enumerate(names) if i},
       "NN99_vs": {n: float(Mtx[1, i]) for i, n in enumerate(names) if i != 1}}

# 🔴 이 실험의 핵심 비교 — 14열 제거가 상관을 올렸는가 내렸는가
key = {
    "AA_67col_A1@1400_3seed(exp048, out-of-time)": "AA67",
    "AA_81col_A1@1400_3seed(exp049, out-of-time)": "AA81",
    "lg_p_Q(arms, tr2024 표본내)": "team_Q",
    "blend_now_logit_ex_d(arms, 합성 = coef_now·X)": "blend_now",
    "lg_p_AA_deployed(arms, tr2024 표본내)": "AA_deployed",
    "p_EBM(arms, tr2024 표본내)": "p_EBM",
    "lg_p_v6(arms, tr2024 표본내)": "v6",
}
res["corr_shift_85_minus_99"] = {
    short: {"nn85": float(Mtx[0, names.index(full)]),
            "nn99": float(Mtx[1, names.index(full)]),
            "delta": float(Mtx[0, names.index(full)] - Mtx[1, names.index(full)])}
    for full, short in key.items()}
res["corr_NN85_vs_NN99"] = float(Mtx[0, 1])

# 확률 공간 상관 (참고) + out-of-time 팔들의 fold_2024 Brier
prob = {"NN85_seedavg": NN85, "NN99_seedavg": NN99,
        "AA_67col(exp048)": AA67, "AA_81col(exp049)": AA81}
pn = list(prob)
Mp = np.corrcoef(np.vstack([prob[n] for n in pn]))
res["corr_prob_space_out_of_time_only"] = {
    a: {b: float(Mp[i, j]) for j, b in enumerate(pn)} for i, a in enumerate(pn)}
res["brier_out_of_time"] = {
    n: {"brier_of_averaged_pred": float(np.mean((p - y) ** 2)),
        "score_of_averaged_pred": L.conv(float(np.mean((p - y) ** 2)))}
    for n, p in prob.items()}
res["reference_check"] = {
    "corr(lg_p_AA_deployed, lg_p_Q) 기대 0.9776": float(
        np.corrcoef(X[3], X[1])[0, 1]),
    "corr(NN99, AA67) 기대 0.9014 (exp_061 실측 0.9013795597911097)": float(
        Mtx[1, names.index("AA_67col_A1@1400_3seed(exp048, out-of-time)")]),
    "note": "두 값이 재현되면 상관 산출 방식이 exp_061·장부와 같은 눈금"}

gt = v.game_type.to_numpy()
res["NN85_vs_by_segment"] = {}
for nm, mk in (("R", gt == "R"), ("F", gt == "F")):
    res["NN85_vs_by_segment"][nm] = {
        n: float(np.corrcoef(series[names[0]][mk], series[n][mk])[0, 1])
        for n in names[1:]}
json.dump(res, open(f"{OUT}/corr.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"corr_NN85_vs_NN99": res["corr_NN85_vs_NN99"],
                  "corr_shift_85_minus_99": res["corr_shift_85_minus_99"],
                  "brier_out_of_time": res["brier_out_of_time"],
                  "reference_check": res["reference_check"]},
                 indent=1, ensure_ascii=False))

"""exp_064 — 배포 p6 99열 NN 예측과 기존 팔들의 상관. fold_2024 valid_all 253,507행.

exp_063 src/corr85.py 승계. 계열에 셋을 나란히 둔다:
  NN_ship (이번), NN99 (exp_061 우리 p6), NN85 (exp_063 14열 제거).

⚠️ 눈금 주의:
  · exp_048/exp_049 A1@1400 = out-of-time(<=2023 학습) — NN 과 같은 눈금.
  · ~/SungjinPark/arms/* (X_arms 5팔) = mate 패키지 배포본 계측 = tr2024 표본 내.
    상관은 읽을 수 있으나 그 팔들의 점수는 out-of-time 이 아니다.
  · S14: arms/* 벡터는 이름이 아니라 저장 형태 그대로 쓴다(X_arms 원본 사용).
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
E63 = f"{H}/SungjinPark/exp063_nn85/out"

NNS = np.load(f"{OUT}/preds/A99ship_seedavg.npy")
NN99 = np.load(f"{E61}/preds/A_pilot_seedavg.npy")
NN85 = np.load(f"{E63}/preds/A85_seedavg.npy")

df = pd.read_parquet(f"{L.EXP60}/cache/frame99.parquet",
                     columns=["season", "game_type", "control_success", "row_id"])
v = df[df.season == 2024]
y = v.control_success.to_numpy(np.float64)
assert len(v) == len(NNS) == len(NN99) == len(NN85) == 253507

ids = np.load(f"{H}/SungjinPark/arms/ids.npy", allow_pickle=True)
assert np.array_equal(np.asarray(ids).astype(str), v.row_id.to_numpy().astype(str))
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
    "NN_ship_seedavg(exp064, out-of-time)": L.lg(NNS),
    "NN99_seedavg(exp061 우리 p6, out-of-time)": L.lg(NN99),
    "NN85_seedavg(exp063 14열 제거, out-of-time)": L.lg(NN85),
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
idx = {n: i for i, n in enumerate(names)}
res = {"n_rows": int(len(v)),
       "space": "로짓 (블렌드가 로짓에서 더해지므로 이 공간에서 잰다)",
       "coef_now": coef.tolist(),
       "corr_matrix": {a: {b: float(Mtx[i, j]) for j, b in enumerate(names)}
                       for i, a in enumerate(names)},
       "NN_ship_vs": {n: float(Mtx[0, idx[n]]) for n in names[1:]}}

key = {
    "AA_67col_A1@1400_3seed(exp048, out-of-time)": "AA67",
    "AA_81col_A1@1400_3seed(exp049, out-of-time)": "AA81",
    "lg_p_Q(arms, tr2024 표본내)": "team_Q",
    "p_EBM(arms, tr2024 표본내)": "p_EBM",
    "blend_now_logit_ex_d(arms, 합성 = coef_now·X)": "blend_now",
    "lg_p_AA_deployed(arms, tr2024 표본내)": "AA_deployed",
    "lg_p_v6(arms, tr2024 표본내)": "v6",
}
res["corr_three_way"] = {
    short: {"nn_ship": float(Mtx[0, idx[full]]),
            "nn99_ours_p6": float(Mtx[1, idx[full]]),
            "nn85_dropped": float(Mtx[2, idx[full]]),
            "ship_minus_99": float(Mtx[0, idx[full]] - Mtx[1, idx[full]]),
            "ship_minus_85": float(Mtx[0, idx[full]] - Mtx[2, idx[full]])}
    for full, short in key.items()}
res["corr_between_NNs"] = {
    "ship_vs_ours99": float(Mtx[0, 1]), "ship_vs_85": float(Mtx[0, 2]),
    "ours99_vs_85": float(Mtx[1, 2])}

prob = {"NN_ship_seedavg": NNS, "NN99_seedavg": NN99, "NN85_seedavg": NN85,
        "AA_67col(exp048)": AA67, "AA_81col(exp049)": AA81}
pn = list(prob)
Mp = np.corrcoef(np.vstack([prob[n] for n in pn]))
res["corr_prob_space_out_of_time_only"] = {
    a: {b: float(Mp[i, j]) for j, b in enumerate(pn)} for i, a in enumerate(pn)}
res["brier_out_of_time"] = {
    n: {"brier": float(np.mean((p - y) ** 2)),
        "score": L.conv(float(np.mean((p - y) ** 2)))} for n, p in prob.items()}
res["reference_check"] = {
    "corr(lg_p_AA_deployed, lg_p_Q) 기대 0.9776": float(np.corrcoef(X[3], X[1])[0, 1]),
    "corr(NN99, AA67) 기대 0.9013795597911097(exp_061)":
        float(Mtx[1, idx["AA_67col_A1@1400_3seed(exp048, out-of-time)"]]),
    "corr(NN85, AA67) 기대 0.9021 근방(exp_063)":
        float(Mtx[2, idx["AA_67col_A1@1400_3seed(exp048, out-of-time)"]]),
    "note": "재현되면 상관 산출 눈금이 exp_061·exp_063·장부와 같다"}

gt = v.game_type.to_numpy()
res["NN_ship_vs_by_segment"] = {}
for nm, mk in (("R", gt == "R"), ("F", gt == "F")):
    res["NN_ship_vs_by_segment"][nm] = {
        n: float(np.corrcoef(series[names[0]][mk], series[n][mk])[0, 1])
        for n in names[1:]}
json.dump(res, open(f"{OUT}/corr.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"corr_between_NNs": res["corr_between_NNs"],
                  "corr_three_way": res["corr_three_way"],
                  "brier_out_of_time": res["brier_out_of_time"],
                  "reference_check": res["reference_check"]},
                 indent=1, ensure_ascii=False))

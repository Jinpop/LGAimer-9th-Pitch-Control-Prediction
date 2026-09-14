"""exp_061 — NN 예측과 기존 팔들의 상관. fold_2024 valid_all 253,507행.

⚠️ 눈금 주의 (이 파일이 단정하는 것은 자기가 계산한 상관·Brier 뿐):
  · exp_048/exp_049 A1@1400 = **out-of-time**(<=2023 학습) — NN 과 같은 눈금.
  · ~/SungjinPark/arms/* (X_arms 5팔) = mate 패키지 배포본 계측 = **tr2024 표본 내**.
    상관 자체는 읽을 수 있으나 그 팔들의 점수는 out-of-time 이 아니다.
  · 행 순서는 arms/ids.npy 와 frame99 season==2024 의 row_id 가 일치함을 assert 로 확인.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

H = os.path.expanduser("~")
if os.environ.get("EXP061_OUT"):
    L.OUT = f"{L.B}/" + os.environ["EXP061_OUT"]
OUT = L.OUT
NNP = os.environ.get("EXP061_NN_PRED", f"{OUT}/preds/A_pick_seedavg.npy")
CORRNAME = os.environ.get("EXP061_CORR_NAME", "corr.json")
NN = np.load(NNP)

df = pd.read_parquet(f"{L.EXP60}/cache/frame99.parquet",
                     columns=["season", "game_type", "control_success", "row_id"])
v = df[df.season == 2024]
y = v.control_success.to_numpy(np.float64)
assert len(v) == len(NN) == 253507

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

series = {
    "NN99_seedavg(exp061, out-of-time)": L.lg(NN),
    "AA_67col_A1@1400_3seed(exp048, out-of-time)": L.lg(mean3("exp_048", "A1")),
    "AA_81col_A1@1400_3seed(exp049, out-of-time)": L.lg(mean3("exp_049", "A1")),
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
       "space": "logit/score (블렌드가 로짓에서 더해지므로 이 공간에서 잰다)",
       "coef_now": coef.tolist(),
       "corr_matrix": {a: {b: float(Mtx[i, j]) for j, b in enumerate(names)}
                       for i, a in enumerate(names)},
       "NN_vs": {n: float(Mtx[0, i]) for i, n in enumerate(names) if i}}

# 확률 공간 상관 (참고) + 각 팔의 fold_2024 Brier
prob = {"NN99_seedavg": NN,
        "AA_67col(exp048)": mean3("exp_048", "A1"),
        "AA_81col(exp049)": mean3("exp_049", "A1")}
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
    "note": "이 값이 0.9776 을 재현하면 상관 산출 방식이 장부와 같은 것"}
gt = v.game_type.to_numpy()
res["NN_vs_by_segment"] = {}
for nm, mk in (("R", gt == "R"), ("F", gt == "F")):
    res["NN_vs_by_segment"][nm] = {
        n: float(np.corrcoef(series[names[0]][mk], series[n][mk])[0, 1])
        for n in names[1:]}
res["nn_pred_path"] = NNP
json.dump(res, open(f"{OUT}/{CORRNAME}", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"NN_vs": res["NN_vs"],
                  "brier_out_of_time": res["brier_out_of_time"],
                  "reference_check": res["reference_check"]},
                 indent=1, ensure_ascii=False))

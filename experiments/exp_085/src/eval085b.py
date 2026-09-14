"""exp_085 stage 3b — 부가 측정 보강. 관문 판정은 건드리지 않는다(eval085.py 가 정본).

왜 필요한가: eval085 의 `on_NN_AA` 기준선은 **4시드 평균 NN** 이고 새 팔은 **1시드**다.
시드 평균은 그 자체로 +점을 사므로(S18: score(n)=887.58−35.99/n) 같은 변동 등급에서
다시 재야 읽을 수 있다. 여기서 재는 것:
  (a) 기준선 = NN 1시드 단독
  (b) 기준선 = NN 1시드 + AA67, w 는 그 쌍의 자체 최적
  (c) 3원 기준선 위에 새 팔 **둘 다** (2차원 격자)
  (d) 4시드 NN+AA 기준선 위에 새 팔 둘 다
  (e) 새 팔끼리의 상관
"""
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp085/out"
REF = 0.2498069
K = 100000.0 / REF
EPS = 1e-6
rng = np.random.default_rng(42)


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["season", "control_success", "pitcher_id"])
v = tr[tr.season == 2024]
y = v.control_success.to_numpy(np.float64)
N = len(y)
blk = pd.factorize(v.pitcher_id)[0]
nb = int(blk.max()) + 1


def brier(p):
    return float(np.mean((np.asarray(p, np.float64) - y) ** 2))


def se_of(d):
    bs = np.bincount(blk, weights=d, minlength=nb)
    tot = bs[rng.integers(0, nb, size=(1000, nb))].sum(axis=1)
    return float(tot.std(ddof=1)) * K / N


A = {
    "AA67": np.load(f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed42_at1400.npy"),
    "AA81": np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed42_at1400.npy"),
    "S1": np.load(f"{H}/experiments/exp_053/pred/pred_S1_u2024_seed42_at1400.npy"),
    "S2": np.load(f"{H}/experiments/exp_053/pred/pred_S2_u2024_seed42_at1400.npy"),
    "T70": np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed42_at1400.npy"),
    "NN_s42": np.load(f"{H}/exp064/out/preds/A99ship_s42.npy"),
    "NN_avg4": np.load(f"{H}/exp064/out/preds/A99ship_seedavg.npy"),
    "FM": np.load(f"{OUT}/preds/F1_FM_s42_r2_best.npy"),
    "EMB": np.load(f"{OUT}/preds/F2_EMB_s42_r2_best.npy"),
}
Z = {k: lg(a.astype(np.float64)) for k, a in A.items()}
M = {"note": "관문 판정 아님 — 다음 단계 설계용 부가 측정"}

grid = np.round(np.arange(0.0, 0.801, 0.005), 4)


def add_one(zbase, name):
    b0 = brier(expit(zbase))
    out = {"base_score": conv(b0)}
    for k in ("FM", "EMB"):
        g = np.array([K * (b0 - brier(expit((1 - w) * zbase + w * Z[k])))
                      for w in grid])
        i = int(np.argmax(g))
        pm = expit((1 - grid[i]) * zbase + grid[i] * Z[k])
        se = se_of((expit(zbase) - y) ** 2 - (pm - y) ** 2)
        out[k] = {"w_star": float(grid[i]), "gain_pts": float(g[i]),
                  "se_pts": se, "t": float(g[i] / se) if se else 0.0,
                  "blend_score": conv(brier(pm)),
                  "w_at_cap": bool(grid[i] >= 0.8 - 1e-9)}
    return out


# (a) NN 1시드 단독
M["base_NN_s42_alone"] = add_one(Z["NN_s42"], "NN_s42")

# (b) NN 1시드 + AA67, 그 쌍의 자체 최적 w
b_aa = brier(A["AA67"])
best = (0.0, np.inf)
for w in grid:
    b = brier(expit((1 - w) * Z["AA67"] + w * Z["NN_s42"]))
    if b < best[1]:
        best = (float(w), b)
w_nn, b_pair = best
zpair = (1 - w_nn) * Z["AA67"] + w_nn * Z["NN_s42"]
M["base_NN_s42_plus_AA67"] = {"w_NN": w_nn, "pair_score": conv(b_pair)}
M["base_NN_s42_plus_AA67"].update(add_one(zpair, "pair"))

# (c)(d) 두 새 팔 동시 투입 — 2차원 격자
g2 = np.round(np.arange(0.0, 0.601, 0.01), 4)
for tag, zbase in (("joint_on_3way",
                    0.50 * Z["AA67"] + 0.28 * Z["S2"] + 0.22 * Z["S1"]),
                   ("joint_on_NNavg4_AA67",
                    0.75 * Z["NN_avg4"] + 0.25 * Z["AA67"]),
                   ("joint_on_NN_s42", Z["NN_s42"])):
    b0 = brier(expit(zbase))
    bestj = {"gain_pts": -1e18}
    for wf in g2:
        for we in g2:
            if wf + we > 0.8:
                continue
            z = (1 - wf - we) * zbase + wf * Z["FM"] + we * Z["EMB"]
            g = K * (b0 - brier(expit(z)))
            if g > bestj["gain_pts"]:
                bestj = {"w_FM": float(wf), "w_EMB": float(we), "gain_pts": float(g)}
    z = ((1 - bestj["w_FM"] - bestj["w_EMB"]) * zbase
         + bestj["w_FM"] * Z["FM"] + bestj["w_EMB"] * Z["EMB"])
    pm = expit(z)
    bestj["se_pts"] = se_of((expit(zbase) - y) ** 2 - (pm - y) ** 2)
    bestj["t"] = bestj["gain_pts"] / bestj["se_pts"] if bestj["se_pts"] else 0.0
    bestj["base_score"] = conv(b0)
    bestj["blend_score"] = conv(brier(pm))
    M[tag] = bestj

# (e) 새 팔끼리 + 전 팔 상관 한 줄
names = list(Z)
C = np.corrcoef(np.vstack([Z[n] for n in names]))
ix = {n: i for i, n in enumerate(names)}
M["corr_FM_EMB"] = float(C[ix["FM"], ix["EMB"]])
M["corr_new_vs_all"] = {k: {n: float(C[ix[k], ix[n]]) for n in names if n != k}
                        for k in ("FM", "EMB")}

json.dump(M, open(f"{OUT}/metrics_addendum.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))

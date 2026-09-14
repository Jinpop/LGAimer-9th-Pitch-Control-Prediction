"""exp_096 진단 — **독립 fold** 준비 (train <= 2022 -> valid 2023).

왜: rank 팔의 이득이 fold_2024 하나에만 얹힌 것인지 보려면 다른 fold 가 필요하다.
    이 fold 에는 z_mate·z_aa·z_T70 이 없으므로 **혼합 이득은 못 잰다** — 잴 수 있는 것은
    같은 구조·같은 예산에서 `rank` 입력이 `표준화` 입력보다 **팔 단독으로 나은가** 하나다.
    그것만으로도 "입력 변환 자체가 값을 한다"의 독립 확인이 된다.

exp_089 prep089.py 의 규약을 그대로 옮긴다.  달라지는 것은 구간뿐:
    학습 : (season <= 2022) & ~((season <= 2021) & game_type == "F")
    평가 : season == 2023
표준화 mu/sd · 범주 어휘 · 분위표를 **전부 이 학습 구간에서만** 만든다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.special import ndtri

H = "/home/<user>/SungjinPark"
B = f"{H}/exp096"
E60 = f"{H}/exp060_nn"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
W = f"{B}/workb"
os.makedirs(W, exist_ok=True)
t0 = time.time()
NQ = 1001

P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
NUM, CAT = P["NUM"], P["CAT"]
assert len(NUM) == 94 and len(CAT) == 5
df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + CAT + ["row_id", "season"])))
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success"])
N = len(df)
assert len(tr) == N == 1475092
assert np.array_equal(df.row_id.to_numpy().astype(str),
                      tr.row_id.to_numpy().astype(str))
season = tr.season.to_numpy(np.int64)
gt = tr.game_type.to_numpy().astype(str)
y = tr.control_success.to_numpy(np.float32)
pid_all = pd.read_csv(f"{H}/data/train.csv", usecols=["pitcher_id"]).pitcher_id.to_numpy()

trm = (season <= 2022) & ~((season <= 2021) & (gt == "F"))
vam = season == 2023
Xraw = df[NUM].astype("float32").to_numpy()
mu = np.nanmean(Xraw[trm], 0).astype(np.float32)
sd = np.nanstd(Xraw[trm], 0).astype(np.float32)
sd = np.where(sd < 1e-6, np.float32(1.0), sd)
with np.errstate(invalid="ignore"):
    X = (Xraw - mu) / sd
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

grid = np.linspace(0.0, 1.0, NQ)
Xr = np.empty_like(X)
for j in range(X.shape[1]):
    col = X[:, j].astype(np.float64)
    q = np.quantile(col[trm], grid)                 # 🔴 학습 구간 행만
    if q[-1] - q[0] < 1e-12:
        Xr[:, j] = 0.0
        continue
    u = np.clip(np.interp(col, q, grid), 1.0 / (2 * NQ), 1.0 - 1.0 / (2 * NQ))
    Xr[:, j] = ndtri(u).astype(np.float32)

CS = {c: df[c].astype(str) for c in CAT}
levels = {c: sorted(map(str, pd.unique(CS[c][trm]))) for c in CAT}
codes = np.zeros((N, len(CAT)), np.int32)
for j, c in enumerate(CAT):
    lut = {v: i + 1 for i, v in enumerate(levels[c])}
    codes[:, j] = CS[c].map(lut).fillna(0).astype(np.int64).to_numpy()

np.save(f"{W}/X_le22.npy", X)
np.save(f"{W}/X_rank_le22.npy", Xr)
np.save(f"{W}/codes_le22.npy", codes)
np.save(f"{W}/trm.npy", trm)
np.save(f"{W}/vam.npy", vam)
np.save(f"{W}/y.npy", y)
np.save(f"{W}/pit_valid.npy", pid_all[vam])
json.dump({"cards": [len(levels[c]) + 1 for c in CAT], "CAT": CAT},
          open(f"{W}/schema.json", "w"), ensure_ascii=False, indent=1)
M = {"exp_id": "exp_096", "stage": "foldb_prep",
     "train_rule": "(season<=2022) & ~((season<=2021) & game_type=='F')",
     "valid_rule": "season == 2023",
     "n_train": int(trm.sum()), "n_valid": int(vam.sum()),
     "cards": [len(levels[c]) + 1 for c in CAT],
     "🔴_stats_source": "mu/sd/vocab/quantiles 전부 학습 구간(trm)에서만",
     "🔴_valid_rows_used_for_any_table": 0,
     "wall_sec": round(time.time() - t0, 1)}
json.dump(M, open(f"{B}/out/foldb_prep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))

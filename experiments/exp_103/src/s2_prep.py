"""exp_103 stage2 prep — 딥 팔 입력을 94 → 111 열로 넓힌다.

exp_096 prep_f64.py 의 규약을 글자 그대로 승계한다:
  · mu/sd 는 **float64 누산**, **학습 구간(trm_oot23) 행에서만** (2024 행 0)
  · 표준화 뒤 nan/inf → 0 (딥 팔은 NaN 을 못 먹는다.
    🔴 이건 피처 설계의 sentinel 채우기가 아니라 신경망 입력 규약이고,
       결측은 `bat_form_miss`·`bat_cs_short` 두 플래그 열이 별도로 들고 있다.)
새 17열은 NUM 목록 **뒤에 붙인다** — 앞 94열의 mu/sd·순서가 바뀌지 않아
exp_097 판본과 앞부분이 비트 동일해야 한다(게이트로 확인).
"""
import json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "3"
import numpy as np, pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp103"
W89, W97 = f"{H}/exp089/work", f"{H}/exp097/work"
E60, PKG = f"{H}/exp060_nn", f"{H}/audit_deep/pkg/DEEP/mate"
t0 = time.time()
P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
NUM, CAT = P["NUM"], P["CAT"]
FCOLS = json.load(open(f"{B}/work/batform2_cols.json"))["cols"]

df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUM + ["row_id"])))
tr = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id"], encoding="utf-8-sig")
assert np.array_equal(df.row_id.to_numpy().astype(str), tr.row_id.to_numpy().astype(str))
F = np.load(f"{B}/work/batform2.npy")
Xraw = np.hstack([df[NUM].astype("float32").to_numpy(), F]).astype(np.float32)
assert Xraw.shape[1] == len(NUM) + len(FCOLS)

trm = np.load(f"{W97}/trm_oot23.npy")
vam = np.load(f"{W97}/valid_mask.npy")
assert int((trm & vam).sum()) == 0
mu = np.nanmean(Xraw[trm].astype(np.float64), axis=0)
sd = np.nanstd(Xraw[trm].astype(np.float64), axis=0)
sd = np.where(sd < 1e-6, 1.0, sd)
with np.errstate(invalid="ignore"):
    X = (Xraw - mu.astype(np.float32)) / sd.astype(np.float32)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
np.save(f"{B}/work/X111_f64o.npy", X)

# 🔴 게이트: 앞 94열이 exp_097 의 f64 판본과 비트 동일해야 한다 (변인은 뒤 17열뿐)
X97 = np.load(f"{W97}/X_f64o.npy", mmap_mode="r")
d = np.abs(np.asarray(X97, np.float64) - X[:, :94].astype(np.float64))
M = {"exp_id": "exp_103", "stage": "s2_prep",
     "n_cols": int(X.shape[1]), "n_num_base": len(NUM), "n_num_new": len(FCOLS),
     "new_cols": FCOLS,
     "first94_max_abs_diff_vs_exp097_X_f64o": float(d.max()),
     "first94_bitwise_identical": bool((np.asarray(X97) == X[:, :94]).all()),
     "new_col_mu_sd": {c: {"mu": float(mu[94 + i]), "sd": float(sd[94 + i])}
                       for i, c in enumerate(FCOLS)},
     "mu_sd_source": "trm_oot23 행만 (2024 행 0)",
     "n_train_rows": int(trm.sum()),
     "wall_sec": round(time.time() - t0, 1)}
assert M["first94_bitwise_identical"], M["first94_max_abs_diff_vs_exp097_X_f64o"]
json.dump(M, open(f"{B}/out/s2_prep.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items() if k != "new_col_mu_sd"},
                 ensure_ascii=False, indent=1))

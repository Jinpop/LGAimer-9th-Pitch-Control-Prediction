"""exp_096 — A3 순위변환 입력 + A1/A0 파라미터수 맞춤.

A3: 수치 94열을 분위수-정규 점수로 바꾼다.
  🔴 분위표는 **학습 구간(oot23 = season<=2023, exp_089 마스크) 행에서만** 만든다.
     2024 행은 표를 만드는 데 한 톨도 안 쓴다 (동결표 → 전 행 적용).
  입력은 exp_089 의 X_oot23.npy (이미 표준화 + NaN→0).  순위는 단조 변환에 불변이므로
  표준화 위에서 순위를 매기는 것과 원값 위에서 매기는 것이 같다.  NaN 이었던 칸은
  '학습 구간 평균'의 순위를 받는다 — exp_089 의 평균 대입 규약과 같은 뜻이다.

A0: A1(FTAttn)과 파라미터 수가 가장 가까운 TokenMLP 의 D 를 찾는다 (S104 귀무 대조).
"""
import json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
from scipy.special import ndtri

H = "/home/<user>/SungjinPark"
B, W89 = f"{H}/exp096", f"{H}/exp089/work"
sys.path.insert(0, f"{B}/src")
os.makedirs(f"{B}/work", exist_ok=True)
t0 = time.time()
SPLIT = "oot23"
NQ = 1001

X = np.load(f"{W89}/X_{SPLIT}.npy")
trm = np.load(f"{W89}/trm_{SPLIT}.npy")
vam = np.load(f"{W89}/valid_mask.npy")
N, F = X.shape
grid = np.linspace(0.0, 1.0, NQ)
Xr = np.empty_like(X)
n_const = 0
qtab = np.empty((F, NQ), np.float64)
for j in range(F):
    col = X[:, j].astype(np.float64)
    q = np.quantile(col[trm], grid)
    qtab[j] = q
    if q[-1] - q[0] < 1e-12:
        Xr[:, j] = 0.0
        n_const += 1
        continue
    u = np.interp(col, q, grid)
    u = np.clip(u, 1.0 / (2 * NQ), 1.0 - 1.0 / (2 * NQ))
    Xr[:, j] = ndtri(u).astype(np.float32)
np.save(f"{B}/work/X_rank_{SPLIT}.npy", Xr)
np.save(f"{B}/work/qtab_{SPLIT}.npy", qtab)

M = {"exp_id": "exp_096", "stage": "prep",
     "rank_transform": {
         "n_cols": int(F), "n_quantile_knots": NQ,
         "quantiles_fit_on": "trm_oot23 (season<=2023, n=%d)" % int(trm.sum()),
         "🔴_2024_rows_used_for_table": 0,
         "n_constant_cols_zeroed": n_const,
         "out_mean_valid": float(Xr[vam].mean()),
         "out_sd_valid": float(Xr[vam].std()),
         "in_mean_valid": float(X[vam].mean()), "in_sd_valid": float(X[vam].std())}}

# ── 파라미터수 맞춤 ───────────────────────────────────────────────────
import torch                                                   # noqa: E402
from arch096 import TokenMLP, FTAttn, n_params                  # noqa: E402
sch = json.load(open(f"{W89}/schema_{SPLIT}.json"))
CARDS = sch["cards"]
A1 = FTAttn(F, CARDS, D=int(os.environ.get("ATT_D", "128")),
            blocks=int(os.environ.get("ATT_BLK", "3")),
            heads=int(os.environ.get("ATT_HEAD", "8")),
            mult=int(os.environ.get("ATT_MULT", "2")))
np_a1 = n_params(A1)
np_ref = n_params(TokenMLP(F, CARDS, D=96))
cand = {D: n_params(TokenMLP(F, CARDS, D=D)) for D in range(4, 97)}
best = min(cand, key=lambda D: abs(cand[D] - np_a1))
M["param_match"] = {
    "A1_attn_params": np_a1, "deep_ref_TokenMLP_D96_params": np_ref,
    "A0_ctrl_D": int(best), "A0_ctrl_params": int(cand[best]),
    "rel_gap_A0_vs_A1": float(abs(cand[best] - np_a1) / np_a1),
    "note": "A0 은 S104 귀무 대조 — '어텐션이라서'와 '파라미터가 그만큼이라서'를 가른다"}
M["wall_sec"] = round(time.time() - t0, 1)
json.dump(M, open(f"{B}/out/prep096.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(M, ensure_ascii=False, indent=1))

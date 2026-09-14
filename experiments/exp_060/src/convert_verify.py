"""exp_060 stage4 — torch state → 출하 npz 포맷 변환 + numpy 순전파 재현 검증.

키 규약(출하 mate/nn_infer.py): emb{i}, mlp_{0,4,8,10}_weight/bias,
mlp_{1,5}_weight/bias/running_mean/running_var(/num_batches_tracked).
검증: 같은 valid 2024 프레임에 대해 numpy 순전파(predict_nn 사본) vs torch 예측의
max_abs_diff. 추론 시간도 실측(참고치 — 게이트용 단독 실측은 본판에서).
"""
import json
import os
import time

import numpy as np
import pandas as pd
import torch

B = os.path.expanduser("~/SungjinPark/exp060_nn")
TAG = os.environ.get("EXP060_TAG", "")
SPD = os.path.expanduser("~/SungjinPark/sp_def_recv")
OUT = f"{B}/out"

state = torch.load(f"{OUT}/nn99_a42{TAG}.pt", map_location="cpu")
M = json.load(open(f"{OUT}/nn99_a42{TAG}_meta.json"))
CATS, DENSE = M["CATS"], M["DENSE"]

npz = {}
for i in range(len(CATS)):
    npz[f"emb{i}"] = state[f"embs.{i}.weight"].numpy()
for i in (0, 4, 8, 10):
    npz[f"mlp_{i}_weight"] = state[f"mlp.{i}.weight"].numpy()
    npz[f"mlp_{i}_bias"] = state[f"mlp.{i}.bias"].numpy()
for i in (1, 5):
    npz[f"mlp_{i}_weight"] = state[f"mlp.{i}.weight"].numpy()
    npz[f"mlp_{i}_bias"] = state[f"mlp.{i}.bias"].numpy()
    npz[f"mlp_{i}_running_mean"] = state[f"mlp.{i}.running_mean"].numpy()
    npz[f"mlp_{i}_running_var"] = state[f"mlp.{i}.running_var"].numpy()
    npz[f"mlp_{i}_num_batches_tracked"] = state[f"mlp.{i}.num_batches_tracked"].numpy()
np.savez(f"{OUT}/nn99_a42{TAG}.npz", **npz)
print("npz keys:", sorted(npz.keys()))


# ---- 출하 nn_infer.predict_nn 사본 (mate/nn_infer.py와 동일 산술) ----
def _relu(z):
    return np.maximum(z, 0.0)


def predict_nn(df, npz_path, meta_path, chunk=100000):
    W = np.load(npz_path)
    Mm = json.load(open(meta_path))
    CATSl, DENSEl = Mm["CATS"], Mm["DENSE"]
    mu = np.array(Mm["mu"], dtype=np.float32)
    sd = np.array(Mm["sd"], dtype=np.float32)
    codes = np.zeros((len(df), len(CATSl)), dtype=np.int64)
    for i, c in enumerate(CATSl):
        lut = {v: j + 1 for j, v in enumerate(Mm["levels"][c])}
        codes[:, i] = df[c].astype(str).map(lut).fillna(0).astype(np.int64).values
    X = df[DENSEl].astype("float32").values
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    embs = [W[f"emb{i}"] for i in range(len(CATSl))]
    lin = lambda z, i: z @ W[f"mlp_{i}_weight"].T + W[f"mlp_{i}_bias"]  # noqa: E731

    def bn(z, i):
        g = W[f"mlp_{i}_weight"]
        b = W[f"mlp_{i}_bias"]
        m = W[f"mlp_{i}_running_mean"]
        v = W[f"mlp_{i}_running_var"]
        return (z - m) / np.sqrt(v + 1e-5) * g + b

    out = np.empty(len(df), dtype=np.float64)
    for s in range(0, len(df), chunk):
        e = min(s + chunk, len(df))
        z = np.concatenate([embs[i][codes[s:e, i]] for i in range(len(CATSl))]
                           + [X[s:e]], axis=1)
        z = lin(z, 0)
        z = bn(z, 1)
        z = _relu(z)
        z = lin(z, 4)
        z = bn(z, 5)
        z = _relu(z)
        z = lin(z, 8)
        z = _relu(z)
        z = lin(z, 10)
        out[s:e] = 1.0 / (1.0 + np.exp(-z.ravel()))
    return out


df = pd.read_parquet(f"{B}/cache/frame99.parquet")
SP6 = [c for c in DENSE if c.startswith("sp_")]
df[SP6] = df[SP6].fillna(0.0)
dv = df[df.season == 2024]
assert len(dv) == 253507

t0 = time.time()
p_np = predict_nn(dv, f"{OUT}/nn99_a42{TAG}.npz", f"{OUT}/nn99_a42{TAG}_meta.json")
t_inf = time.time() - t0
p_torch = np.load(f"{OUT}/pred_valid2024_seed42{TAG}.npy")
d = np.abs(p_np - p_torch)
yv = dv["control_success"].to_numpy(np.float64)
res = {"max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
       "numpy_brier": float(np.mean((p_np - yv) ** 2)),
       "torch_brier": float(np.mean((p_torch - yv) ** 2)),
       "numpy_inference_sec_253507rows": round(t_inf, 2)}
json.dump(res, open(f"{OUT}/verify_seed42{TAG}.json", "w"), indent=1)
print(json.dumps(res, indent=1))
print("convert_verify done", flush=True)

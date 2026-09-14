"""exp_062 step4 — 44열 차이가 배포 NN 출력에 얼마를 움직이는가.

추론기는 **출하본** `mate/nn_infer.predict_nn` 을 그대로 쓴다(배포 경로와 동일 코드).
모델은 exp_061 배포 npz 4시드(≤2024 재적합).

입력 3종 (나머지 55열·범주 5열은 전부 동일, 44열만 교체):
  T  = frame99 저장본           (우리 ≤2023 표 = **학습 시 입력**)
  O  = 우리 ≤2024 표
  S  = 출하 ≤2024 표            (**배포 때 NN 이 실제로 받게 될 것**)
주 비교 = S vs O (as-of 일치). 참고 = S vs T, O vs T.

⚠️ 배포 npz 는 ≤2024 로 적합돼 2024행이 표본 내다 — Brier 의 **수준**은 점수가 아니다.
   두 입력의 **차**만 읽는다.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import numpy as np
import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp062_gate")
B60 = os.path.expanduser("~/SungjinPark/exp060_nn")
SHIP = f"{B}/ship/mate"
sys.path.insert(0, SHIP)
import nn_infer  # noqa: E402  (출하본 numpy 추론기)

REF = 0.2498069
SC = 100000.0 / REF            # Brier 1.0 당 점수 (부호는 음)
SEEDS = [42, 777, 2024, 101]
MDIR = os.path.expanduser("~/SungjinPark/exp061_nn/out2")

f99 = pd.read_parquet(f"{B60}/cache/frame99.parquet")
f99 = f99[f99.season == 2024].reset_index(drop=True)
y = f99.control_success.values.astype("float64")
ours2024 = pd.read_parquet(f"{B}/work/f44_ours2024.parquet").reset_index(drop=True)
ship = pd.read_parquet(f"{B}/work/f44_ship.parquet").reset_index(drop=True)
COLS = [c for c in ours2024.columns if c != "row_id"]
assert np.array_equal(f99.row_id.values, ship.row_id.values)
assert np.array_equal(f99.row_id.values, ours2024.row_id.values)


def swap(base, src):
    d = base.copy()
    for c in COLS:
        d[c] = src[c].values.astype("float32")
    return d


FR = {"T": f99, "O": swap(f99, ours2024), "S": swap(f99, ship)}


def lg(p):
    return np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))


res = {"n_rows": int(len(f99)), "seeds": SEEDS, "per_seed": {}, "note":
       "배포 npz 는 ≤2024 적합 — 2024행 표본 내. Brier 수준은 점수가 아니고 차만 읽는다."}
L = {}
for k, fr in FR.items():
    ls = []
    for s in SEEDS:
        p = nn_infer.predict_nn(fr, f"{MDIR}/nn99_d{s}.npz", f"{MDIR}/nn99_d{s}_meta.json")
        ls.append(lg(p))
    L[k] = np.mean(ls, axis=0)
    L[k + "_perseed"] = ls
    pm = 1.0 / (1.0 + np.exp(-L[k]))
    res["per_seed"][k] = {"brier_seedavg": float(np.mean((pm - y) ** 2)),
                          "brier_each": [float(np.mean((1/(1+np.exp(-l)) - y) ** 2)) for l in ls]}
    print(f"{k}: brier(seedavg logit) = {res['per_seed'][k]['brier_seedavg']:.12f}", flush=True)


def pair(a, b):
    dl = L[a] - L[b]
    pa = 1.0 / (1.0 + np.exp(-L[a]))
    pb = 1.0 / (1.0 + np.exp(-L[b]))
    ba = float(np.mean((pa - y) ** 2))
    bb = float(np.mean((pb - y) ** 2))
    per = []
    for i, s in enumerate(SEEDS):
        d = L[a + "_perseed"][i] - L[b + "_perseed"][i]
        per.append({"seed": s, "logit_max_abs": float(np.abs(d).max()),
                    "logit_mean_abs": float(np.abs(d).mean())})
    return {"logit_max_abs": float(np.abs(dl).max()),
            "logit_mean_abs": float(np.abs(dl).mean()),
            "logit_rms": float(np.sqrt(np.mean(dl ** 2))),
            "prob_max_abs": float(np.abs(pa - pb).max()),
            "prob_mean_abs": float(np.abs(pa - pb).mean()),
            "rows_moved_gt_1e-9": int((np.abs(dl) > 1e-9).sum()),
            "brier_a": ba, "brier_b": bb, "brier_diff_a_minus_b": ba - bb,
            "score_diff_a_minus_b": -(ba - bb) * SC,
            "per_seed": per}


res["S_vs_O_asof_matched_MAIN"] = pair("S", "O")
res["S_vs_T_reference_asof_mismatched"] = pair("S", "T")
res["O_vs_T_reference_asof_mismatched"] = pair("O", "T")
json.dump(res, open(f"{B}/out/nn_delta.json", "w"), indent=1)
print(json.dumps({k: v for k, v in res.items()
                  if k.startswith(("S_", "O_"))}, indent=1)[:4000])

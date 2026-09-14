"""exp_064 stage 2-b — exp_062 step4 재실행. 44열 차이가 배포 NN 출력을 얼마나 움직이나.

추론기는 출하 `mate/nn_infer.predict_nn` 그대로. 모델은 exp_061 배포 npz 4시드.

입력 4종 (2024행 253,507. 44열만 갈아끼운다):
  T = frame99 저장본                       (우리 ≤2023 표 = exp_061 학습 입력)
  S = 출하 ≤2024 표 44열                    (배포 때 NN 이 받을 것) — exp_062 의 S
  N = 우리 ≤2024 표 + **배포 p6 3종** 44열   (이번 실험이 만든 것)
  V = frame99 에서 **14열만** 배포 p6 로 교체 (실제 stage 3 학습 프레임의 2024 슬라이스)

주 게이트 = N vs S. 사전 등록: |Brier 차| < 1점. 기대 정확히 0.00000.
참고 = V vs T (14열 교체가 배포 NN 을 얼마나 움직이는가), S vs T (exp_062 재현).

⚠️ 배포 npz 는 ≤2024 적합이라 2024행이 표본 내다 — Brier 의 **수준**은 점수가 아니다.
"""
import json
import os
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import numpy as np
import pandas as pd

H = os.path.expanduser("~")
B = f"{H}/SungjinPark/exp064"
E62 = f"{H}/SungjinPark/exp062_gate"
EXP60 = f"{H}/SungjinPark/exp060_nn"
SHIP = f"{E62}/ship/mate"
sys.path.insert(0, SHIP)
import nn_infer  # noqa: E402

REF = 0.2498069
SC = 100000.0 / REF
SEEDS = [42, 777, 2024, 101]
MDIR = f"{H}/SungjinPark/exp061_nn/out2"
THRESH_POINTS = 1.0
COLS14 = ["p6_curve", "p6_cutter", "p6_fastball", "p6_offspeed", "p6_sinker",
          "p6_slider", "i3_anom_curve", "i3_anom_cutter", "i3_anom_fastball",
          "i3_anom_offspeed", "i3_anom_sinker", "i3_anom_slider", "i3_anom_l1",
          "idea_matchup_pitch"]

f99 = pd.read_parquet(f"{EXP60}/cache/frame99.parquet")
f99 = f99[f99.season == 2024].reset_index(drop=True)
y = f99.control_success.values.astype("float64")
ship = pd.read_parquet(f"{E62}/work/f44_ship.parquet").reset_index(drop=True)
mine = pd.read_parquet(f"{B}/work/f44_shipp6_2024.parquet").reset_index(drop=True)
fs = pd.read_parquet(f"{B}/cache/frame99_ship.parquet")
fs = fs[fs.season == 2024].reset_index(drop=True)
COLS = [c for c in ship.columns if c != "row_id"]
assert len(COLS) == 44
for d in (ship, mine, fs):
    assert np.array_equal(f99.row_id.values, d.row_id.values)


def swap(base, src, cols):
    d = base.copy()
    for c in cols:
        d[c] = src[c].values.astype("float32")
    return d


FR = {"T": f99, "S": swap(f99, ship, COLS), "N": swap(f99, mine, COLS),
      "V": swap(f99, fs, COLS14)}


def lg(p):
    q = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(q / (1 - q))


res = {"n_rows": int(len(f99)), "seeds": SEEDS, "thresh_points": THRESH_POINTS,
       "note": "배포 npz 는 ≤2024 적합 — 2024행 표본 내. 차만 읽는다."}
L, per = {}, {}
for k, fr in FR.items():
    ls = [lg(nn_infer.predict_nn(fr, f"{MDIR}/nn99_d{s}.npz",
                                 f"{MDIR}/nn99_d{s}_meta.json")) for s in SEEDS]
    L[k] = np.mean(ls, axis=0)
    per[k] = ls
    pm = 1.0 / (1.0 + np.exp(-L[k]))
    res.setdefault("brier", {})[k] = float(np.mean((pm - y) ** 2))
    print(f"{k}: brier(seedavg logit) = {res['brier'][k]:.12f}", flush=True)


def pair(a, b):
    dl = L[a] - L[b]
    pa = 1.0 / (1.0 + np.exp(-L[a]))
    pb = 1.0 / (1.0 + np.exp(-L[b]))
    ba, bb = float(np.mean((pa - y) ** 2)), float(np.mean((pb - y) ** 2))
    return {"logit_max_abs": float(np.abs(dl).max()),
            "logit_mean_abs": float(np.abs(dl).mean()),
            "prob_max_abs": float(np.abs(pa - pb).max()),
            "rows_moved_gt_1e-9": int((np.abs(dl) > 1e-9).sum()),
            "brier_a": ba, "brier_b": bb, "brier_diff_a_minus_b": ba - bb,
            "score_diff_a_minus_b": -(ba - bb) * SC,
            "per_seed_logit_max_abs": [float(np.abs(per[a][i] - per[b][i]).max())
                                       for i in range(len(SEEDS))]}


res["MAIN_N_vs_S"] = pair("N", "S")
res["REF_V_vs_T_14col_swap_only"] = pair("V", "T")
res["REF_S_vs_T_exp062_repro"] = pair("S", "T")
res["exp062_recorded_S_vs_O_score_diff"] = -2.97796
res["prereg_check"] = {
    "cond_abs_score_diff_lt_1point":
        bool(abs(res["MAIN_N_vs_S"]["score_diff_a_minus_b"]) < THRESH_POINTS),
    "score_diff": res["MAIN_N_vs_S"]["score_diff_a_minus_b"],
    "exactly_zero": bool(res["MAIN_N_vs_S"]["brier_diff_a_minus_b"] == 0.0)}
json.dump(res, open(f"{B}/out/gate2_nn_delta.json", "w"), indent=1,
          ensure_ascii=False)
print(json.dumps({k: v for k, v in res.items()
                  if k.startswith(("MAIN", "REF", "prereg", "brier"))},
                 indent=1, ensure_ascii=False))
if not res["prereg_check"]["cond_abs_score_diff_lt_1point"]:
    raise SystemExit("🔴 STAGE2-b FAIL: |Brier 차| >= 1점")
print("GATE2B_OK", flush=True)

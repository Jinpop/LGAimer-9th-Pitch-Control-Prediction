"""exp_062 step5 — 어긋남이 어디에 몰려 있는가.

출하 pitchmodel6 은 pitcher_id 수준 571개, 우리 것은 685개(교집합 564, 출하 전용 7, 우리 전용 121).
수준 목록에 없는 pitcher_id 는 LightGBM 에서 결측 범주가 된다.
→ 2024행을 '양쪽이 아는 투수' / '한쪽만 아는 투수' 로 갈라 44열 차이와 Brier 차를 각각 잰다.
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
sys.path.insert(0, f"{B}/ship/mate")
import nn_infer  # noqa: E402

REF, SC = 0.2498069, 100000.0 / 0.2498069
SEEDS = [42, 777, 2024, 101]
MDIR = os.path.expanduser("~/SungjinPark/exp061_nn/out2")

sa = json.load(open(f"{B}/ship/mate/model/tables/pmcat6.json"))["pitcher_id"]
sb = json.load(open(f"{B}/work/tables_up2024/pmcat6.json"))["pitcher_id"]
S_ids, O_ids = set(map(int, sa)), set(map(int, sb))

f99 = pd.read_parquet(f"{B60}/cache/frame99.parquet")
f99 = f99[f99.season == 2024].reset_index(drop=True)
tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"), encoding="utf-8-sig")
cur = tr[tr.season == 2024].reset_index(drop=True)
assert np.array_equal(cur.row_id.values, f99.row_id.values)
pid = cur.pitcher_id.values
in_s = np.isin(pid, list(S_ids))
in_o = np.isin(pid, list(O_ids))
seg = {"both_known": in_s & in_o, "ours_only": ~in_s & in_o,
       "ship_only": in_s & ~in_o, "neither": ~in_s & ~in_o}

ours = pd.read_parquet(f"{B}/work/f44_ours2024.parquet").reset_index(drop=True)
ship = pd.read_parquet(f"{B}/work/f44_ship.parquet").reset_index(drop=True)
COLS = [c for c in ours.columns if c != "row_id"]

y = f99.control_success.values.astype("float64")


def swap(base, src):
    d = base.copy()
    for c in COLS:
        d[c] = src[c].values.astype("float32")
    return d


def lg(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


L = {}
for k, fr in [("O", swap(f99, ours)), ("S", swap(f99, ship))]:
    L[k] = np.mean([lg(nn_infer.predict_nn(fr, f"{MDIR}/nn99_d{s}.npz",
                                           f"{MDIR}/nn99_d{s}_meta.json")) for s in SEEDS], axis=0)
    np.save(f"{B}/out/logit_{k}.npy", L[k])

pS = 1 / (1 + np.exp(-L["S"]))
pO = 1 / (1 + np.exp(-L["O"]))

r = {"pitcher_levels": {"ship": len(S_ids), "ours": len(O_ids),
                        "common": len(S_ids & O_ids),
                        "ship_only": len(S_ids - O_ids), "ours_only": len(O_ids - S_ids)},
     "segments": {}}
for name, m in seg.items():
    n = int(m.sum())
    if n == 0:
        r["segments"][name] = {"n": 0}
        continue
    bS = float(np.mean((pS[m] - y[m]) ** 2))
    bO = float(np.mean((pO[m] - y[m]) ** 2))
    r["segments"][name] = {
        "n": n, "row_frac": n / len(m),
        "p6_fastball_max_abs": float(np.abs(ship.p6_fastball.values[m]
                                            - ours.p6_fastball.values[m]).max()),
        "idea_matchup_exact_frac": float((ship.idea_matchup_pitch.values[m]
                                          == ours.idea_matchup_pitch.values[m]).mean()),
        "logit_mean_abs": float(np.abs(L["S"][m] - L["O"][m]).mean()),
        "logit_max_abs": float(np.abs(L["S"][m] - L["O"][m]).max()),
        "brier_S": bS, "brier_O": bO, "brier_diff_S_minus_O": bS - bO,
        "contrib_score_全体기준": -(bS - bO) * (n / len(m)) * SC}
json.dump(r, open(f"{B}/out/diag_segments.json", "w"), indent=1)
print(json.dumps(r, indent=1, ensure_ascii=False))

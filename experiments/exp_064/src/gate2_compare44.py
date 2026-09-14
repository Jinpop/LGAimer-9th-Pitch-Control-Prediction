"""exp_064 stage 2-a — exp_062 '비교 B' 재실행.

exp_062 비교 B = 우리 ≤2024 재구축 표로 만든 44열  vs  출하 ≤2024 표로 만든 44열.
그때 14열이 어긋났다. 이번엔 그 표에서 **p6 3종만 배포본으로 갈아끼워** 다시 잰다.

사전 등록 통과조건 (S8 — 결과를 보고 바꾸지 않는다):
  · 14열 max|diff| <= 1e-6   (기대: 정확히 0.0)
비교는 exp_062 compare_44.py 의 cmp() 를 그대로 승계한다(NaN==NaN 을 일치로 본다).
"""
import json
import os

import numpy as np
import pandas as pd

H = os.path.expanduser("~")
B = f"{H}/SungjinPark/exp064"
E62 = f"{H}/SungjinPark/exp062_gate"
EXP60 = f"{H}/SungjinPark/exp060_nn"
TOL = 1e-6
COLS14 = ["p6_curve", "p6_cutter", "p6_fastball", "p6_offspeed", "p6_sinker",
          "p6_slider", "i3_anom_curve", "i3_anom_cutter", "i3_anom_fastball",
          "i3_anom_offspeed", "i3_anom_sinker", "i3_anom_slider", "i3_anom_l1",
          "idea_matchup_pitch"]

ship = pd.read_parquet(f"{E62}/work/f44_ship.parquet")
mine = pd.read_parquet(f"{B}/work/f44_shipp6_2024.parquet")
c14 = pd.read_parquet(f"{B}/work/p6_ship_2024.parquet")
COLS = [c for c in ship.columns if c != "row_id"]
assert len(COLS) == 44
assert np.array_equal(ship.row_id.values, mine.row_id.values)
assert np.array_equal(ship.row_id.values, c14.row_id.values)
assert len(ship) == 253507


def cmp(a, b, cols, cast32=False):
    """exp_062 compare_44.cmp 승계. NaN==NaN 을 일치로 본다."""
    res = {}
    for c in cols:
        x = a[c].values.astype("float32" if cast32 else "float64")
        y = b[c].values.astype("float32" if cast32 else "float64")
        nx, ny = np.isnan(x), np.isnan(y)
        both = ~nx & ~ny
        d = np.zeros(len(x), dtype="float64")
        d[both] = np.abs(x[both].astype("float64") - y[both].astype("float64"))
        res[c] = {"max_abs": float(d.max()), "mean_abs": float(d.mean()),
                  "exact_frac": float(((d == 0.0) & both).sum() + (nx & ny).sum())
                  / len(x),
                  "nan_only_one_side": int((nx ^ ny).sum()),
                  "n_nan_a": int(nx.sum()), "n_nan_b": int(ny.sum())}
    return res


out = {"n_rows": int(len(ship)), "tol": TOL,
       "B_rerun_ours_up2024_with_ship_p6_vs_ship": cmp(mine, ship, COLS),
       "cols14_file_vs_ship": cmp(c14, ship, COLS14)}

# 프레임에 실제로 들어간 float32 값도 대조한다
f99s = pd.read_parquet(f"{B}/cache/frame99_ship.parquet",
                       columns=COLS14 + ["season", "row_id"])
f99s = f99s[f99s.season == 2024]
assert np.array_equal(f99s.row_id.values, ship.row_id.values)
out["frame99_ship_float32_vs_ship_float32"] = cmp(f99s, ship, COLS14, cast32=True)

for k in ("B_rerun_ours_up2024_with_ship_p6_vs_ship", "cols14_file_vs_ship",
          "frame99_ship_float32_vs_ship_float32"):
    r = out[k]
    cols = list(r)
    out[k + "__summary"] = {
        "n_cols": len(cols),
        "max_over_cols": max(r[c]["max_abs"] for c in cols),
        "n_cols_ge_tol": sum(r[c]["max_abs"] >= TOL for c in cols),
        "n_cols_exactly_zero": sum(r[c]["max_abs"] == 0.0 for c in cols),
        "n_nan_mismatch_total": sum(r[c]["nan_only_one_side"] for c in cols),
        "cols_ge_tol": [c for c in cols if r[c]["max_abs"] >= TOL]}

m14 = out["cols14_file_vs_ship__summary"]
m44 = out["B_rerun_ours_up2024_with_ship_p6_vs_ship__summary"]
out["prereg_check"] = {
    "cond_14col_max_le_1e-6": bool(m14["max_over_cols"] <= TOL),
    "cols14_max_abs": m14["max_over_cols"],
    "exactly_zero_14": bool(m14["max_over_cols"] == 0.0),
    "all_44_max_abs": m44["max_over_cols"],
    "exactly_zero_44": bool(m44["max_over_cols"] == 0.0),
    "nan_mismatch_44": m44["n_nan_mismatch_total"]}
json.dump(out, open(f"{B}/out/gate2_compare44.json", "w"), indent=1,
          ensure_ascii=False)
print(json.dumps({k: v for k, v in out.items()
                  if k.endswith("__summary") or k == "prereg_check"},
                 indent=1, ensure_ascii=False))
if not out["prereg_check"]["cond_14col_max_le_1e-6"]:
    raise SystemExit("🔴 STAGE2-a FAIL: 14열 max|diff| > 1e-6")
print("GATE2A_OK", flush=True)

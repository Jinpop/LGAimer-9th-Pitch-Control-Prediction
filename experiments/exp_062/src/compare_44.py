"""exp_062 step3 — 44열 정의 일치 대조.

세 입력 (전부 train season==2024 253,507행, 같은 행·같은 순서):
  ours2023  : 우리 ≤2023 표 (exp_060 work/tables_2024) — **학습 시 실제로 쓴 as-of**
  ours2024  : 우리 ≤2024 표 (exp_062 work/tables_up2024)
  ship      : 출하 zip mate/model/tables (≤2024)

보고 셋:
  (0) 재현  ours2023  vs frame99 저장본        -> 0 이어야 한다
  (A) 주경로 ship      vs ours2024  (as-of 일치) -> 정의 일치 게이트
  (R) 참고  ship      vs ours2023  (as-of 불일치, 해석 금지)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp062_gate")
B60 = os.path.expanduser("~/SungjinPark/exp060_nn")

ours2023 = pd.read_parquet(f"{B}/work/f44_ours2023.parquet")
ours2024 = pd.read_parquet(f"{B}/work/f44_ours2024.parquet")
ship = pd.read_parquet(f"{B}/work/f44_ship.parquet")
COLS = [c for c in ours2023.columns if c != "row_id"]
assert len(COLS) == 44

f99 = pd.read_parquet(f"{B60}/cache/frame99.parquet")
f99 = f99[f99.season == 2024]
assert len(f99) == len(ours2023) == 253507
assert np.array_equal(f99.row_id.values, ours2023.row_id.values)
assert np.array_equal(ship.row_id.values, ours2023.row_id.values)
assert np.array_equal(ours2024.row_id.values, ours2023.row_id.values)


def cmp(a, b, cast32):
    """a, b: DataFrame. NaN==NaN 을 일치로 본다."""
    res = {}
    for c in COLS:
        x = a[c].values.astype("float32" if cast32 else "float64")
        y = b[c].values.astype("float32" if cast32 else "float64")
        nx, ny = np.isnan(x), np.isnan(y)
        both = ~nx & ~ny
        d = np.zeros(len(x), dtype="float64")
        d[both] = np.abs(x[both].astype("float64") - y[both].astype("float64"))
        nan_mismatch = int((nx ^ ny).sum())
        exact = float(((d == 0.0) & both).sum() + (nx & ny).sum()) / len(x)
        res[c] = {
            "max_abs": float(d.max()) if len(d) else 0.0,
            "mean_abs": float(d.mean()) if len(d) else 0.0,
            "exact_frac": exact,
            "nan_only_one_side": nan_mismatch,
            "n_nan_a": int(nx.sum()), "n_nan_b": int(ny.sum()),
            "std_a": float(np.nanstd(x)), "rel_max": (float(d.max() / np.nanstd(x))
                                                      if np.nanstd(x) > 0 else float("nan")),
        }
    return res


out = {
    "n_rows": int(len(ours2023)),
    "cols": COLS,
    "repro_ours2023_vs_frame99_float32": cmp(ours2023, f99, True),
    "A_ship_vs_ours2024_asof_matched": cmp(ship, ours2024, False),
    "R_ship_vs_ours2023_asof_MISMATCHED": cmp(ship, ours2023, False),
}
json.dump(out, open(f"{B}/out/compare44.json", "w"), indent=1)

for k in ("repro_ours2023_vs_frame99_float32", "A_ship_vs_ours2024_asof_matched",
          "R_ship_vs_ours2023_asof_MISMATCHED"):
    r = out[k]
    print(f"\n===== {k} =====")
    print(f"{'col':24s} {'max|diff|':>12s} {'mean|diff|':>12s} {'exact%':>8s} {'nan1side':>9s}")
    for c in COLS:
        v = r[c]
        print(f"{c:24s} {v['max_abs']:12.6g} {v['mean_abs']:12.6g} "
              f"{100*v['exact_frac']:8.4f} {v['nan_only_one_side']:9d}")
    bad = [c for c in COLS if r[c]["max_abs"] >= 1e-6 or r[c]["nan_only_one_side"]]
    print(f"-> cols with max|diff| >= 1e-6 (or NaN 불일치): {len(bad)}/44 {bad}")

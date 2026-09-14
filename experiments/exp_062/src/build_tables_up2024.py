"""exp_062 step1 — 우리 코드·우리 trackman 매핑으로 **≤2024 as-of 표**를 만든다.

출하 `mate/model/tables/` 도 ≤2024 as-of 다. 따라서 이 표와 출하 표의 차이는
**as-of 가 아니라 '누가 만들었는가'** 만 남는다 (exp_060 러너가 신고한 위험 ①②).

코드는 출하본 pipeline.py(sha a3181a10…, 출하 mate/pipeline.py 와 바이트 동일)를 그대로 import.
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import numpy as np
import pandas as pd

B60 = os.path.expanduser("~/SungjinPark/exp060_nn")
B = os.path.expanduser("~/SungjinPark/exp062_gate")
sys.path.insert(0, B60)          # 출하본 pipeline.py 사본 (읽기 전용)
import pipeline as P  # noqa: E402

TDIR = sys.argv[1] if len(sys.argv) > 1 else f"{B}/work/tables_up2024"
t0 = time.time()

tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig")
hist = tr                      # ≤2024 전체 (train 은 2019~2024)
assert int(hist.season.max()) == 2024
print(f"hist rows {len(hist)}  seasons {sorted(hist.season.unique())}", flush=True)

P.build_tables(hist, TDIR)
print(f"[{time.time()-t0:.1f}s] build_tables done", flush=True)

tm = pd.read_parquet(f"{B60}/work/tm_mapped.parquet")   # 읽기 전용
tm_hist = tm[tm.season < 2025]
assert len(tm_hist) == len(tm)
t1 = time.time()
m, lv = P.build_pitch_model(tm_hist, TDIR)
print(f"[{time.time()-t0:.1f}s] pitch model fit {(time.time()-t1)/60:.2f}min "
      f"on {len(tm_hist)} tm rows (seasons {sorted(tm.season.unique())})", flush=True)

Xh = hist[P.PM_FEATS].copy()
for c in P.PM_CATS:
    Xh[c] = pd.Categorical(Xh[c], categories=lv[c])
Ph = m.predict(Xh)
hp = hist[["balls_before", "strikes_before"]].copy()
for i, t in enumerate(P.TYPES):
    hp[f"p6_{t}"] = Ph[:, i]
P.build_anom_table(hp, TDIR)
print(f"[{time.time()-t0:.1f}s] anom table done -> {TDIR}", flush=True)

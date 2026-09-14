"""exp_060 stage1 — ins_* 16열 + sp_* 6열, sp_def README §2 연도별 루프 그대로.

hist = train[season < Y]. deploy_splits.npz / anchor.npz(배포용)는 일절 쓰지 않는다.
2019: ins는 더미 앵커(README 코드 그대로), sp는 NaN(→ 학습 스크립트에서 0.0 채움).
"""
import os
import sys
import time

import numpy as np
import pandas as pd

B = os.path.expanduser("~/SungjinPark/exp060_nn")
SPD = os.path.expanduser("~/SungjinPark/sp_def_recv")
sys.path.insert(0, SPD)
import inseason  # noqa: E402
import splits  # noqa: E402

t0 = time.time()
tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig")

# ---- ins_* 16열 (README §2 코드 그대로) ----
parts = []
for Y in sorted(tr.season.unique()):
    cur, hist = tr[tr.season == Y], tr[tr.season < Y]
    if len(hist) == 0:                      # 2019: 앞선 시즌 없음 -> 더미 앵커
        d = tr.iloc[:1].copy()
        d["pitcher_id"] = -1
        d["batter_id"] = -1
        A = inseason.build_anchor(d)
    else:
        A = inseason.build_anchor(hist)
    I = inseason.apply(cur, A)
    I.index = cur.index
    parts.append(I)
    print(f"ins Y={Y} rows {len(cur)}", flush=True)
INS = pd.concat(parts).reindex(tr.index)
assert len(INS) == len(tr) and not INS.isna().any().any(), "ins_* 결측 없어야 함"
INS.astype("float32").to_parquet(f"{B}/work/ins16.parquet")

# ---- sp_* 6열 (README §2 코드 그대로 — 연도별 season < Y 표) ----
SP6 = [f"sp_{a[0]}" for a in splits.AXES] + ["sp_pit_bhand_all", "sp_bat_phand_all"]
parts = []
for Y in sorted(tr.season.unique()):
    cur, hist = tr[tr.season == Y], tr[tr.season < Y]
    if len(hist) == 0:
        parts.append(pd.DataFrame(np.nan, index=cur.index, columns=SP6))
        print(f"sp Y={Y} -> NaN (no hist)", flush=True)
        continue
    o = splits.apply(cur, splits.build(hist))
    o.index = cur.index
    parts.append(o[SP6])
    print(f"sp Y={Y} rows {len(cur)}", flush=True)
SP = pd.concat(parts).reindex(tr.index)
assert len(SP) == len(tr)
n2019 = int(SP[SP6[0]].isna().sum())
print(f"sp NaN rows (2019 expected 237413): {n2019}")
SP.astype("float32").to_parquet(f"{B}/work/sp6.parquet")
print(f"inssp done elapsed {(time.time()-t0)/60:.2f}min", flush=True)

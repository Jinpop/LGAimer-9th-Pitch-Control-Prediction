"""exp_064 stage 1 — 한 시즌의 **구종모델 파생 14열**을 배포 p6 3종으로 다시 계산.

사용법: python p6_recompute.py <YEAR>

계산 경로는 출하 `mate/pipeline.py` 의 `apply_features` 그대로다(새 코드 없음).
그 함수가 내놓는 22열 중 **14열만** 꺼내 쓴다 — 나머지 8열(idea_cnt_te·i3_te48·
sit_*)은 타깃 인코딩이라 as-of 규율이 걸려 있고, 이 실험은 그것을 건드리지 않는다.
🔴 그래서 14열 외에는 **디스크에 남기지 않는다**(실수로 프레임에 섞이는 길을 없앤다).

예외: YEAR==2024 는 stage 2 게이트용으로 44열 전체를 별도 파일에 따로 저장한다.
그 파일은 게이트 대조에만 쓰이고 학습 프레임에 들어가지 않는다.

배포 p6 는 빈티지가 없다 — 전 연도가 같은 한 벌을 쓴다(팀원 rebuild_tables.py
build_vintage 규약). 그래서 시즌별로 표를 갈아끼우지 않는다.
"""
import json
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import numpy as np
import pandas as pd

H = os.path.expanduser("~")
B = f"{H}/SungjinPark/exp064"
SHIP_MATE = f"{H}/SungjinPark/exp062_gate/ship/mate"
sys.path.insert(0, SHIP_MATE)          # 출하 mate/pipeline.py 를 그대로 import
import pipeline as P  # noqa: E402

TDIR = f"{B}/work/tables_shipp6"
Y = int(sys.argv[1])
t0 = time.time()

COLS14 = (P.PC + [f"i3_anom_{t}" for t in P.TYPES]
          + ["i3_anom_l1", "idea_matchup_pitch"])
assert len(COLS14) == 14

tr = pd.read_csv(f"{H}/SungjinPark/data/train.csv", encoding="utf-8-sig")
cur = tr[tr.season == Y]
print(f"Y={Y} rows {len(cur)}", flush=True)

feats = P.apply_features(cur, TDIR)
assert feats.index.equals(cur.index)
assert feats.shape[1] == 22, feats.shape

out = feats[COLS14].copy()
out["row_id"] = cur.row_id.values
out.to_parquet(f"{B}/work/p6_ship_{Y}.parquet")     # float64 보존

# ── 범주 미등재 계측 (출하 pmcat6 는 투수 571명뿐) ───────────────────────────
lv = json.load(open(f"{TDIR}/pmcat6.json"))
cen = {"season": Y, "n_rows": int(len(cur))}
for c in P.PM_CATS:
    known = set(lv[c])
    miss = ~cur[c].isin(known)
    cen[c] = {"n_levels_in_table": len(known),
              "n_rows_unregistered": int(miss.sum()),
              "frac_rows_unregistered": float(miss.mean()),
              "n_distinct_unregistered": int(cur.loc[miss, c].nunique())}
cen["nan_by_col"] = {c: int(out[c].isna().sum()) for c in COLS14}
cen["desc_by_col"] = {c: {"mean": float(np.nanmean(out[c].values)),
                          "std": float(np.nanstd(out[c].values))}
                      for c in COLS14}
json.dump(cen, open(f"{B}/out/census_{Y}.json", "w"), indent=1, ensure_ascii=False)

if Y == 2024:
    ratios = P.apply_ratio_features(cur, feats)
    f44 = pd.concat([feats, ratios], axis=1)
    assert f44.shape[1] == 44, f44.shape
    f44 = f44.copy()
    f44["row_id"] = cur.row_id.values
    f44.to_parquet(f"{B}/work/f44_shipp6_2024.parquet")
    print(f"Y=2024 gate 44col saved {f44.shape}", flush=True)

print(f"Y={Y} done 14col {out.shape} "
      f"unreg_pitcher {cen['pitcher_id']['frac_rows_unregistered']*100:.4f}% "
      f"elapsed {(time.time()-t0)/60:.2f}min", flush=True)

"""exp_062 step2 — train season==2024 행 253,507개에 대해 **v6 파생 44열**을 만든다.

사용법: python make44.py <table_dir> <out_parquet>
계산 경로는 출하 `mate/v6_infer.py` 의 두 줄과 동일하다:
    feats  = P.apply_features(df, TBL_DIR)
    ratios = P.apply_ratio_features(df, feats)
표만 갈아끼운다. float64 로 저장(비교 정밀도 보존).
"""
import os
import sys
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, "6")

import pandas as pd

B60 = os.path.expanduser("~/SungjinPark/exp060_nn")
sys.path.insert(0, B60)
import pipeline as P  # noqa: E402

TDIR, OUT = sys.argv[1], sys.argv[2]
t0 = time.time()

tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig")
cur = tr[tr.season == 2024]
assert len(cur) == 253507, len(cur)

feats = P.apply_features(cur, TDIR)
ratios = P.apply_ratio_features(cur, feats)
out = pd.concat([feats, ratios], axis=1)
assert out.index.equals(cur.index)
assert out.shape[1] == 44, out.shape
out["row_id"] = cur.row_id.values
out.to_parquet(OUT)
print(f"{TDIR} -> {OUT}  shape {out.shape}  {(time.time()-t0)/60:.2f}min", flush=True)

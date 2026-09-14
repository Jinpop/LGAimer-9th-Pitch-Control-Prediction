"""exp_071 stage 1 사전점검 — 프레임 열 대조 + 학습행 수 확인."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"
import json, hashlib
import numpy as np, pandas as pd

H = "/home/<user>/SungjinPark"
FR = f"{H}/exp064/cache/frame99_ship.parquet"
META = json.load(open(f"{H}/exp071/spec/nn_a42_meta.json"))
CATS, DENSE = META["CATS"], META["DENSE"]

import pyarrow.parquet as pq
sch = pq.read_schema(FR)
cols = list(sch.names)
print("프레임 열수", len(cols))
need = set(CATS + DENSE) - {"cnt_state"}
missing = sorted(need - set(cols))
print("결측 열", missing)
print("보유", len(need) - len(missing), "/", len(need))
# row_id 존재?
for k in ("row_id","season","game_type","control_success"):
    print(f"  {k} in frame:", k in cols)

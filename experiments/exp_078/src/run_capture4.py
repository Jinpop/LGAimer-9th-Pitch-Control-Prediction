"""exp_078 stage1 보조 — 파트너 v6 내부 4팔(pl/pc/pn/pf) 로짓을 tr2024 프레임에서 포획.
학습 0건.  추론 1회 (v6 경로만).  ~/SungjinPark/exp078/work 에만 쓴다."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import sys, time
CAP = "/home/<user>/SungjinPark/arms/mate_cap"
os.chdir(CAP)
sys.path.insert(0, CAP)
sys.path.insert(0, "/home/<user>/SungjinPark/exp078/src")
t0 = time.time()
import v6_capture_infer as V
ids, preds = V.run(model_dir="./model", test_dir="./data")
import numpy as np
np.save("/home/<user>/SungjinPark/exp078/work/tr2024_v6_pv.npy",
        np.asarray(preds, np.float64))
print(f"[DONE] rows={len(preds)}  wall={(time.time()-t0)/60:.2f}min", flush=True)

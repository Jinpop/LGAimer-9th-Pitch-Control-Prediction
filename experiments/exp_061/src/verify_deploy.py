"""exp_061 stage4 검증 — 배포 npz 4벌에 대해 출하 predict_nn(numpy·CPU) vs torch 예측.

파일럿(exp_060) 실측 max_abs_diff 1.19e-07 자릿수를 유지하는지 본다.
추론 시간은 경합 없는 단독 실행에서 재는 것이 규약(§규약7 G6) — 여기 값은 참고치.
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

if os.environ.get("EXP061_SMOKE") == "1":
    L.OUT = f"{L.B}/out_smoke"
if os.environ.get("EXP061_OUT"):
    L.OUT = f"{L.B}/" + os.environ["EXP061_OUT"]
OUT = L.OUT
df, DENSE, _ = L.load_frame()
dv = df[df.season == 2024]
assert len(dv) == 253507

res = {}
for npz in sorted(glob.glob(f"{OUT}/nn99_d*.npz")):
    base = npz[:-4]
    sd = os.path.basename(base).replace("nn99_d", "")
    t0 = time.time()
    p_np = L.predict_nn(dv, npz, f"{base}_meta.json")
    t = time.time() - t0
    p_t = np.load(f"{OUT}/preds/D_s{sd}_pred2024_insample.npy")
    d = np.abs(p_np - p_t)
    res[f"seed_{sd}"] = {
        "max_abs_diff": float(d.max()), "mean_abs_diff": float(d.mean()),
        "numpy_inference_sec_253507rows": round(t, 2),
        "note": "2024행은 배포 모델의 학습 표본 내 — 순전파 동일성 확인용이지 점수가 아니다"}
    print(f"{os.path.basename(npz)} max_abs_diff {d.max():.3e} ({t:.2f}s)",
          flush=True)
json.dump(res, open(f"{OUT}/verify_deploy.json", "w"), indent=1,
          ensure_ascii=False)
print(json.dumps(res, indent=1, ensure_ascii=False))

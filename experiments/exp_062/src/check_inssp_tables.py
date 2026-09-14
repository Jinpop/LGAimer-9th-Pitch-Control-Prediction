"""exp_062 부록 — 44열 밖 22열(ins_* 16 · sp_* 6)의 동결표도 같은 위험을 지는가.

배포 추론은 `mate/model/anchor.npz`·`deploy_splits.npz`(팀원 동결, ≤2024)를 쓴다.
우리 학습 프레임은 sha 동일한 inseason.py/splits.py 로 연도 루프(hist<Y)를 돌려 만들었다.
→ ≤2024 as-of 를 맞춰 우리가 같은 표를 만들어 보고 팀원 동결표와 대조한다.
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
sys.path.insert(0, f"{B}/ship/mate")
import inseason  # noqa: E402
import splits  # noqa: E402

tr = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"), encoding="utf-8-sig")
A_ours = inseason.build_anchor(tr)
T_ours = splits.build(tr)
A_ship = inseason.load(f"{B}/ship/mate/model/anchor.npz")
T_ship = splits.load(f"{B}/ship/mate/model/deploy_splits.npz")


def summ_flat(a, b, tag):
    """평평한 dict[str, ndarray] 대조 (inseason anchor)"""
    ka, kb = set(a.keys()), set(b.keys())
    r = {"keys_only_ours": sorted(ka - kb), "keys_only_ship": sorted(kb - ka), "cols": {}}
    for k in sorted(ka & kb):
        x, y = np.asarray(a[k]), np.asarray(b[k])
        if x.shape != y.shape:
            r["cols"][k] = {"shape_ours": list(x.shape), "shape_ship": list(y.shape),
                            "equal": False}
            continue
        r["cols"][k] = {"shape": list(x.shape),
                        "max_abs": float(np.abs(x.astype("float64") - y.astype("float64")).max()),
                        "exact_frac": float(np.mean(x == y))}
    print(f"--- {tag} ---")
    print(json.dumps(r, indent=1))
    return r


def summ_splits(To, Ts, tag):
    """splits.load 구조: {'league': float, 'axes': {name: {ids,D,n,cn,p_all,nc,...}}}"""
    r = {"league": {"ours": To["league"], "ship": Ts["league"],
                    "equal": To["league"] == Ts["league"]}, "axes": {}}
    for nm in To["axes"]:
        a, b = To["axes"][nm], Ts["axes"][nm]
        d = {}
        for k in ("ids", "D", "n", "cn", "p_all"):
            x, y = np.asarray(a[k]), np.asarray(b[k])
            if x.shape != y.shape:
                d[k] = {"shape_ours": list(x.shape), "shape_ship": list(y.shape),
                        "equal": False}
                continue
            d[k] = {"shape": list(x.shape),
                    "max_abs": float(np.abs(x.astype("float64") - y.astype("float64")).max()),
                    "exact_frac": float(np.mean(x == y))}
        d["nc"] = [int(a["nc"]), int(b["nc"])]
        r["axes"][nm] = d
    print(f"--- {tag} ---")
    print(json.dumps(r, indent=1))
    return r


out = {"anchor": summ_flat(A_ours, A_ship, "inseason anchor (ours <=2024 vs ship frozen)"),
       "splits": summ_splits(T_ours, T_ship, "splits (ours <=2024 vs ship frozen)")}
json.dump(out, open(f"{B}/out/check_inssp.json", "w"), indent=1)

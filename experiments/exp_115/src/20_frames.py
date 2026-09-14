#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_115 — 프레임별 미지 주체율.  네 셀 전부 한 표에 놓는다."""
import json, sys
import numpy as np, pandas as pd
H = "/home/<user>/SungjinPark"; B = f"{H}/build_pb3_probe"
sys.path.insert(0, f"{B}/pkg/PAR")
import splits as SPL
T = SPL.load(f"{B}/pkg/PAR/model/deploy_splits.npz")
FR = f"{H}/build_CA/frames"
CALF = f"{H}/build_AC/work/tr2024/test.csv"
COLS = ["sp_pit_bhand", "sp_bat_phand", "sp_pit_strk2", "sp_pit_ball3"]
o = {}
for name, path in [("cal_train2024", CALF)] + \
        [(f, f"{FR}/{f}/test.csv") for f in ("cov", "band", "real5", "keep110")]:
    d = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    S = SPL.apply(d, T)
    e = {"n": int(len(d))}
    for c in COLS:
        v = S[c].to_numpy(np.float64)
        e[c] = {"table_miss": int((v == 0).sum()),
                "miss_pct": float(100 * (v == 0).mean()),
                "sd": float(v.std())}
    o[name] = e
o["note"] = ("주체가 투수인 셀(pit_*)은 미지율이 같고, 타자 셀(bat_phand)만 다르다. "
             "real5 는 실제 대회 표본 5행이라 배포 미지율의 유일한 직접 관측이다(S38·S51). "
             "미지 행도 표값 0.0 이 유한이라 delta=(0-center)/scale 로 균일하게 움직인다.")
json.dump(o, open(f"{B}/work/frames20.json", "w"), ensure_ascii=False, indent=1)
print(f"{'frame':16s} {'n':>7s}  " + "  ".join(f"{c[3:]:>12s}" for c in COLS))
for k, v in o.items():
    if k == "note":
        continue
    print(f"{k:16s} {v['n']:7d}  " +
          "  ".join(f"{v[c]['table_miss']:5d}({v[c]['miss_pct']:5.1f}%)" for c in COLS))
print("FRAMES_OK")

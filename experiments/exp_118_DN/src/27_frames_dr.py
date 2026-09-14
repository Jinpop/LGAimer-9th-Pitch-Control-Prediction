#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""프레임별 새 축 원시 열 커버리지 실측 — 게이트 해석용 (S38·S61·S131)."""
import json

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
FDIR = f"{H}/build_CA/frames"
COL = "asof_pitcher_strike_rate"

D10 = json.load(open(f"{B}/work/derive12.json", encoding="utf-8"))
center = float(D10["cal_frame"]["center_nanmean"])
scale = float(D10["cal_frame"]["scale_nanstd_ddof0"])

o = {}
for fr in ("cov", "band", "real5", "keep110", "caD", "caE", "mixB", "mixC", "degen",
           "cov_shuf_al"):
    d = pd.read_csv(f"{FDIR}/{fr}/test.csv", encoding="utf-8-sig", low_memory=False)
    if COL not in d.columns:
        o[fr] = {"n": int(len(d)), "🔴": f"{COL} 열이 이 프레임에 없다"}
        continue
    v = pd.to_numeric(d[COL], errors="coerce").to_numpy(np.float64)
    fin = np.isfinite(v)
    dl = np.where(fin, (v - center) / scale, 0.0)
    o[fr] = {"n": int(len(d)),
             "n_nonfinite_no_shift": int((~fin).sum()),
             "nonfinite_pct": float(100.0 * (~fin).mean()),
             "mean_delta": float(dl.mean()),
             "mean_shift_at_u_minus_0008": float(-0.008 * dl.mean()),
             "max_abs_delta": float(np.abs(dl).max()),
             "max_abs_shift_at_0008": float(0.008 * np.abs(dl).max())}
o["🔴"] = (
 "⚠️ 대역 프레임(cov·band·caD·caE·mixB·mixC·degen)은 train-2024 행을 2025 로 덮은 "
 "것이라 이 열의 결측률이 배포 결측률을 대표하지 않는다(S32).  🔴 real5(실제 대회 표본 "
 "5행)와 keep110 의 결측 수를 함께 본다 — S61 이 요구하는 'KEEP 행이 바꾼 상수를 실제로 "
 "타는가' 는 keep110 의 유한 행 수로 판정한다.")
p = f"{B}/work/derive12.json"
D10["frames_probe"] = o
json.dump(D10, open(p, "w"), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))

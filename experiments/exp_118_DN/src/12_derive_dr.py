#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""후보 DN 1단 — 새 축 `ctrl_asof_bmid` 의 center/scale 을 train 에서 동결한다.

원시 열 = test.csv / train.csv 의 `asof_batter_middle_rate` (변환 없음).
center = train season==2024 253,507행에서 그 열의 nanmean
scale  = 같은 행에서 nanstd(ddof=0)
축 2 `ctrl_asof_psr` · 축 5 `ctrl_asof_mid` 와 완전히 같은 규약이다.  test 통계 0.
"""
import hashlib
import json

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
TR = f"{H}/data/train.csv"
COL = "asof_pitcher_strike_rate"
SIB5 = "asof_pitcher_middle_rate"     # 축 5 (부모 DM 이 실은 것)
SIBB = "asof_batter_middle_rate"      # 후보 DN 의 축
MIRROR = "asof_pitcher_ball_rate"     # 거울 열 (이 열과 r = -0.825)
SIB2 = "asof_pitcher_success_rate"    # 축 2


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


cols = ["season", COL, SIB5, SIB2, SIBB, MIRROR, "asof_batter_n",
        "asof_pitcher_n", "control_success"]
df = pd.read_csv(TR, usecols=cols, encoding="utf-8-sig")
d24 = df[df["season"] == 2024]
c = d24[COL].to_numpy(np.float64)

center = float(np.nanmean(c))
scale = float(np.nanstd(c))          # ddof=0
fin = np.isfinite(c)
delta_all = np.nan_to_num((c - center) / scale, nan=0.0)   # 비유한 -> 이동 0.0
ad = np.abs((c - center) / scale)[fin]

corr = {}
for other in (SIB2, SIB5, SIBB, MIRROR):
    o = d24[other].to_numpy(np.float64)
    m = np.isfinite(c) & np.isfinite(o)
    corr[other] = {"r": float(np.corrcoef(c[m], o[m])[0, 1]), "n": int(m.sum())}

# 형제 두 축의 같은 눈금 꼬리 (비교용)
sib = {}
for nm, k in ((SIB5, "axis5"), (SIB2, "axis2")):
    v = d24[nm].to_numpy(np.float64)
    cc, ss = float(np.nanmean(v)), float(np.nanstd(v))
    q = np.abs((v - cc) / ss)
    q = q[np.isfinite(q)]
    sib[k] = {"col": nm, "center": cc, "scale": ss,
              "abs_delta_p50": float(np.percentile(q, 50)),
              "abs_delta_p99": float(np.percentile(q, 99)),
              "abs_delta_p999": float(np.percentile(q, 99.9)),
              "abs_delta_max": float(q.max())}

season_level = {}
for s in sorted(int(x) for x in df["season"].dropna().unique()):
    v = df.loc[df["season"] == s, COL].to_numpy(np.float64)
    season_level[str(s)] = {"n": int(len(v)),
                            "mean_in_2024_sd_units": float((np.nanmean(v) - center) / scale),
                            "n_nonfinite": int(np.isnan(v).sum())}

lab = df["control_success"].to_numpy(np.float64)
label_corr = {}
for s_ in range(2019, 2025):
    m_ = (df["season"].to_numpy() == s_) & np.isfinite(df[COL].to_numpy(np.float64)) \
        & np.isfinite(lab)
    label_corr[str(s_)] = float(np.corrcoef(df[COL].to_numpy(np.float64)[m_], lab[m_])[0, 1])
m_ = np.isfinite(df[COL].to_numpy(np.float64)) & np.isfinite(lab)
label_corr["all"] = float(np.corrcoef(df[COL].to_numpy(np.float64)[m_], lab[m_])[0, 1])

n = d24["asof_pitcher_n"].to_numpy(np.float64)
buckets = {}
for lo, hi, lab in ((0, 10, "n<10"), (10, 50, "10<=n<50"), (50, 200, "50<=n<200"),
                    (200, np.inf, "n>=200")):
    m = fin & (n >= lo) & (n < hi)
    buckets[lab] = {"rows": int(m.sum()),
                    "mean_abs_delta": float(np.abs((c - center) / scale)[m].mean()),
                    "max_abs_delta": float(np.abs((c - center) / scale)[m].max())}

o = {
 "column": COL,
 "train_csv_sha256": sh(TR),
 "n_rows_train": int(len(df)),
 "cal_frame": {
   "definition": "train.csv season == 2024",
   "n_rows": int(len(d24)),
   "n_finite": int(fin.sum()),
   "n_nonfinite": int((~fin).sum()),
   "nonfinite_pct": float(100.0 * (~fin).mean()),
   "center_nanmean": center,
   "center_repr": repr(center),
   "scale_nanstd_ddof0": scale,
   "scale_repr": repr(scale),
   "mean_delta_all_rows_nonfinite_as_zero": float(delta_all.mean()),
   "mean_shift_at_u_minus_0008": float(-0.008 * delta_all.mean()),
   "abs_delta_p50": float(np.percentile(ad, 50)),
   "abs_delta_p99": float(np.percentile(ad, 99)),
   "abs_delta_p999": float(np.percentile(ad, 99.9)),
   "abs_delta_max": float(ad.max()),
   "max_abs_u_times_delta_at_0008": float(0.008 * ad.max()),
 },
 "orthogonality_to_shipped_raw_column_axes": corr,
 "siblings_same_scale": sib,
 "small_sample_buckets_by_asof_pitcher_n": buckets,
 "season_level_drift": season_level,
 "label_correlation_by_season": label_corr,
 "S277_is_this_quantity_already_re_centred_by_shipped_code": {
   "hits": ["aa_infer.py:526 INS_KEY_COLS (원시 열, 재중심 없음)",
            "model_v6/features.txt #21 · mate/model/features.txt #21 · "
            "mate/model/features_sp.txt #21 (트리 팔 피처, 재중심 없음)",
            "mate/model/deep/prep.json NUM[16] mu=0.4418756663799286 "
            "sd=0.03961697965860367 (딥 팔이 train<=2024 전량 눈금으로 표준화한다)"],
   "reading": ("이 열은 이미 세 팔 전부의 입력이다.  형제 축 5 `ctrl_asof_mid` · 형제 후보 DN 의 축도 "
               "정확히 같은 처지였다(부모 DM 이 자기 S73 스윕에 적었다).  따라서 이 "
               "사후 축의 값은 새 정보를 더하는 데 있지 않고, 팔들이 이 방향에 "
               "얼마나 실었는지를 전역 스칼라 하나로 깎거나 더하는 데 있다.  "
               "🔴 딥 팔의 prep.json mu/sd 는 train<=2024 전량(1,475,092행) 눈금이라 "
               "우리 center/scale(2024만)과 다른 값이다 — 두 재중심은 같은 양이 아니다."),
 },
}
json.dump(o, open(f"{B}/work/derive12.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in o.items() if k != "season_level_drift"},
                 ensure_ascii=False, indent=1))
print("DERIVE10_OK center=", repr(center), " scale=", repr(scale))

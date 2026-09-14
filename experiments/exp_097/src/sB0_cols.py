"""exp_097b stageB0 — 결함이 몇 열인지 표준화 *출력* 에서 직접 읽는다.

표준화가 옳다면 학습행에서 각 열의 sd 는 1 이어야 한다(nan->0 대체분만큼만 내려간다).
결함면(X_oot23)과 수리면(X_f64o)의 열별 mean/sd 를 학습 마스크 위에서 재서
어느 열이 죽었는지 프레임 없이 판정한다.
"""
import json
import sys

import numpy as np

A = np.load(sys.argv[1], mmap_mode="r")     # 결함면
B = np.load(sys.argv[2], mmap_mode="r")     # 수리면
trm = np.load(sys.argv[3])
names = json.load(open(sys.argv[4]))
NUM = names["NUM"] if isinstance(names, dict) and "NUM" in names else names
idx = np.flatnonzero(trm.astype(bool))
print("train rows", len(idx), "cols", A.shape[1], "names", len(NUM), flush=True)
rows = []
CH = 200000
for j in range(A.shape[1]):
    sa = na = qa = 0.0
    sb = qb = 0.0
    for s in range(0, len(idx), CH):
        ii = idx[s:s + CH]
        a = np.asarray(A[ii, j], dtype=np.float64)
        b = np.asarray(B[ii, j], dtype=np.float64)
        sa += a.sum(); qa += (a * a).sum()
        sb += b.sum(); qb += (b * b).sum()
        na += len(a)
    ma, mb = sa / na, sb / na
    va = max(qa / na - ma * ma, 0.0)
    vb = max(qb / na - mb * mb, 0.0)
    rows.append({"j": j, "col": NUM[j] if j < len(NUM) else "?",
                 "def_mean": ma, "def_sd": float(np.sqrt(va)),
                 "fix_mean": mb, "fix_sd": float(np.sqrt(vb)),
                 "sd_ratio_fix_over_def": (float(np.sqrt(vb)) / float(np.sqrt(va)))
                 if va > 0 else float("inf")})
bad = [r for r in rows if abs(r["def_sd"] - 1.0) > 0.01]
worst = sorted(rows, key=lambda r: -abs(r["def_sd"] - 1.0))[:10]
res = {"n_train_rows": int(len(idx)), "n_cols": int(A.shape[1]),
       "n_cols_def_sd_off_gt_1pct": len(bad),
       "cols_def_sd_off_gt_1pct": [(r["col"], r["def_sd"], r["fix_sd"]) for r in bad],
       "n_cols_fix_sd_off_gt_1pct": len([r for r in rows if abs(r["fix_sd"] - 1.0) > 0.01]),
       "cols_fix_sd_off_gt_1pct": [(r["col"], r["fix_sd"]) for r in rows
                                   if abs(r["fix_sd"] - 1.0) > 0.01],
       "worst10_by_def_sd": worst, "rows": rows}
json.dump(res, open(sys.argv[5], "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "rows"},
                 ensure_ascii=False, indent=1)[:2500])

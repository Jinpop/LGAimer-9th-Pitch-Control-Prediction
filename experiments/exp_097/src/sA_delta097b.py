"""exp_097b stageA A3 — 종단 submission.csv 차이를 확률·점수 단위로 잰다.

S21: CSV 는 float_precision='round_trip' 으로 읽는다.
점수 상한: Brier 변화 = mean(dp^2) + 2*mean(dp*(p-y)) 이고
          |둘째 항| <= sqrt(mean(dp^2) * mean((p-y)^2)) <= sqrt(mean(dp^2)*REF)
          이므로 어떤 라벨에서도 |점수 이동| <= K*(m2 + 2*sqrt(m2*REF)).
"""
import json
import sys

import numpy as np
import pandas as pd

REF = 0.2498069
K = 400309.19882517256


def load(p):
    d = pd.read_csv(p, float_precision="round_trip")
    d = d.sort_values(d.columns[0], kind="stable")
    return d[d.columns[0]].to_numpy().astype(str), d[d.columns[1]].to_numpy(np.float64)


base_p = sys.argv[1]
out = sys.argv[-1]
ids0, p0 = load(base_p)
R = {"base": base_p, "n_rows": int(len(p0)), "K": K, "REF": REF, "variants": {}}
for spec in sys.argv[2:-1]:
    tag, path = spec.split("=", 1)
    ids, p = load(path)
    assert np.array_equal(ids, ids0), ("row_id 정렬 불일치", tag)
    d = p - p0
    m2 = float(np.mean(d * d))
    bound = K * (m2 + 2.0 * np.sqrt(m2 * REF))
    R["variants"][tag] = {
        "path": path,
        "byte_identical": bool(open(path, "rb").read() == open(base_p, "rb").read()),
        "n_rows_moved": int((d != 0).sum()),
        "max_abs_dp": float(np.abs(d).max()),
        "rms_dp": float(np.sqrt(m2)),
        "mean_dp_signed": float(d.mean()),
        "K_mean_dp2_score_units": K * m2,
        "max_abs_score_move_bound_pts": float(bound),
        "rows_over_1e-6": int((np.abs(d) > 1e-6).sum()),
        "rows_over_1e-9": int((np.abs(d) > 1e-9).sum()),
        "rows_over_1e-12": int((np.abs(d) > 1e-12).sum()),
        "q": {q: float(np.quantile(np.abs(d), q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
    }
json.dump(R, open(out, "w"), indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))

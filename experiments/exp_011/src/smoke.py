"""exp_011 smoke test -- exercise every code path on a subsample before the real run.

Not a result. Catches shape/lookup/NaN bugs in minutes instead of an hour.
"""

import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import (  # noqa: E402
    DERIVED_COLS,
    PRIOR_MODES,
    RAW_RATE_COLS,
    SHRUNK_COLS,
    TARGET,
    build_features,
    build_prior_table,
)
from train import base_features, build_model, fit_predict, gauges  # noqa: E402

DATA = os.path.expanduser("~/SungjinPark/data")

t0 = time.time()
df = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
# keep every (season, game_type) cell represented
sub = df.groupby(["season", "game_type"], group_keys=False).apply(
    lambda g: g.head(12000)
).reset_index(drop=True)
print(f"subsample {sub.shape} from {df.shape} in {time.time()-t0:.1f}s", flush=True)

priors = build_prior_table(sub, sorted(int(s) for s in sub["season"].unique()) + [2025])
print("prior modes:", list(priors["prior"]), "gt keys sample:",
      list(priors["prior_gt"]["cum"]["p_success"])[:4], flush=True)

BASE = base_features()
assert len(BASE) == 47

cache = {}
for mode in PRIOR_MODES:
    for gt in (False, True):
        f, added = build_features(sub, priors, {"pitcher": 100, "batter": 100, "mix": 100},
                                  mode=mode, add_derived=True, gt_split=gt)
        assert added == SHRUNK_COLS + DERIVED_COLS, added
        nan = f[SHRUNK_COLS].isna().to_numpy().sum()
        lo = float(f[SHRUNK_COLS].to_numpy().min())
        hi = float(f[SHRUNK_COLS].to_numpy().max())
        print(f"  mode={mode:7s} gt_split={int(gt)} shrunk nan={nan} range=[{lo:.4f},{hi:.4f}]",
              flush=True)
        assert nan == 0 and 0.0 <= lo and hi <= 1.0
        cache[(mode, gt)] = f["sh_p_success"].to_numpy(float)
for mode in PRIOR_MODES:
    d = float(np.max(np.abs(cache[(mode, True)] - cache[(mode, False)])))
    nF = int((sub["game_type"] == "F").sum())
    print(f"  gt_split effect on sh_p_success, mode={mode}: max|diff|={d:.5f} "
          f"(F rows in subsample: {nF})", flush=True)
    assert d > 0, "gt_split produced no change -- lookup is not splitting"

# the real fit path, both mask arms, tiny forest
f, _ = build_features(sub, priors, {"pitcher": 100, "batter": 100, "mix": 100},
                      mode="trend3", add_derived=True, gt_split=True)
feats_RC = [c for c in BASE if c not in RAW_RATE_COLS] + SHRUNK_COLS + DERIVED_COLS
for tag, feats, mask in [("RB", BASE + SHRUNK_COLS + DERIVED_COLS, False),
                         ("RC", feats_RC, False),
                         ("MB", BASE + SHRUNK_COLS + DERIVED_COLS, True)]:
    res = fit_predict(f, feats, 2022, 2023, mask_F=mask)
    g, is_r = gauges(f, res)
    print(f"  {tag} mask={int(mask)} n_train={res['n_train']} "
          f"all={g['valid_all']['brier']:.6f} R={g['valid_R']['brier']:.6f} "
          f"F={g['valid_F']['brier']:.6f} mtry={res['max_features_resolved']}", flush=True)

# test stub path
test = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")
tb, added = build_features(test, priors, {"pitcher": 100, "batter": 100, "mix": 100},
                           mode="trend3", add_derived=True, gt_split=True)
single = np.vstack([
    build_features(test.iloc[[i]].copy(), priors, {"pitcher": 100, "batter": 100, "mix": 100},
                   mode="trend3", add_derived=True, gt_split=True)[0][added].to_numpy(float)
    for i in range(len(test))
])
dev = float(np.nanmax(np.abs(single - tb[added].to_numpy(float))))
print(f"  test stub: rows={len(tb)} added={len(added)} row-independence dev={dev:.3e}", flush=True)
assert dev == 0.0
model = build_model(BASE + SHRUNK_COLS + DERIVED_COLS)
model.fit(f[BASE + SHRUNK_COLS + DERIVED_COLS], f[TARGET].to_numpy())
p = model.predict_proba(tb[BASE + SHRUNK_COLS + DERIVED_COLS])[:, 1]
print(f"  stub preds: {np.round(p, 4).tolist()}", flush=True)
print(f"SMOKE OK in {(time.time()-t0)/60:.2f} min", flush=True)

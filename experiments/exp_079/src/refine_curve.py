"""exp_079 stage1 보강 — 봉우리 부근 10그루 간격 정밀 격자.

거친 격자(50 간격)에서 argmax 550 · 이득 +2.79098 이 나왔다.  사전 등록 문턱
+3.0 바로 아래이므로 **격자 해상도가 판정을 가르는지**를 먼저 닫는다.
🔴 문턱을 바꾸지 않는다 — 같은 양을 더 촘촘히 잴 뿐이다 (S8).
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import json         # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import catboost as cb  # noqa: E402

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
SEEDS = [42, 777, 2024, 101]
FINE = list(range(360, 701, 10))
T0 = time.time()
sys.path.insert(0, f"{OUT}/src")
import surface079 as S              # noqa: E402

FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
df = pd.read_parquet(FRAME, columns=sorted(set(FE + ["season"])))
vam = np.where(df["season"].to_numpy() == 2024)[0]
Xv = df[FE].iloc[vam].copy()
for c in CATC:
    Xv[c] = Xv[c].astype(str)
del df
print(f"프레임 {time.time()-T0:.1f}s · 정밀격자 {len(FINE)}점", flush=True)

LG = {}
for s in SEEDS:
    m = cb.CatBoostClassifier()
    m.load_model(f"{OUT}/models/oot_s{s}_i1600_t6.cbm")
    for k in FINE:
        p = m.predict_proba(Xv, ntree_end=k, thread_count=6)[:, 1].astype(np.float64)
        LG.setdefault(k, []).append(S.lg(p))
    print(f"  s{s} 완료 {time.time()-T0:.1f}s", flush=True)

l354 = np.mean([S.lg(np.load(f"{OUT}/cache/p_oot_s{s}_i1600_t6_k354.npy"))
                for s in SEEDS], axis=0)
origin, _ = S.surface_pts(l354)
arm0, _ = S.arm_pts(l354)
mu0, sd0 = float(l354.mean()), float(l354.std())

CUR = {"354": {"blend_pts": origin, "delta": 0.0, "arm_pts": arm0,
               "arm_delta": 0.0, "sd_ratio": 1.0, "scale_matched_delta": 0.0}}
for k in FINE:
    lk = np.mean(LG[k], axis=0)
    bp, _ = S.surface_pts(lk)
    ap, _ = S.arm_pts(lk)
    mu, sd = float(lk.mean()), float(lk.std())
    bm, _ = S.surface_pts(mu0 + (lk - mu) * (sd0 / sd))
    CUR[str(k)] = {"blend_pts": bp, "delta": bp - origin, "arm_pts": ap,
                   "arm_delta": ap - arm0, "sd_ratio": sd / sd0,
                   "scale_matched_delta": bm - origin}

ks = sorted(int(k) for k in CUR)
kb = max(ks, key=lambda k: CUR[str(k)]["blend_pts"])
g = CUR[str(kb)]["delta"]
R = {"exp_id": "exp_079", "part": "stage1_refine", "fine_grid": FINE,
     "origin_our_354_blend_pts": origin, "curve": CUR,
     "argmax_trees": kb, "gain_pts": g,
     "scale_matched_gain_at_argmax": CUR[str(kb)]["scale_matched_delta"],
     "flat_region": {f"within_{t}pt": [k for k in ks
                                       if CUR[str(kb)]["blend_pts"]
                                       - CUR[str(k)]["blend_pts"] <= t]
                     for t in (0.10, 0.25, 0.50, 1.00)},
     "verdict_unchanged_by_resolution": bool(g < 3.0),
     "🔴": "사전 등록 문턱 +3.0 / +8.0 을 바꾸지 않았다.  정밀격자는 같은 양을 "
           "더 촘촘히 잰 것이며, 봉우리가 문턱 아래임을 확인하기 위한 것이다."}
json.dump(R, open(f"{OUT}/out/stage1_refine.json", "w"), indent=1,
          ensure_ascii=False)
print("\n트리수  블렌드Δ    척도맞춤Δ   팔Δ      sd비", flush=True)
for k in ks:
    c = CUR[str(k)]
    print(f"{k:5d}  {c['delta']:+8.5f}  {c['scale_matched_delta']:+8.5f}  "
          f"{c['arm_delta']:+8.4f}  {c['sd_ratio']:.5f}", flush=True)
print(f"\n정밀 argmax {kb}그루 · 이득 {g:+.5f}점 "
      f"(척도맞춤 {CUR[str(kb)]['scale_matched_delta']:+.5f})", flush=True)
print(f"봉우리가 문턱 +3.0 아래인가: {g < 3.0}", flush=True)
print(f"평탄(±0.25점) {R['flat_region']['within_0.25pt']}", flush=True)
print(f"벽시계 {(time.time()-T0)/60:.2f}분", flush=True)

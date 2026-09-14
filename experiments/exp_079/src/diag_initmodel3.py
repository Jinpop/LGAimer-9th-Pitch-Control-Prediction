"""exp_079 진단 3 — 정책 4종 완주 + 직렬화 가능성.

진단 2 가 원인을 확정했다:
  weights=[1.0, 0.0] (새 트리 기여 = 0) 인데도
    KeepAllTables                 → 접두 정확히 0.0
    IntersectingCountersAverage   → max|Δ| 2.382827e-02, 252,237행
  = 편차는 **추가된 트리가 아니라 CTR 표 병합**이 만든다.
    `fit(init_model=...)` 은 기본 정책 IntersectingCountersAverage 를 쓴다.

남은 질문: 접두를 보존하는 KeepAllTables 모델을 **저장할 수 있는가**.
진단 2 에서 `.cbm` 저장이 `Some CTR data are lost` 로 실패했다.  저장이 안 되면
출하할 수 없으므로 config stage2 의 설계가 성립하지 않는다.
"""
import os

THREADS = 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)

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
SRC = f"{MODEL}/cat_42.cbm"
NEWP = f"{OUT}/models/diag_newonly_s42_i346.cbm"
T0 = time.time()
R = {"exp_id": "exp_079", "part": "diag_initmodel3",
     "확정된_원인": "fit(init_model=) 의 접두 편차는 추가된 트리가 아니라 "
                 "CTR 표 병합(기본 정책 IntersectingCountersAverage)이 만든다. "
                 "weights=[1,0] 분리 시험에서 새 트리 기여가 0 인데도 같은 "
                 "편차(2.382827e-02, 252,237행)가 나왔다."}

FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
ci = [FE.index(c) for c in CATC]
df = pd.read_parquet(FRAME, columns=sorted(set(FE + ["season", "game_type",
                                                     "control_success"])))
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm = np.where(~((s <= 2022) & (g == "F")))[0]
vam = np.where(s == 2024)[0]
y = df["control_success"].to_numpy(np.int64)
Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
Xtr, ytr, Xv = Xc.iloc[trm], y[trm], Xc.iloc[vam]
del Xc

m0 = cb.CatBoostClassifier()
m0.load_model(SRC)
p0 = m0.predict_proba(Xv, thread_count=THREADS)[:, 1].astype(np.float64)
mN = cb.CatBoostClassifier()
mN.load_model(NEWP)
print(f"준비 {time.time()-T0:.1f}s · m0 {m0.tree_count_} · mN {mN.tree_count_}",
      flush=True)


def prob(m, ntree_end=0):
    r = np.asarray(m.predict(Xv, prediction_type="Probability",
                             ntree_end=ntree_end, thread_count=THREADS),
                   np.float64)
    return r[:, 1] if r.ndim == 2 else r


POL = {}
for pol in ("KeepAllTables", "IntersectingCountersAverage",
            "LeaveMostDiversifiedTable", "FailIfCtrsIntersects"):
    rec = {}
    try:
        ms = cb.sum_models([m0, mN], weights=[1.0, 1.0], ctr_merge_policy=pol)
    except Exception as e:                                   # noqa: BLE001
        rec["merge_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        POL[pol] = rec
        print(f"[{pol}] 병합 실패: {rec['merge_error'][:110]}", flush=True)
        continue
    pp = prob(ms, 354)
    d = np.abs(pp - p0)
    rec.update({"tree_count": int(ms.tree_count_),
                "prefix_max_abs_diff": float(d.max()),
                "prefix_n_diff_rows": int((d > 0).sum()),
                "prefix_bitwise_equal": bool(np.array_equal(pp, p0)),
                "full_pred_mean": float(prob(ms).mean())})
    for tag, fn in (("cbm", lambda p: ms.save_model(p)),
                    ("cbm_with_pool",
                     lambda p: ms.save_model(p, pool=cb.Pool(
                         Xtr.iloc[:20000], ytr[:20000], cat_features=ci))),
                    ("json", lambda p: ms.save_model(p, format="json"))):
        path = f"{OUT}/models/sum_{pol}_s42_T700.{tag}"
        try:
            fn(path)
            rec[f"save_{tag}"] = {"ok": True,
                                  "bytes": os.path.getsize(path)}
        except Exception as e:                               # noqa: BLE001
            rec[f"save_{tag}"] = {"ok": False,
                                  "error": f"{type(e).__name__}: {str(e)[:200]}"}
    POL[pol] = rec
    print(f"[{pol}] 접두동일 {rec['prefix_bitwise_equal']} |Δ| "
          f"{rec['prefix_max_abs_diff']:.6e} · 저장 cbm "
          f"{rec['save_cbm']['ok']} / pool {rec['save_cbm_with_pool']['ok']} "
          f"/ json {rec['save_json']['ok']}", flush=True)

R["policies"] = POL
ok_pref = [p for p, v in POL.items() if v.get("prefix_bitwise_equal")]
ok_save = [p for p in ok_pref
           if any(POL[p].get(f"save_{t}", {}).get("ok")
                  for t in ("cbm", "cbm_with_pool", "json"))]
R["prefix_preserving_policies"] = ok_pref
R["prefix_preserving_AND_saveable"] = ok_save
R["판정"] = ("접두를 보존하면서 저장까지 되는 정책이 있으면 config stage2 설계가 "
           "복구된다.  없으면 배포 CatBoost 를 '앞 354그루 고정 + 트리 추가' 로 "
           "출하하는 길이 catboost 1.2.10 에서 막혀 있는 것이다.")
R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/diag_initmodel3.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\n접두 보존 정책 {ok_pref}\n보존+저장 가능 {ok_save}", flush=True)
print(f"기록 {OUT}/out/diag_initmodel3.json · {R['wall_min']:.2f}분", flush=True)

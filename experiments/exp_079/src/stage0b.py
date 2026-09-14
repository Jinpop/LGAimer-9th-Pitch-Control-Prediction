"""exp_079 stage0 재시도 — G0a 1차 실패 원인 규명 + 변형 시도.

1차(stage0.py, out/stage0.json 에 보존)는 아래로 실패했다:
    CatBoostError: model.cpp:1747: Cannot sum models with different class params

배포 cat_42.cbm 의 `class_names` 는 정수 [0, 1] 인데 exp_077 계열은 y 를
float64 로 넘겨 [0.0, 1.0] 이 된다.  이것은 **메타데이터 불일치**이지 모델
차이가 아니다 — 그러나 그 가정 자체를 실측으로 확인해야 한다(G0a3).

이 파일은 stage0.py 를 지우거나 덮어쓰지 않는다.  산출은 out/stage0b.json.
"""
import os
import sys

THREADS = 6
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(THREADS)

import hashlib          # noqa: E402
import json             # noqa: E402
import time             # noqa: E402

import numpy as np      # noqa: E402
import pandas as pd     # noqa: E402
import catboost as cb   # noqa: E402
from scipy.special import expit, logit   # noqa: E402

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BO/pkg_BO/mate/model"
FRAME = f"{H}/exp071/cache/frame071.parquet"
OUT = f"{H}/exp079"
C77 = f"{H}/exp077/cache"
CATC = ["game_type", "base_state", "pitcher_team_id", "batter_team_id",
        "idea_matchup_pitch"]
REF = 0.2498069
T0 = time.time()
R = {"exp_id": "exp_079", "part": "stage0b", "threads_used": THREADS,
     "catboost_version": cb.__version__,
     "run1_failure": "CatBoostError: catboost/libs/model/model.cpp:1747: "
                     "Cannot sum models with different class params "
                     "(out/stage0.json 에 보존)"}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
ci = [FE.index(c) for c in CATC]
assert len(FE) == 99 and ci == [4, 10, 13, 14, 45]

need = sorted(set(FE + ["season", "game_type", "control_success", "row_id"]))
df = pd.read_parquet(FRAME, columns=need)
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm_oot = np.where((s <= 2023) & ~((s <= 2022) & (g == "F")))[0]
trm_dep = np.where(~((s <= 2022) & (g == "F")))[0]
vam = np.where(s == 2024)[0]
y_all = df["control_success"].to_numpy(np.float64)
y_int = df["control_success"].to_numpy(np.int64)
rid_frame = df["row_id"].to_numpy().astype(str)[vam]
assert len(trm_oot) == 1116277 and len(trm_dep) == 1369784 and len(vam) == 253507
Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
print(f"프레임 준비 {time.time()-T0:.1f}s", flush=True)

P = dict(learning_rate=0.06, depth=6, l2_leaf_reg=10,
         loss_function="Logloss", bootstrap_type="Bernoulli", subsample=0.8,
         has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=0, thread_count=THREADS)

m_dep = cb.CatBoostClassifier()
m_dep.load_model(f"{MODEL}/cat_42.cbm")
DP = m_dep.get_all_params()
p_dep = m_dep.predict_proba(Xc.iloc[vam], thread_count=THREADS)[:, 1].astype(np.float64)
R["deploy_ref"] = {
    "sha256": sha(f"{MODEL}/cat_42.cbm"), "tree_count": int(m_dep.tree_count_),
    "class_names_repr": repr(DP.get("class_names")),
    "class_names_types": [type(v).__name__ for v in DP.get("class_names", [])],
    "classes_count": DP.get("classes_count"),
    "boost_from_average": DP.get("boost_from_average"),
    "pred_vs_exp077_cache_maxabs":
        float(np.abs(p_dep - np.load(f"{C77}/dep_pred2024_seed42.npy")).max())}
print("배포 class_names:", R["deploy_ref"]["class_names_repr"],
      R["deploy_ref"]["class_names_types"], flush=True)

# ── G0a  init_model 접두 항등 — 변형 시도 ───────────────────────────
print("\n=== G0a init_model 접두 항등 (변형 시도) ===", flush=True)
VAR = {}


def try_variant(name, kw, yv):
    t = time.time()
    mm = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                               iterations=20, **{**P, **kw})
    rec = {"kwargs": {k: repr(v) for k, v in kw.items()},
           "y_dtype": str(yv.dtype)}
    try:
        mm.fit(Xc.iloc[trm_dep], yv[trm_dep], cat_features=ci,
               init_model=f"{MODEL}/cat_42.cbm")
    except Exception as e:                                # noqa: BLE001
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["PASS"] = False
        VAR[name] = rec
        print(f"  [{name}] FAIL  {rec['error'][:120]}", flush=True)
        return None
    rec["ext_tree_count"] = int(mm.tree_count_)
    pp = mm.predict_proba(Xc.iloc[vam], ntree_end=354,
                          thread_count=THREADS)[:, 1].astype(np.float64)
    d = np.abs(pp - p_dep)
    rec["prefix_max_abs_diff"] = float(d.max())
    rec["prefix_n_diff_rows"] = int((d > 0).sum())
    rec["prefix_bitwise_equal"] = bool(np.array_equal(pp, p_dep))
    pf = mm.predict_proba(Xc.iloc[vam],
                          thread_count=THREADS)[:, 1].astype(np.float64)
    rec["ext374_vs_dep354_max_abs_diff"] = float(np.abs(pf - p_dep).max())
    rec["sec"] = time.time() - t
    rec["PASS"] = bool(rec["prefix_bitwise_equal"] and rec["ext_tree_count"] == 374)
    VAR[name] = rec
    print(f"  [{name}] tree {rec['ext_tree_count']} · 접두 |Δ| "
          f"{rec['prefix_max_abs_diff']:.3e} · 비트동일 "
          f"{rec['prefix_bitwise_equal']} · PASS {rec['PASS']}", flush=True)
    return mm


try_variant("v1_int_labels", {}, y_int)
try_variant("v2_class_names_int", {"class_names": [0, 1]}, y_all)
R["G0a_variants"] = VAR
ok = [k for k, v in VAR.items() if v.get("PASS")]
R["G0a_init_model_prefix_identity"] = {
    "PASS": bool(ok), "passing_variants": ok,
    "원인": "1차 실패는 라벨 dtype 이 float64 라 class_names 가 [0.0, 1.0] 로 "
            "잡혀 배포본의 정수 [0, 1] 과 어긋난 것. 모델 산술 문제가 아니다."}
json.dump(R, open(f"{OUT}/out/stage0b.json", "w"), indent=1, ensure_ascii=False)
if not ok:
    print("\n[STOP] G0a 전 변형 실패 — 즉시 정지.", flush=True)
    sys.exit(3)
print(f"[G0a] PASS  변형 {ok}", flush=True)

# ── G0a3  라벨 dtype 이 모델을 바꾸는가 (exp_077 비교가능성) ─────────
print("\n=== G0a3 라벨 dtype 중립성 ===", flush=True)
t = time.time()
pf_, pi_ = {}, {}
for nm, yv, box in (("float64", y_all, pf_), ("int64", y_int, pi_)):
    mm = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                               iterations=30, **P)
    mm.fit(Xc.iloc[trm_oot], yv[trm_oot], cat_features=ci)
    box["p"] = mm.predict_proba(Xc.iloc[vam],
                                thread_count=THREADS)[:, 1].astype(np.float64)
R["G0a3_label_dtype_neutral"] = {
    "iters": 30, "mask": "oot",
    "max_abs_diff": float(np.abs(pf_["p"] - pi_["p"]).max()),
    "bitwise_equal": bool(np.array_equal(pf_["p"], pi_["p"])), "sec": time.time() - t,
    "해석": "참이면 exp_077(float 라벨) 산출물과 이 실험(int 라벨) 산출물이 "
            "같은 모델 계열이다 — 라벨 dtype 은 class_names 메타만 바꾼다."}
print(json.dumps(R["G0a3_label_dtype_neutral"], indent=1, ensure_ascii=False),
      flush=True)

# ── G0e  실행길이 불변 smoke (OOT, 30 vs 60) ────────────────────────
print("\n=== G0e 실행길이 불변 smoke ===", flush=True)
t = time.time()
mB = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                           iterations=60, **P)
mB.fit(Xc.iloc[trm_oot], y_all[trm_oot], cat_features=ci)
pB30 = mB.predict_proba(Xc.iloc[vam], ntree_end=30,
                        thread_count=THREADS)[:, 1].astype(np.float64)
R["G0e_runlength_invariance_smoke"] = {
    "iters_short": 30, "iters_long": 60,
    "max_abs_diff": float(np.abs(pB30 - pf_["p"]).max()),
    "n_diff_rows": int((np.abs(pB30 - pf_["p"]) > 0).sum()),
    "bitwise_equal": bool(np.array_equal(pB30, pf_["p"])), "sec": time.time() - t,
    "해석": "참이면 긴 적합을 ntree_end 로 자른 것이 짧은 적합과 같다 = "
            "stage1 곡선의 각 점이 '그 그루수까지 학습한 모델'이다."}
print(json.dumps(R["G0e_runlength_invariance_smoke"], indent=1,
                 ensure_ascii=False), flush=True)
R["G0d_thread_count"] = {
    "fit_thread_count": THREADS,
    "note": "S17 — CatBoost 에서 스레드 수는 시드급 모델 변수다(exp_077 D1: "
            "threads 2 vs 6 에서 rho 0.973547 · 점수 3.301점 차). 이 실험의 "
            "모든 적합은 thread_count=6, exp_077 OOT 적합과 같은 조건."}
R["G0b_masks"] = {"n_oot_le2023": int(len(trm_oot)),
                  "n_deploy_le2024": int(len(trm_dep)),
                  "n_valid_2024": int(len(vam)),
                  "counted_directly": "np.where 직접 계수. 0.9·N 역산 아님 (S17)",
                  "PASS": True}
R["G0c_features"] = {"n_features_sp": 99, "cat_index": ci,
                     "sha256_features_sp": sha(f"{MODEL}/features_sp.txt"),
                     "PASS": True}
R["G0b_row_alignment"] = {
    "frame_2024_vs_exp069_row_id_equal": bool(np.array_equal(
        rid_frame, np.asarray(np.load(f"{H}/exp069/oot/row_id_oot2024.npy",
                                      allow_pickle=True)).astype(str)))}

# ── G0f  정직면 산술 항등 ───────────────────────────────────────────
print("\n=== G0f 정직면 산술 항등 ===", flush=True)
sys.path.insert(0, f"{OUT}/src")
import surface079 as S                                      # noqa: E402
base_pts, base_b = S.surface_pts(lg(S.V["p_cat"]))
R["G0f_surface_identity"] = {
    "base_pts_partner_cat": base_pts, "base_brier": base_b,
    "expected_exp078": 928.501461058684,
    "abs_diff": abs(base_pts - 928.501461058684),
    "PASS": bool(abs(base_pts - 928.501461058684) < 1e-9)}
print(json.dumps(R["G0f_surface_identity"], indent=1, ensure_ascii=False),
      flush=True)
assert R["G0f_surface_identity"]["PASS"], "정직면 산술 항등 실패"

R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/stage0b.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/out/stage0b.json · 벽시계 {R['wall_min']:.2f}분", flush=True)

"""exp_079 stage0 — 게이트.

G0a  init_model 접두 항등 (이 실험의 핵심 장치, 중단 조건)
G0b  학습 마스크 행수 (0.9·N 역산 금지, S17)
G0c  피처 99열 · 범주 인덱스 · sha
G0d  thread_count 고정·기록
G0e  실행길이 불변 smoke — trunc(긴 적합, k) == 짧은 적합 (stage1 곡선의 해석 근거)
G0f  정직면 산술 항등 — exp_078 stage4 기준 928.501461058684 재현
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
SEEDS = [42, 777, 2024, 101]
REF = 0.2498069
T0 = time.time()
R = {"exp_id": "exp_079", "part": "stage0", "threads_used": THREADS,
     "catboost_version": cb.__version__}


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


# ── G0c 피처 ────────────────────────────────────────────────────────
FE = [c for c in open(f"{MODEL}/features_sp.txt").read().split("\n") if c]
ci = [FE.index(c) for c in CATC]
R["G0c_features"] = {
    "n_features_sp": len(FE), "cat_index": ci,
    "cat_index_expected": [4, 10, 13, 14, 45],
    "sha256_features_sp": sha(f"{MODEL}/features_sp.txt"),
    "sha256_expected_exp077":
        "783c650c6dd78cdcd641d0f18e35469ced247544b5d1ebf8d138171e04498924"}
R["G0c_features"]["PASS"] = bool(
    len(FE) == 99 and ci == [4, 10, 13, 14, 45]
    and R["G0c_features"]["sha256_features_sp"]
    == R["G0c_features"]["sha256_expected_exp077"])
assert R["G0c_features"]["PASS"], R["G0c_features"]
print(f"[G0c] PASS  99열 · cat {ci}", flush=True)

# ── 프레임 ──────────────────────────────────────────────────────────
need = sorted(set(FE + ["season", "game_type", "control_success", "row_id"]))
df = pd.read_parquet(FRAME, columns=need)
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
trm_oot = np.where((s <= 2023) & ~((s <= 2022) & (g == "F")))[0]
trm_dep = np.where(~((s <= 2022) & (g == "F")))[0]
vam = np.where(s == 2024)[0]
y_all = df["control_success"].to_numpy(np.float64)
rid_frame = df["row_id"].to_numpy().astype(str)[vam]
R["G0b_masks"] = {
    "n_oot_le2023": int(len(trm_oot)), "n_oot_expected": 1116277,
    "n_deploy_le2024": int(len(trm_dep)), "n_deploy_expected": 1369784,
    "n_valid_2024": int(len(vam)), "n_valid_expected": 253507,
    "n_frame_rows": int(len(df)),
    "counted_directly": "np.where 로 직접 센 값. 0.9·N 역산 아님 (S17)"}
R["G0b_masks"]["PASS"] = bool(len(trm_oot) == 1116277
                              and len(trm_dep) == 1369784
                              and len(vam) == 253507)
assert R["G0b_masks"]["PASS"], R["G0b_masks"]
print(f"[G0b] PASS  oot {len(trm_oot)} · deploy {len(trm_dep)} "
      f"· valid {len(vam)}", flush=True)

# 행 정렬 — 팀원 OOT 납품 벡터와 같은 순서인가
rid_oot = np.asarray(np.load(f"{H}/exp069/oot/row_id_oot2024.npy",
                             allow_pickle=True)).astype(str)
R["G0b_row_alignment"] = {
    "frame_2024_vs_exp069_row_id_equal": bool(np.array_equal(rid_frame,
                                                             rid_oot)),
    "n": int(len(rid_frame))}
assert R["G0b_row_alignment"]["frame_2024_vs_exp069_row_id_equal"]
print("[G0b] 행 정렬 PASS — frame071 s==2024 순서 == exp069 납품 순서",
      flush=True)

Xc = df[FE].copy()
for c in CATC:
    Xc[c] = Xc[c].astype(str)
del df
y = y_all
print(f"프레임 준비 {time.time()-T0:.1f}s", flush=True)

# ── 배포 하이퍼 (실측값 그대로) ──────────────────────────────────────
P = dict(learning_rate=0.06, depth=6, l2_leaf_reg=10,
         loss_function="Logloss", bootstrap_type="Bernoulli", subsample=0.8,
         has_time=False, one_hot_max_size=2, border_count=64,
         boosting_type="Plain", verbose=200, thread_count=THREADS)
R["G0d_thread_count"] = {
    "fit_thread_count": THREADS,
    "note": "S17 — CatBoost 에서 스레드 수는 시드급 모델 변수다(exp_077 D1: "
            "threads 2 vs 6 에서 rho 0.9736 · 점수 3.301점 차). 이 실험의 "
            "모든 적합은 thread_count=6 이며 exp_077 의 OOT 적합과 같은 조건."}

# ── G0a  init_model 접두 항등 (중단 조건) ────────────────────────────
print("\n=== G0a init_model 접두 항등 ===", flush=True)
m_dep = cb.CatBoostClassifier()
m_dep.load_model(f"{MODEL}/cat_42.cbm")
t = time.time()
p_dep = m_dep.predict_proba(Xc.iloc[vam], thread_count=THREADS)[:, 1].astype(np.float64)
sec_pred354 = time.time() - t
cached = np.load(f"{C77}/dep_pred2024_seed42.npy")
G0a = {"deploy_cat42_sha256": sha(f"{MODEL}/cat_42.cbm"),
       "deploy_tree_count": int(m_dep.tree_count_),
       "deploy_pred_vs_exp077_cache_maxabs": float(np.abs(p_dep - cached).max()),
       "sec_predict_354trees_253507rows": sec_pred354}

t = time.time()
m_ext = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                              iterations=20, **P)
err = None
try:
    m_ext.fit(Xc.iloc[trm_dep], y[trm_dep], cat_features=ci,
              init_model=f"{MODEL}/cat_42.cbm")
except Exception as e:                                    # noqa: BLE001
    err = f"{type(e).__name__}: {e}"
G0a["init_model_fit_error"] = err
if err is None:
    G0a["ext_tree_count"] = int(m_ext.tree_count_)
    p_pref = m_ext.predict_proba(Xc.iloc[vam], ntree_end=354,
                                 thread_count=THREADS)[:, 1].astype(np.float64)
    d = np.abs(p_pref - p_dep)
    G0a["prefix_max_abs_diff"] = float(d.max())
    G0a["prefix_n_diff_rows"] = int((d > 0).sum())
    G0a["prefix_bitwise_equal"] = bool(np.array_equal(p_pref, p_dep))
    p_full = m_ext.predict_proba(Xc.iloc[vam],
                                 thread_count=THREADS)[:, 1].astype(np.float64)
    G0a["ext_vs_dep_max_abs_diff_fulltrees"] = float(np.abs(p_full - p_dep).max())
    G0a["sec_fit_20_extension_deploy"] = time.time() - t
    G0a["PASS"] = bool(G0a["prefix_bitwise_equal"]
                       and G0a["ext_tree_count"] == 374)
else:
    G0a["PASS"] = False
R["G0a_init_model_prefix_identity"] = G0a
print(json.dumps(G0a, indent=1, ensure_ascii=False), flush=True)
json.dump(R, open(f"{OUT}/out/stage0.json", "w"), indent=1, ensure_ascii=False)
if not G0a["PASS"]:
    print("\n[STOP] G0a 실패 — 설계가 성립하지 않는다. 즉시 정지.", flush=True)
    sys.exit(3)
print("[G0a] PASS  접두 354그루 비트 동일", flush=True)

# ── G0e  실행길이 불변 smoke (OOT, 30 vs 60) ────────────────────────
print("\n=== G0e 실행길이 불변 smoke ===", flush=True)
t = time.time()
mA = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                           iterations=30, **P)
mA.fit(Xc.iloc[trm_oot], y[trm_oot], cat_features=ci)
pA = mA.predict_proba(Xc.iloc[vam], thread_count=THREADS)[:, 1].astype(np.float64)
mB = cb.CatBoostClassifier(random_seed=42, allow_writing_files=False,
                           iterations=60, **P)
mB.fit(Xc.iloc[trm_oot], y[trm_oot], cat_features=ci)
pB30 = mB.predict_proba(Xc.iloc[vam], ntree_end=30,
                        thread_count=THREADS)[:, 1].astype(np.float64)
R["G0e_runlength_invariance_smoke"] = {
    "iters_short": 30, "iters_long": 60,
    "max_abs_diff": float(np.abs(pB30 - pA).max()),
    "n_diff_rows": int((np.abs(pB30 - pA) > 0).sum()),
    "bitwise_equal": bool(np.array_equal(pB30, pA)),
    "sec": time.time() - t,
    "해석": "참이면 긴 적합을 ntree_end 로 자른 것이 짧은 적합과 같다 = "
            "stage1 곡선의 각 점이 '그 그루수까지 학습한 모델'이다."}
print(json.dumps(R["G0e_runlength_invariance_smoke"], indent=1,
                 ensure_ascii=False), flush=True)

# ── G0f  정직면 산술 항등 ───────────────────────────────────────────
print("\n=== G0f 정직면 산술 항등 ===", flush=True)
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
yo = y_all[vam]
BO = json.load(open(f"{H}/exp079/src/bo_constants.json"))
AA = (42, 7, 2024)
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s_}_at1400.npy")) for s_ in AA],
               axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s_}_at1400.npy")) for s_ in AA],
                axis=0)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}


def rc(z):
    c, yb = 0.0, float(yo.mean())
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - yb) < 1e-14:
            break
        d_ = (mu - yb) / float((p * (1 - p)).mean())
        c -= d_
        if abs(d_) < 1e-14:
            break
    return c


def surface_pts(lg_cat):
    """exp_078 stage34 의 oot_pts 와 동일 산술. p_cat 로짓만 바깥에서 받는다."""
    lv = (W4["LGB"] * lg(V["p_lgb"]) + W4["CAT"] * lg_cat
          + W4["NN"] * lg(V["p_nn"]) + W4["LEAF"] * lg(V["p_leaf"]))
    z = (BO["w_out"] * ((1 - BO["w_ebm"])
                        * (BO["w_v6_inner"] * lv
                           + (1 - BO["w_v6_inner"]) * lg(V["p_q"]))
                        + BO["w_ebm"] * lg(V["p_ebm"])
                        + BO["beta_ins"] * V["off"])
         + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70)
    p = expit(z + rc(z))
    b = float(np.mean((p - yo) ** 2))
    return pts(b), b


base_pts, base_b = surface_pts(lg(V["p_cat"]))
R["G0f_surface_identity"] = {
    "base_pts_partner_cat": base_pts, "base_brier": base_b,
    "expected_exp078": 928.501461058684,
    "abs_diff": abs(base_pts - 928.501461058684),
    "BO_constants_used": BO,
    "PASS": bool(abs(base_pts - 928.501461058684) < 1e-9)}
print(json.dumps({k: v for k, v in R["G0f_surface_identity"].items()
                  if k != "BO_constants_used"}, indent=1, ensure_ascii=False),
      flush=True)
assert R["G0f_surface_identity"]["PASS"], "정직면 산술 항등 실패"

R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/stage0.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/out/stage0.json · 벽시계 {R['wall_min']:.2f}분", flush=True)

"""exp_077 진단 — G0d 가 대역 밖으로 나온 이유를 좁힌다.

D1 스레드   : 같은 시드·같은 마스크·같은 프레임에서 threads 2 vs 6
D2 시드대역 : 우리 12시드 서로 간 ρ (같은 코드·같은 프레임, 시드만 다름)
D3 4평균대역: 서로 겹치지 않는 4시드 평균 두 개의 ρ — cat4 재현 비교의 올바른 기준자
D4 메타     : 배포 .cbm 이 기록한 catboost 버전·학습 시각 (버전 드리프트 여부)
D5 위치     : 우리 배포4 재현 점수가 4시드 평균 분포의 어디인가
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import itertools
import json
import time

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
MODEL = f"{H}/build_BN/pkg_BN/mate/model"
OOT = f"{H}/exp069/oot"
E = f"{H}/exp077"
C = f"{E}/cache"
EPS, REF, N = 1e-6, 0.2498069, 253507
DEPLOY_SEEDS = [42, 777, 2024, 101]
NEW_SEEDS = [1, 7, 13, 77, 202, 404, 808, 1234]
ALL12 = DEPLOY_SEEDS + NEW_SEEDS
T0 = time.time()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


y = np.load(f"{C}/y2024.npy").astype(np.float64)
ybar = float(y.mean())


def score_z(z):
    c = 0.0
    for _ in range(60):
        p = expit(z + c)
        d = (float(p.mean()) - ybar) / float((p * (1.0 - p)).mean())
        c -= d
        if abs(d) < 1e-15:
            break
    p = expit(z + c)
    b = float(np.mean((p - y) ** 2))
    return b, 100000.0 * (1.0 - b / REF)


L = {s: lg(np.load(f"{C}/_cat_oot_s{s}_t6.npy")) for s in ALL12}
D = {}

# ── D1 스레드 ──
f2 = f"{C}/_cat_oot_s42_t2.npy"
if os.path.exists(f2):
    L2 = lg(np.load(f2))
    j2 = json.load(open(f"{C}/_cat_oot_s42_t2.json"))
    j6 = json.load(open(f"{C}/_cat_oot_s42_t6.json"))
    D["D1_threads"] = {
        "rho_logit_t2_vs_t6_same_seed42": float(np.corrcoef(L2, L[42])[0, 1]),
        "rms_logit_diff": float(np.sqrt(np.mean((L2 - L[42]) ** 2))),
        "max_abs_prob_diff": float(np.abs(expit(L2) - expit(L[42])).max()),
        "score_t2": score_z(L2)[1], "score_t6": score_z(L[42])[1],
        "sec_fit_t2": j2["sec_fit"], "sec_fit_t6": j6["sec_fit"],
        "learn_logloss_t2": j2["learn_best_error"],
        "learn_logloss_t6": j6["learn_best_error"],
        "해설": ("같은 시드·같은 마스크·같은 프레임.  달라진 것은 스레드 수뿐이다. "
                 "여기서 나오는 ρ 가 '스레드만 바뀌었을 때의 재현 상한' 이다.")}

# ── D2 시드 대역 (우리 코드·우리 프레임) ──
pr = {}
for a, b in itertools.combinations(ALL12, 2):
    pr[f"{a}_{b}"] = float(np.corrcoef(L[a], L[b])[0, 1])
v = np.array(list(pr.values()))
D["D2_our_seed_pair_band"] = {
    "n_pairs": len(pr), "min": float(v.min()), "max": float(v.max()),
    "mean": float(v.mean()), "sd": float(v.std(ddof=1)),
    "해설": "같은 코드·같은 프레임에서 시드만 다를 때 예측 로짓이 얼마나 닮는가."}

# ── D3 4시드 평균 두 개의 ρ (겹치지 않게) ──
m4 = []
for cmb in itertools.combinations(range(8), 4):
    rest = [i for i in range(8) if i not in cmb]
    A = np.mean([L[NEW_SEEDS[i]] for i in cmb], axis=0)
    B = np.mean([L[NEW_SEEDS[i]] for i in rest], axis=0)
    m4.append(float(np.corrcoef(A, B)[0, 1]))
m4 = np.array(m4[:35])          # 70개 조합은 35쌍의 중복
D["D3_disjoint_4mean_band"] = {
    "n_pairs": int(m4.size), "min": float(m4.min()), "max": float(m4.max()),
    "mean": float(m4.mean()), "sd": float(m4.std(ddof=1)),
    "해설": ("서로 겹치지 않는 4시드 평균 둘 사이의 ρ.  '우리 cat4 재현 vs "
             "팀원 cat4' 를 재는 올바른 기준자다 — 둘 다 4개를 평균한 벡터이므로 "
             "단일 시드쌍 대역과 비교하면 틀린다.")}
L_mate = lg(np.load(f"{OOT}/p_cat_oot2024.npy"))
L_ours4 = np.mean([L[s] for s in DEPLOY_SEEDS], axis=0)
rho_repro = float(np.corrcoef(L_ours4, L_mate)[0, 1])
D["D3_disjoint_4mean_band"]["rho_ours4_vs_mate4"] = rho_repro
D["D3_disjoint_4mean_band"]["repro_inside_band"] = bool(
    m4.min() <= rho_repro <= m4.max())
D["D3_disjoint_4mean_band"]["repro_above_band"] = bool(rho_repro > m4.max())

# ── D4 배포 아티팩트 메타 ──
meta = {}
for s in DEPLOY_SEEDS:
    m = CatBoostClassifier(); m.load_model(f"{MODEL}/cat_{s}.cbm")
    md = dict(m.get_metadata())
    meta[str(s)] = {k: (v[:400] if isinstance(v, str) else str(v)[:400])
                    for k, v in md.items()}
mine = CatBoostClassifier(); mine.load_model(f"{C}/_cat_deploy_s42_t6.cbm")
meta["ours_deploy_s42_t6"] = {k: (v[:400] if isinstance(v, str) else str(v)[:400])
                              for k, v in dict(mine.get_metadata()).items()}
D["D4_artifact_metadata"] = meta

# ── D5 4시드 평균 점수 분포 위에서의 위치 ──
sc4 = []
for cmb in itertools.combinations(range(8), 4):
    sc4.append(score_z(np.mean([L[NEW_SEEDS[i]] for i in cmb], axis=0))[1])
sc4 = np.array(sc4)
s_ours4 = score_z(L_ours4)[1]
s_mate4 = score_z(L_mate)[1]
D["D5_4seed_mean_score_distribution"] = {
    "n_subsets_from_our_8_new": int(sc4.size),
    "min": float(sc4.min()), "max": float(sc4.max()),
    "mean": float(sc4.mean()), "sd": float(sc4.std(ddof=1)),
    "our_deploy4_repro": s_ours4,
    "mate_deploy4_delivered": s_mate4,
    "gap_pts": s_ours4 - s_mate4,
    "gap_in_sd_units": (s_ours4 - s_mate4) / float(sc4.std(ddof=1)),
    "percentile_of_ours4": float((sc4 < s_ours4).mean() * 100),
    "percentile_of_mate4": float((sc4 < s_mate4).mean() * 100),
    "해설": ("우리 새 8시드에서 만든 4시드 평균 70개의 점수 분포.  "
             "우리 재현본과 팀원 납품본의 차이를 이 분포의 산포로 재면 "
             "'시드 뽑기 운' 인지 '체계적 차이' 인지 가늠할 수 있다.")}

# ── D6 배포쌍 잎값 상관은 정보가 없다 ──
b0 = json.load(open(f"{E}/out/stage0.json"))["G0e_deploy_seed_pair_band"]
lf = [v["rho_leaf"] for v in b0["pairs"].values() if v["rho_leaf"] is not None]
D["D6_leaf_correlation_is_uninformative"] = {
    "deploy_sibling_rho_leaf": lf,
    "our_refit_vs_deploy42_rho_leaf": json.load(
        open(f"{E}/out/stage2.json"))["G0d_identity"]["rho_leaf"],
    "판정": ("🔴 배포 형제 모델끼리도 잎값 상관이 0.0008~0.0211 로 사실상 0 이다. "
             "독립적으로 자란 트리의 잎값 배열은 원소 위치가 서로 다른 분기를 "
             "가리키므로 상관이 의미를 갖지 않는다.  config 가 요구한 '잎값 상관' "
             "은 이 모델 계열에서 판정 근거가 될 수 없고, 예측 상관만 쓸 수 있다.")}

D["wall_min"] = (time.time() - T0) / 60.0
json.dump(D, open(f"{E}/out/diag.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in D.items() if k != "D4_artifact_metadata"},
                 ensure_ascii=False, indent=1)[:3000])
print("--- D4 메타 요약 ---")
for k, v in D["D4_artifact_metadata"].items():
    print(k, {kk: vv for kk, vv in v.items()
              if kk in ("catboost_version_info", "train_finish_time",
                        "model_guid", "params")}.get("catboost_version_info", "?")[:120])

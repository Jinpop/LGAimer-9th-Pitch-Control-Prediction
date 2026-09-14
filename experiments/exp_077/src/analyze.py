"""exp_077 마무리 분석 — 짝지은 Δ 의 산포와 1/n 모형 대조."""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import itertools
import json

import numpy as np
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT = f"{H}/exp069/oot"
OOT3 = f"{H}/exp070/oot3"
E = f"{H}/exp077"
C = f"{E}/cache"
EPS, REF, N = 1e-6, 0.2498069, 253507
DEPLOY_SEEDS = [42, 777, 2024, 101]
NEW_SEEDS = [1, 7, 13, 77, 202, 404, 808, 1234]
ALL12 = DEPLOY_SEEDS + NEW_SEEDS
W_LGB, W_CAT, W_LEAF = 0.05, 0.68, 0.12
W_V6_IN, W_EBM, BETA_INS = 0.70, -0.10, 0.03
W_OUT, C_AA, C_T70 = 0.70, 0.25244, 0.04756
ALPHA, BETA_SKEW = 1.07911, -0.059151
AA_SEEDS = (42, 7, 2024)


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


L_lgb = lg(np.load(f"{OOT}/p_lgb_oot2024.npy"))
L_leaf = lg(np.load(f"{OOT}/p_leaf_oot2024.npy"))
L_q = lg(np.load(f"{OOT}/p_q_oot2024.npy"))
L_ebm = lg(np.load(f"{OOT3}/p_ebm_oot2024.npy"))
off = np.load(f"{OOT}/off_oot2024.npy")
L_mate = lg(np.load(f"{OOT}/p_cat_oot2024.npy"))
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA_SEEDS], axis=0)
L = {s: lg(np.load(f"{C}/_cat_oot_s{s}_t6.npy")) for s in ALL12}


def blend(L_cat):
    v6 = (W_LGB * L_lgb + W_CAT * L_cat + W_LEAF * L_leaf) / (W_LGB + W_CAT + W_LEAF)
    inner = (1 - W_EBM) * (W_V6_IN * v6 + (1 - W_V6_IN) * L_q) + W_EBM * L_ebm \
        + BETA_INS * off
    z = W_OUT * inner + C_AA * L_aa + C_T70 * L_t70
    return score_z(ALPHA * z + BETA_SKEW * z * z)[1]


def arm(L_cat):
    return score_z(L_cat)[1]


A = {}
L12 = np.mean([L[s] for s in ALL12], axis=0)
s12_b, s12_a = blend(L12), arm(L12)

# ── 짝지은 Δ 산포: 어느 4개에서 출발하느냐에 따라 12로 갈 때의 이득 ──
d_b, d_a, s4b = [], [], []
for cmb in itertools.combinations(ALL12, 4):
    Lc = np.mean([L[s] for s in cmb], axis=0)
    b_ = blend(Lc)
    s4b.append(b_)
    d_b.append(s12_b - b_)
    d_a.append(s12_a - arm(Lc))
d_b, d_a, s4b = np.array(d_b), np.array(d_a), np.array(s4b)
A["paired_delta_4_to_12_all_495_subsets"] = {
    "n_subsets": int(d_b.size),
    "blend": {"mean": float(d_b.mean()), "sd": float(d_b.std(ddof=1)),
              "min": float(d_b.min()), "max": float(d_b.max()),
              "pct_positive": float((d_b > 0).mean() * 100),
              "pct_above_1.0": float((d_b >= 1.0).mean() * 100),
              "p05": float(np.percentile(d_b, 5)),
              "p50": float(np.percentile(d_b, 50)),
              "p95": float(np.percentile(d_b, 95))},
    "arm": {"mean": float(d_a.mean()), "sd": float(d_a.std(ddof=1)),
            "min": float(d_a.min()), "max": float(d_a.max()),
            "pct_positive": float((d_a > 0).mean() * 100)},
    "해설": ("12시드 고정 · 출발 4시드를 495가지로 바꿔가며 잰 이득.  "
             "🔴 '4 -> 12 의 이득' 은 하나의 수가 아니라 이 분포다. "
             "어느 4개에서 출발했느냐가 이득만큼 크게 움직인다."),
}

# 실제 두 출발점의 위치
d_ours4 = s12_b - blend(np.mean([L[s] for s in DEPLOY_SEEDS], axis=0))
L12m = (4.0 * L_mate + sum(L[s] for s in NEW_SEEDS)) / 12.0
d_mate4 = blend(L12m) - blend(L_mate)
A["two_actual_starting_points"] = {
    "우리_배포4_재현": {"delta_blend": d_ours4,
                        "percentile_in_495": float((d_b < d_ours4).mean() * 100)},
    "팀원_배포4_납품": {"delta_blend": d_mate4,
                        "note": "출발점이 우리 495 분포 밖(다른 코드 경로)이라 백분위 없음"},
}

# ── 구간별 이득 ──
seg = {}
for a_, b_ in [(1, 2), (2, 4), (4, 6), (6, 8), (8, 12), (4, 8), (4, 12), (6, 12)]:
    va, vb = [], []
    for cmb in itertools.combinations(range(12), a_):
        va.append(blend(np.mean([L[ALL12[i]] for i in cmb], axis=0)))
    for cmb in itertools.combinations(range(12), b_):
        vb.append(blend(np.mean([L[ALL12[i]] for i in cmb], axis=0)))
    seg[f"{a_}->{b_}"] = {"mean_from": float(np.mean(va)), "mean_to": float(np.mean(vb)),
                          "delta_of_means": float(np.mean(vb) - np.mean(va))}
A["segment_gains_blend_subset_means"] = seg

# ── 1/n 모형 ──
sc = json.load(open(f"{E}/out/stage2.json"))["seed_curve"]
A["one_over_n_model"] = {"blend": sc["model_blend"], "arm": sc["model_arm"]}
A["orchestrator_extrapolation_check"] = {
    "지휘자_모형": "score(n) = 887.58 − 35.99/n (AA81 3시드에서 실측)",
    "지휘자_예측_4to12_팔눈금": 35.99 * (1 / 4 - 1 / 12),
    "우리_팔_b계수": sc["model_arm"]["b"],
    "우리_팔_예측_4to12": sc["model_arm"]["pred_delta_4_to_12"],
    "우리_팔_실측_4to12": float(np.mean(d_a)),
    "우리_블렌드_실측_4to12": float(np.mean(d_b)),
    "판정": ("🔴 외삽이 과대했다.  v6-Cat 의 1/n 계수는 AA81 에서 읽은 35.99 가 "
             "아니라 그보다 작다.  그리고 팔 눈금 이득이 블렌드로 전달되는 몫은 "
             "1 이 아니다(S4).")}
A["arm_to_blend_transfer_within_surface"] = {
    "arm_delta_mean": float(np.mean(d_a)), "blend_delta_mean": float(np.mean(d_b)),
    "ratio": float(np.mean(d_b) / np.mean(d_a)),
    "해설": ("같은 정직면 안에서 팔 눈금 이득이 블렌드 점수로 얼마나 옮겨지는가. "
             "이것은 LB 전이비가 아니다(S1·S2) — 면 안쪽의 희석률이다.")}

json.dump(A, open(f"{E}/out/analyze.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(A, ensure_ascii=False, indent=1)[:3500])

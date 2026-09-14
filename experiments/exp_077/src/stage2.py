"""exp_077 stage 1 합성 + stage 2 정직면 측정 + G0d 동일성.

정직면 = 팀원 OOT 납품(<=2023 학습 -> 2024 예측) 위의 배포 산술.
🔴 배포 상수는 하나도 건드리지 않는다.  움직이는 것은 cat 시드 수뿐이다.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import hashlib
import itertools
import json
import time

import numpy as np
from catboost import CatBoostClassifier
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
PKG = f"{H}/build_BN/pkg_BN"
MODEL = f"{PKG}/mate/model"
OOT = f"{H}/exp069/oot"
OOT3 = f"{H}/exp070/oot3"
E = f"{H}/exp077"
C = f"{E}/cache"

EPS = 1e-6
REF = 0.2498069
N = 253507
T0 = time.time()

DEPLOY_SEEDS = [42, 777, 2024, 101]
NEW_SEEDS = [1, 7, 13, 77, 202, 404, 808, 1234]
ALL12 = DEPLOY_SEEDS + NEW_SEEDS

# ── 배포 상수 (build_BN/pkg_BN/blend.json + mate/model/blend.json) ──
W_LGB, W_CAT, W_NN, W_LEAF = 0.05, 0.68, 0.15, 0.12   # v6 내부
W_V6_IN = 0.70          # mate/model/blend.json w_v6 (BN 이 0.5 -> 0.70)
W_EBM = -0.10           # mate/model/blend.json w_ebm
BETA_INS = 0.03         # mate/model/blend.json beta_ins
W_OUT = 0.70            # blend.json w_v6 (바깥 — 파트너 팔 지분)
C_AA = 0.25244          # blend.json c_aa
C_T70 = 0.04756         # blend.json smooth_weights.T70
ALPHA = 1.07911         # blend.json alpha
BETA_SKEW = -0.059151   # blend.json beta_skew
AA_SEEDS = (42, 7, 2024)

# exp_070 metrics_core.json 실측 (같은 자·같은 벡터) — 눈금 게이트
GATE = {"p_lgb": 892.2037, "p_cat": 921.9975, "p_leaf": 734.3269,
        "p_q": 894.626, "p_v6": 926.4687, "p_ebm": 687.988,
        "AA81": 875.5823, "T70": 824.3331}
GATE_START = 933.879134169302        # 내부 3축 (0.5, -0.10, 0.03), 재센터링
GATE_FULL6 = 928.1835495351443       # 6상수 배포 조합 (w_v6=0.5, 2차항 없음)
# exp_072 실측 전이 배율 대역
XFER = (0.3929, 0.5463)
PREREG_THRESHOLD = 1.0

M = {"exp_id": "exp_077", "stage": "1_assemble__2_honest_surface",
     "started": time.strftime("%F %T")}


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


y = np.load(f"{C}/y2024.npy").astype(np.float64)
assert y.shape == (N,)
ybar = float(y.mean())
ref_local = float(np.mean((ybar - y) ** 2))


def recenter_c(z):
    c0 = 0.0
    for _ in range(60):
        p = expit(z + c0)
        d = (float(p.mean()) - ybar) / float((p * (1.0 - p)).mean())
        c0 -= d
        if abs(d) < 1e-15:
            break
    return c0


def score_z(z):
    c = recenter_c(z)
    p = expit(z + c)
    b = float(np.mean((p - y) ** 2))
    return b, conv(b), 100000.0 * (1.0 - b / ref_local), c


# ═══════════════ 1. 재료 ═══════════════
FILES = {"p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
         "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_q": f"{OOT}/p_q_oot2024.npy",
         "p_v6_3arm": f"{OOT}/p_v6_3arm_renorm_oot2024.npy",
         "off": f"{OOT}/off_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy"}
V, sha_used = {}, {}
for k, p in FILES.items():
    sha_used[os.path.basename(p)] = sha_file(p)
    V[k] = np.load(p).astype(np.float64)
    assert V[k].shape == (N,), k
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64))
                for s in AA_SEEDS], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy").astype(np.float64))
                 for s in AA_SEEDS], axis=0)
L_q, L_ebm, off = lg(V["p_q"]), lg(V["p_ebm"]), V["off"]
L_lgb, L_leaf = lg(V["p_lgb"]), lg(V["p_leaf"])
L_cat_mate = lg(V["p_cat"])
M["sha256_inputs"] = sha_used

# 우리 12시드
ours, missing = {}, []
for s in ALL12:
    f = f"{C}/_cat_oot_s{s}_t6.npy"
    if os.path.exists(f):
        ours[s] = np.load(f).astype(np.float64)
        assert ours[s].shape == (N,)
    else:
        missing.append(s)
assert not missing, f"OOT 시드 누락 {missing}"
L_ours = {s: lg(ours[s]) for s in ALL12}
M["our_seed_files_sha256"] = {str(s): sha_file(f"{C}/_cat_oot_s{s}_t6.npy")
                              for s in ALL12}


def lgmean(seeds, use_mate4=False):
    if use_mate4:
        n_all = 4 + len([s for s in seeds if s in NEW_SEEDS])
        acc = 4.0 * L_cat_mate + sum(L_ours[s] for s in seeds if s in NEW_SEEDS)
        return acc / n_all
    return np.mean([L_ours[s] for s in seeds], axis=0)


# ═══════════════ 2. 눈금 게이트 ═══════════════
t = time.time()
arms = {}
for k, Lx in [("p_lgb", L_lgb), ("p_cat", L_cat_mate), ("p_leaf", L_leaf),
              ("p_q", L_q), ("p_v6", lg(V["p_v6_3arm"])), ("p_ebm", L_ebm),
              ("AA81", L_aa), ("T70", L_t70)]:
    b, s_ref, s_loc, c = score_z(Lx)
    arms[k] = {"brier": b, "score_REF": s_ref, "score_ref_local": s_loc,
               "recenter_c": c, "exp070_4dp": GATE[k],
               "abs_diff_vs_exp070": abs(s_ref - GATE[k]),
               "match_4dp": bool(abs(s_ref - GATE[k]) < 5e-4)}
gate_arms = all(v["match_4dp"] for v in arms.values())


def inner_z(L_v6, w_v6=W_V6_IN):
    return ((1.0 - W_EBM) * (w_v6 * L_v6 + (1.0 - w_v6) * L_q)
            + W_EBM * L_ebm + BETA_INS * off)


b, s_ref, s_loc, c = score_z(inner_z(lg(V["p_v6_3arm"]), 0.5))
gate_start = bool(abs(s_ref - GATE_START) < 1e-5)
M["scale_gates"] = {
    "single_arms": arms, "single_arms_PASS": bool(gate_arms),
    "start_qualification": {"score_REF": s_ref, "exp070": GATE_START,
                            "abs_diff": abs(s_ref - GATE_START),
                            "PASS": gate_start}}
z_full6 = W_OUT * inner_z(lg(V["p_v6_3arm"]), 0.5) + C_AA * L_aa + C_T70 * L_t70
b6, s6, l6, c6 = score_z(z_full6)
M["scale_gates"]["deployment_full6_no_quadratic"] = {
    "score_REF": s6, "exp070": GATE_FULL6, "abs_diff": abs(s6 - GATE_FULL6),
    "PASS": bool(abs(s6 - GATE_FULL6) < 1e-5)}
assert gate_arms and gate_start, "🔴 눈금 게이트 실패 — 자가 다르다.  중단."
print(f"[눈금] 단일팔 4자리 일치 {gate_arms} · 시작자격 {s_ref:.9f} "
      f"· 6상수 {s6:.9f}  {time.time()-t:.0f}s", flush=True)


# ═══════════════ 3. v6 조립 항등 게이트 ═══════════════
def v6_from_cat(L_cat):
    """NN 자리를 비운 3종 재정규화 — 팀원 `_renorm` 판과 같은 산술."""
    return (W_LGB * L_lgb + W_CAT * L_cat + W_LEAF * L_leaf) / (
        W_LGB + W_CAT + W_LEAF)


z_v6_mate = v6_from_cat(L_cat_mate)
p_v6_mine = expit(z_v6_mate)
d_id = float(np.abs(p_v6_mine - V["p_v6_3arm"]).max())
M["v6_assembly_identity"] = {
    "max_abs_prob_diff_vs_mate_renorm": d_id,
    "renorm_denominator": W_LGB + W_CAT + W_LEAF,
    "PASS": bool(d_id < 1e-12),
    "해설": ("팀원 cat4 를 넣어 조립한 우리 v6 가 팀원의 "
             "p_v6_3arm_renorm_oot2024.npy 와 같은 벡터인가.  통과하면 "
             "조립 산술·가중·재정규화가 전부 같다는 뜻이다.")}
assert d_id < 1e-12, f"🔴 v6 조립이 팀원 판과 다르다 {d_id}"
n_clip = int((np.abs(z_v6_mate) > -np.log(EPS / (1 - EPS))).sum())
M["v6_assembly_identity"]["logit_clip_rows"] = n_clip
print(f"[조립] v6 항등 max|Δp| {d_id:.3e}  클립행 {n_clip}", flush=True)

# ═══════════════ 4. G0d 동일성 (배포 마스크 재적합 vs 배포 아티팩트) ═══════════════
G0D = {}
dep_fit = f"{C}/_cat_deploy_s42_t6.cbm"
if os.path.exists(dep_fit):
    mine = CatBoostClassifier(); mine.load_model(dep_fit)
    dep = CatBoostClassifier(); dep.load_model(f"{MODEL}/cat_42.cbm")
    lm = np.asarray(mine.get_leaf_values(), np.float64)
    ld = np.asarray(dep.get_leaf_values(), np.float64)
    Lmine = lg(np.load(f"{C}/_cat_deploy_s42_t6.npy"))
    Ldep = lg(np.load(f"{C}/dep_pred2024_seed42.npy"))
    band = json.load(open(f"{E}/out/stage0.json"))["G0e_deploy_seed_pair_band"]
    lo, hi = band["rho_logit_pred_band"]
    rho_pred = float(np.corrcoef(Lmine, Ldep)[0, 1])
    cross = {str(s): float(np.corrcoef(
        Lmine, lg(np.load(f"{C}/dep_pred2024_seed{s}.npy")))[0, 1])
        for s in DEPLOY_SEEDS[1:]}
    G0D = {
        "our_refit": "deploy 마스크(<=2024, 1,369,784행) · seed 42 · threads 6",
        "tree_count_ours": int(mine.tree_count_),
        "tree_count_deploy": int(dep.tree_count_),
        "leaf_array_size_ours": int(lm.size), "leaf_array_size_deploy": int(ld.size),
        "rho_leaf": (float(np.corrcoef(lm, ld)[0, 1]) if lm.size == ld.size else None),
        "rho_logit_pred_2024": rho_pred,
        "rms_logit_diff": float(np.sqrt(np.mean((Lmine - Ldep) ** 2))),
        "max_abs_prob_diff": float(np.abs(expit(Lmine) - expit(Ldep)).max()),
        "deploy_seed_pair_band": [lo, hi],
        "rho_vs_other_deploy_seeds": cross,
        "판정": ("같은 시드 재적합이 다른 시드쌍 대역보다 위면 같은 계열이고 "
                 "시드도 재현된다.  대역 안이면 계열은 같으나 시드 재현은 "
                 "안 된 것 — S17(재학습 바이트 재현 불가) 이 예고한 결과."),
        "above_band": bool(rho_pred > hi),
        "within_or_above_band": bool(rho_pred >= lo)}
    print(f"[G0d] ρ(우리재적합42, 배포42) = {rho_pred:.8f}  "
          f"대역 {lo:.8f}~{hi:.8f}  대역위={G0D['above_band']}", flush=True)
M["G0d_identity"] = G0D

# ═══════════════ 5. 우리 4시드 재현 vs 팀원 납품 ═══════════════
L_ours4 = lgmean(DEPLOY_SEEDS)
p_ours4 = expit(L_ours4)
b_m, s_m, l_m, _ = score_z(L_cat_mate)
b_o, s_o, l_o, _ = score_z(L_ours4)
per_seed = {}
for s in DEPLOY_SEEDS:
    bb, ss, ll, _ = score_z(L_ours[s])
    per_seed[str(s)] = {"score_REF": ss, "score_ref_local": ll,
                        "rho_logit_vs_mate_mean": float(
                            np.corrcoef(L_ours[s], L_cat_mate)[0, 1])}
M["cat4_reproduction"] = {
    "rho_logit": float(np.corrcoef(L_ours4, L_cat_mate)[0, 1]),
    "rms_logit_diff": float(np.sqrt(np.mean((L_ours4 - L_cat_mate) ** 2))),
    "max_abs_prob_diff": float(np.abs(p_ours4 - V["p_cat"]).max()),
    "mean_abs_prob_diff": float(np.abs(p_ours4 - V["p_cat"]).mean()),
    "score_REF_mate": s_m, "score_REF_ours": s_o,
    "score_ref_local_mate": l_m, "score_ref_local_ours": l_o,
    "delta_pts_ours_minus_mate": s_o - s_m,
    "per_seed_ours": per_seed,
    "🔴변인": ("팀원 thread_count=2 · 스냅샷 저장 사용 · 자기 feat6.parquet. "
               "우리 thread_count=6 · 스냅샷 없음 · exp071/cache/frame071.parquet. "
               "따라서 바이트 동일은 기대 대상이 아니다(S17)."),
}
print(f"[재현] cat4 ρ {M['cat4_reproduction']['rho_logit']:.8f}  "
      f"우리 {s_o:.4f} vs 팀원 {s_m:.4f} ({s_o - s_m:+.4f}점)", flush=True)

# ═══════════════ 6. 시드 곡선 + 정직면 Δ ═══════════════


def blend_score(L_cat):
    z = W_OUT * inner_z(v6_from_cat(L_cat)) + C_AA * L_aa + C_T70 * L_t70
    f = ALPHA * z + BETA_SKEW * z * z
    return score_z(f)


def arm_score(L_cat):
    return score_z(L_cat)


rng = np.random.default_rng(77)
curve = {}
for n in range(1, 13):
    combos = list(itertools.combinations(range(12), n))
    if len(combos) > 40:
        idx = rng.choice(len(combos), 40, replace=False)
        combos = [combos[i] for i in idx]
    sc, ar = [], []
    for cmb in combos:
        seeds = [ALL12[i] for i in cmb]
        Lc = lgmean(seeds)
        sc.append(blend_score(Lc)[1])
        ar.append(arm_score(Lc)[1])
    curve[n] = {"n_subsets": len(combos),
                "blend_score_mean": float(np.mean(sc)),
                "blend_score_sd": float(np.std(sc, ddof=1)) if len(sc) > 1 else 0.0,
                "blend_score_min": float(np.min(sc)), "blend_score_max": float(np.max(sc)),
                "arm_score_mean": float(np.mean(ar)),
                "arm_score_sd": float(np.std(ar, ddof=1)) if len(ar) > 1 else 0.0}
    print(f"  [곡선] n={n:2d}  블렌드 {curve[n]['blend_score_mean']:9.4f} "
          f"(±{curve[n]['blend_score_sd']:.4f}, {curve[n]['n_subsets']}조합)  "
          f"팔 {curve[n]['arm_score_mean']:9.4f}", flush=True)

# 1/n 모형 적합 (n>=2 평균 위에서 최소제곱)
ns = np.array([n for n in range(2, 13)], float)
for key in ("blend", "arm"):
    ys = np.array([curve[int(n)][f"{key}_score_mean"] for n in ns])
    A = np.column_stack([np.ones_like(ns), -1.0 / ns])
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ys - pred) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    curve[f"model_{key}"] = {
        "form": "score(n) = a - b/n", "a": float(coef[0]), "b": float(coef[1]),
        "R2": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "resid_max_abs": float(np.max(np.abs(ys - pred))),
        "pred_4": float(coef[0] - coef[1] / 4), "pred_12": float(coef[0] - coef[1] / 12),
        "pred_delta_4_to_12": float(coef[1] * (1 / 4 - 1 / 12)),
        "pred_infinity": float(coef[0])}
M["seed_curve"] = curve

# 실제 채택 경로: 배포 4 -> +새 시드 순차
seq = {}
for k in range(0, 9):
    seeds = DEPLOY_SEEDS + NEW_SEEDS[:k]
    Lc = lgmean(seeds)
    b_, s_, l_, c_ = blend_score(Lc)
    ab, asr, al, _ = arm_score(Lc)
    seq[4 + k] = {"seeds": seeds, "blend_score_REF": s_, "blend_brier": b_,
                  "blend_score_ref_local": l_, "arm_score_REF": asr}
M["sequential_ours"] = seq

# ═══════════════ 7. 주지표 ═══════════════
L4o, L12o = lgmean(DEPLOY_SEEDS), lgmean(ALL12)
b4o, s4o, l4o, c4o = blend_score(L4o)
b12o, s12o, l12o, c12o = blend_score(L12o)
a4o = arm_score(L4o)[1]
a12o = arm_score(L12o)[1]

L4m = L_cat_mate
L12m = lgmean(NEW_SEEDS, use_mate4=True)
b4m, s4m, l4m, _ = blend_score(L4m)
b12m, s12m, l12m, _ = blend_score(L12m)
a4m, a12m = arm_score(L4m)[1], arm_score(L12m)[1]

PRI = s12o - s4o
SEC = s12m - s4m
M["primary"] = {
    "정의": "Δ = score(v6_12) − score(v6_4), 정직면 재센터링, REF 눈금",
    "우리12_vs_우리4": {
        "brier_v6_4": b4o, "brier_v6_12": b12o,
        "score_v6_4": s4o, "score_v6_12": s12o, "delta_pts": PRI,
        "score_ref_local_v6_4": l4o, "score_ref_local_v6_12": l12o,
        "arm_score_cat4": a4o, "arm_score_cat12": a12o,
        "arm_delta_pts": a12o - a4o,
        "note": "한 코드 경로 — cat 시드 수만 다르다.  주지표."},
    "팀원4+우리8_vs_팀원4": {
        "score_v6_4": s4m, "score_v6_12": s12m, "delta_pts": SEC,
        "arm_score_cat4": a4m, "arm_score_cat12": a12m,
        "arm_delta_pts": a12m - a4m,
        "note": "배포가 실제로 밟을 경로(배포 4시드는 그대로 두고 8개를 더한다)."},
}


def blend_p(L_cat, c):
    z = W_OUT * inner_z(v6_from_cat(L_cat)) + C_AA * L_aa + C_T70 * L_t70
    return expit(ALPHA * z + BETA_SKEW * z * z + c)


_d = np.abs(blend_p(L12o, c12o) - blend_p(L4o, c4o))
M["primary"]["pred_diff_v6_12_vs_v6_4"] = {
    "max_abs": float(_d.max()), "mean_abs": float(_d.mean()),
    "🔴": "S9 — 이것은 후보 간 예측 차이지 보정 이동폭이 아니고, 점수로 환산되지 않는다."}
M["transfer"] = {
    "exp_072_measured_range": list(XFER),
    "🔴": "S2 — 전이비는 축마다 다르고 부호까지 갈렸다(−8.27~2.03). 범위로만 쓴다.",
    "primary_delta_pts": PRI,
    "lb_projection_range": [PRI * XFER[0], PRI * XFER[1]],
    "secondary_delta_pts": SEC,
    "lb_projection_range_secondary": [SEC * XFER[0], SEC * XFER[1]]}
M["preregistered_verdict"] = {
    "rule": "정직면 Δ >= +1.0 -> 배포 시드 추가 학습해 후보로. 미만이면 축을 닫는다.",
    "threshold": PREREG_THRESHOLD, "primary_delta": PRI, "secondary_delta": SEC,
    "PASS_primary": bool(PRI >= PREREG_THRESHOLD),
    "PASS_secondary": bool(SEC >= PREREG_THRESHOLD)}
print(f"\n=== 주지표 ===  우리12−우리4 Δ {PRI:+.5f}점 · "
      f"팀원4+우리8−팀원4 Δ {SEC:+.5f}점 · 문턱 {PREREG_THRESHOLD}", flush=True)
print(f"    팔 눈금 Δ  우리 {a12o - a4o:+.5f} · 혼합 {a12m - a4m:+.5f}", flush=True)

# ═══════════════ 8. 산출 벡터 ═══════════════
np.save(f"{E}/out/p_cat4_oot2024.npy", expit(L4o))
np.save(f"{E}/out/p_cat12_oot2024.npy", expit(L12o))
np.save(f"{E}/out/p_cat12_mate4plus8_oot2024.npy", expit(L12m))
np.save(f"{E}/out/p_v6_12_oot2024.npy", expit(v6_from_cat(L12o)))
np.save(f"{E}/out/p_v6_4_oot2024.npy", expit(v6_from_cat(L4o)))
M["outputs"] = {k: sha_file(f"{E}/out/{k}") for k in
                ["p_cat4_oot2024.npy", "p_cat12_oot2024.npy",
                 "p_cat12_mate4plus8_oot2024.npy",
                 "p_v6_12_oot2024.npy", "p_v6_4_oot2024.npy"]}
M["wall_min"] = (time.time() - T0) / 60.0
json.dump(M, open(f"{E}/out/stage2.json", "w"), ensure_ascii=False, indent=1)
print(f"\n=== stage2 완료 {M['wall_min']:.2f}분 ===", flush=True)

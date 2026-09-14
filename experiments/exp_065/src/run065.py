"""exp_065 — exp_064 NN 을 여섯째 팔로 넣으면 블렌드가 좋아지는가.  학습 0건.

두 면에서 잰다:
  면 A(정직면)  z_A = w*lg(NN_oot) + (1-w)*lg(AA_oot)   — 양쪽 다 out-of-time
  면 B(현행면)  현행 5팔 블렌드(z) 에 w*lg(NN) 을 더하고 alpha/beta_skew 재적합 -> d_adj 재못박기
                🔴 동료 팔이 tr2024 표본 내라 이 값은 **하한**이다.

🔴 항등 앵커 먼저: w=0 이 현행 블렌드를 바이트 동일하게 재현해야 한다.
🔴 S14: arms/* 는 이름이 아니라 값의 범위로 확인한다 (p_ebm 은 이미 로짓).
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp065/out"
os.makedirs(OUT, exist_ok=True)
EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
NN_SEEDS = (42, 777, 2024, 101)
AA_SEEDS = (42, 7, 2024)
T_START = time.time()
WALL = {}


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


M = {"exp_id": "exp_065", "type": "블렌드 위 팔 측정 (학습 0건)"}
t = time.time()

# ══════════════════════════════════════════════════════════════════════
# 0. 정렬 — arms · train.csv · frame99 이 같은 행 순서인가
# ══════════════════════════════════════════════════════════════════════
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success", "pitcher_id"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
N = len(tr)
assert N == 253507, N
y = tr.control_success.to_numpy(np.float64)
rid = tr.row_id.to_numpy().astype(str)
assert np.array_equal(ids, rid), "arms/ids.npy 가 row_id 정렬과 불일치"

f99 = pd.read_parquet(f"{H}/exp060_nn/cache/frame99.parquet",
                      columns=["season", "row_id", "control_success"])
v99 = f99[f99.season == 2024]
assert len(v99) == N
f99_same_order = bool(np.array_equal(v99.row_id.to_numpy().astype(str), rid))
if f99_same_order:
    perm = None
else:
    order = pd.Series(np.arange(N), index=v99.row_id.to_numpy().astype(str))
    perm = order.reindex(rid).to_numpy()
    assert not np.isnan(perm).any(), "frame99 에 없는 row_id"
    perm = perm.astype(int)
yv = v99.control_success.to_numpy(np.float64)
assert np.array_equal(yv if perm is None else yv[perm], y), "라벨 불일치"
M["alignment"] = {"n_rows": N,
                  "arms_ids_eq_rowid_sorted": True,
                  "frame99_2024_same_order_as_rowid_sorted": f99_same_order,
                  "permutation_applied_to_NN_preds": perm is not None,
                  "y_mean": float(y.mean()),
                  "y_sha256": sha(y)}


def nnload(path):
    a = np.load(path).astype(np.float64)
    assert a.shape == (N,), (path, a.shape)
    return a if perm is None else a[perm]


# ══════════════════════════════════════════════════════════════════════
# 0b. S14 — arms/* 를 값의 범위로 확인한다 (이름으로 판단 금지)
# ══════════════════════════════════════════════════════════════════════
X = np.load(f"{H}/arms/X_arms.npy")
coef_now = np.load(f"{H}/arms/coef_now.npy")
assert X.shape == (5, N)
XN = ["X0_lg_p_v6", "X1_lg_p_q", "X2_p_ebm", "X3_lg_p_AA_dep", "X4_off"]
M["S14_range_check"] = {}
for i, nm in enumerate(XN):
    lo, hi = float(X[i].min()), float(X[i].max())
    inside01 = bool(lo >= 0.0 and hi <= 1.0)
    M["S14_range_check"][nm] = {
        "min": lo, "max": hi, "mean": float(X[i].mean()),
        "inside_0_1": inside01,
        "verdict": "확률로 보임 → lg() 필요" if inside01 else "이미 로짓 → lg() 씌우지 말 것"}
assert not M["S14_range_check"]["X2_p_ebm"]["inside_0_1"], \
    "p_ebm 이 [0,1] 안이다 — S14 전제가 깨졌다"
M["S14_range_check"]["coef_now"] = coef_now.tolist()
WALL["s0_align_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 1. 부모 상수 · z 재계산 · 항등 앵커 G1/G2
# ══════════════════════════════════════════════════════════════════════
t = time.time()
bl = json.load(open(f"{H}/build_BG/pkg_BG/blend.json", encoding="utf-8"))
A0, B0, D0, T0 = bl["alpha"], bl["beta_skew"], bl["d_adj"], bl["target_rate"]
C = {"aa": 0.25244, "S2": 0.0, "S1": 0.0, "T70": 0.04756}
p_v19 = np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy")
p_aa = np.load(f"{H}/build_AY/work/pcal_aa.npy")
p_S2 = np.load(f"{H}/build_AY/work/pcal_S2.npy")
p_S1 = np.load(f"{H}/build_AY/work/pcal_S1.npy")
p_T70 = np.load(f"{H}/build_AY/work/pcal_T70.npy")
z = (0.7 * lg(p_v19) + C["aa"] * lg(p_aa) + C["S2"] * lg(p_S2)
     + C["S1"] * lg(p_S1) + C["T70"] * lg(p_T70))

z_ref = np.load(f"{H}/audit_BH/z_tr.npy")
r0_ref = np.load(f"{H}/audit_BH/r0_tr.npy")
G1 = bool(np.array_equal(z, z_ref))
# 🔴 결합 순서 두 가지가 1 ulp 다르다 (assemble_BL3.py 58행 vs 60행).
#    저장된 audit_BH/r0_tr.npy 는 B*(z*z) 판, 출하 script.py 는 B*z*z = (B*z)*z 판.
pA_lin = expit(A0 * z + B0 * (z * z) + D0)      # 저장 벡터와 같은 순서
pA = expit(A0 * z + B0 * z * z + D0)            # 출하 산술과 같은 순서 (하류 전부 이것)
G2 = bool(np.array_equal(y - pA_lin, r0_ref))
ASSOC = {"max_abs_p_diff_between_orders": float(np.max(np.abs(pA - pA_lin))),
         "n_rows_differing": int(np.sum(pA != pA_lin)),
         "note": ("B*(z*z) 와 (B*z)*z 는 부동소수 결합 순서가 달라 최대 1 ulp 어긋난다. "
                  "assemble_BL3.py 도 저장 벡터 대조에는 앞 판, 출하 산술에는 뒤 판을 쓴다. "
                  "이 실험의 하류 계산은 전부 출하 순서다.")}


def anc_brentq(f, target=None):
    tg = T0 if target is None else target
    return brentq(lambda d: expit(f + d).mean() - tg, -5.0, 5.0, xtol=1e-15)


def anc(f, target=None):
    """Newton. 🔴 순수 함수여야 한다 — 시작점·반복조건이 입력에만 의존한다."""
    tg = T0 if target is None else target
    d = 0.0
    for _ in range(80):
        p = expit(f + d)
        h = float(np.mean(p * (1.0 - p)))
        step = (float(p.mean()) - tg) / h
        d -= step
        if abs(step) < 1e-16:
            break
    return d


f0 = A0 * z + B0 * z * z
d_bq = anc_brentq(f0)
d_nw = anc(f0)
SHIFT = D0 - d_nw
b_par = float(np.mean((pA - y) ** 2))
M["anchor"] = {
    "parent": "BG (build_BG/pkg_BG/blend.json)",
    "alpha": A0, "beta_skew": B0, "d_adj": D0, "target_rate": T0,
    "arm_coefs": {"partner_v19": 0.7, **C},
    "G1_z_byte_identical_to_audit_BH_z_tr": G1,
    "G2_r0_byte_identical_to_audit_BH_r0_tr": G2,
    "association_order": ASSOC,
    "d_brentq": d_bq, "d_newton": d_nw,
    "newton_minus_brentq": float(d_nw - d_bq),
    "SHIFT": SHIFT,
    "parent_brier": b_par, "parent_score": conv(b_par),
    "parent_mean": float(pA.mean()),
    "z_sha256": sha(z), "pA_sha256": sha(pA)}
assert abs(d_nw - d_bq) < 1e-13, "Newton d-solve 가 brentq 와 어긋난다"
assert G1 and G2, "🔴 항등 앵커 G1/G2 실패 — 나머지 측정이 무의미하다. 중단."


def build(zz, a, b):
    """출하 산술: f = a*zz + b*zz^2, d = anc(f) + SHIFT."""
    f = a * zz + b * zz * zz
    dd = anc(f) + SHIFT
    return expit(f + dd), dd


def brier_of(zz, a, b):
    return float(np.mean((build(zz, a, b)[0] - y) ** 2))


WALL["s1_anchor_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 2. 재료 — NN(out-of-time) · AA(out-of-time) · 항등 앵커 G3
# ══════════════════════════════════════════════════════════════════════
t = time.time()
NN = nnload(f"{H}/exp064/out/preds/A99ship_seedavg.npy")
NNS = {s: nnload(f"{H}/exp064/out/preds/A99ship_s{s}.npy") for s in NN_SEEDS}
AA = np.mean([np.load(f"{H}/experiments/exp_048/pred/"
                      f"pred_A1_u2024_seed{s}_at1400.npy").astype(np.float64)
              for s in AA_SEEDS], axis=0)
assert AA.shape == (N,)
lgNN = lg(NN)
lgAA = lg(AA)

# 🔴 항등 앵커 G3 — w=0 을 NN 팔 경로로 통과시켜 부모를 바이트 동일 재현
p_chk, d_chk = build(z + 0.0 * lgNN, A0, B0)
G3 = bool(np.array_equal(p_chk, pA))
M["anchor"]["G3_builder_w0_byte_identical_to_parent"] = G3
M["anchor"]["G3_d_returned"] = d_chk
M["anchor"]["G3_d_minus_D0"] = float(d_chk - D0)
M["anchor"]["G3_max_abs_p_diff"] = float(np.max(np.abs(p_chk - pA)))
M["anchor"]["G3_p_sha256"] = sha(p_chk)
assert G3, ("🔴 항등 앵커 G3 실패 — w=0 빌더가 부모를 재현하지 못한다. 중단. "
            f"max|dp| = {np.max(np.abs(p_chk - pA)):.3e}")
print(f"[앵커] G1 {G1} G2 {G2} G3 {G3}  부모 {conv(b_par):.5f}점")

lgNN_from_seeds = np.mean([lg(NNS[s]) for s in NN_SEEDS], axis=0)
b_nn = float(np.mean((NN - y) ** 2))
b_aa = float(np.mean((AA - y) ** 2))
M["materials"] = {
    "NN": {"src": "exp064/out/preds/A99ship_seedavg.npy (4시드 로짓평균, out-of-time)",
           "brier": b_nn, "score": conv(b_nn),
           "exp064_recorded_brier": 0.24756767168405058,
           "recompute_delta": b_nn - 0.24756767168405058,
           "seedavg_is_logit_mean_max_abs_diff":
               float(np.max(np.abs(lgNN - lgNN_from_seeds)))},
    "AA": {"src": f"exp_048 A1@1400 시드{AA_SEEDS} 확률평균 (out-of-time)",
           "brier": b_aa, "score": conv(b_aa),
           "exp063_recorded_meanofbriers": 0.24787535671986385,
           "note": "exp_063 의 0.2478754 는 mean-of-briers. 여기 값은 3시드 평균예측의 Brier"},
    "per_seed_NN": {str(s): {"brier": float(np.mean((NNS[s] - y) ** 2)),
                             "score": conv(float(np.mean((NNS[s] - y) ** 2)))}
                    for s in NN_SEEDS},
    "corr_logit_NN_AA": float(np.corrcoef(lgNN, lgAA)[0, 1]),
    "corr_logit_NN_zblend": float(np.corrcoef(lgNN, z)[0, 1]),
    "corr_logit_AA_zblend": float(np.corrcoef(lgAA, z)[0, 1])}
assert abs(b_nn - 0.24756767168405058) < 1e-12, "NN 벡터가 exp_064 기록과 다르다"
print(f"[재료] NN {conv(b_nn):.5f}점  AA {conv(b_aa):.5f}점  "
      f"corr(lgNN,lgAA) {M['materials']['corr_logit_NN_AA']:.6f}")
WALL["s2_materials_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 3. 면 A (정직면) — 둘 다 out-of-time
# ══════════════════════════════════════════════════════════════════════
t = time.time()


def faceA(w, ln):
    return expit(w * ln + (1.0 - w) * lgAA)


def gridsolve(fn, hi, step=0.005):
    ws = np.round(np.arange(0.0, hi + 1e-12, step), 6)
    bs = np.array([fn(float(w)) for w in ws])
    return ws, bs, int(np.argmin(bs))


A_RES = {"form": "p = expit(w*lg(NN) + (1-w)*lg(AA)) — d 없음(둘 다 이미 보정된 확률)",
         "baseline": "AA 단독 (w=0)", "grids": {}}
ws1, bs1, j1 = gridsolve(lambda w: float(np.mean((faceA(w, lgNN) - y) ** 2)), 0.40)
cap_hit = (j1 == len(ws1) - 1)
A_RES["grids"]["0_to_0.40"] = {
    "argmin_w": float(ws1[j1]), "brier": float(bs1[j1]), "score": conv(float(bs1[j1])),
    "cap_touched": bool(cap_hit),
    "verdict": "🔴 무효 — argmin 이 격자 끝(0.40)에 닿았다. 범위 확대" if cap_hit else "유효"}
if cap_hit:
    ws1, bs1, j1 = gridsolve(lambda w: float(np.mean((faceA(w, lgNN) - y) ** 2)), 1.00)
    A_RES["grids"]["0_to_1.00"] = {
        "argmin_w": float(ws1[j1]), "brier": float(bs1[j1]),
        "score": conv(float(bs1[j1])),
        "cap_touched": bool(j1 == len(ws1) - 1),
        "verdict": "유효" if j1 < len(ws1) - 1 else "🔴 무효 — w=1 에 닿음"}
wA, bA = float(ws1[j1]), float(bs1[j1])
A_RES.update({"optimal_w_NN": wA, "brier_at_opt": bA, "score_at_opt": conv(bA),
              "gain_vs_AA_alone_pts": K * (b_aa - bA),
              "gain_vs_NN_alone_pts": K * (b_nn - bA)})
A_RES["curve"] = {f"{float(w):.3f}": {"brier": float(b), "score": conv(float(b))}
                  for w, b in zip(ws1, bs1)
                  if abs(round(float(w) * 1000) % 50) == 0 or abs(float(w) - wA) < 1e-9}

# d 재못박기 변이 (같은 상수를 baseline·candidate 양쪽에 똑같이 — NN 을 편들지 않는다)
b_aa_d = float(np.mean((expit(lgAA + anc(lgAA)) - y) ** 2))


def faceA_d(w, ln):
    zz = w * ln + (1.0 - w) * lgAA
    return expit(zz + anc(zz))


ws2, bs2, j2 = gridsolve(lambda w: float(np.mean((faceA_d(w, lgNN) - y) ** 2)), 1.00)
A_RES["variant_with_d_repin_to_BG_target"] = {
    "target_rate": T0,
    "AA_alone_brier": b_aa_d, "AA_alone_score": conv(b_aa_d),
    "argmin_w": float(ws2[j2]), "brier": float(bs2[j2]), "score": conv(float(bs2[j2])),
    "gain_vs_AA_alone_pts": K * (b_aa_d - float(bs2[j2])),
    "cap_touched": bool(j2 == len(ws2) - 1)}

gs, A_RES["per_seed_at_w_star"] = [], {}
for s in NN_SEEDS:
    bsd = float(np.mean((faceA(wA, lg(NNS[s])) - y) ** 2))
    gs.append(K * (b_aa - bsd))
    A_RES["per_seed_at_w_star"][str(s)] = {"brier": bsd, "score": conv(bsd),
                                           "gain_vs_AA_pts": K * (b_aa - bsd)}
gs = np.array(gs)
sd = float(gs.std(ddof=1))
A_RES["stability"] = {"gains_pts": gs.tolist(), "mean": float(gs.mean()),
                      "std_ddof1": sd, "mean_over_std": float(gs.mean() / sd),
                      "sign_all_positive": bool((gs > 0).all()),
                      "sign_count": f"{int((gs > 0).sum())}/4",
                      "pass_3sigma": bool(gs.mean() >= 3.0 * sd)}
A_PASS = bool(wA > 0.0 and A_RES["gain_vs_AA_alone_pts"] >= 3.0)
A_RES["gate"] = {"rule": "최적 w_NN > 0 그리고 AA 단독 대비 이득 >= +3점", "pass": A_PASS}
M["face_A_honest"] = A_RES
WALL["s3_faceA_min"] = (time.time() - t) / 60.0
print(f"[면A] w* {wA}  {conv(bA):.5f}점  vs AA {A_RES['gain_vs_AA_alone_pts']:+.5f}  "
      f"vs NN단독 {A_RES['gain_vs_NN_alone_pts']:+.5f}  게이트 {A_PASS}")
print(f"      개별시드 이득 {np.round(gs, 4).tolist()}  평균/std {gs.mean() / sd:.3f}")
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

if not A_PASS:
    M["verdict"] = {"face_A_pass": False, "adopt": False,
                    "stop": "면 A 정지점 발동 — 축을 닫고 보고. 면 B 로 가지 않는다."}
    WALL["total_min"] = (time.time() - T_START) / 60.0
    M["wall_clock"] = WALL
    json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
    print("면 A 정지점 발동 — 종료")
    raise SystemExit(0)

# ══════════════════════════════════════════════════════════════════════
# 4. 면 B (현행면, 하한) — 현행 5팔 블렌드에 NN 을 여섯째로
# ══════════════════════════════════════════════════════════════════════
t = time.time()


def refit_ab(zz, x0=(A0, B0)):
    r = minimize(lambda q: brier_of(zz, float(q[0]), float(q[1])),
                 np.array(x0, np.float64), method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-16, "maxiter": 4000,
                          "maxfev": 4000})
    return float(r.x[0]), float(r.x[1]), float(r.fun), int(r.nfev)


B_RES = {"note": ("🔴 이 값은 하한이다 — 동료 팔 5개가 tr2024 표본 내이고 NN 만 "
                  "out-of-time 이라 NN 이 부당하게 불리하다."),
         "parent_brier": b_par, "parent_score": conv(b_par)}

# ── ①-a  5팔 비율 고정 · w 만 추가 · alpha/beta 는 BG 값 고정 · d 재못박기 ──
ws, bs, j = gridsolve(lambda w: brier_of(z + w * lgNN, A0, B0), 0.40)
cap_a = (j == len(ws) - 1)
first_hi = 0.40
if cap_a:
    ws, bs, j = gridsolve(lambda w: brier_of(z + w * lgNN, A0, B0), 1.20)
B_RES["form1a_fixed_alpha_beta"] = {
    "desc": "z' = z + w*lg(NN); alpha/beta_skew 는 BG 값 고정, d_adj 만 재못박기",
    "grid_first_hi": first_hi, "grid_cap_0.40_touched_first": bool(cap_a),
    "grid_hi_used": float(ws[-1]),
    "argmin_w": float(ws[j]), "brier": float(bs[j]), "score": conv(float(bs[j])),
    "cap_touched": bool(j == len(ws) - 1),
    "gain_vs_parent_pts": K * (b_par - float(bs[j])),
    "curve": {f"{float(w):.3f}": conv(float(b)) for w, b in zip(ws, bs)
              if abs(round(float(w) * 1000) % 25) == 0 or abs(float(w) - float(ws[j])) < 1e-9}}
print(f"  [면B ①-a] w* {float(ws[j]):.3f}  {conv(float(bs[j])):.5f}점  "
      f"{K * (b_par - float(bs[j])):+.5f}")

# ── ①-b  w 추가 + alpha/beta 재적합 (config 가 지정한 확립된 순서) ──
a_b0, b_b0, br_b0, nf0 = refit_ab(z)
B_RES["form1b_baseline_w0_refit"] = {
    "alpha": a_b0, "beta_skew": b_b0, "brier": br_b0, "score": conv(br_b0),
    "gain_vs_parent_pts": K * (b_par - br_b0), "nfev": nf0,
    "note": "재적합만으로 생기는 몫 — NN 귀속분에서 빼야 한다"}
print(f"  [면B ①-b w=0 기준선] a {a_b0:.6f} b {b_b0:+.6f}  {conv(br_b0):.5f}점  "
      f"재적합몫 {K * (b_par - br_b0):+.5f}")

wgrid = [0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13, 0.16, 0.20,
         0.25, 0.30, 0.35, 0.40]
rows = {}
for w in wgrid:
    a_, b_, br_, nf_ = refit_ab(z + w * lgNN)
    rows[f"{w:.3f}"] = {"alpha": a_, "beta_skew": b_, "brier": br_, "score": conv(br_),
                        "nfev": nf_, "gain_vs_refit_baseline_pts": K * (br_b0 - br_),
                        "gain_vs_parent_pts": K * (b_par - br_)}
    print(f"  [면B ①-b] w {w:.3f}  a {a_:.6f} b {b_:+.6f}  {conv(br_):.5f}점  "
          f"NN귀속 {K * (br_b0 - br_):+.5f}")
best_w = min(rows, key=lambda k: rows[k]["brier"])
cap_1b = abs(float(best_w) - 0.40) < 1e-12
if cap_1b:
    for w in (0.50, 0.60, 0.80, 1.00, 1.20):
        a_, b_, br_, nf_ = refit_ab(z + w * lgNN)
        rows[f"{w:.3f}"] = {"alpha": a_, "beta_skew": b_, "brier": br_,
                            "score": conv(br_), "nfev": nf_,
                            "gain_vs_refit_baseline_pts": K * (br_b0 - br_),
                            "gain_vs_parent_pts": K * (b_par - br_)}
        print(f"  [면B ①-b 확장] w {w:.3f}  {conv(br_):.5f}점  "
              f"NN귀속 {K * (br_b0 - br_):+.5f}")
    best_w = min(rows, key=lambda k: rows[k]["brier"])
wB = float(best_w)
a_s, b_s = rows[best_w]["alpha"], rows[best_w]["beta_skew"]
zbest = z + wB * lgNN
zstar = (-a_s / (2 * b_s)) if b_s != 0 else float("inf")
B_RES["form1b_refit_alpha_beta"] = {
    "desc": ("z' = z + w*lg(NN) 뒤 alpha/beta_skew 재적합 -> d_adj 재못박기 "
             "(config 지정 순서). 기준선은 w=0 에서 같은 재적합"),
    "grid": rows, "grid_cap_0.40_touched_first": bool(cap_1b), "best_w": wB,
    "brier": rows[best_w]["brier"], "score": rows[best_w]["score"],
    "alpha": a_s, "beta_skew": b_s,
    "NN_attributable_gain_pts": rows[best_w]["gain_vs_refit_baseline_pts"],
    "total_gain_vs_parent_pts": rows[best_w]["gain_vs_parent_pts"],
    "monotonicity": {"z_star": zstar, "z_max": float(zbest.max()),
                     "z_min": float(zbest.min()),
                     "margin_x": (float(zstar / max(zbest.max(), 1e-12))
                                  if np.isfinite(zstar) else None),
                     "safe": bool((not np.isfinite(zstar))
                                  or zstar > zbest.max() * 3
                                  or zstar < zbest.min() * 3)}}

# ── ②  전체 재적합 — 팔 계수 전부 자유 ────────────────────────────────
dec = 0.7 * lg(p_v19) - (0.385 * X[0] + 0.385 * X[1] - 0.07 * X[2] + 0.021 * X[4])
B_RES["decomposition_check"] = {
    "resid_min": float(dec.min()), "resid_max": float(dec.max()),
    "span": float(dec.max() - dec.min()), "mean": float(dec.mean()),
    "exact_constant": bool(float(dec.max() - dec.min()) < 1e-9),
    "meaning": ("0.7*lg(p_v19) == 0.385*lg_v6 + 0.385*lg_Q - 0.07*p_EBM + 0.021*off "
                "+ 0.7*d_mate 인가. 상수면 6팔 분해가 정확하고 상수는 d 가 흡수한다")}

ARMS_FULL = [("lg_v6", X[0], 0.385), ("lg_Q", X[1], 0.385),
             ("p_EBM", X[2], -0.07), ("off", X[4], 0.021),
             ("lg_p_aa", lg(p_aa), C["aa"]), ("lg_p_T70", lg(p_T70), C["T70"])]
ARMS_COARSE = [("lg_p_v19", lg(p_v19), 0.7),
               ("lg_p_aa", lg(p_aa), C["aa"]), ("lg_p_T70", lg(p_T70), C["T70"])]


def parent_equiv(arms):
    """alpha=1 표현에서의 부모 등가점: c' = A0*c, beta' = B0/A0^2."""
    return np.r_[A0 * np.array([a[2] for a in arms], np.float64), B0 / (A0 * A0)]


def full_refit(arms, with_nn, x0):
    Mx = np.vstack([a[1] for a in arms] + ([lgNN] if with_nn else []))
    r = minimize(lambda q: brier_of(q[:-1] @ Mx, 1.0, float(q[-1])),
                 np.asarray(x0, np.float64), method="Nelder-Mead",
                 options={"xatol": 1e-9, "fatol": 1e-16,
                          "maxiter": 40000, "maxfev": 40000})
    names = [a[0] for a in arms] + (["lg_NN"] if with_nn else [])
    return ({"coef": {n: float(x) for n, x in zip(names, r.x[:-1])},
             "beta_skew": float(r.x[-1]), "alpha_fixed": 1.0,
             "brier": float(r.fun), "score": conv(float(r.fun)),
             "nfev": int(r.nfev), "converged": bool(r.success)}, r.x, Mx)


for tag, arms in (("full6", ARMS_FULL), ("coarse3", ARMS_COARSE)):
    pe = parent_equiv(arms)
    Mx0 = np.vstack([a[1] for a in arms])
    b_start = brier_of(pe[:-1] @ Mx0, 1.0, float(pe[-1]))
    r5, x5, _ = full_refit(arms, False, pe)
    r6, x6, Mx6 = full_refit(arms, True, np.r_[x5[:-1], 0.0, x5[-1]])
    B_RES[f"form2_{tag}"] = {
        "parent_equiv_start_brier": b_start,
        "parent_equiv_start_score": conv(b_start),
        "parent_equiv_minus_parent_pts": K * (b_par - b_start),
        "arms_without_NN": r5, "arms_with_NN": r6,
        "gain_reweight_only_vs_parent_pts": K * (b_par - r5["brier"]),
        "NN_attributable_gain_pts": K * (r5["brier"] - r6["brier"]),
        "total_gain_vs_parent_pts": K * (b_par - r6["brier"]),
        "🔴_caveat": ("표본 내 전면 재가중이다 — reweight.py 가 경고한 과적합 경로. "
                      "수축표를 같이 본다.")}
    base_vec = np.r_[pe[:-1], 0.0, pe[-1]]
    shr = {}
    for lam in (0.25, 0.5, 0.75, 1.0):
        vv = base_vec + lam * (x6 - base_vec)
        bb = brier_of(vv[:-1] @ Mx6, 1.0, float(vv[-1]))
        shr[str(lam)] = {"brier": bb, "score": conv(bb),
                         "gain_vs_parent_pts": K * (b_par - bb),
                         "w_NN": float(vv[-2])}
    B_RES[f"form2_{tag}"]["shrinkage"] = shr
    print(f"  [면B ② {tag}] 시작점 {K * (b_par - b_start):+.5f}  "
          f"재가중만 {K * (b_par - r5['brier']):+.5f}  "
          f"NN귀속 {K * (r5['brier'] - r6['brier']):+.5f}  "
          f"합 {K * (b_par - r6['brier']):+.5f}  w_NN {r6['coef']['lg_NN']:+.5f}")

B_RES["reweight_headroom_pts"] = {
    "full6": (B_RES["form2_full6"]["NN_attributable_gain_pts"]
              - B_RES["form1b_refit_alpha_beta"]["NN_attributable_gain_pts"]),
    "coarse3": (B_RES["form2_coarse3"]["NN_attributable_gain_pts"]
                - B_RES["form1b_refit_alpha_beta"]["NN_attributable_gain_pts"]),
    "meaning": "① 비율고정 대비 ② 전체재적합이 더 뽑아내는 몫 = 재가중 여지"}

gsb, B_RES["per_seed_at_w_star"] = [], {
    "convention": ("①-b 의 w*·alpha*·beta* 를 고정하고 NN 벡터만 시드별로 바꿔 넣는다. "
                   "기준선은 w=0 재적합 값")}
for s in NN_SEEDS:
    bb = brier_of(z + wB * lg(NNS[s]), a_s, b_s)
    gsb.append(K * (br_b0 - bb))
    B_RES["per_seed_at_w_star"][str(s)] = {"brier": bb, "score": conv(bb),
                                           "gain_vs_refit_baseline_pts": K * (br_b0 - bb)}
gsb = np.array(gsb)
sdb = float(gsb.std(ddof=1))
B_RES["stability"] = {"gains_pts": gsb.tolist(), "mean": float(gsb.mean()),
                      "std_ddof1": sdb,
                      "mean_over_std": float(gsb.mean() / sdb) if sdb > 0 else None,
                      "sign_all_positive": bool((gsb > 0).all()),
                      "sign_count": f"{int((gsb > 0).sum())}/4",
                      "pass_3sigma": bool(gsb.mean() >= 3.0 * sdb)}
B_PASS = bool(B_RES["form1b_refit_alpha_beta"]["NN_attributable_gain_pts"] >= 3.0
              and (gsb > 0).all() and gsb.mean() >= 3.0 * sdb)
B_RES["gate"] = {"rule": "면 B(하한) NN 귀속 이득 >= +3점 · 부호 4/4 · 평균 >= 3*std(ddof=1)",
                 "pass": B_PASS}
M["face_B_current_lower_bound"] = B_RES
WALL["s4_faceB_min"] = (time.time() - t) / 60.0
print(f"      개별시드 이득 {np.round(gsb, 4).tolist()}  "
      f"평균/std {gsb.mean() / sdb if sdb > 0 else float('nan'):.3f}")

M["verdict"] = {
    "face_A_pass": A_PASS, "face_B_pass": B_PASS,
    "adopt": bool(A_PASS and B_PASS),
    "prereg": "면 A 통과 그리고 면 B(하한) 이득 >= +3점 · 부호 4/4 · 평균 >= 3*std"}
WALL["total_min"] = (time.time() - T_START) / 60.0
M["wall_clock"] = WALL
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(M["verdict"], indent=1, ensure_ascii=False))
print(f"총 {WALL['total_min']:.2f}분")

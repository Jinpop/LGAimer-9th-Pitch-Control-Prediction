"""exp_066 — exp_065 면 A 의 하나뿐인 양성 측정을, 더 센 상대(AA81) 로 다시 잰다.

순서(config 지정, 이 순서를 지킨다):
  ① 재현   : AA67(exp_048 A1@1400 3시드 확률평균) 상대로 exp_065 값을 바이트급 재현
  ② 본론   : AA81(exp_049 A1@1400 3시드 확률평균) 로 상대를 갈아끼우고 같은 형태로 격자 적합
  ③ 부수   : T70 을 세번째 팔로 (재적합 불필요 — exp_055 가 이미 split_A out-of-time 벡터를 갖고 있다)

🔴 두 이득을 나눠 적는다:
    vs_AA단독 = AA 위에 NN 을 얹은 값  (판정 대상)
    vs_NN단독 = NN 위에 AA 를 얹은 값  (다른 양)
🔴 등급을 사후에 만들지 않는다. 게이트는 config 의 하나뿐이고,
   크기는 exp_065 의 +108.39147 대비 몇 % 로 남았는지로만 적는다.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp066/out"
os.makedirs(OUT, exist_ok=True)
EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
NN_SEEDS = (42, 777, 2024, 101)
AA_SEEDS = (42, 7, 2024)
T_START = time.time()
WALL = {}

# exp_065 가 기록한 값 — 재현 대조용 상수 (여기서 고쳐 쓰지 않는다)
E065 = {
    "AA67_brier": 0.247808741181198,
    "AA67_score": 799.8813558800832,
    "NN_brier": 0.24756767168405058,
    "NN_score": 896.3836931443536,
    "w_star": 0.75,
    "blend_brier": 0.24753797180675718,
    "blend_score": 908.2728272288731,
    "gain_vs_AA_alone_pts": 108.3914713487961,
    "gain_vs_NN_alone_pts": 11.889134084526383,
    "corr_logit_NN_AA": 0.8979219511101727,
    "per_seed_gain_pts": [62.979908049248195, 77.98566656094255,
                          80.69035584234855, 82.2996133790871],
}


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1.0 - EPS))


def conv(b):
    return 100000.0 * (1.0 - b / REF)


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, np.float64).tobytes()).hexdigest()


def shafile(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


M = {"exp_id": "exp_066",
     "type": "정직면 재확인 — 상대(AA) 판본 교체. 학습 0건(T70 재적합 불필요, §T70_재적합_불필요 참조)",
     "unit": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
              "n_valid_all": 253507,
              "note": "fold_2022 / fold_2023 는 이 실험의 측정 대상이 아니다 — "
                      "재료 벡터가 fold_2024 out-of-time 예측만 존재한다."},
     "score_conversion": {"formula": "100000 * (1 - Brier / 0.2498069)",
                          "ref_brier": REF,
                          "sensitivity": "Brier 1e-6 = 0.4003점. 6자리 Brier 환산은 ±0.2002 불확실(S3)"}}
t = time.time()

# ══════════════════════════════════════════════════════════════════════
# 0. 정렬 — exp_065 와 같은 행 순서인가
# ══════════════════════════════════════════════════════════════════════
ids = np.asarray(np.load(f"{H}/arms/ids.npy", allow_pickle=True)).astype(str)
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success"])
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

# train.csv 자연 순서(2024 구간)가 row_id 정렬과 같은가 — 저장 예측 벡터의 순서 전제
tr_nat = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id", "season"])
nat24 = tr_nat[tr_nat.season == 2024].row_id.to_numpy().astype(str)
nat_eq_sorted = bool(np.array_equal(nat24, rid))
M["alignment"] = {"n_rows": N,
                  "arms_ids_eq_rowid_sorted": True,
                  "frame99_2024_same_order_as_rowid_sorted": f99_same_order,
                  "permutation_applied_to_NN_preds": perm is not None,
                  "train_csv_natural_order_2024_eq_rowid_sorted": nat_eq_sorted,
                  "y_mean": float(y.mean()),
                  "y_sha256": sha(y),
                  "y_sha256_exp065_recorded":
                      "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"}
assert (M["alignment"]["y_sha256"]
        == M["alignment"]["y_sha256_exp065_recorded"]), "🔴 라벨 벡터가 exp_065 와 다르다"
del tr_nat


def nnload(path):
    a = np.load(path).astype(np.float64)
    assert a.shape == (N,), (path, a.shape)
    return a if perm is None else a[perm]


# ══════════════════════════════════════════════════════════════════════
# 0b. S14 — arms/* 를 값의 범위로 확인 (이름으로 판단 금지)
# ══════════════════════════════════════════════════════════════════════
X = np.load(f"{H}/arms/X_arms.npy")
assert X.shape == (5, N)
XN = ["X0_lg_p_v6", "X1_lg_p_q", "X2_p_ebm", "X3_lg_p_AA_dep", "X4_off"]
M["S14_range_check"] = {}
for i, nm in enumerate(XN):
    lo, hi = float(X[i].min()), float(X[i].max())
    inside01 = bool(lo >= 0.0 and hi <= 1.0)
    M["S14_range_check"][nm] = {
        "min": lo, "max": hi, "inside_0_1": inside01,
        "verdict": "확률로 보임 → lg() 필요" if inside01 else "이미 로짓 → lg() 씌우지 말 것"}
assert not M["S14_range_check"]["X2_p_ebm"]["inside_0_1"], \
    "p_ebm 이 [0,1] 안이다 — S14 전제가 깨졌다"
M["S14_range_check"]["note"] = ("이 실험의 면 A 산술은 arms/* 를 쓰지 않는다. "
                                "S14 는 환경 동일성 확인으로만 기록한다.")
WALL["s0_align_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 1. 1-ulp 함정 확인 + BG 항등 앵커 (환경 동일성 — 면 A 는 z 를 쓰지 않는다)
# ══════════════════════════════════════════════════════════════════════
t = time.time()
bl = json.load(open(f"{H}/build_BG/pkg_BG/blend.json", encoding="utf-8"))
A0, B0, D0, T0 = bl["alpha"], bl["beta_skew"], bl["d_adj"], bl["target_rate"]
C = {"aa": 0.25244, "S2": 0.0, "S1": 0.0, "T70": 0.04756}
p_v19 = np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy")
p_aa = np.load(f"{H}/build_AY/work/pcal_aa.npy")
p_S2 = np.load(f"{H}/build_AY/work/pcal_S2.npy")
p_S1 = np.load(f"{H}/build_AY/work/pcal_S1.npy")
p_T70c = np.load(f"{H}/build_AY/work/pcal_T70.npy")
z = (0.7 * lg(p_v19) + C["aa"] * lg(p_aa) + C["S2"] * lg(p_S2)
     + C["S1"] * lg(p_S1) + C["T70"] * lg(p_T70c))
z_ref = np.load(f"{H}/audit_BH/z_tr.npy")
r0_ref = np.load(f"{H}/audit_BH/r0_tr.npy")
G1 = bool(np.array_equal(z, z_ref))
# 🔴 1-ulp 함정: 저장 벡터(audit_BH/r0_tr.npy)는 B*(z*z) 판, 출하 산술은 (B*z)*z 판.
pA_lin = expit(A0 * z + B0 * (z * z) + D0)   # 저장 벡터와 같은 결합 순서
pA = expit(A0 * z + B0 * z * z + D0)       # 출하 산술 순서
G2 = bool(np.array_equal(y - pA_lin, r0_ref))
b_par = float(np.mean((pA - y) ** 2))
M["env_anchor_BG"] = {
    "purpose": "환경 동일성 확인 전용 — 면 A 산술은 z 를 쓰지 않는다",
    "G1_z_byte_identical_to_audit_BH_z_tr": G1,
    "G2_r0_byte_identical_to_audit_BH_r0_tr": G2,
    "ulp_trap_checked_first": True,
    "association_order": {
        "max_abs_p_diff_between_orders": float(np.max(np.abs(pA - pA_lin))),
        "n_rows_differing": int(np.sum(pA != pA_lin)),
        "note": ("B*(z*z) 와 (B*z)*z 는 결합 순서가 달라 최대 1 ulp 어긋난다. "
                 "저장 벡터 대조에는 앞 판, 출하 산술에는 뒤 판. "
                 "exp_065 실측 133행 / 1.11e-16 와 대조한다.")},
    "exp065_recorded_n_rows_differing": 133,
    "exp065_recorded_max_abs_p_diff": 1.1102230246251565e-16,
    "parent_brier": b_par, "parent_score": conv(b_par),
    "z_sha256": sha(z), "pA_sha256": sha(pA),
    "exp065_recorded_z_sha256":
        "1512ce4db5815d070e66a5b7a4ac56388253e3b0f3101b5c48b385508c595099",
    "exp065_recorded_pA_sha256":
        "6b730f92283f3c40fecab792c9e842c97798c5cc501d47874b9fab2c170a1c9b"}
assert G1 and G2, "🔴 BG 항등 앵커 실패 — 환경이 exp_065 와 다르다. 중단."
assert M["env_anchor_BG"]["z_sha256"] == M["env_anchor_BG"]["exp065_recorded_z_sha256"]
assert M["env_anchor_BG"]["pA_sha256"] == M["env_anchor_BG"]["exp065_recorded_pA_sha256"]
del p_v19, p_aa, p_S2, p_S1, p_T70c, z, z_ref, r0_ref, pA, pA_lin
WALL["s1_env_anchor_min"] = (time.time() - t) / 60.0
print(f"[환경앵커] G1 {G1} G2 {G2}  1-ulp 행수 "
      f"{M['env_anchor_BG']['association_order']['n_rows_differing']}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 2. 재료
# ══════════════════════════════════════════════════════════════════════
t = time.time()
NN = nnload(f"{H}/exp064/out/preds/A99ship_seedavg.npy")
NNS = {s: nnload(f"{H}/exp064/out/preds/A99ship_s{s}.npy") for s in NN_SEEDS}

P48 = {s: f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed{s}_at1400.npy"
       for s in AA_SEEDS}
P49A0 = {s: f"{H}/experiments/exp_049/pred/pred_A0_u2024_seed{s}_at1400.npy"
         for s in AA_SEEDS}
P49A1 = {s: f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy"
         for s in AA_SEEDS}
P55T = {s: f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy"
        for s in AA_SEEDS}

AA67S = {s: np.load(P48[s]).astype(np.float64) for s in AA_SEEDS}
AA81S = {s: np.load(P49A1[s]).astype(np.float64) for s in AA_SEEDS}
T70S = {s: np.load(P55T[s]).astype(np.float64) for s in AA_SEEDS}
for d_ in (AA67S, AA81S, T70S):
    for s in AA_SEEDS:
        assert d_[s].shape == (N,), (s, d_[s].shape)
AA67 = np.mean([AA67S[s] for s in AA_SEEDS], axis=0)
AA81 = np.mean([AA81S[s] for s in AA_SEEDS], axis=0)
T70 = np.mean([T70S[s] for s in AA_SEEDS], axis=0)

lgNN, lgAA67, lgAA81, lgT70 = lg(NN), lg(AA67), lg(AA81), lg(T70)
b_nn = float(np.mean((NN - y) ** 2))
b_aa67 = float(np.mean((AA67 - y) ** 2))
b_aa81 = float(np.mean((AA81 - y) ** 2))
b_t70 = float(np.mean((T70 - y) ** 2))

M["materials"] = {
    "NN": {"src": "exp064/out/preds/A99ship_seedavg.npy (4시드 로짓평균, out-of-time)",
           "sha256_file": shafile(f"{H}/exp064/out/preds/A99ship_seedavg.npy"),
           "brier": b_nn, "score": conv(b_nn),
           "exp065_recorded_brier": E065["NN_brier"],
           "delta_vs_exp065": b_nn - E065["NN_brier"]},
    "AA67": {"src": f"exp_048 A1@1400 시드{AA_SEEDS} 확률평균 (67열, out-of-time)",
             "files": {str(s): P48[s] for s in AA_SEEDS},
             "sha256_files": {str(s): shafile(P48[s]) for s in AA_SEEDS},
             "sha256_mean_vector": sha(AA67),
             "brier": b_aa67, "score": conv(b_aa67),
             "exp065_recorded_brier": E065["AA67_brier"],
             "delta_vs_exp065": b_aa67 - E065["AA67_brier"],
             "per_seed_brier": {str(s): float(np.mean((AA67S[s] - y) ** 2))
                                for s in AA_SEEDS}},
    "AA81": {"src": f"exp_049 A1@1400 시드{AA_SEEDS} 확률평균 (81열, out-of-time)",
             "files": {str(s): P49A1[s] for s in AA_SEEDS},
             "sha256_files": {str(s): shafile(P49A1[s]) for s in AA_SEEDS},
             "sha256_mean_vector": sha(AA81),
             "brier": b_aa81, "score": conv(b_aa81),
             "per_seed_brier": {str(s): float(np.mean((AA81S[s] - y) ** 2))
                                for s in AA_SEEDS},
             "exp049_recorded_mean_of_briers_at1400": 0.24777260667585732,
             "exp049_recorded_mean_of_briers_score": 814.34633,
             "note": ("exp_049 가 기록한 0.2477726 은 mean-of-briers. "
                      "여기 값은 3시드 **평균예측**의 Brier — exp_065 의 AA67 과 같은 정의")},
    "T70": {"src": f"exp_055 T70@1400 시드{AA_SEEDS} 확률평균 (split_A, out-of-time)",
            "files": {str(s): P55T[s] for s in AA_SEEDS},
            "sha256_files": {str(s): shafile(P55T[s]) for s in AA_SEEDS},
            "sha256_mean_vector": sha(T70),
            "brier": b_t70, "score": conv(b_t70),
            "per_seed_brier": {str(s): float(np.mean((T70S[s] - y) ** 2))
                               for s in AA_SEEDS},
            "exp055_recorded_per_seed_brier": {
                "42": 0.24783713924616857, "7": 0.24784420685480416,
                "2024": 0.24786031240823916}},
    "per_seed_NN": {str(s): {"brier": float(np.mean((NNS[s] - y) ** 2)),
                             "score": conv(float(np.mean((NNS[s] - y) ** 2)))}
                    for s in NN_SEEDS},
    "corr_logit": {
        "NN_AA67": float(np.corrcoef(lgNN, lgAA67)[0, 1]),
        "NN_AA81": float(np.corrcoef(lgNN, lgAA81)[0, 1]),
        "NN_T70": float(np.corrcoef(lgNN, lgT70)[0, 1]),
        "AA81_T70": float(np.corrcoef(lgAA81, lgT70)[0, 1]),
        "AA67_AA81": float(np.corrcoef(lgAA67, lgAA81)[0, 1]),
        "AA67_T70": float(np.corrcoef(lgAA67, lgT70)[0, 1])}}

# 출처 대조 — exp_049 A0 가 exp_048 A1 과 바이트 동일이어야 판본 비교가 성립
prov = {str(s): {"exp048_A1_sha": shafile(P48[s]),
                 "exp049_A0_sha": shafile(P49A0[s]),
                 "bit_identical": shafile(P48[s]) == shafile(P49A0[s])}
        for s in AA_SEEDS}
M["materials"]["provenance_AA67_eq_exp049_control"] = {
    "per_seed": prov,
    "ALL_BIT_IDENTICAL": all(v["bit_identical"] for v in prov.values()),
    "meaning": ("exp_049 의 대조군 A0(67열)가 exp_048 A1 과 바이트 동일 → "
                "AA67 과 AA81 은 **같은 하네스·같은 OOF·같은 시드** 위의 "
                "열 수 차이 하나만 다른 판본이다")}
assert M["materials"]["provenance_AA67_eq_exp049_control"]["ALL_BIT_IDENTICAL"]

# T70 이 정말 split_A 로 적합됐는가 — 재적합이 불필요한 근거를 기계로 확인
tz = np.load(f"{H}/build_AH/localref/oof/stack_fold2024_seed42.npz",
             allow_pickle=False)
n_tr_full = int(tz["tr_full"].astype(bool).sum())
n_va = int(tz["va"].astype(bool).sum())
M["T70_재적합_불필요"] = {
    "config_가정": ("배포 T70(build_AY/seeds/seed_*_T70) 은 n_full 1,369,784 = 2024 포함 "
                    "표본내라 못 쓴다 — 맞다. 그래서 run_t70_fits.sh 를 split_A 로 "
                    "다시 돌리라고 했다."),
    "실측": ("그러나 exp_055 가 **이미** split_A 마스크로 T70 을 적합하고 2024 "
             "out-of-time 예측을 저장해 두었다(experiments/exp_055/pred/). "
             "재적합은 같은 산출물을 다시 만드는 일이라 하지 않았다."),
    "mask_source": f"{H}/build_AH/localref/oof/stack_fold2024_seed42.npz",
    "n_train_rows": n_tr_full,
    "n_valid_rows": n_va,
    "expected_split_A_n_train": 1116277,
    "mask_is_split_A": bool(n_tr_full == 1116277 and n_va == N),
    "deploy_T70_n_full": 1369784,
    "alpha": 0.70, "arms": "c67", "rounds": 1400, "seeds": list(AA_SEEDS)}
assert M["T70_재적합_불필요"]["mask_is_split_A"], \
    f"🔴 exp_055 T70 의 학습 마스크가 split_A 가 아니다: {n_tr_full}"
del tz
WALL["s2_materials_min"] = (time.time() - t) / 60.0
print(f"[재료] NN {conv(b_nn):.5f}  AA67 {conv(b_aa67):.5f}  "
      f"AA81 {conv(b_aa81):.5f}  T70 {conv(b_t70):.5f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 3. 재현 — AA67 상대로 exp_065 면 A 를 그대로 다시 낸다
# ══════════════════════════════════════════════════════════════════════
t = time.time()


def faceA(w, ln, la):
    return expit(w * ln + (1.0 - w) * la)


def gridsolve(fn, hi, step=0.005):
    ws = np.round(np.arange(0.0, hi + 1e-12, step), 6)
    bs = np.array([fn(float(w)) for w in ws])
    return ws, bs, int(np.argmin(bs))


def two_arm(ln, la, b_a, b_n, tag_a, tag_n, hi=1.00):
    ws, bs, j = gridsolve(
        lambda w: float(np.mean((faceA(w, ln, la) - y) ** 2)), hi)
    w_, b_ = float(ws[j]), float(bs[j])
    cap = bool(j == 0 or j == len(ws) - 1)
    if j == 0:
        cv = ("🔴 argmin 이 격자 아래 끝(w=0)에 닿았다 — 볼록 혼합의 정의역 끝이다. "
              f"뜻: 이 형태 안에서 {tag_n} 을 얹을 이유가 없다. "
              "게이트의 'w > 0' 조건이 이것을 그대로 잡는다(범위 확대로 풀리지 않는다).")
    elif j == len(ws) - 1:
        cv = f"🔴 무효 — argmin 이 격자 위 끝(w={hi})에 닿았다. 범위를 넓혀야 한다"
    else:
        cv = "유효 — argmin 이 격자 내부"
    return {"form": f"p = expit(w*lg({tag_n}) + (1-w)*lg({tag_a})) — d 없음",
            "grid": {"lo": 0.0, "hi": hi, "step": 0.005,
                     "domain_note": ("볼록 혼합이므로 w 는 [0,1] 로 제한된다. "
                                     "음수 최적이 참이면 argmin 은 0 에 앉는다.")},
            f"optimal_w_{tag_n}": w_, "optimal_w": w_,
            "brier_at_opt": b_, "score_at_opt": conv(b_),
            f"gain_vs_{tag_a}_alone_pts": K * (b_a - b_),
            f"gain_vs_{tag_n}_alone_pts": K * (b_n - b_),
            "cap_touched": cap,
            "cap_verdict": cv,
            "curve": {f"{float(w):.3f}": {"brier": float(b), "score": conv(float(b))}
                      for w, b in zip(ws, bs)
                      if abs(round(float(w) * 1000) % 50) == 0
                      or abs(float(w) - w_) < 1e-9}}, w_, b_


REP, w67, b67 = two_arm(lgNN, lgAA67, b_aa67, b_nn, "AA67", "NN")
REP["baseline"] = "AA67 단독 (w=0)"
chk = {
    "AA67_alone_brier": {"now": b_aa67, "exp065": E065["AA67_brier"],
                         "abs_diff": abs(b_aa67 - E065["AA67_brier"])},
    "NN_alone_brier": {"now": b_nn, "exp065": E065["NN_brier"],
                       "abs_diff": abs(b_nn - E065["NN_brier"])},
    "w_star": {"now": w67, "exp065": E065["w_star"],
               "abs_diff": abs(w67 - E065["w_star"])},
    "blend_brier": {"now": b67, "exp065": E065["blend_brier"],
                    "abs_diff": abs(b67 - E065["blend_brier"])},
    "blend_score": {"now": conv(b67), "exp065": E065["blend_score"],
                    "abs_diff": abs(conv(b67) - E065["blend_score"])},
    "gain_vs_AA_alone_pts": {"now": REP["gain_vs_AA67_alone_pts"],
                             "exp065": E065["gain_vs_AA_alone_pts"],
                             "abs_diff": abs(REP["gain_vs_AA67_alone_pts"]
                                             - E065["gain_vs_AA_alone_pts"])},
    "gain_vs_NN_alone_pts": {"now": REP["gain_vs_NN_alone_pts"],
                             "exp065": E065["gain_vs_NN_alone_pts"],
                             "abs_diff": abs(REP["gain_vs_NN_alone_pts"]
                                             - E065["gain_vs_NN_alone_pts"])},
    "corr_logit_NN_AA67": {"now": M["materials"]["corr_logit"]["NN_AA67"],
                           "exp065": E065["corr_logit_NN_AA"],
                           "abs_diff": abs(M["materials"]["corr_logit"]["NN_AA67"]
                                           - E065["corr_logit_NN_AA"])},
}
_all_exact = all(v["abs_diff"] == 0.0 for v in chk.values())
_all_pass = all(v["abs_diff"] < 1e-12 for v in chk.values())
REP["reproduction_vs_exp065"] = dict(chk, tol=1e-12,
                                     ALL_EXACT=_all_exact, PASS=_all_pass)
M["step1_reproduction_AA67"] = REP
print(f"[재현] w* {w67}  {conv(b67):.5f}점  vs AA67 "
      f"{REP['gain_vs_AA67_alone_pts']:+.5f}  vs NN "
      f"{REP['gain_vs_NN_alone_pts']:+.5f}  "
      f"정확일치 {REP['reproduction_vs_exp065']['ALL_EXACT']}", flush=True)
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)
assert REP["reproduction_vs_exp065"]["PASS"], \
    "🔴 재현 실패 — 판본 비교가 무의미하다. 중단하고 보고."
WALL["s3_reproduce_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 4. ① 본론 — AA81 로 상대를 갈아끼운다 (같은 형태)
# ══════════════════════════════════════════════════════════════════════
t = time.time()
MAIN, w81, b81 = two_arm(lgNN, lgAA81, b_aa81, b_nn, "AA81", "NN")
MAIN["baseline"] = "AA81 단독 (w=0)"
MAIN["🔴_두_이득"] = {
    "vs_AA81단독_pts": MAIN["gain_vs_AA81_alone_pts"],
    "vs_AA81단독_뜻": "AA81 위에 NN 을 얹으면 얼마인가 — **판정 대상**",
    "vs_NN단독_pts": MAIN["gain_vs_NN_alone_pts"],
    "vs_NN단독_뜻": "NN 위에 AA81 을 얹으면 얼마인가 — 다른 양이다"}
MAIN["exp065_대비"] = {
    "exp065_vs_AA단독_pts": E065["gain_vs_AA_alone_pts"],
    "exp066_vs_AA단독_pts": MAIN["gain_vs_AA81_alone_pts"],
    "남은_비율_pct": 100.0 * MAIN["gain_vs_AA81_alone_pts"] / E065["gain_vs_AA_alone_pts"],
    "exp065_vs_NN단독_pts": E065["gain_vs_NN_alone_pts"],
    "exp066_vs_NN단독_pts": MAIN["gain_vs_NN_alone_pts"],
    "vs_NN단독_남은_비율_pct":
        100.0 * MAIN["gain_vs_NN_alone_pts"] / E065["gain_vs_NN_alone_pts"],
    "상대가_세진_폭_pts": conv(b_aa81) - conv(b_aa67),
    "🔴_주의": ("이 두 비율은 서로 다른 양의 비율이다. 하나로 뭉뚱그리지 말 것. "
                "등급 구간은 만들지 않는다 — 비율 자체가 정보다.")}

gs, MAIN["per_seed_at_w_star"] = [], {
    "convention": ("exp_065 와 같다 — w* 를 고정하고 NN 벡터만 시드별로 바꿔 넣는다. "
                   "기준선은 AA81 단독")}
for s in NN_SEEDS:
    bsd = float(np.mean((faceA(w81, lg(NNS[s]), lgAA81) - y) ** 2))
    gs.append(K * (b_aa81 - bsd))
    MAIN["per_seed_at_w_star"][str(s)] = {"brier": bsd, "score": conv(bsd),
                                          "gain_vs_AA81_pts": K * (b_aa81 - bsd)}
gs = np.array(gs)
sd = float(gs.std(ddof=1))
MAIN["stability"] = {"gains_pts": gs.tolist(), "mean": float(gs.mean()),
                     "std_ddof1": sd,
                     "mean_over_std": float(gs.mean() / sd) if sd > 0 else None,
                     "sign_all_positive": bool((gs > 0).all()),
                     "sign_count": f"{int((gs > 0).sum())}/4",
                     "pass_3sigma": bool(gs.mean() >= 3.0 * sd),
                     "exp065_per_seed_gains_pts": E065["per_seed_gain_pts"]}

g_w = bool(w81 > 0.0)
g_3 = bool(MAIN["gain_vs_AA81_alone_pts"] >= 3.0)
g_sign = bool((gs > 0).all())
g_sig = bool(gs.mean() >= 3.0 * sd)
MAIN["gate"] = {
    "rule": ("사전등록: 최적 w_NN > 0  그리고  AA81 단독 대비 >= +3점  그리고  "
             "개별 4시드 부호 4/4  그리고  평균 >= 3*std(ddof=1)"),
    "condition_w_gt_0": g_w,
    "condition_gain_ge_3pts": g_3,
    "condition_sign_4of4": g_sign,
    "condition_mean_ge_3sigma": g_sig,
    "PASS": bool(g_w and g_3 and g_sign and g_sig),
    "cap_valid": not MAIN["cap_touched"]}
M["step2_main_AA81"] = MAIN
print(f"[본론] w* {w81}  {conv(b81):.5f}점  vs AA81단독 "
      f"{MAIN['gain_vs_AA81_alone_pts']:+.5f}  vs NN단독 "
      f"{MAIN['gain_vs_NN_alone_pts']:+.5f}  게이트 {MAIN['gate']['PASS']}", flush=True)
print(f"       exp_065 대비 남은 비율 "
      f"{MAIN['exp065_대비']['남은_비율_pct']:.2f}%  "
      f"개별시드 {np.round(gs, 4).tolist()}  평균/std "
      f"{(gs.mean() / sd) if sd > 0 else float('nan'):.3f}", flush=True)
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)
WALL["s4_main_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 5. ② 부수 — T70 을 세번째 팔로 (③ 파생 = 우리 재료만의 정직면 최고점)
# ══════════════════════════════════════════════════════════════════════
t = time.time()
SUB = {"form": ("p = expit(w_NN*lg(NN) + w_T70*lg(T70) "
                "+ (1-w_NN-w_T70)*lg(AA81)) — d 없음"),
       "T70_출처": "exp_055 pred_T70_u2024_*@1400 (split_A 적합, 재적합 없음)"}


def b3(w1, w2):
    zz = w1 * lgNN + w2 * lgT70 + (1.0 - w1 - w2) * lgAA81
    return float(np.mean((expit(zz) - y) ** 2))


# 2팔 대조들 (전부 같은 격자 규약)
AT, wAT, bAT = two_arm(lgT70, lgAA81, b_aa81, b_t70, "AA81", "T70")
AT["baseline"] = "AA81 단독 (w=0)"
NT_, wNT, bNT = two_arm(lgT70, lgNN, b_nn, b_t70, "NN", "T70")
NT_["baseline"] = "NN 단독 (w=0)"
SUB["two_arm_AA81_plus_T70"] = AT
SUB["two_arm_NN_plus_T70"] = NT_
SUB["two_arm_AA81_plus_NN"] = {"optimal_w_NN": w81, "brier": b81,
                               "score": conv(b81),
                               "ref": "step2_main_AA81"}

step = 0.005
axis = np.round(np.arange(0.0, 1.0 + 1e-12, step), 6)
best = (None, None, np.inf)
for w1 in axis:
    w2max = 1.0 - float(w1)
    w2s = axis[axis <= w2max + 1e-12]
    if len(w2s) == 0:
        continue
    for i0 in range(0, len(w2s), 40):          # 메모리 상한을 위한 조각내기
        wc = w2s[i0:i0 + 40]
        Zb = (float(w1) * lgNN[None, :] + wc[:, None] * lgT70[None, :]
              + (1.0 - float(w1) - wc)[:, None] * lgAA81[None, :])
        Bb = np.mean((expit(Zb) - y[None, :]) ** 2, axis=1)
        k = int(np.argmin(Bb))
        if float(Bb[k]) < best[2]:
            best = (float(w1), float(wc[k]), float(Bb[k]))
        del Zb, Bb
w1s, w2s_, b3s = best
w3s = 1.0 - w1s - w2s_
on_edge = [nm for nm, v in (("w_NN=0", w1s), ("w_T70=0", w2s_),
                            ("w_AA81=0", w3s)) if abs(v) < 1e-12]
on_edge += ["w_NN=1"] if abs(w1s - 1.0) < 1e-12 else []
on_edge += ["w_T70=1"] if abs(w2s_ - 1.0) < 1e-12 else []
SUB["three_arm"] = {
    "grid": {"lo": 0.0, "hi": 1.0, "step": step,
             "domain": "단체(simplex) w_NN + w_T70 <= 1, 전부 >= 0"},
    "optimal": {"w_NN": w1s, "w_T70": w2s_, "w_AA81": w3s},
    "brier": b3s, "score": conv(b3s),
    "cap_touched": bool(len(on_edge) > 0),
    "cap_edges": on_edge,
    "cap_verdict": ("🔴 무효 — argmin 이 단체 경계에 닿았다: " + ", ".join(on_edge)
                    if on_edge else "유효 — argmin 이 단체 내부"),
    "gain_vs_AA81_alone_pts": K * (b_aa81 - b3s),
    "gain_vs_NN_alone_pts": K * (b_nn - b3s),
    "gain_vs_T70_alone_pts": K * (b_t70 - b3s),
    "gain_vs_2arm_AA81_NN_pts": K * (b81 - b3s),
    "gain_vs_2arm_AA81_T70_pts": K * (bAT - b3s),
    "NN_귀속분_3팔세계_pts": K * (bAT - b3s),
    "NN_귀속분_뜻": ("3팔 최적점 − (AA81+T70) 2팔 최적점. "
                     "3팔 세계에서 NN 을 더 얹어 얻은 몫"),
    "T70_귀속분_pts": K * (b81 - b3s),
    "T70_귀속분_뜻": "3팔 최적점 − (AA81+NN) 2팔 최적점"}

gs3, SUB["three_arm"]["per_seed_at_w_star"] = [], {
    "convention": "3팔 w* 고정, NN 벡터만 시드별 교체. 기준선은 (AA81+T70) 2팔 최적점"}
for s in NN_SEEDS:
    zz = (w1s * lg(NNS[s]) + w2s_ * lgT70 + w3s * lgAA81)
    bb = float(np.mean((expit(zz) - y) ** 2))
    gs3.append(K * (bAT - bb))
    SUB["three_arm"]["per_seed_at_w_star"][str(s)] = {
        "brier": bb, "score": conv(bb), "gain_vs_2arm_AA81_T70_pts": K * (bAT - bb)}
gs3 = np.array(gs3)
sd3 = float(gs3.std(ddof=1))
SUB["three_arm"]["stability"] = {
    "gains_pts": gs3.tolist(), "mean": float(gs3.mean()), "std_ddof1": sd3,
    "mean_over_std": float(gs3.mean() / sd3) if sd3 > 0 else None,
    "sign_all_positive": bool((gs3 > 0).all()),
    "sign_count": f"{int((gs3 > 0).sum())}/4",
    "pass_3sigma": bool(gs3.mean() >= 3.0 * sd3)}
SUB["🔴_주의"] = ("② 는 부수 측정이다. 사전 등록 게이트는 ① 하나뿐이며 "
                  "여기에는 게이트가 없다. 지분 0.04756 짜리 팔이라 작을 것으로 "
                  "예상됐고, 크게 나오면 의심하라는 것이 config 의 지시다.")
M["step3_sub_T70"] = SUB
print(f"[부수] 3팔 w_NN {w1s} w_T70 {w2s_} w_AA81 {w3s:.3f}  "
      f"{conv(b3s):.5f}점  vs AA81단독 "
      f"{SUB['three_arm']['gain_vs_AA81_alone_pts']:+.5f}  "
      f"NN귀속 {SUB['three_arm']['NN_귀속분_3팔세계_pts']:+.5f}  "
      f"cap {SUB['three_arm']['cap_verdict']}", flush=True)

# ③ 파생 — 우리 재료만의 정직면 최고점
cands = {"AA81_alone": b_aa81, "NN_alone": b_nn, "T70_alone": b_t70,
         "AA81+NN": b81, "AA81+T70": bAT, "NN+T70": bNT, "AA81+NN+T70": b3s}
bestname = min(cands, key=lambda k: cands[k])
M["step4_derived_best_honest_face"] = {
    "candidates": {k: {"brier": v, "score": conv(v)} for k, v in cands.items()},
    "best": bestname, "brier": cands[bestname], "score": conv(cands[bestname]),
    "🔴_용도": ("LB 프로브 시작점 근거로만 쓴다. S1 — fold_2024 값이고 "
                "배포는 <=2024 재적합이라 같은 모델이 아니다. LB 예측치로 옮기지 마라.")}
WALL["s5_sub_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 6. 판정 (판정 범위를 넘지 않는다)
# ══════════════════════════════════════════════════════════════════════
M["verdict"] = {
    "reproduction_PASS": REP["reproduction_vs_exp065"]["PASS"],
    "reproduction_EXACT": REP["reproduction_vs_exp065"]["ALL_EXACT"],
    "gate_①_PASS": MAIN["gate"]["PASS"],
    "gate_①_breakdown": {k: v for k, v in MAIN["gate"].items()
                         if k.startswith("condition")},
    "판정대상_vs_AA81단독_pts": MAIN["gain_vs_AA81_alone_pts"],
    "exp065_대비_남은_비율_pct": MAIN["exp065_대비"]["남은_비율_pct"],
    "🔴_판정_범위": ("이 실험은 C단 진행 여부를 결정하지 않는다. "
                     "게이트 통과/탈락과 숫자까지가 산출이다.")}

M["train_time_min"] = 0.0
M["inference_time_sec"] = None
M["train_time_note"] = ("학습 0건. ② 의 T70 재적합은 exp_055 가 이미 같은 마스크로 "
                        "적합해 저장한 벡터가 있어 수행하지 않았다(§T70_재적합_불필요).")
M["threads_env"] = {k: os.environ.get(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
WALL["total_min"] = (time.time() - T_START) / 60.0
M["wall_clock_min"] = WALL
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(M["verdict"], indent=1, ensure_ascii=False))
print(f"총 {WALL['total_min']:.2f}분")

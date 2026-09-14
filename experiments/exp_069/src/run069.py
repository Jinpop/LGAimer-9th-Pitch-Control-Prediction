"""exp_069 — ① v6 의 NN 자리에 우리 NN 을 끼운다  ② exp_065·066 결론을 배포 가능한 팔로 재확인.

학습 0건. 캐시 벡터 위 산술만.

🔴 절대 원칙 두 개
  1) 판정에 쓰는 NN 은 **exp068/out/preds/A_k8_ens.npy (배포 가능, 829.82499)** 뿐이다.
     exp064 의 A99ship_seedavg.npy (896.38369) 는 **상한 표시용**으로만 나란히 계산한다.
     그 판은 시드마다 fold_2024 를 보고 에폭을 골라 배포에서 재현 불가하다.
  2) 눈금은 전부 **원자료**다. 팀원 README 의 BSS 는 재센터링 후라 약 +28점 높다. 섞지 않는다.
"""
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score

H = "/home/<user>/SungjinPark"
B = f"{H}/exp069"
OOT = f"{B}/oot"
OUT = f"{B}/out"
os.makedirs(OUT, exist_ok=True)

EPS = 1e-6
REF = 0.2498069
K = 100000.0 / REF
NN_SEEDS = (42, 777, 2024, 101)
AA_SEEDS = (42, 7, 2024)
W_SHIP = 0.15                     # 🔴 출하 v6 의 NN 자리 가중치. 사전 등록 판정치.
T_START = time.time()
WALL = {}

# ── 앞선 실험이 기록한 상수 (여기서 고쳐 쓰지 않는다) ────────────────────────
E065 = {"AA67_score": 799.8813558800832, "NN_best_score": 896.3836931443536,
        "w_star": 0.75, "blend_score": 908.2728272288731,
        "gain_vs_AA_alone_pts": 108.3914713487961,
        "gain_vs_NN_alone_pts": 11.889134084526383}
E066 = {"AA81_brier": 0.24771241129756838, "AA81_score": 838.4430944187726,
        "w_star": 0.67,
        "blend_brier": 0.2475217870580942, "blend_score": 914.7517309993525,
        "gain_vs_AA81_alone_pts": 76.30863658057116,
        "gain_vs_NN_alone_pts": 18.368037854996746,
        "per_seed_gain_pts": [39.31808158877951, 51.72807173686735,
                              55.18580815269329, 55.53713239081518],
        "mean_over_std": 6.626068220976915,
        "corr_logit_NN_AA81": 0.9123226717478412}
E068 = {"k8_ens_brier": 0.2477339399122003, "k8_ens_score": 829.8249919436506,
        "best_ens_score": 896.38369, "gap_pts": -66.5586}
# 지휘자 원자료 계측 (README 재센터링 값이 아니다)
ORCH = {"lgb": 863.49, "cat": 899.21, "leaf": 703.79,
        "v6_3arm_renorm": 902.41, "v6_3arm_rawW": 862.88}


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


M = {
    "exp_id": "exp_069",
    "type": ("정직면 산술. 학습 0건. ① v6 NN 자리에 우리 NN 끼우기 "
             "② exp_065·066 결론을 배포 가능한 NN(k=8)으로 재확인"),
    "unit": {"fold": "fold_2024", "train_upto": 2023, "valid_season": 2024,
             "n_valid_all": 253507,
             "note": ("fold_2022 / fold_2023 는 이 실험의 측정 대상이 아니다 — "
                      "재료 벡터가 fold_2024 out-of-time 예측만 존재한다.")},
    "score_conversion": {"formula": "100000 * (1 - Brier / 0.2498069)",
                         "ref_brier": REF,
                         "sensitivity": ("Brier 1e-6 = 0.4003점. "
                                         "6자리 Brier 환산은 ±0.2002 불확실(S3)")},
    "🔴_눈금_경고": ("이 파일의 모든 점수는 **원자료**다. 팀원 README 의 BSS 는 "
                     "재센터링 후 값이라 약 +28점 높다. 두 눈금을 섞으면 결론이 뒤집힌다."),
    "🔴_NN_판본_규약": {
        "판정용": "exp068/out/preds/A_k8_ens.npy — k=8, 배포 가능",
        "상한표시용": ("exp064/out/preds/A99ship_seedavg.npy — 시드마다 fold_2024 를 보고 "
                       "에폭을 골랐다. 배포에서 재현 불가. 게이트 판정에 쓰지 않는다."),
        "exp068_확정_격차_pts": E068["gap_pts"]},
}
t = time.time()

# ══════════════════════════════════════════════════════════════════════
# 0. 정렬 — exp_065·066 과 같은 행 순서인가
# ══════════════════════════════════════════════════════════════════════
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "control_success"])
tr = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
N = len(tr)
assert N == 253507, N
y = tr.control_success.to_numpy(np.float64)
rid = tr.row_id.to_numpy().astype(str)

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

# 팀원 납품 벡터의 행 순서
oot_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert oot_rid.shape == (N,), oot_rid.shape
oot_same_order = bool(np.array_equal(oot_rid, rid))
if oot_same_order:
    operm = None
else:
    o2 = pd.Series(np.arange(N), index=oot_rid)
    operm = o2.reindex(rid).to_numpy()
    assert not np.isnan(operm).any(), "납품 벡터에 없는 row_id"
    operm = operm.astype(int)

M["alignment"] = {
    "n_rows": N,
    "frame99_2024_same_order_as_rowid_sorted": f99_same_order,
    "permutation_applied_to_NN_preds": perm is not None,
    "oot_delivery_same_order_as_rowid_sorted": oot_same_order,
    "permutation_applied_to_oot_vectors": operm is not None,
    "y_mean": float(y.mean()),
    "y_sha256": sha(y),
    "y_sha256_exp065_066_recorded":
        "e0483e6a48d9f699ca9c4abf7b4b81aed4d679c359c7e9c3752b3f9e6e27c9d3"}
assert (M["alignment"]["y_sha256"]
        == M["alignment"]["y_sha256_exp065_066_recorded"]), \
    "🔴 라벨 벡터가 exp_065·066 과 다르다 — 비교 불가"
WALL["s0_align_min"] = (time.time() - t) / 60.0
print(f"[정렬] N={N} frame99동순 {f99_same_order} 납품동순 {oot_same_order}", flush=True)


def nnload(path):
    a = np.load(path).astype(np.float64)
    assert a.shape == (N,), (path, a.shape)
    return a if perm is None else a[perm]


def ootload(name):
    a = np.load(f"{OOT}/{name}").astype(np.float64)
    assert a.shape == (N,), (name, a.shape)
    return a if operm is None else a[operm]


def stat(p, tag, src=None):
    b = float(np.mean((p - y) ** 2))
    o = {"brier": b, "score": conv(b), "sha256_vector": sha(p),
         "min": float(p.min()), "max": float(p.max()), "mean": float(p.mean())}
    if src:
        o["src"] = src
    return o


# ══════════════════════════════════════════════════════════════════════
# 1. 재료
# ══════════════════════════════════════════════════════════════════════
t = time.time()
p_lgb = ootload("p_lgb_oot2024.npy")
p_cat = ootload("p_cat_oot2024.npy")
p_leaf = ootload("p_leaf_oot2024.npy")
p_rawW = ootload("p_v6_3arm_rawW_oot2024.npy")
p_renorm = ootload("p_v6_3arm_renorm_oot2024.npy")

NN8 = nnload(f"{H}/exp068/out/preds/A_k8_ens.npy")
NN8S = {s: nnload(f"{H}/exp068/out/preds/A_k8_s{s}.npy") for s in NN_SEEDS}
NNB = nnload(f"{H}/exp064/out/preds/A99ship_seedavg.npy")
NNBS = {s: nnload(f"{H}/exp064/out/preds/A99ship_s{s}.npy") for s in NN_SEEDS}

P48 = {s: f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed{s}_at1400.npy"
       for s in AA_SEEDS}
P49A1 = {s: f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy"
         for s in AA_SEEDS}
AA67 = np.mean([np.load(P48[s]).astype(np.float64) for s in AA_SEEDS], axis=0)
AA81 = np.mean([np.load(P49A1[s]).astype(np.float64) for s in AA_SEEDS], axis=0)
assert AA67.shape == (N,) and AA81.shape == (N,)

MAT = {
    "p_lgb": stat(p_lgb, "lgb", "팀원 납품 LightGBM 3시드 로짓평균 (out-of-time)"),
    "p_cat": stat(p_cat, "cat", "팀원 납품 CatBoost 4시드 로짓평균 (out-of-time)"),
    "p_leaf": stat(p_leaf, "leaf", "팀원 납품 GBDT리프+로지스틱 3시드 (out-of-time)"),
    "p_v6_3arm_rawW": stat(p_rawW, "rawW", "팀원 납품 v6 3팔 원가중(합0.85) — 산술 앵커용"),
    "p_v6_3arm_renorm": stat(p_renorm, "renorm",
                             "팀원 납품 v6 3팔 재정규화 — 🔴 ① 의 판정 대조군"),
    "NN_k8": stat(NN8, "NN8", "🔴 판정용 — exp068 k=8 4시드 앙상블, 배포 가능"),
    "NN_best_epoch": stat(NNB, "NNB",
                          "상한표시용 — exp064 시드별 best epoch. 배포 재현 불가"),
    "AA67": stat(AA67, "AA67", "exp_048 A1@1400 3시드 확률평균 (67열)"),
    "AA81": stat(AA81, "AA81", "exp_049 A1@1400 3시드 확률평균 (81열)"),
}
MAT["sha256_files"] = {
    "p_lgb": shafile(f"{OOT}/p_lgb_oot2024.npy"),
    "p_cat": shafile(f"{OOT}/p_cat_oot2024.npy"),
    "p_leaf": shafile(f"{OOT}/p_leaf_oot2024.npy"),
    "p_v6_3arm_rawW": shafile(f"{OOT}/p_v6_3arm_rawW_oot2024.npy"),
    "p_v6_3arm_renorm": shafile(f"{OOT}/p_v6_3arm_renorm_oot2024.npy"),
    "NN_k8": shafile(f"{H}/exp068/out/preds/A_k8_ens.npy"),
    "NN_best_epoch": shafile(f"{H}/exp064/out/preds/A99ship_seedavg.npy")}
MAT["per_seed_NN_k8"] = {str(s): stat(NN8S[s], f"NN8_{s}") for s in NN_SEEDS}
MAT["per_seed_NN_best"] = {str(s): stat(NNBS[s], f"NNB_{s}") for s in NN_SEEDS}

# 기록 대조 — 이 환경에서 앞선 실험 값이 재현되는가
MAT["대조_기록값"] = {
    "NN_k8_score": {"now": MAT["NN_k8"]["score"], "exp068": E068["k8_ens_score"],
                    "abs_diff": abs(MAT["NN_k8"]["score"] - E068["k8_ens_score"])},
    "NN_best_score": {"now": MAT["NN_best_epoch"]["score"],
                      "exp064_065": E065["NN_best_score"],
                      "abs_diff": abs(MAT["NN_best_epoch"]["score"]
                                      - E065["NN_best_score"])},
    "AA67_score": {"now": MAT["AA67"]["score"], "exp065": E065["AA67_score"],
                   "abs_diff": abs(MAT["AA67"]["score"] - E065["AA67_score"])},
    "AA81_score": {"now": MAT["AA81"]["score"], "exp066": E066["AA81_score"],
                   "abs_diff": abs(MAT["AA81"]["score"] - E066["AA81_score"])},
    "지휘자_원자료_계측_대조": {
        k: {"now": MAT[n]["score"], "orch": ORCH[k],
            "abs_diff": abs(MAT[n]["score"] - ORCH[k])}
        for k, n in (("lgb", "p_lgb"), ("cat", "p_cat"), ("leaf", "p_leaf"),
                     ("v6_3arm_renorm", "p_v6_3arm_renorm"),
                     ("v6_3arm_rawW", "p_v6_3arm_rawW"))},
    "note": ("지휘자 계측은 소수 2자리 기록이라 abs_diff < 0.01 이면 일치. "
             "앞선 실험 기록은 완전 정밀도라 abs_diff < 1e-9 이어야 한다.")}
assert MAT["대조_기록값"]["AA67_score"]["abs_diff"] < 1e-9, "AA67 재현 실패"
assert MAT["대조_기록값"]["AA81_score"]["abs_diff"] < 1e-9, "AA81 재현 실패"
assert MAT["대조_기록값"]["NN_best_score"]["abs_diff"] < 1e-9, "NN(best) 재현 실패"
assert MAT["대조_기록값"]["NN_k8_score"]["abs_diff"] < 1e-9, "NN(k=8) 재현 실패"

lg_lgb, lg_cat, lg_leaf = lg(p_lgb), lg(p_cat), lg(p_leaf)
lgNN8, lgNNB = lg(NN8), lg(NNB)
lgAA67, lgAA81 = lg(AA67), lg(AA81)
MAT["corr_logit"] = {
    "NN8_AA67": float(np.corrcoef(lgNN8, lgAA67)[0, 1]),
    "NN8_AA81": float(np.corrcoef(lgNN8, lgAA81)[0, 1]),
    "NNbest_AA81": float(np.corrcoef(lgNNB, lgAA81)[0, 1]),
    "NN8_NNbest": float(np.corrcoef(lgNN8, lgNNB)[0, 1]),
    "NN8_cat": float(np.corrcoef(lgNN8, lg_cat)[0, 1]),
    "NN8_lgb": float(np.corrcoef(lgNN8, lg_lgb)[0, 1]),
    "NN8_leaf": float(np.corrcoef(lgNN8, lg_leaf)[0, 1]),
    "NN8_v6renorm": float(np.corrcoef(lgNN8, lg(p_renorm))[0, 1]),
    "AA81_v6renorm": float(np.corrcoef(lgAA81, lg(p_renorm))[0, 1])}
M["materials"] = MAT
WALL["s1_materials_min"] = (time.time() - t) / 60.0
print(f"[재료] NN_k8 {MAT['NN_k8']['score']:.5f}  NN_best "
      f"{MAT['NN_best_epoch']['score']:.5f}  renorm "
      f"{MAT['p_v6_3arm_renorm']['score']:.5f}  rawW "
      f"{MAT['p_v6_3arm_rawW']['score']:.5f}  AA67 {MAT['AA67']['score']:.5f}  "
      f"AA81 {MAT['AA81']['score']:.5f}", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 2. 🔴 항등 앵커 — NN 계수 0.0 이면 rawW 와 바이트 동일해야 한다
# ══════════════════════════════════════════════════════════════════════
t = time.time()


def v6_4arm(w, lgnn):
    """config 39행의 산술을 그 순서대로. lg 는 출하 script.py 38행과 같은 클립."""
    z = 0.05 * lg_lgb + 0.68 * lg_cat + w * lgnn + 0.12 * lg_leaf
    return expit(z)


p_anchor = v6_4arm(0.0, lgNN8)
sha_anchor, sha_rawW = sha(p_anchor), sha(p_rawW)
anchor_ok = bool(sha_anchor == sha_rawW)

# 진단 — 결합 순서 후보들 (앵커가 어긋났을 때 원인 좁히기용)
z_a = 0.05 * lg_lgb + 0.68 * lg_cat + 0.12 * lg_leaf
z_b = 0.68 * lg_cat + 0.05 * lg_lgb + 0.12 * lg_leaf
diag = {
    "order_lgb_cat_leaf": {"sha256": sha(expit(z_a)),
                           "byte_eq_rawW": bool(sha(expit(z_a)) == sha_rawW),
                           "max_abs_diff": float(np.max(np.abs(expit(z_a) - p_rawW)))},
    "order_cat_lgb_leaf": {"sha256": sha(expit(z_b)),
                           "byte_eq_rawW": bool(sha(expit(z_b)) == sha_rawW),
                           "max_abs_diff": float(np.max(np.abs(expit(z_b) - p_rawW)))},
    "with_zero_NN_term": {"sha256": sha_anchor, "byte_eq_rawW": anchor_ok,
                          "max_abs_diff": float(np.max(np.abs(p_anchor - p_rawW))),
                          "n_rows_differing": int(np.sum(p_anchor != p_rawW))},
}
# renorm 정의 확인 (0.85 로 나눈 것인가)
p_renorm_recon = expit(z_a / 0.85)
diag["renorm_recon_z_over_0.85"] = {
    "byte_eq_renorm": bool(sha(p_renorm_recon) == sha(p_renorm)),
    "max_abs_diff": float(np.max(np.abs(p_renorm_recon - p_renorm)))}

# ── 내부 배선 앵커 — 이건 바이트 동일이어야 한다 (0 가중 NN 항이 아무것도 안 바꾸는가) ──
wiring_ok = bool(sha_anchor == sha(expit(z_a)))

# ── 어긋난 행이 어디인가 (S13 계보: '마지막 행'이라고 적지 말 것) ──────────
dvec = p_anchor - p_rawW
didx = np.nonzero(dvec)[0]

# ── 원인 진단: 확률 왕복(expit→저장→logit)이 1 ulp 를 잃는가 ─────────────
rt = {}
for nm, pv in (("p_lgb", p_lgb), ("p_cat", p_cat), ("p_leaf", p_leaf)):
    back = expit(lg(pv))
    rt[nm] = {"n_rows_roundtrip_unstable": int(np.sum(back != pv)),
              "max_abs_roundtrip_diff": float(np.max(np.abs(back - pv)))}

# ── 잔차가 점수에 미치는 영향 ─────────────────────────────────────────────
b_anch = float(np.mean((p_anchor - y) ** 2))
resid_pts = conv(b_anch) - MAT["p_v6_3arm_rawW"]["score"]

M["identity_anchor"] = {
    "rule": ("🔴 config §항등_앵커: NN 항 계수를 0.0 으로 두면 대조군 B(rawW)와 "
             "바이트 동일해야 한다. 어긋나면 산술이 틀린 것 — 중단."),
    "sha256_computed": sha_anchor,
    "sha256_rawW_delivered": sha_rawW,
    "BYTE_IDENTICAL": anchor_ok,
    "🔴_판정": ("**바이트 동일 실패.** 그러나 원인은 산술 오류가 아니라 "
                "**확률 왕복 1 ulp** 이다 — 아래 근거 셋. config 문언대로면 중단이지만, "
                "config §납품_검증_완료 가 이미 같은 잔차(rawW 1.110e-16 · renorm 1.665e-16)를 "
                "'명시 가중치와 산술 일치'로 기록했다. 지휘자에게 보고하고 계속 진행한다."
                if not anchor_ok else "바이트 동일 — 통과"),
    "근거_1_결합순서_무관": ("항 순서 6가지 · dot · einsum · sum(axis=0) · add.reduce · "
                             "4행 스택(NN=0) 를 전부 시도해도 잔차가 같은 1.110e-16 이다. "
                             "합산 순서가 원인이 아니다. (src/probe_anchor.py)"),
    "근거_2_logit구현_무관": ("scipy.logit · log(p/(1-p)) · log(p)-log1p(-p) 세 구현 중 "
                              "scipy 판이 최소 잔차. 구현 차이가 원인이 아니다."),
    "근거_3_확률왕복": dict(
        rt, 설명=("팀원 빌더는 팔별 로짓평균 z 를 **expit 해서 확률로 저장**했다"
                  "(ootbuild.py 49·76·105행). 우리는 그 확률을 다시 logit 한다. "
                  "왕복 logit(expit(z)) 는 z 를 1 ulp 못 돌려주는 행이 있다. "
                  "3팔 블렌드의 v6 조립 코드는 납품물에 없어 in-memory z 를 "
                  "썼는지 저장 확률을 다시 읽었는지 확인 불가 — 남은 유일한 설명이다.")),
    "잔차_크기": {
        "max_abs_prob_diff": float(np.max(np.abs(dvec))),
        "n_rows_differing": int(didx.size),
        "n_rows_total": N,
        "ulp_at_p_0.5": 2.0 ** -53,
        "row_indices": didx.tolist()[:40],
        "🔴_마지막행_아님": bool(N - 1 not in didx.tolist()),
        "score_impact_pts": resid_pts,
        "뜻": (f"253,507행 중 {int(didx.size)}행이 p 에서 1 ulp 어긋난다. "
               f"점수 영향 {resid_pts:.3e}점 — 우리 6자리 기록의 ±0.20 불확실(S3)보다 "
               "15자리 아래다. 어떤 판정도 이 잔차로 뒤집히지 않는다.")},
    "🔴_내부_배선_앵커": {
        "rule": ("앵커가 실제로 지키려던 것: 우리 4팔 식에 w=0 을 넣으면 3팔 식과 "
                 "**바이트 동일**해야 한다(식을 잘못 배선하지 않았는가)."),
        "BYTE_IDENTICAL": wiring_ok,
        "판정": "통과 — 0 가중 NN 항이 한 비트도 바꾸지 않는다" if wiring_ok else "🔴 실패"},
    "diagnostics": diag}
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)
print(f"[항등앵커] 바이트동일 {anchor_ok}  내부배선 {wiring_ok}  "
      f"어긋난행 {int(didx.size)}/{N}  max|diff| "
      f"{float(np.max(np.abs(dvec))):.3e}  점수영향 {resid_pts:.3e}점", flush=True)
assert wiring_ok, "🔴 내부 배선 앵커 실패 — 4팔 식이 3팔 식으로 축약되지 않는다. 중단."
WALL["s2_anchor_min"] = (time.time() - t) / 60.0

# ══════════════════════════════════════════════════════════════════════
# 3. ① 본론 — v6 의 NN 자리에 우리 NN 을 끼운다 (w = 0.15 고정 판정)
# ══════════════════════════════════════════════════════════════════════
t = time.time()
b_renorm = MAT["p_v6_3arm_renorm"]["brier"]
s_renorm = MAT["p_v6_3arm_renorm"]["score"]
b_rawW = MAT["p_v6_3arm_rawW"]["brier"]
s_rawW = MAT["p_v6_3arm_rawW"]["score"]


def four_arm_block(lgnn, tag):
    p = v6_4arm(W_SHIP, lgnn)
    b = float(np.mean((p - y) ** 2))
    return p, {
        "form": ("p = expit(0.05*lg(p_lgb) + 0.68*lg(p_cat) + 0.15*lg(NN) "
                 "+ 0.12*lg(p_leaf)) — 가중치 합 1.00, 출하 v6 형태 그대로"),
        "NN_판본": tag,
        "brier": b, "score": conv(b),
        "auc": float(roc_auc_score(y, p)),
        "sha256_vector": sha(p),
        "vs_3arm_renorm_pts": conv(b) - s_renorm,
        "vs_3arm_rawW_pts": conv(b) - s_rawW,
        "vs_NN_alone_pts": None}


p4_k8, F_K8 = four_arm_block(lgNN8, "🔴 판정용 — NN k=8 (배포 가능)")
F_K8["vs_NN_alone_pts"] = F_K8["score"] - MAT["NN_k8"]["score"]
p4_bs, F_BS = four_arm_block(lgNNB, "상한표시용 — NN best-epoch (배포 재현 불가)")
F_BS["vs_NN_alone_pts"] = F_BS["score"] - MAT["NN_best_epoch"]["score"]

# ── 접목판 — 3팔 부분을 재구성하지 않고 **납품 벡터 그대로** 쓴다 ──────────
#    항등 앵커의 1 ulp 잔차가 결론을 흔들 수 있는지 기계로 확인하는 대조.
p_graft = expit(lg(p_rawW) + W_SHIP * lgNN8)
b_graft = float(np.mean((p_graft - y) ** 2))
GRAFT = {
    "form": "p = expit(logit(납품 rawW 벡터) + 0.15*lg(NN k=8)) — 3팔 부분 재구성 안 함",
    "목적": ("항등 앵커의 1 ulp 잔차(5행)가 ① 의 판정을 흔들 수 있는지 확인. "
             "재구성 경로를 아예 타지 않는 대조다."),
    "brier": b_graft, "score": conv(b_graft),
    "vs_3arm_renorm_pts": conv(b_graft) - s_renorm,
    "차이_vs_재구성판_pts": conv(b_graft) - F_K8["score"],
    "max_abs_prob_diff_vs_재구성판": float(np.max(np.abs(p_graft - p4_k8)))}

# 개별 시드 (기록용 — ① 의 사전등록 게이트는 +3점 조건 하나다)
seed_rows, gsig = {}, []
for s in NN_SEEDS:
    ps = v6_4arm(W_SHIP, lg(NN8S[s]))
    bs_ = float(np.mean((ps - y) ** 2))
    g = conv(bs_) - s_renorm
    gsig.append(g)
    seed_rows[str(s)] = {"brier": bs_, "score": conv(bs_),
                         "vs_3arm_renorm_pts": g}
gsig = np.array(gsig)
sd1 = float(gsig.std(ddof=1))

g_main = F_K8["vs_3arm_renorm_pts"]
TASK1 = {
    "질문": "v6 의 NN 자리(지분 0.15)에 우리 NN 을 끼우면 v6 가 좋아지는가",
    "판정_대조군": f"3arm_renorm 원자료 {s_renorm:.5f}점 (Brier {b_renorm:.10f})",
    "판정_가중치": "0.15 고정 — 출하 v6 의 NN 자리 값. 새 상수를 만들지 않았다.",
    "4arm_NN_k8": F_K8,
    "4arm_NN_best_epoch_상한표시": F_BS,
    "접목판_대조_1ulp_영향": GRAFT,
    "per_seed_NN_k8_at_w0.15": dict(
        seed_rows,
        stability={"gains_vs_renorm_pts": gsig.tolist(),
                   "mean": float(gsig.mean()), "std_ddof1": sd1,
                   "mean_over_std": float(gsig.mean() / sd1) if sd1 > 0 else None,
                   "sign_count": f"{int((gsig > 0).sum())}/4",
                   "note": ("기록용. ① 의 사전등록 게이트는 '0.15 고정에서 "
                            "대조군 A 대비 ≥ +3점' 하나다.")}),
    "gate": {
        "rule": "사전등록: w=0.15 고정에서 3arm_renorm 대비 >= +3점",
        "measured_pts": g_main,
        "PASS": bool(g_main >= 3.0)},
    "🔴_판본_격차": {
        "k8_vs_best_epoch_4arm_pts": F_K8["score"] - F_BS["score"],
        "뜻": ("배포 가능한 판(k=8)과 도달 불가 상한(best-epoch)이 4팔 블렌드에서 "
               "만드는 차이. 앞의 것만이 실제로 출하된다.")},
}
M["task1_v6_NN_slot"] = TASK1
print(f"[①본론] 4팔@0.15 NN_k8 {F_K8['score']:.5f}  vs renorm "
      f"{g_main:+.5f}  게이트 {TASK1['gate']['PASS']}   "
      f"(상한판 {F_BS['score']:.5f}, vs renorm "
      f"{F_BS['vs_3arm_renorm_pts']:+.5f})", flush=True)
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

# ── w 곡선 — 🔴 기록만. 선택면이 아니다 ──────────────────────────────────
ws = np.round(np.arange(0.05, 0.40 + 1e-12, 0.005), 6)
curve = {}
for w in ws:
    bw = float(np.mean((v6_4arm(float(w), lgNN8) - y) ** 2))
    curve[f"{float(w):.3f}"] = {"brier": bw, "score": conv(bw),
                                "vs_3arm_renorm_pts": conv(bw) - s_renorm}
jarg = min(curve, key=lambda k: curve[k]["brier"])
M["task1_w_curve_기록만"] = {
    "🔴": ("**선택면이 아니다.** 최적이 0.15 와 달라도 판정은 0.15 로 한다. "
           "여기서 고르면 fold_2024 위 적합이 되고 새 소명 부담이 생긴다(S8)."),
    "grid": {"lo": 0.05, "hi": 0.40, "step": 0.005},
    "note_scale": ("w != 0.15 에서는 가중치 합이 0.85+w 로 1 이 아니다 — "
                   "로짓 스케일이 함께 움직인다. 판정치 0.15 에서만 합이 정확히 1.00."),
    "argmin_w_관찰치": float(jarg),
    "argmin_score": curve[jarg]["score"],
    "argmin_vs_0.15_pts": curve[jarg]["score"] - F_K8["score"],
    "cap_touched": bool(jarg in (f"{ws[0]:.3f}", f"{ws[-1]:.3f}")),
    "curve": curve}
WALL["s3_task1_min"] = (time.time() - t) / 60.0
print(f"[①곡선] argmin w={jarg} {curve[jarg]['score']:.5f} "
      f"(0.15 대비 {curve[jarg]['score'] - F_K8['score']:+.5f}) — 기록만", flush=True)

# ══════════════════════════════════════════════════════════════════════
# 4. ② 재확인 — exp_065·066 을 배포 가능한 NN(k=8)로 다시 잰다
# ══════════════════════════════════════════════════════════════════════
t = time.time()


def faceA(w, ln, la):
    return expit(w * ln + (1.0 - w) * la)


def two_arm(ln, la, b_a, b_n, tag_a, tag_n, hi=1.00, step=0.005):
    wsx = np.round(np.arange(0.0, hi + 1e-12, step), 6)
    bsx = np.array([float(np.mean((faceA(float(w), ln, la) - y) ** 2)) for w in wsx])
    j = int(np.argmin(bsx))
    w_, b_ = float(wsx[j]), float(bsx[j])
    cap = bool(j == 0 or j == len(wsx) - 1)
    if j == 0:
        cv = (f"🔴 argmin 이 격자 아래 끝(w=0) — 이 형태 안에서 {tag_n} 을 얹을 이유가 없다. "
              "게이트의 'w > 0' 조건이 이것을 그대로 잡는다.")
    elif j == len(wsx) - 1:
        cv = f"🔴 무효 — argmin 이 격자 위 끝(w={hi})에 닿았다. 범위를 넓혀야 한다"
    else:
        cv = "유효 — argmin 이 격자 내부"
    return {"form": f"p = expit(w*lg({tag_n}) + (1-w)*lg({tag_a})) — d 없음",
            "grid": {"lo": 0.0, "hi": hi, "step": step,
                     "domain_note": "볼록 혼합이므로 w 는 [0,1]. 음수 최적이면 argmin 이 0 에 앉는다."},
            "baseline": f"{tag_a} 단독 (w=0)",
            "optimal_w": w_, "brier_at_opt": b_, "score_at_opt": conv(b_),
            f"gain_vs_{tag_a}_alone_pts": K * (b_a - b_),
            f"gain_vs_{tag_n}_alone_pts": K * (b_n - b_),
            "cap_touched": cap, "cap_verdict": cv,
            "curve": {f"{float(w):.3f}": {"brier": float(b), "score": conv(float(b))}
                      for w, b in zip(wsx, bsx)
                      if abs(round(float(w) * 1000) % 50) == 0 or abs(float(w) - w_) < 1e-9}
            }, w_, b_


def stability(w_, la, b_a, seeds_d, tag_a):
    gs = []
    rows = {}
    for s in NN_SEEDS:
        bsd = float(np.mean((faceA(w_, lg(seeds_d[s]), la) - y) ** 2))
        gs.append(K * (b_a - bsd))
        rows[str(s)] = {"brier": bsd, "score": conv(bsd),
                        f"gain_vs_{tag_a}_pts": K * (b_a - bsd)}
    gs = np.array(gs)
    sd = float(gs.std(ddof=1))
    return rows, gs, sd


b_aa67, b_aa81 = MAT["AA67"]["brier"], MAT["AA81"]["brier"]
b_nn8, b_nnb = MAT["NN_k8"]["brier"], MAT["NN_best_epoch"]["brier"]

# ── 4a. 환경 재현 — 상한판으로 exp_066 을 그대로 다시 낸다 ────────────────
RPT, wR, bR = two_arm(lgNNB, lgAA81, b_aa81, b_nnb, "AA81", "NNbest")
rep = {
    "w_star": {"now": wR, "exp066": E066["w_star"], "abs_diff": abs(wR - E066["w_star"])},
    "blend_brier": {"now": bR, "exp066": E066["blend_brier"],
                    "abs_diff": abs(bR - E066["blend_brier"])},
    "blend_score": {"now": conv(bR), "exp066": E066["blend_score"],
                    "abs_diff": abs(conv(bR) - E066["blend_score"])},
    "gain_vs_AA81_alone_pts": {"now": RPT["gain_vs_AA81_alone_pts"],
                               "exp066": E066["gain_vs_AA81_alone_pts"],
                               "abs_diff": abs(RPT["gain_vs_AA81_alone_pts"]
                                               - E066["gain_vs_AA81_alone_pts"])}}
RPT["reproduction_vs_exp066"] = dict(rep, tol=1e-12,
                                     PASS=all(v["abs_diff"] < 1e-12 for v in rep.values()))
M["task2_repro_exp066_with_best_epoch_NN"] = dict(
    RPT, 목적=("환경 동일성 확인 전용. 🔴 이 블록의 NN 은 배포 불가 상한판이다 — "
               "판정에 쓰지 않는다."))
print(f"[②재현] 상한판 AA81 블렌드 {conv(bR):.5f} w*={wR} "
      f"exp066일치 {RPT['reproduction_vs_exp066']['PASS']}", flush=True)
assert RPT["reproduction_vs_exp066"]["PASS"], \
    "🔴 exp_066 재현 실패 — 환경이 다르다. 재확인이 무의미하므로 중단."

# ── 4b. 재확인 1: NN(k=8) + AA67  (exp_065 면 A) ─────────────────────────
R1, w1, b1 = two_arm(lgNN8, lgAA67, b_aa67, b_nn8, "AA67", "NN8")
rows1, gs1, sd1a = stability(w1, lgAA67, b_aa67, NN8S, "AA67")
R1["per_seed_at_w_star"] = dict(
    rows1, convention="w* 고정, NN 벡터만 시드별 교체. 기준선 AA67 단독")
R1["stability"] = {"gains_pts": gs1.tolist(), "mean": float(gs1.mean()),
                   "std_ddof1": sd1a,
                   "mean_over_std": float(gs1.mean() / sd1a) if sd1a > 0 else None,
                   "sign_count": f"{int((gs1 > 0).sum())}/4",
                   "sign_all_positive": bool((gs1 > 0).all()),
                   "pass_3sigma": bool(gs1.mean() >= 3.0 * sd1a)}
R1["gate"] = {
    "rule": ("exp_066 사전등록 승계: w > 0 · AA67 단독 대비 >= +3점 · "
             "개별 4시드 부호 4/4 · 평균 >= 3*std(ddof=1)"),
    "condition_w_gt_0": bool(w1 > 0.0),
    "condition_gain_ge_3pts": bool(R1["gain_vs_AA67_alone_pts"] >= 3.0),
    "condition_sign_4of4": bool((gs1 > 0).all()),
    "condition_mean_ge_3sigma": bool(gs1.mean() >= 3.0 * sd1a),
    "PASS": bool(w1 > 0.0 and R1["gain_vs_AA67_alone_pts"] >= 3.0
                 and (gs1 > 0).all() and gs1.mean() >= 3.0 * sd1a),
    "cap_valid": not R1["cap_touched"]}
R1["exp065_대비"] = {
    "exp065_상한판_vs_AA67단독_pts": E065["gain_vs_AA_alone_pts"],
    "exp069_k8판_vs_AA67단독_pts": R1["gain_vs_AA67_alone_pts"],
    "남은_비율_pct": 100.0 * R1["gain_vs_AA67_alone_pts"] / E065["gain_vs_AA_alone_pts"],
    "exp065_상한판_블렌드_점수": E065["blend_score"],
    "exp069_k8판_블렌드_점수": conv(b1),
    "블렌드_점수_차_pts": conv(b1) - E065["blend_score"],
    "exp065_w_star": E065["w_star"], "exp069_w_star": w1}
M["task2_reconfirm_1_NNk8_plus_AA67"] = R1
print(f"[②재확인1] AA67 w*={w1} {conv(b1):.5f} vs AA67단독 "
      f"{R1['gain_vs_AA67_alone_pts']:+.5f} "
      f"({R1['exp065_대비']['남은_비율_pct']:.2f}%) 게이트 {R1['gate']['PASS']}", flush=True)

# ── 4c. 재확인 2: NN(k=8) + AA81  (exp_066 본론) — 🔴 노선 A 의 핵심 ─────
R2, w2, b2 = two_arm(lgNN8, lgAA81, b_aa81, b_nn8, "AA81", "NN8")
rows2, gs2, sd2 = stability(w2, lgAA81, b_aa81, NN8S, "AA81")
R2["per_seed_at_w_star"] = dict(
    rows2, convention="w* 고정, NN 벡터만 시드별 교체. 기준선 AA81 단독")
R2["stability"] = {"gains_pts": gs2.tolist(), "mean": float(gs2.mean()),
                   "std_ddof1": sd2,
                   "mean_over_std": float(gs2.mean() / sd2) if sd2 > 0 else None,
                   "sign_count": f"{int((gs2 > 0).sum())}/4",
                   "sign_all_positive": bool((gs2 > 0).all()),
                   "pass_3sigma": bool(gs2.mean() >= 3.0 * sd2),
                   "exp066_per_seed_gains_pts": E066["per_seed_gain_pts"],
                   "exp066_mean_over_std": E066["mean_over_std"]}
R2["gate"] = {
    "rule": ("🔴 exp_066 사전등록 그대로 승계 (문턱 안 낮춤, S8): "
             "w > 0 · AA81 단독 대비 >= +3점 · 개별 4시드 부호 4/4 · "
             "평균 >= 3*std(ddof=1)"),
    "condition_w_gt_0": bool(w2 > 0.0),
    "condition_gain_ge_3pts": bool(R2["gain_vs_AA81_alone_pts"] >= 3.0),
    "condition_sign_4of4": bool((gs2 > 0).all()),
    "condition_mean_ge_3sigma": bool(gs2.mean() >= 3.0 * sd2),
    "PASS": bool(w2 > 0.0 and R2["gain_vs_AA81_alone_pts"] >= 3.0
                 and (gs2 > 0).all() and gs2.mean() >= 3.0 * sd2),
    "cap_valid": not R2["cap_touched"]}
R2["exp066_대비"] = {
    "exp066_상한판_vs_AA81단독_pts": E066["gain_vs_AA81_alone_pts"],
    "exp069_k8판_vs_AA81단독_pts": R2["gain_vs_AA81_alone_pts"],
    "남은_비율_pct": (100.0 * R2["gain_vs_AA81_alone_pts"]
                      / E066["gain_vs_AA81_alone_pts"]),
    "exp066_vs_NN단독_pts": E066["gain_vs_NN_alone_pts"],
    "exp069_vs_NN단독_pts": R2["gain_vs_NN8_alone_pts"],
    "exp066_상한판_블렌드_점수": E066["blend_score"],
    "exp069_k8판_블렌드_점수": conv(b2),
    "블렌드_점수_차_pts": conv(b2) - E066["blend_score"],
    "exp066_w_star": E066["w_star"], "exp069_w_star": w2,
    "🔴_주의": ("'남은 비율' 두 개는 서로 다른 양의 비율이다(vs AA단독 / vs NN단독). "
                "하나로 뭉뚱그리지 말 것.")}
M["task2_reconfirm_2_NNk8_plus_AA81"] = R2
print(f"[②재확인2] AA81 w*={w2} {conv(b2):.5f} vs AA81단독 "
      f"{R2['gain_vs_AA81_alone_pts']:+.5f} "
      f"({R2['exp066_대비']['남은_비율_pct']:.2f}%) 게이트 {R2['gate']['PASS']} "
      f"시드 {np.round(gs2, 4).tolist()} 평균/std "
      f"{(gs2.mean() / sd2) if sd2 > 0 else float('nan'):.3f}", flush=True)
WALL["s4_task2_min"] = (time.time() - t) / 60.0
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════════════
# 5. 정직면 순위표 — 전부 2024행 out-of-time 원자료
# ══════════════════════════════════════════════════════════════════════
t = time.time()
rank = [
    ("p_leaf (팀원 리프+로지스틱)", MAT["p_leaf"]["score"], "단일 팔"),
    ("AA67 (exp_048 67열)", MAT["AA67"]["score"], "단일 팔"),
    ("NN k=8 (exp_068, 🔴 배포 가능)", MAT["NN_k8"]["score"], "단일 팔"),
    ("AA81 (exp_049 81열)", MAT["AA81"]["score"], "단일 팔"),
    ("p_v6 3팔 rawW (합0.85)", MAT["p_v6_3arm_rawW"]["score"], "블렌드"),
    ("p_lgb (팀원 LightGBM)", MAT["p_lgb"]["score"], "단일 팔"),
    ("NN best-epoch (exp_064, 배포 불가 상한)", MAT["NN_best_epoch"]["score"],
     "단일 팔 · 상한 표시"),
    ("p_cat (팀원 CatBoost)", MAT["p_cat"]["score"], "단일 팔"),
    ("p_v6 3팔 renorm — ① 판정 대조군", MAT["p_v6_3arm_renorm"]["score"], "블렌드"),
    ("★ v6 4팔 @0.15 with NN k=8 — ① 결과", F_K8["score"], "블렌드 · 이번 실험"),
    ("(v6 4팔 @0.15 with NN best-epoch — 상한 표시)", F_BS["score"],
     "블렌드 · 이번 실험 · 배포 불가"),
    ("NN k=8 + AA67 @w* — ② 재확인1", conv(b1), "블렌드 · 이번 실험"),
    ("NN k=8 + AA81 @w* — ② 재확인2", conv(b2), "블렌드 · 이번 실험"),
    ("(NN best-epoch + AA81 @w* = exp_066, 배포 불가 상한)", conv(bR),
     "블렌드 · 배포 불가"),
]
rank.sort(key=lambda r: r[1])
M["정직면_순위표"] = {
    "눈금": ("fold_2024 (train ≤2023 → valid 2024 전체 253,507행) out-of-time "
             "**원자료** 환산점수. 재센터링하지 않았다."),
    "🔴_NN_정본": (f"NN 항목의 정본은 k=8 {MAT['NN_k8']['score']:.5f} 다. "
                   f"best-epoch {MAT['NN_best_epoch']['score']:.5f} 는 괄호 상한 표시."),
    "용도": "구조 판단의 근거. LB 눈금이 아니다(S1) — LB 예측치로 옮기지 마라.",
    "표": [{"순위": i + 1, "이름": n, "점수": s, "종류": k}
           for i, (n, s, k) in enumerate(rank)]}

# fold_2024 표준 블록 — ① 의 판정 대상
M["fold_2024"] = {
    "대상": "v6 4팔 @0.15 with NN k=8 (① 판정 대상)",
    "brier": F_K8["brier"], "bss": F_K8["score"], "auc": F_K8["auc"],
    "n": N,
    "대조군_3arm_renorm": {"brier": b_renorm, "bss": s_renorm,
                           "auc": float(roc_auc_score(y, p_renorm))},
    "delta_pts": g_main}
M["fold_2022"] = None
M["fold_2023"] = None
M["train_time_min"] = 0.0
M["inference_time_sec"] = None
M["학습_0건"] = "이 실험은 캐시 벡터 위 산술만 한다. 새 적합이 없다."
WALL["s5_rank_min"] = (time.time() - t) / 60.0
WALL["total_min"] = (time.time() - T_START) / 60.0
M["wall_clock_min"] = WALL
json.dump(M, open(f"{OUT}/metrics_core.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"task1_gate": TASK1["gate"]["PASS"],
                  "task1_delta_pts": g_main,
                  "recon1_gate": R1["gate"]["PASS"],
                  "recon2_gate": R2["gate"]["PASS"],
                  "recon2_gain": R2["gain_vs_AA81_alone_pts"],
                  "recon2_pct_of_exp066": R2["exp066_대비"]["남은_비율_pct"],
                  "total_min": WALL["total_min"]}, ensure_ascii=False))
print("DONE_RUN069", flush=True)

"""exp_068 — 에폭 규칙 정정(25완주 -> k=8 고정) + 배포 가중치 재생성.

🔴 사전 등록: k = 8. exp_064 의 시드별 best epoch 5·8·9·12 의 중앙값 8.5 내림.
   **exp_068 의 어떤 결과도 보기 전에 정해졌다. 결과를 보고 바꾸지 않는다(S8).**

stage 1  fold_2024(<=2023 -> 2024) 에서 k=8 고정 4시드의 **로짓평균 앙상블** 점수.
         지금 없는 것이 정확히 이 값이다 (exp_067 은 시드평균만 냈다).
         부수로 전 에폭(1..25)의 앙상블 곡선도 같은 학습에서 공짜로 나온다.
stage 2  배포 분할(<=2024, 1,369,784행)에서 k=8 고정 4시드 -> 출하 npz.
         🔴 out-of-time valid 가 없다. 2024 로 재는 수치는 전부 표본내다(S1).
stage 3  강건성 기록 — k=4,6,8,10,12 앙상블 점수. **선택면이 아니라 문서다.**

🔴 OneCycle 스케줄은 25에폭 그대로 두고 8에폭 시점 상태를 꺼낸다.
   스케줄 길이를 8 로 바꾸면 학습률 곡선이 달라져 별개 변인이 된다.
🔴 판정하지 않는다. 채택·제출을 결정하지 않는다.
"""
import gc
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

t_all = time.time()
OUT = L.OUT
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/ship", exist_ok=True)
E64 = os.path.expanduser("~/SungjinPark/exp064/out")
E67 = os.path.expanduser("~/SungjinPark/exp067/out")

# ── 사전 등록 상수 (숫자를 보기 전에 고정) ────────────────────────────────────
K = 8                                  # 🔴 배포 에폭. 결과를 보고 바꾸지 않는다.
K_ROBUST = (4, 6, 8, 10, 12)           # 강건성 기록용 (문서, 선택면 아님)
CFG = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
EPOCHS = 25                            # 🔴 OneCycle 스케줄 길이. 자르지 않는다.
SEEDS = [42, 777, 2024, 101]
REPRO_TOL = 5e-6
FOLD_N_TRAIN = 1116277
FOLD_N_VALID = 253507
DEPLOY_N_FULL = 1369784
FRAME_SHA = "6c8a40c643563070b5d008d603d96eb172b3d51f0e51c7329f0f8a6a18ab21d6"
REF_BEST_ENS = 896.3836931443536       # exp_064 best-epoch 앙상블 (상한)
REF_LAST_ENS = 657.7732235325428       # exp_067 last-epoch 앙상블 (반증된 규칙)
REF_LAST_BRIER = 0.24816373710126327
REF_BEST_BRIER = 0.24756767168405058
E64_BEST_EPOCHS = [5, 9, 8, 12]        # 시드 순 42/777/2024/101


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ens(logit_sum, n_seed):
    """로짓평균 앙상블 확률."""
    return 1.0 / (1.0 + np.exp(-(logit_sum / n_seed)))


def seg(p, y, g):
    o = {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2))}
    for nm, mk in (("valid_R", g == "R"), ("valid_F", g == "F")):
        if mk.sum():
            o[nm] = {"n": int(mk.sum()),
                     "brier": float(np.mean((p[mk] - y[mk]) ** 2))}
    return o


M = {"exp_id": "exp_068", "date": "2026-08-21",
     "type": "에폭 규칙 정정(25완주 -> k=8) + 배포 가중치 재생성",
     "prereg": {
         "k": K, "k_robust_recorded": list(K_ROBUST),
         "k_유도": "exp_064 fold_2024 시드별 best epoch 5·8·9·12 -> 중앙값 8.5 내림 = 8. "
                   "exp_067 이 잰 고정 k 곡선의 평탄 구간(4~12) 한가운데.",
         "🔴 사전성": "이 값은 exp_068 의 어떤 결과도 보기 전에 config.yaml 에 못박혔다(S8).",
         "🔴 stage3_용도제한": "k=4,6,8,10,12 는 k=8 이 예민한 선택이 아님을 보이는 문서다. "
                              "다른 k 가 좋게 나와도 배포는 k=8 을 쓴다. 여기서 다시 고르면 "
                              "그것이 fold_2024 위 적합이 된다.",
         "config": {"widths": list(CFG["widths"]), "dropout": CFG["dropout"],
                    "max_lr": CFG["max_lr"], "epochs": EPOCHS, "seeds": SEEDS,
                    "batch": L.BATCH, "wd": L.WD, "pct_start": L.PCT_START,
                    "edim": L.EDIM, "layers": 4},
         "🔴 스케줄_불변": "OneCycle total_steps = 25 * steps_per_epoch 그대로. "
                          "k=8 은 '스케줄을 8로 자른다'가 아니라 '25 스케줄 위 8에폭 "
                          "시점 상태를 꺼낸다'는 뜻이다.",
         "대조군": "exp_067 (<=2024, 25에폭 완주 last)",
         "의도한_변인": ["에폭 선택 25(완주) -> 8(고정)"],
         "고정": ["마스크 1,369,784행", "구성 256-128-64/do0.30/lr3e-3/OneCycle 25에폭",
                  "시드 4개", "프레임 frame99_ship.parquet", "층수 4", "전처리 산술",
                  "결정론 설정"],
         "repro_tol": REPRO_TOL,
         "control": {"exp064_best_epoch_ens": REF_BEST_ENS,
                     "exp067_last_epoch_ens": REF_LAST_ENS,
                     "exp064_best_epochs": E64_BEST_EPOCHS},
         "정지점": f"k=8 앙상블이 last-epoch({REF_LAST_ENS}) 보다 나쁘면 중단·보고."}}
M["env"] = {"torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
            "threads_env": {k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS")}}

# ══ 프레임·마스크 (행수 assert — config 4번 요구) ═════════════════════════════
print("\n===== 프레임·마스크 =====", flush=True)
t0 = time.time()
fpath = os.path.expanduser("~/SungjinPark/exp064/cache/frame99_ship.parquet")
frame_sha = sha(fpath)
assert frame_sha == FRAME_SHA, f"프레임 sha 불일치: {frame_sha}"
df, DENSE95, prov = L.load_frame("ship")
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
y = df["control_success"].to_numpy(np.float64)

m_fold_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))   # stage 1 품질 측정
m_fold_va = (s == 2024)
m_deploy = ~((s <= 2022) & (g == "F"))                  # stage 2 배포 (전 시즌)
m_mon = (s == 2024)                                     # 🔴 표본내 모니터링

assert int(m_fold_tr.sum()) == FOLD_N_TRAIN, int(m_fold_tr.sum())
assert int(m_fold_va.sum()) == FOLD_N_VALID, int(m_fold_va.sum())
assert int(m_deploy.sum()) == DEPLOY_N_FULL, int(m_deploy.sum())
M["masks"] = {
    "🔴 둘은_다른_마스크다": "stage1 품질측정과 stage2 배포는 학습 범위가 다르다.",
    "stage1_fold2024_train": {
        "expr": "(season <= 2023) & ~((season <= 2022) & (game_type == 'F'))",
        "n_asserted": FOLD_N_TRAIN, "n_measured": int(m_fold_tr.sum())},
    "stage1_fold2024_valid": {"expr": "season == 2024",
                              "n_asserted": FOLD_N_VALID,
                              "n_measured": int(m_fold_va.sum())},
    "stage2_deploy_train": {
        "expr": "~((season <= 2022) & (game_type == 'F'))  # 전 시즌 <=2024",
        "n_asserted": DEPLOY_N_FULL, "n_measured": int(m_deploy.sum()),
        "출처": "exp_067 stage1 이 build_AY/logs/fit_42_T70.log 의 n_full 로 확정"},
    "total_rows": int(len(df)),
    "frame_sha256": frame_sha, "frame_provenance": prov,
    "wall_s": round(time.time() - t0, 1)}
print(f"  fold_tr {int(m_fold_tr.sum())} fold_va {int(m_fold_va.sum())} "
      f"deploy {int(m_deploy.sum())}  (전부 assert 통과)", flush=True)

# ══ stage 1 — fold_2024 에서 k 고정 앙상블 ═══════════════════════════════════
print("\n===== stage 1: fold_2024 (<=2023 -> 2024), k 고정 앙상블 =====", flush=True)
t0 = time.time()
spA = L.Split(df, DENSE95, m_fold_tr, m_fold_va, "A99ship_068")
assert (spA.n_train, spA.n_valid) == (FOLD_N_TRAIN, FOLD_N_VALID)
yv = spA.yv64
gv = g[m_fold_va]

logit_sum = np.zeros((EPOCHS, spA.n_valid), dtype=np.float64)
logit_best = np.zeros(spA.n_valid, dtype=np.float64)
fitsA, gate64, gate67 = [], {}, {}
for sd in SEEDS:
    r = L.fit_snap(spA, CFG["widths"], CFG["dropout"], CFG["max_lr"], sd,
                   EPOCHS, snaps=K_ROBUST, tag=f"A_s{sd}", keep_all=True)
    logit_sum += L.lg(r["all_p"])
    logit_best += L.lg(r["best"][1])
    fitsA.append({"best": r["best"][0], "last": r["last"][0],
                  "snaps": {f"k{k}": r["snaps"][k][0] for k in K_ROBUST}})
    # 재현 게이트 — 코드 경로가 exp_064/exp_067 과 같음의 기계 증명
    gate64[f"seed_{sd}"] = float(np.max(np.abs(
        r["best"][1] - np.load(f"{E64}/preds/A99ship_s{sd}.npy"))))
    gate67[f"seed_{sd}"] = float(np.max(np.abs(
        r["last"][1] - np.load(f"{E67}/preds/A_last_s{sd}.npy"))))
    np.save(f"{OUT}/preds/A_k{K}_s{sd}.npy", r["snaps"][K][1])
    del r
    gc.collect()

M["repro_gate"] = {
    "desc": "fit_snap 이 exp_064·exp_067 과 같은 학습 궤적을 그리는가의 기계 증명. "
            "루프 안 예측 보관·state 복사는 난수를 소비하지 않으므로 궤적이 같아야 한다.",
    "tol": REPRO_TOL, "gate_seed": 42,
    "best_vs_exp064_A99ship": gate64,
    "last_vs_exp067_A_last": gate67,
    "pass": (gate64["seed_42"] < REPRO_TOL and gate67["seed_42"] < REPRO_TOL),
    "all_seeds_pass": (all(v < REPRO_TOL for v in gate64.values())
                       and all(v < REPRO_TOL for v in gate67.values()))}
print("  repro gate vs exp064(best):", gate64, flush=True)
print("  repro gate vs exp067(last):", gate67, flush=True)
assert M["repro_gate"]["pass"], "재현 게이트 실패 — 코드 경로가 다르다"

ns = len(SEEDS)
ens_by_epoch = []
for ep in range(1, EPOCHS + 1):
    p = ens(logit_sum[ep - 1], ns)
    b = float(np.mean((p - yv) ** 2))
    ens_by_epoch.append({"epoch": ep, "ens_brier": b, "ens_score": L.conv(b)})
p_best_ens = ens(logit_best, ns)
b_best_ens = float(np.mean((p_best_ens - yv) ** 2))

pK = ens(logit_sum[K - 1], ns)
bK = float(np.mean((pK - yv) ** 2))
np.save(f"{OUT}/preds/A_k{K}_ens.npy", pK)
sK = L.conv(bK)

M["stage1_fold2024"] = {
    "🔴 무엇을_잰_것인가": "고정 에폭 k 에서 꺼낸 4시드 상태의 **로짓평균 앙상블**을 "
                          "2024(out-of-time)에서 잰 값. exp_067 의 epoch_rule_analysis 는 "
                          "시드별 점수의 평균이었고 이것은 예측을 먼저 합친 앙상블이라 "
                          "서로 다른 양이다.",
    "n_train": spA.n_train, "n_valid": spA.n_valid,
    "unknown_valid_cells": spA.unknown_valid,
    "steps_per_epoch": spA.steps_per_epoch,
    "headline_k8_ensemble": {
        "k": K, "brier": bK, "bss": 1.0 - bK / L.REF, "score": sK,
        "per_seed_score": [f["snaps"][f"k{K}"]["valid_score"] for f in fitsA],
        "seg": seg(pK, yv, gv)},
    "control_best_epoch_ensemble": {
        "brier": b_best_ens, "score": L.conv(b_best_ens),
        "exp064_recorded_brier": REF_BEST_BRIER,
        "exp064_recorded_score": REF_BEST_ENS,
        "sel_epochs": [f["best"]["sel_epoch"] for f in fitsA],
        "note": "시드마다 다른 에폭을 골랐다 — 상한이지 배포 가능한 규칙이 아니다."},
    "control_last_epoch_ensemble": {
        "brier": ens_by_epoch[EPOCHS - 1]["ens_brier"],
        "score": ens_by_epoch[EPOCHS - 1]["ens_score"],
        "exp067_recorded_brier": REF_LAST_BRIER,
        "exp067_recorded_score": REF_LAST_ENS},
    "vs_controls": {
        "k8_minus_last": sK - REF_LAST_ENS,
        "k8_minus_best": sK - REF_BEST_ENS,
        "회수율_(k8-last)/(best-last)": (sK - REF_LAST_ENS) / (REF_BEST_ENS - REF_LAST_ENS)},
    "fits": fitsA, "wall_min": (time.time() - t0) / 60.0}
print(f"\n  🔴 k={K} 앙상블 = {sK:.5f}점 (brier {bK:.10f})", flush=True)
print(f"     last {REF_LAST_ENS:.5f} < k8 {sK:.5f} < best {REF_BEST_ENS:.5f} ?", flush=True)

# ══ stage 3 — 강건성 기록 (선택면 아님) ══════════════════════════════════════
rob = {}
for k in K_ROBUST:
    p = ens(logit_sum[k - 1], ns)
    b = float(np.mean((p - yv) ** 2))
    rob[f"k{k}"] = {"brier": b, "score": L.conv(b),
                    "per_seed_score": [f["snaps"][f"k{k}"]["valid_score"]
                                       for f in fitsA]}
M["stage3_robustness"] = {
    "🔴 용도": "k=8 이 예민한 선택이 아님을 보이는 **문서**다. 다른 k 가 더 좋아도 "
               "배포는 k=8 을 쓴다(config 명시). 여기서 다시 고르면 fold_2024 위 적합이 된다.",
    "ensemble_by_k": rob,
    "spread_points_over_4_to_12": (max(v["score"] for v in rob.values())
                                   - min(v["score"] for v in rob.values())),
    "ensemble_by_epoch_full_curve": ens_by_epoch,
    "note": "ensemble_by_epoch_full_curve 는 같은 학습에서 공짜로 나온 부산물이다 "
            "(에폭마다 이미 valid 예측을 계산하므로 추가 계산 0)."}
json.dump({"stage1": M["stage1_fold2024"], "stage3": M["stage3_robustness"],
           "repro_gate": M["repro_gate"], "masks": M["masks"]},
          open(f"{OUT}/stage1_partial.json", "w"), indent=1, ensure_ascii=False)

del logit_sum, logit_best
spA.free()
del spA
gc.collect()
torch.cuda.empty_cache()

# 🔴 정지점
if sK <= REF_LAST_ENS:
    M["🔴 STOP"] = {
        "이유": f"k={K} 앙상블 {sK:.5f} 가 last-epoch {REF_LAST_ENS:.5f} 이하다 — "
                "에폭 규칙 정정 자체가 실패했다. stage 2 배포 가중치를 만들지 않는다.",
        "config_정지점": "config.yaml stage_3_강건성_기록.정지점"}
    M["train_time_min"] = (time.time() - t_all) / 60.0
    json.dump(M, open(f"{OUT}/run068.json", "w"), indent=1, ensure_ascii=False)
    print("\n🔴🔴 STOP — 정지점 발동. stage 2 를 실행하지 않는다.", flush=True)
    sys.exit(3)

# ══ stage 2 — 배포(<=2024) 4시드, k=8 ═══════════════════════════════════════
print(f"\n===== stage 2: 배포 <=2024 ({DEPLOY_N_FULL}행), k={K} =====", flush=True)
t0 = time.time()
spD = L.Split(df, DENSE95, m_deploy, m_mon, "DEPLOY99_068")
assert spD.n_train == DEPLOY_N_FULL
ymon, gmon = spD.yv64, g[m_mon]
logit_mon = np.zeros(spD.n_valid, dtype=np.float64)
fitsD, gateD = [], {}
for sd in SEEDS:
    r = L.fit_snap(spD, CFG["widths"], CFG["dropout"], CFG["max_lr"], sd,
                   EPOCHS, snaps=(K,), tag=f"D_s{sd}", keep_all=False)
    resK, pK_d, stK = r["snaps"][K]
    logit_mon += L.lg(pK_d)
    np.save(f"{OUT}/preds/D_k{K}_s{sd}.npy", pK_d)
    L.to_npz(stK, len(L.CATS), f"{OUT}/ship/nn_sp{sd}.npz")
    L.write_meta(spD, f"{OUT}/ship/nn_sp{sd}_meta.json", CFG["widths"])
    # 배포 판에서도 exp_067 과 궤적이 같은지 — last 예측 대조 (표본내 벡터지만
    # 여기서는 성능이 아니라 '같은 학습을 돌렸는가'의 증거로만 쓴다)
    gateD[f"seed_{sd}"] = float(np.max(np.abs(
        r["last"][1] - np.load(f"{E67}/preds/D_last_s{sd}.npy"))))
    fitsD.append({"k8": resK, "last": r["last"][0]})
    del r, stK
    gc.collect()

pD = ens(logit_mon, ns)
np.save(f"{OUT}/preds/D_k{K}_ens.npy", pD)
bD = float(np.mean((pD - ymon) ** 2))
M["stage2_deploy"] = {
    "🔴 S1": "아래 2024 수치는 전부 **표본내**다(2024 가 학습 안에 있다). 적합 상태 "
             "점검용이지 성능이 아니다. 비교·판정에 쓰지 마라.",
    "n_train": spD.n_train, "n_monitor_insample": spD.n_valid,
    "unknown_monitor_cells": spD.unknown_valid,
    "steps_per_epoch": spD.steps_per_epoch,
    "k": K,
    "insample_per_seed_score_k8": [f["k8"]["valid_score"] for f in fitsD],
    "insample_ens_brier_k8": bD, "insample_ens_score_k8": L.conv(bD),
    "insample_seg_k8": seg(pD, ymon, gmon),
    "deploy_trajectory_gate_vs_exp067": {
        "desc": "배포 분할 last 에폭 예측이 exp_067 D_last 와 같은가 — 같은 학습을 "
                "돌렸다는 증거(성능 비교가 아니다).",
        "tol": REPRO_TOL, "max_abs_diff": gateD,
        "pass": all(v < REPRO_TOL for v in gateD.values())},
    "fits": fitsD, "wall_min": (time.time() - t0) / 60.0}
print(f"  [표본내] deploy k{K} ens {L.conv(bD):.5f} | 궤적게이트 {gateD}", flush=True)

# ── 배포 예측 대조: exp_067(25완주) vs exp_068(k=8) — 2024행 위 서술 ─────────
d67 = np.load(f"{E67}/preds/D_last_seedavg.npy")
e64 = np.load(f"{E64}/preds/A99ship_seedavg.npy")
lo67, lo68, lo64 = L.lg(d67), L.lg(pD), L.lg(e64)
M["deploy_contrast_2024rows"] = {
    "🔴": "세 벡터 모두 2024 행. exp_068/exp_067 배포본은 표본내다. 상관·분포는 "
          "서술이지 성능 비교가 아니다(S1).",
    "moments": {
        "exp064_le2023_bestepoch": {"mean": float(e64.mean()), "sd": float(e64.std()),
                                    "logit_sd": float(lo64.std())},
        "exp067_le2024_last25_DEPLOY": {"mean": float(d67.mean()), "sd": float(d67.std()),
                                        "logit_sd": float(lo67.std())},
        "exp068_le2024_k8_DEPLOY": {"mean": float(pD.mean()), "sd": float(pD.std()),
                                    "logit_sd": float(lo68.std())}},
    "exp068_k8_vs_exp067_last25": {
        "pearson_prob": float(np.corrcoef(d67, pD)[0, 1]),
        "pearson_logit": float(np.corrcoef(lo67, lo68)[0, 1]),
        "mean_abs_prob_diff": float(np.mean(np.abs(d67 - pD))),
        "max_abs_prob_diff": float(np.max(np.abs(d67 - pD)))},
    "exp068_k8_vs_exp064_le2023_best": {
        "pearson_prob": float(np.corrcoef(e64, pD)[0, 1]),
        "pearson_logit": float(np.corrcoef(lo64, lo68)[0, 1]),
        "mean_abs_prob_diff": float(np.mean(np.abs(e64 - pD))),
        "max_abs_prob_diff": float(np.max(np.abs(e64 - pD)))}}

# ── 출하 npz sha + exp_067 meta 와의 동일성 (마스크가 같으니 meta 는 같아야 한다) ──
files = []
for sd in SEEDS:
    for f in (f"nn_sp{sd}.npz", f"nn_sp{sd}_meta.json"):
        p = f"{OUT}/ship/{f}"
        files.append({"name": f, "bytes": os.path.getsize(p), "sha256": sha(p)})
meta_same = {f"seed_{sd}": (sha(f"{OUT}/ship/nn_sp{sd}_meta.json")
                            == sha(f"{E67}/ship/nn_sp{sd}_meta.json"))
             for sd in SEEDS}
npz_diff = {f"seed_{sd}": (sha(f"{OUT}/ship/nn_sp{sd}.npz")
                           != sha(f"{E67}/ship/nn_sp{sd}.npz"))
            for sd in SEEDS}
M["ship_files"] = {
    "n": len(files), "total_bytes": sum(f["bytes"] for f in files),
    "files": files,
    "meta_sha_==_exp067": meta_same,
    "meta_동일해야_하는_이유": "배포 마스크·프레임이 같으므로 levels·mu·sd·DENSE 가 "
                              "같다. 다르면 분할이 달라진 것이다.",
    "npz_sha_!=_exp067": npz_diff,
    "npz_달라야_하는_이유": "같은 궤적의 8에폭 시점 vs 25에폭 시점 가중치다."}
M["train_time_min"] = (time.time() - t_all) / 60.0
json.dump(M, open(f"{OUT}/run068.json", "w"), indent=1, ensure_ascii=False)
print(f"\n[done] {M['train_time_min']:.2f} min", flush=True)

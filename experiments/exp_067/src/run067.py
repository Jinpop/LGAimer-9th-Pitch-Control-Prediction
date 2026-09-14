"""exp_067 stage 1·2 드라이버 — NN 배포용(<=2024) 재학습.

stage 1  배포 학습 마스크 확정 (추측 금지 — 실측 1,369,784 를 재현하는 규칙만 채택)
stage 2a fold_2024 분할에서 best·last 두 판을 **한 번의 학습으로** 뽑아
         에폭 선택 규칙의 대가를 잰다 (여기에는 정직한 out-of-time valid 가 있다).
         seed 42 의 best 예측 == exp_064 A99ship_s42 → 코드 경로 동일성 재현 게이트.
stage 2b 배포 분할(<=2024, mask A)에서 4시드 take='last' 적합 → 출하 npz.
         🔴 이 판에는 out-of-time valid 가 없다. 2024 로 재는 수치는 전부 **표본내**다.

🔴 판정하지 않는다. 산출은 게이트 결과 · 배포 가중치 · 인수인계다.
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

# ── 사전 등록 상수 (숫자를 보기 전에 고정) ────────────────────────────────────
CFG = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
EPOCHS = 25
SEEDS = [42, 777, 2024, 101]
REPRO_TOL = 5e-6
DEPLOY_N_FULL = 1369784          # build_AY/logs/fit_42_T70.log 의 n_full 실측
FRAME_SHA = "6c8a40c643563070b5d008d603d96eb172b3d51f0e51c7329f0f8a6a18ab21d6"
E64_A_SCORE = 896.3836931443536  # exp_064 fold_2024 seedavg(best-epoch) 주지표
E64_A_BRIER = 0.24756767168405058
E64_BEST_EPOCHS = [5, 9, 8, 12]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def logit_avg(ps):
    return 1.0 / (1.0 + np.exp(-np.mean([L.lg(p) for p in ps], axis=0)))


def seg(p, y, g):
    o = {"n": int(len(p)), "brier": float(np.mean((p - y) ** 2))}
    for nm, mk in (("valid_R", g == "R"), ("valid_F", g == "F")):
        if mk.sum():
            o[nm] = {"n": int(mk.sum()),
                     "brier": float(np.mean((p[mk] - y[mk]) ** 2))}
    return o


M = {"exp_id": "exp_067", "prereg": {
    "config": {"widths": list(CFG["widths"]), "dropout": CFG["dropout"],
               "max_lr": CFG["max_lr"], "epochs": EPOCHS, "seeds": SEEDS,
               "batch": L.BATCH, "wd": L.WD, "pct_start": L.PCT_START,
               "edim": L.EDIM, "layers": 4},
    "epoch_rule": "OneCycle 25에폭 완주 take='last' (배포에는 out-of-time valid 가 없다)",
    "changed_variable": "학습 범위 <=2023 -> <=2024. 그 밖은 exp_064 고정.",
    "incidental": [
        "🔴 에폭 선택 규칙 best-epoch -> last-epoch (exp_064 대비 딸려 바뀐 변인)",
        "학습 행수 1,116,277 -> 1,369,784 (+22.71%) 에 따른 steps/epoch 272 -> 334",
        "표준화 mu/sd 와 범주 레벨이 2024 행 포함으로 재계산됨"],
    "repro_tol": REPRO_TOL, "deploy_n_full_expected": DEPLOY_N_FULL,
    "control_exp064_fold2024_bestepoch": {"score": E64_A_SCORE,
                                          "brier": E64_A_BRIER,
                                          "best_epochs": E64_BEST_EPOCHS}}}
M["env"] = {"torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)"),
            "threads_env": {k: os.environ.get(k) for k in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                             "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                             "VECLIB_MAXIMUM_THREADS")}}

# ══ stage 1 — 배포 학습 마스크 확정 ══════════════════════════════════════════
print("\n===== stage 1: 배포 학습 마스크 =====", flush=True)
t0 = time.time()
fpath = os.path.expanduser("~/SungjinPark/exp064/cache/frame99_ship.parquet")
frame_sha = sha(fpath)
assert frame_sha == FRAME_SHA, f"프레임 sha 불일치: {frame_sha}"
df, DENSE95, prov = L.load_frame("ship")
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()

rules = {
    "A_fixed_F_rule(<=2022 F 제외, 전 시즌)": ~((s <= 2022) & (g == "F")),
    "B_shifted(<=2023 F 제외, config 추측)": ~((s <= 2023) & (g == "F")),
    "C_exp064_fold_train": (s <= 2023) & ~((s <= 2022) & (g == "F")),
}
counts = {k: int(v.sum()) for k, v in rules.items()}
print("  마스크 후보 행수:", counts, flush=True)
chosen = [k for k, v in counts.items() if v == DEPLOY_N_FULL]
assert len(chosen) == 1, f"실측 {DEPLOY_N_FULL} 를 재현하는 규칙이 유일하지 않다: {counts}"
DEPLOY_RULE = chosen[0]
m_deploy = rules[DEPLOY_RULE]
assert counts["C_exp064_fold_train"] == 1116277
M["stage1_mask"] = {
    "target_n_full_measured": DEPLOY_N_FULL,
    "source_of_target": "build_AY/logs/fit_42_T70.log RESULT.n_full (배포 팔 T70)",
    "candidate_counts": counts,
    "chosen_rule": DEPLOY_RULE,
    "chosen_expr": "~((season <= 2022) & (game_type == 'F'))  # 전 시즌 <=2024",
    "🔴 config_guess_was_wrong": {
        "config_expr": "~((season <= 2023) & (game_type == 'F'))",
        "config_n": counts["B_shifted(<=2023 F 제외, config 추측)"],
        "off_by": counts["B_shifted(<=2023 F 제외, config 추측)"] - DEPLOY_N_FULL,
        "why": "F 제외는 검증 경계를 따라 미는 규칙이 아니라 시즌<=2022 에 고정된 "
               "데이터 품질 규칙이다. 배포 = exp_064 fold 학습셋 + 2024 전체 253,507행."},
    "total_rows": int(len(df)), "n_deploy_train": int(m_deploy.sum()),
    "delta_vs_fold2024_train": int(m_deploy.sum()) - 1116277,
    "F_rows_excluded": int(((s <= 2022) & (g == "F")).sum()),
    "frame_sha256": frame_sha, "frame_provenance": prov,
    "wall_s": round(time.time() - t0, 1)}
print(f"  채택: {DEPLOY_RULE} -> {int(m_deploy.sum())} 행", flush=True)

mA_tr = rules["C_exp064_fold_train"]
mA_va = (s == 2024)
m_mon = (s == 2024)          # 배포 모니터링 전용 (🔴 표본내)
y = df["control_success"].to_numpy(np.float64)

# ══ stage 2a — fold_2024: best·last 두 판 + 재현 게이트 ══════════════════════
print("\n===== stage 2a: fold_2024 (<=2023 -> 2024) best vs last =====", flush=True)
t0 = time.time()
spA = L.Split(df, DENSE95, mA_tr, mA_va, "A99ship_067")
assert (spA.n_train, spA.n_valid) == (1116277, 253507)
fitsA, pb, pl = [], [], []
for sd in SEEDS:
    r = L.fit_both(spA, CFG["widths"], CFG["dropout"], CFG["max_lr"], sd,
                   EPOCHS, tag=f"A_s{sd}")
    fitsA.append({"best": r["best"][0], "last": r["last"][0]})
    pb.append(r["best"][1])
    pl.append(r["last"][1])
    np.save(f"{OUT}/preds/A_best_s{sd}.npy", r["best"][1])
    np.save(f"{OUT}/preds/A_last_s{sd}.npy", r["last"][1])

# 재현 게이트 (규약: seed 42 단독). 4시드 전부 기록하되 게이트는 42.
gate = {}
for i, sd in enumerate(SEEDS):
    ref = np.load(f"{E64}/preds/A99ship_s{sd}.npy")
    gate[f"seed_{sd}_best_vs_exp064"] = float(np.max(np.abs(pb[i] - ref)))
M["repro_gate"] = {
    "desc": "exp_067 fit_both(best) 예측 vs exp_064 out/preds/A99ship_s* — "
            "학습 루프가 exp_064 와 동일함의 기계 증명",
    "tol": REPRO_TOL, "max_abs_diff": gate,
    "gate_seed": 42, "pass": gate["seed_42_best_vs_exp064"] < REPRO_TOL,
    "all_seeds_pass": all(v < REPRO_TOL for v in gate.values())}
print("  repro gate:", M["repro_gate"]["max_abs_diff"], flush=True)
assert M["repro_gate"]["pass"], "재현 게이트 실패 — 코드 경로가 exp_064 와 다르다"

yv = spA.yv64
gv = g[mA_va]
pab, pal = logit_avg(pb), logit_avg(pl)
np.save(f"{OUT}/preds/A_best_seedavg.npy", pab)
np.save(f"{OUT}/preds/A_last_seedavg.npy", pal)
bb = float(np.mean((pab - yv) ** 2))
bl = float(np.mean((pal - yv) ** 2))
M["stage2a_fold2024"] = {
    "n_train": spA.n_train, "n_valid": spA.n_valid,
    "unknown_valid_cells": spA.unknown_valid,
    "best": {"per_seed_score": [f["best"]["valid_score"] for f in fitsA],
             "sel_epochs": [f["best"]["sel_epoch"] for f in fitsA],
             "seedavg_logit_brier": bb, "seedavg_logit_score": L.conv(bb),
             "seg": seg(pab, yv, gv)},
    "last": {"per_seed_score": [f["last"]["valid_score"] for f in fitsA],
             "sel_epochs": [f["last"]["sel_epoch"] for f in fitsA],
             "seedavg_logit_brier": bl, "seedavg_logit_score": L.conv(bl),
             "seg": seg(pal, yv, gv)},
    "D_last_minus_best": {
        "per_seed_score": [f["last"]["valid_score"] - f["best"]["valid_score"]
                           for f in fitsA],
        "seedavg_brier": bl - bb, "seedavg_score": L.conv(bl) - L.conv(bb),
        "🔴": "이것이 '에폭 선택 규칙' 변인의 fold_2024 대가다. 배포에서 그대로 "
              "나타난다는 보장은 없다(S1) — 크기의 눈금으로만 읽는다."},
    "exp064_control": {"score": E64_A_SCORE, "brier": E64_A_BRIER},
    "fits": fitsA, "wall_min": (time.time() - t0) / 60.0}
print(f"  best seedavg {L.conv(bb):.5f} | last seedavg {L.conv(bl):.5f} "
      f"| D {L.conv(bl) - L.conv(bb):+.5f}", flush=True)
spA.free()
del spA
gc.collect()
torch.cuda.empty_cache()

# ══ stage 2b — 배포(<=2024) 4시드 take='last' ═══════════════════════════════
print("\n===== stage 2b: 배포 <=2024 (4시드, take='last') =====", flush=True)
t0 = time.time()
spD = L.Split(df, DENSE95, m_deploy, m_mon, "DEPLOY99_067")
assert spD.n_train == DEPLOY_N_FULL
fitsD, pD_last, pD_insbest = [], [], []
for sd in SEEDS:
    r = L.fit_both(spD, CFG["widths"], CFG["dropout"], CFG["max_lr"], sd,
                   EPOCHS, tag=f"D_s{sd}")
    fitsD.append({"last": r["last"][0], "insample_best": r["best"][0]})
    pD_last.append(r["last"][1])
    pD_insbest.append(r["best"][1])
    np.save(f"{OUT}/preds/D_last_s{sd}.npy", r["last"][1])
    # 출하 npz — 배포본은 last
    L.to_npz(r["last"][2], len(L.CATS), f"{OUT}/ship/nn_sp{sd}.npz")
    # 대조 보관용 (🔴 표본내 2024 로 고른 것 — 배포에 쓰면 안 된다)
    L.to_npz(r["best"][2], len(L.CATS), f"{OUT}/ship/_contrast_insbest_{sd}.npz")
    L.write_meta(spD, f"{OUT}/ship/nn_sp{sd}_meta.json", CFG["widths"])

pDav = logit_avg(pD_last)
np.save(f"{OUT}/preds/D_last_seedavg.npy", pDav)
ymon = spD.yv64
gmon = g[m_mon]
bD = float(np.mean((pDav - ymon) ** 2))
M["stage2b_deploy"] = {
    "🔴 S1": "아래 2024 수치는 전부 **표본내**다(2024 가 학습 안에 있다). "
             "성능이 아니라 적합 상태 점검용이다. 비교·판정에 쓰지 마라.",
    "n_train": spD.n_train, "n_monitor_insample": spD.n_valid,
    "unknown_monitor_cells": spD.unknown_valid,
    "steps_per_epoch": spD.steps_per_epoch,
    "insample_per_seed_score_LAST": [f["last"]["valid_score"] for f in fitsD],
    "insample_seedavg_brier_LAST": bD,
    "insample_seedavg_score_LAST": L.conv(bD),
    "insample_seg_LAST": seg(pDav, ymon, gmon),
    "insample_best_epochs_IGNORED": [f["insample_best"]["sel_epoch"] for f in fitsD],
    "fits": fitsD, "wall_min": (time.time() - t0) / 60.0}
print(f"  [표본내] deploy seedavg {L.conv(bD):.5f}", flush=True)

# ── 배포 대조표: exp_064(<=2023) vs exp_067(<=2024) 예측 상관·분포 (2024 행) ──
e64 = np.load(f"{E64}/preds/A99ship_seedavg.npy")
lo64, lo67 = L.lg(e64), L.lg(pDav)
M["deploy_contrast_2024rows"] = {
    "🔴": "exp_067 쪽은 표본내다. 상관·분포는 서술이지 성능 비교가 아니다.",
    "pearson_prob": float(np.corrcoef(e64, pDav)[0, 1]),
    "pearson_logit": float(np.corrcoef(lo64, lo67)[0, 1]),
    "spearman_like_rank": float(np.corrcoef(np.argsort(np.argsort(e64)),
                                            np.argsort(np.argsort(pDav)))[0, 1]),
    "mean_exp064": float(e64.mean()), "mean_exp067_deploy": float(pDav.mean()),
    "sd_exp064": float(e64.std()), "sd_exp067_deploy": float(pDav.std()),
    "logit_sd_exp064": float(lo64.std()), "logit_sd_exp067_deploy": float(lo67.std()),
    "max_abs_prob_diff": float(np.max(np.abs(e64 - pDav))),
    "mean_abs_prob_diff": float(np.mean(np.abs(e64 - pDav)))}
M["train_time_min"] = (time.time() - t_all) / 60.0
json.dump(M, open(f"{OUT}/run067.json", "w"), indent=1, ensure_ascii=False)
print(f"\n[done] {M['train_time_min']:.2f} min", flush=True)

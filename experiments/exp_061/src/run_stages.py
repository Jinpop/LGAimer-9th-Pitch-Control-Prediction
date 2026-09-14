"""exp_061 드라이버 — stage1 용량 스윕 -> stage2 시드평균 -> stage3 분할B -> stage4 배포.

🔴 판정 규칙은 전부 이 파일 안에 **사전 등록**돼 있다 (결과를 보고 바꾸지 않는다):
  - stage1 정지점: 16셀 중 파일럿(837.06272) 대비 +10점 이상이 없으면 용량 축을 닫고
    파일럿 구성(256-128-64, dropout 0.30, max_lr 3e-3)으로 stage 2 진행. 재탐색 금지.
  - stage3 게이트: 분할 A 와 분할 B 에서 부호가 갈리면 stage 4 로 넘기지 않는다.
      S_cap  = score(낙점 4시드평균) - score(파일럿 4시드평균)   (낙점==파일럿이면 미정의)
      S_seed = score(4시드 로짓평균) - mean(시드별 score)         (낙점 구성)
  - stage4 에폭: 분할 A 각 시드의 best epoch 를 **그 시드에** 그대로 쓴다
    (행 수 비례 외삽 금지). OneCycle 은 15에폭짜리 사이클을 그대로 두고 E 에폭에서 멈춘다
    — 분할 A 가 best 를 낸 그 시점의 LR 궤적 위치를 재현하기 위함.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

t_all = time.time()
SMOKE = os.environ.get("EXP061_SMOKE") == "1"
if SMOKE:                       # 전 경로 1에폭 리허설 (본판 산출물과 분리)
    L.OUT = f"{L.B}/out_smoke"
OUT = L.OUT
os.makedirs(OUT, exist_ok=True)
os.makedirs(f"{OUT}/preds", exist_ok=True)

# ── 사전 등록 상수 ────────────────────────────────────────────────────────────
PILOT = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
PILOT_SCORE = 837.06272          # exp_060 c15/full15 seed42 실측
STAGE1_MARGIN = 10.0
EPOCHS = 15                      # OneCycle 완주 (patience 무효)
SEEDS = [42, 777, 2024, 101]
WIDTHS = [(256, 128, 64), (512, 256, 128), (1024, 512, 256), (768, 384, 192)]
DROPOUTS = [0.30, 0.15]
LRS = [3e-3, 1e-3]
AA_REF_MEANBRIER = 0.24787535671986385   # exp_048 A1@1400 3시드 mean-of-briers
AA_REF_SCORE = 773.21454

if SMOKE:                       # 규칙은 그대로 두고 격자만 줄인다
    EPOCHS = 1
    SEEDS = [42, 777]
    WIDTHS = [(256, 128, 64), (512, 256, 128)]
    DROPOUTS = [0.30]
    LRS = [3e-3]
    STAGE1_MARGIN = -1e9        # cap_open 분기까지 태우기 위한 리허설 전용 값

M = {"exp_id": "exp_061", "prereg": {
    "pilot": {"widths": list(PILOT["widths"]), "dropout": PILOT["dropout"],
              "max_lr": PILOT["max_lr"], "score": PILOT_SCORE},
    "stage1_margin": STAGE1_MARGIN, "epochs": EPOCHS, "seeds": SEEDS,
    "stage3_gate": "sign(S_cap) and sign(S_seed) must agree between split A and B",
    "stage4_epoch_rule": "per-seed split-A best epoch, OneCycle cycle kept at 15ep"}}

df, DENSE, prov = L.load_frame()
M["frame_provenance"] = prov
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
mA_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
mA_va = (s == 2024)
mB_tr = (s >= 2020) & (s <= 2022) & ~((s <= 2022) & (g == "F"))
mB_va = (s == 2023) & (g == "R")
mD_tr = (s <= 2024) & ~((s <= 2022) & (g == "F"))
assert (int(mA_tr.sum()), int(mA_va.sum())) == (1116277, 253507)
assert (int(mB_tr.sum()), int(mB_va.sum())) == (659125, 219839)
assert int(mD_tr.sum()) == 1369784

PRED = {}          # tag -> p_valid (분할 A)
RES1 = []


def cellkey(w, do, lr, seed):
    return f"w{w[0]}_do{do}_lr{lr}_s{seed}"


def logit_avg(ps):
    return 1.0 / (1.0 + np.exp(-np.mean([L.lg(p) for p in ps], axis=0)))


def seg_metrics(p, y, gt):
    from sklearn.metrics import roc_auc_score, log_loss
    o = {"brier": float(np.mean((p - y) ** 2)),
         "auc": float(roc_auc_score(y, p)),
         "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))}
    o["score"] = L.conv(o["brier"])
    for nm, mk in (("valid_R", gt == "R"), ("valid_F", gt == "F")):
        if mk.sum():
            o[nm] = {"n": int(mk.sum()),
                     "brier": float(np.mean((p[mk] - y[mk]) ** 2))}
    return o


# ══ stage 1 — 용량 스윕 (seed 42, 분할 A) ═════════════════════════════════════
print("\n===== stage 1 =====", flush=True)
t1 = time.time()
spA = L.Split(df, DENSE, mA_tr, mA_va, "A")
M["split_A"] = {"n_train": spA.n_train, "n_valid": spA.n_valid,
                "unknown_valid_cells": spA.unknown_valid,
                "desc": "train season<=2023 (mask: season<=2022 & F 제외) -> valid 2024 전체"}
for w in WIDTHS:
    for do in DROPOUTS:
        for lr in LRS:
            k = cellkey(w, do, lr, 42)
            r, p, _ = L.fit(spA, w, do, lr, 42, EPOCHS, tag=k)
            RES1.append(r)
            PRED[k] = p
            np.save(f"{OUT}/preds/pA_{k}.npy", p)
M["stage1"] = {"cells": RES1, "wall_min": (time.time() - t1) / 60.0}
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1)

# ---- 재현 게이트: 파일럿 셀 == exp_060 full15 ----
pk = cellkey(PILOT["widths"], PILOT["dropout"], PILOT["max_lr"], 42)
try:
    p60 = np.load(os.path.expanduser(
        "~/SungjinPark/exp060_nn/out/pred_valid2024_seed42_full15.npy"))
    d = float(np.abs(PRED[pk] - p60).max())
    M["repro_gate_vs_exp060_full15"] = {
        "max_abs_diff": d, "pass": bool(d < 5e-6),
        "exp060_brier": float(np.mean((p60 - spA.yv64) ** 2)),
        "exp061_brier": float(np.mean((PRED[pk] - spA.yv64) ** 2))}
    print(f"[repro gate] max_abs_diff {d:.3e}", flush=True)
except Exception as e:            # noqa: BLE001
    M["repro_gate_vs_exp060_full15"] = {"error": repr(e)}

# ---- 사전 등록 정지점 ----
best = max(RES1, key=lambda r: r["valid_score"])
if best["valid_score"] >= PILOT_SCORE + STAGE1_MARGIN:
    PICK = {"widths": tuple(best["widths"]), "dropout": best["dropout"],
            "max_lr": best["max_lr"]}
    cap_open = True
else:
    PICK = dict(PILOT)
    cap_open = False
M["stage1_decision"] = {
    "best_cell": best["tag"], "best_score": best["valid_score"],
    "margin_over_pilot": best["valid_score"] - PILOT_SCORE,
    "capacity_axis_open": cap_open,
    "pick": {"widths": list(PICK["widths"]), "dropout": PICK["dropout"],
             "max_lr": PICK["max_lr"]}}
print(f"[stage1] best {best['tag']} {best['valid_score']:.5f} "
      f"(pilot {PILOT_SCORE}) -> capacity_open={cap_open} pick={PICK}",
      flush=True)
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1)


# ══ stage 2 — 시드 평균 (분할 A) ══════════════════════════════════════════════
def seed_block(sp, cfg, seeds, prefix, cache=None):
    """구성 하나를 여러 시드로 적합. cache 에 (config,seed) 예측이 있으면 재사용."""
    rs, ps, states = [], [], {}
    for sd in seeds:
        k = cellkey(cfg["widths"], cfg["dropout"], cfg["max_lr"], sd)
        r, p, st = L.fit(sp, cfg["widths"], cfg["dropout"], cfg["max_lr"], sd,
                         EPOCHS, tag=f"{prefix}_{k}")
        if cache is not None and k in cache:
            r["reproduction_vs_stage1"] = {
                "max_abs_diff": float(np.abs(p - cache[k]).max())}
        rs.append(r)
        ps.append(p)
        states[sd] = st
        np.save(f"{OUT}/preds/{prefix}_{k}.npy", p)
    pav = logit_avg(ps)
    np.save(f"{OUT}/preds/{prefix}_seedavg.npy", pav)
    per = [r["valid_score"] for r in rs]
    ens_b = float(np.mean((pav - sp.yv64) ** 2))
    out = {"fits": rs, "per_seed_score_mean": float(np.mean(per)),
           "per_seed_score_sd": float(np.std(per, ddof=1)),
           "per_seed_brier_mean": float(np.mean([r["valid_brier"] for r in rs])),
           "seedavg_logit_brier": ens_b, "seedavg_logit_score": L.conv(ens_b),
           "S_seed": L.conv(ens_b) - float(np.mean(per)),
           "best_epochs": [r["best_epoch"] for r in rs]}
    return out, pav, states


print("\n===== stage 2 =====", flush=True)
t2 = time.time()
gtA = g[mA_va]
s2_pick, pA_pick, statesA = seed_block(spA, PICK, SEEDS, "A_pick", cache=PRED)
s2_pick["metrics_seedavg"] = seg_metrics(pA_pick, spA.yv64, gtA)
M["stage2"] = {"pick": s2_pick, "wall_min": (time.time() - t2) / 60.0}
if cap_open:
    s2_pil, pA_pil, _ = seed_block(spA, PILOT, SEEDS, "A_pilot", cache=PRED)
    M["stage2"]["pilot_contrast"] = s2_pil
    S_cap_A = s2_pick["seedavg_logit_score"] - s2_pil["seedavg_logit_score"]
else:
    S_cap_A = None
M["stage2"]["S_cap_A"] = S_cap_A
M["stage2"]["S_seed_A"] = s2_pick["S_seed"]
M["stage2"]["vs_AA_arm"] = {
    "AA_A1_1400_3seed_mean_of_briers": AA_REF_MEANBRIER,
    "AA_score": AA_REF_SCORE,
    "NN_per_seed_brier_mean": s2_pick["per_seed_brier_mean"],
    "NN_per_seed_score_mean": s2_pick["per_seed_score_mean"],
    "delta_same_convention": s2_pick["per_seed_score_mean"] - AA_REF_SCORE}
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1)
print(f"[stage2] S_seed_A {s2_pick['S_seed']:.5f} S_cap_A {S_cap_A}", flush=True)

spA.free()

# ══ stage 3 — 분할 B ══════════════════════════════════════════════════════════
print("\n===== stage 3 =====", flush=True)
t3 = time.time()
spB = L.Split(df, DENSE, mB_tr, mB_va, "B")
M["split_B"] = {"n_train": spB.n_train, "n_valid": spB.n_valid,
                "unknown_valid_cells": spB.unknown_valid,
                "desc": "train 2020~2022 R -> valid 2023 R. 환산 ref 는 2024 눈금이라 "
                        "수준은 분할 간 비교 불가 — 분할 안의 차(Δ)만 비교한다"}
s3_pick, pB_pick, _ = seed_block(spB, PICK, SEEDS, "B_pick")
M["stage3"] = {"pick": s3_pick}
if cap_open:
    s3_pil, pB_pil, _ = seed_block(spB, PILOT, SEEDS, "B_pilot")
    M["stage3"]["pilot_contrast"] = s3_pil
    S_cap_B = s3_pick["seedavg_logit_score"] - s3_pil["seedavg_logit_score"]
else:
    S_cap_B = None
M["stage3"]["S_cap_B"] = S_cap_B
M["stage3"]["S_seed_B"] = s3_pick["S_seed"]
M["stage3"]["wall_min"] = (time.time() - t3) / 60.0

gate = {"S_seed_A": s2_pick["S_seed"], "S_seed_B": s3_pick["S_seed"],
        "S_seed_agree": bool(np.sign(s2_pick["S_seed"]) == np.sign(s3_pick["S_seed"])),
        "S_cap_A": S_cap_A, "S_cap_B": S_cap_B}
gate["S_cap_agree"] = (None if not cap_open
                       else bool(np.sign(S_cap_A) == np.sign(S_cap_B)))
gate["pass"] = bool(gate["S_seed_agree"] and (gate["S_cap_agree"] is not False))
M["stage3"]["gate"] = gate
spB.free()
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1)
print(f"[stage3] gate {json.dumps(gate)}", flush=True)

# ══ stage 4 — 배포 재적합 (<=2024 전체) ═══════════════════════════════════════
if not gate["pass"]:
    M["stage4"] = {"skipped": True,
                   "reason": "stage3 gate 부호 불일치 — 사전 등록 규칙대로 중단"}
    print("[stage4] SKIPPED (gate fail)", flush=True)
else:
    print("\n===== stage 4 =====", flush=True)
    t4 = time.time()
    spD = L.Split(df, DENSE, mD_tr, mA_va, "D")   # valid=2024 는 표본 내(검증 전용 아님)
    M["split_D"] = {"n_train": spD.n_train,
                    "desc": "train season<=2024 (mask 동일) 전체 재적합. "
                            "valid 2024 는 표본 내라 점수 판정에 쓰지 않는다 "
                            "— numpy 재현 검증용 순전파 입력일 뿐"}
    ep_by_seed = {r["seed"]: r["best_epoch"] for r in s2_pick["fits"]}
    dep = []
    for sd in SEEDS:
        E = ep_by_seed[sd]
        r, p, st = L.fit(spD, PICK["widths"], PICK["dropout"], PICK["max_lr"],
                         sd, E, sched_epochs=EPOCHS, take="final",
                         tag=f"D_s{sd}_ep{E}")
        r["epoch_rule"] = (f"분할 A seed {sd} best epoch = {E} 그대로. "
                           f"OneCycle total_steps 는 15에폭 사이클 유지")
        base = f"{OUT}/nn99_d{sd}"
        keys = L.to_npz(st, len(L.CATS), f"{base}.npz")
        torch.save(st, f"{base}.pt")
        json.dump({"CATS": L.CATS, "levels": spD.levels, "DENSE": DENSE,
                   "mu": [float(v) for v in spD.mu],
                   "sd": [float(v) for v in spD.sd],
                   "edims": [L.EDIM] * len(L.CATS), "H1": PICK["widths"][0],
                   "widths": list(PICK["widths"]), "dropout": PICK["dropout"],
                   "seed": sd, "epochs": E, "npz_keys": keys},
                  open(f"{base}_meta.json", "w"))
        np.save(f"{OUT}/preds/D_s{sd}_pred2024_insample.npy", p)
        dep.append(r)
    M["stage4"] = {"fits": dep, "wall_min": (time.time() - t4) / 60.0,
                   "epoch_by_seed": ep_by_seed}
    spD.free()

M["total_wall_min"] = (time.time() - t_all) / 60.0
M["gpu"] = torch.cuda.get_device_name(0)
M["torch"] = torch.__version__
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1)
print("\nALL_STAGES_DONE", flush=True)

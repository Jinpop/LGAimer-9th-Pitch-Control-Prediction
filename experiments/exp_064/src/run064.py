"""exp_064 stage 3 드라이버 — 99열(14열 = **배포 p6**) 4시드 재학습.

대조군 둘 (같은 구성·같은 4시드·같은 분할):
  · exp_061 out2 파일럿 99열(우리 재구축 p6)  A 888.15660 / B 746.18208
  · exp_063 85열(14열 제거)                  A 878.58760 / B 724.26956

이 실험의 의도한 변인은 **14열의 산출원 하나**다(우리 685쌍 재구축 → 배포본 그대로).
딸려 바뀌는 것: pitcher_id 어휘 685→571, 2019행 14열이 NaN → 값, 표준화 mu/sd.

🔴 판정하지 않는다. 산출은 게이트 결과·NN 단독 점수·상관뿐이다(config S8).

단계
  0. 원본 frame99 provenance sha 4종 확인
  1. 재현 게이트 — 원본 frame99 99열 seed 42 를 이 코드로 적합해
     exp_061 out2/preds/A_pilot_s42 와 abs_diff < 5e-6 인지 (규약: seed 42 단독)
  2. 분할 A — frame99_ship 99열 4시드
  3. 분할 B — frame99_ship 99열 4시드
"""
import gc
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
os.makedirs(f"{OUT}/curves", exist_ok=True)

E61 = os.path.expanduser("~/SungjinPark/exp061_nn/out2")
E63 = os.path.expanduser("~/SungjinPark/exp063_nn85/out")

# ── 사전 등록 상수 (숫자를 보기 전에 고정) ────────────────────────────────────
CFG = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
EPOCHS = 25
SEEDS = [42, 777, 2024, 101]
REPRO_TOL = 5e-6
NN99_A_SCORE = 888.1566009436015     # exp_061 out2 pilot seedavg (우리 p6)
NN99_B_SCORE = 746.182077782176
NN85_A_SCORE = 878.5875993947867     # exp_063 85열 seedavg
NN85_B_SCORE = 724.2695600160332
GATE_LOSS_EXP062 = -2.97796
PROV_EXPECT = {
    "train_csv_sha256":
        "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "features_sp_sha256":
        "783c650c6dd78cdcd641d0f18e35469ced247544b5d1ebf8d138171e04498924",
    "pipeline_py_sha256":
        "a3181a10a1c621524460f14bc06b3c7f188fe23955814bb5e5b890e0df3016c9",
    "splits_py_sha256":
        "5e8884ed7c8d918e6c4fc332d1743c5437d21f72b8e2021a6f37fc4a8dd2ceb4",
}

_REQ = ([f"{E61}/preds/A_pilot_s{s}.npy" for s in SEEDS]
        + [f"{E61}/preds/B_pilot_s{s}.npy" for s in SEEDS]
        + [f"{E61}/preds/A_pilot_seedavg.npy", f"{E61}/preds/B_pilot_seedavg.npy"]
        + [f"{E63}/preds/A85_s{s}.npy" for s in SEEDS]
        + [f"{E63}/preds/B85_s{s}.npy" for s in SEEDS]
        + [f"{E63}/preds/A85_seedavg.npy", f"{E63}/preds/B85_seedavg.npy"])
for _f in _REQ:
    assert os.path.exists(_f), f"대조군 예측 없음: {_f}"
    assert len(np.load(_f)) in (253507, 219839), _f
print(f"[precheck] 대조군 예측 {len(_REQ)}개 확인", flush=True)

M = {"exp_id": "exp_064", "prereg": {
    "config": {"widths": list(CFG["widths"]), "dropout": CFG["dropout"],
               "max_lr": CFG["max_lr"], "epochs": EPOCHS, "seeds": SEEDS},
    "controls": {"exp_061_99col_ours_p6": {"A": NN99_A_SCORE, "B": NN99_B_SCORE},
                 "exp_063_85col_dropped": {"A": NN85_A_SCORE, "B": NN85_B_SCORE}},
    "changed_variable": "14열의 산출원 — 우리 685쌍 재구축 p6 -> 배포본 p6 그대로",
    "incidental": ["pitcher_id 어휘 685 -> 571 (미등재는 결측 처리)",
                   "2019행 14열: NaN -> 값 (배포 p6 는 빈티지가 없다)",
                   "14열 분포 변화에 따른 표준화 mu/sd 재계산"],
    "repro_tol": REPRO_TOL, "gate_loss_exp062": GATE_LOSS_EXP062,
    "verdict_scope": "🔴 팔 채택을 판정하지 않는다 — 게이트·점수·상관만 낸다"}}


def logit_avg(ps):
    return 1.0 / (1.0 + np.exp(-np.mean([L.lg(q) for q in ps], axis=0)))


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


def seed_block(sp, DEN, prefix):
    rs, ps = [], []
    for sd in SEEDS:
        r, p, _ = L.fit(sp, CFG["widths"], CFG["dropout"], CFG["max_lr"], sd,
                        EPOCHS, tag=f"{prefix}_s{sd}")
        rs.append(r)
        ps.append(p)
        np.save(f"{OUT}/preds/{prefix}_s{sd}.npy", p)
    pav = logit_avg(ps)
    np.save(f"{OUT}/preds/{prefix}_seedavg.npy", pav)
    per = [r["valid_score"] for r in rs]
    eb = float(np.mean((pav - sp.yv64) ** 2))
    return ({"fits": rs, "n_dense": len(DEN), "per_seed_score": per,
             "per_seed_score_mean": float(np.mean(per)),
             "per_seed_score_sd": float(np.std(per, ddof=1)),
             "per_seed_brier_mean": float(np.mean([r["valid_brier"] for r in rs])),
             "seedavg_logit_brier": eb, "seedavg_logit_score": L.conv(eb),
             "S_seed": L.conv(eb) - float(np.mean(per)),
             "best_epochs": [r["best_epoch"] for r in rs]}, pav)


def ref(path, y):
    p = np.load(path)
    b = float(np.mean((p - y) ** 2))
    return p, b, L.conv(b)


def masks(df):
    s = df["season"].to_numpy()
    g = df["game_type"].to_numpy()
    mA_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
    mA_va = (s == 2024)
    mB_tr = (s >= 2020) & (s <= 2022) & ~((s <= 2022) & (g == "F"))
    mB_va = (s == 2023) & (g == "R")
    assert (int(mA_tr.sum()), int(mA_va.sum())) == (1116277, 253507)
    assert (int(mB_tr.sum()), int(mB_va.sum())) == (659125, 219839)
    return mA_tr, mA_va, mB_tr, mB_va, g


# ══ 1. 재현 게이트 — 원본 frame99, 99열, seed 42 ══════════════════════════════
print("\n===== stage 0: repro gate (원본 frame99, 99col, seed 42) =====", flush=True)
t0 = time.time()
df0, DENSE95, prov0 = L.load_frame("orig")
for k, v in PROV_EXPECT.items():
    assert prov0[k] == v, f"provenance {k} 불일치: {prov0[k]} != {v}"
assert len(DENSE95) == 95
M["frame_provenance_orig"] = prov0
mA_tr, mA_va, mB_tr, mB_va, g0 = masks(df0)
spR = L.Split(df0, DENSE95, mA_tr, mA_va, "A99orig")
rR, pR, _ = L.fit(spR, CFG["widths"], CFG["dropout"], CFG["max_lr"], 42, EPOCHS,
                  tag="repro_A99_s42")
np.save(f"{OUT}/preds/repro_A99_s42.npy", pR)
d = float(np.abs(pR - np.load(f"{E61}/preds/A_pilot_s42.npy")).max())
M["repro_gate"] = {
    "desc": "원본 frame99 99열 seed42 재적합 vs exp_061 out2/preds/A_pilot_s42",
    "max_abs_diff": d, "tol": REPRO_TOL, "pass": bool(d < REPRO_TOL), "fit": rR,
    "exp061_recorded_score": 829.6718195295538,
    "exp063_recorded_max_abs_diff": 0.0,
    "wall_min": (time.time() - t0) / 60.0}
print(f"[repro gate] max_abs_diff {d:.3e} pass={d < REPRO_TOL}", flush=True)
assert d < REPRO_TOL, f"재현 게이트 실패 max_abs_diff={d:.3e}"
spR.free()
del spR, df0
gc.collect()
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

# ══ 2·3. 배포 p6 프레임 ══════════════════════════════════════════════════════
df, DENSE95b, prov = L.load_frame("ship")
assert DENSE95b == DENSE95, "DENSE 열 순서가 달라졌다"
mA_tr, mA_va, mB_tr, mB_va, g = masks(df)
M["frame_ship"] = {
    "path": f"{L.B}/cache/frame99_ship.parquet",
    "stage1_gate": json.load(open(f"{L.B}/out/stage1_assemble.json"))[
        "invariance_gate"],
    "unregistered_census": json.load(open(f"{L.B}/out/stage1_assemble.json"))[
        "unregistered_census"]}

print("\n===== stage A: 99col(ship p6) x 4 seeds (split A) =====", flush=True)
tA = time.time()
spA = L.Split(df, DENSE95, mA_tr, mA_va, "A99ship")
M["split_A"] = {"n_train": spA.n_train, "n_valid": spA.n_valid,
                "unknown_valid_cells": spA.unknown_valid,
                "exp063_unknown_valid_cells": 584,
                "desc": "train season<=2023 (season<=2022 & F 제외) -> valid 2024 전체"}
a, pA = seed_block(spA, DENSE95, "A99ship")
yA = spA.yv64
a["metrics_seedavg"] = seg_metrics(pA, yA, g[mA_va])
_, b99A, s99A = ref(f"{E61}/preds/A_pilot_seedavg.npy", yA)
_, b85A, s85A = ref(f"{E63}/preds/A85_seedavg.npy", yA)
M["stage_A"] = {
    "nn99ship": a,
    "control_exp061_99col_ours_p6": {"brier": b99A, "score": s99A,
                                     "recorded": NN99_A_SCORE,
                                     "recompute_delta": s99A - NN99_A_SCORE},
    "control_exp063_85col": {"brier": b85A, "score": s85A,
                             "recorded": NN85_A_SCORE,
                             "recompute_delta": s85A - NN85_A_SCORE},
    "D_vs_exp061_99col": a["seedavg_logit_score"] - s99A,
    "D_vs_exp063_85col": a["seedavg_logit_score"] - s85A,
    "wall_min": (time.time() - tA) / 60.0}
print(f"[stage A] ship99 {a['seedavg_logit_score']:.5f} | ours99 {s99A:.5f} "
      f"| 85col {s85A:.5f}", flush=True)
spA.free()
del spA
gc.collect()
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)

print("\n===== stage B: 99col(ship p6) x 4 seeds (split B) =====", flush=True)
tB = time.time()
spB = L.Split(df, DENSE95, mB_tr, mB_va, "B99ship")
M["split_B"] = {"n_train": spB.n_train, "n_valid": spB.n_valid,
                "unknown_valid_cells": spB.unknown_valid,
                "desc": "train 2020~2022 R -> valid 2023 R. 환산 ref 는 2024 눈금 — "
                        "분할 안의 차(Δ)만 읽는다"}
bb, pB = seed_block(spB, DENSE95, "B99ship")
yB = spB.yv64
_, b99B, s99B = ref(f"{E61}/preds/B_pilot_seedavg.npy", yB)
_, b85B, s85B = ref(f"{E63}/preds/B85_seedavg.npy", yB)
M["stage_B"] = {
    "nn99ship": bb,
    "control_exp061_99col_ours_p6": {"brier": b99B, "score": s99B,
                                     "recorded": NN99_B_SCORE,
                                     "recompute_delta": s99B - NN99_B_SCORE},
    "control_exp063_85col": {"brier": b85B, "score": s85B,
                             "recorded": NN85_B_SCORE,
                             "recompute_delta": s85B - NN85_B_SCORE},
    "D_vs_exp061_99col": bb["seedavg_logit_score"] - s99B,
    "D_vs_exp063_85col": bb["seedavg_logit_score"] - s85B,
    "wall_min": (time.time() - tB) / 60.0}
print(f"[stage B] ship99 {bb['seedavg_logit_score']:.5f} | ours99 {s99B:.5f} "
      f"| 85col {s85B:.5f}", flush=True)
spB.free()

# ── 시드별 짝지은 차 ─────────────────────────────────────────────────────────
paired = {}
for split, pre, r61, r63 in (("A", "A99ship", "A_pilot", "A85"),
                             ("B", "B99ship", "B_pilot", "B85")):
    yy = yA if split == "A" else yB
    rows = []
    for sd in SEEDS:
        ps = np.load(f"{OUT}/preds/{pre}_s{sd}.npy")
        p9 = np.load(f"{E61}/preds/{r61}_s{sd}.npy")
        p8 = np.load(f"{E63}/preds/{r63}_s{sd}.npy")
        sc = [L.conv(float(np.mean((q - yy) ** 2))) for q in (ps, p9, p8)]
        rows.append({"seed": sd, "score_ship99": sc[0], "score_ours99": sc[1],
                     "score_85": sc[2], "delta_vs_ours99": sc[0] - sc[1],
                     "delta_vs_85": sc[0] - sc[2]})
    for key in ("delta_vs_ours99", "delta_vs_85"):
        ds = [r[key] for r in rows]
        paired.setdefault(split, {})[key] = {
            "mean": float(np.mean(ds)), "sd": float(np.std(ds, ddof=1)),
            "n_negative": int(sum(x < 0 for x in ds))}
    paired[split]["per_seed"] = rows
M["paired_by_seed"] = paired
M["paired_by_seed"]["note"] = ("같은 시드끼리 짝지은 차. 시드 산포(exp_063 A sd 29.9점)가 "
                              "앙상블 차를 압도했으므로 짝지음이 잡음이 적다")

M["total_wall_min"] = (time.time() - t_all) / 60.0
M["gpu"] = torch.cuda.get_device_name(0)
M["torch"] = torch.__version__
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1, ensure_ascii=False)
print("\nALL_STAGES_DONE", flush=True)

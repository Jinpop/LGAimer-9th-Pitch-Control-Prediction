"""exp_063 드라이버 — 85열(구종모델 파생 14열 제거) 재적합.

대조군 = exp_061 pass2 **파일럿 팔**(256-128-64 / dropout 0.30 / max_lr 3e-3 /
OneCycle 25에폭 / 4시드 로짓평균). 그 팔에서 달라진 것은 **입력 열 14개 제거 하나**다.

🔴 사전 등록 (config.yaml S8 — 결과를 보고 바꾸지 않는다):
  S_14 = score(85열 4시드 로짓평균, 분할 A) - score(99열 4시드 로짓평균, 분할 A)
    · S_14 >= -5    -> 85열 노선 채택 권고 (실질 비교는 S_14 + 2.98)
    · -15 < S_14 < -5 -> 상관 변화로 가른다 (99열판 AA67 상관 0.9014 기준)
    · S_14 <= -15   -> 팀원 trackman 매핑 대기
  ⚠️ 분할 A·B 부호가 갈리면 위 판정을 적용하지 않고 그 사실을 보고한다.

단계
  0. 프레임 provenance sha 4종 재확인 (읽기 전용 재사용)
  1. 재현 게이트 — 99열 seed 42 를 이 코드로 다시 적합해 exp_061 out2/A_pilot_s42
     와 abs_diff < 5e-6 인지. 통과하면 S_14 를 '열 제거' 하나에 귀속할 수 있다.
  2. 분할 A 85열 4시드
  3. 분할 B 85열 4시드
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
L.OUT = f"{L.B}/out"
OUT = L.OUT
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/curves", exist_ok=True)

E61 = os.path.expanduser("~/SungjinPark/exp061_nn/out2")

# ── 사전 등록 상수 (숫자를 보기 전에 고정) ────────────────────────────────────
CFG = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
EPOCHS = 25
SEEDS = [42, 777, 2024, 101]
REPRO_TOL = 5e-6
NN99_A_SCORE = 888.1566009436015     # exp_061 out2 stage2 pilot_lr3e-3 seedavg
NN99_B_SCORE = 746.182077782176      # exp_061 out2 stage3 pilot_lr3e-3 seedavg
GATE_LOSS_EXP062 = -2.97796          # 배포 불일치 실측(85열이면 구조상 0)
THRESH = {"adopt": -5.0, "reject": -15.0}
AA_REF_MEANBRIER = 0.24787535671986385   # exp_048 A1@1400 3시드 mean-of-briers
AA_REF_SCORE = 773.21454
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

# ── 발사 즉시 참조 파일 존재·형상 확인 (15분 뒤에 죽지 않게) ────────────────
_REQ = ([f"{E61}/preds/A_pilot_s{sd}.npy" for sd in SEEDS]
        + [f"{E61}/preds/B_pilot_s{sd}.npy" for sd in SEEDS]
        + [f"{E61}/preds/A_pilot_seedavg.npy", f"{E61}/preds/B_pilot_seedavg.npy"])
for _f in _REQ:
    assert os.path.exists(_f), f"대조군 예측 없음: {_f}"
    _n = len(np.load(_f))
    assert _n in (253507, 219839), (_f, _n)
print(f"[precheck] 대조군 예측 {len(_REQ)}개 확인", flush=True)

M = {"exp_id": "exp_063", "prereg": {
    "config": {"widths": list(CFG["widths"]), "dropout": CFG["dropout"],
               "max_lr": CFG["max_lr"], "epochs": EPOCHS, "seeds": SEEDS},
    "control": "exp_061 out2 pilot arm (99열, 같은 구성·같은 시드)",
    "changed_variable": "입력 열 14개 제거 (99 -> 85; DENSE 95 -> 81) 하나뿐",
    "S_14_def": "score(85열 4시드 로짓평균) - score(99열 4시드 로짓평균), 분할별",
    "thresholds": THRESH, "repro_tol": REPRO_TOL,
    "nn99_A_score": NN99_A_SCORE, "nn99_B_score": NN99_B_SCORE,
    "gate_loss_exp062": GATE_LOSS_EXP062}}

# ── 0. 프레임 ────────────────────────────────────────────────────────────────
df, DENSE95, prov = L.load_frame()
for k, v in PROV_EXPECT.items():
    assert prov[k] == v, f"provenance {k} 불일치: {prov[k]} != {v}"
DENSE81 = L.drop14(DENSE95)
assert len(DENSE95) == 95 and len(DENSE81) == 81
M["frame_provenance"] = prov
M["provenance_recheck"] = {"expected": PROV_EXPECT, "pass": True}
M["columns"] = {
    "dropped_14": L.DROP14, "n_dense_99col": len(DENSE95),
    "n_dense_85col": len(DENSE81), "cats": L.CATS,
    "n_input_99col": len(DENSE95) + 4, "n_input_85col": len(DENSE81) + 4,
    "kept_on_purpose": ["idea_cnt_te", "i3_te48"],
    "kept_check": [c for c in ("idea_cnt_te", "i3_te48") if c in DENSE81]}
assert M["columns"]["kept_check"] == ["idea_cnt_te", "i3_te48"]

s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
mA_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
mA_va = (s == 2024)
mB_tr = (s >= 2020) & (s <= 2022) & ~((s <= 2022) & (g == "F"))
mB_va = (s == 2023) & (g == "R")
assert (int(mA_tr.sum()), int(mA_va.sum())) == (1116277, 253507)
assert (int(mB_tr.sum()), int(mB_va.sum())) == (659125, 219839)


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
    return ({"fits": rs, "n_dense": len(DEN),
             "per_seed_score": per,
             "per_seed_score_mean": float(np.mean(per)),
             "per_seed_score_sd": float(np.std(per, ddof=1)),
             "per_seed_brier_mean": float(np.mean([r["valid_brier"] for r in rs])),
             "seedavg_logit_brier": eb, "seedavg_logit_score": L.conv(eb),
             "S_seed": L.conv(eb) - float(np.mean(per)),
             "best_epochs": [r["best_epoch"] for r in rs]}, pav)


def ref99(path, y):
    """exp_061 out2 저장 예측에서 같은 y·같은 conv 로 되계산 (눈금 동일 보증)."""
    p = np.load(path)
    b = float(np.mean((p - y) ** 2))
    return p, b, L.conv(b)


# ══ 1. 재현 게이트 — 99열 seed 42 (규약: 재현 게이트는 seed 42 단독) ══════════
print("\n===== stage 0: repro gate (99col, seed 42) =====", flush=True)
t0 = time.time()
spA99 = L.Split(df, DENSE95, mA_tr, mA_va, "A99")
r99, p99_s42, _ = L.fit(spA99, CFG["widths"], CFG["dropout"], CFG["max_lr"], 42,
                        EPOCHS, tag="repro_A99_s42")
np.save(f"{OUT}/preds/repro_A99_s42.npy", p99_s42)
ref = np.load(f"{E61}/preds/A_pilot_s42.npy")
d = float(np.abs(p99_s42 - ref).max())
M["repro_gate"] = {
    "desc": "이 코드로 99열 seed42 를 재적합해 exp_061 out2/preds/A_pilot_s42 와 대조",
    "max_abs_diff": d, "tol": REPRO_TOL, "pass": bool(d < REPRO_TOL),
    "fit": r99,
    "exp061_recorded_score": 829.6718195295538}
print(f"[repro gate] max_abs_diff {d:.3e} pass={d < REPRO_TOL}", flush=True)
assert d < REPRO_TOL, f"재현 게이트 실패 max_abs_diff={d:.3e} — 코드 경로가 다르다"
spA99.free()
M["repro_gate"]["wall_min"] = (time.time() - t0) / 60.0
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1,
          ensure_ascii=False)

# ══ 2. 분할 A — 85열 4시드 ═══════════════════════════════════════════════════
print("\n===== stage A: 85col x 4 seeds (split A) =====", flush=True)
tA = time.time()
spA = L.Split(df, DENSE81, mA_tr, mA_va, "A85")
M["split_A"] = {"n_train": spA.n_train, "n_valid": spA.n_valid,
                "unknown_valid_cells": spA.unknown_valid,
                "desc": "train season<=2023 (mask: season<=2022 & F 제외) "
                        "-> valid 2024 전체 253,507행"}
a85, pA85 = seed_block(spA, DENSE81, "A85")
yA = spA.yv64
a85["metrics_seedavg"] = seg_metrics(pA85, yA, g[mA_va])
pA99, bA99, sA99 = ref99(f"{E61}/preds/A_pilot_seedavg.npy", yA)
M["stage_A"] = {
    "nn85": a85,
    "nn99_control_recomputed": {
        "src": f"{E61}/preds/A_pilot_seedavg.npy",
        "seedavg_logit_brier": bA99, "seedavg_logit_score": sA99,
        "recorded_in_exp061": NN99_A_SCORE,
        "recompute_delta": sA99 - NN99_A_SCORE},
    "S_14_A": a85["seedavg_logit_score"] - sA99,
    "wall_min": (time.time() - tA) / 60.0}
M["stage_A"]["vs_AA_arm"] = {
    "AA_A1_1400_3seed_mean_of_briers": AA_REF_MEANBRIER,
    "AA_score_mean_of_briers": AA_REF_SCORE,
    "NN85_per_seed_brier_mean": a85["per_seed_brier_mean"],
    "NN85_per_seed_score_mean": a85["per_seed_score_mean"],
    "delta_same_convention": a85["per_seed_score_mean"] - AA_REF_SCORE}
print(f"[stage A] 85col {a85['seedavg_logit_score']:.5f} vs 99col {sA99:.5f} "
      f"-> S_14_A {M['stage_A']['S_14_A']:.5f}", flush=True)
spA.free()
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1,
          ensure_ascii=False)

# ══ 3. 분할 B — 85열 4시드 ═══════════════════════════════════════════════════
print("\n===== stage B: 85col x 4 seeds (split B) =====", flush=True)
tB = time.time()
spB = L.Split(df, DENSE81, mB_tr, mB_va, "B85")
M["split_B"] = {"n_train": spB.n_train, "n_valid": spB.n_valid,
                "unknown_valid_cells": spB.unknown_valid,
                "desc": "train 2020~2022 R -> valid 2023 R. 환산 ref 는 2024 눈금이라 "
                        "수준은 분할 간 비교 불가 — 분할 안의 차(Δ)만 읽는다"}
b85, pB85 = seed_block(spB, DENSE81, "B85")
yB = spB.yv64
_, bB99, sB99 = ref99(f"{E61}/preds/B_pilot_seedavg.npy", yB)
M["stage_B"] = {
    "nn85": b85,
    "nn99_control_recomputed": {
        "src": f"{E61}/preds/B_pilot_seedavg.npy",
        "seedavg_logit_brier": bB99, "seedavg_logit_score": sB99,
        "recorded_in_exp061": NN99_B_SCORE,
        "recompute_delta": sB99 - NN99_B_SCORE},
    "S_14_B": b85["seedavg_logit_score"] - sB99,
    "wall_min": (time.time() - tB) / 60.0}
print(f"[stage B] 85col {b85['seedavg_logit_score']:.5f} vs 99col {sB99:.5f} "
      f"-> S_14_B {M['stage_B']['S_14_B']:.5f}", flush=True)
spB.free()

# ── 시드별 짝지은 차 (같은 시드끼리 85 vs 99) ────────────────────────────────
paired = {}
for split, prefix, ref_prefix in (("A", "A85", "A_pilot"), ("B", "B85", "B_pilot")):
    yy = yA if split == "A" else yB
    rows = []
    for sd in SEEDS:
        p8 = np.load(f"{OUT}/preds/{prefix}_s{sd}.npy")
        p9 = np.load(f"{E61}/preds/{ref_prefix}_s{sd}.npy")
        b8 = float(np.mean((p8 - yy) ** 2))
        b9 = float(np.mean((p9 - yy) ** 2))
        rows.append({"seed": sd, "score_85": L.conv(b8), "score_99": L.conv(b9),
                     "delta": L.conv(b8) - L.conv(b9)})
    ds = [r["delta"] for r in rows]
    paired[split] = {"per_seed": rows, "mean_delta": float(np.mean(ds)),
                     "sd_delta": float(np.std(ds, ddof=1)),
                     "n_negative": int(sum(d < 0 for d in ds))}
M["paired_by_seed"] = paired
M["paired_by_seed"]["note"] = ("같은 시드끼리 짝지은 차. 시드 산포(exp_061 sd 8~36점)가 "
                              "격자 효과보다 컸으므로 짝지음이 앙상블 차보다 잡음이 적다")

M["total_wall_min"] = (time.time() - t_all) / 60.0
M["gpu"] = torch.cuda.get_device_name(0)
M["torch"] = torch.__version__
json.dump(M, open(f"{OUT}/metrics_partial.json", "w"), indent=1,
          ensure_ascii=False)
print("\nALL_STAGES_DONE", flush=True)

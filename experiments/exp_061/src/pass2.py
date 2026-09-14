"""exp_061 pass 2 — 에폭 예산 확인(지휘자 2026-08-20 승인 범위 확장) + 필요 시 stage 2~4 재실행.

배경: stage 1 에서 max_lr 1e-3 이 best epoch 를 5 -> 11 로 밀었다. 15에폭 OneCycle 사이클이
이 학습률에서 짧을 수 있다. pass 1(15에폭)은 그대로 두고 **덧붙여** 잰다 — 돌고 있는 잡을
죽이지 않고, 15 vs 25 는 에폭 예산만 바뀐 단일 변인 대조가 된다.

🔴 사전 등록 (숫자를 보기 전에 고정):
  · 예산 부족 판정: best_epoch > 0.8 x budget 이면 그 예산은 포화를 못 봤다 -> 올려 재실행.
    (실행시간_실측 §효율 규약의 "argmin 이 cap 에 닿으면 그 fit 은 무효" 를 에폭에 적용)
  · 채택: 낙점 구성의 25에폭 점수가 15에폭 대비 +10 이상이면 EPOCHS_FINAL=25(또는 35),
    아니면 15 를 유지하고 pass 1 결과가 정본이다.
  · 예산이 바뀌면 stage 2/3/4 를 그 예산으로 전부 다시 돌린다(파일럿 대조 포함).
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
L.OUT = f"{L.B}/out2"
OUT = L.OUT
os.makedirs(f"{OUT}/preds", exist_ok=True)

PILOT = {"widths": (256, 128, 64), "dropout": 0.30, "max_lr": 3e-3}
SEEDS = [42, 777, 2024, 101]
BASE_EPOCHS = 15
LADDER = [25, 35]
MARGIN = 10.0
SAT = 0.8                        # best_epoch > SAT*budget 이면 예산 부족

M1 = json.load(open(f"{L.B}/out/metrics.json"))
pk = M1["stage1_decision"]["pick"]
PICK = {"widths": tuple(pk["widths"]), "dropout": pk["dropout"],
        "max_lr": pk["max_lr"]}
base_cell = [c for c in M1["stage1"]["cells"]
             if tuple(c["widths"]) == PICK["widths"]
             and c["dropout"] == PICK["dropout"]
             and c["max_lr"] == PICK["max_lr"]][0]
BASE_SCORE = base_cell["valid_score"]
print(f"pick {PICK} base(15ep) {BASE_SCORE:.5f} best_epoch {base_cell['best_epoch']}",
      flush=True)

M = {"exp_id": "exp_061_pass2", "prereg": {
    "budget_insufficient_rule": f"best_epoch > {SAT} * budget",
    "adopt_margin": MARGIN, "ladder": LADDER, "base_epochs": BASE_EPOCHS,
    "pick": {"widths": list(PICK["widths"]), "dropout": PICK["dropout"],
             "max_lr": PICK["max_lr"]},
    "base_15ep_score": BASE_SCORE, "base_15ep_best_epoch": base_cell["best_epoch"]}}

df, DENSE, prov = L.load_frame()
s = df["season"].to_numpy()
g = df["game_type"].to_numpy()
mA_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
mA_va = (s == 2024)
mB_tr = (s >= 2020) & (s <= 2022) & ~((s <= 2022) & (g == "F"))
mB_va = (s == 2023) & (g == "R")
mD_tr = (s <= 2024) & ~((s <= 2022) & (g == "F"))

spA = L.Split(df, DENSE, mA_tr, mA_va, "A")

# ── 에폭 사다리 ───────────────────────────────────────────────────────────────
ladder = []
chosen = BASE_EPOCHS
chosen_score = BASE_SCORE
for bud in LADDER:
    r, p, _ = L.fit(spA, PICK["widths"], PICK["dropout"], PICK["max_lr"], 42,
                    bud, tag=f"ep{bud}_s42")
    r["budget_insufficient"] = bool(r["best_epoch"] > SAT * bud)
    r["delta_vs_15ep"] = r["valid_score"] - BASE_SCORE
    ladder.append(r)
    np.save(f"{OUT}/preds/A_ep{bud}_s42.npy", p)
    if r["valid_score"] > chosen_score:
        chosen, chosen_score = bud, r["valid_score"]
    if not r["budget_insufficient"]:
        break                    # 포화를 봤다 — 더 올릴 근거 없음
M["epoch_ladder"] = ladder
adopt = bool(chosen != BASE_EPOCHS
             and (chosen_score - BASE_SCORE) >= MARGIN)
EP = chosen if adopt else BASE_EPOCHS
M["epoch_decision"] = {"best_budget_seen": chosen, "best_score": chosen_score,
                       "delta_vs_15ep": chosen_score - BASE_SCORE,
                       "adopted": adopt, "EPOCHS_FINAL": EP,
                       "rule": f"delta >= {MARGIN} 이어야 채택"}
print(f"[epoch] {json.dumps(M['epoch_decision'])}", flush=True)
json.dump(M, open(f"{OUT}/metrics_pass2.json", "w"), indent=1, ensure_ascii=False)

if not adopt:
    print("EPOCH_BUDGET_UNCHANGED — pass 1(15에폭)이 정본. stage 2~4 재실행 없음.",
          flush=True)
    M["total_wall_min"] = (time.time() - t_all) / 60.0
    json.dump(M, open(f"{OUT}/metrics_pass2.json", "w"), indent=1, ensure_ascii=False)
    print("PASS2_DONE", flush=True)
    sys.exit(0)

# ── 예산이 바뀌었다 -> stage 2/3/4 전부 그 예산으로 재실행 ────────────────────
def logit_avg(ps):
    return 1.0 / (1.0 + np.exp(-np.mean([L.lg(q) for q in ps], axis=0)))


def seg_metrics(p, y, gt):
    from sklearn.metrics import roc_auc_score, log_loss
    o = {"brier": float(np.mean((p - y) ** 2)), "auc": float(roc_auc_score(y, p)),
         "logloss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15)))}
    o["score"] = L.conv(o["brier"])
    for nm, mk in (("valid_R", gt == "R"), ("valid_F", gt == "F")):
        if mk.sum():
            o[nm] = {"n": int(mk.sum()),
                     "brier": float(np.mean((p[mk] - y[mk]) ** 2))}
    return o


def seed_block(sp, cfg, prefix):
    rs, ps, states = [], [], {}
    for sd in SEEDS:
        r, p, st = L.fit(sp, cfg["widths"], cfg["dropout"], cfg["max_lr"], sd,
                         EP, tag=f"{prefix}_s{sd}")
        rs.append(r); ps.append(p); states[sd] = st
        np.save(f"{OUT}/preds/{prefix}_s{sd}.npy", p)
    pav = logit_avg(ps)
    np.save(f"{OUT}/preds/{prefix}_seedavg.npy", pav)
    per = [r["valid_score"] for r in rs]
    eb = float(np.mean((pav - sp.yv64) ** 2))
    return ({"fits": rs, "per_seed_score_mean": float(np.mean(per)),
             "per_seed_score_sd": float(np.std(per, ddof=1)),
             "per_seed_brier_mean": float(np.mean([r["valid_brier"] for r in rs])),
             "seedavg_logit_brier": eb, "seedavg_logit_score": L.conv(eb),
             "S_seed": L.conv(eb) - float(np.mean(per)),
             "best_epochs": [r["best_epoch"] for r in rs]}, pav, states)

print(f"\n===== pass2 stage 2 (EPOCHS={EP}) =====", flush=True)
s2p, pA_pick, _ = seed_block(spA, PICK, "A_pick")
s2p["metrics_seedavg"] = seg_metrics(pA_pick, spA.yv64, g[mA_va])
s2l, pA_pil, _ = seed_block(spA, PILOT, "A_pilot")
S_cap_A = s2p["seedavg_logit_score"] - s2l["seedavg_logit_score"]
M["stage2"] = {"pick": s2p, "pilot_contrast": s2l, "S_cap_A": S_cap_A,
               "S_seed_A": s2p["S_seed"]}
json.dump(M, open(f"{OUT}/metrics_pass2.json", "w"), indent=1, ensure_ascii=False)
spA.free()

print(f"\n===== pass2 stage 3 (EPOCHS={EP}) =====", flush=True)
spB = L.Split(df, DENSE, mB_tr, mB_va, "B")
s3p, _, _ = seed_block(spB, PICK, "B_pick")
s3l, _, _ = seed_block(spB, PILOT, "B_pilot")
S_cap_B = s3p["seedavg_logit_score"] - s3l["seedavg_logit_score"]
gate = {"S_seed_A": s2p["S_seed"], "S_seed_B": s3p["S_seed"],
        "S_seed_agree": bool(np.sign(s2p["S_seed"]) == np.sign(s3p["S_seed"])),
        "S_cap_A": S_cap_A, "S_cap_B": S_cap_B,
        "S_cap_agree": bool(np.sign(S_cap_A) == np.sign(S_cap_B))}
gate["pass"] = bool(gate["S_seed_agree"] and gate["S_cap_agree"])
M["stage3"] = {"pick": s3p, "pilot_contrast": s3l, "S_cap_B": S_cap_B,
               "S_seed_B": s3p["S_seed"], "gate": gate}
print(f"[pass2 stage3] gate {json.dumps(gate)}", flush=True)
spB.free()
json.dump(M, open(f"{OUT}/metrics_pass2.json", "w"), indent=1, ensure_ascii=False)

if not gate["pass"]:
    M["stage4"] = {"skipped": True, "reason": "분할 A·B 부호 불일치 — 사전 등록대로 중단"}
    print("[pass2 stage4] SKIPPED (gate fail)", flush=True)
else:
    print(f"\n===== pass2 stage 4 (EPOCHS from split A best) =====", flush=True)
    spD = L.Split(df, DENSE, mD_tr, mA_va, "D")
    ep_by_seed = {r["seed"]: r["best_epoch"] for r in s2p["fits"]}
    dep = []
    for sd in SEEDS:
        E = ep_by_seed[sd]
        r, p, st = L.fit(spD, PICK["widths"], PICK["dropout"], PICK["max_lr"],
                         sd, E, sched_epochs=EP, take="final",
                         tag=f"D_s{sd}_ep{E}")
        r["epoch_rule"] = (f"분할 A seed {sd} best epoch = {E} 그대로. "
                           f"OneCycle 사이클은 {EP}에폭 유지. 행 수 비례 외삽 없음")
        base = f"{OUT}/nn99_d{sd}"
        keys = L.to_npz(st, len(L.CATS), f"{base}.npz")
        torch.save(st, f"{base}.pt")
        json.dump({"CATS": L.CATS, "levels": spD.levels, "DENSE": DENSE,
                   "mu": [float(v) for v in spD.mu],
                   "sd": [float(v) for v in spD.sd],
                   "edims": [L.EDIM] * len(L.CATS), "H1": PICK["widths"][0],
                   "widths": list(PICK["widths"]), "dropout": PICK["dropout"],
                   "seed": sd, "epochs": E, "sched_epochs": EP,
                   "npz_keys": keys}, open(f"{base}_meta.json", "w"))
        np.save(f"{OUT}/preds/D_s{sd}_pred2024_insample.npy", p)
        dep.append(r)
    M["stage4"] = {"fits": dep, "epoch_by_seed": ep_by_seed}
    spD.free()

M["total_wall_min"] = (time.time() - t_all) / 60.0
json.dump(M, open(f"{OUT}/metrics_pass2.json", "w"), indent=1, ensure_ascii=False)
print("PASS2_DONE", flush=True)

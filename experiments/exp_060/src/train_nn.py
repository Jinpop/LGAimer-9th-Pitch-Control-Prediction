"""exp_060 stage3 — 99열 NN, GPU, seed 42 단독 (파일럿).

분할 A: train season<=2023 (마스크: season<=2022 & game_type=='F' 제외)
        → valid 2024 전체 (valid_all).
구조(출하 numpy 추론기 하드코딩 호환, config 고정):
  임베딩 5개(base_state, game_type, pitcher_team_id, batter_team_id, cnt_state; 각 6차원)
  + dense 95 → Linear(125,256) BN ReLU Drop(0.3) → Linear(256,128) BN ReLU Drop(0.3)
  → Linear(128,64) ReLU → Linear(64,1)
  (Sequential 인덱스: 0 lin, 1 bn, 2 relu, 3 drop, 4 lin, 5 bn, 6 relu, 7 drop,
   8 lin, 9 relu, 10 lin — npz 키 mlp_{0,1,4,5,8,10} 규약과 일치)
전처리(출하 predict_nn과 동일 산술): sp_* NaN→0.0(원값, 모듈의 미지 주체 값),
  dense astype(float32) → (X-mu)/sd(float32, mu/sd는 학습 분할 nanmean/nanstd)
  → nan_to_num(0) — 남은 NaN(2019 v6 파생 등)은 표준화 공간의 0(=평균 대입).
학습: AdamW + OneCycle(max_lr 3e-3, wd 1e-4, pct_start 0.3), batch 4096(drop_last),
  BCEWithLogits, max 60 epoch, early stop patience 5 (valid 2024 Brier).
"""
import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 42
MAX_EPOCHS = int(os.environ.get("EXP060_MAX_EPOCHS", "60"))
TAG = os.environ.get("EXP060_TAG", "")
PATIENCE = int(os.environ.get("EXP060_PATIENCE", "5"))
BATCH = 4096
MAX_LR = 3e-3
WD = 1e-4
PCT_START = 0.3
EDIM = 6
B = os.path.expanduser("~/SungjinPark/exp060_nn")
SPD = os.path.expanduser("~/SungjinPark/sp_def_recv")
OUT = f"{B}/out"
os.makedirs(OUT, exist_ok=True)

t_all = time.time()
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
dev = "cuda"
assert torch.cuda.is_available()

df = pd.read_parquet(f"{B}/cache/frame99.parquet")
FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8")
       if l.strip()]
CATS = ["base_state", "game_type", "pitcher_team_id", "batter_team_id",
        "cnt_state"]
DENSE = [c for c in FSP if c not in CATS]
assert len(DENSE) == 95 and all(c in df.columns for c in CATS)

SP6 = [c for c in DENSE if c.startswith("sp_")]
df[SP6] = df[SP6].fillna(0.0)          # config: 2019 sp_* NaN -> 0.0 (원값)

season = df["season"].to_numpy()
gt = df["game_type"].to_numpy()
train_mask = (season <= 2023) & ~((season <= 2022) & (gt == "F"))
valid_mask = season == 2024
n_tr, n_va = int(train_mask.sum()), int(valid_mask.sum())
print(f"train {n_tr}  valid {n_va}  (masked-out {len(df)-n_tr-n_va})", flush=True)
assert n_va == 253507, n_va

# ---- 범주 레벨(학습 분할 기준) + 코드 (predict_nn과 같은 LUT 규약: 미지=0) ----
levels = {}
for c in CATS:
    vals = df.loc[train_mask, c].unique()
    try:
        lv = sorted(vals, key=lambda x: int(x))
    except (ValueError, TypeError):
        lv = sorted(map(str, vals))
    levels[c] = [str(v) for v in lv]
codes = np.zeros((len(df), len(CATS)), dtype=np.int64)
for i, c in enumerate(CATS):
    lut = {v: j + 1 for j, v in enumerate(levels[c])}
    codes[:, i] = df[c].astype(str).map(lut).fillna(0).astype(np.int64).values
n_unknown_valid = int((codes[valid_mask] == 0).sum())
print(f"valid unknown-category cells: {n_unknown_valid}", flush=True)

# ---- dense 표준화 (predict_nn과 같은 float32 산술) ----
X = df[DENSE].astype("float32").values
mu = np.nanmean(X[train_mask], axis=0).astype(np.float32)
sd = np.nanstd(X[train_mask], axis=0).astype(np.float32)
sd[sd < 1e-6] = 1.0
X = (X - mu) / sd
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
y = df["control_success"].to_numpy(np.float32)

Xt = torch.from_numpy(X[train_mask]).to(dev)
Ct = torch.from_numpy(codes[train_mask]).to(dev)
yt = torch.from_numpy(y[train_mask]).to(dev)
Xv = torch.from_numpy(X[valid_mask]).to(dev)
Cv = torch.from_numpy(codes[valid_mask]).to(dev)
yv64 = y[valid_mask].astype(np.float64)
del X, codes


class Net(nn.Module):
    def __init__(self, nlev, din, ed=EDIM):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(k + 1, ed) for k in nlev])
        self.mlp = nn.Sequential(
            nn.Linear(din + len(nlev) * ed, 256), nn.BatchNorm1d(256),
            nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xc, xd):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xd], dim=1)).squeeze(1)


nlev = [len(levels[c]) for c in CATS]
model = Net(nlev, len(DENSE)).to(dev)
n_params = sum(p.numel() for p in model.parameters())
print(f"input width {len(DENSE) + len(nlev)*EDIM}  params {n_params}", flush=True)

steps_per_epoch = n_tr // BATCH          # drop_last
opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=MAX_LR, total_steps=MAX_EPOCHS * steps_per_epoch,
    pct_start=PCT_START, div_factor=25.0, final_div_factor=1e4)
lossf = nn.BCEWithLogitsLoss()
gcpu = torch.Generator().manual_seed(SEED)


@torch.no_grad()
def eval_valid():
    model.eval()
    ps = []
    for s in range(0, len(Xv), 65536):
        z = model(Cv[s:s + 65536], Xv[s:s + 65536])
        ps.append(torch.sigmoid(z).double().cpu().numpy())
    p = np.concatenate(ps)
    return p, float(np.mean((p - yv64) ** 2))


best = {"brier": np.inf, "epoch": -1, "state": None}
bad = 0
curve = []
stopped = "max_epochs"
for ep in range(1, MAX_EPOCHS + 1):
    te0 = time.time()
    model.train()
    perm = torch.randperm(n_tr, generator=gcpu)
    ep_loss, nb = 0.0, 0
    for s in range(0, steps_per_epoch * BATCH, BATCH):
        idx = perm[s:s + BATCH].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(Ct[idx], Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        sched.step()
        ep_loss += loss.item()
        nb += 1
    t_train = time.time() - te0
    tv0 = time.time()
    _, brier = eval_valid()
    t_eval = time.time() - tv0
    curve.append({"epoch": ep, "train_logloss": ep_loss / nb,
                  "valid_brier": brier, "t_train_s": round(t_train, 2),
                  "t_eval_s": round(t_eval, 2),
                  "lr": sched.get_last_lr()[0]})
    print(f"ep {ep:3d}  loss {ep_loss/nb:.6f}  valid brier {brier:.8f}  "
          f"train {t_train:.1f}s eval {t_eval:.1f}s", flush=True)
    if brier < best["brier"]:
        best = {"brier": brier, "epoch": ep,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}}
        bad = 0
    else:
        bad += 1
        if bad >= PATIENCE:
            stopped = f"early_stop_ep{ep}"
            break

model.load_state_dict(best["state"])
p_valid, brier = eval_valid()
assert abs(brier - best["brier"]) < 1e-12
from sklearn.metrics import roc_auc_score, log_loss  # noqa: E402
auc = float(roc_auc_score(yv64, p_valid))
ll = float(log_loss(yv64, np.clip(p_valid, 1e-15, 1 - 1e-15)))
REF = 0.2498069
conv = 100000.0 * (1.0 - brier / REF)
gt_v = gt[valid_mask]
seg = {}
for name, m in [("valid_R", gt_v == "R"), ("valid_F", gt_v == "F")]:
    bs = float(np.mean((p_valid[m] - yv64[m]) ** 2))
    seg[name] = {"n": int(m.sum()), "brier": bs}

np.save(f"{OUT}/pred_valid2024_seed42{TAG}.npy", p_valid)
torch.save(best["state"], f"{OUT}/nn99_a42{TAG}.pt")
summary = {
    "seed": SEED, "tag": TAG, "n_train": n_tr, "n_valid": n_va,
    "input_width": len(DENSE) + len(nlev) * EDIM,
    "n_dense": len(DENSE), "cats": CATS, "n_levels": nlev, "edim": EDIM,
    "n_params": n_params,
    "hyperparams": {"batch": BATCH, "max_lr": MAX_LR, "weight_decay": WD,
                    "pct_start": PCT_START, "div_factor": 25.0,
                    "final_div_factor": 1e4, "max_epochs": MAX_EPOCHS,
                    "patience": PATIENCE, "optimizer": "AdamW",
                    "scheduler": "OneCycleLR", "loss": "BCEWithLogits",
                    "drop_last": True, "dropout": 0.30},
    "stopped": stopped, "best_epoch": best["epoch"], "epochs_run": len(curve),
    "valid_brier": brier, "valid_score": conv, "valid_auc": auc,
    "valid_logloss": ll, "segments": seg,
    "epoch_wall_s": {"mean_train": float(np.mean([c["t_train_s"] for c in curve])),
                     "mean_eval": float(np.mean([c["t_eval_s"] for c in curve]))},
    "total_wall_min": (time.time() - t_all) / 60.0,
    "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    "unknown_cat_cells_valid": n_unknown_valid,
}
json.dump(summary, open(f"{OUT}/train_summary{TAG}.json", "w"), indent=1)
json.dump(curve, open(f"{OUT}/curve_seed42{TAG}.json", "w"), indent=1)
json.dump({"CATS": CATS, "levels": levels, "DENSE": DENSE,
           "mu": [float(v) for v in mu], "sd": [float(v) for v in sd],
           "edims": [EDIM] * len(CATS), "H1": 256},
          open(f"{OUT}/nn99_a42{TAG}_meta.json", "w"))
print(json.dumps({k: v for k, v in summary.items() if k != "hyperparams"},
                 indent=1), flush=True)
print("train_nn done", flush=True)

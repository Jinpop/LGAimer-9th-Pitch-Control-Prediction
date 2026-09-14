"""exp_085 stage 2 — 팔 1개 학습 (F1_FM 또는 F2_EMB), fold_2024, 시드 42 단독.

사용: python src/train085.py <F1_FM|F2_EMB> <cuda:N> <runtag>

최적화 레시피는 **exp_064 `nnlib.fit` 에서 그대로 승계**한다:
  AdamW(lr=max_lr, wd 1e-4) + OneCycleLR(total_steps = epochs × steps_per_epoch,
  pct_start 0.3, div 25, final_div 1e4), batch 4096 drop_last, BCEWithLogits,
  epochs 25 **고정(조기 종료 없음)**, take="best"(valid Brier 최소 에폭의 상태).
관문 실험이라 하이퍼파라미터를 탐색하지 않는다 — 검증된 레시피를 물려받는 것이
낚시를 피하는 방법이다.

⚠️ run r1 기록: 처음 실행은 schedule 을 60에폭으로 잡고 patience 5 조기 종료를 걸어
   **warm-up 구간에서 잘렸다**(FM 6에폭·EMB 10에폭, LR 이 한 번도 안 내려감).
   그건 승계 레시피가 아니라 내 구현 오류다. r2 가 exp_064 프로토콜 그대로이며
   등록 측정값이다. r1 산출물은 지우지 않고 남겨 둔다.

⚠️ take="best" 는 valid 2024 Brier 를 본다 — exp_060/exp_064 NN 팔과 같은 관례이고
   AA·S1·S2·T70(반복 수 pinned)에는 없는 이점이다. 마지막 에폭 값도 함께 적는다.

🔴 규칙 2-4: 추론이 행별 독립인지 기계로 검사한다(청크 크기 2종 + 순열 왕복, 비트 동일).
"""
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
import torch.nn as nn

ARM = sys.argv[1]
DEV = sys.argv[2] if len(sys.argv) > 2 else "cuda:0"
RUNTAG = sys.argv[3] if len(sys.argv) > 3 else "r2"
SEED = 42
EPOCHS = 25
BATCH = 4096
MAX_LR = 3e-3
WD = 1e-4
PCT_START = 0.3
H = "/home/<user>/SungjinPark"
B = f"{H}/exp085"
W = f"{B}/work"
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)

torch.set_num_threads(6)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False

schema = json.load(open(f"{W}/schema.json"))
FIELDS = schema["fields"]
IX = {n: i for i, n in enumerate(FIELDS)}
NLEV = [schema["levels_n"][n] for n in FIELDS]

Xd = np.load(f"{W}/Xd.npy")
codes = np.load(f"{W}/codes.npy").astype(np.int64)
y = np.load(f"{W}/y.npy")
trm = np.load(f"{W}/train_mask.npy")
vam = np.load(f"{W}/valid_mask.npy")
D = Xd.shape[1]
n_tr = int(trm.sum())

dev = torch.device(DEV)
Xt = torch.from_numpy(Xd[trm]).to(dev)
Ct = torch.from_numpy(codes[trm]).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(Xd[vam]).to(dev)
Cv = torch.from_numpy(codes[vam]).to(dev)
yv = y[vam].astype(np.float64)
del Xd, codes


class FM(nn.Module):
    """인수분해 기계 — 지정된 4쌍의 저계수 쌍선형 상호작용 + 전 열 선형항."""

    K = 16
    FF = ["pitcher_id", "batter_id", "batter_hand", "dom6"]
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 3)]

    def __init__(self):
        super().__init__()
        self.fidx = [IX[n] for n in self.FF]
        self.emb = nn.ModuleList(
            [nn.Embedding(NLEV[i] + 1, self.K) for i in self.fidx])
        self.lin_c = nn.ModuleList(
            [nn.Embedding(k + 1, 1) for k in NLEV])          # 전 범주 열 선형항
        self.lin_d = nn.Linear(D, 1)                          # 수치열 선형항
        for e in self.emb:
            nn.init.normal_(e.weight, 0.0, 0.01)
        for e in self.lin_c:
            nn.init.zeros_(e.weight)
        nn.init.zeros_(self.lin_d.weight)
        nn.init.zeros_(self.lin_d.bias)

    def forward(self, xc, xd):
        z = self.lin_d(xd).squeeze(1)
        for i, e in enumerate(self.lin_c):
            z = z + e(xc[:, i]).squeeze(1)
        v = [self.emb[j](xc[:, self.fidx[j]]) for j in range(len(self.FF))]
        for a, b in self.PAIRS:
            z = z + (v[a] * v[b]).sum(dim=1)
        return z


class EMBMLP(nn.Module):
    """ID 임베딩 MLP — pitcher_id/batter_id/dom6 임베딩 ⊕ 기존 저카디널리티 임베딩
    ⊕ 기존 수치열 → 256-128-64 → 로짓."""

    NEW = {"pitcher_id": 32, "batter_id": 32, "dom6": 16}
    OLD = ["base_state", "game_type", "pitcher_team_id", "batter_team_id", "cnt_state"]
    OLD_DIM = 6

    def __init__(self):
        super().__init__()
        self.names = list(self.NEW) + self.OLD
        self.dims = [self.NEW[n] for n in self.NEW] + [self.OLD_DIM] * len(self.OLD)
        self.fidx = [IX[n] for n in self.names]
        self.emb = nn.ModuleList(
            [nn.Embedding(NLEV[i] + 1, d) for i, d in zip(self.fidx, self.dims)])
        din = D + sum(self.dims)
        self.mlp = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xc, xd):
        e = [emb(xc[:, self.fidx[i]]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat(e + [xd], dim=1)).squeeze(1)


model = (FM() if ARM == "F1_FM" else EMBMLP()).to(dev)
n_params = sum(p.numel() for p in model.parameters())
steps = n_tr // BATCH
opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=PCT_START,
    div_factor=25.0, final_div_factor=1e4)
lossf = nn.BCEWithLogitsLoss()
gcpu = torch.Generator().manual_seed(SEED)
print(f"{ARM} {RUNTAG} params {n_params} dense {D} steps/epoch {steps}", flush=True)


@torch.no_grad()
def predict(chunk=65536, order=None):
    model.eval()
    C_, X_ = (Cv, Xv) if order is None else (Cv[order], Xv[order])
    o = []
    for s in range(0, len(X_), chunk):
        o.append(torch.sigmoid(model(C_[s:s + chunk], X_[s:s + chunk])).double().cpu())
    return torch.cat(o).numpy()


t0 = time.time()
best = {"brier": np.inf, "epoch": -1, "state": None}
curve = []
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = torch.randperm(n_tr, generator=gcpu)
    tot = 0.0
    for s in range(0, steps * BATCH, BATCH):
        idx = perm[s:s + BATCH].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(Ct[idx], Xt[idx]), yt[idx])
        loss.backward()
        opt.step()
        sched.step()
        tot += loss.item()
    p = predict()
    br = float(np.mean((p - yv) ** 2))
    curve.append({"epoch": ep, "train_logloss": tot / steps, "valid_brier": br,
                  "score": 100000.0 * (1.0 - br / 0.2498069),
                  "lr": sched.get_last_lr()[0]})
    print(f"ep {ep:3d} loss {tot/steps:.6f} brier {br:.9f} "
          f"score {100000.0*(1.0-br/0.2498069):.5f} lr {sched.get_last_lr()[0]:.6g}",
          flush=True)
    if br < best["brier"]:
        best = {"brier": br, "epoch": ep,
                "state": {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}}
last_p = p
last_brier = curve[-1]["valid_brier"]
model.load_state_dict(best["state"])
best_p = predict()
wall = time.time() - t0

# ── 규칙 2-4 · 행 독립성 기계 검사 ──────────────────────────────────────
p_small = predict(chunk=4096)
g = torch.Generator().manual_seed(7)
sh = torch.randperm(len(yv), generator=g)
p_sh = predict(chunk=65536, order=sh.to(dev))
inv = np.empty(len(yv), dtype=np.int64)
inv[sh.numpy()] = np.arange(len(yv))
p_unsh = p_sh[inv]
row_indep = {
    "chunk_65536_vs_4096_max_abs_diff": float(np.max(np.abs(best_p - p_small))),
    "shuffled_roundtrip_max_abs_diff": float(np.max(np.abs(best_p - p_unsh))),
    "bitwise_identical_chunk": bool(np.array_equal(best_p, p_small)),
    "bitwise_identical_shuffle": bool(np.array_equal(best_p, p_unsh)),
}
print("row-independence:", json.dumps(row_indep), flush=True)

np.save(f"{OUT}/preds/{ARM}_s{SEED}_{RUNTAG}_best.npy", best_p)
np.save(f"{OUT}/preds/{ARM}_s{SEED}_{RUNTAG}_last.npy", last_p)
torch.save(best["state"], f"{W}/{ARM}_s{SEED}_{RUNTAG}_best.pt")
res = {
    "arm": ARM, "runtag": RUNTAG, "seed": SEED, "device": DEV,
    "n_params": int(n_params), "dense_width": int(D), "n_train": n_tr,
    "n_valid": int(len(yv)), "epochs": EPOCHS, "early_stop": False,
    "take": "best", "best_epoch": best["epoch"],
    "best_valid_brier": best["brier"],
    "best_valid_score": 100000.0 * (1.0 - best["brier"] / 0.2498069),
    "last_valid_brier": last_brier,
    "last_valid_score": curve[-1]["score"],
    "fit_wall_min": round(wall / 60.0, 3),
    "row_independence_rule_2_4": row_indep,
    "curve": curve,
    "recipe": {"opt": "AdamW", "max_lr": MAX_LR, "wd": WD, "batch": BATCH,
               "sched": "OneCycleLR total_steps=25*steps, pct_start 0.3, "
                        "div 25, final_div 1e4",
               "loss": "BCEWithLogits",
               "src": "exp_064 nnlib.fit 승계 (epochs 25, take=best, 조기종료 없음)"},
    "pred_sha256_best": hashlib.sha256(
        np.ascontiguousarray(best_p).tobytes()).hexdigest(),
}
if ARM == "F1_FM":
    res["structure"] = {"k": FM.K, "fields": FM.FF,
                        "pairs": [[FM.FF[a], FM.FF[b]] for a, b in FM.PAIRS],
                        "linear": "수치 157열 Linear + 범주 10필드 레벨별 스칼라 + 절편"}
else:
    res["structure"] = {"new_id_emb": EMBMLP.NEW, "old_emb": EMBMLP.OLD,
                        "old_dim": EMBMLP.OLD_DIM,
                        "mlp": "256 BN ReLU Drop.3 - 128 BN ReLU Drop.3 - 64 ReLU - 1"}
json.dump(res, open(f"{OUT}/{ARM}_s{SEED}_{RUNTAG}.json", "w"),
          ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "curve"},
                 ensure_ascii=False, indent=1))

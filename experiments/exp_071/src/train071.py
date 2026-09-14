"""exp_071 stage 2 — 팀원 배포 NN 구조(159차원)를 OOT 분할로 재학습.

🔴 우리가 새로 정하는 변인(아티팩트에 없다): dropout 0.30 · AdamW wd 1e-4 ·
   batch 4096 drop_last · OneCycle max_lr 3e-3 pct_start 0.3 div25 final1e4 ·
   25에폭 스케줄 완주. exp_064/068 레시피 승계.
구조·전처리·레벨규약·edims 는 배포 사양(nn_a42_meta.json) 그대로.
mu/sd·levels 는 **우리 학습행(≤2023 마스크)** 에서 새로 계산한다 — 배포 meta 값을
그대로 쓰면 2024 통계가 OOT 벡터로 새어든다.
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[v] = "6"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

H = "/home/<user>/SungjinPark"
E = f"{H}/exp071"
OUT, CACHE, MODEL = f"{E}/out", f"{E}/cache", f"{E}/model"
os.makedirs(MODEL, exist_ok=True)
torch.set_num_threads(6)

BATCH, WD, PCT_START, MAX_LR, DROPOUT, EPOCHS = 4096, 1e-4, 0.3, 3e-3, 0.30, 25
SEEDS = [42, 777, 2024, 101]
DEV = "cuda"
T0 = time.time()

META = json.load(open(f"{E}/mate/model/nn_a42_meta.json"))
CATS, DENSE, EDIMS, H1 = META["CATS"], META["DENSE"], META["edims"], META["H1"]

df = pd.read_parquet(f"{CACHE}/frame071.parquet")
s = df.season.to_numpy()
g = df.game_type.astype(str).to_numpy()
m_tr = (s <= 2023) & ~((s <= 2022) & (g == "F"))
m_24 = (s == 2024)
assert int(m_tr.sum()) == 1116277 and int(m_24.sum()) == 253507

# ── 레벨·표준화 상수: 학습행에서만 ────────────────────────────────
levels = {}
for c in CATS:
    vals = df.loc[m_tr, c].unique()
    try:
        lv = sorted(vals, key=lambda x: int(x))
    except (ValueError, TypeError):
        lv = sorted(map(str, vals))
    levels[c] = [str(v) for v in lv]
codes = np.zeros((len(df), len(CATS)), np.int64)
for i, c in enumerate(CATS):
    lut = {v: j + 1 for j, v in enumerate(levels[c])}          # ★ +1, 0 = 미지
    codes[:, i] = df[c].astype(str).map(lut).fillna(0).astype(np.int64).values
X = df[DENSE].astype("float32").values
assert list(df[DENSE].columns) == META["DENSE"] and X.shape[1] == 73
mu = np.nanmean(X[m_tr], axis=0).astype(np.float32)
sd = np.nanstd(X[m_tr], axis=0).astype(np.float32)
sd[sd < 1e-6] = 1.0
X = (X - mu) / sd                                              # ★ 표준화 먼저
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)   # ★ 그 다음 0
y = df["control_success"].to_numpy(np.float32)
row24 = df.loc[m_24, "row_id"].to_numpy()

INFO = {
    "n_train": int(m_tr.sum()), "n_2024": int(m_24.sum()),
    "levels_n": {c: len(levels[c]) for c in CATS},
    "levels_n_배포": {c: len(META["levels"][c]) for c in CATS},
    "unknown_cells_2024": {c: int((codes[m_24, i] == 0).sum())
                           for i, c in enumerate(CATS)},
    "unknown_rows_2024_any": int((codes[m_24] == 0).any(1).sum()),
    "mu_sd_출처": "우리 학습행(1,116,277) — 배포 meta 값 재사용 안 함",
    "mu_max_abs_diff_vs_배포": float(np.max(np.abs(mu - np.array(META["mu"], np.float32)))),
    "sd_max_abs_diff_vs_배포": float(np.max(np.abs(sd - np.array(META["sd"], np.float32)))),
    "레시피_우리변인": {"dropout": DROPOUT, "opt": "AdamW", "wd": WD, "batch": BATCH,
                        "drop_last": True, "sched": "OneCycleLR", "max_lr": MAX_LR,
                        "pct_start": PCT_START, "div_factor": 25.0,
                        "final_div_factor": 1e4, "sched_epochs": EPOCHS,
                        "loss": "BCEWithLogitsLoss", "출처": "exp_064/068 레시피 승계"},
}
print(json.dumps({k: INFO[k] for k in ("n_train", "n_2024", "levels_n",
                                       "unknown_rows_2024_any")}, ensure_ascii=False),
      flush=True)

Xt = torch.from_numpy(X[m_tr]).to(DEV)
Ct = torch.from_numpy(codes[m_tr]).to(DEV)
yt = torch.from_numpy(y[m_tr]).to(DEV)
Xv = torch.from_numpy(X[m_24]).to(DEV)
Cv = torch.from_numpy(codes[m_24]).to(DEV)
yv = y[m_24].astype(np.float64)
n_tr = int(m_tr.sum())
SPE = n_tr // BATCH
NLEV = [len(levels[c]) for c in CATS]
print(f"[split] steps/epoch {SPE}", flush=True)


class DeployNN(nn.Module):
    """배포 구조 그대로. ★ BatchNorm 은 앞 두 블록에만."""

    def __init__(self, nlev, edims, n_dense, h1=256, p_drop=0.30):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n + 1, d) for n, d in zip(nlev, edims)])
        d_in = sum(edims) + n_dense
        h2, h3 = h1 // 2, h1 // 4
        self.mlp = nn.Sequential(
            nn.Linear(d_in, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(p_drop),
            nn.Linear(h2, h3), nn.ReLU(),
            nn.Linear(h3, 1))

    def forward(self, c, xd):
        z = torch.cat([e(c[:, i]) for i, e in enumerate(self.emb)] + [xd], dim=1)
        return self.mlp(z).squeeze(1)


def export_npz(model, path):
    sd_ = model.state_dict()
    out = {}
    for i, e in enumerate(model.emb):
        out[f"emb{i}"] = sd_[f"emb.{i}.weight"].cpu().numpy().astype(np.float32)
    for k, v in sd_.items():
        if k.startswith("mlp."):
            out["mlp_" + k[4:].replace(".", "_")] = v.cpu().numpy()
    np.savez(path, **out)


def seed_all(sd_):
    random.seed(sd_); np.random.seed(sd_)
    torch.manual_seed(sd_); torch.cuda.manual_seed_all(sd_)


@torch.no_grad()
def pred24(model):
    model.eval()
    ps = []
    for a in range(0, len(Xv), 65536):
        z = model(Cv[a:a + 65536], Xv[a:a + 65536])
        ps.append(torch.sigmoid(z).double().cpu().numpy())
    return np.concatenate(ps)


CURVES, STATES = {}, {}
EPRED = {}
for sd_ in SEEDS:
    t = time.time()
    seed_all(sd_)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    model = DeployNN(NLEV, EDIMS, len(DENSE), h1=H1, p_drop=DROPOUT).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=MAX_LR, total_steps=EPOCHS * SPE, pct_start=PCT_START,
        div_factor=25.0, final_div_factor=1e4)
    lossf = nn.BCEWithLogitsLoss()
    gcpu = torch.Generator().manual_seed(sd_)
    P = np.empty((EPOCHS, len(yv)), np.float64)
    cur = []
    for ep in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_tr, generator=gcpu)
        el, nb = 0.0, 0
        for a in range(0, SPE * BATCH, BATCH):
            idx = perm[a:a + BATCH].to(DEV)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(Ct[idx], Xt[idx]), yt[idx])
            loss.backward(); opt.step(); sch.step()
            el += loss.item(); nb += 1
        p = pred24(model)
        P[ep - 1] = p
        br = float(np.mean((p - yv) ** 2))
        cur.append({"epoch": ep, "train_logloss": el / nb, "brier_2024": br,
                    "lr": sch.get_last_lr()[0]})
        STATES[(sd_, ep)] = {k: v.detach().cpu().clone()
                             for k, v in model.state_dict().items()}
    CURVES[sd_] = cur
    EPRED[sd_] = P
    np.save(f"{CACHE}/p24_epochs_seed{sd_}.npy", P.astype(np.float32))
    be = int(min(cur, key=lambda r: r["brier_2024"])["epoch"])
    print(f"[seed {sd_}] {(time.time()-t)/60:.2f}분  best_epoch {be} "
          f"brier {min(r['brier_2024'] for r in cur):.6f}  "
          f"last {cur[-1]['brier_2024']:.6f}", flush=True)

REF = 0.2498069
conv = lambda b: 100000.0 * (1.0 - b / REF)
lg = lambda p: np.log(np.clip(p, 1e-12, 1 - 1e-12) / (1 - np.clip(p, 1e-12, 1 - 1e-12)))
expit = lambda z: 1.0 / (1.0 + np.exp(-z))

best_ep = {sd_: int(min(CURVES[sd_], key=lambda r: r["brier_2024"])["epoch"])
           for sd_ in SEEDS}
k = int(np.floor(np.median(list(best_ep.values()))))
INFO["best_epoch_per_seed"] = best_ep
INFO["k_고정"] = k
INFO["k_유도"] = (f"시드별 best epoch {sorted(best_ep.values())} 의 중앙값 "
                  f"{np.median(list(best_ep.values()))} 내림 → k={k}")


def ens(idx_by_seed):
    L = np.mean([lg(EPRED[sd_][idx_by_seed[sd_] - 1]) for sd_ in SEEDS], axis=0)
    return expit(L)


p_k = ens({sd_: k for sd_ in SEEDS})
p_best = ens(best_ep)
p_last = ens({sd_: EPOCHS for sd_ in SEEDS})
ENS = {}
for kk in [2, 4, 6, 8, 10, 12, 14, 16, 20, 25]:
    pk = ens({sd_: kk for sd_ in SEEDS})
    b = float(np.mean((pk - yv) ** 2))
    ENS[str(kk)] = {"brier": b, "score_REF_raw": conv(b)}
INFO["k_강건성_원시Brier"] = ENS
INFO["앙상블_원시"] = {
    "k고정": {"k": k, "brier": float(np.mean((p_k - yv) ** 2)),
              "score_REF_raw": conv(float(np.mean((p_k - yv) ** 2)))},
    "best_epoch": {"brier": float(np.mean((p_best - yv) ** 2)),
                   "score_REF_raw": conv(float(np.mean((p_best - yv) ** 2)))},
    "last25": {"brier": float(np.mean((p_last - yv) ** 2)),
               "score_REF_raw": conv(float(np.mean((p_last - yv) ** 2)))},
    "시드별_k고정": {str(sd_): float(np.mean((EPRED[sd_][k - 1] - yv) ** 2)) for sd_ in SEEDS},
    "🔴": "원시 Brier — 재센터링 전. exp_070 곡면 점수와 직접 비교 금지"}

np.save(f"{OUT}/p_nn_oot2024.npy", p_k.astype(np.float64))
np.save(f"{OUT}/p_nn_oot2024_bestepoch.npy", p_best.astype(np.float64))
np.save(f"{OUT}/p_nn_oot2024_last25.npy", p_last.astype(np.float64))
np.save(f"{OUT}/row_id_oot2024_ours.npy", row24)

our_meta = {"CATS": CATS, "levels": levels, "DENSE": DENSE,
            "mu": [float(x) for x in mu], "sd": [float(x) for x in sd],
            "edims": EDIMS, "H1": H1}
for sd_ in SEEDS:
    m2 = DeployNN(NLEV, EDIMS, len(DENSE), h1=H1, p_drop=DROPOUT).to(DEV)
    m2.load_state_dict(STATES[(sd_, k)])
    export_npz(m2, f"{MODEL}/nn_ours_oot_a{sd_}.npz")
    json.dump(our_meta, open(f"{MODEL}/nn_ours_oot_a{sd_}_meta.json", "w"))

# 왕복 검증 — 배포 nn_infer 가 우리 npz 를 그대로 읽는가
sys.path.insert(0, f"{E}/mate")
import nn_infer  # noqa: E402
sub = df.loc[m_24].head(20000).reset_index(drop=True)
rt = nn_infer.predict_nn(sub, f"{MODEL}/nn_ours_oot_a42.npz",
                         f"{MODEL}/nn_ours_oot_a42_meta.json")
INFO["출하왕복_max_abs_vs_torch"] = float(np.max(np.abs(rt - EPRED[42][k - 1][:20000])))
print(f"[왕복] nn_infer vs torch  max|Δ| = {INFO['출하왕복_max_abs_vs_torch']:.3e}", flush=True)

INFO["curves"] = {str(s_): CURVES[s_] for s_ in SEEDS}
INFO["train_time_min"] = (time.time() - T0) / 60.0
json.dump(INFO, open(f"{OUT}/train071.json", "w"), indent=1, ensure_ascii=False)
print(f"[train] 총 {INFO['train_time_min']:.2f}분  k={k} "
      f"best_epoch={best_ep}", flush=True)
print("DONE_TRAIN071", flush=True)

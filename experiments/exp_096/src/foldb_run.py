"""exp_096 진단 — 독립 fold(<=2022 -> 2023) 에서 rank vs 표준화 팔 단독 대조.

사용: foldb_run.py <INPUT std|rank> <SEED> <DEV> <TAG>
구조·최적화기·스텝은 fold_2024 판본과 **글자 그대로 같다** (TokenMLP D=96, AdamW 0.01,
lr 1e-3 상수, batch 4096, 1632스텝, init ftt).  바뀌는 것은 학습 구간과 입력 행렬뿐이다.
"""
import hashlib, json, os, random, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

INP, SEED, DEV, TAG = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
H = "/home/<user>/SungjinPark"
B, W = f"{H}/exp096", f"{H}/exp096/workb"
sys.path.insert(0, f"{B}/src")
from arch096 import TokenMLP, n_params                      # noqa: E402

LR, BATCH, TOTAL = 1e-3, 4096, 1632
os.makedirs(f"{B}/out/predsb", exist_ok=True)
torch.set_num_threads(1)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CARDS = json.load(open(f"{W}/schema.json"))["cards"]
XP = f"{W}/X_rank_le22.npy" if INP == "rank" else f"{W}/X_le22.npy"
X = np.load(XP, mmap_mode="r")
C = np.load(f"{W}/codes_le22.npy", mmap_mode="r")
y = np.load(f"{W}/y.npy")
trm = np.load(f"{W}/trm.npy")
vam = np.load(f"{W}/vam.npy")
n_tr = int(trm.sum())
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)

model = TokenMLP(Xt.shape[1], CARDS, D=96, p_drop=0.0, init="ftt").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
t0 = time.time()
curve, done, ep = [], 0, 0
while done < TOTAL:
    ep += 1
    model.train()
    perm = torch.randperm(n_tr, generator=g)
    tot, cnt = 0.0, 0
    for s in range(0, steps * BATCH, BATCH):
        if done >= TOTAL:
            break
        idx = perm[s:s + BATCH].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(Ct[idx], Xt[idx]), yt[idx])
        loss.backward(); opt.step()
        tot += loss.item(); cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_logloss": tot / max(cnt, 1)})
model.eval()
with torch.no_grad():
    z = torch.cat([model(Cv[s:s + 32768], Xv[s:s + 32768]).double().cpu()
                   for s in range(0, len(yv), 32768)]).numpy()
p = 1.0 / (1.0 + np.exp(-z))
br = float(np.mean((p - yv) ** 2))
np.save(f"{B}/out/predsb/z_{TAG}.npy", z)
res = {"tag": TAG, "input": INP, "seed": SEED, "X_path": XP,
       "n_train": n_tr, "n_valid": int(len(yv)), "total_steps": TOTAL,
       "n_params": n_params(model), "valid_brier_raw": br,
       "valid_score_raw": 100000.0 * (1.0 - br / 0.2498069),
       "z_sd": float(z.std()), "fit_wall_sec": round(time.time() - t0, 1),
       "curve": curve,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{B}/out/fitb_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False))

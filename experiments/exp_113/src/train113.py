"""exp_113 — 딥 팔 1 fit. exp_103b train103b.py 의 코드 경로를 **그대로** 승계한다.

사용: train113.py <SEED> <DEV> <STEPS> <TAG> [<EXTRA_BLK.npy>]
바뀐 것 없음 — 인자만 줄였다(SPLIT=f64o, INIT=ftt, DROP=0, OPT=adamw0.01 고정).
동결 레시피: lr 1e-3 상수(스케줄러 없음) · batch 4096 · D 96 · AdamW(wd 0.01) · dropout 0 · init ftt.

🔴 EXTRA 가 없으면 94열 경로 = exp_097/exp_102 판본과 **비트 동일**해야 한다(게이트).
🔴 `deep_infer._fold` 대신 일반화한 fold — 배포 함수가 토큰 수 99 를 하드코딩한다(S101).
   n_num == 94 일 때 배포 `_fold` 와 비트 동일함을 매 fit 확인한다.
"""
import hashlib
import json
import os
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

SEED, DEV, TOTAL, TAG = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
EXTRA = sys.argv[5] if len(sys.argv) > 5 else ""

H = "/home/<user>/SungjinPark"
B = f"{H}/exp113"
# 🔴 읽기 전용. EXP113_W 로 tmpfs 사본을 가리킬 수 있다 — **바이트가 같으므로 z 는 안 변한다**
#    (게이트: 같은 seed·steps 를 두 경로에서 돌려 z sha256 동일 확인).
W = os.environ.get("EXP113_W", f"{H}/exp102/work")
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{H}/exp102/src")
import deep_infer as DI                      # noqa: E402
from arch102 import TokenMLP, raw_npz        # noqa: E402

LR, BATCH, SPLIT = 1e-3, 4096, "f64o"
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/fits", exist_ok=True)
if os.path.exists(f"{OUT}/fits/{TAG}.json"):
    print(f"skip {TAG}")
    sys.exit(0)
torch.set_num_threads(1)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CARDS = json.load(open(f"{W}/schema_{SPLIT}.json"))["cards"]
X = np.load(f"{W}/X_{SPLIT}.npy", mmap_mode="r")
C = np.load(f"{W}/codes_{SPLIT}.npy", mmap_mode="r")
y = np.load(f"{W}/y.npy")
trm = np.load(f"{W}/trm_{SPLIT}.npy")
vam = np.load(f"{W}/valid_mask.npy")
n_tr = int(trm.sum())
dev = torch.device(DEV)


def block(mask):
    A = np.ascontiguousarray(X[mask])
    if EXTRA:
        E = np.load(EXTRA, mmap_mode="r")
        A = np.ascontiguousarray(np.hstack([A, np.asarray(E[mask], np.float32)]))
    return A


Xt = torch.from_numpy(block(trm)).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(block(vam)).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)
N_NUM = Xt.shape[1]

model = TokenMLP(N_NUM, CARDS, p_drop=0.0, init="ftt").to(dev)
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
        loss.backward()
        opt.step()
        tot += loss.item(); cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_logloss": tot / max(cnt, 1)})

model.eval()
with torch.no_grad():
    zz = [model(Cv[s:s + 65536], Xv[s:s + 65536]).double().cpu()
          for s in range(0, len(yv), 65536)]
    z = torch.cat(zz).numpy()
wall = time.time() - t0
br = float(np.mean((1.0 / (1.0 + np.exp(-z)) - yv) ** 2))


def fold_gen(Wt, n_num, n_cat):
    """배포 `_fold` 의 상수만 일반화. 산식은 글자 그대로 같다."""
    W0 = Wt["m.0.weight"]
    D = Wt["w"].shape[1]
    B0 = W0.reshape(W0.shape[0], n_num + n_cat, D)
    A = np.einsum("id,oid->io", Wt["w"], B0[:, :n_num, :])
    c0 = np.einsum("id,oid->o", Wt["b"], B0[:, :n_num, :]) + Wt["m.0.bias"]
    T = [Wt[f"ce.{j}.weight"] @ B0[:, n_num + j, :].T for j in range(n_cat)]
    return A.astype(np.float64), c0.astype(np.float64), T


raw = raw_npz(model.cpu())
A, c0, T = fold_gen(raw, N_NUM, len(CARDS))
gate = None
if N_NUM == 94:
    A2, c2, T2 = DI._fold(raw)
    gate = {"fold_A": float(np.abs(A - A2).max()), "fold_c0": float(np.abs(c0 - c2).max()),
            "fold_T": float(max(np.abs(T[j] - T2[j]).max() for j in range(len(T))))}

np.save(f"{OUT}/preds/z_{TAG}.npy", z)
res = {"tag": TAG, "seed": SEED, "n_num": int(N_NUM), "extra": EXTRA,
       "total_steps": TOTAL, "n_train": n_tr, "lr": LR, "batch": BATCH,
       "lr_schedule": "constant (no scheduler)", "opt": "adamw0.01",
       "init": "ftt", "drop": 0.0,
       "valid_brier_raw": br, "valid_score_raw": 100000.0 * (1 - br / 0.2498069),
       "z_sd": float(z.std()), "fit_wall_sec": round(wall, 1),
       "deployed_fold_gate": gate, "curve": curve,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{OUT}/fits/{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, ensure_ascii=False))

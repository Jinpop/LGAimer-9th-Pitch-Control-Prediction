"""exp_114 — 딥 팔 1 fit + 중간 체크포인트 z 덤프. train113.py 의 학습 루프를 **글자 그대로** 승계한다.

사용: train114.py <SEED> <DEV> <TOTAL> <TAG> [<FRAC>]
  FRAC 은 학습 행 부분표집 비율(기본 1.0). 부분표집은 **모델 시드와 무관한 고정 난수**로 뽑는다
  — 그래야 부분표집 잡음이 시드 잡음과 안 섞인다.

🔴 S98 이용: LR 스케줄러가 없고 상수 lr 이라 같은 시드면 긴 적합의 앞부분이 짧은 적합과 비트 동일하다.
   그래서 4352 를 한 번 굽고 CKPTS 지점마다 z 를 뜬다. **그 비트 동일성은 게이트로 확인한다**
   (seed42 의 1632 체크포인트 == exp_097 z_F1632_f64o_s42, sha16 b1da6808c7f16dea).
🔴 체크포인트 평가가 학습을 건드리면 안 된다 — dropout 0 · LayerNorm(러닝 통계 없음) · no_grad ·
   순열 생성기 g 는 CPU 이고 forward 가 안 건드린다. 그래도 판정은 sha 대조로 한다.
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
FRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0

H = "/home/<user>/SungjinPark"
B = f"{H}/exp114"
W = os.environ.get("EXP113_W", f"{H}/exp102/work")
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{H}/exp102/src")
import deep_infer as DI                      # noqa: E402
from arch102 import TokenMLP, raw_npz        # noqa: E402

LR, BATCH, SPLIT = 1e-3, 4096, "f64o"
SUBSAMPLE_RNG = 20260827          # 모델 시드와 분리된 고정 난수
CKPTS = [544, 720, 816, 952, 1088, 1224, 1360, 1496, 1632, 1768, 1904, 2040,
         2160, 2176, 2312, 2448, 2720, 2992, 3264, 3536, 3808, 4080, 4352]
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
n_full = int(trm.sum())
if FRAC < 1.0:
    idx_all = np.flatnonzero(trm)
    keep = np.random.default_rng(SUBSAMPLE_RNG).permutation(len(idx_all))[
        :int(round(len(idx_all) * FRAC))]
    keep.sort()
    trm = np.zeros_like(trm)
    trm[idx_all[keep]] = True
n_tr = int(trm.sum())
dev = torch.device(DEV)

Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)
N_NUM = Xt.shape[1]
assert N_NUM == 94, N_NUM

model = TokenMLP(N_NUM, CARDS, p_drop=0.0, init="ftt").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
t0 = time.time()


def dump(k):
    model.eval()
    with torch.no_grad():
        zz = [model(Cv[s:s + 65536], Xv[s:s + 65536]).double().cpu()
              for s in range(0, len(yv), 65536)]
        z = torch.cat(zz).numpy()
    model.train()
    np.save(f"{OUT}/preds/z_{TAG}__k{k}.npy", z)
    br = float(np.mean((1.0 / (1.0 + np.exp(-z)) - yv) ** 2))
    return {"steps": k, "valid_brier_raw": br,
            "valid_score_raw": 100000.0 * (1 - br / 0.2498069),
            "z_sd": float(z.std()), "z_mean": float(z.mean()),
            "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}


want = sorted(k for k in CKPTS if k <= TOTAL)
ck, curve, done, ep = [], [], 0, 0
model.train()
while done < TOTAL:
    ep += 1
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
        if done in want:
            ck.append(dump(done))
    curve.append({"epoch": ep, "steps": cnt, "train_logloss": tot / max(cnt, 1)})

wall = time.time() - t0


def fold_gen(Wt, n_num, n_cat):
    W0 = Wt["m.0.weight"]
    D = Wt["w"].shape[1]
    B0 = W0.reshape(W0.shape[0], n_num + n_cat, D)
    A = np.einsum("id,oid->io", Wt["w"], B0[:, :n_num, :])
    c0 = np.einsum("id,oid->o", Wt["b"], B0[:, :n_num, :]) + Wt["m.0.bias"]
    T = [Wt[f"ce.{j}.weight"] @ B0[:, n_num + j, :].T for j in range(n_cat)]
    return A.astype(np.float64), c0.astype(np.float64), T


raw = raw_npz(model.cpu())
A, c0, T = fold_gen(raw, N_NUM, len(CARDS))
A2, c2, T2 = DI._fold(raw)
gate = {"fold_A": float(np.abs(A - A2).max()), "fold_c0": float(np.abs(c0 - c2).max()),
        "fold_T": float(max(np.abs(T[j] - T2[j]).max() for j in range(len(T))))}

res = {"tag": TAG, "seed": SEED, "frac": FRAC, "n_num": int(N_NUM),
       "n_train_full": n_full, "n_train": n_tr, "steps_per_epoch": steps,
       "total_steps": TOTAL, "lr": LR, "batch": BATCH,
       "lr_schedule": "constant (no scheduler)", "opt": "adamw0.01",
       "init": "ftt", "drop": 0.0, "subsample_rng": SUBSAMPLE_RNG,
       "fit_wall_sec": round(wall, 1), "deployed_fold_gate": gate,
       "ckpts": ck, "curve": curve}
json.dump(res, open(f"{OUT}/fits/{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"tag": TAG, "n_tr": n_tr, "spe": steps, "wall": round(wall, 1),
                  "n_ck": len(ck), "gate": gate}, ensure_ascii=False))

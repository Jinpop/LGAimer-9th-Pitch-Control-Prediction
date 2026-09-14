"""exp_103 stage2 — 확장 입력(111열) 위의 딥 팔 1 fit.

exp_097 train097.py 의 코드 경로를 그대로 승계한다. 바뀐 것은 **입력 X 하나**:
    exp_097 fix : X_f64o.npy      (94열)
    exp_103     : X111_f64o.npy   (94열 + 타자 폼 17열, 앞 94열은 비트 동일)
레시피 동결: lr 1e-3 상수(스케줄러 없음) · batch 4096 · D 96 · AdamW(wd 0.01) · dropout 0 · init ftt.
사용: s2_train.py <SEED> <DEV> <STEPS> <TAG> [XFILE]
"""
import hashlib, json, os, random, sys, time
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np, torch, torch.nn as nn

SEED, DEV, TOTAL, TAG = int(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
XFILE = sys.argv[5] if len(sys.argv) > 5 else "X111_f64o.npy"
H = "/home/<user>/SungjinPark"
B, W97 = f"{H}/exp103", f"{H}/exp097/work"
sys.path.insert(0, f"{H}/experiments/exp_097/src")
sys.path.insert(0, f"{B}/src")
from arch097 import TokenMLP, raw_npz  # noqa: E402

LR, BATCH = 1e-3, 4096
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
torch.set_num_threads(2)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

CARDS = json.load(open(f"{W97}/schema_oot23.json"))["cards"]
X = np.load(f"{B}/work/{XFILE}" if XFILE.startswith("X111") else f"{W97}/{XFILE}",
            mmap_mode="r")
C = np.load(f"{W97}/codes_oot23.npy", mmap_mode="r")
y = np.load(f"{W97}/y.npy")
trm = np.load(f"{W97}/trm_oot23.npy")
vam = np.load(f"{W97}/valid_mask.npy")
n_tr = int(trm.sum())
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)

model = TokenMLP(Xt.shape[1], CARDS, p_drop=0.0, init="ftt").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
t0 = time.time(); curve = []; done = 0; ep = 0
while done < TOTAL:
    ep += 1
    model.train()
    perm = torch.randperm(n_tr, generator=g)
    tot = cnt = 0
    for s in range(0, steps * BATCH, BATCH):
        if done >= TOTAL:
            break
        idx = perm[s:s + BATCH].to(dev)
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(Ct[idx], Xt[idx]), yt[idx])
        loss.backward(); opt.step()
        tot += loss.item(); cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_logloss": tot / max(cnt, 1)})
    print(f"ep {ep} steps {cnt} loss {tot/max(cnt,1):.6f}", flush=True)

model.eval()
with torch.no_grad():
    zz = [model(Cv[s:s + 65536], Xv[s:s + 65536]).double().cpu()
          for s in range(0, len(yv), 65536)]
    z = torch.cat(zz).numpy()
np.save(f"{OUT}/preds/z_{TAG}.npy", z)
br = float(np.mean((1 / (1 + np.exp(-z)) - yv) ** 2))
res = {"tag": TAG, "seed": SEED, "xfile": XFILE, "n_num": int(Xt.shape[1]),
       "total_steps": TOTAL, "n_train": n_tr, "lr": LR, "batch": BATCH,
       "lr_schedule": "constant (no scheduler)", "opt": "adamw0.01",
       "init": "ftt", "drop": 0.0, "cards": CARDS,
       "valid_brier_raw": br, "valid_score_raw": 100000.0 * (1 - br / 0.2498069),
       "z_sd": float(z.std()), "fit_wall_sec": round(time.time() - t0, 1),
       "curve": curve,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{OUT}/fit_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "cards"}, ensure_ascii=False))

"""exp_091 — 딥 팔 1 fit. **exp_090 train090.py 의 코드 경로를 그대로 승계한다.**

사용: train091.py <SPLIT le2024|oot23> <SEED> <DEV> <INIT> <DROP> <OPT> <TAG>
환경: STEPS091=<총 옵티마이저 스텝>   (0 이면 steps_per_epoch x EPOCH)
      EP091=<epoch>                  (기본 2 — STEPS091 이 있으면 무시된다)
      HID091=256,128,64              (은닉 폭·깊이. 기본 = 현행 구조)
      NT091=4                        (torch CPU 스레드. 기본 4 = exp_090 과 동일)

exp_090 대비 **바뀐 것 넷뿐이고 기본값에서는 전부 z 에 안 닿는다**:
  ① 산출 디렉터리가 exp091/out
  ② 환경변수 이름 STEPS090→STEPS091, EP090→EP091
  ③ arch091.TokenMLP 의 `hid` 인자 (기본값 (256,128,64) = arch090 과 레이어 인덱스까지 동일)
  ④ CPU 스레드 수를 NT091 로 뺐다 (기본 4 = exp_090)
🔴 ①~④ 가 z 를 안 바꾼다는 것은 **재현 게이트**로 증명한다: exp_090 의 K1632_oot23_s42 를
   같은 인자로 다시 적합해 z_sha256 이 같은지 본다. NT091=1 로도 한 번 더 돌려
   **스레드 수가 z 를 안 바꾼다는 것까지 기계로 확인**한 뒤에만 동시 실행 수를 올린다.

동결 레시피 (팀원 prep.json): lr 1e-3 상수(스케줄러 없음) · batch 4096 · D 96 · init ftt.
이번 실험이 움직이는 축: 시드 수 / 은닉 폭·깊이 / dropout · weight decay. 학습량은 1632 고정.
"""
import hashlib, json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

SPLIT = sys.argv[1]
SEED = int(sys.argv[2])
DEV = sys.argv[3]
INIT = sys.argv[4]
DROP = float(sys.argv[5])
OPT = sys.argv[6]
TAG = sys.argv[7]

H = "/home/<user>/SungjinPark"
B = f"{H}/exp091"
W89 = f"{H}/exp089/work"                      # 🔴 읽기 전용
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                       # noqa: E402
from arch091 import TokenMLP, raw_npz, foldable, HID_BASE   # noqa: E402

LR, BATCH = 1e-3, 4096
EPOCH = int(os.environ.get("EP091", "2"))
HID = tuple(int(x) for x in os.environ.get(
    "HID091", ",".join(str(h) for h in HID_BASE)).split(","))
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/npz", exist_ok=True)
torch.set_num_threads(int(os.environ.get("NT091", "4")))
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sch = json.load(open(f"{W89}/schema_{SPLIT}.json"))
CARDS = sch["cards"]
X = np.load(f"{W89}/X_{SPLIT}.npy", mmap_mode="r")
C = np.load(f"{W89}/codes_{SPLIT}.npy", mmap_mode="r")
y = np.load(f"{W89}/y.npy")
trm = np.load(f"{W89}/trm_{SPLIT}.npy")
vam = np.load(f"{W89}/valid_mask.npy")
n_tr = int(trm.sum())
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)

model = TokenMLP(Xt.shape[1], CARDS, p_drop=DROP, init=INIT, hid=HID).to(dev)
if OPT == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=LR)
elif OPT.startswith("adamw"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(OPT[len("adamw"):]))
else:
    raise ValueError(OPT)
lossf = nn.BCEWithLogitsLoss()
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
TOTAL = int(os.environ.get("STEPS091", "0")) or steps * EPOCH
t0 = time.time()
curve = []
done, ep = 0, 0
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
    print(f"ep {ep} steps {cnt} loss {tot/max(cnt,1):.6f}", flush=True)

model.eval()
with torch.no_grad():
    zz = []
    for s in range(0, len(yv), 65536):
        zz.append(model(Cv[s:s + 65536], Xv[s:s + 65536]).double().cpu())
    z = torch.cat(zz).numpy()
wall = time.time() - t0
p = 1.0 / (1.0 + np.exp(-z))
br = float(np.mean((p - yv) ** 2))

raw = raw_npz(model.cpu())
fp = {}
FOLD = foldable(model)
if FOLD:                                   # 깊이 3 에서만 팀원 _fold 가 먹는다
    A, c0, T = DI._fold(raw)
    fp = {"A": float(A.std()), "c0": float(c0.std())}
    fp.update({f"T{j}": float(T[j].std()) for j in range(5)})
    for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
              "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight"):
        fp[k] = float(raw[k].std())
    fp["m.10.bias_value"] = float(raw["m.10.bias"].ravel()[0])

np.save(f"{OUT}/preds/z_{TAG}.npy", z)

npz_path = None
if os.environ.get("SAVE_NPZ") == "1" and FOLD:
    out = {"A": A.astype(np.float64), "c0": c0.astype(np.float64)}
    for j in range(5):
        out[f"T{j}"] = T[j].astype(np.float32)
    for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
              "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight", "m.10.bias"):
        out[k] = raw[k].astype(np.float32)
    npz_path = f"{OUT}/npz/deepf_{TAG}.npz"
    np.savez(npz_path, **out)

res = {"tag": TAG, "split": SPLIT, "seed": SEED, "init": INIT, "drop": DROP,
       "opt": OPT, "lr": LR, "lr_schedule": "constant (no scheduler)",
       "hid": list(HID), "depth": len(HID), "foldable_by_mate_deep_infer": FOLD,
       "torch_threads": int(os.environ.get("NT091", "4")),
       "epoch_env": EPOCH, "batch": BATCH, "cards": CARDS,
       "n_train": n_tr, "n_valid": int(len(yv)), "steps_per_epoch": steps,
       "total_steps": int(TOTAL),
       "n_params": int(sum(q.numel() for q in model.parameters())),
       "valid_brier_raw": br,
       "valid_score_raw": 100000.0 * (1.0 - br / 0.2498069),
       "z_sd": float(z.std()), "z_mean": float(z.mean()),
       "fit_wall_sec": round(wall, 1), "curve": curve,
       "folded_fingerprint_sd": fp, "npz_path": npz_path,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{OUT}/fit_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("cards", "folded_fingerprint_sd", "curve")},
                 ensure_ascii=False))

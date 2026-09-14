"""exp_090 — 딥 팔 1 fit. **exp_089 train089.py 의 코드 경로를 그대로 승계한다.**

사용: train090.py <SPLIT le2024|oot23> <SEED> <DEV> <INIT> <DROP> <OPT> <TAG>
환경: STEPS090=<총 옵티마이저 스텝>   (0 이면 steps_per_epoch x EPOCH)
      EP090=<epoch>                  (기본 2 — STEPS090 이 있으면 무시된다)
      SAVE_NPZ=1                     (접힌 배포 npz 를 out/npz/ 에 쓴다. z 계산 뒤에만 실행)

exp_089 대비 **바뀐 것 셋뿐이고 전부 z 에 닿지 않는다**:
  ① 입력 work 디렉터리를 exp089/work 로 읽는다 (읽기 전용 재사용 — prep 을 다시 안 돌린다)
  ② 산출 디렉터리가 exp090/out
  ③ SAVE_NPZ 분기 추가 — z 를 다 계산하고 model.cpu() 한 뒤에만 파일을 쓴다
  ⓐ 환경변수 이름 STEPS089→STEPS090, EP089→EP090
🔴 이것이 z 를 안 바꾼다는 것은 **재현 게이트**로 증명한다: exp_089 의 S720_oot23_s42 를
   같은 인자로 다시 적합해 z_sha256 이 같은지 본다.

동결 레시피 (팀원 prep.json): lr 1e-3 · batch 4096 · D 96 · AdamW(wd 0.01) · dropout 0 · init ftt.
🔴 **학습률 스케줄러는 없다.** 팀원 `prep.json` 이 `lr` 을 단일값 0.001 로 기록했고
   exp_089 가 그 규약으로 복제해 ρ 0.985651 을 얻었다. 따라서 **총 스텝을 바꿔도 LR 궤적은
   안 바뀐다** — 매 스텝 lr = 1e-3 상수다. (OneCycle/코사인이었다면 스텝 수가 궤적을 통째로
   바꿔 "학습량" 축과 "스케줄 모양" 축이 섞였을 것이다. 여기서는 안 섞인다.)
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
B = f"{H}/exp102"
W89 = f"{H}/exp102/work"                      # 🔴 읽기 전용
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                       # noqa: E402
from arch102 import TokenMLP, raw_npz         # noqa: E402

LR, BATCH = 1e-3, 4096
EPOCH = int(os.environ.get("EP090", "2"))
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/npz", exist_ok=True)
torch.set_num_threads(4)
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

model = TokenMLP(Xt.shape[1], CARDS, p_drop=DROP, init=INIT).to(dev)
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
TOTAL = int(os.environ.get("STEPS090", "0")) or steps * EPOCH
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
A, c0, T = DI._fold(raw)
fp = {"A": float(A.std()), "c0": float(c0.std())}
fp.update({f"T{j}": float(T[j].std()) for j in range(5)})
for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
          "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight"):
    fp[k] = float(raw[k].std())
fp["m.10.bias_value"] = float(raw["m.10.bias"].ravel()[0])

np.save(f"{OUT}/preds/z_{TAG}.npy", z)

npz_path = None
if os.environ.get("SAVE_NPZ") == "1":
    # 🔴 팀원 배포 아티팩트와 **같은 접힌 형식** — 키 17개, dtype 도 같게 맞춘다.
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
                  if k not in ("cards", "folded_fingerprint_sd")},
                 ensure_ascii=False))

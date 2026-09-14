"""exp_089 — 복원한 딥 팔 1 fit.

사용: train089.py <SPLIT le2024|oot23> <SEED> <DEV> <INIT> <DROP> <OPT> <TAG>

동결 레시피 (팀원 `prep.json` 에서 읽은 것): lr 1e-3 · epoch 2 · batch 4096 · D 96.
probe 로 식별하는 것: INIT(w/b 초기화) · DROP(Dropout p) · OPT(adam|adamw<wd>).
손실은 BCEWithLogits, 스케줄러 없음(epoch 2 에 lr 이 단일값으로 기록돼 있다).

산출: valid(season==2024, 253,507행) 로짓 + 접힌 지문(A·c0·T·m.* 의 sd).
🔴 조기 종료·최선 에폭 선택 없음 — 팀원 팔에는 held-out 이 없다(≤2024 전량 학습).
   epoch 2 를 다 돌고 **마지막 상태**를 쓴다. 두 구간에 같은 규칙을 적용한다.
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
OPT = sys.argv[6]                      # "adam" | "adamw0.01" | "adamw0.0001"
TAG = sys.argv[7]

H = "/home/<user>/SungjinPark"
B = f"{H}/exp089"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                       # noqa: E402
from arch089 import TokenMLP, raw_npz         # noqa: E402

LR, BATCH = 1e-3, 4096
# 🔴 EPOCH 은 팀원 prep.json 이 2 로 못박은 값이다. 환경변수 EP089 는 **강건성 대조**
#    전용 — 학습 구간이 줄면 총 옵티마이저 스텝도 줄어드는 교란을 분리하려고 둔다.
EPOCH = int(os.environ.get("EP089", "2"))
W, OUT = f"{B}/work", f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
torch.set_num_threads(4)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False    # 재현성 — TF32 끔
torch.backends.cudnn.allow_tf32 = False

sch = json.load(open(f"{W}/schema_{SPLIT}.json"))
CARDS = sch["cards"]
X = np.load(f"{W}/X_{SPLIT}.npy", mmap_mode="r")
C = np.load(f"{W}/codes_{SPLIT}.npy", mmap_mode="r")
y = np.load(f"{W}/y.npy")
trm = np.load(f"{W}/trm_{SPLIT}.npy")
vam = np.load(f"{W}/valid_mask.npy")
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
# 🔴 STEPS089 는 **총 옵티마이저 스텝 수를 못박는** 강건성 대조 전용이다.
#    epoch 을 고정하면 학습 구간이 줄 때 스텝 수도 같이 줄어 "구간이 짧다"와
#    "덜 학습됐다" 두 변인이 겹친다. 배포 팔은 1,475,092//4096 x 2 = 720 스텝이다.
TOTAL = int(os.environ.get("STEPS089", "0")) or steps * EPOCH
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
res = {"tag": TAG, "split": SPLIT, "seed": SEED, "init": INIT, "drop": DROP,
       "opt": OPT, "lr": LR, "epoch": EPOCH, "batch": BATCH, "cards": CARDS,
       "n_train": n_tr, "n_valid": int(len(yv)), "steps_per_epoch": steps,
       "total_steps": int(TOTAL),
       "n_params": int(sum(q.numel() for q in model.parameters())),
       "valid_brier_raw": br,
       "valid_score_raw": 100000.0 * (1.0 - br / 0.2498069),
       "z_sd": float(z.std()), "z_mean": float(z.mean()),
       "fit_wall_sec": round(wall, 1), "curve": curve,
       "folded_fingerprint_sd": fp,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{OUT}/fit_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))

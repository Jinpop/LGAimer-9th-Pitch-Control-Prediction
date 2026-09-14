"""exp_096 — 새 팔 1 fit.  **exp_090 train090.py 의 코드 경로를 승계한다.**

사용: train096.py <ARM> <SEED> <DEV> <TAG>
  ARM ∈ {mlp96, mlpP<D>, attn, cos, rank, seq}
환경: STEPS096(기본 1632) · DROP(기본 0.0) · OPT(기본 adamw0.01) · MICRO(기본 0=전배치)
      ATT_D / ATT_BLK / ATT_HEAD / ATT_MULT — A1 형상
      SEQK(기본 16) · SEQD(기본 64) · SEQC(기본 64)

동결 레시피 (팀원 prep.json, exp_089/090 승계): lr 1e-3 상수 · batch 4096 · AdamW(wd 0.01)
· dropout 0 · init ftt · 스케줄러 없음.  **바꾸는 것만 인자로 노출한다.**

🔴 변인 기록: 각 fit 의 json 에 arm/steps/lr_schedule/입력행렬/파라미터수를 전부 적는다.
"""
import hashlib, json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

ARM = sys.argv[1]
SEED = int(sys.argv[2])
DEV = sys.argv[3]
TAG = sys.argv[4]

H = "/home/<user>/SungjinPark"
B = f"{H}/exp096"
W89 = f"{H}/exp089/work"                      # 🔴 읽기 전용
sys.path.insert(0, f"{B}/src")
from arch096 import TokenMLP, FTAttn, SeqArm, n_params    # noqa: E402

SPLIT = "oot23"                               # 정직면 — season<=2023 학습만 (config 규율)
LR, BATCH = 1e-3, 4096
DROP = float(os.environ.get("DROP", "0.0"))
OPT = os.environ.get("OPT", "adamw0.01")
INIT = "ftt"
TOTAL = int(os.environ.get("STEPS096", "1632"))
MICRO = int(os.environ.get("MICRO", "0")) or BATCH
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
torch.set_num_threads(1)                      # 🔴 CPU 4코어 상한 — GPU 레인 2개 x 1
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sch = json.load(open(f"{W89}/schema_{SPLIT}.json"))
CARDS = sch["cards"]
# 🔴 감사 후속(2026-08-25): mlp64 분기 추가. 기존 두 경로는 한 글자도 안 바뀐다.
XPATH = ({"rank": f"{B}/work/X_rank_{SPLIT}.npy",
          "mlp64": f"{B}/work/X_f64_{SPLIT}.npy"}.get(ARM, f"{W89}/X_{SPLIT}.npy"))
X = np.load(XPATH, mmap_mode="r")
C = np.load(f"{W89}/codes_{SPLIT}.npy", mmap_mode="r")
y = np.load(f"{W89}/y.npy")
trm = np.load(f"{W89}/trm_{SPLIT}.npy")
vam = np.load(f"{W89}/valid_mask.npy")

SEQ = ARM == "seq"
if SEQ:
    # 동결 시퀀스: (행 -> 표 인덱스) + (표 인덱스 -> K x Fs).  seqprep096.py 가 만든다.
    SEQV = os.environ.get("SEQ_VARIANT", "A")             # A=라벨 없음 / B=라벨 채널 켬
    SQ = np.load(f"{B}/work/seq_table.npy" if SEQV == "B"
                 else f"{B}/work/seq_table_A.npy")        # (n_key, Fs+1, K) float32
    SIDX = np.load(f"{B}/work/seq_rowidx.npy")            # (N,) int32, -1 = 이력 없음
    Fs = SQ.shape[1] - 1
    KSEQ = SQ.shape[2]
    # 🔴 S104 귀무 대조: 시퀀스 내용은 그대로 두고 **행<->시퀀스 연결만** 끊는다.
    #    파라미터 수·학습 행·주변 분포가 전부 같고 정보만 사라진다.
    SEQ_NULL = os.environ.get("SEQ_NULL", "0") == "1"
    if SEQ_NULL:
        SQ = SQ[np.random.default_rng(20260825).permutation(SQ.shape[0])]
    ZERO = np.zeros((1, SQ.shape[1], KSEQ), np.float32)
    SQ = np.concatenate([SQ, ZERO], 0)                    # 마지막 슬롯 = 이력 없음
    SIDX = np.where(SIDX < 0, SQ.shape[0] - 1, SIDX).astype(np.int64)
    # 🔴 학습은 이력 있는 행만 쓴다? 아니다 — 전 행을 쓰고 마스크 채널이 없음을 알린다.
    trm = trm & np.load(f"{B}/work/seq_trainable.npy")    # season >= 최소시즌+1 인 행만

n_tr = int(trm.sum())
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)
if SEQ:
    SQt = torch.from_numpy(SQ).to(dev)
    St = torch.from_numpy(SIDX[trm]).to(dev)
    Sv = torch.from_numpy(SIDX[vam]).to(dev)

n_num = Xt.shape[1]
if ARM in ("mlp96", "cos", "mlp64"):
    model = TokenMLP(n_num, CARDS, D=96, p_drop=DROP, init=INIT)
elif ARM == "rank":
    model = TokenMLP(n_num, CARDS, D=96, p_drop=DROP, init=INIT)
elif ARM.startswith("mlpP"):
    model = TokenMLP(n_num, CARDS, D=int(ARM[4:]), p_drop=DROP, init=INIT)
elif ARM == "attn":
    model = FTAttn(n_num, CARDS, D=int(os.environ.get("ATT_D", "128")),
                   blocks=int(os.environ.get("ATT_BLK", "3")),
                   heads=int(os.environ.get("ATT_HEAD", "8")),
                   mult=int(os.environ.get("ATT_MULT", "2")),
                   p_drop=DROP, init=INIT)
elif ARM == "seq":
    model = SeqArm(n_num, CARDS, Fs=Fs, K=KSEQ, D=int(os.environ.get("SEQD", "64")),
                   Cs=int(os.environ.get("SEQC", "64")), p_drop=DROP)
else:
    raise ValueError(ARM)
model = model.to(dev)
NP = n_params(model)

if OPT == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=LR)
elif OPT.startswith("adamw"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(OPT[len("adamw"):]))
else:
    raise ValueError(OPT)
SCHED = "cosine 1e-3 -> 1e-5" if ARM == "cos" else "constant 1e-3 (no scheduler)"
LR_MIN = 1e-5


def set_lr(step):
    if ARM != "cos":
        return LR
    lr = LR_MIN + 0.5 * (LR - LR_MIN) * (1.0 + np.cos(np.pi * step / TOTAL))
    for g_ in opt.param_groups:
        g_["lr"] = lr
    return float(lr)


lossf_sum = nn.BCEWithLogitsLoss(reduction="sum")
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
t0 = time.time()
curve, done, ep = [], 0, 0


def fwd(idx):
    if SEQ:
        return model(Ct[idx], Xt[idx], SQt[St[idx]])
    return model(Ct[idx], Xt[idx])


while done < TOTAL:
    ep += 1
    model.train()
    perm = torch.randperm(n_tr, generator=g)
    tot, cnt = 0.0, 0
    for s in range(0, steps * BATCH, BATCH):
        if done >= TOTAL:
            break
        cur_lr = set_lr(done)
        idx_all = perm[s:s + BATCH].to(dev)
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for m0 in range(0, BATCH, MICRO):
            idx = idx_all[m0:m0 + MICRO]
            loss = lossf_sum(fwd(idx), yt[idx]) / BATCH
            loss.backward()
            acc += loss.item()
        opt.step()
        tot += acc; cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_logloss": tot / max(cnt, 1),
                  "lr_at_epoch_start": cur_lr})
    print(f"ep {ep} steps {cnt} loss {tot/max(cnt,1):.6f} lr {cur_lr:.2e}", flush=True)

model.eval()
with torch.no_grad():
    zz = []
    # 🔴 감사 후속: attn 은 예측 청크 32768 에서 CUDA OOM 으로 죽었다(laneA fit 0건).
    PCH = int(os.environ.get("PRED_CHUNK", "32768"))
    for s in range(0, len(yv), PCH):
        sl = slice(s, s + PCH)
        out = (model(Cv[sl], Xv[sl], SQt[Sv[sl]]) if SEQ
               else model(Cv[sl], Xv[sl]))
        zz.append(out.double().cpu())
    z = torch.cat(zz).numpy()
wall = time.time() - t0
p = 1.0 / (1.0 + np.exp(-z))
br = float(np.mean((p - yv) ** 2))
np.save(f"{OUT}/preds/z_{TAG}.npy", z)

res = {"tag": TAG, "arm": ARM, "split": SPLIT, "seed": SEED, "init": INIT,
       "drop": DROP, "opt": OPT, "lr": LR, "lr_schedule": SCHED,
       "batch": BATCH, "micro_batch": MICRO, "cards": CARDS,
       "X_path": XPATH, "n_train": n_tr, "n_valid": int(len(yv)),
       "steps_per_epoch": steps, "total_steps": int(TOTAL), "n_params": NP,
       "valid_brier_raw": br,
       "valid_score_raw": 100000.0 * (1.0 - br / 0.2498069),
       "z_sd": float(z.std()), "z_mean": float(z.mean()),
       "fit_wall_sec": round(wall, 1), "curve": curve,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
if SEQ:
    res.update({"seq_variant": SEQV, "seq_null_control": SEQ_NULL,
                "seq_label_channel_used": SEQV == "B",
                "seq_K": int(KSEQ), "seq_Fs": int(Fs),
                "seq_D": int(os.environ.get("SEQD", "64")),
                "seq_C": int(os.environ.get("SEQC", "64"))})
if ARM == "attn":
    res.update({k: os.environ.get(k, d) for k, d in
                (("ATT_D", "128"), ("ATT_BLK", "3"), ("ATT_HEAD", "8"),
                 ("ATT_MULT", "2"))})
json.dump(res, open(f"{OUT}/fit_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "cards"},
                 ensure_ascii=False))

"""exp_093 — 딥 팔 1 fit. **exp_090/091/092 의 코드 경로를 그대로 승계한다.**

사용: train093.py <SPLIT le2024|oot23> <SEED> <DEV> <INIT> <DROP> <OPT> <TAG>
환경: STEPS093=<총 옵티마이저 스텝>  (0 이면 steps_per_epoch x EPOCH)
      EP093=<epoch>                 (기본 2 — STEPS093 가 있으면 무시)
      NT093=1                       (torch CPU 스레드. S102 — z 에 안 닿는다)
      TARGET093= bce | cells5 | collapsed5

🔴 **이 실험의 변인은 출력층·목적함수 둘뿐이다.** 그 외 전부 exp_090 과 같다
   (lr 1e-3 상수·batch 4096·D 96·init ftt·adamw0.01·drop 0·1632스텝·같은 프레임·같은 trm).

TARGET093 세 값
  bce         : `n_out=1` + `BCEWithLogitsLoss`  = exp_090/091/092 의 기준선.
                🔴 이 값에서 z 는 exp_090 과 **바이트 동일**해야 한다(재현 게이트).
  cells5      : `n_out=5` + 소프트 타깃 교차엔트로피. 클래스
                [is_mid_only, is_rev_only, is_both, is_wayoff_true, success].
                ★이 실험이 재려는 것★
  collapsed5  : `n_out=5` 인데 **실패 4셀에 균등 0.25** 를 준다.
                → 주변 성공확률 타깃이 cells5·bce 와 정확히 같고 **유형 정보만 없다.**
                🔑 그래서 `n_out` 1→5 가 딸고 오는 것(파라미터 +260 · 초기화 난수 소비 이동)의
                   몫만 따로 재는 **귀무 대조**다. cells5 의 ρ 를 이것 없이 해석하지 마라.

성공 로짓 읽기 (n_out=5)
  p_succ = softmax(logits)[:,4] = 1 − Σ(4셀)  →  z = logits[:,4] − logsumexp(logits[:,:4])
  (수학적으로 정확히 logit(p_succ). 나눗셈·클리핑 없음.)

🔴 리크 게이트
  · 타깃 `T5_oot23.npy` 는 예측면(2024) 행이 NaN 이다. 학습 마스크와 예측면이 서로소임을 assert.
  · 저장하는 것은 z(=성공 로짓)뿐. 셀 확률·셀 라벨을 어떤 산출물에도 안 쓴다.
🔴 S102: 사후 기록은 `np.save(z)` **뒤**에 둔다.
"""
import hashlib, json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn

SPLIT, SEED, DEV, INIT = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
DROP, OPT, TAG = float(sys.argv[5]), sys.argv[6], sys.argv[7]

H = "/home/<user>/SungjinPark"
B = f"{H}/exp104"
W89 = f"{H}/exp102/work"                      # 🔴 읽기 전용
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                       # noqa: E402
from arch093 import TokenMLP, raw_npz, foldable, HID_BASE   # noqa: E402

LR, BATCH = 1e-3, 4096
EPOCH = int(os.environ.get("EP093", "2"))
HID = tuple(int(x) for x in os.environ.get(
    "HID093", ",".join(str(h) for h in HID_BASE)).split(","))
TARGET = os.environ.get("TARGET093", "bce")
assert TARGET in ("bce", "cells5", "collapsed5")
N_OUT = 1 if TARGET == "bce" else 5
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
torch.set_num_threads(int(os.environ.get("NT093", "1")))
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
assert not (trm & vam).any(), "🔴 학습 마스크와 예측면이 겹친다"
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)

# ── 5분류 소프트 타깃 (학습 행만) ──────────────────────────────────────
TT, TGT_META = None, {"kind": "bce", "n_out": 1}
if N_OUT == 5:
    T5 = np.load(f"{H}/exp093/work/T5_oot23.npy")
    assert len(T5) == len(trm) and T5.shape[1] == 5
    assert np.isnan(T5[vam]).all(), "🔴 예측면 라벨이 NaN 이 아니다"
    t_tr = T5[trm].astype(np.float32)
    assert np.isfinite(t_tr).all()
    assert np.allclose(t_tr.sum(1), 1.0, atol=1e-5)
    assert np.allclose(t_tr[:, 4], y[trm], atol=1e-6), "🔴 주변 성공확률 타깃 != y"
    if TARGET == "collapsed5":                      # 유형 정보를 지운 귀무 대조
        fail = 1.0 - t_tr[:, 4]
        t_tr = np.concatenate([np.repeat((fail * 0.25)[:, None], 4, 1),
                               t_tr[:, 4:5]], 1).astype(np.float32)
        assert np.allclose(t_tr.sum(1), 1.0, atol=1e-5)
    TT = torch.from_numpy(t_tr).to(dev)
    TGT_META = {"kind": TARGET, "n_out": 5,
                "class_order": ["is_mid_only", "is_rev_only", "is_both",
                                "is_wayoff_true", "success"],
                "train_class_prior": [float(v) for v in t_tr.mean(0)],
                "marginal_success_target_equals_y": True}
    del t_tr, T5

model = TokenMLP(Xt.shape[1], CARDS, p_drop=DROP, init=INIT, hid=HID,
                 n_out=N_OUT).to(dev)
if OPT == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=LR)
elif OPT.startswith("adamw"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(OPT[len("adamw"):]))
else:
    raise ValueError(OPT)

bce = nn.BCEWithLogitsLoss()
lsm = nn.LogSoftmax(dim=1)


def step_loss(idx):
    o = model(Ct[idx], Xt[idx])
    if N_OUT == 1:
        return bce(o, yt[idx])
    return -(TT[idx] * lsm(o)).sum(1).mean()


g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
TOTAL = int(os.environ.get("STEPS093", "0")) or steps * EPOCH
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
        loss = step_loss(idx)
        loss.backward()
        opt.step()
        tot += loss.item(); cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_loss": tot / max(cnt, 1)})
    print(f"ep {ep} steps {cnt} loss {tot/max(cnt,1):.6f}", flush=True)

model.eval()
with torch.no_grad():
    zz, cc = [], []
    for s in range(0, len(yv), 65536):
        o = model(Cv[s:s + 65536], Xv[s:s + 65536]).double()
        if N_OUT == 1:
            zz.append(o.cpu())
        else:
            # logit(p_succ) = l4 − logsumexp(l_0..3).  나눗셈·클리핑 없음.
            zz.append((o[:, 4] - torch.logsumexp(o[:, :4], dim=1)).cpu())
            cc.append(torch.softmax(o, dim=1)[:, :4].sum(1).cpu())
    z = torch.cat(zz).numpy()
wall = time.time() - t0
p = 1.0 / (1.0 + np.exp(-z))
br = float(np.mean((p - yv) ** 2))

# 🔴 S102 — z 를 **먼저** 저장한다. 사후 기록은 그 뒤.
np.save(f"{OUT}/preds/z_{TAG}.npy", z)

# 🔴 산출물에 셀 확률을 남기지 않는다 — 항등 확인용 스칼라 한 개만 적는다.
ident = None
if N_OUT == 5:
    pf = torch.cat(cc).numpy()
    ident = float(np.abs((1.0 - pf) - p).max())        # p_succ == 1 − Σ(4셀) 인가
    del pf

raw = raw_npz(model.cpu())
FOLD = foldable(model)
fp = {}
if FOLD:
    A, c0, T = DI._fold(raw)
    fp = {"A": float(A.std()), "c0": float(c0.std())}
    fp.update({f"T{j}": float(T[j].std()) for j in range(5)})
    for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
              "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight"):
        fp[k] = float(raw[k].std())
    fp["m.10.bias_value"] = float(raw["m.10.bias"].ravel()[0])

res = {"tag": TAG, "split": SPLIT, "seed": SEED, "init": INIT, "drop": DROP,
       "opt": OPT, "lr": LR, "lr_schedule": "constant (no scheduler)",
       "hid": list(HID), "depth": len(HID), "n_out": N_OUT,
       "target_spec": TARGET, "target_meta": TGT_META,
       "foldable_by_mate_deep_infer": FOLD,
       "torch_threads": int(os.environ.get("NT093", "1")),
       "epoch_env": EPOCH, "batch": BATCH, "cards": CARDS,
       "n_train": n_tr, "n_valid": int(len(yv)), "steps_per_epoch": steps,
       "total_steps": int(TOTAL),
       "n_params": int(sum(q.numel() for q in model.parameters())),
       "z_is": "logit(p_succ). n_out=5 면 l4 − logsumexp(l_0..3)",
       "identity_psucc_eq_1_minus_cellsum_maxabs": ident,
       "valid_brier_raw": br,
       "valid_score_raw": 100000.0 * (1.0 - br / 0.2498069),
       "z_sd": float(z.std()), "z_mean": float(z.mean()),
       "fit_wall_sec": round(wall, 1), "curve": curve,
       "folded_fingerprint_sd": fp,
       "z_sha256": hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()}
json.dump(res, open(f"{OUT}/fit_{TAG}.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items()
                  if k not in ("cards", "folded_fingerprint_sd", "curve")},
                 ensure_ascii=False))

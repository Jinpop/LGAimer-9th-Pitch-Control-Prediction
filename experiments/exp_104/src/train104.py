"""exp_092 — 딥 팔 1 fit. **exp_091 train091.py 의 코드 경로를 그대로 승계한다.**

사용: train092.py <SPLIT le2024|oot23> <SEED> <DEV> <INIT> <DROP> <OPT> <TAG>
환경: STEPS092=<총 옵티마이저 스텝>   (0 이면 steps_per_epoch x EPOCH)
      EP092=<epoch>                  (기본 2 — STEPS092 가 있으면 무시된다)
      HID092=256,128,64              (은닉 폭·깊이. 기본 = 현행 구조. 이번 실험은 안 건드린다)
      NT092=1                        (torch CPU 스레드. S102 — z 에 안 닿는다)
      MASK092=<쉼표로 구분한 수치열 인덱스>  (비면 마스킹 없음)
      LOSS092=bce | focal1 | focal2 | ls0.02 | ls0.05
      ROWS092=<bool npy 경로>   학습 행을 추가로 제한한다 (축 F 의 커버리지 구멍용)
      OFF092=<float npy 경로>   **학습 시에만** 로짓 오프셋을 더한다 (축 F 잔차 타깃)

🔴 **exp_091 대비 바뀐 것 둘, 그리고 기본값에서는 둘 다 z 에 안 닿는다**:
  ① MASK092 — 수치 94열 중 지정 열을 **표준화 공간의 0(=학습 구간 평균)으로 대입**한다.
     🔴 **열을 빼는 게 아니라 0 으로 덮는다** (S101 — 팀원 `_fold` 가 `reshape(256, 99, 96)` 으로
        99토큰을 하드코딩하므로 열 수가 바뀌면 출하가 깨진다). 마스킹된 열의 토큰은
        `0*w_j + b_j = b_j` 상수라 첫 Linear 의 편향에 흡수된다 = 정보가 0 이다.
     🔴 **mu/sd 는 다시 안 잡는다.** 표준화가 열별 통계라 살아남은 열의 mu/sd 는 마스킹과
        무관하게 같다 — 즉 "다시 잡을지"가 변인이 아니다(이 선택을 report 에 명시한다).
  ② LOSS092 — 학습 손실만 바꾼다. ⚠️ **손실을 바꾸면 스텝당 기울기 크기가 달라져
     1632 스텝의 *유효* 학습량이 함께 바뀐다** — 이건 딸려 오는 변인이고 그렇게 기록한다.
  ③ ROWS092 / OFF092 — 축 F. 손실이 `BCE(model(x) + o_i, y_i)` 가 되어 모델이 **잔차**를 배운다.
     🔴 **저장하는 z 는 `model(x)` 뿐이다 — 오프셋을 안 더한다.** 잔차 자체가 블렌드에 꽂을
        팔이기 때문이다(mate 가 이미 기저를 대고 있다). 오프셋을 더한 읽기는 평가 단계에서
        같은 벡터로 따로 만든다 — 한 번의 적합에서 두 읽기가 나온다.
     🔴 오프셋은 **그 행에 대해 out-of-time** 이어야 한다 (s0_offF092.py 의 게이트 넷).
🔴 MASK092·ROWS092·OFF092 가 비고 LOSS092=bce 면 exp_090 의 z 와 **바이트 동일**해야 한다
   — 재현 게이트가 증명한다.
🔴 S102: 사후 기록(raw_npz/_fold)은 `np.save(z)` **뒤**에 둔다.

동결 레시피 (팀원 prep.json): lr 1e-3 상수(스케줄러 없음) · batch 4096 · D 96 · init ftt.
학습량 1632 스텝 · 5시드 · 폭 256/깊이 3 · dropout 0 · wd 0.01 은 이번 실험에서 고정이다.
"""
import hashlib, json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

SPLIT = sys.argv[1]
SEED = int(sys.argv[2])
DEV = sys.argv[3]
INIT = sys.argv[4]
DROP = float(sys.argv[5])
OPT = sys.argv[6]
TAG = sys.argv[7]

H = "/home/<user>/SungjinPark"
B = f"{H}/exp104"
W89 = f"{H}/exp102/work"                      # 🔴 읽기 전용
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
sys.path.insert(0, PKG)
sys.path.insert(0, f"{B}/src")
import deep_infer as DI                       # noqa: E402
from arch102b import TokenMLP, raw_npz, foldable, HID_BASE   # noqa: E402

LR, BATCH = 1e-3, 4096
EPOCH = int(os.environ.get("EP092", "2"))
HID = tuple(int(x) for x in os.environ.get(
    "HID092", ",".join(str(h) for h in HID_BASE)).split(","))
_mk = os.environ.get("MASK092", "").strip()
MASKCOLS = [int(x) for x in _mk.split(",")] if _mk else []
LOSS = os.environ.get("LOSS092", "bce")
ROWS_P = os.environ.get("ROWS092", "").strip()
OFF_P = os.environ.get("OFF092", "").strip()
OUT = f"{B}/out"
os.makedirs(f"{OUT}/preds", exist_ok=True)
os.makedirs(f"{OUT}/npz", exist_ok=True)
torch.set_num_threads(int(os.environ.get("NT092", "1")))
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
n_tr_full = int(trm.sum())
if ROWS_P:                                   # 축 F — 학습 행 제한
    rsel = np.load(ROWS_P)
    assert rsel.dtype == np.bool_ and len(rsel) == len(trm)
    trm = trm & rsel
n_tr = int(trm.sum())
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
yv = y[vam].astype(np.float64)

# ── 축 F — 학습 시에만 더하는 out-of-time 로짓 오프셋 ────────────────────────
ot = None
OFF_STATS = None
if OFF_P:
    offa = np.load(OFF_P)
    assert len(offa) == len(vam)
    o_tr = offa[trm]
    assert np.isfinite(o_tr).all(), "🔴 학습 행 오프셋에 NaN — ROWS092 로 커버 행만 남겨라"
    ot = torch.from_numpy(o_tr.astype(np.float32)).to(dev)
    OFF_STATS = {"path": OFF_P, "n_train_rows": int(len(o_tr)),
                 "train_mean": float(o_tr.mean()), "train_sd": float(o_tr.std()),
                 "valid_finite_share": float(np.isfinite(offa[vam]).mean())}

# ── 피처 마스킹 (표준화 공간의 0 = 학습 구간 평균). 형상 99토큰 유지 ──────────
if MASKCOLS:
    assert max(MASKCOLS) < Xt.shape[1] and min(MASKCOLS) >= 0
    assert len(set(MASKCOLS)) == len(MASKCOLS)
    mi = torch.as_tensor(sorted(MASKCOLS), dtype=torch.long, device=dev)
    Xt[:, mi] = 0.0
    Xv[:, mi] = 0.0
    assert float(Xt[:, mi].abs().max()) == 0.0
    assert float(Xv[:, mi].abs().max()) == 0.0

model = TokenMLP(Xt.shape[1], CARDS, p_drop=DROP, init=INIT, hid=HID).to(dev)
if OPT == "adam":
    opt = torch.optim.Adam(model.parameters(), lr=LR)
elif OPT.startswith("adamw"):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=float(OPT[len("adamw"):]))
else:
    raise ValueError(OPT)


def make_loss(spec):
    """🔴 spec='bce' 는 exp_090/091 과 **글자 그대로 같은 객체**여야 한다."""
    if spec == "bce":
        f = nn.BCEWithLogitsLoss()
        return (lambda lo, tg: f(lo, tg)), {"kind": "bce"}
    if spec.startswith("focal"):
        gm = float(spec[len("focal"):])

        def fl(lo, tg):                       # FL = (1-p_t)^gamma * BCE, alpha 없음
            logp, log1mp = Fn.logsigmoid(lo), Fn.logsigmoid(-lo)
            p = torch.sigmoid(lo)
            pt = tg * p + (1.0 - tg) * (1.0 - p)
            logpt = tg * logp + (1.0 - tg) * log1mp
            return (-((1.0 - pt) ** gm) * logpt).mean()
        return fl, {"kind": "focal", "gamma": gm, "alpha": None}
    if spec.startswith("ls"):
        eps = float(spec[2:])
        f = nn.BCEWithLogitsLoss()
        return (lambda lo, tg: f(lo, tg * (1.0 - eps) + 0.5 * eps)),\
               {"kind": "label_smoothing", "eps": eps}
    raise ValueError(spec)


lossf, LOSS_META = make_loss(LOSS)
g = torch.Generator().manual_seed(SEED)
steps = n_tr // BATCH
TOTAL = int(os.environ.get("STEPS092", "0")) or steps * EPOCH
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
        out_l = model(Ct[idx], Xt[idx])
        if ot is not None:
            out_l = out_l + ot[idx]
        loss = lossf(out_l, yt[idx])
        loss.backward()
        opt.step()
        tot += loss.item(); cnt += 1; done += 1
    curve.append({"epoch": ep, "steps": cnt, "train_loss": tot / max(cnt, 1)})
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

# 🔴 S102 — z 를 **먼저** 저장한다. 사후 기록은 그 뒤.
np.save(f"{OUT}/preds/z_{TAG}.npy", z)

raw = raw_npz(model.cpu())
fp = {}
FOLD = foldable(model)
if FOLD:
    A, c0, T = DI._fold(raw)
    fp = {"A": float(A.std()), "c0": float(c0.std())}
    fp.update({f"T{j}": float(T[j].std()) for j in range(5)})
    for k in ("m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias", "m.5.weight",
              "m.5.bias", "m.8.weight", "m.8.bias", "m.10.weight"):
        fp[k] = float(raw[k].std())
    fp["m.10.bias_value"] = float(raw["m.10.bias"].ravel()[0])

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
       "torch_threads": int(os.environ.get("NT092", "1")),
       "loss_spec": LOSS, "loss_meta": LOSS_META,
       "n_masked_num_cols": len(MASKCOLS), "masked_num_cols": sorted(MASKCOLS),
       "rows_restrict_path": ROWS_P or None, "n_train_full_mask": n_tr_full,
       "offset_path": OFF_P or None, "offset_stats": OFF_STATS,
       "z_saved_is": "model(x) 만 — 오프셋 미포함 (잔차 팔)",
       "mask_convention": "표준화 공간 0 대입 (= 학습 구간 평균). 열 수 94 유지, mu/sd 재계산 없음",
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
                  if k not in ("cards", "folded_fingerprint_sd", "curve",
                               "masked_num_cols")}, ensure_ascii=False))

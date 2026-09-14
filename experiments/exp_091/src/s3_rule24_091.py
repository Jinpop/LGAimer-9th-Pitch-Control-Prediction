"""exp_091 stage 3 — 규칙 2-4: 이 실험이 만든 z 가 **다른 행의 내용**에 닿는가.

이 실험은 제출물을 내지 않는다. 측정 대상은 `train091.py` 가 저장한 valid 로짓 `z` 다.
그러니 검사할 것은 딱 하나 — **z 의 각 행이 그 행만의 함수인가.**

CLAUDE.md §규칙 2-4 판정대로 **위치와 내용을 가른다**:
  ① 위치 — valid 행을 무작위로 섞어 순전파하고 순열을 되돌린다. delta 가 0.0 이면 위치 무관.
  ② 내용 — KEEP 110행은 값·인덱스 그대로 두고 **나머지 전 행을 두 극단 행 반복으로 교체**(S28).
           KEEP 행의 z 가 정확히 0.0 만큼 움직이면 다른 행의 *내용*은 통로가 아니다.
검사는 학습된 가중치 위에서 한다(50스텝 — 행 독립성은 가중치의 함수가 아니지만,
초기화 상태의 항등은 약한 검사라 실제 학습 경로를 한 번 태운다).
GPU 순전파라 CPU 를 거의 안 쓴다.
"""
import json, os, random, sys, time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import torch
import torch.nn as nn

H = "/home/<user>/SungjinPark"
B = f"{H}/exp091"
W89 = f"{H}/exp089/work"
sys.path.insert(0, f"{B}/src")
from arch091 import TokenMLP                       # noqa: E402

DEV = sys.argv[1] if len(sys.argv) > 1 else "cuda:0"
SEED, SPLIT, BATCH, LR = 42, "oot23", 4096, 1e-3
torch.set_num_threads(2)
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

sch = json.load(open(f"{W89}/schema_{SPLIT}.json"))
X = np.load(f"{W89}/X_{SPLIT}.npy", mmap_mode="r")
C = np.load(f"{W89}/codes_{SPLIT}.npy", mmap_mode="r")
y = np.load(f"{W89}/y.npy")
trm = np.load(f"{W89}/trm_{SPLIT}.npy")
vam = np.load(f"{W89}/valid_mask.npy")
dev = torch.device(DEV)
Xt = torch.from_numpy(np.ascontiguousarray(X[trm])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(C[trm]).astype(np.int64)).to(dev)
yt = torch.from_numpy(y[trm]).to(dev)
Xv = torch.from_numpy(np.ascontiguousarray(X[vam])).to(dev)
Cv = torch.from_numpy(np.ascontiguousarray(C[vam]).astype(np.int64)).to(dev)
n = Xv.shape[0]
assert n == 253507

model = TokenMLP(Xt.shape[1], sch["cards"], p_drop=0.0, init="ftt").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
lossf = nn.BCEWithLogitsLoss()
g = torch.Generator().manual_seed(SEED)
perm_tr = torch.randperm(int(trm.sum()), generator=g)
model.train()
for s in range(0, 50 * BATCH, BATCH):                 # 50 스텝
    idx = perm_tr[s:s + BATCH].to(dev)
    opt.zero_grad(set_to_none=True)
    lossf(model(Ct[idx], Xt[idx]), yt[idx]).backward()
    opt.step()
model.eval()


def fwd(Xa, Ca):
    with torch.no_grad():
        out = []
        for s in range(0, Xa.shape[0], 65536):        # train091.py 와 같은 청크
            out.append(model(Ca[s:s + 65536], Xa[s:s + 65536]).double().cpu())
    return torch.cat(out).numpy()


z0 = fwd(Xv, Cv)

# ① 위치 — 전 행 순열 왕복
rp = np.random.default_rng(91091).permutation(n)
inv = np.argsort(rp)
zp = fwd(Xv[torch.from_numpy(rp).to(dev)], Cv[torch.from_numpy(rp).to(dev)])[inv]
d_pos = np.abs(zp - z0)

# ② 내용 — KEEP 110행 원위치·원문, 나머지 전량 극단 두 행 반복 (S28)
rng = np.random.default_rng(90090)
KEEP = np.sort(rng.choice(n, 110, replace=False))
pool = np.setdiff1d(np.arange(n), KEEP)
lo, hi = int(pool[0]), int(pool[-1])
src = np.where(np.arange(n) % 2 == 0, lo, hi)
src[KEEP] = KEEP
Xd = Xv[torch.from_numpy(src).to(dev)].clone()
Cd = Cv[torch.from_numpy(src).to(dev)].clone()
assert torch.equal(Xd[torch.from_numpy(KEEP).to(dev)], Xv[torch.from_numpy(KEEP).to(dev)])
assert torch.equal(Cd[torch.from_numpy(KEEP).to(dev)], Cv[torch.from_numpy(KEEP).to(dev)])
zd = fwd(Xd, Cd)
d_con = np.abs(zd[KEEP] - z0[KEEP])

R = {"exp_id": "exp_091", "stage": "s3_rule24", "n_rows": int(n),
     "target": "train091.py 가 저장하는 valid 로짓 z (이 실험의 유일한 측정 산출물)",
     "method": "CLAUDE.md §규칙 2-4 — 위치(전 행 순열 왕복)와 내용(S28 축퇴 프레임)을 가른다",
     "position_permutation": {
         "max_abs_delta": float(d_pos.max()),
         "n_moving": int((d_pos > 0).sum()),
         "PASS_position_independent": bool(d_pos.max() == 0.0)},
     "content_separation": {
         "n_keep": int(len(KEEP)),
         "max_abs_delta_on_KEEP": float(d_con.max()),
         "n_moving": int((d_con > 0).sum()),
         "PASS_content_independent": bool(d_con.max() == 0.0)},
     "no_test_frame_statistics": ("이 실험은 test(2025) 프레임을 한 번도 읽지 않는다. "
                                 "mu/sd·어휘는 exp_089 work/ 의 학습 구간 산출물이고, "
                                 "면의 모든 팔은 out-of-time 납품 벡터다."),
     }
R["PASS"] = bool(R["position_permutation"]["PASS_position_independent"]
                 and R["content_separation"]["PASS_content_independent"])
json.dump(R, open(f"{B}/out/s3_rule24.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))

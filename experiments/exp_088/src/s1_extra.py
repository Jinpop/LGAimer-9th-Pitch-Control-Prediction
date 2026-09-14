"""exp_088 stage 1c — 추론 시간 실측 + 딥 팔 자신의 중복도(참조점).

🔑 왜 딥 팔의 중복도를 재나: 관문 A 의 문턱(0.90 / 0.95)은 사전 등록값이지 실적으로
   눈금이 잡힌 값이 아니다. **LB +5.0813180433 을 실제로 산 팔**이 기존 팔과 얼마나
   겹치는지를 같은 자로 재면 그 문턱이 어디쯤인지 알 수 있다.
"""
import json, time
import numpy as np
import torch
from scipy.special import logit

H = "/home/<user>/SungjinPark"
B = f"{H}/exp088"
EPS = 1e-6
R = {"exp_id": "exp_088", "stage": "s1c_extra"}

# ── 1. 딥 팔 vs 기존 팔 (같은 면·같은 자) ────────────────────────────────
X = np.load(f"{H}/arms/X_arms.npy")
bj = json.load(open(f"{H}/audit_deep/pkg/DEEP/mate/model/blend.json", encoding="utf-8"))
w_v6, w_ebm, beta = float(bj["w_v6"]), float(bj["w_ebm"]), float(bj["beta_ins"])
off = np.load(f"{H}/arms/off.npy")
z_mate = (1 - w_ebm) * (w_v6 * X[0] + (1 - w_v6) * X[1]) + w_ebm * X[2] + beta * off
zd = np.load(f"{B}/out/preds/deep_route1_logit_seed5.npy")
rr = lambda a, b: float(np.corrcoef(a, b)[0, 1])
R["deep_arm_own_overlap"] = {
    "vs_mate_v6": rr(zd, X[0]), "vs_mate_Q": rr(zd, X[1]),
    "vs_p_EBM": rr(zd, X[2]), "vs_AA_deployed": rr(zd, X[3]),
    "vs_partner_internal_mix_z_mate": rr(zd, z_mate),
    "w_deep_deployed": float(bj["w_deep"]),
    "LB_delta_bought": 5.0813180433,
    "🔑": ("딥 팔은 파트너 내부 혼합 로짓의 20% 를 갈아끼우며 LB +5.0813180433 을 샀다. "
           "그 팔이 기존 팔들과 갖는 상관이 우리 관문 A 문턱 근처/위라면, "
           "문턱 자체가 LB 실적으로 눈금이 안 잡혀 있다는 뜻이다"),
}

# ── 2. 추론 시간 (CPU, 253,507행, threads 6) ─────────────────────────────
torch.set_num_threads(6)
import sys
sys.argv = ["x", "F1_FM"]
W = f"{B}/work"
schema = json.load(open(f"{W}/schema.json"))
FIELDS = schema["fields"]
IX = {n: i for i, n in enumerate(FIELDS)}
NLEV = [schema["levels_n"][n] for n in FIELDS]
Xd = np.load(f"{W}/Xd.npy", mmap_mode="r")
codes = np.load(f"{W}/codes.npy")
vam = np.load(f"{W}/valid_mask.npy")
D = Xd.shape[1]
Xv = torch.from_numpy(np.ascontiguousarray(Xd[vam]))
Cv = torch.from_numpy(codes[vam].astype(np.int64))

import torch.nn as nn


class FM(nn.Module):
    K = 16
    FF = ["pitcher_id", "batter_id", "batter_hand", "dom6"]
    PAIRS = [(0, 1), (0, 2), (0, 3), (1, 3)]

    def __init__(self):
        super().__init__()
        self.fidx = [IX[n] for n in self.FF]
        self.emb = nn.ModuleList([nn.Embedding(NLEV[i] + 1, self.K) for i in self.fidx])
        self.lin_c = nn.ModuleList([nn.Embedding(k + 1, 1) for k in NLEV])
        self.lin_d = nn.Linear(D, 1)

    def forward(self, xc, xd):
        z = self.lin_d(xd).squeeze(1)
        for i, e in enumerate(self.lin_c):
            z = z + e(xc[:, i]).squeeze(1)
        v = [self.emb[j](xc[:, self.fidx[j]]) for j in range(len(self.FF))]
        for a, b in self.PAIRS:
            z = z + (v[a] * v[b]).sum(dim=1)
        return z


times = []
for s in (42, 7, 2024, 101, 777):
    m = FM()
    m.load_state_dict(torch.load(f"{W}/F1_FM_s{s}_le2024_best.pt", map_location="cpu"))
    m.eval()
    t0 = time.time()
    with torch.no_grad():
        o = []
        for a in range(0, len(Xv), 65536):
            o.append(torch.sigmoid(m(Cv[a:a + 65536], Xv[a:a + 65536])).double())
        p = torch.cat(o).numpy()
    times.append(time.time() - t0)
R["inference_time_sec"] = {
    "rows": int(vam.sum()), "threads": 6, "device": "cpu",
    "per_seed_sec": [round(t, 4) for t in times],
    "sum_5seed_sec": round(sum(times), 4),
    "note": "가중치 적재는 제외한 순전파만. S24 — 추론 시간은 현재 제약이 아니다",
}
json.dump(R, open(f"{B}/out/s1c_extra.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))

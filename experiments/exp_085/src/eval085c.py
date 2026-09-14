"""exp_085 stage 3c — CPU 추론 시간 실측 (채점 서버는 GPU 를 전제하지 않는다).

253,507행(=fold_2024 valid) 을 CPU·threads 6 으로 한 번 통과시키는 시간.
test 는 245,789행이므로 거의 같은 규모다. 학습 0건 — 저장된 state 를 읽어 쓴다.
GPU 예측과의 최대절대차도 함께 적는다(같은 모델인지 확인).
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn

H = "/home/<user>/SungjinPark"
B = f"{H}/exp085"
W, OUT = f"{B}/work", f"{B}/out"
torch.set_num_threads(6)

schema = json.load(open(f"{W}/schema.json"))
FIELDS = schema["fields"]
IX = {n: i for i, n in enumerate(FIELDS)}
NLEV = [schema["levels_n"][n] for n in FIELDS]
Xd = np.load(f"{W}/Xd.npy")
codes = np.load(f"{W}/codes.npy").astype(np.int64)
vam = np.load(f"{W}/valid_mask.npy")
D = Xd.shape[1]
Xv = torch.from_numpy(Xd[vam])
Cv = torch.from_numpy(codes[vam])
del Xd, codes


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


class EMBMLP(nn.Module):
    NEW = {"pitcher_id": 32, "batter_id": 32, "dom6": 16}
    OLD = ["base_state", "game_type", "pitcher_team_id", "batter_team_id", "cnt_state"]
    OLD_DIM = 6

    def __init__(self):
        super().__init__()
        self.names = list(self.NEW) + self.OLD
        self.dims = [self.NEW[n] for n in self.NEW] + [self.OLD_DIM] * len(self.OLD)
        self.fidx = [IX[n] for n in self.names]
        self.emb = nn.ModuleList(
            [nn.Embedding(NLEV[i] + 1, d) for i, d in zip(self.fidx, self.dims)])
        din = D + sum(self.dims)
        self.mlp = nn.Sequential(
            nn.Linear(din, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xc, xd):
        e = [emb(xc[:, self.fidx[i]]) for i, emb in enumerate(self.emb)]
        return self.mlp(torch.cat(e + [xd], dim=1)).squeeze(1)


R = {"n_rows": int(vam.sum()), "threads": 6, "device": "cpu",
     "note": "test 는 245,789행 — 여기 잰 253,507행과 3% 차"}
for arm, cls in (("F1_FM", FM), ("F2_EMB", EMBMLP)):
    m = cls()
    m.load_state_dict(torch.load(f"{W}/{arm}_s42_r2_best.pt", map_location="cpu"))
    m.eval()
    with torch.no_grad():
        for _ in range(2):                       # warm-up
            m(Cv[:8192], Xv[:8192])
        t0 = time.time()
        o = []
        for s in range(0, len(Xv), 65536):
            o.append(torch.sigmoid(m(Cv[s:s + 65536], Xv[s:s + 65536])).double().numpy())
        dt = time.time() - t0
    p = np.concatenate(o)
    gpu = np.load(f"{OUT}/preds/{arm}_s42_r2_best.npy")
    R[arm] = {"cpu_infer_sec": round(dt, 3),
              "max_abs_diff_vs_gpu_pred": float(np.max(np.abs(p - gpu)))}
    print(arm, R[arm], flush=True)

json.dump(R, open(f"{OUT}/infer_time.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))

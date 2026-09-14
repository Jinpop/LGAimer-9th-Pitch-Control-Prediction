"""exp_067 공통 — exp_064 src/nnlib.py 승계. 편집 3곳(작업 dir 분리, 프레임 경로, fit_both 추가).

승계(비트 동일 의도): 피처 99열·범주 5개(edim 6)·전처리 산술(float32 표준화 후
nan_to_num 0)·AdamW+OneCycle·BCEWithLogits·batch 4096 drop_last·결정론 설정.
개조: ① 폭/dropout/max_lr/seed/epochs 를 인자로 ② 분할(train/valid 마스크)을 인자로
③ 한 프로세스 안에서 split 텐서를 재사용해 여러 fit 을 돌린다.

층 수는 4층 고정 — 출하 mate/nn_infer.py predict_nn 이 인덱스를 하드코딩한다
(lin0 -> bn1 -> relu -> drop3 -> lin4 -> bn5 -> relu -> drop7 -> lin8 -> relu -> lin10).
"""
import json
import os
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

EXP60 = os.path.expanduser("~/SungjinPark/exp060_nn")
SPD = os.path.expanduser("~/SungjinPark/sp_def_recv")
B064 = os.path.expanduser("~/SungjinPark/exp064")   # 읽기 전용 (프레임)
B = os.path.expanduser("~/SungjinPark/exp067")
OUT = f"{B}/out"
BATCH = 4096
WD = 1e-4
PCT_START = 0.3
EDIM = 6
REF = 0.2498069
CATS = ["base_state", "game_type", "pitcher_team_id", "batter_team_id",
        "cnt_state"]


def conv(brier):
    """LB 환산 점수 (CLAUDE.md 규약). 6자리 Brier 환산은 +-0.20 불확실."""
    return 100000.0 * (1.0 - brier / REF)


def lg(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def load_frame(which="orig"):
    """which='orig' -> exp_060 캐시(우리 재구축 p6, exp_061·exp_063 이 쓴 것).
    which='ship' -> exp_064 가 만든 배포 p6 14열 교체판.
    둘 다 읽기 전용 — exp060_nn 은 절대 수정하지 않는다."""
    assert which in ("orig", "ship")
    prov = json.load(open(f"{EXP60}/cache/provenance.json"))
    path = (f"{EXP60}/cache/frame99.parquet" if which == "orig"
            else f"{B064}/cache/frame99_ship.parquet")
    df = pd.read_parquet(path)
    assert len(df) == 1475092 == prov["n_rows"], (len(df), prov["n_rows"])
    FSP = [l.strip() for l in open(f"{SPD}/features_sp.txt", encoding="utf-8")
           if l.strip()]
    assert len(FSP) == 99
    DENSE = [c for c in FSP if c not in CATS]
    assert len(DENSE) == 95 and all(c in df.columns for c in CATS)
    SP6 = [c for c in DENSE if c.startswith("sp_")]
    df[SP6] = df[SP6].fillna(0.0)      # exp_060 과 동일: 2019 sp_* NaN -> 0.0
    return df, DENSE, prov


class Split:
    """한 분할의 GPU 텐서·표준화 상수·범주 LUT. 여러 fit 이 공유한다."""

    def __init__(self, df, DENSE, train_mask, valid_mask, name, dev="cuda"):
        self.name = name
        self.DENSE = DENSE
        self.dev = dev
        n_tr = int(train_mask.sum())
        n_va = int(valid_mask.sum())
        self.n_train, self.n_valid = n_tr, n_va

        levels = {}
        for c in CATS:
            vals = df.loc[train_mask, c].unique()
            try:
                lv = sorted(vals, key=lambda x: int(x))
            except (ValueError, TypeError):
                lv = sorted(map(str, vals))
            levels[c] = [str(v) for v in lv]
        self.levels = levels
        codes = np.zeros((len(df), len(CATS)), dtype=np.int64)
        for i, c in enumerate(CATS):
            lut = {v: j + 1 for j, v in enumerate(levels[c])}
            codes[:, i] = (df[c].astype(str).map(lut).fillna(0)
                           .astype(np.int64).values)
        self.unknown_valid = int((codes[valid_mask] == 0).sum())

        X = df[DENSE].astype("float32").values
        mu = np.nanmean(X[train_mask], axis=0).astype(np.float32)
        sd = np.nanstd(X[train_mask], axis=0).astype(np.float32)
        sd[sd < 1e-6] = 1.0
        X = (X - mu) / sd
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        self.mu, self.sd = mu, sd
        y = df["control_success"].to_numpy(np.float32)

        self.Xt = torch.from_numpy(X[train_mask]).to(dev)
        self.Ct = torch.from_numpy(codes[train_mask]).to(dev)
        self.yt = torch.from_numpy(y[train_mask]).to(dev)
        self.Xv = torch.from_numpy(X[valid_mask]).to(dev)
        self.Cv = torch.from_numpy(codes[valid_mask]).to(dev)
        self.yv64 = y[valid_mask].astype(np.float64)
        self.nlev = [len(levels[c]) for c in CATS]
        self.steps_per_epoch = n_tr // BATCH
        del X, codes
        print(f"[split {name}] train {n_tr} valid {n_va} "
              f"unknown_valid_cells {self.unknown_valid} "
              f"steps/epoch {self.steps_per_epoch}", flush=True)

    def free(self):
        for a in ("Xt", "Ct", "yt", "Xv", "Cv"):
            setattr(self, a, None)
        torch.cuda.empty_cache()


class Net(nn.Module):
    def __init__(self, nlev, din, widths, dropout, ed=EDIM):
        super().__init__()
        h1, h2, h3 = widths
        self.embs = nn.ModuleList([nn.Embedding(k + 1, ed) for k in nlev])
        self.mlp = nn.Sequential(
            nn.Linear(din + len(nlev) * ed, h1), nn.BatchNorm1d(h1),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, h3), nn.ReLU(), nn.Linear(h3, 1))

    def forward(self, xc, xd):
        e = [emb(xc[:, i]) for i, emb in enumerate(self.embs)]
        return self.mlp(torch.cat(e + [xd], dim=1)).squeeze(1)


def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(sp, widths, dropout, max_lr, seed, epochs, sched_epochs=None,
        take="best", tag=""):
    """한 번의 학습. sched_epochs 는 OneCycle total_steps 산정용 에폭 수
    (기본 = epochs). take='best' 면 valid Brier 최소 상태, 'final' 이면 마지막 상태.
    반환: dict(요약) + p_valid(np.float64) + state(cpu)."""
    _seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    sched_epochs = sched_epochs or epochs
    model = Net(sp.nlev, len(sp.DENSE), widths, dropout).to(sp.dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=sched_epochs * sp.steps_per_epoch,
        pct_start=PCT_START, div_factor=25.0, final_div_factor=1e4)
    lossf = nn.BCEWithLogitsLoss()
    gcpu = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def eval_valid():
        model.eval()
        ps = []
        for s in range(0, len(sp.Xv), 65536):
            z = model(sp.Cv[s:s + 65536], sp.Xv[s:s + 65536])
            ps.append(torch.sigmoid(z).double().cpu().numpy())
        p = np.concatenate(ps)
        return p, float(np.mean((p - sp.yv64) ** 2))

    best = {"brier": np.inf, "epoch": -1, "state": None}
    curve = []
    for ep in range(1, epochs + 1):
        te0 = time.time()
        model.train()
        perm = torch.randperm(sp.n_train, generator=gcpu)
        ep_loss, nb = 0.0, 0
        for s in range(0, sp.steps_per_epoch * BATCH, BATCH):
            idx = perm[s:s + BATCH].to(sp.dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(sp.Ct[idx], sp.Xt[idx]), sp.yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1
        t_train = time.time() - te0
        _, brier = eval_valid()
        curve.append({"epoch": ep, "train_logloss": ep_loss / nb,
                      "valid_brier": brier, "t_train_s": round(t_train, 2),
                      "lr": sched.get_last_lr()[0]})
        if brier < best["brier"]:
            best = {"brier": brier, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    if take == "final":
        state = {k: v.detach().cpu().clone()
                 for k, v in model.state_dict().items()}
        sel_epoch = epochs
    else:
        state = best["state"]
        model.load_state_dict(state)
        sel_epoch = best["epoch"]
    p_valid, brier = eval_valid()
    res = {"tag": tag, "split": sp.name, "widths": list(widths),
           "dropout": dropout, "max_lr": max_lr, "seed": seed,
           "epochs_run": epochs, "sched_epochs": sched_epochs, "take": take,
           "n_params": n_params, "sel_epoch": sel_epoch,
           "best_epoch": best["epoch"], "best_brier": best["brier"],
           "valid_brier": brier, "valid_score": conv(brier),
           "n_train": sp.n_train, "n_valid": sp.n_valid,
           "fit_wall_min": (time.time() - t0) / 60.0}
    print(f"[fit {tag}] w{widths} do{dropout} lr{max_lr} s{seed} "
          f"ep{epochs}/{sched_epochs} take={take} params {n_params} "
          f"sel_ep {sel_epoch} brier {brier:.10f} score {conv(brier):.5f} "
          f"({res['fit_wall_min']:.2f}min)", flush=True)
    os.makedirs(f"{OUT}/curves", exist_ok=True)
    json.dump(curve, open(f"{OUT}/curves/curve_{tag}.json", "w"), indent=1)
    return res, p_valid, state


def to_npz(state, nlev_len, path):
    """torch state -> 출하 npz 포맷 (mate/nn_infer.py 키 규약)."""
    npz = {}
    for i in range(nlev_len):
        npz[f"emb{i}"] = state[f"embs.{i}.weight"].numpy()
    for i in (0, 4, 8, 10):
        npz[f"mlp_{i}_weight"] = state[f"mlp.{i}.weight"].numpy()
        npz[f"mlp_{i}_bias"] = state[f"mlp.{i}.bias"].numpy()
    for i in (1, 5):
        npz[f"mlp_{i}_weight"] = state[f"mlp.{i}.weight"].numpy()
        npz[f"mlp_{i}_bias"] = state[f"mlp.{i}.bias"].numpy()
        npz[f"mlp_{i}_running_mean"] = state[f"mlp.{i}.running_mean"].numpy()
        npz[f"mlp_{i}_running_var"] = state[f"mlp.{i}.running_var"].numpy()
        npz[f"mlp_{i}_num_batches_tracked"] = \
            state[f"mlp.{i}.num_batches_tracked"].numpy()
    np.savez(path, **npz)
    return sorted(npz.keys())


def predict_nn(df, npz_path, meta_path, chunk=100000):
    """출하 mate/nn_infer.py predict_nn 사본 (numpy·CPU). 산술 동일."""
    W = np.load(npz_path)
    Mm = json.load(open(meta_path))
    CATSl, DENSEl = Mm["CATS"], Mm["DENSE"]
    mu = np.array(Mm["mu"], dtype=np.float32)
    sd = np.array(Mm["sd"], dtype=np.float32)
    codes = np.zeros((len(df), len(CATSl)), dtype=np.int64)
    for i, c in enumerate(CATSl):
        lut = {v: j + 1 for j, v in enumerate(Mm["levels"][c])}
        codes[:, i] = df[c].astype(str).map(lut).fillna(0).astype(np.int64).values
    X = df[DENSEl].astype("float32").values
    X = (X - mu) / sd
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    embs = [W[f"emb{i}"] for i in range(len(CATSl))]

    def lin(z, i):
        return z @ W[f"mlp_{i}_weight"].T + W[f"mlp_{i}_bias"]

    def bn(z, i):
        return ((z - W[f"mlp_{i}_running_mean"])
                / np.sqrt(W[f"mlp_{i}_running_var"] + 1e-5)
                * W[f"mlp_{i}_weight"] + W[f"mlp_{i}_bias"])

    out = np.empty(len(df), dtype=np.float64)
    for s in range(0, len(df), chunk):
        e = min(s + chunk, len(df))
        z = np.concatenate([embs[i][codes[s:e, i]] for i in range(len(CATSl))]
                           + [X[s:e]], axis=1)
        z = lin(z, 0)
        z = np.maximum(bn(z, 1), 0.0)
        z = lin(z, 4)
        z = np.maximum(bn(z, 5), 0.0)
        z = np.maximum(lin(z, 8), 0.0)
        z = lin(z, 10)
        out[s:e] = 1.0 / (1.0 + np.exp(-z.ravel()))
    return out


# ── 구종모델 파생 14열 (exp_062 실측 max|diff|>0 전량). exp_064 는 제거가 아니라 교체 ──
# 🔴 이 목록이 exp_063 의 유일한 변인이다. 출하 mate/model/tables/pitchmodel6.txt
#    (+pmcat6.json·anom_cnt.csv)가 우리 학습용 표와 달라 exp_062 게이트가 실패했다.
#    idea_cnt_te·i3_te48 은 exact_frac 1.0 이므로 남긴다.
DROP14 = [
    "p6_curve", "p6_cutter", "p6_fastball", "p6_offspeed", "p6_sinker",
    "p6_slider",
    "i3_anom_curve", "i3_anom_cutter", "i3_anom_fastball", "i3_anom_offspeed",
    "i3_anom_sinker", "i3_anom_slider", "i3_anom_l1",
    "idea_matchup_pitch",
]


def drop14(DENSE):
    """DENSE95 -> DENSE81. 순서는 원본 순서를 그대로 보존한다."""
    assert len(DROP14) == 14 == len(set(DROP14))
    missing = [c for c in DROP14 if c not in DENSE]
    assert not missing, f"DROP14 중 DENSE 에 없는 열: {missing}"
    out = [c for c in DENSE if c not in set(DROP14)]
    assert len(out) == len(DENSE) - 14
    return out


# ══ exp_067 추가 — 한 번의 학습에서 best·last 두 상태를 모두 꺼낸다 ══════════
# 🔴 학습 루프는 fit() 과 **한 글자도 다르지 않다**. 달라지는 것은 루프가 끝난
#    뒤 어느 상태를 꺼내는가뿐이다. eval_valid 는 @no_grad + model.eval() 이라
#    난수를 소비하지 않으므로 두 번 불러도 학습 결과가 바뀌지 않는다.
#    → seed 42 best 예측이 exp_064 A99ship_s42 와 일치하면 이 사실이 기계 증명된다.
def fit_both(sp, widths, dropout, max_lr, seed, epochs, sched_epochs=None,
             tag=""):
    """반환: dict(best=..., last=...) 각각 (res, p_valid, state)."""
    _seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    t0 = time.time()
    sched_epochs = sched_epochs or epochs
    model = Net(sp.nlev, len(sp.DENSE), widths, dropout).to(sp.dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=max_lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=max_lr, total_steps=sched_epochs * sp.steps_per_epoch,
        pct_start=PCT_START, div_factor=25.0, final_div_factor=1e4)
    lossf = nn.BCEWithLogitsLoss()
    gcpu = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def eval_valid():
        model.eval()
        ps = []
        for s in range(0, len(sp.Xv), 65536):
            z = model(sp.Cv[s:s + 65536], sp.Xv[s:s + 65536])
            ps.append(torch.sigmoid(z).double().cpu().numpy())
        p = np.concatenate(ps)
        return p, float(np.mean((p - sp.yv64) ** 2))

    best = {"brier": np.inf, "epoch": -1, "state": None}
    curve = []
    for ep in range(1, epochs + 1):
        te0 = time.time()
        model.train()
        perm = torch.randperm(sp.n_train, generator=gcpu)
        ep_loss, nb = 0.0, 0
        for s in range(0, sp.steps_per_epoch * BATCH, BATCH):
            idx = perm[s:s + BATCH].to(sp.dev)
            opt.zero_grad(set_to_none=True)
            loss = lossf(model(sp.Ct[idx], sp.Xt[idx]), sp.yt[idx])
            loss.backward()
            opt.step()
            sched.step()
            ep_loss += loss.item()
            nb += 1
        t_train = time.time() - te0
        _, brier = eval_valid()
        curve.append({"epoch": ep, "train_logloss": ep_loss / nb,
                      "valid_brier": brier, "t_train_s": round(t_train, 2),
                      "lr": sched.get_last_lr()[0]})
        if brier < best["brier"]:
            best = {"brier": brier, "epoch": ep,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
    wall = (time.time() - t0) / 60.0
    st_last = {k: v.detach().cpu().clone()
               for k, v in model.state_dict().items()}
    p_last, b_last = eval_valid()
    model.load_state_dict(best["state"])
    p_best, b_best = eval_valid()

    def _res(take, sel_ep, brier):
        return {"tag": f"{tag}_{take}", "split": sp.name, "widths": list(widths),
                "dropout": dropout, "max_lr": max_lr, "seed": seed,
                "epochs_run": epochs, "sched_epochs": sched_epochs,
                "take": take, "n_params": n_params, "sel_epoch": sel_ep,
                "best_epoch": best["epoch"], "best_brier": best["brier"],
                "valid_brier": brier, "valid_score": conv(brier),
                "n_train": sp.n_train, "n_valid": sp.n_valid,
                "fit_wall_min": wall}
    os.makedirs(f"{OUT}/curves", exist_ok=True)
    json.dump(curve, open(f"{OUT}/curves/curve_{tag}.json", "w"), indent=1)
    print(f"[fit2 {tag}] s{seed} best_ep {best['epoch']} "
          f"brier {b_best:.10f} score {conv(b_best):.5f} | "
          f"last_ep {epochs} brier {b_last:.10f} score {conv(b_last):.5f} "
          f"({wall:.2f}min)", flush=True)
    return {"best": (_res("best", best["epoch"], b_best), p_best, best["state"]),
            "last": (_res("last", epochs, b_last), p_last, st_last)}


def write_meta(sp, path, widths, ed=EDIM):
    """출하 mate/nn_infer.py 가 읽는 meta.json (키 규약: CATS/levels/DENSE/mu/sd)."""
    M = {"CATS": list(CATS), "levels": sp.levels, "DENSE": list(sp.DENSE),
         "mu": [float(x) for x in sp.mu], "sd": [float(x) for x in sp.sd],
         "edims": [ed] * len(CATS), "H1": int(widths[0])}
    json.dump(M, open(path, "w"))
    return M

"""exp_080 stage2c — q_beta·q_k 정직면 주사 범위 확장 (경계에 닿아 측정이 미완이었다).

🔴 이것은 판정 문턱 변경이 아니라 **측정 범위 오류의 수정**이다 (S27 구분).
   문턱 +2.0 은 그대로다.  바꾸는 것은 내가 임의로 잡았던 주사 구간뿐이다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, time
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp080/out"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
AA = (42, 7, 2024)
BO = dict(c_aa=0.25244, c_T70=0.04756, w_out=0.7, w_v6_inner=0.6668397489,
          w_ebm=-0.1, beta_ins=0.03)
MQ = json.load(open(f"{H}/build_BO/pkg_BO/mate/model_q/meta.json"))
BETA_Q, EBK = float(MQ["offset"]["beta"]), float(MQ["offset"]["eb_k"])
t0 = time.time()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id", "batter_hand", "pitcher_hand"])
tro = tr[tr.season == 2024].sort_values("row_id", kind="stable")
yo = tro.control_success.to_numpy(np.float64)
YB = float(yo.mean())
m = (tr.season <= 2023) & ~((tr.season <= 2022) & (tr.game_type == "F"))
dd = tr[m & tr.control_success.notna()].copy()
for c in ("pitcher_id", "batter_hand", "pitcher_hand"):
    dd[c] = dd[c].astype("Int64")
ov = dd.groupby("pitcher_id").control_success.mean()
g = dd.groupby(["pitcher_id", "batter_hand"]).control_success.agg(["sum", "count"]).reset_index()
g["overall"] = g.pitcher_id.map(ov)
NUMv = g["sum"].to_numpy(np.float64) - g["count"].to_numpy(np.float64) * g.overall.to_numpy(np.float64)
KK = [f"{int(a)}|{int(b)}" for a, b in zip(g.pitcher_id, g.batter_hand)]
NUM23, NC23 = dict(zip(KK, NUMv)), dict(zip(KK, g["count"].to_numpy(np.int64)))
lgp = dd.groupby("pitcher_hand").control_success.mean()
lpb = dd.groupby(["pitcher_hand", "batter_hand"]).control_success.agg(["mean", "count"])
LD23 = {f"{int(a)}|{int(b)}": float(r["mean"] - lgp.loc[a]) for (a, b), r in lpb.iterrows()}


def keys_of(df, a, b):
    ka, kb = pd.to_numeric(df[a], errors="coerce"), pd.to_numeric(df[b], errors="coerce")
    ok = (ka.notna() & kb.notna()).to_numpy()
    out = np.array([""] * len(df), dtype=object)
    out[ok] = [f"{int(x)}|{int(y)}" for x, y in zip(ka[ok].astype(int), kb[ok].astype(int))]
    return out


kp_o, kl_o = keys_of(tro, "pitcher_id", "batter_hand"), keys_of(tro, "pitcher_hand", "batter_hand")
hit_o = np.array([k in NUM23 for k in kp_o])
num_o = np.array([NUM23.get(k, 0.0) for k in kp_o], np.float64)
nc_o = np.array([NC23.get(k, 0) for k in kp_o], np.float64)
Lrow = np.array([LD23.get(k, 0.0) for k in kl_o], np.float64)

V = {n: np.load(p).astype(np.float64) for n, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
L4 = {n: lg(V[n]) for n in ("p_lgb", "p_cat", "p_leaf", "p_nn")}
LV0 = (W4["LGB"] * L4["p_lgb"] + W4["CAT"] * L4["p_cat"]
       + W4["NN"] * L4["p_nn"] + W4["LEAF"] * L4["p_leaf"])
LE = lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy"))
                for s in AA], axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy"))
                 for s in AA], axis=0)


def rc(z):
    c = 0.0
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - YB) < 1e-14:
            break
        d = (mu - YB) / float((p * (1 - p)).mean())
        c -= d
        if abs(d) < 1e-14:
            break
    return c


def zof(lq):
    return (BO["w_out"] * ((1 - BO["w_ebm"])
                           * (BO["w_v6_inner"] * LV0 + (1 - BO["w_v6_inner"]) * lq)
                           + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])
            + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70)


def pts_of_z(z):
    p = expit(z + rc(z))
    return pts(float(np.mean((p - yo) ** 2)))


def d_of(k=EBK):
    return np.where(hit_o, num_o / (nc_o + k), 0.0) + np.where(hit_o, 0.0, Lrow)


def lq_of(beta=BETA_Q, k=EBK):
    raw = V["p_q"] + beta * d_of(k)
    return lg(np.clip(raw, EPS, 1 - EPS)), int(((raw < EPS) | (raw > 1 - EPS)).sum())


LQ_DEP, _ = lq_of()
MU0, SD0 = float(LQ_DEP.mean()), float(LQ_DEP.std())
base = pts_of_z(zof(LQ_DEP))
R = {"exp_id": "exp_080", "part": "stage2c_extended_scan",
     "honest_base_with_deployed_q_offset": base,
     "why": ("stage2b 에서 q_beta 는 상한 1.6, q_k 는 하한 20 에 argmax 가 닿았다 — "
             "그 두 값은 내가 임의로 잡은 구간이지 사전 등록된 것이 아니다. "
             "🔴 문턱 +2.0 은 그대로 두고 주사 구간만 물리적 한계까지 넓힌다 (S27).")}


def scan(f, lo, hi, n=41):
    def gr(x):
        lq, nc = f(x)
        return pts_of_z(zof(lq)) - base, nc

    def gs(x):
        lq, _ = f(x)
        mu, sd = float(lq.mean()), float(lq.std())
        return pts_of_z(zof(MU0 + (lq - mu) * (SD0 / sd))) - base, sd / SD0

    rr = minimize_scalar(lambda x: -(gr(x)[0]), bounds=(lo, hi), method="bounded",
                         options={"xatol": (hi - lo) * 1e-7})
    xs = float(rr.x)
    graw, nclip = gr(xs)
    gsd, sdr = gs(xs)
    grid = {}
    for v in np.linspace(lo, hi, n):
        a, c = gr(v)
        b, _ = gs(v)
        grid[f"{v:.6g}"] = {"raw": a, "sd_matched": b, "n_clipped": c}
    return {"scan_range": [lo, hi], "argmax": xs, "at_boundary":
            bool(min(abs(xs - lo), abs(xs - hi)) < (hi - lo) * 1e-4),
            "raw_gain_pts": graw, "n_clipped_at_argmax": nclip,
            "sd_matched_gain_pts": gsd, "sd_ratio": sdr,
            "information_share": gsd / graw if graw else None,
            "sd_matched_argmax_scan": max(grid, key=lambda z: grid[z]["sd_matched"]),
            "sd_matched_max_on_grid": max(v["sd_matched"] for v in grid.values()),
            "grid": grid}


R["q_beta"] = scan(lambda b: lq_of(beta=b), 0.0, 6.0, 61)
R["q_k"] = scan(lambda k: lq_of(k=k), 1.0, 1200.0, 61)
for nm in ("q_beta", "q_k"):
    v = R[nm]
    print(f"[{nm}] 구간 {v['scan_range']} argmax {v['argmax']:.6g} 경계닿음 "
          f"{v['at_boundary']} raw {v['raw_gain_pts']:+.5f}점 sd맞춤 "
          f"{v['sd_matched_gain_pts']:+.5f}점 (정보 {100*(v['information_share'] or 0):.1f}%) "
          f"clip {v['n_clipped_at_argmax']}행 · sd맞춤 격자최대 "
          f"{v['sd_matched_max_on_grid']:+.5f}점 @ {v['sd_matched_argmax_scan']}", flush=True)
R["one_dof_in_sample_floor_honest_pts"] = float(
    K * float(np.mean((yo - expit(zof(LQ_DEP) + rc(zof(LQ_DEP)))) ** 2)) / len(yo))
R["debiased"] = {nm: R[nm]["sd_matched_gain_pts"] - R["one_dof_in_sample_floor_honest_pts"]
                 for nm in ("q_beta", "q_k")}
R["wall_sec"] = time.time() - t0
json.dump(R, open(f"{OUT}/s4_extend.json", "w"), indent=1, ensure_ascii=False)
print(f"1자유도 표본내 바닥 {R['one_dof_in_sample_floor_honest_pts']:.5f}점 · "
      f"편의보정 {R['debiased']}", flush=True)
print(f"기록 {OUT}/s4_extend.json · {R['wall_sec']:.1f}s", flush=True)

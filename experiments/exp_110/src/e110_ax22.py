# exp_110 stage3 — re-read the 2026-08-16 22-axis screen as a VERTEX gain.
# The original (axis_scan.py) reported gain(s=1) = 2*K*L - K*G, which is
# sign- AND scale-dependent: a large negative can mean "L is large and the sign
# is reversed" OR "L~0 and we applied pure noise at full strength".  These are
# not separable from one number.  Here we report K*L and K*G separately, so the
# sign-invariant maximum  K*L^2/G  is available for every axis.
# Read-only: no retraining, uses the SAME stored predictions and the SAME tables.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import glob, json
import numpy as np, pandas as pd

H = "/home/<user>/SungjinPark"
REF, EPS, K_PERS, NB = 0.2498069, 1e-6, 400, 1000
K_PTS = 100000.0 / REF
BETA = {"R": 0.597044, "F": 0.492041}
rng = np.random.default_rng(42)

tr = pd.read_csv(f"{H}/data/train.csv")
tr["cnt"] = tr.balls_before.astype(str) + "-" + tr.strikes_before.astype(str)
tr["inn3"] = np.clip((tr.inning - 1) // 2, 0, 4)
tr["lev"] = pd.qcut(tr.li, 5, labels=False, duplicates="drop")
tr["sdf"] = np.clip(tr.score_diff_pitcher_team, -4, 4)
tr["month"] = tr.game_month
tr["bn"] = pd.qcut(tr.asof_batter_n, 5, labels=False, duplicates="drop")
tr["pn"] = pd.qcut(tr.asof_pitcher_n, 5, labels=False, duplicates="drop")
tr["mix"] = pd.qcut(tr.asof_pitcher_breaking_rate, 5, labels=False, duplicates="drop")
tr["form"] = pd.qcut(tr.asof_pitcher_prev3_game_success_rate, 5, labels=False, duplicates="drop")
tr["bsucc"] = pd.qcut(tr.asof_batter_success_rate, 5, labels=False, duplicates="drop")

def role_map(past):
    g = past.groupby("pitcher_id").inning.mean()
    return (g <= 3).map({True: "SP", False: "RP"})

AXES = [
 ("[ctrl] pitcher_hand x batter_hand", ["pitcher_hand","batter_hand"], "pitcher_id"),
 ("batter_id",["batter_id"],"batter_id"),
 ("batter_id x pitcher_hand",["batter_id","pitcher_hand"],"batter_id"),
 ("batter_team",["batter_team_id"],"batter_id"),
 ("pitcher_team",["pitcher_team_id"],"pitcher_id"),
 ("pitcher x inn3",["pitcher_id","inn3"],"pitcher_id"),
 ("pitcher x count",["pitcher_id","cnt"],"pitcher_id"),
 ("pitcher x outs",["pitcher_id","outs_before"],"pitcher_id"),
 ("inn3 (global)",["inn3"],"pitcher_id"),
 ("count (global)",["cnt"],"pitcher_id"),
 ("count x pitcher_hand",["cnt","pitcher_hand"],"pitcher_id"),
 ("base_state",["base_state"],"pitcher_id"),
 ("leverage q5",["lev"],"pitcher_id"),
 ("score_diff",["sdf"],"pitcher_id"),
 ("month",["month"],"pitcher_id"),
 ("role x inn3",["role","inn3"],"pitcher_id"),
 ("pitcher-exp q5 x count",["pn","cnt"],"pitcher_id"),
 ("batter-exp q5",["bn"],"batter_id"),
 ("batter-succ q5 x pitcher_hand",["bsucc","pitcher_hand"],"batter_id"),
 ("pitcher breaking q5",["mix"],"pitcher_id"),
 ("pitcher prev3 form q5",["form"],"pitcher_id"),
 ("pitcher x batter history",["pitcher_id","batter_id"],"pitcher_id"),
]
SHRINK = [20, 60, 200, 600, 2000]

def deployed_offset(cur, past):
    ov = past.groupby("pitcher_id").control_success.mean()
    cell = past.groupby(["pitcher_id","batter_hand"]).control_success.agg(["sum","count"])
    s_map, n_map = cell["sum"].to_dict(), cell["count"].to_dict()
    key = list(zip(cur.pitcher_id, cur.batter_hand))
    hit = np.array([k in n_map for k in key])
    o = np.nan_to_num(cur.pitcher_id.map(ov).to_numpy())
    s_ = np.array([s_map.get(k,0.0) for k in key], float)
    n_ = np.array([n_map.get(k,0) for k in key], float)
    d_s = (s_ - n_*o)/(n_ + K_PERS)
    f = past.groupby(["pitcher_hand","batter_hand"]).control_success.mean()
    c = past.groupby("pitcher_hand").control_success.mean()
    L = np.nan_to_num(np.array([f.get((a,b),np.nan)-c.get(a,np.nan)
                                for a,b in zip(cur.pitcher_hand, cur.batter_hand)]))
    return np.where(hit, d_s, L)

PRED = {s: f"{H}/experiments/exp_030/pred/pred_C5B3_u{s}_seed*_st2.npy" for s in (2022,2023)}
tabs = []
for s in (2022, 2023):
    p = np.mean([np.load(x) for x in sorted(glob.glob(PRED[s]))], axis=0)
    cur = tr[tr.season == s].copy(); past = tr[tr.season <= s-1]
    off = deployed_offset(cur, past)
    beta = np.array([BETA.get(g,0.59072) for g in cur.game_type])
    r = cur.control_success.to_numpy(float) - np.clip(p + beta*off, EPS, 1-EPS)
    cur["resid"] = r - r.mean()
    cur["role"] = cur.pitcher_id.map(role_map(past))
    tabs.append(cur)
TAB = pd.concat(tabs, ignore_index=True)
print(f"table material: 2022+2023 {len(TAB):,} rows", flush=True)

S = 2024
cur24 = tr[tr.season == S].copy(); past24 = tr[tr.season <= S-1]
cur24["role"] = cur24.pitcher_id.map(role_map(past24))
off24 = deployed_offset(cur24, past24)
beta24 = np.array([BETA.get(g,0.59072) for g in cur24.game_type])
y24 = cur24.control_success.to_numpy(float); n_tot = len(cur24)

EVAL = [("51col exp_030", f"{H}/experiments/exp_030/pred/pred_C5B3_u2024_seed*_st2.npy"),
        ("67col exp_048", f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed*_at1400.npy")]
RES = {}
for label, pat in EVAL:
    f = sorted(glob.glob(pat))
    if not f: continue
    p = np.mean([np.load(x) for x in f], axis=0)
    p_off = np.clip(p + beta24*off24, EPS, 1-EPS)
    g = np.mean(y24 - p_off)
    base = np.clip(p_off + g, EPS, 1-EPS)
    r0 = y24 - base
    print("\n" + "="*104); print(f"### {label}  eval 2024 ({n_tot:,})  base Brier {np.mean(r0**2):.7f}", flush=True)
    print(f" {'axis':<32}{'cov%':>6}{'gain(1)':>10}{'K*L':>10}{'K*G':>10}{'vertex':>9}{'s*':>8}{'t_L':>7}")
    print(" " + "-"*100)
    rows = []
    for name, cols, bunit in AXES:
        if any(c not in TAB.columns for c in cols): continue
        gt = TAB.groupby(cols, observed=True).resid.agg(["mean","count"])
        keys24 = list(zip(*[cur24[c] for c in cols])) if len(cols) > 1 else list(cur24[cols[0]])
        m, n = gt["mean"].to_dict(), gt["count"].to_dict()
        mv = np.array([m.get(k, np.nan) for k in keys24])
        nv = np.array([n.get(k, 0) for k in keys24], float)
        cov = 100.0*np.mean(~np.isnan(mv)); mv = np.nan_to_num(mv)
        blk = pd.factorize(cur24[bunit])[0]; nb = blk.max()+1
        best = None
        for k in SHRINK:
            c = nv*mv/(nv+k)
            b = np.mean((np.clip(base+c, EPS, 1-EPS)-y24)**2)
            if best is None or b < best[0]: best = (b, k, c)
        _, k, c = best
        # applied gain at s=1 (what the original reported)
        d1 = (base-y24)**2 - (np.clip(base+c, EPS, 1-EPS)-y24)**2
        gain1 = K_PTS*d1.mean()
        # separate L and G.  c is a probability-space additive move (mean re-anchored).
        c0 = c - c.mean()
        L = float(np.mean(c0*r0)); G = float(np.mean(c0*c0))
        KL, KG = K_PTS*L, K_PTS*G
        vertex = KL*KL/KG if KG > 0 else 0.0
        s_star = L/G if G > 0 else 0.0
        bs = np.bincount(blk, weights=c0*r0, minlength=nb)
        draws = rng.integers(0, nb, size=(NB, nb))
        se = float(bs[draws].sum(axis=1).std(ddof=1))/n_tot
        t_L = L/se if se > 0 else 0.0
        rows.append((vertex, name, cov, gain1, KL, KG, s_star, t_L, k))
        RES.setdefault(label, {})[name] = dict(cov=cov, gain_at_1=gain1, K_L=KL,
            K_G=KG, vertex_gain_pts=vertex, s_star=s_star, t_of_L=t_L, shrink_k=k)
    for vtx, name, cov, g1, KL, KG, ss, tL, k in sorted(rows, key=lambda r: -r[0]):
        print(f" {name:<32}{cov:>6.1f}{g1:>10.3f}{KL:>10.2f}{KG:>10.1f}{vtx:>9.3f}{ss:>8.3f}{tL:>7.2f}")
json.dump(RES, open(f"{H}/exp110_ax22.json","w"), ensure_ascii=False, indent=1)
print("\nWROTE ~/SungjinPark/exp110_ax22.json")

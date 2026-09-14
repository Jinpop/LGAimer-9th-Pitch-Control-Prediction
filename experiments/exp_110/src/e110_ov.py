# exp_110 stage3b — overlap among the axes the vertex re-read raised, on the
# 67-col base.  Joint solve (eigen-truncated, S44) + pairwise rho.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import glob, json, numpy as np, pandas as pd
exec(open("/home/<user>/SungjinPark/e110_ax22.py").read().split("EVAL = [")[0]
     .replace('json.dump', '#json.dump'))
H = "/home/<user>/SungjinPark"
p = np.mean([np.load(x) for x in sorted(glob.glob(
    f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed*_at1400.npy"))], axis=0)
p_off = np.clip(p + beta24*off24, EPS, 1-EPS)
g = np.mean(y24 - p_off); base = np.clip(p_off + g, EPS, 1-EPS); r0 = y24 - base
SHRINK = [20, 60, 200, 600, 2000]
PICK = ["batter_id", "batter_id x pitcher_hand", "batter_team",
        "[ctrl] pitcher_hand x batter_hand", "pitcher x batter history",
        "pitcher_team", "leverage q5", "base_state",
        "batter-succ q5 x pitcher_hand", "pitcher prev3 form q5"]
AX = {n: (c, b) for n, c, b in AXES}
V, names = [], []
for name in PICK:
    cols, bunit = AX[name]
    gt = TAB.groupby(cols, observed=True).resid.agg(["mean", "count"])
    keys24 = list(zip(*[cur24[c] for c in cols])) if len(cols) > 1 else list(cur24[cols[0]])
    m, n_ = gt["mean"].to_dict(), gt["count"].to_dict()
    mv = np.nan_to_num(np.array([m.get(k, np.nan) for k in keys24]))
    nv = np.array([n_.get(k, 0) for k in keys24], float)
    best = None
    for k in SHRINK:
        c = nv*mv/(nv+k)
        b = np.mean((np.clip(base+c, EPS, 1-EPS)-y24)**2)
        if best is None or b < best[0]: best = (b, k, c)
    c0 = best[2] - best[2].mean()
    V.append(c0); names.append(name)
V = np.array(V)
K_PTS = 100000.0/REF
R = np.corrcoef(V)
print("pairwise rho (67col base):")
print("      " + " ".join(f"{i:>6d}" for i in range(len(names))))
for i, n_ in enumerate(names):
    print(f"{i:2d} {n_[:30]:<32}" + " ".join(f"{R[i,j]:+6.3f}" for j in range(len(names))))
L = V @ r0 / len(r0); G = V @ V.T / len(r0)
KL, KG = K_PTS*L, K_PTS*G
print("\nsingle-axis vertex:", {names[i]: round(KL[i]**2/KG[i, i], 3) for i in range(len(names))})
def joint(idx):
    Gs = KG[np.ix_(idx, idx)]; Ls = KL[idx]
    w, Q = np.linalg.eigh(Gs); keep = w > w.max()*1e-10
    u = Q[:, keep] @ (Q[:, keep].T @ Ls / w[keep])
    return float(2*Ls @ u - u @ Gs @ u), u, int(keep.sum())
for lab, idx in [("batter trio (0,1,2)", [0,1,2]),
                 ("top4 (0,1,2,6)", [0,1,2,6]),
                 ("all 10", list(range(10)))]:
    gain, u, rk = joint(idx)
    print(f"{lab:22s} joint vertex = {gain:8.3f} pts (naive sum "
          f"{sum(KL[i]**2/KG[i,i] for i in idx):8.3f}, rank {rk}/{len(idx)})")
json.dump({"names": names, "rho": R.tolist(), "K_L": KL.tolist(),
           "K_G": KG.tolist()}, open(f"{H}/exp110_overlap.json", "w"), indent=1)
print("WROTE ~/SungjinPark/exp110_overlap.json")

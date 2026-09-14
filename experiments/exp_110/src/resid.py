# Residualize each exp_076/075 axis against the directions ALREADY SHIPPED or
# already model inputs, using the recorded pairwise correlation matrix + M.
# M is linear in the direction, so M(u_perp) follows from Gram-Schmidt on rho.
import json, numpy as np
K, EPS, KMC2 = 400309.19882517256, 0.02, 28920.30698468467
B = "<PROJECT_ROOT>/experiments"

def load(e, blocks, ctrl):
    d = json.load(open(f"{B}/{e}/out/screen.json"))
    M, cov = {}, d.get("row_coverage", {})
    for b in blocks:
        for k, v in d[b].items():
            M.setdefault(k, v["M"])
    P = d["pairwise_corr"]
    R = {}
    for k, v in P.items():
        a, b2 = k.split("|"); R[(a, b2)] = v; R[(b2, a)] = v
    for k in M: R[(k, k)] = 1.0
    names = sorted(M)
    def rho(a, b2): return R.get((a, b2))
    out = []
    for a in names:
        if a in ctrl or a.startswith("perm"): continue
        C = [c for c in ctrl if c in M and rho(a, c) is not None]
        if not C: continue
        # Gram matrix of controls, and rho vector of a vs controls
        G = np.array([[rho(c1, c2) if rho(c1, c2) is not None else 0.0
                       for c2 in C] for c1 in C], float)
        r = np.array([rho(a, c) for c in C], float)
        try: beta = np.linalg.solve(G + 1e-9*np.eye(len(C)), r)
        except Exception: continue
        var_perp = max(1.0 - float(r @ beta), 1e-12)
        M_perp = M[a] - float(sum(beta[i]*M[c] for i, c in enumerate(C)))
        q = cov.get(a, {}).get("row_finite", 1.0)
        KL_raw = M[a]*K/EPS
        # perp direction renormalised to unit variance
        KL_perp = (M_perp/np.sqrt(var_perp))*K/EPS
        KG = KMC2*q
        out.append((a, M[a], KL_raw, KL_raw**2/KG, np.sqrt(var_perp),
                    KL_perp, KL_perp**2/KG))
    return sorted(out, key=lambda x: -x[6])

SETS = [
  ("exp_076", ["screen_pitcher_block", "supplementary_screen"],
   ["ctrl_asof_psr", "ctrl_prev3_suc", "supp_ins_p_success", "supp_ins_p_success_dev"]),
  ("exp_075", ["screen_batter_block", "supplementary_screen"],
   ["ctrl_asof_psr", "ctrl_asof_bsr", "ctrl_asof_bmr"]),
  ("exp_074", ["screen", "supplementary_screen"],
   ["ctrl_asof_psr", "ctrl_prev3_suc", "ins_p_success", "ins_p_success_dev"]),
]
res = {}
for e, blocks, ctrl in SETS:
    print("="*74); print(e, " controls:", ctrl)
    print(f"{'axis':30s} {'K*L':>8s} {'gain':>7s} | {'sd_perp':>7s} {'K*L_p':>8s} {'gain_p':>7s}")
    for a, m, kl, g, sp, klp, gp in load(e, blocks, ctrl):
        print(f"{a:30s} {kl:8.1f} {g:7.3f} | {sp:7.3f} {klp:8.1f} {gp:7.4f}")
        res.setdefault(e, {})[a] = dict(K_L_raw=kl, gain_raw=g, sd_perp=sp,
                                        K_L_perp=klp, gain_perp=gp)
json.dump(res, open("<SCRATCH>/resid.json","w"), indent=1)

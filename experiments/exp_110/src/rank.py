import json, math
K = 400309.19882517256
EPS = 0.02              # exp_073~076 s2_screen.py:83  EPS_PROBE
KMC2 = 28920.30698468467  # exp_109 K*mean(c^2), continuous-direction curvature law (S201)
B = "<PROJECT_ROOT>/experiments"

rows = []
# ---- exp_073~076 screens ------------------------------------------------
SRC = {"exp_073": ["screen"],
       "exp_074": ["screen", "supplementary_screen"],
       "exp_075": ["screen_batter_block", "supplementary_screen"],
       "exp_076": ["screen_pitcher_block", "supplementary_screen"]}
for e, keys in SRC.items():
    d = json.load(open(f"{B}/{e}/out/screen.json"))
    cov = d.get("row_coverage", {})
    for kk in keys:
        for name, v in d[kk].items():
            nnf = v.get("n_nonfinite", 0)
            q = cov.get(name, {}).get("row_finite")
            if q is None:
                q = 1.0 - nnf / 253507.0
            KL = v["M"] * K / EPS
            KG = KMC2 * q
            rows.append(dict(exp=e, block=kk, axis=name, M=v["M"], t=v["t"],
                             q=q, KL=KL, KG=KG, gain=KL * KL / KG,
                             passed=v.get("pass_t2")))
# ---- exp_095 stage2 -----------------------------------------------------
d = json.load(open(f"{B}/exp_095/out/s2_leftover.json"))
for name, v in d["candidates"].items():
    rows.append(dict(exp="exp_095", block="s2_leftover", axis=name,
                     M=v["L"] * EPS, t=v.get("boot_t_of_L"), q=None,
                     KL=v["L"] * K, KG=v["K_G"], gain=v["vertex_gain_pts"],
                     passed=v["exceeds_null_A_p95"],
                     rho=v["rho_vs_bought_axes"], status=v["status"],
                     cite=v["code_citation"]))
rows.sort(key=lambda r: -r["gain"])
print(f"{'exp':8s} {'axis':32s} {'t':>7s} {'cov':>6s} {'K*L':>9s} {'K*G':>9s} {'maxgain':>8s} p")
for r in rows:
    print(f"{r['exp']:8s} {r['axis']:32s} {r['t'] if r['t'] is not None else 0:7.2f} "
          f"{(r['q'] or 1.0):6.3f} {r['KL']:9.1f} {r['KG']:9.0f} {r['gain']:8.4f} "
          f"{'Y' if r['passed'] else '.'}")
json.dump(rows, open("<SCRATCH>/rank.json","w"))

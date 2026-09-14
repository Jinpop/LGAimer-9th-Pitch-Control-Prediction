"""exp_114 stage5 — 출하 형태의 짝지은 델타. w 를 배포값 0.40 에 **고정**했을 때
스텝만 바꾸는 이동이 얼마인가. (s3 의 판정은 후보 자신의 argmax w 를 썼다.)
그리고 s4 가 낸 배포 최적 스텝 주변을 표로 낸다.
"""
import json
import sys

sys.path.insert(0, "/home/<user>/SungjinPark/exp114/src")
import numpy as np  # noqa: E402
from surf114 import (B, BUNDLE, CKPTS, TRANSFER, Surface, zbundle,  # noqa: E402
                     bootstrapper)

PRE = json.load(open(f"{B}/out/s2_sigma.json"))["PREREG"]
S4 = json.load(open(f"{B}/out/s4_scale.json"))
S = Surface()
boot = bootstrapper(S.pit)
tagf = lambda s: f"M_s{s}"
A_K, AW = 2160, 0.40
CAND = [1088, 1224, 1360, 1496, 1632, 1768]

R = {"exp_id": "exp_114", "stage": "s5_ship", "anchor": {"steps": A_K, "w": AW},
     "설명": "w 를 배포값 0.40 에 고정하고 스텝만 옮기는 이동 — 출하 시 실제로 하는 일"}
zc_a = {b: (1 - AW) * S.Z_MATE + AW * zbundle(tagf, b, A_K) for b in range(5)}
anc = {b: S.sc(S.root(zc_a[b]), "abd") - S.base_abd for b in range(5)}
za = np.mean([zc_a[b] for b in range(5)], 0)
sqa = (S.prob(S.root(za), "abd") - S.y) ** 2

rows = {}
for k in CAND:
    zb = {b: (1 - AW) * S.Z_MATE + AW * zbundle(tagf, b, k) for b in range(5)}
    g = {b: S.sc(S.root(zb[b]), "abd") - S.base_abd for b in range(5)}
    d = [g[b] - anc[b] for b in range(5)]
    m, sdd = float(np.mean(d)), float(np.std(d, ddof=1))
    sig_fit = max(sdd / np.sqrt(5), PRE["sigma_floor_for_mean"])
    zz = np.mean([zb[b] for b in range(5)], 0)
    bt = boot((S.prob(S.root(zz), "abd") - S.y) ** 2 - sqa)
    sig = float(np.hypot(sig_fit, bt["boot_se_pts"]))
    rows[str(k)] = {"gain_mean": float(np.mean(list(g.values()))),
                    "delta_mean_pts": m, "delta_per_bundle": d,
                    "delta_sd_ddof1": sdd, "sigma_fit_used": float(sig_fit),
                    "boot_se_pts": bt["boot_se_pts"], "boot_ci95_pts": bt["ci95_pts"],
                    "sigma_combined": sig, "t_combined": m / sig,
                    "lb_equivalent": m * TRANSFER,
                    "n_bundles_positive": int(sum(1 for x in d if x > 0)),
                    "PASS": bool(m >= PRE["min_pts_from_LB"] and m / sig >= PRE["min_t"])}
    print(k, round(m, 3), "t", round(m / sig, 3), rows[str(k)]["PASS"], flush=True)

R["anchor_gain_per_bundle"] = {str(b): anc[b] for b in range(5)}
R["anchor_gain_mean"] = float(np.mean(list(anc.values())))
R["candidates_fixed_w0.40"] = rows
kb = max(rows, key=lambda k: rows[k]["delta_mean_pts"])
R["best_fixed_w"] = {"steps": int(kb)} | rows[kb]

p = S4["law_kstar_free_w"]["p_exponent"]
n_oot, n_dep = 1116277, 1475092
R["deploy_map"] = {
    "p_exponent_used": p,
    "map": "k_OOT = k_deploy * (n_OOT/n_deploy)^p",
    "factor": float((n_oot / n_dep) ** p),
    "shipped_D2160_to_OOT": float(2160 * (n_oot / n_dep) ** p),
    "kstar_deploy_from_s4": S4["law_kstar_free_w"]["kstar_at_n_deploy"],
    "existing_exp090_packages": ["deploy_D1440", "deploy_D1632", "deploy_D2160"],
    "권고": "D1632 (이미 구워져 있다) 또는 s4 점추정 1588"}
json.dump(R, open(f"{B}/out/s5_ship.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "candidates_fixed_w0.40"},
                 ensure_ascii=False, indent=1))

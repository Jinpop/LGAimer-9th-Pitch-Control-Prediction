"""exp_114 stage2 — 1σ 실측. **후보를 하나도 보기 전에** 돈다.

재는 것은 오직 **배포 앵커 한 점의 묶음 간 흩어짐**이다 — config 가 지정한 귀무
("스텝을 바꾸는 대신 시드만 바꾼 같은 개수의 묶음")가 정확히 이것이다.
스텝 축 비교(1632 대 2160)는 **후보 결과**이므로 여기서 계산하지 않는다.
"""
import json
import sys

sys.path.insert(0, "/home/<user>/SungjinPark/exp114/src")
import numpy as np  # noqa: E402
from surf114 import (B, BUNDLE, LB_FLOOR, TRANSFER, BASE_EXPECT,  # noqa: E402
                     Surface, zbundle)

ANCHOR_STEPS, ANCHOR_W = 2160, 0.40        # config 지정 배포 배치 (스텝 정합 읽기)
S = Surface()
R = {"exp_id": "exp_114", "stage": "s2_sigma",
     "면": "exp_097 sB1 / exp_113 f4 그대로",
     "G1_surface": {"w0_rc": S.base_rc, "expected": BASE_EXPECT,
                    "abs_diff": abs(S.base_rc - BASE_EXPECT),
                    "PASS": bool(abs(S.base_rc - BASE_EXPECT) < 1e-9)}}
assert R["G1_surface"]["PASS"], S.base_rc

tagf = lambda s: f"M_s{s}"
g = {}
for b in range(5):
    zd = zbundle(tagf, b, ANCHOR_STEPS)
    g[b] = S.sc(S.root((1 - ANCHOR_W) * S.Z_MATE + ANCHOR_W * zd), "abd") - S.base_abd
    print(b, g[b], flush=True)

vals = np.array([g[b] for b in range(5)])
sd_pt = float(vals.std(ddof=1))
nb = len(vals)
pairs = [(i, j) for i in range(nb) for j in range(i + 1, nb)]
null_pair = [g[i] - g[j] for i, j in pairs]

R["anchor"] = {"steps": ANCHOR_STEPS, "w": ANCHOR_W,
               "gain_per_bundle": {str(b): g[b] for b in range(5)},
               "mean": float(vals.mean()), "sd_ddof1": sd_pt,
               "min": float(vals.min()), "max": float(vals.max())}
R["seed_only_null"] = {
    "설명": "같은 (스텝, w) 를 서로소 묶음끼리 짝지은 차 — 스텝 축이 시드 재뽑기와 구별되는지의 눈금",
    "pairs": [{"b": [i, j], "delta": g[i] - g[j]} for i, j in pairs],
    "sd_ddof1": float(np.std(null_pair, ddof=1)),
    "max_abs": float(np.max(np.abs(null_pair)))}

sigma_floor_mean = sd_pt * np.sqrt(2.0) / np.sqrt(nb)
min_pts = LB_FLOOR / TRANSFER
R["PREREG"] = {
    "판정량": "Δ̄ = mean_b [ gain_b(k,w) − gain_b(2160, 0.40) ]  (같은 묶음 안 짝지음, 묶음 5개)",
    "sigma_fit_point_sd": sd_pt,
    "sigma_floor_for_mean": float(sigma_floor_mean),
    "sigma_rule": "σ_fit = max( sd_b(Δ_b)/√5 , sigma_floor_for_mean ); "
                  "σ_combined = hypot(σ_fit, 짝지은 투수블록 부트스트랩 se)",
    "min_t": 2.0,
    "min_pts_from_LB": float(min_pts),
    "전이비": TRANSFER, "LB_검출한계": LB_FLOOR,
    "행동_문턱": f"Δ̄ >= {min_pts:.5f} 점 AND t >= 2.0 (둘 다)",
    "t2_equivalent_pts_at_floor_sigma": float(2.0 * sigma_floor_mean),
    "확정_문턱_점수": float(max(min_pts, 2.0 * sigma_floor_mean))}
json.dump(R, open(f"{B}/out/s2_sigma.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(R["anchor"] | R["PREREG"], ensure_ascii=False, indent=1))

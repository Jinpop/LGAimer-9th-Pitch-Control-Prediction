#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""표본 밖(out-of-time 정직면)에서 같은 K*L 을 다시 잰다.

표본 안(cov·band)에서는 양성 대조가 ★부호까지★ 틀렸다 — 배포 팔이 <=2024 학습인데
그 프레임이 2024 행이라 잔차가 정직하지 않다(S210).  여기서는 exp_111 이 만든
★<=2023 학습 팔을 2024 행에 적용한★ OOT 예측을 쓴다.

  base_oot = y24 - r_oot          (exp111/r_oot.npy — 전역 평균 앵커 이후 잔차)
  K*L = K * mean( r_oot * p0*(1-p0) * z )
  K*G = K * mean( (p0*(1-p0) * z)^2 )

⚠️ 이 면의 값은 ★순위에만★ 쓴다.  걸음 크기로 쓰지 마라 (S232·S245·S252 —
   정직면 유래 몫 100% 축은 4/4 로 졌고 합계 -38.6).
⚠️ 기반이 배포와 다르다: p_oot 는 사후 축(3·4·5·6)을 하나도 안 실은 블렌드다.
"""
import json
import sys

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK = f"{B}/pkg", f"{B}/work"
FR24 = f"{H}/build_AC/work/tr2024/test.csv"
K = 400309.19882517256
U = 0.008

COLS = ["asof_batter_middle_rate", "asof_batter_success_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_success_rate"]

sys.path.insert(0, f"{PKG}/PAR")
import inseason                                                   # noqa: E402
import splits as SPL                                              # noqa: E402

tr = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "control_success"] + COLS)
t24 = tr[tr["season"] == 2024].reset_index(drop=True)
ids24 = pd.read_csv(FR24, encoding="utf-8-sig", usecols=["row_id"])["row_id"].to_numpy()
assert (t24["row_id"].to_numpy() == ids24).all(), "tr2024 프레임 행 순서 불일치"
y = t24["control_success"].to_numpy(np.float64)
r_oot = np.load(f"{H}/exp111/r_oot.npy")
p_raw = np.load(f"{H}/exp111/p_oot.npy")
assert len(r_oot) == len(y) == 253507
p0 = y - r_oot
print(f"base_oot: n={len(p0)} min={p0.min():.6f} max={p0.max():.6f} "
      f"mean={p0.mean():.8f}  y_mean={y.mean():.8f}  mean(r)={r_oot.mean():.3e}")
assert (p0 > 0).all() and (p0 < 1).all(), "base_oot 가 확률 범위를 벗어난다"
brier_oot = float(((p0 - y) ** 2).mean())
print(f"OOT 정직면 Brier={brier_oot:.6f}  환산={100000 * (1 - brier_oot / 0.2498069):.2f}점"
      f"   (원 p_oot Brier={float(((p_raw - y) ** 2).mean()):.6f})")

CS = {c: (float(np.nanmean(t24[c].to_numpy(np.float64))),
          float(np.nanstd(t24[c].to_numpy(np.float64)))) for c in COLS}
w = p0 * (1.0 - p0)

# 출하 5축 delta (배포 규약 그대로)
tdf = pd.read_csv(FR24, encoding="utf-8-sig", low_memory=False)
PBL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
AX = PBL["axis_offsets"]
ins = inseason.apply(tdf, inseason.load(f"{PKG}/PAR/model/ins/deploy_anchor.npz"))
cur = np.clip(ins["ins_p_success"].to_numpy(np.float64), 0.02, 0.98)
pri = np.clip(cur - ins["ins_p_success_dev"].to_numpy(np.float64), 0.02, 0.98)
off = np.nan_to_num(np.log(cur / (1 - cur)) - np.log(pri / (1 - pri)))
spT = SPL.load(f"{PKG}/PAR/model/deploy_splits.npz")
spA = SPL.apply(tdf, spT)
sd1, sd2 = float(AX[2]["sd1"]), float(AX[2]["sd2"])
raw = {"beta_ins_off": off,
       "ctrl_asof_psr": tdf["asof_pitcher_success_rate"].to_numpy(np.float64),
       "sp_pit_split": (spA["sp_pit_bhand"].to_numpy(np.float64) / sd1
                        + spA["sp_pit_strk2"].to_numpy(np.float64) / sd2),
       "sp_bat_phand": spA["sp_bat_phand"].to_numpy(np.float64),
       "ctrl_asof_mid": tdf["asof_pitcher_middle_rate"].to_numpy(np.float64)}
D = []
for a in AX:
    r = raw[a["name"]]
    d = np.zeros(len(y), dtype=np.float64)
    f = np.isfinite(r)
    d[f] = (r[f] - float(a["center"])) / float(a["scale"])
    D.append(d)
D = np.stack(D, axis=1)
Wt = w ** 2
M = (D * Wt[:, None]).T @ D / len(y)


def resid(b):
    v = (D * Wt[:, None]).T @ b / len(y)
    coef, *_ = np.linalg.lstsq(M, v, rcond=None)
    return b - D @ coef


res = {}
for c in COLS:
    cen, sc = CS[c]
    v = t24[c].to_numpy(np.float64)
    z = np.zeros(len(y), dtype=np.float64)
    f = np.isfinite(v)
    z[f] = (v[f] - cen) / sc
    KL = K * float((r_oot * w * z).mean())
    KG = K * float(((w * z) ** 2).mean())
    zr = resid(z)
    KLr = K * float((r_oot * w * zr).mean())
    KGr = K * float(((w * zr) ** 2).mean())
    res[c] = {"n_finite": int(f.sum()), "center": cen, "scale": sc,
              "K_L": KL, "K_G": KG,
              "K_L_residualised_vs_5_shipped_axes": KLr, "K_G_residualised": KGr,
              "retained_after_residualisation_pct":
                (100.0 * abs(KLr) / abs(KL)) if KL else None,
              "break_even_needed_abs_K_L": KG * U / 2.0,
              "passes_break_even": abs(KL) > KG * U / 2.0,
              "passes_break_even_residualised": abs(KLr) > KGr * U / 2.0,
              "u_star": KL / KG, "ceiling_pts": KL * KL / KG,
              "u_star_residualised": KLr / KGr if KGr else None,
              "ceiling_pts_residualised": KLr * KLr / KGr if KGr else None}

pc = res["asof_pitcher_middle_rate"]
EXP = -251.2                     # 배포 기반 LB 두 점에서 역산 (u=-0.008, gain +2.4731)
res["🔴_POSITIVE_CONTROL"] = {
 "column": "asof_pitcher_middle_rate",
 "expected_K_L_from_deployed_LB": EXP,
 "measured_K_L_on_OOT": pc["K_L"],
 "sign_matches": (pc["K_L"] < 0) == (EXP < 0),
 "ratio": pc["K_L"] / EXP,
 "abs_pct_error": 100.0 * abs(pc["K_L"] / EXP - 1.0),
 "second_control_asof_pitcher_success_rate": {
   "measured_K_L_on_OOT": res["asof_pitcher_success_rate"]["K_L"],
   "expected_sign": "+ (라벨 상관 +0.0843, 출하 u=+0.006544)",
   "sign_matches": res["asof_pitcher_success_rate"]["K_L"] > 0,
   "blend_json_back_solved_on_base_BG": 165.65,
   "ratio": res["asof_pitcher_success_rate"]["K_L"] / 165.65},
 "🔴": ("기반이 배포와 다르므로 크기의 일치는 기대하지 않는다.  ★부호★ 와 자릿수가 "
        "판정 기준이다.  둘 다 맞으면 이 면을 순위 근거로 쓴다(걸음 크기로는 쓰지 않는다)."),
}
out = {"K": K, "u_reference": U, "surface": "out-of-time 정직면 (<=2023 학습 팔 -> 2024 행)",
       "base_source": "exp111/r_oot.npy + p_oot.npy (exp_111 s2_oot.py)",
       "base_caveat": ("이 기반은 사후 축(3·4·5·6)을 하나도 안 실은 블렌드다.  배포 기반과 "
                       "다르므로 K*L 의 ★크기★ 를 걸음으로 옮기지 마라 — 순위에만 쓴다."),
       "oot_brier": brier_oot, "n": int(len(y)), "cols": res}
json.dump(out, open(f"{WORK}/linear_KL_oot.json", "w"), ensure_ascii=False, indent=1)

print(f"\n양성대조 asof_pitcher_middle_rate: 측정 K*L={pc['K_L']:.1f}  기대(배포 LB)={EXP:.1f}"
      f"  부호일치={res['🔴_POSITIVE_CONTROL']['sign_matches']}"
      f"  비율={pc['K_L'] / EXP:.3f}")
sc2 = res["🔴_POSITIVE_CONTROL"]["second_control_asof_pitcher_success_rate"]
print(f"2차대조 asof_pitcher_success_rate: 측정 K*L={sc2['measured_K_L_on_OOT']:.1f}"
      f"  부호일치={sc2['sign_matches']}  vs BG역산 165.65 비율={sc2['ratio']:.3f}")
hdr = (f"{'column':<38}{'K*L':>10}{'K*L_res':>10}{'유지%':>8}{'K*G':>9}"
       f"{'u*_res':>10}{'천장_res':>10}{'분기':>7}")
print(f"\n=== OOT 정직면 (표본 밖) ===\n{hdr}")
for c in sorted(COLS, key=lambda x: -abs(res[x]["ceiling_pts_residualised"] or 0)):
    d = res[c]
    print(f"{c:<38}{d['K_L']:>10.1f}{d['K_L_residualised_vs_5_shipped_axes']:>10.1f}"
          f"{(d['retained_after_residualisation_pct'] or 0):>8.1f}{d['K_G']:>9.0f}"
          f"{d['u_star_residualised']:>10.5f}{d['ceiling_pts_residualised']:>10.2f}"
          f"{('통과' if d['passes_break_even_residualised'] else '미달'):>7}")
print("\nLINEAR_OOT_OK -> work/linear_KL_oot.json")

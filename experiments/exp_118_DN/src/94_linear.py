#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""미청구 원본 열들의 선형항 K*L 을 재학습 0 · 추론 0 으로 잰다.

    Brier(u) - Brier(0) = 2*mean((p0-y)*dp) + mean(dp^2),   dp = p0*(1-p0)*u*z
    gain = -K*(Brier(u)-Brier(0)) = 2*K*u*L - K*u^2*G
      K*L = K * mean( (y - p0) * p0*(1-p0) * z )
      K*G = K * mean( (p0*(1-p0) * z)^2 )

🔴 양성 대조가 이 측정의 유일한 근거다(S274).  `asof_pitcher_middle_rate` 는 출하 축 5 이고
   그 축의 LB 실측에서 역산한 K*L 이 약 -269 다 (u=-0.008, gain +2.4730710258, K*G~28,700).
   이 식이 그 값을 못 맞히면 ★그 프레임에서는 이 측정을 순위 근거로 쓰지 마라★.
🔴 프레임 성격: cov·band 는 train season==2024 행을 2025 로 덮은 것이고 배포 모델은
   <=2024 학습이라 ★표본 안(암기 섞임)★ 이다.  S210 이 표본 안 암기 +218점을 관측했다.
"""
import json
import sys

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
PKG, WORK, RUNS = f"{B}/pkg", f"{B}/work", f"{B}/runs"
FDIR = f"{H}/build_CA/frames"
TRAIN = f"{H}/data/train.csv"
K = 400309.19882517256
U = 0.008
FRAMES = ("cov", "band")

COLS = ["asof_batter_middle_rate", "asof_batter_success_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev5_game_middle_rate",
        "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_success_rate"]
CONTROLS = {"asof_pitcher_middle_rate": {"u": -0.008, "lb_gain": 2.4730710258,
                                         "note": "출하 축 5 (부모 DM 이 이 기반에서 실측)"},
            "asof_pitcher_success_rate": {"u": 0.006544, "lb_gain": None,
                                          "note": "출하 축 2 — LB 는 다른 기반(BG)에서 샀다"}}

sys.path.insert(0, f"{PKG}/PAR")
import inseason                                                   # noqa: E402
import splits as SPL                                              # noqa: E402

# ── train season==2024 center/scale (모든 열, 결측은 z=0) ─────────────
tr = pd.read_csv(TRAIN, usecols=["row_id", "season", "control_success"] + COLS,
                 encoding="utf-8-sig")
t24 = tr[tr["season"] == 2024]
CS = {}
for c in COLS:
    v = t24[c].to_numpy(np.float64)
    CS[c] = (float(np.nanmean(v)), float(np.nanstd(v)))
LAB = dict(zip(tr["row_id"].tolist(),
               tr["control_success"].to_numpy(np.float64).tolist()))
print("center/scale 동결 완료 (train season==2024 %d행)" % len(t24))

PBL = json.load(open(f"{PKG}/PAR/blend.json", encoding="utf-8"))
AX = PBL["axis_offsets"]
out = {"K": K, "u_reference": U,
       "frame_class": {
         "cov": "표본 안 — train season==2024 행 245,789개를 2025 로 덮은 대역 프레임. "
                "배포 모델이 <=2024 학습이라 암기가 섞인다(S210).",
         "band": "표본 안 — cov 와 같은 성질."},
       "positive_control_spec": {
         "asof_pitcher_middle_rate": (
           "출하 축 5.  u=-0.008 에서 LB gain +2.4730710258, 같은 빌드가 잰 K*G≈28,503(cov)."
           "  gain = 2*K*L*u - K*G*u^2 -> K*L = (gain + K*G*u^2)/(2*u) = -269.5."
           "  이 식이 그 값을 ±20% 안에서 맞혀야 순위 근거로 쓴다."),
         "asof_pitcher_success_rate": (
           "출하 축 2.  blend.json 이 기록한 L_i_back_solved=0.0004138189513845949 -> "
           "K*L = +165.65 (기반 BG, 다른 기반이므로 참고값).")},
       "frames": {}}

for fr in FRAMES:
    tdf = pd.read_csv(f"{FDIR}/{fr}/test.csv", encoding="utf-8-sig", low_memory=False)
    sub = pd.read_csv(f"{RUNS}/par__{fr}.csv", encoding="utf-8-sig",
                      float_precision="round_trip")
    ids = sub["row_id"].to_numpy()
    p0 = sub["control_success"].to_numpy(np.float64)
    ti = pd.Index(tdf["row_id"].to_numpy())
    pos = ti.get_indexer(pd.Index(ids))
    assert (pos >= 0).all(), "프레임에 없는 row_id"
    y = np.array([LAB.get(r, np.nan) for r in ids], dtype=np.float64)
    n_lab = int(np.isfinite(y).sum())
    m = np.isfinite(y)
    w = p0 * (1.0 - p0)

    # ── 출하 5축의 delta 벡터 (script.py 와 같은 산술) ────────────────
    ins = inseason.apply(tdf, inseason.load(f"{PKG}/PAR/model/ins/deploy_anchor.npz"))
    cur = np.clip(ins["ins_p_success"].to_numpy(np.float64), 0.02, 0.98)
    pri = np.clip(cur - ins["ins_p_success_dev"].to_numpy(np.float64), 0.02, 0.98)
    off = np.nan_to_num(np.log(cur / (1 - cur)) - np.log(pri / (1 - pri)))
    spT = SPL.load(f"{PKG}/PAR/model/deploy_splits.npz")
    spA = SPL.apply(tdf, spT)
    sd1, sd2 = float(AX[2]["sd1"]), float(AX[2]["sd2"])
    raw = {
      "beta_ins_off": off,
      "ctrl_asof_psr": tdf["asof_pitcher_success_rate"].to_numpy(np.float64),
      "sp_pit_split": (spA["sp_pit_bhand"].to_numpy(np.float64) / sd1
                       + spA["sp_pit_strk2"].to_numpy(np.float64) / sd2),
      "sp_bat_phand": spA["sp_bat_phand"].to_numpy(np.float64),
      "ctrl_asof_mid": tdf["asof_pitcher_middle_rate"].to_numpy(np.float64),
    }
    D = []
    for a in AX:
        r = raw[a["name"]][pos]
        d = np.zeros(len(ids), dtype=np.float64)
        f = np.isfinite(r)
        d[f] = (r[f] - float(a["center"])) / float(a["scale"])
        D.append(d)
    D = np.stack(D, axis=1)                      # (n, 5) 출하 축 방향

    # G-계량 <a,b>_G = mean(w^2 * a * b) 에서 그람-슈미트 잔차화
    def gram(A, b):
        Wt = (w[m] ** 2)
        M = (A[m] * Wt[:, None]).T @ A[m] / m.sum()
        v = (A[m] * Wt[:, None]).T @ b[m] / m.sum()
        coef, *_ = np.linalg.lstsq(M, v, rcond=None)
        return b - A @ coef

    res = {}
    KG_ref = None
    for c in COLS:
        cen, sc = CS[c]
        v = pd.to_numeric(tdf[c], errors="coerce").to_numpy(np.float64)[pos]
        z = np.zeros(len(ids), dtype=np.float64)
        f = np.isfinite(v)
        z[f] = (v[f] - cen) / sc
        KL = K * float(((y[m] - p0[m]) * w[m] * z[m]).mean())
        KG = K * float(((w[m] * z[m]) ** 2).mean())
        zr = gram(D, z)
        KLr = K * float(((y[m] - p0[m]) * w[m] * zr[m]).mean())
        KGr = K * float(((w[m] * zr[m]) ** 2).mean())
        if c == "asof_batter_middle_rate":
            KG_ref = KG
        res[c] = {
          "n_finite": int(f.sum()), "center": cen, "scale": sc,
          "K_L": KL, "K_G": KG,
          "K_L_residualised_vs_5_shipped_axes": KLr,
          "K_G_residualised": KGr,
          "retained_after_residualisation_pct":
            (100.0 * abs(KLr) / abs(KL)) if KL else None,
          "break_even_needed_abs_K_L": KG * U / 2.0,
          "passes_break_even": abs(KL) > KG * U / 2.0,
          "passes_break_even_residualised": abs(KLr) > KGr * U / 2.0,
          "u_star": (KL / KG) if KG else None,
          "ceiling_pts": (KL * KL / KG) if KG else None,
          "u_star_residualised": (KLr / KGr) if KGr else None,
          "ceiling_pts_residualised": (KLr * KLr / KGr) if KGr else None,
          "gain_at_u_0008_sign_of_K_L": (2 * K * (U if KL > 0 else -U) * (KL / K)
                                         - KG * U * U),
        }
    # 양성 대조 판정
    pc = res["asof_pitcher_middle_rate"]
    exp_KL = (CONTROLS["asof_pitcher_middle_rate"]["lb_gain"]
              + pc["K_G"] * 0.008 ** 2) / (2 * -0.008)
    res["🔴_POSITIVE_CONTROL"] = {
      "column": "asof_pitcher_middle_rate",
      "expected_K_L_from_LB": exp_KL,
      "measured_K_L": pc["K_L"],
      "ratio_measured_over_expected": pc["K_L"] / exp_KL,
      "abs_pct_error": 100.0 * abs(pc["K_L"] / exp_KL - 1.0),
      "PASS_within_20pct": abs(pc["K_L"] / exp_KL - 1.0) <= 0.20,
      "second_control_asof_pitcher_success_rate": {
        "measured_K_L": res["asof_pitcher_success_rate"]["K_L"],
        "blend_json_back_solved_K_L_on_base_BG": 165.65,
        "ratio": res["asof_pitcher_success_rate"]["K_L"] / 165.65},
      "🔴": ("양성 대조가 떨어지면 이 표의 순위를 근거로 쓰지 마라 — 값이 아니라 다른 "
             "것을 재고 있는 것이다(S274)."),
    }
    res["n_rows"] = int(len(ids))
    res["n_rows_with_label"] = n_lab
    out["frames"][fr] = res
    print(f"[{fr}] n={len(ids)} 라벨={n_lab}  "
          f"양성대조 K*L 측정={pc['K_L']:.1f} 기대={exp_KL:.1f} "
          f"오차={100 * abs(pc['K_L'] / exp_KL - 1):.1f}%")

json.dump(out, open(f"{WORK}/linear_KL.json", "w"), ensure_ascii=False, indent=1)
hdr = f"{'column':<38}{'K*L':>10}{'K*L_res':>10}{'유지%':>8}{'K*G':>10}{'u*':>10}{'천장pts':>10}{'손익분기':>8}"
for fr in FRAMES:
    print(f"\n=== {fr} (표본 안) ===\n{hdr}")
    r = out["frames"][fr]
    for c in COLS:
        d = r[c]
        print(f"{c:<38}{d['K_L']:>10.1f}{d['K_L_residualised_vs_5_shipped_axes']:>10.1f}"
              f"{(d['retained_after_residualisation_pct'] or 0):>8.1f}{d['K_G']:>10.0f}"
              f"{d['u_star']:>10.5f}{d['ceiling_pts']:>10.2f}"
              f"{('통과' if d['passes_break_even'] else '미달'):>8}")
print("\nLINEAR_OK -> work/linear_KL.json")

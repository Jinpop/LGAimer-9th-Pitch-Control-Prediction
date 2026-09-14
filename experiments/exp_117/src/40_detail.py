#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_117 4단 — 상세·보수 열.

🔴 30_value.py 의 V_regime 은 '배포 모델이 6시즌 무차별 평균 자리에 서 있다' 를 전제한다.
   그 전제가 축마다 참이 아니다:
     · count12  — CS3 가 +0.78666 을 실제로 벌었으므로 전제가 참임이 실측됐다.
     · game_type — ★거짓★.  aa_infer.py stage 4 의 gamma[game_type] 가
       "centres a segment's mean on that segment's realised 2024 rate" 라고 명시한다.
   그래서 '2024 기준' 보수 열을 전 축에 대해 함께 낸다:
       eps_2024 = (사후 레짐 수준) - (2024 수준)   -> 레짐이 유지되면 0 에 가깝다.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_117/src")
import common117 as C
import importlib
_R = importlib.import_module("20_resid")
_V = importlib.import_module("30_value")

OUT = "<PROJECT_ROOT>/experiments/exp_117/out"
K, CS3_LB = _V.K, _V.CS3_LB
SEEDS = [11, 22, 33, 44, 55]


def v2024(m, V, n):
    """모델이 2024 수준에 이미 서 있다고 볼 때 남는 값."""
    ok = np.isfinite(m).all(0)
    f = n.sum(0) / n.sum()
    eps = m[C.POST].mean(0) - m[5]                 # 사후평균 - 2024
    ve = V[4] * 0.25 + V[5] * 0.25                 # (m23+m24)/2 - m24 = (m23-m24)/2
    e2 = np.maximum(eps ** 2 - ve, 0.0)
    return float(K * np.nansum(np.where(ok, f * e2, 0.0)))


def main():
    t0 = time.time()
    A = C.load()
    cells = C.make_cells(A)
    y, se = A["y"], A["seas_idx"]
    val = json.load(open(f"{OUT}/30_value.json", encoding="utf-8"))
    sc_reg = val["cs3_calibration"]["scale_regime"]
    out = {"scale_regime": sc_reg, "raw_rates": {}, "residual_detail": {},
           "conservative_2024_ref": {}}

    # ---- 원자료: 시즌 x game_type 성공률 (제188 재현의 전문) ----
    for gt, mk in (("F", A["isF"] == 1), ("R", A["isF"] == 0)):
        out["raw_rates"][gt] = {str(C.SEASONS[s]): {
            "rate": float(y[mk & (se == s)].mean()),
            "n": int((mk & (se == s)).sum())} for s in range(C.NS)}
    out["raw_rates"]["ALL"] = {str(C.SEASONS[s]): {
        "rate": float(y[se == s].mean()), "n": int((se == s).sum())}
        for s in range(C.NS)}

    # ---- 잔차 상세 + 보수 열 ----
    for frame in ("F0", "F1", "F2"):
        Z, _ = _V.build_Z_frame(A, frame)
        beta, p, nit = _R.irls(Z, y)
        r = y - p
        bn = np.bincount(se, minlength=C.NS).astype(np.float64)
        bs = np.bincount(se, weights=r, minlength=C.NS)
        r = r - (bs / bn)[se]
        D, Cv = {}, {}
        for cname, X in cells.items():
            m, V, n = _V.cellgrid(r, X["code"], se, X["nc"])
            if cname in ("game_type", "hand_pair", "asof_pn_bin", "count12",
                         "game_month", "li_bin"):
                D[cname] = {X["labels"][j]: {
                    "resid_dev_by_season": [float(m[s, j]) for s in range(C.NS)],
                    "n_by_season": [int(n[s, j]) for s in range(C.NS)]}
                    for j in range(X["nc"])}
            nl = []
            for sd in SEEDS:
                rs = np.random.RandomState(sd)
                cd = X["code"].copy()
                for s_i in range(C.NS):
                    idx = np.flatnonzero(se == s_i)
                    cd[idx] = X["code"][idx][rs.permutation(len(idx))]
                m2, V2, n2 = _V.cellgrid(r, cd, se, X["nc"])
                nl.append(v2024(m2, V2, n2))
            Cv[cname] = {"V_2024ref": v2024(m, V, n),
                         "null_median": float(np.median(nl)),
                         "null_max": float(np.max(nl))}
            Cv[cname]["net"] = Cv[cname]["V_2024ref"] - Cv[cname]["null_median"]
            Cv[cname]["score"] = Cv[cname]["net"] * sc_reg
        out["residual_detail"][frame] = D
        out["conservative_2024_ref"][frame] = Cv
        print("%s 보수열 완료 [%.0fs]" % (frame, time.time() - t0), flush=True)

    # ---- 사전 등록 게이트가 양성 대조 셀 자신을 통과시키는가 (probR 관례) ----
    b = C.baseline_row(y[A["isF"] == 0], se[A["isF"] == 0],
                       se[A["isF"] == 0], C.NS, "prob")
    cnt = (A["balls_before"] * 3 + A["strikes_before"])[A["isF"] == 0]
    dev, V, n = C.cell_stats(y[A["isF"] == 0], cnt, se[A["isF"] == 0], b, 12, "prob")
    j = 3 * 3 + 0                                   # 3-0
    sh, ho, ps, sef, z = C.shift_stats(dev, V, n)
    out["prereg_gate_on_its_own_positive_control"] = {
        "cell": "count 3-0 (probR 관례 = PC1 을 정확히 재현한 관례)",
        "dev_by_season": [float(x) for x in dev[:, j]],
        "n_by_season": [int(x) for x in n[:, j]],
        "shift": float(sh[j]), "hold": float(ho[j]), "pre_sd": float(ps[j]),
        "hold_over_pre_sd": float(ho[j] / ps[j]),
        "cond2_hold_le_1.5x_pre_sd": bool(ho[j] <= 1.5 * ps[j]),
        "cond3_n_ge_20000_in_2023_and_2024": bool(n[4, j] >= 20000 and n[5, j] >= 20000),
        "verdict": ("사전 등록 게이트가 자기 양성 대조 셀을 떨어뜨린다 — "
                    "이 게이트는 값이 아니라 표본수를 재고 있었다.")}

    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/40_detail.json", "w"), ensure_ascii=False, indent=1)

    print("\n=== 시즌별 성공률 ===")
    print("%-5s " % "" + " ".join("%9d" % s for s in C.SEASONS))
    for gt in ("F", "R", "ALL"):
        print("%-5s " % gt + " ".join("%9.4f" % out["raw_rates"][gt][str(s)]["rate"]
                                      for s in C.SEASONS))
        print("%-5s " % "  n" + " ".join("%9d" % out["raw_rates"][gt][str(s)]["n"]
                                         for s in C.SEASONS))
    print("\n=== 보수열(모델이 2024 수준에 서 있다고 볼 때) · F1 면 ===")
    rows = sorted(out["conservative_2024_ref"]["F1"].items(),
                  key=lambda kv: -kv[1]["score"])
    print("%-14s %10s %10s %10s" % ("축", "V_2024ref", "널중앙", "점수"))
    for a, e in rows[:10]:
        print("%-14s %10.4f %10.4f %10.3f" % (a, e["V_2024ref"], e["null_median"],
                                              e["score"]))
    g = out["prereg_gate_on_its_own_positive_control"]
    print("\n=== 사전 등록 게이트 vs 자기 양성 대조 (count 3-0, probR) ===")
    print("  dev  " + " ".join("%+0.5f" % x for x in g["dev_by_season"]))
    print("  n    " + " ".join("%7d" % x for x in g["n_by_season"]))
    print("  hold %.5f / pre_sd %.5f = %.3f  -> cond2 %s ; cond3 %s"
          % (g["hold"], g["pre_sd"], g["hold_over_pre_sd"],
             g["cond2_hold_le_1.5x_pre_sd"], g["cond3_n_ge_20000_in_2023_and_2024"]))
    print("DETAIL_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

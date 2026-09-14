#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_116 — (a) 새로 추가된 live 축의 음성 대조  (b) pair_pb 싼 확인."""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_116/src")
import common as C

OUT = "<PROJECT_ROOT>/experiments/exp_116/out"
K, WN, MC = 200., "sqmin", 50           # 하니스 H1
SEEDS = [11, 22, 33, 44, 55]


def pairs(se):
    P = [("<=%d|%d" % (Y, Y + 1), se <= Y, se == Y + 1)
         for Y in (2019, 2020, 2021, 2022, 2023)]
    P.append(("REF 2019-22|2023-24", se <= 2022, se >= 2023))
    return P


def rho(DA, cA, DB, cB, wn=WN, mc=MC):
    m = (cA >= mc) & (cB >= mc)
    if m.sum() < 3:
        return float("nan"), 0
    return C.wcorr(DA[m], DB[m], C.WEIGHTS[wn](cA[m], cB[m])), int(m.sum())


def main():
    t0 = time.time()
    A = C.load(); ax, li_thr, meta = C.make_axes(A)
    y = A["control_success"].astype(np.float64); se = A["season"]
    PR = pairs(se)
    out = {"harness": {"k": K, "weight": WN, "min_cn": MC}, "null": {}, "pair_pb": {}}

    # ---------- (a) 음성 대조: 창 안 셀코드 무작위 재배정 (셀 크기 정확 보존) ----
    for name in ("pit_strk1", "pit_ball1", "pit_ball2"):
        X = ax[name]; nl = {}
        for sd in SEEDS:
            rs = np.random.RandomState(sd); vals = []
            for tag, mA, mB in PR:
                bb = []
                for m in (mA, mB):
                    cd = X["code"][m].copy(); rs.shuffle(cd)
                    bb.append(C.build_axis(y[m], X["subj"][m], X["P"], cd, X["nc"], K))
                vals.append(rho(bb[0][0], bb[0][1], bb[1][0], bb[1][1])[0])
            fv = [v for v in vals[:-1] if np.isfinite(v)]
            nl[str(sd)] = {"rolling_median": float(np.median(fv)), "ref": vals[-1]}
        mm = [nl[str(s)]["rolling_median"] for s in SEEDS]
        out["null"][name] = {"per_seed": nl,
                             "median_of_medians": float(np.median(mm)),
                             "max_abs_median": float(max(abs(v) for v in mm)),
                             "max_abs_ref": float(max(abs(nl[str(s)]["ref"]) for s in SEEDS))}
        print("null %-10s  med-of-med %+0.4f  max|med| %.4f  max|ref| %.4f"
              % (name, out["null"][name]["median_of_medians"],
                 out["null"][name]["max_abs_median"],
                 out["null"][name]["max_abs_ref"]), flush=True)

    # ---------- (b) pair_pb : k 를 흔들어 수축 후 분산이 남는가 ----------------
    X = ax["pair_pb"]
    mA, mB = (se <= 2022), (se >= 2023)
    ref = ax["pit_bhand"]
    pb = {}
    for k in (1., 3., 5., 10., 25., 75., 200.):
        DA, cA = C.build_axis(y[mA], X["subj"][mA], X["P"], X["code"][mA], X["nc"], k)
        DB, cB = C.build_axis(y[mB], X["subj"][mB], X["P"], X["code"][mB], X["nc"], k)
        RA, rcA = C.build_axis(y[mA], ref["subj"][mA], ref["P"],
                               ref["code"][mA], ref["nc"], k)
        nz = (cA > 0)
        both = (cA > 0) & (cB > 0)
        # 배포 관례로 실제 배포되는 값 = 행별 조회이므로 '행 기준' sd 가 진짜 진폭
        rowvA = DA[X["subj"][mA], X["code"][mA]]
        rowvR = RA[ref["subj"][mA], ref["code"][mA]]
        e = {"cells_total": int(X["P"] * X["nc"]),
             "cells_nonempty_A": int(nz.sum()),
             "cells_both_windows": int(both.sum()),
             "mean_cn_nonempty_A": float(cA[nz].mean()),
             "median_cn_nonempty_A": float(np.median(cA[nz])),
             "shrink_retained_frac_median": float(np.median(cA[nz] / (cA[nz] + k))),
             "sd_D_cells_nonempty": float(DA[nz].std()),
             "sd_D_rowwise": float(rowvA.std()),
             "sd_D_rowwise_pit_bhand_same_k": float(rowvR.std()),
             "ratio_rowsd_vs_pit_bhand": float(rowvA.std() / max(rowvR.std(), 1e-30))}
        for mc in (1, 5, 20, 50):
            r, n = rho(DA, cA, DB, cB, mc=mc)
            e[f"rho_ref_window_min_cn_{mc}"] = r
            e[f"n_cells_min_cn_{mc}"] = n
        pb[str(k)] = e
        print("pair_pb k=%6.0f  nonempty %7d  mean cn %6.2f  retained %.4f  rowsd %.5f "
              "(bhand %.5f, %.2fx)  rho(mc1) %+0.4f n=%d  rho(mc5) %+0.4f n=%d"
              % (k, e["cells_nonempty_A"], e["mean_cn_nonempty_A"],
                 e["shrink_retained_frac_median"], e["sd_D_rowwise"],
                 e["sd_D_rowwise_pit_bhand_same_k"], e["ratio_rowsd_vs_pit_bhand"],
                 e["rho_ref_window_min_cn_1"], e["n_cells_min_cn_1"],
                 e["rho_ref_window_min_cn_5"], e["n_cells_min_cn_5"]), flush=True)
    out["pair_pb"] = pb
    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/40_null_pair.json", "w"), ensure_ascii=False, indent=1)
    print("NULLPAIR_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_116 1단 — (주체 x 상황) 스플릿 셀의 연도 전이 상관 스크린.

라벨은 쓰지만 학습·LB 슬롯은 0 이다.  출하 splits.py 의 build() 산식을 창별로 돌린다.

하니스 패널: 양성 대조 4개를 |dRho| <= 0.05 로 동시에 재현하는 변형만 쓴다
(사전 등록: out/00_prereg.json).  H1 이 주 하니스, 나머지는 강건성 띠.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_116/src")
import common as C

OUT = "<PROJECT_ROOT>/experiments/exp_116/out"
PUB = {"pit_bhand": 0.515, "bat_phand": 0.414, "pit_strk2": 0.284, "pit_ball3": 0.254}
# (이름, k, 가중, 모양, min_cn) — 전부 flat 이라 셀 수가 2 든 830 이든 계산된다
PANEL = [("H1", 200., "sqmin", "flat", 50),
         ("H2", 100., "sum",   "flat", 1),
         ("H3", 200., "sqmin", "flat", 100),
         ("H4", 100., "sqmin", "flat", 50),
         ("H5", 150., "sqmin", "flat", 50)]
NULL_SEEDS = [11, 22, 33, 44, 55]


def pairs(se):
    P = [(f"<=%d|%d" % (Y, Y + 1), se <= Y, se == Y + 1) for Y in (2019, 2020, 2021, 2022, 2023)]
    P.append(("REF 2019-22|2023-24", se <= 2022, se >= 2023))
    return P


def rho_from(DA, cA, DB, cB, wname, mc):
    m = (cA >= mc) & (cB >= mc)
    if m.sum() < 3:
        return float("nan"), 0
    return C.wcorr(DA[m], DB[m], C.WEIGHTS[wname](cA[m], cB[m])), int(m.sum())


def main():
    t0 = time.time()
    A = C.load()
    ax, li_thr, meta = C.make_axes(A)
    y = A["control_success"].astype(np.float64)
    se = A["season"]
    PR = pairs(se)
    ks = sorted({p[1] for p in PANEL})

    out = {"meta": meta, "li_threshold_frozen_on_train": li_thr,
           "panel": [{"tag": t, "k": k, "weight": w, "shape": s, "min_cn": m}
                     for t, k, w, s, m in PANEL],
           "year_pairs": [p[0] for p in PR], "axes": {}}

    for name, X in ax.items():
        e = {"kind": X["kind"], "n_cells": X["nc"], "n_subject": X["P"], "rho": {}}
        # 창별 build 를 k 마다 한 번씩만 (캐시)
        cache = {}
        for tag, mA, mB in PR:
            for side, m in (("A", mA), ("B", mB)):
                for k in ks:
                    cache[(tag, side, k)] = C.build_axis(
                        y[m], X["subj"][m], X["P"], X["code"][m], X["nc"], k)
        for ht, k, wn, sh, mc in PANEL:
            r = {}
            for tag, _, _ in PR:
                DA, cA = cache[(tag, "A", k)]
                DB, cB = cache[(tag, "B", k)]
                v, ncell = rho_from(DA, cA, DB, cB, wn, mc)
                r[tag] = {"rho": v, "n_cells_used": ncell}
            roll = [r[p[0]]["rho"] for p in PR[:-1]]
            roll = [v for v in roll if np.isfinite(v)]
            e["rho"][ht] = {"by_pair": r,
                            "rolling_median": float(np.median(roll)) if roll else float("nan"),
                            "rolling_min": float(min(roll)) if roll else float("nan"),
                            "rolling_max": float(max(roll)) if roll else float("nan"),
                            "ref": r[PR[-1][0]]["rho"]}
        # ---- 음성 대조: 창 안에서 셀 코드를 무작위 재배정 (셀 크기 정확 보존) ----
        nl = {}
        for sd in NULL_SEEDS:
            rs = np.random.RandomState(sd)
            k, wn, mc = PANEL[0][1], PANEL[0][2], PANEL[0][4]
            vals = []
            for tag, mA, mB in PR:
                bb = []
                for m in (mA, mB):
                    cd = X["code"][m].copy()
                    rs.shuffle(cd)
                    bb.append(C.build_axis(y[m], X["subj"][m], X["P"], cd, X["nc"], k))
                v, _ = rho_from(bb[0][0], bb[0][1], bb[1][0], bb[1][1], wn, mc)
                vals.append(v)
            fv = [v for v in vals[:-1] if np.isfinite(v)]
            nl[str(sd)] = {"rolling_median": float(np.median(fv)) if fv else float("nan"),
                           "ref": vals[-1]}
        e["null_shuffled_cell"] = nl
        nm = [nl[str(s)]["rolling_median"] for s in NULL_SEEDS]
        nm = [v for v in nm if np.isfinite(v)]
        e["null_summary"] = {"median_of_medians": float(np.median(nm)) if nm else float("nan"),
                             "max_abs": float(max(abs(v) for v in nm)) if nm else float("nan")}
        out["axes"][name] = e
        print("%-12s %-14s nc=%4d  H1 med %+0.4f  [%+0.4f,%+0.4f]  ref %+0.4f   null|max| %.4f"
              % (name, X["kind"], X["nc"], e["rho"]["H1"]["rolling_median"],
                 e["rho"]["H1"]["rolling_min"], e["rho"]["H1"]["rolling_max"],
                 e["rho"]["H1"]["ref"], e["null_summary"]["max_abs"]), flush=True)

    # ---- 양성 대조 재현 판정 ----
    rep = {}
    for ht, *_ in PANEL:
        d = {a: out["axes"][a]["rho"][ht]["ref"] for a in PUB}
        rep[ht] = {"ref_rho": d,
                   "max_abs_err_vs_published": max(abs(d[a] - PUB[a]) for a in PUB),
                   "rank_preserved": (d["pit_bhand"] > d["bat_phand"] >
                                      d["pit_strk2"] > d["pit_ball3"])}
    out["positive_control_reproduction"] = {"published": PUB, "by_harness": rep,
                                            "tolerance": 0.05}
    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/20_screen.json", "w"), ensure_ascii=False, indent=1)
    print("\n재현 판정 (REF 창):")
    for ht, v in rep.items():
        print("  %s maxerr %.4f  rank %s  %s" % (ht, v["max_abs_err_vs_published"],
                                                 v["rank_preserved"],
                                                 {a: round(x, 4) for a, x in v["ref_rho"].items()}))
    print("SCREEN_OK  %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

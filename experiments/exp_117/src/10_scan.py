#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_117 0단+1단 — 양성 대조 재현 · 리그 셀 레벨 이동 전수 스캔 · 널 바닥.

라벨은 쓰지만 학습 0 · LB 슬롯 0.
사전 등록: out/00_prereg.json (측정 전에 작성됨).
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_117/src")
import common117 as C

OUT = "<PROJECT_ROOT>/experiments/exp_117/out"
PREREG = json.load(open(f"{OUT}/00_prereg.json", encoding="utf-8"))

PANEL = [("P1", "logit", "all", "season"),
         ("P2", "logit", "R",   "season"),
         ("P3", "prob",  "all", "season"),
         ("P4", "prob",  "R",   "season"),
         ("P5", "logit", "all", "season_x_gametype")]
NULL_SEEDS = [11, 22, 33, 44, 55]
MIN_N = 20000          # 사전 등록 cond_3
HOLD_MULT = 1.5        # 사전 등록 cond_2


def panel_arrays(A, rows, basekind, scale):
    m = np.ones(len(A["y"]), bool) if rows == "all" else (A["isF"] == 0)
    y = A["y"][m]
    se = A["seas_idx"][m]
    if basekind == "season":
        grp, ng = se, C.NS
    else:
        grp, ng = se * 2 + A["isF"][m], C.NS * 2
    b = C.baseline_row(y, se, grp, ng, scale)
    return m, y, se, b


def main():
    t0 = time.time()
    A = C.load()
    ok_frozen, got = C.check_frozen(A)
    cells = C.make_cells(A)
    out = {"n_rows": int(len(A["y"])),
           "frozen_bins": C.FROZEN,
           "frozen_bins_recomputed_from_train": got,
           "frozen_bins_match": bool(ok_frozen),
           "panel": [{"tag": t, "scale": s, "rows": r, "baseline": b}
                     for t, s, r, b in PANEL],
           "min_n_per_post_season": MIN_N, "hold_mult": HOLD_MULT,
           "null_seeds": NULL_SEEDS}
    assert ok_frozen, f"동결 경계 불일치: {got}"

    # ---------------------------------------------------- 양성 대조 2 (게임타입 F)
    pc2 = {}
    for wtag, pre, post in [("W1", np.isin(A["season"], [2019, 2020, 2021, 2022]),
                             np.isin(A["season"], [2023, 2024])),
                            ("W2", A["season"] == 2022, A["season"] == 2023)]:
        e = {}
        for gt, mask in (("F", A["isF"] == 1), ("R", A["isF"] == 0)):
            e[f"{gt}_pre"] = float(A["y"][pre & mask].mean())
            e[f"{gt}_post"] = float(A["y"][post & mask].mean())
            e[f"{gt}_n_pre"] = int((pre & mask).sum())
            e[f"{gt}_n_post"] = int((post & mask).sum())
        tg = PREREG["positive_control_2_game_type_F"]["target"]
        e["err_F_pre"] = abs(e["F_pre"] - tg["F_pre"])
        e["err_F_post"] = abs(e["F_post"] - tg["F_post"])
        e["err_R_pre"] = abs(e["R_pre"] - tg["R_pre"])
        e["err_R_post"] = abs(e["R_post"] - tg["R_post"])
        e["pass_primary"] = bool(e["err_F_pre"] <= 0.010 and e["err_F_post"] <= 0.010)
        e["pass_R_descriptive"] = bool(e["err_R_pre"] <= 0.010 and e["err_R_post"] <= 0.010)
        pc2[wtag] = e
    pc2["PASS"] = bool(any(pc2[w]["pass_primary"] for w in ("W1", "W2")))
    out["positive_control_2_game_type_F"] = pc2
    print("PC2 게임타입F:")
    for w in ("W1", "W2"):
        e = pc2[w]
        print("  %s  F %.4f -> %.4f (오차 %.4f/%.4f)  R %.4f -> %.4f (%.4f/%.4f)  primary=%s"
              % (w, e["F_pre"], e["F_post"], e["err_F_pre"], e["err_F_post"],
                 e["R_pre"], e["R_post"], e["err_R_pre"], e["err_R_post"],
                 e["pass_primary"]))
    print("  PC2 PASS =", pc2["PASS"], flush=True)

    # -------------------------------------------------------------- 패널 스캔
    res = {}
    devstore = {}
    for ptag, scale, rows, basekind in PANEL:
        m, y, se, b = panel_arrays(A, rows, basekind, scale)
        res[ptag] = {}
        for cname, X in cells.items():
            code = X["code"][m]
            dev, V, n = C.cell_stats(y, code, se, b, X["nc"], scale)
            sh, ho, ps, sef, z = C.shift_stats(dev, V, n)
            devstore[(ptag, cname)] = (dev, n)
            # ---- 널: 시즌 안 셀코드 무작위 재배정 (셀 크기 정확 보존) ----
            nullz = []
            for sd in NULL_SEEDS:
                rs = np.random.RandomState(sd)
                cd = code.copy()
                for s_i in range(C.NS):
                    idx = np.flatnonzero(se == s_i)
                    cd[idx] = code[idx][rs.permutation(len(idx))]
                d2, V2, n2 = C.cell_stats(y, cd, se, b, X["nc"], scale)
                _, _, _, _, z2 = C.shift_stats(d2, V2, n2)
                big2 = (n2[4] >= MIN_N) & (n2[5] >= MIN_N)
                v = np.abs(z2[big2])
                nullz.append(v[np.isfinite(v)])
            nz = np.concatenate(nullz) if nullz else np.array([])
            p95 = float(np.percentile(nz, 95)) if len(nz) else float("nan")
            nmax = float(nz.max()) if len(nz) else float("nan")
            big = (n[4] >= MIN_N) & (n[5] >= MIN_N)
            per = []
            for j in range(X["nc"]):
                if not np.isfinite(sh[j]):
                    continue
                c1 = bool(np.isfinite(p95) and abs(z[j]) >= p95)
                c2 = bool(ho[j] <= ps[j] * HOLD_MULT)
                c3 = bool(big[j])
                per.append({"cell": X["labels"][j],
                            "n_by_season": [int(v) for v in n[:, j]],
                            "dev_by_season": [float(v) for v in dev[:, j]],
                            "shift": float(sh[j]), "hold": float(ho[j]),
                            "pre_sd": float(ps[j]), "se": float(sef[j]),
                            "z": float(z[j]),
                            "cond1_z_ge_null_p95": c1, "cond2_hold": c2,
                            "cond3_size": c3, "stage1_pass": bool(c1 and c2 and c3)})
            res[ptag][cname] = {"family": X["family"], "n_cells": X["nc"],
                                "null_p95_absz": p95, "null_max_absz": nmax,
                                "null_n_samples": int(len(nz)),
                                "cells": per,
                                "n_pass": int(sum(p["stage1_pass"] for p in per))}
        done = sum(v["n_pass"] for v in res[ptag].values())
        print("패널 %s (%s/%s/%s) 1단 통과 셀 %d  [%.0fs]"
              % (ptag, scale, rows, basekind, done, time.time() - t0), flush=True)
    out["stage1"] = res

    # ---------------------------------------------------- 양성 대조 1 (카운트 3-0)
    tg = PREREG["positive_control_1_count_3_0"]["target_by_season"]
    tv = np.array([tg[str(s)] for s in C.SEASONS])
    pc1 = {}
    for conv, ptag in (("logitR", "P2"), ("probR", "P4")):
        dev, n = devstore[(ptag, "count12")]
        j = C.make_cells(A)["count12"]["labels"].index("3-0")
        v = dev[:, j]
        err = float(np.nanmax(np.abs(v - tv)))
        pc1[conv] = {"panel": ptag, "measured": [float(x) for x in v],
                     "target": [float(x) for x in tv],
                     "n_by_season": [int(x) for x in n[:, j]],
                     "max_abs_err": err,
                     "pass_primary": bool(err <= 0.0025),
                     "pass_secondary_descriptive": bool(err <= 0.0050)}
    pc1["PASS"] = bool(any(pc1[c]["pass_primary"] for c in ("logitR", "probR")))
    out["positive_control_1_count_3_0"] = pc1
    print("\nPC1 카운트 3-0:")
    for c in ("logitR", "probR"):
        e = pc1[c]
        print("  %-7s %s  maxerr %.5f  primary=%s"
              % (c, " ".join("%+.5f" % x for x in e["measured"]),
                 e["max_abs_err"], e["pass_primary"]))
    print("  target  " + " ".join("%+.5f" % x for x in tv))
    print("  PC1 PASS =", pc1["PASS"], flush=True)

    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/10_scan.json", "w"), ensure_ascii=False, indent=1)

    # ---------------------------------------------------------------- 요약 출력
    print("\n=== 1단 통과 셀 (패널별) ===")
    agg = {}
    for ptag, *_ in PANEL:
        for cname, e in out["stage1"][ptag].items():
            for p in e["cells"]:
                if p["stage1_pass"]:
                    agg.setdefault((cname, p["cell"]), []).append(ptag)
    if not agg:
        print("  없다 (전 패널 0)")
    for (cn, cl), ps in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        print("  %-14s %-12s  패널 %d/5  %s" % (cn, cl, len(ps), ",".join(ps)))
    print("SCAN_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

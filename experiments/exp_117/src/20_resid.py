#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_117 2단 — 배포 축에 대한 직교 잔차에서 레벨 이동 통계를 다시 낸다.

🔴 exp_116 에서 최고 후보가 잔차화로 -80% 무너졌다. 이 단계가 진짜 관문이다.

통제 기저 Z (사전 등록 out/00_prereg.json §stage2_threshold):
  season 더미 · count12 더미 · CS3 12칸 기울기 방향 x (season-2021.5) ·
  3-0 지시자 · 3-0 x post2023 · 배포 스플릿 4축(splits.py 산식, k=75, train 전체) · isF
로지스틱 적합 -> 행별 잔차 r = y - p_hat -> 셀/시즌 평균 잔차로 shift 재계산.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_117/src")
import common117 as C

OUT = "<PROJECT_ROOT>/experiments/exp_117/out"
K_SPLIT = 75.0          # 출하 splits.py AXES 의 k (teammate_v19/extracted/pack/sp/splits.py:26-29)
NULL_SEEDS = [11, 22, 33, 44, 55]
MIN_N, HOLD_MULT = 20000, 1.5


# ------------------------------------------------------------- 배포 스플릿 4축
def split_axis_rowvals(y, subj_raw, code, nc, k=K_SPLIT):
    """출하 splits.build() 산식 그대로. train 전체를 히스토리로 쓴다(통제용)."""
    ids, idx = np.unique(subj_raw, return_inverse=True)
    P = len(ids)
    league = float(y.mean())
    n = np.bincount(idx, minlength=P).astype(np.float64)
    s = np.bincount(idx, weights=y, minlength=P)
    p_all = (s + k * league) / (n + k)
    ln = np.bincount(code, minlength=nc).astype(np.float64)
    ls = np.bincount(code, weights=y, minlength=nc)
    lrate = (ls + k * league) / (ln + k)
    lgc = C.lg(lrate) - C.lg(league)
    flat = idx * nc + code
    cn = np.bincount(flat, minlength=P * nc).astype(np.float64).reshape(P, nc)
    cs = np.bincount(flat, weights=y, minlength=P * nc).reshape(P, nc)
    rate = (cs + k * p_all[:, None]) / (cn + k)
    D = (C.lg(rate) - C.lg(p_all)[:, None]) - lgc[None, :]
    return D[idx, code]


def count_trend_dir(A):
    """CS3 12칸 OLS 시즌 기울기의 ★부호★ (12개 자유 파라미터 아님)."""
    cnt = A["balls_before"] * 3 + A["strikes_before"]
    y, se = A["y"], A["seas_idx"]
    b = C.baseline_row(y, se, se, C.NS, "logit")
    dev, _, n = C.cell_stats(y, cnt, se, b, 12, "logit")
    x = np.arange(C.NS, dtype=np.float64)
    x = x - x.mean()
    slope = (dev * x[:, None]).sum(0) / (x ** 2).sum()
    return cnt, np.sign(slope), slope


def build_Z(A, variant):
    """통제 기저. 반환 (Z, 열이름)."""
    y, se = A["y"], A["seas_idx"]
    cols, names = [], []

    def push(v, nm):
        cols.append(np.asarray(v, np.float64)); names.append(nm)

    for s in range(1, C.NS):                       # season 더미 (첫 칸 드롭)
        push(se == s, f"season_{C.SEASONS[s]}")
    cnt, sgn, slope = count_trend_dir(A)
    for j in range(1, 12):                         # count12 더미 (첫 칸 드롭)
        push(cnt == j, f"count_{j//3}-{j%3}")
    push(sgn[cnt] * (A["season"] - 2021.5), "CS3_trend_dir_x_season")
    is30 = (A["balls_before"] == 3) & (A["strikes_before"] == 0)
    post = (A["season"] >= 2023)
    push(is30, "ind_3_0")
    push(is30 & post, "ind_3_0_x_post2023")
    push(A["isF"], "isF")
    if variant != "R3":                            # 배포 스플릿 4축
        push(split_axis_rowvals(y, A["pitcher_id"], A["batter_hand"] - 1, 2), "sp_pit_bhand")
        push(split_axis_rowvals(y, A["batter_id"], A["pitcher_hand"] - 1, 2), "sp_bat_phand")
        push(split_axis_rowvals(y, A["pitcher_id"],
                                (A["strikes_before"] >= 2).astype(np.int64), 2), "sp_pit_strk2")
        push(split_axis_rowvals(y, A["pitcher_id"],
                                (A["balls_before"] >= 3).astype(np.int64), 2), "sp_pit_ball3")
    if variant == "R4":                            # season x count12 (최강 카운트 통제)
        for s in range(1, C.NS):
            for j in range(1, 12):
                push((se == s) & (cnt == j), f"s{C.SEASONS[s]}_c{j//3}-{j%3}")
    if variant == "R5":                            # season x isF
        for s in range(1, C.NS):
            push((se == s) & (A["isF"] == 1), f"s{C.SEASONS[s]}_F")
    Z = np.column_stack([np.ones(len(y))] + cols)
    return Z, ["intercept"] + names, slope


def irls(Z, y, iters=30, tol=1e-11):
    beta = np.zeros(Z.shape[1])
    beta[0] = np.log(y.mean() / (1 - y.mean()))
    for it in range(iters):
        eta = Z @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.maximum(p * (1 - p), 1e-9)
        g = Z.T @ (y - p)
        H = Z.T @ (Z * w[:, None])
        H[np.diag_indices_from(H)] += 1e-8
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    eta = Z @ beta
    return beta, 1.0 / (1.0 + np.exp(-eta)), it + 1


def resid_stats(r, code, seas_idx, nc, base_row):
    flat = seas_idx * nc + code
    n = np.bincount(flat, minlength=C.NS * nc).astype(np.float64).reshape(C.NS, nc)
    s = np.bincount(flat, weights=r, minlength=C.NS * nc).reshape(C.NS, nc)
    s2 = np.bincount(flat, weights=r * r, minlength=C.NS * nc).reshape(C.NS, nc)
    b = np.bincount(flat, weights=base_row, minlength=C.NS * nc).reshape(C.NS, nc)
    nn = np.maximum(n, 1.0)
    m = s / nn
    V = np.maximum(s2 / nn - m ** 2, 1e-12) / nn
    dev = m - b / nn
    dev[n == 0] = np.nan
    V[n == 0] = np.nan
    return dev, V, n


def main():
    t0 = time.time()
    A = C.load()
    cells = C.make_cells(A)
    VARIANTS = [("R1", "all", "season"), ("R2", "R", "season"),
                ("R3", "all", "season"), ("R4", "all", "season"),
                ("R5", "all", "season_x_gametype")]
    out = {"k_split": K_SPLIT, "min_n": MIN_N, "hold_mult": HOLD_MULT,
           "null_seeds": NULL_SEEDS, "variants": {}, "fits": {}}

    for vtag, rows, basekind in VARIANTS:
        Z, znames, slope = build_Z(A, vtag)
        m = np.ones(len(A["y"]), bool) if rows == "all" else (A["isF"] == 0)
        y = A["y"][m]; se = A["seas_idx"][m]
        beta, p, nit = irls(Z[m], y)
        r = y - p
        if basekind == "season":
            grp, ng = se, C.NS
        else:
            grp, ng = se * 2 + A["isF"][m], C.NS * 2
        bn = np.bincount(grp, minlength=ng).astype(np.float64)
        bs = np.bincount(grp, weights=r, minlength=ng)
        base_row = (bs / np.maximum(bn, 1.0))[grp]
        out["fits"][vtag] = {"n_cols": int(Z.shape[1]), "n_rows": int(m.sum()),
                             "iters": int(nit), "rows": rows, "baseline": basekind,
                             "mean_resid": float(r.mean()),
                             "brier_resid": float((r ** 2).mean()),
                             "count_trend_slope": [float(x) for x in slope],
                             "count_trend_sign": [int(np.sign(x)) for x in slope],
                             "z_columns": znames if vtag in ("R1", "R3") else znames[:24]}
        # 원면(같은 행 집합, prob 척도) 대조 — 비율의 분모
        b_raw = C.baseline_row(y, se, grp, ng, "prob")
        V = {}
        for cname, X in cells.items():
            code = X["code"][m]
            rd, rV, n = resid_stats(r, code, se, X["nc"], base_row)
            rsh, rho_, rps, rse, rz = C.shift_stats(rd, rV, n)
            wd, wV, _ = C.cell_stats(y, code, se, b_raw, X["nc"], "prob")
            wsh, who, wps, wse, wz = C.shift_stats(wd, wV, n)
            nullz = []
            for sd in NULL_SEEDS:
                rs = np.random.RandomState(sd)
                cd = code.copy()
                for s_i in range(C.NS):
                    idx = np.flatnonzero(se == s_i)
                    cd[idx] = code[idx][rs.permutation(len(idx))]
                d2, V2, n2 = resid_stats(r, cd, se, X["nc"], base_row)
                _, _, _, _, z2 = C.shift_stats(d2, V2, n2)
                big2 = (n2[4] >= MIN_N) & (n2[5] >= MIN_N)
                v = np.abs(z2[big2]); nullz.append(v[np.isfinite(v)])
            nz = np.concatenate(nullz)
            p95 = float(np.percentile(nz, 95)) if len(nz) else float("nan")
            big = (n[4] >= MIN_N) & (n[5] >= MIN_N)
            per = []
            for j in range(X["nc"]):
                if not np.isfinite(rsh[j]):
                    continue
                c1 = bool(np.isfinite(p95) and abs(rz[j]) >= p95)
                c2 = bool(rho_[j] <= rps[j] * HOLD_MULT)
                c3 = bool(big[j])
                per.append({"cell": X["labels"][j],
                            "resid_shift": float(rsh[j]), "raw_shift": float(wsh[j]),
                            "ratio_resid_over_raw": float(rsh[j] / wsh[j])
                            if abs(wsh[j]) > 1e-12 else float("nan"),
                            "resid_z": float(rz[j]), "raw_z": float(wz[j]),
                            "resid_hold": float(rho_[j]), "resid_pre_sd": float(rps[j]),
                            "resid_dev_by_season": [float(x) for x in rd[:, j]],
                            "n_by_season": [int(x) for x in n[:, j]],
                            "cond1": c1, "cond2": c2, "cond3": c3,
                            "stage2_pass": bool(c1 and c2 and c3)})
            V[cname] = {"family": X["family"], "null_p95_absz": p95,
                        "null_n_samples": int(len(nz)), "cells": per,
                        "n_pass": int(sum(q["stage2_pass"] for q in per))}
        out["variants"][vtag] = V
        print("%s cols=%d iters=%d  2단 통과 %d  [%.0fs]"
              % (vtag, Z.shape[1], nit, sum(v["n_pass"] for v in V.values()),
                 time.time() - t0), flush=True)

    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/20_resid.json", "w"), ensure_ascii=False, indent=1)

    agg = {}
    for vtag, *_ in VARIANTS:
        for cname, e in out["variants"][vtag].items():
            for q in e["cells"]:
                if q["stage2_pass"]:
                    agg.setdefault((cname, q["cell"]), []).append(vtag)
    print("\n=== 2단(직교 잔차) 통과 셀 ===")
    if not agg:
        print("  없다 (전 변형 0)")
    for (cn, cl), vs in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        q = [x for x in out["variants"]["R1"][cn]["cells"] if x["cell"] == cl]
        rr = q[0]["ratio_resid_over_raw"] if q else float("nan")
        print("  %-14s %-12s  변형 %d/5  %s   R1 비율 %.3f"
              % (cn, cl, len(vs), ",".join(vs), rr))
    print("RESID_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

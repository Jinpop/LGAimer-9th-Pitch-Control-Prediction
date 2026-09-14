#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_116 1단b — `pit_strk1` 재현 + 배포축 직교 잔차 연도전이.

질문 하나: 배포된 `sp_pit_strk2`(2스트) 등과의 중복분을 뺀 뒤에도
           `pit_strk1`(1스트+) 의 연도 전이가 남는가.

방법: 창 W 안에서
  D_strk1[p,c]  <- 출하 splits.build() 산식 그대로
  pred_j[p,c]   <- 배포축 j 의 ★행값★ 을 같은 셀로 평균 (같은 창 표로 지음)
  잔차          <- cn 가중 WLS 의 잔차 (창마다 독립 적합 = 더 엄격한 시험)
  rho           <- 잔차_A vs 잔차_B (하니스 H1)
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_116/src")
import common as C

OUT = "<PROJECT_ROOT>/experiments/exp_116/out"
PANEL = [("H1", 200., "sqmin", 50), ("H2", 100., "sum", 1),
         ("H3", 200., "sqmin", 100), ("H4", 100., "sqmin", 50),
         ("H5", 150., "sqmin", 50)]
DEPLOYED = ["pit_bhand", "pit_strk2", "bat_phand", "pit_ball3"]
TARGETS = ["pit_strk1", "pit_ball1", "pit_ball2", "pit_strk2", "pit_ball3",
           "pit_bhand", "bat_phand"]


def pairs(se):
    P = [("<=%d|%d" % (Y, Y + 1), se <= Y, se == Y + 1)
         for Y in (2019, 2020, 2021, 2022, 2023)]
    P.append(("REF 2019-22|2023-24", se <= 2022, se >= 2023))
    return P


def cellmeans(rowv, subj, code, nc, P, cn):
    """행값을 (주체, 셀) 로 평균."""
    flat = subj * nc + code
    s = np.bincount(flat, weights=rowv, minlength=P * nc).reshape(P, nc)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cn > 0, s / np.maximum(cn, 1.0), 0.0)


def wls_resid(Y, X, w):
    """cn 가중 최소제곱 잔차.  X 는 (n, m), 절편 포함."""
    sw = np.sqrt(w)[:, None]
    Xw, Yw = X * sw, Y * sw[:, 0]
    b, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
    return Y - X @ b, b


def main():
    t0 = time.time()
    A = C.load()
    ax, li_thr, meta = C.make_axes(A)
    y = A["control_success"].astype(np.float64)
    se = A["season"]
    psr = A["asof_psr"].copy()
    psr_isnan = ~np.isfinite(psr)
    PR = pairs(se)
    out = {"li_threshold_frozen_on_train": li_thr,
           "psr_nan_frac": float(psr_isnan.mean()), "axes": {}}

    for tname in TARGETS:
        X = ax[tname]
        rec = {"raw": {}, "resid": {}, "overlap": {}}
        for ht, k, wn, mc in PANEL:
            raw, res, ov = {}, {}, {}
            for tag, mA, mB in PR:
                Ds, cs_, Rs = {}, {}, {}
                for side, m in (("A", mA), ("B", mB)):
                    lg_ = y[m]
                    D, cn = C.build_axis(lg_, X["subj"][m], X["P"],
                                         X["code"][m], X["nc"], k)
                    # ---- 예측자: 배포축 행값의 셀 평균 -----------------
                    cols = [np.ones((X["P"], X["nc"]))]
                    names = ["const"]
                    for j in DEPLOYED:
                        if j == tname:
                            continue
                        J = ax[j]
                        Dj, _ = C.build_axis(lg_, J["subj"][m], J["P"],
                                             J["code"][m], J["nc"], k)
                        rowv = Dj[J["subj"][m], J["code"][m]]
                        cols.append(cellmeans(rowv, X["subj"][m], X["code"][m],
                                              X["nc"], X["P"], cn))
                        names.append(j)
                    # ctrl_asof_psr 대리 + 인시즌 폼 대리
                    lgm = float(lg_.mean())
                    pv = np.where(psr_isnan[m], lgm, psr[m])
                    cols.append(cellmeans(pv, X["subj"][m], X["code"][m],
                                          X["nc"], X["P"], cn))
                    names.append("asof_psr")
                    npall = np.bincount(X["subj"][m], minlength=X["P"]).astype(float)
                    spall = np.bincount(X["subj"][m], weights=lg_, minlength=X["P"])
                    subj_rate = (spall + k * lgm) / (npall + k)
                    cols.append(cellmeans(pv - subj_rate[X["subj"][m]],
                                          X["subj"][m], X["code"][m],
                                          X["nc"], X["P"], cn))
                    names.append("ins_form_proxy")
                    Ds[side], cs_[side], Rs[side] = D, cn, np.stack(
                        [c.ravel() for c in cols], 1)
                m_ = (cs_["A"] >= mc) & (cs_["B"] >= mc)
                fl = m_.ravel()
                if fl.sum() < 10:
                    raw[tag] = float("nan"); res[tag] = float("nan"); continue
                wa = C.WEIGHTS[wn](cs_["A"].ravel()[fl], cs_["B"].ravel()[fl])
                ya, yb = Ds["A"].ravel()[fl], Ds["B"].ravel()[fl]
                raw[tag] = C.wcorr(ya, yb, wa)
                ra, ba = wls_resid(ya, Rs["A"][fl], wa)
                rb, bb = wls_resid(yb, Rs["B"][fl], wa)
                res[tag] = C.wcorr(ra, rb, wa)
                if ht == "H1":
                    v1 = C.wcorr(ya, Rs["A"][fl][:, names.index("pit_strk2")]
                                 if "pit_strk2" in names else ya, wa)
                    ov[tag] = {"R2_A": float(1 - (wa * ra ** 2).sum() /
                                             max((wa * (ya - (wa * ya).sum() / wa.sum()) ** 2).sum(), 1e-30)),
                               "rho_with_strk2_pred": v1,
                               "n_cells": int(fl.sum())}
            def summ(d):
                r = [d[p[0]] for p in PR[:-1] if np.isfinite(d.get(p[0], np.nan))]
                return {"by_pair": d,
                        "rolling_median": float(np.median(r)) if r else float("nan"),
                        "rolling_min": float(min(r)) if r else float("nan"),
                        "rolling_max": float(max(r)) if r else float("nan"),
                        "ref": d.get(PR[-1][0], float("nan"))}
            rec["raw"][ht] = summ(raw)
            rec["resid"][ht] = summ(res)
            if ht == "H1":
                rec["overlap"] = ov
        out["axes"][tname] = rec
        print("%-10s  raw H1 med %+0.4f ref %+0.4f | resid H1 med %+0.4f ref %+0.4f | R2(REF) %.3f"
              % (tname, rec["raw"]["H1"]["rolling_median"], rec["raw"]["H1"]["ref"],
                 rec["resid"]["H1"]["rolling_median"], rec["resid"]["H1"]["ref"],
                 rec["overlap"][PR[-1][0]]["R2_A"]), flush=True)

    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/30_strk1.json", "w"), ensure_ascii=False, indent=1)
    print("STRK1_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

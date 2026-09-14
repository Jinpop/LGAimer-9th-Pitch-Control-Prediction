#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_117 3단 준비 — 값(value) 측정.  1·2단이 '유의한가'만 묻고 '얼마인가'를 안 묻는다.

두 통계를 낸다. 둘 다 CS3 카운트 축(실측 LB +0.78666)을 눈금으로 쓴다.

  (A) 레짐 값  V = K * sum_c f_c * eps_c^2,  eps_c = mean(rdev[2023,2024]) - mean(rdev[전체])
      = 시즌 무차별 평균으로 학습한 모델이 2025 에서 이 셀에 대해 덜 미는 양.
      🔴 잡음 편향 보정: eps^2 -> eps^2 - var(eps) (음수는 0 으로 자른다).

  (B) 표본 밖 홀드아웃 (S185 · S190).  2024 를 감추고 <=2023 만으로 두 예측을 만든다:
        naive  = mean(rdev[2019..2023])     (시즌 무차별 = 모델이 하는 것)
        regime = rdev[2023]                 (레짐 외삽 = 우리가 팔려는 것)
      질량 가중 제곱오차 감소를 점수로 환산한다. 이게 진짜 표본 밖 증거다.

잔차면 둘:
  F0 = 배포 통제만 (season 더미 + isF + 스플릿 4축).  count12 의 값이 살아 있어야 눈금이 된다.
  F1 = F0 + CS3 카운트 성분 (12더미 · 기울기방향 · 3-0 레벨).  ★순증★ 을 재는 면.
"""
import json, sys, time
import numpy as np
sys.path.insert(0, "<PROJECT_ROOT>/experiments/exp_117/src")
import common117 as C
import importlib
_R = importlib.import_module("20_resid")          # 2단 코드 재사용 (재구현 금지)
split_axis_rowvals = _R.split_axis_rowvals
count_trend_dir = _R.count_trend_dir
irls = _R.irls

OUT = "<PROJECT_ROOT>/experiments/exp_117/out"
K = 400309.19882517256          # 점수 = 100000*(1-Brier/0.2498069) -> 1 Brier = K 점
CS3_LB = 0.78666                # CS3 카운트 축 실측 LB 이득 (팀 원장 정정본)
NULL_SEEDS = [11, 22, 33, 44, 55]
W_PRE = 4.0 / 6.0               # 시즌 무차별 평균 대비 사후창 초과분 계수 (=2/3)


def build_Z_frame(A, frame):
    """F0 = 배포 통제만.  F1 = F0 + CS3 카운트 성분."""
    y, se = A["y"], A["seas_idx"]
    cols, names = [], []

    def push(v, nm):
        cols.append(np.asarray(v, np.float64)); names.append(nm)

    for s in range(1, C.NS):
        push(se == s, f"season_{C.SEASONS[s]}")
    push(A["isF"], "isF")
    push(split_axis_rowvals(y, A["pitcher_id"], A["batter_hand"] - 1, 2), "sp_pit_bhand")
    push(split_axis_rowvals(y, A["batter_id"], A["pitcher_hand"] - 1, 2), "sp_bat_phand")
    push(split_axis_rowvals(y, A["pitcher_id"],
                            (A["strikes_before"] >= 2).astype(np.int64), 2), "sp_pit_strk2")
    push(split_axis_rowvals(y, A["pitcher_id"],
                            (A["balls_before"] >= 3).astype(np.int64), 2), "sp_pit_ball3")
    if frame in ("F1", "F2"):
        cnt, sgn, _ = count_trend_dir(A)
        for j in range(1, 12):
            push(cnt == j, f"count_{j//3}-{j%3}")
        push(sgn[cnt] * (A["season"] - 2021.5), "CS3_trend_dir_x_season")
        is30 = (A["balls_before"] == 3) & (A["strikes_before"] == 0)
        push(is30, "ind_3_0")
        push(is30 & (A["season"] >= 2023), "ind_3_0_x_post2023")
    if frame == "F2":                       # + season x isF : F 레짐 단절을 통째로 흡수
        for s_ in range(1, C.NS):
            push((se == s_) & (A["isF"] == 1), f"s{C.SEASONS[s_]}_F")
    return np.column_stack([np.ones(len(y))] + cols), ["intercept"] + names


def cellgrid(r, code, se, nc):
    flat = se * nc + code
    n = np.bincount(flat, minlength=C.NS * nc).astype(np.float64).reshape(C.NS, nc)
    s = np.bincount(flat, weights=r, minlength=C.NS * nc).reshape(C.NS, nc)
    s2 = np.bincount(flat, weights=r * r, minlength=C.NS * nc).reshape(C.NS, nc)
    nn = np.maximum(n, 1.0)
    m = s / nn
    V = np.maximum(s2 / nn - m * m, 1e-12) / nn      # var(셀-시즌 평균 잔차)
    m[n == 0] = np.nan; V[n == 0] = np.nan
    return m, V, n


def stats_from_grid(m, V, n):
    """(A) 레짐 값 · (B) 홀드아웃 이득. 둘 다 잡음 편향 보정."""
    ok = np.isfinite(m).all(0)
    N = n.sum()
    f = n.sum(0) / N                                  # 셀 질량
    # ---- (A) eps = 사후창 평균 - 전체 평균  (= W_PRE * shift) ----
    eps = m[C.POST].mean(0) - m.mean(0)
    ve = V[C.POST].sum(0) * (1 / 3.0 - 1 / 6.0) ** 2 + V[C.PRE].sum(0) * (1 / 6.0) ** 2
    e2 = np.maximum(eps ** 2 - ve, 0.0)
    VA = float(K * np.nansum(np.where(ok, f * e2, 0.0)))
    VA_raw = float(K * np.nansum(np.where(ok, f * eps ** 2, 0.0)))
    # ---- (B) 2024 홀드아웃: naive(<=2023 평균) vs regime(2023 단독) ----
    naive = m[:5].mean(0)
    regime = m[4]
    t24 = m[5]
    red = (t24 - naive) ** 2 - (t24 - regime) ** 2
    vb = (V[:5].sum(0) / 25.0) - V[4]                 # (b-a) 의 분산 성분 편향 보정항
    red_bc = red + vb
    VB = float(K * np.nansum(np.where(ok, f * red_bc, 0.0)))
    VB_raw = float(K * np.nansum(np.where(ok, f * red, 0.0)))
    return dict(V_regime=VA, V_regime_uncorrected=VA_raw,
                V_holdout=VB, V_holdout_uncorrected=VB_raw,
                n_cells_used=int(ok.sum()))


def main():
    t0 = time.time()
    A = C.load()
    cells = C.make_cells(A)
    y_all, se_all = A["y"], A["seas_idx"]
    out = {"K": K, "CS3_LB_anchor": CS3_LB, "w_pre": W_PRE,
           "note": ("V_regime · V_holdout 은 '셀 평균 잔차를 완전히 맞췄을 때'의 "
                    "Brier 감소를 점수로 환산한 양이다. 배포 실현분은 그보다 작다 — "
                    "그래서 CS3 카운트 축의 실측 LB +0.78666 로 눈금을 맞춘다."),
           "frames": {}}

    for frame in ("F0", "F1", "F2"):
        Z, znames = build_Z_frame(A, frame)
        beta, p, nit = irls(Z, y_all)
        r = y_all - p
        # season 더미가 있으므로 시즌 평균 잔차는 0 에 가깝다 — 기준선을 명시적으로 뺀다
        bn = np.bincount(se_all, minlength=C.NS).astype(np.float64)
        bs = np.bincount(se_all, weights=r, minlength=C.NS)
        r = r - (bs / bn)[se_all]
        E = {"n_cols": int(Z.shape[1]), "iters": int(nit), "cols": znames,
             "brier_resid": float((r ** 2).mean()), "axes": {}}
        for cname, X in cells.items():
            m, V, n = cellgrid(r, X["code"], se_all, X["nc"])
            st = stats_from_grid(m, V, n)
            nl = {"V_regime": [], "V_holdout": []}
            for sd in NULL_SEEDS:
                rs = np.random.RandomState(sd)
                cd = X["code"].copy()
                for s_i in range(C.NS):
                    idx = np.flatnonzero(se_all == s_i)
                    cd[idx] = X["code"][idx][rs.permutation(len(idx))]
                m2, V2, n2 = cellgrid(r, cd, se_all, X["nc"])
                s2 = stats_from_grid(m2, V2, n2)
                nl["V_regime"].append(s2["V_regime"])
                nl["V_holdout"].append(s2["V_holdout"])
            st["null_V_regime"] = nl["V_regime"]
            st["null_V_holdout"] = nl["V_holdout"]
            st["null_V_regime_median"] = float(np.median(nl["V_regime"]))
            st["null_V_holdout_median"] = float(np.median(nl["V_holdout"]))
            st["null_V_regime_max"] = float(np.max(nl["V_regime"]))
            st["null_V_holdout_max"] = float(np.max(nl["V_holdout"]))
            st["family"] = X["family"]
            st["n_cells"] = X["nc"]
            E["axes"][cname] = st
        out["frames"][frame] = E
        anc = E["axes"]["count12"]
        print("%s cols=%d  count12 눈금: V_regime %.4f (널중앙 %.4f) · V_holdout %.4f "
              "(널중앙 %.4f)  [%.0fs]"
              % (frame, Z.shape[1], anc["V_regime"], anc["null_V_regime_median"],
                 anc["V_holdout"], anc["null_V_holdout_median"], time.time() - t0),
              flush=True)

    # -------------------------------------------------- CS3 눈금으로 환산
    a0 = out["frames"]["F0"]["axes"]["count12"]
    sc_reg = CS3_LB / max(a0["V_regime"] - a0["null_V_regime_median"], 1e-12)
    sc_hol = CS3_LB / max(a0["V_holdout"] - a0["null_V_holdout_median"], 1e-12)
    out["cs3_calibration"] = {
        "count12_F0_V_regime_net": a0["V_regime"] - a0["null_V_regime_median"],
        "count12_F0_V_holdout_net": a0["V_holdout"] - a0["null_V_holdout_median"],
        "scale_regime": sc_reg, "scale_holdout": sc_hol,
        "meaning": ("CS3 는 이 값 전부를 실현하지 않았다(12칸 방향 + 1칸 레벨만). "
                    "따라서 환산값은 '같은 실현률로 배포했을 때'의 기대 LB 이득이다.")}

    # 축별 시즌 상세 (상위 축)
    out["per_season_detail"] = {}
    for cname in ("game_type", "hand_pair", "asof_pn_bin", "count12", "game_month"):
        X = cells[cname]
        det = {}
        for frame in ("F0", "F1", "F2"):
            det[frame] = out["frames"][frame]["axes"][cname]
        out["per_season_detail"][cname] = det

    rows = []
    for cname, e in out["frames"]["F1"]["axes"].items():
        net_r = e["V_regime"] - e["null_V_regime_median"]
        net_h = e["V_holdout"] - e["null_V_holdout_median"]
        e2 = out["frames"]["F2"]["axes"][cname]
        n2r = e2["V_regime"] - e2["null_V_regime_median"]
        n2h = e2["V_holdout"] - e2["null_V_holdout_median"]
        rows.append((cname, e["family"], net_r, net_r * sc_reg, net_h, net_h * sc_hol,
                     e["null_V_regime_max"] * sc_reg, e["null_V_holdout_max"] * sc_hol,
                     n2r * sc_reg, n2h * sc_hol))
    rows.sort(key=lambda x: -x[3])
    out["ranking_F1_incremental"] = [
        {"axis": a, "family": fm, "V_regime_net": vr, "score_regime": sr,
         "V_holdout_net": vh, "score_holdout": sh,
         "null_max_score_regime": nr, "null_max_score_holdout": nh,
         "score_regime_F2": s2r, "score_holdout_F2": s2h}
        for a, fm, vr, sr, vh, sh, nr, nh, s2r, s2h in rows]

    print("\n=== F1(=CS3 이미 통제) 면에서의 순증 · CS3 눈금 환산 점수 ===")
    print("%-14s %-16s %10s %9s | %9s | %8s %8s | %9s %9s"
          % ("축", "계열", "V_reg", "점수_reg", "점수_hold",
             "널max_r", "널max_h", "F2점수_r", "F2점수_h"))
    for a, fm, vr, sr, vh, sh, nr, nh, s2r, s2h in rows:
        print("%-14s %-16s %10.4f %9.3f | %9.3f | %8.3f %8.3f | %9.3f %9.3f"
              % (a, fm, vr, sr, sh, nr, nh, s2r, s2h))

    out["elapsed_sec"] = time.time() - t0
    json.dump(out, open(f"{OUT}/30_value.json", "w"), ensure_ascii=False, indent=1)
    print("VALUE_OK %.1fs" % out["elapsed_sec"])


if __name__ == "__main__":
    main()

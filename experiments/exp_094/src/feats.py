"""제9계열 피처 생성 — 실제 컬럼명 확정판.

생성 계열
  ins_*   당시즌 분해 (제8계열 재구현, 기준선 arm 에도 포함)
  frm_*   F1  다중 시간척도 폼 편차 (prev1/3/5 x {success, middle} 를 커리어·시즌 대비)
  bwd_*   F1b ★ 창 차분으로 최근 5경기를 3구간 시계열로 복원 (b1 / b23 / b45)
  spl_*   F2  2025 시즌 내부 2시점 분해
  qom_*   F3  두 분할의 교차 (outside/inplay 복원 포함)
  tdr_*   F4  팀 드리프트 외삽
  eb_*    F5  투수별 적응 수축

규율: apply 단계는 그 행의 컬럼 + train 유래 동결 룩업만. groupby/정렬/평균 없음.
"""
import numpy as np
import pandas as pd

EPS = 1e-9
SEASON = "season"
PID, BID = "pitcher_id", "batter_id"
PTEAM, BTEAM = "pitcher_team_id", "batter_team_id"
P_N = "asof_pitcher_n"
P_RATES = ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
           "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
           "asof_pitcher_strike_rate"]
P_SHORT = ["success", "reverse", "middle", "ball", "strike"]
B_N = "asof_batter_n"
B_RATES = ["asof_batter_success_rate", "asof_batter_middle_rate"]
B_SHORT = ["success", "middle"]
PREV = {(w, k): f"asof_pitcher_prev{w}_game_{k}_rate"
        for w in (1, 3, 5) for k in ("success", "middle")}
TGT = "control_success"

# 경기당 투구수 근사 (분모 컬럼이 제공되지 않음). 등판 1회 평균.
NG = {1: 35.0, 3: 105.0, 5: 175.0}


def logit(p, lo=1e-4, hi=1 - 1e-4):
    p = np.clip(np.asarray(p, dtype=np.float64), lo, hi)
    return np.log(p / (1.0 - p))


def shrink(rate, n, prior, k):
    n = np.asarray(n, dtype=np.float64)
    return (n * np.asarray(rate, np.float64) + k * np.asarray(prior, np.float64)) \
        / np.maximum(n + k, EPS)


# ----------------------------------------------------------------- 앵커
def build_ins_anchor(df, idcol, ncol, ratecols, seasons):
    """table[(id, S)] = 시즌 S 시작 시점의 커리어 누적 (n, cnt...)

    S 미만 시즌 중 가장 최근 것을 이월한다. 각 (id,S) 는 season<S 데이터만 본다.
    """
    idx = df.groupby([idcol, SEASON], sort=False)[ncol].idxmax()
    last = df.loc[idx, [idcol, SEASON, ncol] + ratecols].copy()
    last[ncol] = last[ncol].astype(np.float64) + 1.0            # 그 투구 1개 포함
    n = last[ncol].to_numpy(np.float64)
    cnt = {rc: np.round(n * last[rc].to_numpy(np.float64)) for rc in ratecols}
    rec = {}
    for i, (idv, sv) in enumerate(zip(last[idcol].to_numpy(), last[SEASON].to_numpy())):
        rec.setdefault(idv, {})[int(sv)] = tuple([n[i]] + [cnt[rc][i] for rc in ratecols])
    smin, smax = min(seasons), max(seasons) + 1
    tbl = {}
    for idv, per in rec.items():
        cur = None
        for S in range(smin, smax + 1):
            if cur is not None:
                tbl[(idv, S)] = cur
            if S in per:
                cur = per[S]
    return tbl


def apply_ins(df, tbl, ncol, ratecols, shorts, prefix, K=50.0):
    n_now = df[ncol].to_numpy(np.float64)
    key = pd.MultiIndex.from_arrays([df[PID if prefix == "ins_p" else BID].to_numpy(),
                                     df[SEASON].to_numpy()])
    got = pd.Series(list(key)).map(tbl)
    A = np.zeros((len(df), 1 + len(ratecols)), np.float64)
    m = got.notna().to_numpy()
    if m.any():
        A[m] = np.array([list(v) for v in got[m].to_numpy()], np.float64)
    ins_n = np.maximum(n_now - A[:, 0], 0.0)
    out = {f"{prefix}_n": ins_n}
    for j, (rc, sh) in enumerate(zip(ratecols, shorts)):
        car = df[rc].to_numpy(np.float64)
        cnt_ins = np.maximum(n_now * np.nan_to_num(car) - A[:, 1 + j], 0.0)
        raw = np.where(ins_n > 0, cnt_ins / np.maximum(ins_n, EPS), np.nan_to_num(car))
        raw = np.clip(raw, 0.0, 1.0)
        s = shrink(raw, ins_n, np.nan_to_num(car, nan=0.5), K)
        out[f"{prefix}_{sh}"] = s
        out[f"{prefix}_{sh}_raw"] = raw
        out[f"{prefix}_{sh}_dev"] = s - np.nan_to_num(car, nan=0.5)
    return pd.DataFrame(out, index=df.index)


def build_team_table(hist):
    """(팀, 시즌) 성공률 -> 수축 선형회귀 계수. hist 는 season < Y 로 잘라서 넣는다."""
    out = {}
    lg = hist.groupby(SEASON)[TGT].mean()
    if len(lg) < 2:
        return out
    x0 = lg.index.to_numpy(np.float64)
    lg_slope = float(np.polyfit(x0, logit(lg.to_numpy()), 1)[0])
    for side, tcol in (("pit", PTEAM), ("bat", BTEAM)):
        ts = hist.groupby([tcol, SEASON])[TGT].agg(["mean", "size"]).reset_index()
        rows = {}
        for t, sub in ts.groupby(tcol):
            x = sub[SEASON].to_numpy(np.float64)
            y = logit(sub["mean"].to_numpy())
            w = sub["size"].to_numpy(np.float64)
            xm = float(np.average(x, weights=w))
            if len(x) < 3:
                rows[t] = (float(np.average(y, weights=w)), lg_slope, xm)
                continue
            X = np.vstack([np.ones_like(x), x - xm]).T
            W = w / w.sum()
            XtW = X.T * W
            beta = np.linalg.solve(XtW @ X + 1e-9 * np.eye(2), XtW @ y)
            slope = 0.5 * float(beta[1]) + 0.5 * lg_slope        # 리그 기울기로 수축
            rows[t] = (float(beta[0]), slope, xm)
        out[side] = rows
    out["lg_slope"] = lg_slope
    out["lg_mean"] = float(logit(hist[TGT].mean()))
    return out


def build_eb_table(hist):
    ps = hist.groupby([PID, SEASON])[TGT].agg(["mean", "size"]).reset_index()
    ps = ps[ps["size"] >= 200]
    ebk = {}
    for pidv, sub in ps.groupby(PID):
        if len(sub) < 3:
            continue
        m = sub["mean"].to_numpy(np.float64)
        n = sub["size"].to_numpy(np.float64)
        v_w = float(np.mean(m * (1 - m) / n))
        v_b = max(float(np.var(m, ddof=1)) - v_w, 1e-6)
        ebk[pidv] = float(np.clip(v_w * float(np.mean(n)) / v_b, 10.0, 800.0))
    return {"k": ebk, "default": float(np.median(list(ebk.values()))) if ebk else 50.0}


# ----------------------------------------------------------------- 적용
def apply_new(df, ins_p, team_tbl, eb_tbl, K=50.0):
    n = len(df)
    F = {}
    car = np.nan_to_num(df["asof_pitcher_success_rate"].to_numpy(np.float64), nan=0.5)
    carm = np.nan_to_num(df["asof_pitcher_middle_rate"].to_numpy(np.float64), nan=0.13)
    car_l, carm_l = logit(car), logit(carm)
    ins_s = ins_p["ins_p_success"].to_numpy(np.float64)
    ins_m = ins_p["ins_p_middle"].to_numpy(np.float64)
    ins_n = ins_p["ins_p_n"].to_numpy(np.float64)
    ins_sl, ins_ml = logit(ins_s), logit(ins_m)

    # ---------- F1 frm ----------
    P = {}
    for (w, k), col in PREV.items():
        v = df[col].to_numpy(np.float64)
        base = car if k == "success" else carm
        v = np.where(np.isfinite(v), v, base)
        P[(w, k)] = v
        sh = shrink(v, NG[w], base, K)
        F[f"frm_p{w}_{k}_car"] = logit(sh) - (car_l if k == "success" else carm_l)
        F[f"frm_p{w}_{k}_ins"] = logit(sh) - (ins_sl if k == "success" else ins_ml)
    F["frm_have"] = np.isfinite(df[PREV[(1, "success")]].to_numpy(np.float64)).astype(np.float64)
    # 비대칭 분해 (부진은 지속, 호조는 회귀) — EBM 단조제약 우회용
    src = F["frm_p3_success_ins"]
    F["frm_neg"] = np.minimum(src, 0.0)
    F["frm_pos"] = np.maximum(src, 0.0)

    # ---------- F1b bwd  ★ 창 차분으로 3구간 시계열 복원 ----------
    for k in ("success", "middle"):
        p1, p3, p5 = P[(1, k)], P[(3, k)], P[(5, k)]
        b23 = np.clip((3 * p3 - p1) / 2.0, 0.0, 1.0)
        b45 = np.clip((5 * p5 - 3 * p3) / 2.0, 0.0, 1.0)
        base = car if k == "success" else carm
        s1 = shrink(p1, NG[1], base, K)
        s23 = shrink(b23, 2 * NG[1], base, K)
        s45 = shrink(b45, 2 * NG[1], base, K)
        l1, l23, l45 = logit(s1), logit(s23), logit(s45)
        F[f"bwd_{k}_b23"] = l23 - (car_l if k == "success" else carm_l)
        F[f"bwd_{k}_b45"] = l45 - (car_l if k == "success" else carm_l)
        F[f"bwd_{k}_slope"] = l1 - l45                      # 최근 5경기 기울기
        F[f"bwd_{k}_accel"] = (l1 - l23) - (l23 - l45)      # 가속도

    # ---------- F2 spl : 2025 시즌 내부 2시점 ----------
    p5s = P[(5, "success")]
    p5n = np.minimum(NG[5], ins_n)
    early_n = np.maximum(ins_n - p5n, 0.0)
    early_r = np.where(early_n > 0,
                       (ins_n * ins_s - p5n * p5s) / np.maximum(early_n, EPS), car)
    early_r = np.clip(early_r, 0.0, 1.0)
    rec_s = shrink(p5s, p5n, car, K)
    ear_s = shrink(early_r, early_n, car, K)
    F["spl_trend"] = logit(rec_s) - logit(ear_s)
    F["spl_early"] = logit(ear_s) - car_l
    F["spl_frac"] = p5n / np.maximum(ins_n, 1.0)
    F["spl_ok"] = (early_n >= 150).astype(np.float64)

    # ---------- F3 qom : 두 분할의 교차 ----------
    def qom(tag, suc, rev, mid, bal, stk):
        out_r = np.clip(1.0 - suc - rev - mid, 0.0, 1.0)      # ★ 미제공 4번째 성분
        inp = np.clip(1.0 - bal - stk, 0.0, 1.0)              # ★ 미제공 3번째 성분
        F[f"qom_{tag}_outside"] = out_r
        F[f"qom_{tag}_inplay"] = inp
        F[f"qom_{tag}_mid_g_str"] = mid / np.maximum(stk, EPS)
        F[f"qom_{tag}_out_g_bal"] = out_r / np.maximum(bal, EPS)
        F[f"qom_{tag}_rev_g_liv"] = rev / np.maximum(1.0 - inp, EPS)
        F[f"qom_{tag}_edge"] = stk - mid
        F[f"qom_{tag}_close"] = bal - out_r
    qom("car", car,
        np.nan_to_num(df["asof_pitcher_reverse_rate"].to_numpy(np.float64), nan=0.22),
        carm,
        np.nan_to_num(df["asof_pitcher_ball_rate"].to_numpy(np.float64), nan=0.37),
        np.nan_to_num(df["asof_pitcher_strike_rate"].to_numpy(np.float64), nan=0.44))
    qom("ins", ins_s, ins_p["ins_p_reverse"].to_numpy(np.float64), ins_m,
        ins_p["ins_p_ball"].to_numpy(np.float64), ins_p["ins_p_strike"].to_numpy(np.float64))
    for kk in ("outside", "inplay", "mid_g_str", "out_g_bal", "edge"):
        F[f"qom_dev_{kk}"] = F[f"qom_ins_{kk}"] - F[f"qom_car_{kk}"]

    # ---------- F4 tdr : 팀 드리프트 외삽 ----------
    ssn = df[SEASON].to_numpy(np.float64)
    for side, tcol in (("pit", PTEAM), ("bat", BTEAM)):
        rows = team_tbl.get(side)
        if not rows:
            F[f"tdr_{side}"] = np.zeros(n)
            continue
        a = pd.Series(df[tcol].to_numpy()).map({k: v[0] for k, v in rows.items()}).to_numpy(np.float64)
        b = pd.Series(df[tcol].to_numpy()).map({k: v[1] for k, v in rows.items()}).to_numpy(np.float64)
        xm = pd.Series(df[tcol].to_numpy()).map({k: v[2] for k, v in rows.items()}).to_numpy(np.float64)
        pred = a + b * (ssn - xm)
        F[f"tdr_{side}"] = np.where(np.isfinite(pred), pred - team_tbl.get("lg_mean", 0.0), 0.0)

    # ---------- F5 eb : 적응 수축 ----------
    kmap = eb_tbl.get("k", {})
    kdef = eb_tbl.get("default", 50.0)
    ki = pd.Series(df[PID].to_numpy()).map(kmap).fillna(kdef).to_numpy(np.float64)
    raw = ins_p["ins_p_success_raw"].to_numpy(np.float64)
    F["eb_ins_suc"] = logit(shrink(raw, ins_n, car, ki)) - car_l
    F["eb_k"] = ki
    F["eb_gain"] = F["eb_ins_suc"] - (ins_sl - car_l)

    out = pd.DataFrame(F, index=df.index)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)


FAMILIES = {
    "F1_frm": "frm_",
    "F1b_bwd": "bwd_",
    "F2_spl": "spl_",
    "F3_qom": "qom_",
    "F4_tdr": "tdr_",
    "F5_eb": "eb_",
}

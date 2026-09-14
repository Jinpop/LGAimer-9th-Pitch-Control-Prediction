#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_116 공용 — splits.py(출하본)의 build() 산식을 창(window)별로 돌리기 위한 얇은 래퍼.

🔴 산식은 출하 splits.py 와 한 글자도 다르지 않다:
      lrate = (ls + k*league) / (ln + k);  lgc = logit(lrate) - logit(league)
      rate  = (cs + k*p_all[:,None]) / (cn + k)
      D     = (logit(rate) - logit(p_all)[:,None]) - lgc[None,:]
   다른 것은 두 가지뿐이고 둘 다 창 비교를 위해 필요하다:
     (1) 주체 인덱스를 창마다 np.unique 하지 않고 ★전역 사전★ 으로 고정한다
         (두 창의 행을 같은 키에 놓기 위함. 창에 없는 주체는 n=0 -> p_all=league -> D=0
          이고 cn=0 이라 상관 계산에서 어차피 제외된다).
     (2) 셀 코드를 numpy 배열에서 만든다 (출하본은 DataFrame 에서 만든다).
"""
import numpy as np
import pandas as pd

EPS = 1e-6
TRAIN = "<PROJECT_ROOT>/data/train.csv"

USECOLS = ["season", "control_success", "pitcher_id", "batter_id",
           "pitcher_hand", "batter_hand", "strikes_before", "balls_before",
           "outs_before", "num_runners_on", "runner_on_1b", "runner_on_2b",
           "runner_on_3b", "li", "score_diff_pitcher_team", "top_bottom",
           "inning", "pitcher_team_id", "batter_team_id", "game_type",
           "asof_pitcher_success_rate"]


def lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def load():
    d = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=USECOLS)
    A = {c: d[c].to_numpy(np.int64) for c in USECOLS
         if c not in ("li", "top_bottom", "game_type",
                      "asof_pitcher_success_rate")}
    A["li"] = d["li"].to_numpy(np.float64)
    A["asof_psr"] = d["asof_pitcher_success_rate"].to_numpy(np.float64)
    A["top_bottom"] = (d["top_bottom"].to_numpy() == "B").astype(np.int64)
    A["game_type"] = d["game_type"].to_numpy()
    return A


def dense(v):
    ids, idx = np.unique(v, return_inverse=True)
    return ids, idx.astype(np.int64)


# ---------------------------------------------------------------- 축 정의
# 🔴 li 문턱은 train 전체에서 동결한다 (S170). 값은 make_axes() 가 되돌려 준다.
def make_axes(A):
    li_thr = float(np.median(A["li"]))                  # train 동결 문턱
    P_ids, P_idx = dense(A["pitcher_id"])
    B_ids, B_idx = dense(A["batter_id"])
    PT_ids, PT_idx = dense(A["pitcher_team_id"])
    BT_ids, BT_idx = dense(A["batter_team_id"])

    ax = {}
    def add(name, subj_idx, nP, code, nc, kind):
        ax[name] = dict(subj=subj_idx, P=nP, code=code.astype(np.int64),
                        nc=int(nc), kind=kind)

    nP, nB = len(P_ids), len(B_ids)
    # --- 양성 대조 4 (출하된 축) -----------------------------------------
    add("pit_bhand", P_idx, nP, A["batter_hand"] - 1, 2, "positive")
    add("bat_phand", B_idx, nB, A["pitcher_hand"] - 1, 2, "positive")
    add("pit_strk2", P_idx, nP, (A["strikes_before"] >= 2), 2, "positive")
    add("pit_ball3", P_idx, nP, (A["balls_before"] >= 3), 2, "positive")
    # --- 신규 후보 -------------------------------------------------------
    add("pit_runner", P_idx, nP, (A["num_runners_on"] >= 1), 2, "cand")
    add("pit_r1b",    P_idx, nP, (A["runner_on_1b"] == 1), 2, "cand")
    add("pit_sp",     P_idx, nP, ((A["runner_on_2b"] == 1) |
                                  (A["runner_on_3b"] == 1)), 2, "cand")
    add("pit_out2",   P_idx, nP, (A["outs_before"] >= 2), 2, "cand")
    add("pit_out0",   P_idx, nP, (A["outs_before"] == 0), 2, "cand")
    add("bat_strk2",  B_idx, nB, (A["strikes_before"] >= 2), 2, "cand")
    add("bat_ball3",  B_idx, nB, (A["balls_before"] >= 3), 2, "cand")
    add("bat_runner", B_idx, nB, (A["num_runners_on"] >= 1), 2, "cand")
    add("pit_hilev",  P_idx, nP, (A["li"] >= li_thr), 2, "cand")
    add("pit_ahead",  P_idx, nP, (A["score_diff_pitcher_team"] > 0), 2, "cand")
    add("pit_tb",     P_idx, nP, A["top_bottom"], 2, "cand")
    add("pit_bteam",  P_idx, nP, BT_idx, len(BT_ids), "cand")
    add("bat_pteam",  B_idx, nB, PT_idx, len(PT_ids), "cand")
    add("pair_pb",    P_idx, nP, B_idx, nB, "cand_pair")
    # --- 음성 대조 -------------------------------------------------------
    add("pit_inning", P_idx, nP, (A["inning"] >= 7), 2, "negative_known")
    # --- 2026-08-28 방향 전환: 팀원 28쌍 재실행에서 유일하게 살아남은 축 -----
    add("pit_strk1",  P_idx, nP, (A["strikes_before"] >= 1), 2, "cand_live")
    add("pit_ball1",  P_idx, nP, (A["balls_before"] >= 1), 2, "cand_live")
    add("pit_ball2",  P_idx, nP, (A["balls_before"] >= 2), 2, "cand_live")
    return ax, li_thr, dict(li_thr=li_thr, n_pitcher=nP, n_batter=nB,
                            n_team=len(BT_ids))


def build_axis(y, subj, P, code, nc, k):
    """출하 splits.build() 의 한 축 부분 — 전역 주체 인덱스판."""
    league = float(y.mean())
    n = np.bincount(subj, minlength=P).astype(np.float64)
    s = np.bincount(subj, weights=y, minlength=P)
    p_all = (s + k * league) / (n + k)
    ln = np.bincount(code, minlength=nc).astype(np.float64)
    ls = np.bincount(code, weights=y, minlength=nc)
    lrate = (ls + k * league) / (ln + k)
    lgc = lg(lrate) - lg(league)
    flat = subj * nc + code
    cn = np.bincount(flat, minlength=P * nc).astype(np.float64).reshape(P, nc)
    cs = np.bincount(flat, weights=y, minlength=P * nc).reshape(P, nc)
    rate = (cs + k * p_all[:, None]) / (cn + k)
    D = (lg(rate) - lg(p_all)[:, None]) - lgc[None, :]
    return D, cn


def wcorr(x, y, w):
    w = np.asarray(w, np.float64)
    sw = w.sum()
    if sw <= 0 or len(x) < 3:
        return float("nan")
    mx = (w * x).sum() / sw
    my = (w * y).sum() / sw
    vx = (w * (x - mx) ** 2).sum() / sw
    vy = (w * (y - my) ** 2).sum() / sw
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float((w * (x - mx) * (y - my)).sum() / sw / np.sqrt(vx * vy))


WEIGHTS = {
    "unw":  lambda a, b: np.ones_like(a),
    "min":  lambda a, b: np.minimum(a, b),
    "sum":  lambda a, b: a + b,
    "harm": lambda a, b: 2 * a * b / np.maximum(a + b, 1.0),
    "cnA":  lambda a, b: a,
    "cnB":  lambda a, b: b,
    "sqmin": lambda a, b: np.sqrt(np.minimum(a, b)),
}


def rho_pair(y, mask_a, mask_b, ax, k, wname="min", shape="flat", min_cn=1):
    """창 A(mask_a) 와 창 B(mask_b) 에서 지은 D 의 상관."""
    DA, cA = build_axis(y[mask_a], ax["subj"][mask_a], ax["P"],
                        ax["code"][mask_a], ax["nc"], k)
    DB, cB = build_axis(y[mask_b], ax["subj"][mask_b], ax["P"],
                        ax["code"][mask_b], ax["nc"], k)
    if shape == "flat":
        m = (cA >= min_cn) & (cB >= min_cn)
        x, yy = DA[m], DB[m]
        w = WEIGHTS[wname](cA[m], cB[m])
    else:                                   # 셀 2개짜리 축의 '차이' 판
        m = (cA >= min_cn).all(1) & (cB >= min_cn).all(1)
        x = (DA[:, 1] - DA[:, 0])[m]
        yy = (DB[:, 1] - DB[:, 0])[m]
        w = WEIGHTS[wname](cA[m].min(1), cB[m].min(1))
    return wcorr(x, yy, w), int(len(x))

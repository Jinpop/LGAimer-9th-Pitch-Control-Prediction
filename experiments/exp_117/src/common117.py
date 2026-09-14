#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_117 공용 — 2023 레짐 경계 리그 셀 레벨 이동 스캔.

리그 셀 = 주체 키(pitcher_id/batter_id)가 없는 현재 행 열만으로 정의되는 셀.
exp_116 하니스(주체 x 상황 스플릿)와 다른 대상이라 산식은 새로 쓰되,
널 대조·패널·가중 상관 관례는 exp_116 을 그대로 따른다.

🔴 배포 취득처(S170): 모든 셀 정의가 현재 행 열이고, 구간 경계는 train 전체에서
   동결해 FROZEN 에 값을 박아 둔다. 배포에서 test 분위수를 다시 계산하지 않는다.
"""
import numpy as np
import pandas as pd

EPS = 1e-9
TRAIN = "<PROJECT_ROOT>/data/train.csv"
SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
NS = len(SEASONS)
PRE = [0, 1, 2, 3]          # 2019..2022
POST = [4, 5]               # 2023..2024

USECOLS = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
           "game_type", "balls_before", "strikes_before", "outs_before",
           "num_runners_on", "base_state", "li", "score_diff_pitcher_team",
           "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
           "asof_pitcher_n", "asof_pitcher_pitchmix_n",
           "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
           "asof_pitcher_offspeed_rate", "control_success"]

# ---------------------------------------------------------------- 동결 경계
# 🔴 train 1,475,092 행 전체에서 계산해 여기에 못 박는다 (S170).
#    scan 스크립트가 실행 시 재계산해 이 값과 일치하는지 assert 한다.
FROZEN = {
    "li_q": [0.4, 0.8, 1.32],                  # train 25/50/75 분위
    "asof_pitcher_n_q": [684.0, 1776.0, 3725.0],
    "score_diff_cuts": [-3.5, -0.5, 0.5, 3.5],  # <=-4 / -3..-1 / 0 / 1..3 / >=4
    "inning_cuts": [3.5, 6.5],                  # <=3 / 4-6 / 7+
}


def lg(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load():
    d = pd.read_csv(TRAIN, encoding="utf-8-sig", usecols=USECOLS)
    A = {}
    for c in ("season", "game_month", "game_dayofweek", "inning",
              "balls_before", "strikes_before", "outs_before",
              "num_runners_on", "pitcher_hand", "batter_hand",
              "asof_pitcher_n", "asof_pitcher_pitchmix_n",
              "control_success", "pitcher_id", "batter_id"):
        A[c] = d[c].to_numpy(np.int64)
    A["score_diff"] = d["score_diff_pitcher_team"].to_numpy(np.int64)
    A["li"] = d["li"].to_numpy(np.float64)
    A["isB"] = (d["top_bottom"].to_numpy() == "B").astype(np.int64)
    A["isF"] = (d["game_type"].to_numpy() == "F").astype(np.int64)
    bs, bsi = np.unique(d["base_state"].to_numpy(), return_inverse=True)
    A["base_state"] = bsi.astype(np.int64)
    A["base_state_labels"] = list(bs)
    mix = d[["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
             "asof_pitcher_offspeed_rate"]].to_numpy(np.float64)
    dom = np.where(np.isnan(mix).any(1), 3, np.nanargmax(
        np.where(np.isnan(mix), -1.0, mix), axis=1))
    A["pitchmix_dom"] = dom.astype(np.int64)
    A["y"] = A["control_success"].astype(np.float64)
    A["seas_idx"] = np.searchsorted(np.array(SEASONS), A["season"])
    return A


def check_frozen(A):
    """동결 경계가 train 에서 실제로 나온 값인지 기계 확인."""
    got = {
        "li_q": [float(np.percentile(A["li"], q)) for q in (25, 50, 75)],
        "asof_pitcher_n_q": [float(np.percentile(A["asof_pitcher_n"].astype(float), q))
                             for q in (25, 50, 75)],
    }
    ok = (np.allclose(got["li_q"], FROZEN["li_q"], atol=1e-9) and
          np.allclose(got["asof_pitcher_n_q"], FROZEN["asof_pitcher_n_q"], atol=1e-9))
    return ok, got


# ---------------------------------------------------------------- 셀 정의
def make_cells(A):
    """name -> dict(code, nc, labels, family)"""
    C = {}

    def add(name, code, labels, family="single"):
        code = np.asarray(code, np.int64)
        C[name] = dict(code=code, nc=int(len(labels)), labels=list(labels),
                       family=family)

    add("base_state", A["base_state"], A["base_state_labels"])
    add("outs", A["outs_before"], ["0", "1", "2"])
    add("runners", A["num_runners_on"], ["0", "1", "2", "3"])
    inn = np.digitize(A["inning"].astype(float), FROZEN["inning_cuts"])
    add("inning_bin", inn, ["<=3", "4-6", "7+"])
    add("top_bottom", A["isB"], ["T", "B"])
    add("game_type", A["isF"], ["R", "F"], family="positive_control")
    hp = (A["pitcher_hand"] - 1) * 2 + (A["batter_hand"] - 1)
    add("hand_pair", hp, ["P1B1", "P1B2", "P2B1", "P2B2"])
    sd = np.digitize(A["score_diff"].astype(float), FROZEN["score_diff_cuts"])
    add("score_diff_bin", sd, ["<=-4", "-3..-1", "0", "1..3", ">=4"])
    lib = np.digitize(A["li"], FROZEN["li_q"])
    add("li_bin", lib, ["Q1", "Q2", "Q3", "Q4"])
    add("game_month", A["game_month"] - 3, [str(m) for m in range(3, 11)])
    add("dayofweek", A["game_dayofweek"], [str(i) for i in range(7)])
    apn = np.digitize(A["asof_pitcher_n"].astype(float),
                      FROZEN["asof_pitcher_n_q"])
    add("asof_pn_bin", apn, ["Q1", "Q2", "Q3", "Q4"])
    add("pitchmix_dom", A["pitchmix_dom"], ["fastball", "breaking", "offspeed",
                                            "unknown"])
    # 카운트 계열 (CS3 영역 — 양성 대조 / 2단에서 반드시 잔차화)
    cnt = A["balls_before"] * 3 + A["strikes_before"]
    add("count12", cnt, ["%d-%d" % (b, s) for b in range(4) for s in range(3)],
        family="count")
    add("balls", A["balls_before"], ["0", "1", "2", "3"], family="count")
    add("strikes", A["strikes_before"], ["0", "1", "2"], family="count")

    # 2원 교차 (셀 수 <= 60)
    add("hand_x_outs", hp * 3 + A["outs_before"],
        ["%s|o%d" % (h, o) for h in ["P1B1", "P1B2", "P2B1", "P2B2"]
         for o in range(3)], family="cross")
    add("hand_x_base", hp * 8 + A["base_state"],
        ["%s|%s" % (h, b) for h in ["P1B1", "P1B2", "P2B1", "P2B2"]
         for b in A["base_state_labels"]], family="cross")
    add("outs_x_base", A["outs_before"] * 8 + A["base_state"],
        ["o%d|%s" % (o, b) for o in range(3)
         for b in A["base_state_labels"]], family="cross")
    add("count_x_hand", cnt * 4 + hp,
        ["%d-%d|%s" % (b, s, h) for b in range(4) for s in range(3)
         for h in ["P1B1", "P1B2", "P2B1", "P2B2"]], family="cross_count")
    return C


# ------------------------------------------------------- 레벨 이동 통계
def cell_stats(y, code, seas_idx, base_row, nc, scale):
    """dev / var / n 을 (NS, nc) 격자로."""
    flat = seas_idx * nc + code
    n = np.bincount(flat, minlength=NS * nc).astype(np.float64).reshape(NS, nc)
    s = np.bincount(flat, weights=y, minlength=NS * nc).reshape(NS, nc)
    b = np.bincount(flat, weights=base_row, minlength=NS * nc).reshape(NS, nc)
    nn = np.maximum(n, 1.0)
    r = s / nn
    if scale == "logit":
        fr = lg(r)
        rc = np.clip(r, 1e-6, 1 - 1e-6)
        V = 1.0 / (nn * rc * (1 - rc))
    else:
        fr = r
        V = r * (1 - r) / nn
    dev = fr - b / nn
    dev[n == 0] = np.nan
    V[n == 0] = np.nan
    return dev, V, n


def shift_stats(dev, V, n):
    """shift / hold / pre_sd / z 를 셀별로."""
    ok = np.isfinite(dev).all(0)
    shift = np.nanmean(dev[POST], 0) - np.nanmean(dev[PRE], 0)
    hold = np.abs(dev[4] - dev[5])
    pre_sd = np.nanstd(dev[PRE], 0)
    var = V[POST].sum(0) / len(POST) ** 2 + V[PRE].sum(0) / len(PRE) ** 2
    se = np.sqrt(var)
    z = shift / np.maximum(se, 1e-300)
    for a in (shift, hold, pre_sd, z):
        a[~ok] = np.nan
    return shift, hold, pre_sd, se, z


def baseline_row(y, seas_idx, group_row, ngroup, scale):
    """행별 기준선 f(rate(기준 그룹))."""
    n = np.bincount(group_row, minlength=ngroup).astype(np.float64)
    s = np.bincount(group_row, weights=y, minlength=ngroup)
    r = s / np.maximum(n, 1.0)
    f = lg(r) if scale == "logit" else r
    return f[group_row]

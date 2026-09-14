"""exp_079 정직면 채점 모듈 — exp_078 stage34_078.py 의 `oot_pts` 와 동일 산술.

면:   ≤2023 학습 → 2024 예측 253,507행 (팀원 OOT 납품과 같은 면)
상수: BO 출하 상수 (build_BO/pkg_BO)
후처리: 전역 재중심화 (mean(p) == mean(y) 로 뉴턴 수렴)
기준: 팀원 납품 p_cat 을 넣으면 928.501461058684 (exp_078 실측)

p_cat 만 바깥에서 로짓으로 받는다.  나머지 팔은 납품 벡터 그대로.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
REF = 0.2498069
K = 100000.0 / REF
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
AA = (42, 7, 2024)


def lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def pts(b):
    return 100000.0 * (1.0 - b / REF)


V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}

_tro = pd.read_csv(f"{H}/data/train.csv",
                   usecols=["row_id", "season", "control_success"])
_tro = _tro[_tro.season == 2024].sort_values("row_id", kind="stable")
yo = _tro.control_success.to_numpy(np.float64)
_rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy",
                          allow_pickle=True)).astype(str)
assert np.array_equal(_rid, _tro.row_id.to_numpy().astype(str))
assert len(yo) == 253507

BO = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "bo_constants.json")))
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA],
               axis=0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA],
                axis=0)
_L = {n: lg(V[n]) for n in ("p_lgb", "p_leaf", "p_nn", "p_q", "p_ebm")}


def rc(z):
    """전역 재중심화 상수 — mean(expit(z+c)) == mean(y)."""
    c, yb = 0.0, float(yo.mean())
    for _ in range(80):
        p = expit(z + c)
        mu = float(p.mean())
        if abs(mu - yb) < 1e-14:
            break
        d = (mu - yb) / float((p * (1 - p)).mean())
        c -= d
        if abs(d) < 1e-14:
            break
    return c


def surface_p(lg_cat):
    lv = (W4["LGB"] * _L["p_lgb"] + W4["CAT"] * lg_cat
          + W4["NN"] * _L["p_nn"] + W4["LEAF"] * _L["p_leaf"])
    z = (BO["w_out"] * ((1 - BO["w_ebm"])
                        * (BO["w_v6_inner"] * lv
                           + (1 - BO["w_v6_inner"]) * _L["p_q"])
                        + BO["w_ebm"] * _L["p_ebm"]
                        + BO["beta_ins"] * V["off"])
         + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70)
    return expit(z + rc(z))


def surface_pts(lg_cat):
    p = surface_p(lg_cat)
    b = float(np.mean((p - yo) ** 2))
    return pts(b), b


def arm_pts(lg_cat):
    """v6 내부 CatBoost 팔 단독 눈금 (재중심화 적용)."""
    p = expit(lg_cat + rc(lg_cat))
    b = float(np.mean((p - yo) ** 2))
    return pts(b), b

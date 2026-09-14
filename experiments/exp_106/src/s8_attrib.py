"""exp_106 s8 — 결합 이득의 귀속. "+12.58 이 어느 축에 사는가".

1자유도 합(6.73)보다 결합(12.58)이 두 배 가까이 크다 = 축들이 심하게 상관돼 있어
좌표 방향 어디에도 없는 대각선에서 이득이 난다(S19). 그 대각선이 무엇인지 가른다.

부분집합 적합 넷
  A  루트 4축만          (alpha · c_aa · c_T70 · w_out)
  B  mate 내부 3축만     (w_v6_inner · w_ebm · w_deep)
  C  우리 팔 지분 동결   — w_out 0.7 고정 · c_aa + c_T70 = 0.3 고정
     🔴 이게 중요한 이유: 정직면의 AA·T70 은 **3시드 ≤2023 복제본**이고 배포는 AA 6시드
        ≤2024 다. 우리 팔이 이 면에서 체계적으로 약하면 "우리 몫을 줄여라" 가
        면의 결함이지 신호가 아니다. 그 축을 얼려 나머지가 얼마를 버는지 본다.
  D  축 하나씩 뺀 6축 적합 (leave-one-out) — 각 축의 대체 불가능한 몫
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                   # noqa: E402
from scipy.optimize import minimize                                  # noqa: E402
import surf106 as S                                                  # noqa: E402

t0 = time.time()
OUT = "/home/<user>/SungjinPark/exp106/out"
AX = S.AXES
th0 = dict(S.THETA0)
z0 = S.zvec(th0)
DEPLOY = S.score(z0, th0["alpha"], th0["beta_skew"])
BOUNDS = {"alpha": (0.5, 2.0), "c_aa": (-0.3, 1.0), "c_T70": (-0.3, 1.0),
          "w_out": (0.2, 1.3), "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.6, 0.6),
          "w_deep": (0.0, 0.95)}


def log(*a):
    print("[%6.1fs]" % (time.time() - t0), *a, flush=True)


def fit(free, mode=None):
    lo = np.array([BOUNDS[a][0] for a in free])
    hi = np.array([BOUNDS[a][1] for a in free])

    def build(x):
        th = dict(th0)
        for a, v in zip(free, x):
            th[a] = float(v)
        if mode == "frozen_share":          # 우리 팔 총 지분 0.3 · w_out 0.7 동결
            th["w_out"] = 0.7
            th["c_T70"] = 0.3 - th["c_aa"]
        return th

    def neg(x):
        return -S.sc_theta(build(x))

    res = minimize(neg, np.array([th0[a] for a in free]), method="Powell",
                   bounds=list(zip(lo, hi)),
                   options={"xtol": 1e-7, "ftol": 1e-9, "maxfev": 40000})
    th = build(res.x)
    zz = S.zvec(th)
    return {"free": list(free), "theta": {a: th[a] for a in AX},
            "이동": {a: th[a] - th0[a] for a in AX},
            "gain_pts": -float(res.fun) - DEPLOY,
            "gain_sd_matched_pts": S.score(S.sd_match(zz, z0), th0["alpha"],
                                           th0["beta_skew"]) - DEPLOY,
            "n_dof": len(free), "nfev": int(res.nfev)}


R = {"exp_id": "exp_106", "stage": "s8_attrib", "deploy": DEPLOY,
     "귀무_자유도당": 0.390759, "subsets": {}}
JOBS = [
    ("A_root4", ["alpha", "c_aa", "c_T70", "w_out"], None),
    ("B_inner3", ["w_v6_inner", "w_ebm", "w_deep"], None),
    ("C_frozen_our_share", ["alpha", "c_aa", "w_v6_inner", "w_ebm", "w_deep"],
     "frozen_share"),
    ("C2_frozen_share_inner_only", ["w_v6_inner", "w_ebm", "w_deep"], "frozen_share"),
]
for name, free, mode in JOBS:
    R["subsets"][name] = fit(free, mode)
    e = R["subsets"][name]
    log("%-28s dof %d  gain %+8.4f (sd맞춤 %+8.4f)  귀무기대 %+5.2f"
        % (name, e["n_dof"], e["gain_pts"], e["gain_sd_matched_pts"],
           e["n_dof"] * 0.390759))
    json.dump(R, open(f"{OUT}/s8_attrib.json", "w"), ensure_ascii=False, indent=1)

R["leave_one_out"] = {}
for a in AX:
    free = [x for x in AX if x != a]
    e = fit(free)
    R["leave_one_out"][a] = e
    log("LOO drop %-12s gain %+8.4f" % (a, e["gain_pts"]))
    json.dump(R, open(f"{OUT}/s8_attrib.json", "w"), ensure_ascii=False, indent=1)

R["실행시간_분"] = (time.time() - t0) / 60.0
json.dump(R, open(f"{OUT}/s8_attrib.json", "w"), ensure_ascii=False, indent=1)
log("done")

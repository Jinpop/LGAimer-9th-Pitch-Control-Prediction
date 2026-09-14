"""exp_106 s10 — 🔴 s2 의 결합해는 **출하 불가**다. 출하 가능한 족에서 다시 푼다.

s2 가 낸 두 해가 둘 다 출하 코드의 assert 를 어긴다:
  ① 7축 해 : c_T70 이 격자 **하한 -0.3 에 붙었다**(내부 최적이 아니다) ·
             w_out + c_aa + c_T70 = 0.822 ≠ 1  → `script.py:391` assert 위반
  ② "출하형": w_out(=루트 blend.json `w_v6`) **1.176** → `script.py:188` assert
             `0.0 <= w_v6 <= 1.0` 위반. 지휘자가 합 제약만 걸고 [0,1] 을 안 걸었다.

그래서 여기서 넷을 푼다
  F1 ship_valid  : w_out∈[0,1] · c_aa = 1 - w_out - c_T70 (합 제약) · 나머지 자유
  F2 wide        : 경계를 크게 넓혀 c_T70 의 **내부 최적이 존재하는지** 확인
  F3 T70frozen   : c_T70 을 출하값 0.04756 에 얼리고 6축 — "+12.58 중 T70 몫이 얼마인가"
  F4 ship_valid + T70frozen
그리고 F1 해를 **투수 반분할 교차적합** 으로 표본 밖에서 검증한다(10분할 x 2방향).
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np                                                   # noqa: E402
from scipy.optimize import minimize                                  # noqa: E402
from scipy.special import expit                                      # noqa: E402
import surf106 as S                                                  # noqa: E402

t0 = time.time()
OUT = "/home/<user>/SungjinPark/exp106/out"
AX = S.AXES
th0 = dict(S.THETA0)
z0 = S.zvec(th0)
DEPLOY = S.score(z0, th0["alpha"], th0["beta_skew"])
SHIP_T70 = th0["c_T70"]


def log(*a):
    print("[%6.1fs]" % (time.time() - t0), *a, flush=True)


def make(kind):
    """(free 축 목록, 하한, 상한, build 함수) 를 돌려준다."""
    if kind == "ship_valid":
        free = ["alpha", "w_out", "c_T70", "w_v6_inner", "w_ebm", "w_deep"]
        b = {"alpha": (0.5, 2.0), "w_out": (0.0, 1.0), "c_T70": (-0.5, 0.5),
             "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.6, 0.6), "w_deep": (0.0, 0.95)}

        def build(x):
            th = dict(th0)
            for a, v in zip(free, x):
                th[a] = float(v)
            th["c_aa"] = 1.0 - th["w_out"] - th["c_T70"]
            return th
    elif kind == "wide":
        free = list(AX)
        b = {"alpha": (0.3, 3.0), "c_aa": (-1.5, 2.0), "c_T70": (-2.0, 1.5),
             "w_out": (0.0, 2.5), "w_v6_inner": (0.0, 1.0), "w_ebm": (-1.0, 1.0),
             "w_deep": (0.0, 0.95)}

        def build(x):
            th = dict(th0)
            for a, v in zip(free, x):
                th[a] = float(v)
            return th
    elif kind == "T70frozen":
        free = ["alpha", "c_aa", "w_out", "w_v6_inner", "w_ebm", "w_deep"]
        b = {"alpha": (0.5, 2.0), "c_aa": (-0.3, 1.0), "w_out": (0.2, 1.3),
             "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.6, 0.6), "w_deep": (0.0, 0.95)}

        def build(x):
            th = dict(th0)
            for a, v in zip(free, x):
                th[a] = float(v)
            th["c_T70"] = SHIP_T70
            return th
    elif kind == "ship_valid_T70frozen":
        free = ["alpha", "w_out", "w_v6_inner", "w_ebm", "w_deep"]
        b = {"alpha": (0.5, 2.0), "w_out": (0.0, 1.0), "w_v6_inner": (0.0, 1.0),
             "w_ebm": (-0.6, 0.6), "w_deep": (0.0, 0.95)}

        def build(x):
            th = dict(th0)
            for a, v in zip(free, x):
                th[a] = float(v)
            th["c_T70"] = SHIP_T70
            th["c_aa"] = 1.0 - th["w_out"] - th["c_T70"]
            return th
    lo = np.array([b[a][0] for a in free])
    hi = np.array([b[a][1] for a in free])
    return free, lo, hi, build


def fit(kind, nstart=5, idx=None):
    free, lo, hi, build = make(kind)

    def sc(th):
        if idx is None:
            return S.sc_theta(th)
        z = S.zvec(th)[idx]
        yy = S.y[idx]
        c, tgt = 0.0, float(yy.mean())
        for _ in range(200):
            p = expit(th["alpha"] * z + th["beta_skew"] * z * z + c)
            g = float(p.mean()) - tgt
            if abs(g) < 1e-15:
                break
            c -= g / float((p * (1 - p)).mean())
        p = expit(th["alpha"] * z + th["beta_skew"] * z * z + c)
        return S.pts(float(np.mean((p - yy) ** 2)))

    def neg(x):
        return -sc(build(x))

    rng = np.random.default_rng(20261029)
    xs = [np.clip(np.array([th0[a] for a in free]), lo, hi)]
    for _ in range(nstart - 1):
        xs.append(lo + rng.random(len(free)) * (hi - lo))
    best, bx = -1e18, None
    for x0 in xs:
        r = minimize(neg, x0, method="Powell", bounds=list(zip(lo, hi)),
                     options={"xtol": 1e-7, "ftol": 1e-9, "maxfev": 30000})
        r = minimize(neg, r.x, method="Powell", bounds=list(zip(lo, hi)),
                     options={"xtol": 1e-8, "ftol": 1e-10, "maxfev": 30000})
        if -r.fun > best:
            best, bx = -float(r.fun), r.x
    th = build(bx)
    at_bound = {a: bool(abs(v - lo[i]) < 1e-6 or abs(v - hi[i]) < 1e-6)
                for i, (a, v) in enumerate(zip(free, bx))}
    return th, best, at_bound, free


R = {"exp_id": "exp_106", "stage": "s10_shipvalid", "deploy_score": DEPLOY,
     "🔴_왜": "s2 의 두 해가 출하 코드 assert 를 어긴다 (c_T70 경계 · w_v6 > 1)",
     "fits": {}}
for kind in ("ship_valid", "wide", "T70frozen", "ship_valid_T70frozen"):
    th, sc_, ab, free = fit(kind)
    zz = S.zvec(th)
    R["fits"][kind] = {
        "free": free, "theta": {a: th[a] for a in AX},
        "이동": {a: th[a] - th0[a] for a in AX},
        "gain_pts": sc_ - DEPLOY,
        "gain_sd_matched_pts": S.score(S.sd_match(zz, z0), th0["alpha"],
                                       th0["beta_skew"]) - DEPLOY,
        "sd_ratio": float(zz.std() / z0.std()),
        "경계에_붙은_축": [a for a, v in ab.items() if v],
        "sum_check_w_out+c_aa+c_T70": th["w_out"] + th["c_aa"] + th["c_T70"],
        "출하_assert_통과": bool(0.0 <= th["w_out"] <= 1.0
                                and abs(th["w_out"] + th["c_aa"] + th["c_T70"] - 1.0) < 1e-9
                                and 0.5 <= th["alpha"] <= 2.0)}
    e = R["fits"][kind]
    log("%-22s gain %+8.4f (sd맞춤 %+8.4f) 경계 %s  출하가능 %s"
        % (kind, e["gain_pts"], e["gain_sd_matched_pts"],
           e["경계에_붙은_축"], e["출하_assert_통과"]))
    json.dump(R, open(f"{OUT}/s10_shipvalid.json", "w"), ensure_ascii=False, indent=1)

# ── F1 을 부분 걸음으로 · 부트스트랩 ────────────────────────────────────
thb = dict(th0, **{a: R["fits"]["ship_valid"]["theta"][a] for a in AX})
R["partial_steps_ship_valid"] = []
for f in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    th = dict(th0, **{a: th0[a] + f * (thb[a] - th0[a]) for a in AX})
    R["partial_steps_ship_valid"].append({"frac": f, "gain_pts": S.sc_theta(th) - DEPLOY})
p_b = S.prob(S.zvec(thb), thb["alpha"], thb["beta_skew"])
p_d = S.prob(z0, th0["alpha"], th0["beta_skew"])
R["boot_ship_valid_vs_deploy"] = S.boot_t((p_b - S.y) ** 2 - (p_d - S.y) ** 2)
log("ship_valid boot t %.3f" % (R["boot_ship_valid_vs_deploy"]["t"] or 0))
json.dump(R, open(f"{OUT}/s10_shipvalid.json", "w"), ensure_ascii=False, indent=1)

# ── 교차적합: F1 족을 반쪽에서 적합해 나머지 반쪽에서 채점 ───────────────
uq, inv = np.unique(S.pit, return_inverse=True)
NB = len(uq)
rng = np.random.default_rng(20261030)


def sc_idx(th, idx):
    z = S.zvec(th)[idx]
    yy = S.y[idx]
    c, tgt = 0.0, float(yy.mean())
    for _ in range(200):
        p = expit(th["alpha"] * z + th["beta_skew"] * z * z + c)
        g = float(p.mean()) - tgt
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    p = expit(th["alpha"] * z + th["beta_skew"] * z * z + c)
    return S.pts(float(np.mean((p - yy) ** 2)))


rows = []
for k in range(10):
    m = rng.random(NB) < 0.5
    iA = np.where(m[inv])[0]
    iB = np.where(~m[inv])[0]
    for fit_i, ev_i, tag in ((iA, iB, "A->B"), (iB, iA, "B->A")):
        th_f, sc_f, _, _ = fit("ship_valid", nstart=2, idx=fit_i)
        d_in = sc_idx(th0, fit_i)
        d_out = sc_idx(th0, ev_i)
        half = dict(th0, **{a: th0[a] + 0.5 * (th_f[a] - th0[a]) for a in AX})
        rows.append({"split": k, "dir": tag,
                     "gain_in": sc_f - d_in,
                     "gain_out": sc_idx(th_f, ev_i) - d_out,
                     "gain_out_halfstep": sc_idx(half, ev_i) - d_out,
                     "theta": {a: th_f[a] for a in AX}})
    log("split %d  in %+7.3f/%+7.3f  OUT %+7.3f/%+7.3f"
        % (k, rows[-2]["gain_in"], rows[-1]["gain_in"],
           rows[-2]["gain_out"], rows[-1]["gain_out"]))
    R["crossfit_ship_valid"] = rows
    json.dump(R, open(f"{OUT}/s10_shipvalid.json", "w"), ensure_ascii=False, indent=1)


def summ(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "se": float(a.std(ddof=1) / np.sqrt(len(a))),
            "frac_positive": float((a > 0).mean()),
            "min": float(a.min()), "max": float(a.max())}


R["crossfit_요약"] = {
    "in_sample_half": summ([r["gain_in"] for r in rows]),
    "OUT_of_sample": summ([r["gain_out"] for r in rows]),
    "OUT_halfstep": summ([r["gain_out_halfstep"] for r in rows]),
    "n_dof": 6,
    "귀무_표본밖_기대(-m*K*B/N_fit)": -6 * 0.390759 * 2}
R["실행시간_분"] = (time.time() - t0) / 60.0
json.dump(R, open(f"{OUT}/s10_shipvalid.json", "w"), ensure_ascii=False, indent=1)
log("요약", json.dumps(R["crossfit_요약"], ensure_ascii=False))

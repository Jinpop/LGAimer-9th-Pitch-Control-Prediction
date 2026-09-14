"""exp_106 s7 — 투수 분할 교차적합. "재적합 이득이 표본 밖에서 살아남는가".

s2 의 +N 점은 **적합한 바로 그 253,507행에서 잰 값**이다. 7자유도를 열면
순수 잡음만으로도 K*Brier/N * 7 = 약 2.73점이 공짜로 나온다(S22·S41).
귀무 분포는 그 몫을 재지만, **그 몫을 뺀 나머지가 표본 밖에서 실현되는지**는
따로 물어야 한다.

설계
  - 투수를 무작위 반으로 갈라(A/B) A 에서 7축을 적합하고 B 에서 점수를 잰다.
    양방향(A->B, B->A) 20분할 x 2 = 40회.
  - d_adj(못박음 상수)는 **평가 쪽에서 다시 못박는다** — 후보와 배포점에 똑같이
    적용되므로 어느 쪽도 편들지 않고, 면의 관례와 같다.
  - 같은 절차를 **개수 맞춤 무작위 방향 7개**에도 돌려 표본 밖 손실의 눈금을 잡는다.
  ⚠️ S34: 표본을 반으로 줄인 대가가 있다. 그래서 귀무를 같은 반쪽에서 돌린다.
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
BOUNDS = {"alpha": (0.5, 2.0), "c_aa": (-0.3, 1.0), "c_T70": (-0.3, 1.0),
          "w_out": (0.2, 1.3), "w_v6_inner": (0.0, 1.0), "w_ebm": (-0.6, 0.6),
          "w_deep": (0.0, 0.95)}


def log(*a):
    print("[%6.1fs]" % (time.time() - t0), *a, flush=True)


def pin_sub(z, alpha, beta, ysub):
    c, tgt = 0.0, float(ysub.mean())
    for _ in range(200):
        p = expit(alpha * z + beta * z * z + c)
        g = float(p.mean()) - tgt
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


def sc_sub(z, alpha, beta, idx):
    zz, yy = z[idx], S.y[idx]
    c = pin_sub(zz, alpha, beta, yy)
    p = expit(alpha * zz + beta * zz * zz + c)
    return S.pts(float(np.mean((p - yy) ** 2)))


def fit_on(idx, x0=None):
    lo = np.array([BOUNDS[a][0] for a in AX])
    hi = np.array([BOUNDS[a][1] for a in AX])

    def neg(x):
        th = dict(th0)
        for a, v in zip(AX, x):
            th[a] = float(v)
        return -sc_sub(S.zvec(th), th["alpha"], th["beta_skew"], idx)

    x_s = np.array([th0[a] for a in AX]) if x0 is None else np.asarray(x0)
    res = minimize(neg, x_s, method="Powell", bounds=list(zip(lo, hi)),
                   options={"xtol": 1e-6, "ftol": 1e-8, "maxfev": 40000})
    th = dict(th0)
    for a, v in zip(AX, res.x):
        th[a] = float(v)
    return th


# ── 실축의 z 공간 방향 (귀무 크기 맞춤용) ────────────────────────────────
wi, we, wd, wo = (th0["w_v6_inner"], th0["w_ebm"], th0["w_deep"], th0["w_out"])
base_in = wi * S.LV + (1.0 - wi) * S.LQ
zm = (1.0 - we) * base_in + we * S.LE + th0["beta_ins"] * S.OFF
zm_full = (1.0 - wd) * zm + wd * S.Z_DEEP
z0 = S.zvec(th0)
DZ_REAL = {"c_aa": S.L_aa, "c_T70": S.L_t70, "w_out": zm_full,
           "w_deep": wo * (S.Z_DEEP - zm),
           "w_v6_inner": wo * (1.0 - wd) * (1.0 - we) * (S.LV - S.LQ),
           "w_ebm": wo * (1.0 - wd) * (S.LE - base_in),
           "alpha": z0}
RMS = np.array([float(np.sqrt((DZ_REAL[a] ** 2).mean())) for a in AX])

uq, inv = np.unique(S.pit, return_inverse=True)
NB = len(uq)
rng = np.random.default_rng(20261027)
NSPLIT = 20
R = {"exp_id": "exp_106", "stage": "s7_crossfit", "n_splits": NSPLIT,
     "설계": "투수 무작위 반분할 · 적합 A / 평가 B (양방향) · d_adj 는 평가 쪽에서 재못박음",
     "n_axes": len(AX)}
DEP_FULL = S.sc_theta(th0)

rows, nullrows = [], []
for k in range(NSPLIT):
    m = rng.random(NB) < 0.5
    iA = np.where(m[inv])[0]
    iB = np.where(~m[inv])[0]
    for (fit_i, ev_i, tag) in ((iA, iB, "A->B"), (iB, iA, "B->A")):
        th_f = fit_on(fit_i)
        zf = S.zvec(th_f)
        dep_in = sc_sub(z0, th0["alpha"], th0["beta_skew"], fit_i)
        dep_out = sc_sub(z0, th0["alpha"], th0["beta_skew"], ev_i)
        g_in = sc_sub(zf, th_f["alpha"], th_f["beta_skew"], fit_i) - dep_in
        g_out = sc_sub(zf, th_f["alpha"], th_f["beta_skew"], ev_i) - dep_out
        # 절반 걸음(위험 관리형 착륙)
        th_h = {a: th0[a] + 0.5 * (th_f[a] - th0[a]) for a in AX}
        th_h = dict(th0, **th_h)
        zh = S.zvec(th_h)
        g_out_half = sc_sub(zh, th_h["alpha"], th_h["beta_skew"], ev_i) - dep_out
        rows.append({"split": k, "dir": tag, "n_fit": int(len(fit_i)),
                     "n_eval": int(len(ev_i)), "gain_in": g_in, "gain_out": g_out,
                     "gain_out_halfstep": g_out_half,
                     "theta": {a: th_f[a] for a in AX}})
        # 귀무: z 공간 무작위 방향 7개, 실축 rms 맞춤 — 같은 반쪽에서 적합·평가
        deltas = []
        for j in range(len(AX)):
            v = rng.standard_normal(S.N)
            v -= v.mean()
            deltas.append(v / float(np.sqrt((v * v).mean())) * RMS[j])
        zf_ = z0[fit_i]
        c_ = pin_sub(zf_, th0["alpha"], th0["beta_skew"], S.y[fit_i])
        p_ = expit(th0["alpha"] * zf_ + th0["beta_skew"] * zf_ ** 2 + c_)
        q_ = p_ * (1 - p_)
        g_ = th0["alpha"] + 2 * th0["beta_skew"] * zf_
        s_ = p_ - S.y[fit_i]
        mq = float(q_.mean())
        E = np.stack([q_ * (g_ * d[fit_i] - float((q_ * g_ * d[fit_i]).mean()) / mq)
                      for d in deltas])
        Ln = -(E @ s_) / len(fit_i)
        Gn = (E @ E.T) / len(fit_i)
        w_, U = np.linalg.eigh(Gn)
        keep = w_ > 1e-10 * w_.max()
        tn = U[:, keep] @ ((U[:, keep].T @ Ln) / w_[keep])
        zpert = z0 + sum(float(tn[j]) * deltas[j] for j in range(len(AX)))
        nullrows.append({
            "split": k, "dir": tag,
            "gain_in": sc_sub(zpert, th0["alpha"], th0["beta_skew"], fit_i) - dep_in,
            "gain_out": sc_sub(zpert, th0["alpha"], th0["beta_skew"], ev_i) - dep_out})
    log("split %d  real in %+7.3f out %+7.3f | %+7.3f %+7.3f  || null out %+7.3f %+7.3f"
        % (k, rows[-2]["gain_in"], rows[-2]["gain_out"],
           rows[-1]["gain_in"], rows[-1]["gain_out"],
           nullrows[-2]["gain_out"], nullrows[-1]["gain_out"]))
    json.dump({**R, "rows": rows, "null_rows": nullrows},
              open(f"{OUT}/s7_crossfit.json", "w"), ensure_ascii=False, indent=1)


def summ(a):
    a = np.asarray(a, float)
    return {"n": int(len(a)), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
            "se": float(a.std(ddof=1) / np.sqrt(len(a))),
            "min": float(a.min()), "max": float(a.max()),
            "frac_positive": float((a > 0).mean())}


R["rows"] = rows
R["null_rows"] = nullrows
R["요약"] = {
    "real_gain_in_sample_halfdata": summ([r["gain_in"] for r in rows]),
    "real_gain_OUT_of_sample": summ([r["gain_out"] for r in rows]),
    "real_gain_OUT_halfstep": summ([r["gain_out_halfstep"] for r in rows]),
    "null_gain_in_sample_halfdata": summ([r["gain_in"] for r in nullrows]),
    "null_gain_OUT_of_sample": summ([r["gain_out"] for r in nullrows]),
    "이론_자유도당_반쪽": 0.390759 * 2,
    "배포점_전표본": DEP_FULL}
R["요약"]["real_minus_null_OUT"] = (R["요약"]["real_gain_OUT_of_sample"]["mean"]
                                    - R["요약"]["null_gain_OUT_of_sample"]["mean"])
R["실행시간_분"] = (time.time() - t0) / 60.0
json.dump(R, open(f"{OUT}/s7_crossfit.json", "w"), ensure_ascii=False, indent=1)
log("요약", json.dumps(R["요약"], ensure_ascii=False, indent=1))

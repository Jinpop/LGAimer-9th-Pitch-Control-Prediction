"""exp_114 stage4 — **스텝 정합인가 에폭 정합인가**를 직접 잰다.

배포 CE 는 6에폭 = 2160 배포스텝(n=1,475,092)이다. OOT 는 n=1,116,277 이라 에폭당 272스텝이고,
  에폭 정합 -> OOT 1632,  스텝 정합 -> OOT 2160.
어느 쪽이 배포의 정직면 상인지는 **최적 스텝이 학습 행 수에 어떻게 따라가는가**로 갈린다.

방법: 같은 정직면(같은 valid 2024 · 같은 Z_MATE)에서 학습 행만 50% / 75% / 100% 로 바꿔
  혼합 이득의 최적 스텝 k*(n) 을 잰다. k* ∝ n^p 로 적합해
    p ≈ 1  -> 에폭 법칙 (OOT 1632 가 배포 2160 의 상)
    p ≈ 0  -> 스텝 법칙 (OOT 2160 이 배포 2160 의 상)
  그리고 배포 n 으로 외삽해 **배포 최적 스텝**을 낸다.
"""
import json
import sys

sys.path.insert(0, "/home/<user>/SungjinPark/exp114/src")
import numpy as np  # noqa: E402
from surf114 import B, BUNDLE, CKPTS, GRID, Surface, zbundle  # noqa: E402

N_DEPLOY = 1475092
FR = {"1.00": ("M_s{}", 1.0), "0.75": ("f75_s{}", 0.75), "0.50": ("f50_s{}", 0.50)}
S = Surface()
R = {"exp_id": "exp_114", "stage": "s4_scale", "n_deploy": N_DEPLOY,
     "질문": "최적 스텝이 학습 행 수에 비례하는가(에폭 법칙) 고정인가(스텝 법칙)"}


def peak_log(ks, gs):
    """상위 5점에 log(step) 2차 적합 -> 연속 꼭지점."""
    o = np.argsort(gs)[::-1][:5]
    x, yv = np.log(np.array(ks, float)[o]), np.array(gs)[o]
    a, b_, c = np.polyfit(x, yv, 2)
    if a >= 0:
        return float(ks[int(np.argmax(gs))]), None
    return float(np.exp(-b_ / (2 * a))), {"a": float(a), "b": float(b_), "c": float(c)}


out = {}
for nm, (pat, fr) in FR.items():
    fit = json.load(open(f"{B}/out/fits/{pat.format(42)}.json"))
    n_tr = fit["n_train"]
    tagf = lambda s, p=pat: p.format(s)
    prof_free, prof_w40, alone = [], [], []
    for k in CKPTS:
        zd = zbundle(tagf, 0, k)
        cur, best = S.curve(zd)
        prof_free.append(best["gain"])
        prof_w40.append(cur[0.40])
        alone.append(S.sc(zd, "rc"))
    pf, qf = peak_log(CKPTS, prof_free)
    pw, qw = peak_log(CKPTS, prof_w40)
    pa, _ = peak_log(CKPTS, alone)
    wat = [max(S.curve(zbundle(tagf, 0, k))[0].items(), key=lambda kv: kv[1])[0]
           for k in [CKPTS[int(np.argmin(np.abs(np.array(CKPTS) - pf)))]]][0]
    # 시드별 흩어짐
    per_seed = []
    for s in BUNDLE[0]:
        gs = [S.curve(np.load(f"{B}/out/preds/z_{tagf(s)}__k{k}.npy"))[1]["gain"]
              for k in CKPTS]
        per_seed.append(peak_log(CKPTS, gs)[0])
    out[nm] = {"frac": fr, "n_train": n_tr,
               "steps_per_epoch": fit["steps_per_epoch"],
               "gain_profile_free_w": dict(zip(map(str, CKPTS), prof_free)),
               "gain_profile_w0.40": dict(zip(map(str, CKPTS), prof_w40)),
               "alone_profile": dict(zip(map(str, CKPTS), alone)),
               "kstar_free_w": pf, "kstar_w0.40": pw, "kstar_alone": pa,
               "argmax_w_at_peak": wat,
               "kstar_per_seed": per_seed,
               "kstar_per_seed_sd": float(np.std(per_seed, ddof=1)),
               "epochs_at_kstar": pf / fit["steps_per_epoch"]}
    print(nm, n_tr, "k*free", round(pf, 1), "k*w40", round(pw, 1),
          "ep", round(pf / fit["steps_per_epoch"], 2), flush=True)

R["by_frac"] = out
ns = np.array([out[k]["n_train"] for k in FR], float)
for key in ("kstar_free_w", "kstar_w0.40", "kstar_alone"):
    ks = np.array([out[k][key] for k in FR], float)
    p, a = np.polyfit(np.log(ns), np.log(ks), 1)
    pred = float(np.exp(a) * N_DEPLOY ** p)
    R[f"law_{key}"] = {
        "p_exponent": float(p), "intercept": float(a),
        "r2": float(1 - np.sum((np.log(ks) - (p * np.log(ns) + a)) ** 2)
                    / max(np.sum((np.log(ks) - np.log(ks).mean()) ** 2), 1e-30)),
        "kstar_at_n_deploy": pred,
        "kstar_at_n_oot": float(out["1.00"][key]),
        "해석": ("에폭 법칙(p≈1)" if p > 0.66 else
                "스텝 법칙(p≈0)" if p < 0.33 else "중간 (p 0.33~0.66)"),
        "OOT_상_of_deploy_2160": float(out["1.00"][key] * (2160.0 / pred))
        if pred > 0 else None}
# 시드별 흩어짐으로 본 p 의 불확실성 (묶음0 5시드 각각으로 p 를 다시 적합)
ps = []
for i in range(5):
    ks = np.array([out[k]["kstar_per_seed"][i] for k in FR], float)
    ps.append(float(np.polyfit(np.log(ns), np.log(ks), 1)[0]))
R["p_per_seed"] = {"values": ps, "mean": float(np.mean(ps)),
                   "sd_ddof1": float(np.std(ps, ddof=1))}
json.dump(R, open(f"{B}/out/s4_scale.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "by_frac"},
                 ensure_ascii=False, indent=1))

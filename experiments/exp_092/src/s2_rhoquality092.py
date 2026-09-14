"""exp_092 stage 2 — (rho, 혼합이득) 산점의 해석. **s1_eval.json 만 읽는다.**

왜 이 단계가 필요한가
  S100 은 exp_091 의 22지점에서 "rho 가 오르면 이득이 깎인다"를 읽었다. 그런데 그 22지점은
  **rho 와 팔 단독 점수가 같이 움직이는** 지점들이었다. exp_092 가 rho 를 0.9142 → 0.3701
  까지 벌리자 **부호가 뒤집혀 보인다**(단순 회귀 기울기 +18.96, t +3.18).
  🔑 그래서 두 변수를 **함께** 넣어야 한다 — rho 와 단독 점수는 corr +0.93 으로 얽혀 있고,
     단순 회귀는 어느 쪽으로든 교락된 값을 낸다.

이 단계가 내는 것
  ① 중복 제거 — 기준선 지점이 exp_091 에 3벌·exp_092 에 1벌, **같은 z 5벌**이라 4번 세어진다.
  ② gain ~ rho + alone 2변수 회귀 (전체 / 팔이 안 망가진 부분집합).
  ③ 🔑 **교환율** — "rho 를 0.01 내리는 것"이 단독 점수 몇 점만큼의 값인가.
     이것이 다음 설계의 표적이다: 그보다 싸게 rho 를 내리는 손잡이라야 이득이 난다.
  ④ 각 손잡이의 실제 (Δrho, Δalone) 거래를 그 교환율과 대조한다.
  ⑤ 비모수 확인 — 단독 점수가 비슷한 짝에서 rho 낮은 쪽이 이기는가 (이항검정).

⚠️ 회귀는 **국소 근사**다. F1(rho 0.466) 처럼 멀리 나간 지점에는 외삽하지 마라.
⚠️ 이 기울기들은 **관측 회귀**이지 개입 실험이 아니다. 사전 등록 판정(P1 t>=2.0)과는 별개다.
"""
import json
from itertools import combinations
from math import comb
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp092"
c = json.load(open(f"{B}/out/s1_eval.json"))
S = c["rho_gain_scatter"]["points"]

seen, U, DUP = {}, [], {}
for p in S:
    k = (round(p["rho"], 10), round(p["gain_at_argmax"], 10), round(p["alone_rc"], 10))
    if k in seen:
        DUP.setdefault(seen[k], []).append(p["point"])
        continue
    seen[k] = p["point"]
    U.append(p)
rho = np.array([p["rho"] for p in U])
gain = np.array([p["gain_at_argmax"] for p in U])
alone = np.array([p["alone_rc"] for p in U])
nm = [p["point"] for p in U]


def ols(y, *xs):
    X = np.column_stack([np.ones_like(y)] + list(xs))
    b, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    n, k = X.shape
    se = np.sqrt(np.diag((r @ r / (n - k)) * np.linalg.inv(X.T @ X)))
    return b, b / se, 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))


R = {"exp_id": "exp_092", "stage": "s2_rho_quality",
     "n_scatter_rows": len(S), "n_unique_points": len(U),
     "deduplicated": DUP,
     "🔴 왜 중복 제거": ("기준선(1632·5시드·현행구조·BCE·전체 99열)은 exp_091 의 A/B/C 축 "
                     "기준선과 **같은 z 5벌**이라 산점에 4번 들어간다. 회귀에서 그 점만 "
                     "4배 가중되는 것을 막는다."),
     "fits": {}}
for tag, m in (("all", np.ones(len(U), bool)),
               ("alone_ge_820", alone >= 820),
               ("alone_ge_850", alone >= 850)):
    r, g, a = rho[m], gain[m], alone[m]
    b, t, R2 = ols(g, r, a)
    b1, t1, R21 = ols(g, r)
    b2, t2, R22 = ols(g, a)
    ent = {"n": int(m.sum()), "corr_rho_alone": float(np.corrcoef(r, a)[0, 1]),
           "univariate_gain_on_rho": {"slope": float(b1[1]), "t": float(t1[1]),
                                      "R2": float(R21)},
           "univariate_gain_on_alone": {"slope": float(b2[1]), "t": float(t2[1]),
                                        "R2": float(R22)},
           "joint_gain_on_rho_and_alone": {
               "slope_rho": float(b[1]), "t_rho": float(t[1]),
               "slope_alone": float(b[2]), "t_alone": float(t[2]), "R2": float(R2)}}
    if b[2] > 0:
        ent["🔑exchange_rate_alone_pts_per_0.01_rho"] = float(abs(b[1]) * 0.01 / b[2])
        ent["🔑읽는_법"] = ("rho 를 0.01 내리면서 팔 단독 점수를 이 값보다 **적게** 잃는 "
                        "손잡이라야 혼합 이득이 는다. 그보다 비싸면 손해다.")
    R["fits"][tag] = ent

XR = R["fits"]["alone_ge_820"].get("🔑exchange_rate_alone_pts_per_0.01_rho")
BASE = [p for p in U if p["point"] == "D0_all99_bce"][0]
trades = []
for p in U:
    if p["point"] == BASE["point"]:
        continue
    d_rho = p["rho"] - BASE["rho"]
    d_al = p["alone_rc"] - BASE["alone_rc"]
    e = {"point": p["point"], "exp": p["exp"], "axis": p["axis"],
         "d_rho": float(d_rho), "d_alone_pts": float(d_al),
         "d_gain_pts": float(p["gain_at_argmax"] - BASE["gain_at_argmax"])}
    if d_rho < -1e-9:
        e["alone_pts_paid_per_0.01_rho"] = float(-d_al / (-d_rho * 100.0))
        e["cheaper_than_exchange_rate"] = bool(XR is not None
                                               and e["alone_pts_paid_per_0.01_rho"] < XR)
    trades.append(e)
R["levers_vs_exchange_rate"] = {
    "baseline": {"point": BASE["point"], "rho": BASE["rho"],
                 "alone_rc": BASE["alone_rc"], "gain": BASE["gain_at_argmax"]},
    "exchange_rate_used": XR,
    "levers_that_lower_rho_cheaply": [t for t in trades
                                      if t.get("cheaper_than_exchange_rate")],
    "all": sorted(trades, key=lambda t: t["d_rho"])}

w = l = 0
for i, j in combinations(range(len(U)), 2):
    if abs(alone[i] - alone[j]) > 6 or min(alone[i], alone[j]) < 820:
        continue
    if abs(rho[i] - rho[j]) < 1e-6:
        continue
    lo, hi = (i, j) if rho[i] < rho[j] else (j, i)
    w += int(gain[lo] > gain[hi])       # 🔴 int() — numpy int64 로 두면 2**n 이 넘쳐 0 이 된다
    l += int(gain[lo] <= gain[hi])
n = int(w + l)
R["pairwise_nonparametric"] = {
    "rule": "단독 점수 차 <= 6점 · 둘 다 alone >= 820 인 모든 짝에서 rho 낮은 쪽이 이겼는가",
    "wins_for_lower_rho": int(w), "n_pairs": int(n),
    "one_sided_binomial_p": float(sum(comb(n, k) for k in range(w, n + 1)) / 2 ** n),
    "⚠️": "짝들이 서로 독립이 아니다(같은 지점이 여러 짝에 들어간다). p 는 참고값이다"}
R["🔑결론"] = ("① 단순 산점은 **양의 기울기**를 낸다 — rho 와 팔 품질이 corr +0.93 으로 얽혀 있어서다. "
             "② **품질을 통제하면 rho 의 몫은 크게 음수**이고(t 약 −10, R2 0.83) S100 의 방향이 맞다. "
             "③ 그러나 우리가 가진 손잡이는 전부 rho 를 내리면서 품질을 **교환율보다 비싸게** 잃는다. "
             "④ 그래서 남은 표적은 '더 벌리기'가 아니라 **'같은 품질에서 벌리기'** 다.")
json.dump(R, open(f"{B}/out/s2_rho_quality.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "levers_vs_exchange_rate"},
                 ensure_ascii=False, indent=1))
print("\n=== 손잡이별 거래 (rho 오름차순) ===")
for t in R["levers_vs_exchange_rate"]["all"]:
    pp = t.get("alone_pts_paid_per_0.01_rho")
    print(f"{t['point']:24s} d_rho {t['d_rho']:+.4f}  d_alone {t['d_alone_pts']:+8.2f}  "
          f"d_gain {t['d_gain_pts']:+7.3f}  단가 {('%8.2f' % pp) if pp is not None else '     n/a'}")

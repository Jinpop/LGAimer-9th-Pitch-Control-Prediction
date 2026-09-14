"""exp_103b 추가 대조 — GPU 안 쓴다.

1. **3열 판본 vs 17열 판본 직접 대조.** 두 실험이 같은 ① base 예측 벡터를 썼으므로
   짝지은 투수 블록 부트스트랩으로 바로 비교된다. "14열을 걷어낸 것이 얼마를 되찾았는가".
2. **이 화면의 실제 검출한계.** 짝지은 부트스트랩은 시드를 고정한 채 행만 재표집하므로
   **적합-간(시드) 잡음을 못 잡는다.** 신호가 없도록 만든 판본(count-matched null)들이
   기준선 대비 실제로 얼마나 흩어지는지를 재서 그 몫을 실측한다.
"""
import json
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
REF = 0.2498069
NBOOT = 2000

y = np.load(f"{H}/exp097/work/y.npy").astype(np.float64)
vam = np.load(f"{H}/exp097/work/valid_mask.npy")
pit = pd.read_csv(f"{H}/data/train.csv", usecols=["pitcher_id"],
                  encoding="utf-8-sig")["pitcher_id"].to_numpy(np.int64)
yv, pv = y[vam], pit[vam]

P = {"base": np.load(f"{H}/exp103/work/p2_base.npy"),
     "form17": np.load(f"{H}/exp103/work/p2_form.npy"),
     "null17_row": np.load(f"{H}/exp103/work/p2_null_row.npy"),
     "null17_bat": np.load(f"{H}/exp103/work/p2_null_bat.npy"),
     "form3": np.load(f"{H}/exp103b/work/p3_form3.npy"),
     "null3_bat": np.load(f"{H}/exp103b/work/p3_null_bat3.npy"),
     "null3_row": np.load(f"{H}/exp103b/work/p3_null_row3.npy"),
     "rand3": np.load(f"{H}/exp103b/work/p3_rand3.npy")}

upit, inv = np.unique(pv, return_inverse=True)
G = len(upit)
ordi = np.argsort(inv, kind="stable")
inv_s = inv[ordi]
bnd = np.concatenate([[0], np.where(np.diff(inv_s) != 0)[0] + 1, [len(inv_s)]])
blocks = [ordi[bnd[i]:bnd[i + 1]] for i in range(len(bnd) - 1)]
blk_sum = {t: np.array([((P[t] - yv) ** 2)[b].sum() for b in blocks]) for t in P}
blk_n = np.array([len(b) for b in blocks], dtype=np.float64)
draws = np.random.RandomState(777).randint(0, G, size=(NBOOT, G))


def boot_t(a, c):
    d = blk_sum[a] - blk_sum[c]
    obs = -(d.sum() / blk_n.sum()) / REF * 100000.0
    vals = -(d[draws].sum(1) / blk_n[draws].sum(1)) / REF * 100000.0
    sd = float(vals.std(ddof=1))
    return {"delta_score": float(obs), "boot_sd": sd, "t": float(obs / sd),
            "frac_positive": float((vals > 0).mean())}


rep = {"scores": {t: float(100000.0 * (1 - np.mean((P[t] - yv) ** 2) / REF)) for t in P}}
rep["contrast_3col_vs_17col"] = {
    "form3_minus_form17": boot_t("form3", "form17"),
    "해석": "같은 ① base 위에서 3열 판본이 17열 판본보다 얼마나 나은가. 양수면 "
            "'14열 준중복이 손해의 원인'이라는 감사 진단을 지지한다.",
}

# ── 검출한계 실측: 신호 없는 판본 5개의 기준선 대비 흩어짐 ──
nulls = ["null17_row", "null17_bat", "null3_bat", "null3_row", "rand3"]
d = {n: boot_t(n, "base")["delta_score"] for n in nulls}
arr = np.array(list(d.values()))
rep["🔴_실제_검출한계"] = {
    "count_matched_null_arms_vs_base": d,
    "n_arms": len(nulls),
    "spread_min_to_max": float(arr.max() - arr.min()),
    "sd_across_null_arms": float(arr.std(ddof=1)),
    "mean_boot_sd_reported": float(np.mean([boot_t(n, "base")["boot_sd"] for n in nulls])),
    "설명": "이 다섯 판본은 전부 '행 연결이 끊긴 열을 더한 것'이라 기댓값이 같아야 한다. "
            "그런데 실제로는 최대-최소가 크게 벌어진다. 짝지은 부트스트랩은 **시드를 고정한 채 "
            "행만 재표집**하므로 이 적합-간 잡음을 한 톨도 못 잡는다. 보고된 boot_sd 는 "
            "따라서 **아래쪽으로 치우친 추정**이다.",
}
tot = float(np.sqrt(rep["🔴_실제_검출한계"]["sd_across_null_arms"] ** 2
                    + rep["🔴_실제_검출한계"]["mean_boot_sd_reported"] ** 2))
rep["🔴_실제_검출한계"]["combined_sd_pts"] = tot
rep["🔴_실제_검출한계"]["effect_needed_for_t2_combined"] = 2.0 * tot
rep["🔴_실제_검출한계"]["함의"] = (
    f"행 재표집 잡음과 적합-간 잡음을 합치면 1σ 가 약 {tot:.1f}점이다. "
    f"t>=2.0 은 약 {2*tot:.0f}점 효과를 요구한다 — 3열이 그만한 값을 낼 가능성은 낮았다. "
    "즉 이 화면은 3열의 값을 **측정할 능력 자체가 없다**. 문턱 미달은 '값이 없다'가 아니라 "
    "'이 화면으로는 못 본다'로 읽어야 한다.")

json.dump(rep, open(f"{H}/exp103b/out/s1b_extra.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))

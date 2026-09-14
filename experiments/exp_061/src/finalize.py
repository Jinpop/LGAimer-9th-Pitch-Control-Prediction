"""exp_061 — metrics_partial + corr + verify 를 표준 metrics.json 으로 합치고
파생 비교값을 계산한다. 여기서 단정하는 것은 이 실험이 측정한 값뿐이다.

파생값 셋:
 1) 축 분해 — 16셀을 폭/dropout/max_lr 축으로 갈라 어느 축이 점수를 움직였는가.
    (정지점은 '용량 축' 이름으로 등록됐지만 격자는 lr·dropout 도 함께 훑는다.
     낙점이 lr 로 이긴 것이라면 '폭이 대역을 바꿨다'고 쓰면 안 된다.)
 2) 시드 이득 두 형태 — 앙상블−시드평균(S_seed) 과 앙상블−seed42(팀원 대조 형태).
 3) AA 팔 대비 차 — 눈금 두 가지를 나란히. AA 장부값 773.21454 는
    **시드별 Brier 평균** 규약이고, 앙상블끼리 견주려면 평균예측 Brier 끼리 봐야 한다.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nnlib as L  # noqa: E402

OUT = L.OUT
M = json.load(open(f"{OUT}/metrics_partial.json"))
for nm, fn in (("correlations", "corr.json"), ("stage4_verify", "verify_deploy.json")):
    p = f"{OUT}/{fn}"
    if os.path.exists(p):
        M[nm] = json.load(open(p))

# ── 1) 축 분해 ────────────────────────────────────────────────────────────────
cells = M["stage1"]["cells"]
by = {c["tag"]: c["valid_score"] for c in cells}
ax = {"by_width_best": {}, "by_dropout_best": {}, "by_max_lr_best": {},
      "table": [{"widths": c["widths"], "dropout": c["dropout"],
                 "max_lr": c["max_lr"], "n_params": c["n_params"],
                 "best_epoch": c["best_epoch"], "brier": c["valid_brier"],
                 "score": c["valid_score"], "fit_min": round(c["fit_wall_min"], 2)}
                for c in cells]}
for key, fld in (("by_width_best", "widths"), ("by_dropout_best", "dropout"),
                 ("by_max_lr_best", "max_lr")):
    for c in cells:
        k = str(c[fld])
        ax[key][k] = max(ax[key].get(k, -1e9), c["valid_score"])
PIL = M["prereg"]["pilot"]
pil_cell = [c for c in cells if c["widths"] == PIL["widths"]
            and c["dropout"] == PIL["dropout"] and c["max_lr"] == PIL["max_lr"]]
ax["pilot_cell_in_grid"] = pil_cell[0] if pil_cell else None
# 파일럿 폭을 고정하고 lr/dropout 만 바꾼 셀들 = '용량이 아닌 축'의 몫
ax["pilot_width_only_lr_dropout"] = {
    f"do{c['dropout']}_lr{c['max_lr']}": c["valid_score"]
    for c in cells if c["widths"] == PIL["widths"]}
# 낙점과 파일럿 사이에 실제로 달라진 것
pk = M["stage1_decision"]["pick"]
ax["pick_vs_pilot_changed_vars"] = [
    n for n, a, b in (("widths", pk["widths"], PIL["widths"]),
                      ("dropout", pk["dropout"], PIL["dropout"]),
                      ("max_lr", pk["max_lr"], PIL["max_lr"])) if a != b]
M["stage1_axis_decomposition"] = ax

# ── 1b) 용량 축 단독 판정 (지휘자 2026-08-20 지시) ────────────────────────────
# 사전 등록 문턱 "+10점"은 *용량 축*을 닫으려고 등록된 것인데 격자에는 lr·dropout 도
# 함께 들어 있다. 그래서 판정을 둘로 가른다:
#   (a) 용량 축 = max_lr·dropout 을 고정한 채 폭만 비교. 파일럿 폭 대비 +10 이상인
#       조가 하나라도 있어야 '폭이 대역을 바꿨다'고 말할 수 있다.
#   (b) 낙점 구성 = 16셀 전체 최선 (축 무관). stage 2 로 보내는 것은 이쪽이다.
PW = tuple(PIL["widths"])
capg = {}
for c in cells:
    capg.setdefault((c["dropout"], c["max_lr"]), []).append(c)
cap = {"groups": {}, "margin": M["prereg"]["stage1_margin"]}
for (do, lr), grp in sorted(capg.items()):
    base = [c for c in grp if tuple(c["widths"]) == PW]
    others = [c for c in grp if tuple(c["widths"]) != PW]
    if not base or not others:
        continue
    b = base[0]
    o = max(others, key=lambda c: c["valid_score"])
    cap["groups"][f"do{do}_lr{lr}"] = {
        "pilot_width_score": b["valid_score"],
        "best_other_width": o["widths"], "best_other_score": o["valid_score"],
        "delta_width_only": o["valid_score"] - b["valid_score"],
        "n_params_pilot": b["n_params"], "n_params_best_other": o["n_params"]}
deltas = [v["delta_width_only"] for v in cap["groups"].values()]
cap["max_delta_width_only"] = max(deltas) if deltas else None
cap["capacity_axis_open"] = bool(deltas and max(deltas) >= cap["margin"])
cap["verdict"] = ("폭이 대역을 바꾼다" if cap["capacity_axis_open"] else
                  "용량 축 닫힘 — max_lr·dropout 을 고정하면 폭 확대가 문턱을 못 넘는다")
cap["note"] = ("이 판정은 폭만 바뀐 대조(같은 dropout·같은 max_lr)에서 나온 것이라 "
               "폭에 귀속 가능하다. 반면 stage1_decision.pick 은 축이 섞여 있어 "
               "어느 하나에도 귀속하지 않는다.")
M["stage1_capacity_verdict"] = cap

# 🔴 키 이름 정정 — run_stages.py 의 stage1_decision["capacity_axis_open"] 은
# 실제로는 "전체 최선이 파일럿 대비 +10 을 넘었는가"(= 낙점이 파일럿과 다른가)이지
# 용량 축 판정이 아니다. 용량 축의 정본은 위 stage1_capacity_verdict 다.
M["stage1_decision"]["pick_differs_from_pilot"] = \
    M["stage1_decision"].pop("capacity_axis_open")
M["stage1_decision"]["_key_rename_note"] = (
    "구 키 capacity_axis_open -> pick_differs_from_pilot 로 정정. "
    "용량 축 판정은 stage1_capacity_verdict.capacity_axis_open 이 정본이다 "
    "(그쪽만 폭만 바뀐 대조에서 나온다).")

# ── 2) 시드 이득 두 형태 ──────────────────────────────────────────────────────
def seed_forms(block):
    f = {r["seed"]: r["valid_score"] for r in block["fits"]}
    o = {"per_seed_score": f,
         "S_seed_ensemble_minus_mean": block["S_seed"],
         "ensemble_minus_seed42": block["seedavg_logit_score"] - f.get(42, float("nan")),
         "per_seed_sd": block["per_seed_score_sd"]}
    return o


for st, nm in (("stage2", "pick"), ("stage3", "pick")):
    if st in M and nm in M[st]:
        M[st]["seed_gain_forms"] = seed_forms(M[st][nm])

# ── 3) AA 팔 대비 ─────────────────────────────────────────────────────────────
AA_MEANBRIER = 0.24787535671986385      # exp_048 A1@1400 3시드 mean-of-briers (장부값)
AA_AVGPRED_BRIER = 0.24780874118119800  # 같은 3시드의 평균예측 Brier (이 실험이 재계산)
s2 = M["stage2"]["pick"]
M["vs_AA_arm_both_conventions"] = {
    "convention_A_mean_of_per_seed_brier": {
        "AA_67col": {"brier": AA_MEANBRIER, "score": L.conv(AA_MEANBRIER)},
        "NN99": {"brier": s2["per_seed_brier_mean"],
                 "score": s2["per_seed_score_mean"]},
        "delta_score": s2["per_seed_score_mean"] - L.conv(AA_MEANBRIER)},
    "convention_B_brier_of_averaged_pred": {
        "AA_67col_prob_avg": {"brier": AA_AVGPRED_BRIER,
                              "score": L.conv(AA_AVGPRED_BRIER)},
        "NN99_logit_avg": {"brier": s2["seedavg_logit_brier"],
                           "score": s2["seedavg_logit_score"]},
        "delta_score": s2["seedavg_logit_score"] - L.conv(AA_AVGPRED_BRIER)},
    "note": "AA 는 3시드·확률평균, NN 은 4시드·로짓평균 — 시드 수와 평균 공간이 다르다. "
            "이 차는 '이 두 산출물의 차'이지 아키텍처 단독 효과가 아니다."}
M["conversion_note"] = ("점수 = 100000*(1 - Brier/0.2498069). 6자리 Brier 에서 "
                        "환산하면 ±0.20 불확실 — 뒤 네 자리는 측정값이 아니다.")
json.dump(M, open(f"{OUT}/metrics.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({"stage1_decision": M["stage1_decision"],
                  "capacity_verdict": M["stage1_capacity_verdict"],
                  "axis": {k: v for k, v in ax.items() if k != "table"},
                  "vs_AA": M["vs_AA_arm_both_conventions"],
                  "gate": M.get("stage3", {}).get("gate")},
                 indent=1, ensure_ascii=False))

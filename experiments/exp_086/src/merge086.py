"""로컬 산출물 조립 — 서버 out/*.json 을 experiments/exp_086/metrics.json 으로 합친다."""
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = f"{D}/out"
m = json.load(open(f"{O}/metrics_core.json"))
m["derivation_from_exp085"] = json.load(open(f"{O}/derive.json"))
m["reuse_sha_check"] = json.load(open(f"{O}/reuse_check.json"))
m["fit_curves"] = {}
for a in ("F1_FM", "F2_EMB"):
    for s in (7, 2024, 101):
        m["fit_curves"][f"{a}_s{s}"] = json.load(open(f"{O}/{a}_s{s}_r2.json"))["curve"]

sb = m["seed_expansion"]
m["fold_2022"] = None
m["fold_2023"] = None
m["fold_2024"] = {
    "note": "이 실험의 유일한 면. config 가 fold_2024 단독으로 지정했다",
    "n_valid_all": m["surface"]["n_valid_all"],
    "arms_seedavg4": {a: {"brier": sb[a]["seedavg_brier"],
                          "score": sb[a]["seedavg_score"],
                          "bss_vs_ref_0.2498069":
                              1.0 - sb[a]["seedavg_brier"] / 0.2498069}
                      for a in ("F1_FM", "F2_EMB")},
    "arms_per_seed": {a: {str(s): sb[a]["per_seed"][str(s)]["score"]
                          for s in sb[a]["seeds"]}
                      for a in ("F1_FM", "F2_EMB")},
    "reference_arms": {k: {"brier": v["brier"], "score": v["score"]}
                       for k, v in m["standalone_out_of_time"].items()
                       if not k.startswith(("F1_", "F2_"))},
    "verdict_brier_gain": {
        a: {"gain_pts": m["VERDICT_primary"]["arms"][a]["gain_pts"],
            "t": m["VERDICT_primary"]["arms"][a]["t"],
            "pass": m["VERDICT_primary"]["arms"][a]["pass"]}
        for a in ("F1_FM", "F2_EMB")},
}
newfits = [sb[a]["per_seed"][str(s)]["fit_wall_min"]
           for a in ("F1_FM", "F2_EMB") for s in (7, 2024, 101)]
m["train_time_min"] = {
    "new_fits_6": newfits, "new_fits_6_sum": round(sum(newfits), 3),
    "per_fit_mean": round(sum(newfits) / len(newfits), 3),
    "note": "시드 42 는 exp_085 r2 재사용(재학습 0). 두 팔은 GPU 2장 동시, "
            "팔 안에서는 시드 3개 순차",
}
m["inference_time_sec"] = {
    "basis": "exp_085 실측 승계 — 구조·가중 형상이 같고 이 실험은 시드만 바꿨다",
    "F1_FM_1seed": 0.092, "F2_EMB_1seed": 0.783,
    "F1_FM_4seed_est": round(0.092 * 4, 3), "F2_EMB_4seed_est": round(0.783 * 4, 3),
    "note": "4시드 평균은 같은 구조를 4번 통과시키므로 선형 외삽. 실측 아님 — "
            "배포 경로로 갈 때 단독 측정 필요(S24: 추론 시간은 현재 제약이 아니다)",
}
json.dump(m, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote", f"{D}/metrics.json", os.path.getsize(f"{D}/metrics.json"), "bytes")

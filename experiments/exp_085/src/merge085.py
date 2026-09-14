"""로컬 산출물 조립 — 서버 out/*.json 을 experiments/exp_085/metrics.json 으로 합친다."""
import json
import os

D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O = f"{D}/out"
m = json.load(open(f"{O}/metrics.json"))
m["addendum_blend_same_variance_class"] = json.load(open(f"{O}/metrics_addendum.json"))
m["inference_time"] = json.load(open(f"{O}/infer_time.json"))
m["prep_full"] = json.load(open(f"{O}/prep.json"))
m["fit_curves"] = {a: json.load(open(f"{O}/{a}_s42_r2.json"))["curve"]
                   for a in ("F1_FM", "F2_EMB")}

f = m["fits"]
m["fold_2022"] = None
m["fold_2023"] = None
m["fold_2024"] = {
    "note": "이 실험의 유일한 면. config 가 fold_2024 단독으로 지정했다",
    "n_valid_all": m["surface"]["n_valid_all"],
    "arms": {a: {"brier": f[a]["best_valid_brier"],
                 "score": f[a]["best_valid_score"],
                 "bss_vs_ref_0.2498069": 1.0 - f[a]["best_valid_brier"] / 0.2498069}
             for a in ("F1_FM", "F2_EMB")},
    "reference_arms": {k: {"brier": v["brier"], "score": v["score"]}
                       for k, v in m["standalone_out_of_time"].items()
                       if not k.startswith(("F1_", "F2_"))},
}
m["train_time_min"] = {
    "F1_FM_r2": f["F1_FM"]["fit_wall_min"], "F2_EMB_r2": f["F2_EMB"]["fit_wall_min"],
    "prep": round(m["prep_full"]["wall_sec"] / 60.0, 3),
    "note": "두 팔은 GPU 2장에 동시 실행(경합 계수 미분리). r1(구현 오류 실행) 0.497 / 0.773분",
}
m["inference_time_sec"] = {"F1_FM": m["inference_time"]["F1_FM"]["cpu_infer_sec"],
                           "F2_EMB": m["inference_time"]["F2_EMB"]["cpu_infer_sec"],
                           "basis": "CPU threads 6, 253,507행 1회 통과"}
json.dump(m, open(f"{D}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("wrote", f"{D}/metrics.json", os.path.getsize(f"{D}/metrics.json"), "bytes")

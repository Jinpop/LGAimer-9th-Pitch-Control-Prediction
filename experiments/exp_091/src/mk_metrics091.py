"""exp_091 — 로컬 metrics.json 조립 (서버에서 회수한 산출물만 읽는다)."""
import json, os

E = "<PROJECT_ROOT>/experiments/exp_091"
O = f"{E}/out"
c = json.load(open(f"{O}/s1_eval.json"))
plan = json.load(open(f"{O}/plan.json"))
try:
    r24 = json.load(open(f"{O}/s3_rule24.json"))
except FileNotFoundError:
    r24 = None
try:
    gate2 = json.load(open(f"{O}/gate_repro_final.json"))
except FileNotFoundError:
    gate2 = None

ORDER = (["A_n05_seeds5", "A_n10_seeds10", "A_n15_seeds15", "A_n20_seeds20"]
         + [k for k in plan["points"] if k.startswith("B_")]
         + [k for k in plan["points"] if k.startswith("C_")])


def row(name):
    v = c["points"][name]
    a, rc = v["abd"], v["rc"]
    pr = v["paired_vs_reference"]
    return {
        "point": name, "axis": v["axis"], "n_seeds": v["n_seeds"],
        "hid": v["hid"], "drop": v["drop"], "opt": v["opt"],
        "n_params": v.get("n_params"),
        "IS_BASELINE": v.get("IS_BASELINE", False),
        "alone_recentred_pts": v["alone_recentred_score"],
        "alone_raw_pts": v["alone_raw_score"],
        "z_sd": v["z_sd"],
        "rho_vs_z_mate_oot": v["rho_vs_z_mate_oot"],
        "within_seed_rho": v["within_seed_rho"],
        "abd_argmax_w": a["argmax_w"],
        "abd_gain_at_argmax": a["gain_at_argmax"],
        "abd_gain_at_w0.20": a["gain_at_w0.20"],
        "abd_gain_at_w0.36": a["gain_at_w0.36"],
        "abd_gain_at_w0.20_sd_matched": a.get("gain_at_w0.20_sd_matched"),
        "abd_gain_at_w0.20_alpha_free": a.get("gain_at_w0.20_alpha_free"),
        "rc_argmax_w": rc["argmax_w"], "rc_gain_at_argmax": rc["gain_at_argmax"],
        "paired_t_P1_own_argmax": pr["P1_own_argmax"].get("t"),
        "paired_t_P2_w0.20": pr["P2_w0.20"].get("t"),
        "paired_t_P3_w0.36": pr["P3_w0.36"].get("t"),
        "boot_t_vs_w0_at_w0.20": v["boot_vs_w0_at_w0.20"]["t"],
        "boot_t_vs_w0_at_argmax": v["boot_vs_w0_at_argmax"].get("t"),
        "single_seed_gain_mean": v["seed_noise"]["single_seed_gain_mean"],
        "single_seed_gain_sd": v["seed_noise"]["single_seed_gain_sd"],
        "jackknife_se_of_mean": v["seed_noise"]["jackknife_se_of_mean"],
    }


V = c["PREREGISTERED_VERDICT"]
M = {
    "exp_id": "exp_091",
    "title": "딥 팔 훑기 2 — 시드 수·구조·정규화. 학습량은 1632스텝 고정",
    "verdict": ("PASS — 통과 지점 있음" if V["PASS"] else
                "FAIL (사전 등록 기각) — 전 지점 P1 t < 2.0. 현행 구조·5시드가 실용 최적"),
    "surface": {
        "name": "exp_089 정직면 (모든 팔 out-of-time), exp_090 과 동일",
        "baseline_w0_rc_pts": c["baseline_w0"]["rc"],
        "baseline_w0_rc_expected": c["baseline_w0"]["rc_expected"],
        "baseline_w0_rc_abs_diff": c["baseline_w0"]["rc_abs_diff"],
        "baseline_w0_abd_pts": c["baseline_w0"]["abd"],
        "z_mate_alone_recentred_pts": c["baseline_w0"]["z_mate_alone_recentred"],
        "K_pts_per_brier": c["K_pts_per_brier"], "ref_brier": c["ref_brier"],
        "n_rows_valid_2024": c["n_rows"], "y_mean": c["y_mean"],
        "primary_convention": "abd (배포 산술: p = expit(alpha*z + beta_skew*z^2 + d))",
        "BO_constants": c["BO_constants"],
        "fixed_total_steps": c["fixed_total_steps"],
    },
    "gates": {
        "baseline_point_reproduction": {
            "PASS": c["gate_baseline_point_PASS"],
            "abs_diff_pts": c["baseline_w0"]["rc_abs_diff"]},
        "reference_point_reproduction": c["gate_reference_reproduction"],
        "code_path_reproduction_vs_exp090": gate2,
        "n_preds_sha_checked": len(c["pred_sha_check"]),
        "preds_sha_all_match": all(v["MATCH"] for v in c["pred_sha_check"].values()),
    },
    "t_sign_convention": c["t_sign_convention"],
    "axis_curves": [row(n) for n in ORDER],
    "PREREGISTERED_VERDICT": V,
    "rule_2_4_permutation": r24,
    "fold_2024": {
        "note": ("이 실험은 단일 모델의 fold Brier 를 내는 실험이 아니다. 비교 기준은 "
                 "exp_089 정직면(모든 팔 out-of-time, valid = season 2024 전체 253,507행)의 "
                 "**혼합 이득(점수 단위)** 이다. fold_2022/2023 은 이 면에 존재하지 않는다."),
        "baseline_pts_abd": c["baseline_w0"]["abd"],
        "baseline_brier_abd": 0.2498069 * (1 - c["baseline_w0"]["abd"] / 100000.0),
        "reference_point": V["reference_point"],
        "reference_gain_at_argmax_pts": V["reference_gain_at_argmax"],
        "best_mix_point": V["S99_best_mix_point"],
        "best_mix_gain_at_argmax_pts": V["S99_best_mix_gain_at_argmax"],
        "best_alone_point": V["S99_best_alone_point"],
        "best_alone_recentred_pts": V["S99_best_alone_recentred_score"],
        "S99_best_mix_and_best_alone_same_point": V["S99_same_point"],
    },
    "seeds": {"S5_inherited": plan["S5"], "S15_new": plan["S15"]},
}
json.dump(M, open(f"{E}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items() if k != "axis_curves"},
                 ensure_ascii=False, indent=1)[:3000])
print(f"\n{len(M['axis_curves'])} points written")

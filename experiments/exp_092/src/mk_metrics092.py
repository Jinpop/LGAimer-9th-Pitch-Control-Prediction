"""exp_092 — 로컬 metrics.json 조립 (서버에서 회수한 산출물만 읽는다)."""
import json, os

E = "<PROJECT_ROOT>/experiments/exp_092"
O = f"{E}/out"
c = json.load(open(f"{O}/s1_eval.json"))
plan = json.load(open(f"{O}/plan.json"))
gF = json.load(open(f"{O}/s0_gateF.json"))
oF = json.load(open(f"{O}/s0_offF.json"))
gate_repro = json.load(open(f"{O}/gate_repro.json"))
s2 = json.load(open(f"{O}/s2_rho_quality.json"))
r24 = {}
for v in ("mask_ins", "offsetF"):
    p = f"{O}/s3_rule24_{v}.json"
    if os.path.exists(p):
        r24[v] = json.load(open(p))

ORDER = [k for k in c["points"] if k.startswith("D0")] \
    + [k for k in c["points"] if k.startswith("D1") or k.startswith("D2")
       or k.startswith("D3") or k.startswith("D4")] \
    + [k for k in c["points"] if k.startswith("D5")] \
    + [k for k in c["points"] if k.startswith("E_")] \
    + [k for k in c["points"] if k.startswith("F")]
assert len(ORDER) == len(c["points"]), (len(ORDER), len(c["points"]))


def row(name):
    v = c["points"][name]
    a, rc = v["abd"], v["rc"]
    pr = v["paired_vs_reference"]
    return {
        "point": name, "axis": v["axis"], "desc": v["desc"],
        "n_masked_num_cols": v["n_masked"], "loss": v["loss"],
        "rows_restrict": v.get("rows_restrict"), "offset_used": v.get("offset_used"),
        "variant": v.get("variant"),
        "IS_BASELINE": v.get("IS_BASELINE", False), "n_seeds": v["n_seeds"],
        "🔑rho_vs_z_mate_oot": v["rho_vs_z_mate_oot"],
        "abd_gain_at_argmax": a["gain_at_argmax"], "abd_argmax_w": a["argmax_w"],
        "abd_gain_at_w0.20": a["gain_at_w0.20"],
        "abd_gain_at_w0.36": a["gain_at_w0.36"],
        "abd_gain_at_w0.20_sd_matched": a.get("gain_at_w0.20_sd_matched"),
        "abd_gain_at_w0.20_alpha_free": a.get("gain_at_w0.20_alpha_free"),
        "alone_recentred_pts": v["alone_recentred_score"],
        "alone_raw_pts": v["alone_raw_score"],
        "z_sd": v["z_sd"], "within_seed_rho": v["within_seed_rho"],
        "rc_argmax_w": rc["argmax_w"], "rc_gain_at_argmax": rc["gain_at_argmax"],
        "paired_t_P1_own_argmax": pr["P1_own_argmax"].get("t"),
        "paired_t_P2_w0.20": pr["P2_w0.20"].get("t"),
        "paired_t_P3_w0.36": pr["P3_w0.36"].get("t"),
        "boot_t_vs_w0_at_argmax": v["boot_vs_w0_at_argmax"].get("t"),
        "single_seed_gain_mean": v["seed_noise"]["single_seed_gain_mean"],
        "single_seed_gain_sd": v["seed_noise"]["single_seed_gain_sd"],
        "jackknife_se_of_mean": v["seed_noise"]["jackknife_se_of_mean"],
    }


V = c["PREREGISTERED_VERDICT"]
M = {
    "exp_id": "exp_092",
    "title": "딥 팔의 rho 를 명시적으로 벌린다 — 피처 부분집합·손실·잔차 타깃",
    "verdict": ("PASS — 통과 지점 있음" if V["PASS"] else
                "FAIL (사전 등록 기각) — 전 지점 P1 t < 2.0"),
    "surface": {
        "name": "exp_089 정직면 (모든 팔 out-of-time), exp_090/091 과 동일",
        "baseline_w0_rc_pts": c["baseline_w0"]["rc"],
        "baseline_w0_rc_expected": c["baseline_w0"]["rc_expected"],
        "baseline_w0_rc_abs_diff": c["baseline_w0"]["rc_abs_diff"],
        "baseline_w0_abd_pts": c["baseline_w0"]["abd"],
        "z_mate_alone_recentred_pts": c["baseline_w0"]["z_mate_alone_recentred"],
        "K_pts_per_brier": c["K_pts_per_brier"], "ref_brier": c["ref_brier"],
        "n_rows_valid_2024": c["n_rows"], "y_mean": c["y_mean"],
        "primary_convention": "abd (배포 산술: p = expit(alpha*z + beta_skew*z^2 + d))",
        "BO_constants": c["BO_constants"], "fixed_total_steps": c["fixed_total_steps"],
    },
    "gates": {
        "baseline_point_reproduction": {
            "PASS": c["gate_baseline_point_PASS"],
            "abs_diff_pts": c["baseline_w0"]["rc_abs_diff"]},
        "reference_point_reproduction": c["gate_reference_reproduction"],
        "code_path_reproduction_vs_exp090": gate_repro,
        "n_preds_sha_checked": len(c["pred_sha_check"]),
        "preds_sha_all_match": all(v["MATCH"] for v in c["pred_sha_check"].values()),
    },
    "t_sign_convention": c["t_sign_convention"],
    "🔑rho_gain_scatter": c["rho_gain_scatter"],
    "🔑rho_vs_arm_quality": s2,
    "D5_null_comparison": c["D5_null_comparison"],
    "axis_curves": [row(n) for n in ORDER],
    "PREREGISTERED_VERDICT": V,
    "F_axis": {"config_named_source_missing": gF, "substitute_built": oF,
               "offset_arm_on_surface": c["offset_arm_on_surface"]},
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
    "seeds": {"S5": plan["S5"]},
    "families_masked": plan["families"],
    "wall_sec_stage1": c["wall_sec"],
}
json.dump(M, open(f"{E}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items()
                  if k not in ("axis_curves", "🔑rho_gain_scatter", "F_axis")},
                 ensure_ascii=False, indent=1)[:4000])
print(f"\n{len(M['axis_curves'])} points written")

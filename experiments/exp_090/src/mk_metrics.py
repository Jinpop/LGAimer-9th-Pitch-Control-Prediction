"""exp_090 — 로컬 metrics.json 조립 (서버에서 회수한 산출물만 읽는다)."""
import json, os

E = "<PROJECT_ROOT>/experiments/exp_090"
O = f"{E}/out"
c = json.load(open(f"{O}/s1_curve.json"))
pl = json.load(open(f"{O}/s1b_plateau.json"))
r24 = json.load(open(f"{O}/s3_rule24.json"))
G = {t: json.load(open(f"{O}/gate_{t}.json")) for t in ("D1632", "D2160", "D1440")}

STEPS = sorted(int(k) for k in c["points"])
curve = []
for n in STEPS:
    v = c["points"][str(n)]
    a, rc = v["abd"], v["rc"]
    sn = pl["seed_noise"][str(n)]
    pv = pl["paired_vs_peak"].get(str(n))
    curve.append({
        "total_steps": n, "epochs_oot23": v["epochs_equiv_oot23"], "source": v["source"],
        "alone_recentred_pts": v["alone_recentred_score"],
        "alone_raw_pts": v["alone_raw_score"],
        "z_sd": v["z_sd"], "rho_vs_z_mate_oot": v["rho_vs_z_mate_oot"],
        "within_seed_rho": v["within_seed_rho"],
        "abd_argmax_w": a["argmax_w"], "abd_gain_at_argmax": a["gain_at_argmax"],
        "abd_t_at_argmax": a["boot_t_at_argmax"],
        "abd_gain_at_w0.20": a["gain_at_w0.20"], "abd_t_at_w0.20": a["boot_t_at_w0.20"],
        "abd_gain_at_w0.20_sd_matched": a["gain_at_w0.20_sd_matched"],
        "abd_gain_at_w0.20_alpha_free": a["gain_at_w0.20_alpha_free"],
        "abd_B_K_mean_delta2": a["B_K_mean_delta2_at_w0.20"],
        "abd_ci95_at_w0.20": a["boot_ci95_at_w0.20"],
        "rc_argmax_w": rc["argmax_w"], "rc_gain_at_argmax": rc["gain_at_argmax"],
        "rc_gain_at_w0.20": rc["gain_at_w0.20"], "rc_t_at_w0.20": rc["boot_t_at_w0.20"],
        "single_seed_gain_mean": sn["single_seed_mean"],
        "single_seed_gain_sd": sn["single_seed_sd"],
        "jackknife_se_of_5seed_mean": sn["jackknife_se_of_mean5"],
        "paired_t_vs_peak_1632": None if pv is None else pv["paired_t"],
    })

M = {
    "exp_id": "exp_090",
    "title": "딥 팔 학습량 훑기 — OOT 정직면에서 고원을 찾고 그 자리로 배포 팔 재학습",
    "verdict": "PASS (사전 등록 통과) — stage 2 수행",
    "surface": {
        "name": "exp_089 정직면 (모든 팔 out-of-time)",
        "baseline_w0_rc_pts": c["baseline_w0"]["rc"],
        "baseline_w0_rc_expected_exp078_080": c["baseline_w0"]["rc_expected"],
        "baseline_w0_rc_abs_diff": c["baseline_w0"]["rc_abs_diff"],
        "baseline_w0_abd_pts": c["baseline_w0"]["abd"],
        "z_mate_alone_recentred_pts": c["baseline_w0"]["z_mate_alone_recentred"],
        "K_pts_per_brier": c["K_pts_per_brier"], "ref_brier": c["ref_brier"],
        "n_rows_valid_2024": c["n_rows"], "y_mean": c["y_mean"],
        "primary_convention": "abd (배포 산술: p = expit(alpha*z + beta_skew*z^2 + d))",
        "BO_constants": c["BO_constants"],
    },
    "gates": {
        "baseline_reproduction": {"PASS": c["baseline_gate_PASS"],
                                  "abs_diff_pts": c["baseline_w0"]["rc_abs_diff"]},
        "code_path_reproduction_vs_exp089": {
            "what": "exp_089 S720_oot23_s42 를 train090.py 로 재적합",
            "z_sha256": "90002c1450b839d596b94e4c0873671270e230a3ab276e7b669136e064184a3b",
            "max_abs_diff": 0.0, "PASS": True},
        "reused_pred_sha_all_match": c["pred_sha_all_match"],
        "n_preds_sha_checked": len(c["pred_sha_check"]),
    },
    "stage1_step_curve": curve,
    "stage1_plateau": {
        "peak_steps": pl["peak_steps"],
        "plateau_members_steps": pl["plateau"]["members"],
        "significantly_below_peak": pl["plateau"]["significantly_below_peak_t>=2"],
        "definition": pl["plateau"]["definition"],
        "deployed_arm_720_paired_t_vs_peak": pl["paired_vs_peak"]["720"]["paired_t"],
        "deployed_recipe_544_paired_t_vs_peak": pl["paired_vs_peak"]["544"]["paired_t"],
    },
    "PREREGISTERED_VERDICT": c["PREREGISTERED_VERDICT"],
    "stage2_deploy": {
        "conversion": c["DEPLOY_STEP_CONVERSION"],
        "vintages": {
            "D1632": {"interpretation": "스텝 맞춤 — OOT 최적 s*=1632 스텝을 그대로",
                      "total_steps": 1632, "epochs_le2024": 1632 / 360,
                      "server_path": "~/SungjinPark/exp090/out/deploy_D1632/"},
            "D2160": {"interpretation": "에폭 맞춤 — s*=6 에폭을 배포 구간의 6 에폭으로",
                      "total_steps": 2160, "epochs_le2024": 6.0,
                      "row_ratio_alternative_steps": 2156.5885026745154,
                      "server_path": "~/SungjinPark/exp090/out/deploy_D2160/"},
            "D1440": {"interpretation": "보수 대안 — 단독 최고점 s*=1088(4 에폭) 에폭 맞춤",
                      "total_steps": 1440, "epochs_le2024": 4.0,
                      "server_path": "~/SungjinPark/exp090/out/deploy_D1440/"},
        },
        "shipped_arm_for_reference": {"total_steps": 720, "epochs_le2024": 2.0},
        "format_gate": {t: {"ALL_PASS": G[t]["ALL_PASS"],
                            "G1_keys_shapes_dtypes": G[t]["G1_keys_shapes_dtypes"]["PASS"],
                            "G2_prep_input_transform": G[t]["G2_prep_input_transform"]["PASS"],
                            "G2_changed_fields": sorted(G[t]["G2_prep_input_transform"]["changed_fields"]),
                            "G3_end_to_end_rc": G[t]["G3_end_to_end"]["rc"],
                            "G3_p_min": G[t]["G3_end_to_end"]["min"],
                            "G3_p_max": G[t]["G3_end_to_end"]["max"],
                            "G3_n_nan": G[t]["G3_end_to_end"]["n_nan"],
                            "G4_max_abs_logit_diff_vs_torch": G[t]["G4_vs_torch_path"]["max_abs_logit_diff"],
                            "G5_permutation_roundtrip_max_abs": G[t]["G5_row_independence"]["max_abs_diff"],
                            "npz_sha256": G[t]["npz_sha256"]} for t in G},
        "rule_2_4_content_separation": {
            t: {"max_abs_delta_on_KEEP110": r24[t]["content_separation_max_abs_delta_on_KEEP"],
                "n_moving": r24[t]["content_separation_n_moving"],
                "PASS": r24[t]["PASS_content_independent"]} for t in G},
        "drift_vs_shipped_deep_arm": {t: r24[t]["vs_shipped_deep_arm"] for t in G},
        "inference_time_sec": {t: r24[t]["inference_sec_253507rows_5seeds_6threads"] for t in G},
    },
    "fold_2024": {
        "note": ("이 실험은 단일 모델의 fold Brier 를 내는 실험이 아니다. 비교 기준은 "
                 "exp_089 정직면(모든 팔 out-of-time, valid = season 2024 전체 253,507행)의 "
                 "**혼합 이득(점수 단위)** 이다. fold_2022/2023 은 이 면에 존재하지 않는다 "
                 "(OOT 납품 벡터가 2024 만 있다)."),
        "baseline_pts_abd": c["baseline_w0"]["abd"],
        "baseline_brier_abd": c["ref_brier"] * (1 - c["baseline_w0"]["abd"] / 100000.0),
        "best_point_steps": pl["peak_steps"],
        "best_point_gain_at_w0.20_pts": pl["gain_at_w0.20"][str(pl["peak_steps"])],
        "deployed_720_gain_at_w0.20_pts": pl["gain_at_w0.20"]["720"],
        "prereg_recipe_544_gain_at_w0.20_pts": pl["gain_at_w0.20"]["544"],
    },
    "limits_carried": [
        "① 정직면 크기는 5.4배 어긋난다 (exp_089 전이비 +0.1866 이 유일하게 변인 통제된 점). "
        "🔴 이 실험의 이득을 LB 점수로 환산하지 마라 — 위치와 부호만 쓴다.",
        "② 꼭지점 위치는 대체로 맞는다 (exp_089: 정직면 스텝맞춤 0.14 vs LB 유도 0.1757).",
        "③ 우리 frame99 와 팀원 프레임이 94열 중 8열 다르다 (i3_anom_*·p6_*, 최대 0.616 sd). "
        "이 실험의 모든 측정에 실려 있다.",
        "④ 복제 게이트의 천장은 0.988034 (팀원 시드 간 rho 0.944428). exp_089 가 0.985651 을 얻었다.",
        "⑤ 이 축의 전이비는 exp_089 에서 -0.57 ~ +1.77 로 하나로 안 나온다 (S93).",
    ],
    "changed_variables_vs_exp089": [
        "학습량(총 옵티마이저 스텝)만 축으로 움직였다. 544/720/816/1088 은 exp_089 산출물을 sha 대조 후 재사용.",
        "LR 스케줄러는 없다 (lr 1e-3 상수, 팀원 prep.json 이 lr 을 단일값으로 기록). "
        "따라서 스텝을 바꿔도 LR 궤적이 안 바뀐다 — '학습량'과 '스케줄 모양'이 안 섞인다.",
        "train090.py 는 train089.py 대비 ① work 를 exp089/work 에서 읽고 ② out 이 exp090 이고 "
        "③ SAVE_NPZ 분기가 추가됐고 ④ 환경변수 이름이 STEPS089→STEPS090 이다. "
        "z 에 안 닿는다는 것은 재현 게이트(sha 동일)로 증명했다.",
        "stage 2 배포 학습의 mu/sd/levels/D/batch 는 팀원 prep.json 그대로 — 배포 팔 대비 변인은 학습량 하나.",
        "부트스트랩 rng 시드·추출 순서를 exp_089 와 같게 유지 → 공유 지점의 t 가 그대로 재현된다.",
    ],
    "server_paths": {
        "root": "~/SungjinPark/exp090/",
        "preds": "~/SungjinPark/exp090/out/preds/",
        "deploy_weights": ["~/SungjinPark/exp090/out/deploy_D1632/",
                           "~/SungjinPark/exp090/out/deploy_D2160/",
                           "~/SungjinPark/exp090/out/deploy_D1440/"],
        "reused_from": "~/SungjinPark/exp089/ (읽기만), ~/SungjinPark/audit_deep/pkg/DEEP/mate (읽기만)",
    },
    "train_time_min": round(1974.2 / 60.0, 2),
    "n_fits": 56,
    "n_fits_breakdown": {"repro_gate": 1, "stage1_new_oot_points": 40,
                         "stage2_deploy": 15, "reused_from_exp089": 20},
}
json.dump(M, open(f"{E}/metrics.json", "w"), ensure_ascii=False, indent=1)
print("written", os.path.getsize(f"{E}/metrics.json"))

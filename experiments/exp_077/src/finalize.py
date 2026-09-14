"""exp_077 산출물 합본 — stage0.json + stage2.json + 적합 타이밍 -> metrics.json."""
import glob
import json
import os
import time

E = "/home/<user>/SungjinPark/exp077"
s0 = json.load(open(f"{E}/out/stage0.json"))
s2 = json.load(open(f"{E}/out/stage2.json"))

fits = {}
tot_fit_min = 0.0
for f in sorted(glob.glob(f"{E}/cache/_cat_*.json")):
    d = json.load(open(f))
    k = f"{d['mask']}_s{d['seed']}_t{d['threads']}"
    fits[k] = {"n_train": d["n_train"], "sec_fit": d["sec_fit"],
               "sec_total": d["sec_total"], "tree_count": d["tree_count"],
               "learn_logloss": d["learn_best_error"], "pred_mean": d["pred_mean"]}
    tot_fit_min += d["sec_fit"] / 60.0

wall = json.load(open(f"{E}/out/wall.json")) if os.path.exists(f"{E}/out/wall.json") else {}

P = s2["primary"]
pri = P["우리12_vs_우리4"]["delta_pts"]
sec = P["팀원4+우리8_vs_팀원4"]["delta_pts"]

M = {
    "exp_id": "exp_077",
    "date": "2026-08-21",
    "type": "시드 확장 측정.  CatBoost 적합 13건 · 제출물 없음 · 상수 변경 0.",
    "base": ("BN (LB 1115.9165956458) 의 배포 산술을 팀원 OOT 정직면"
             "(<=2023 학습 -> 2024 예측 253,507행) 위에서 적용."),
    "질문": "v6 내부 CatBoost 를 4시드 -> 12시드로 늘리면 정직면에서 얼마를 버는가.",

    "fold_2022": None,
    "fold_2023": None,
    "fold_2024": {
        "대상": "정직면 전체 블렌드, v6 내부 cat 시드 수만 변수",
        "brier_v6_4": P["우리12_vs_우리4"]["brier_v6_4"],
        "brier_v6_12": P["우리12_vs_우리4"]["brier_v6_12"],
        "bss_v6_4": P["우리12_vs_우리4"]["score_v6_4"],
        "bss_v6_12": P["우리12_vs_우리4"]["score_v6_12"],
        "delta_pts": pri,
        "n": 253507,
        "환산": "점수 = 100000 × (1 − Brier / 0.2498069).  🔴 S1 — LB 눈금이 아니다."},

    "stage0_gates": {
        "G0a_hyper_PASS": s0["G0a_PASS"],
        "G0a_float32_note": s0["G0a_float32_note"],
        "G0a_per_seed": {k: {kk: v[kk] for kk in
                             ("sha256", "tree_count", "random_seed",
                              "iterations", "learning_rate", "depth",
                              "l2_leaf_reg", "subsample", "border_count",
                              "one_hot_max_size", "boosting_type", "has_time",
                              "cat_feature_indices", "feature_names_match_FE_SP")}
                         for k, v in s0["G0a_hyper"].items()},
        "G0a_param_diff_vs_seed42": s0["G0a_param_diff_vs_seed42"],
        "G0b_features": s0["G0b_features"],
        "G0c_categorical": s0["G0c_categorical"],
        "G0c2_deploy_insample_2024": s0["G0c2_deploy_insample_2024"],
        "G0e_deploy_seed_pair_band": s0["G0e_deploy_seed_pair_band"],
        "G0f_seed_loop": s0["G0f_seed_loop"],
        "mask_counts": s0["mask_counts"],
        "alignment": s0["alignment"],
        "ALL_PASS": s0["ALL_PASS"]},

    "G0d_identity": s2["G0d_identity"],
    "scale_gates": s2["scale_gates"],
    "v6_assembly_identity": s2["v6_assembly_identity"],
    "cat4_reproduction": s2["cat4_reproduction"],
    "seed_curve": s2["seed_curve"],
    "sequential_ours": s2["sequential_ours"],
    "primary": s2["primary"],
    "transfer": s2["transfer"],
    "preregistered_verdict": s2["preregistered_verdict"],

    "fits": fits,
    "train_time_min": tot_fit_min,
    "inference_time_sec": None,
    "wall_clock_min": wall.get("total_min"),
    "wall_clock_breakdown": wall,

    "env": {"python": "3.11.15", "catboost": "1.2.10", "numpy": "1.26.4",
            "pandas": "2.0.3", "sklearn": "1.8.0", "scipy": "1.15.3",
            "threads_per_fit": 6, "concurrent_fits": 3,
            "cpu_quota_cores": 20,
            "note": ("컨테이너 CPU 쿼터가 20코어라 thread_count 6 × 동시 3 = 18 "
                     "로 잡았다.  thread_count 는 실험 내내 6 고정.")},

    "변인_전부": {
        "vs_대조군(v6_4)": "cat 시드 수 4 -> 12.  그 외 전부 동일.",
        "vs_팀원_납품_경로": [
            "thread_count 2(팀원) -> 6(우리)",
            "save_snapshot 사용(팀원) -> 미사용(우리)",
            "feat6.parquet(팀원 환경) -> exp071/cache/frame071.parquet(우리)",
            "allow_writing_files True -> False"],
        "🔴": ("팀원 대비 넷이 다르므로 우리 4시드 재현본은 바이트 동일이 "
               "기대 대상이 아니다(S17).  주지표는 우리 12 vs 우리 4 — "
               "그 쌍은 시드 수 하나만 다르다.")},

    "편향_명시": {
        "p_q": "1시드 (배포는 3시드) — Q 팔이 배포보다 약하다",
        "p_v6": "NN 자리가 비었다 (배포 지분 0.15).  v6_4·v6_12 양쪽 다 비웠다",
        "off": "팀원 납품 in-season 오프셋, season<2024 앵커",
        "방향": ("두 편향은 v6_4·v6_12 양쪽에 똑같이 걸려 있어 Δ 에서는 "
                 "상당 부분 상쇄된다.  그러나 cat 지분이 실제 배포보다 "
                 "크게 잡히는 쪽(NN 이 빠져 재정규화로 0.68/0.85=0.80)이므로 "
                 "Δ 는 위로 부풀 수 있다.")},
    "판정_금지": "채택·제출을 결정하지 않는다.  산출은 게이트 + 시드 곡선 + 정직면 Δ.",
    "산출물": {"서버": f"{E}/out/", "벡터": s2["outputs"]},
    "generated": time.strftime("%F %T"),
}
M["report"] = json.load(open(f"{E}/out/report.json"), ) if os.path.exists(
    f"{E}/out/report.json") else None
json.dump(M, open(f"{E}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
print(f"metrics.json 작성 · 적합 {len(fits)}건 총 {tot_fit_min:.2f}분 · "
      f"주지표 Δ {pri:+.5f} · 보조 Δ {sec:+.5f}")

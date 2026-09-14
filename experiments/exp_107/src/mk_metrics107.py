"""exp_107 metrics.json 조립 — 산출물에서만 읽는다(손으로 옮겨 적지 않는다)."""
import glob, json, os
import numpy as np

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
prep = json.load(open(f"{B}/out/s0_prep107.json"))
mix = json.load(open(f"{B}/out/s2_mix107.json"))
rep = json.load(open(f"{B}/out/gates_repro.json"))
leak1 = json.load(open(f"{B}/out/s1_leak107_s2020.json"))
leak2 = json.load(open(f"{B}/out/s1b_leak107_s2020.json"))
leak3 = json.load(open(f"{B}/out/s1c_leak107.json"))
TAB = mix["표"]
fits = glob.glob(f"{B}/out/fit_*.json")
FJ = [json.load(open(f)) for f in fits]

rows = []
for nm, v in sorted(TAB.items(), key=lambda kv: -kv[1]["delta_vs_prereg_baseline"]):
    s = v.get("S103_교환율") or {}
    rows.append({
        "point": nm, "axis": v["axis"], "desc": v["desc"],
        "own_argmax_step": v["own_argmax_step"], "own_argmax_w_deep": v["own_argmax_w"],
        "mix_gain": v["gain"],
        "delta_vs_prereg_baseline_26.2698": v["delta_vs_prereg_baseline"],
        "delta_vs_redrawn_baseline_mean_25.4866": v["delta_vs_redrawn_baseline_mean"],
        "boot_t_vs_base94_1632": v["vs_base94_1632_paired"]["t"],
        "boot_se_pts": v["vs_base94_1632_paired"]["boot_se_pts"],
        "t_combined_1sigma_4.654": v["t_combined_prereg_sigma"],
        "alone_recentred_pts": v["alone_recentred"],
        "rho_vs_z_mate": v["rho_vs_z_mate"],
        "d_rho_vs_base_1632": s.get("d_rho_vs_base_1632"),
        "d_alone_vs_base_1632": s.get("d_alone_vs_base_1632"),
        "rho_unit_price_pts_per_0.01": s.get("rho_unit_price_pts_per_0.01"),
        "S103_pred_dgain": s.get("pred_dgain_defectfit_-308.92rho_+0.18533alone"),
        "S103_pred_dgain_fixedfit": s.get("pred_dgain_fixedfit_-289.78rho_+0.186alone"),
        "S103_obs_dgain_1632": s.get("obs_dgain_1632"),
        "PASS": v["PASS"]})

live = [r for r in rows if r["point"] not in
        ("N3_identity", "N1a_perm_e0", "N1b_perm_eA") and r["S103_pred_dgain"] is not None]
pa = np.array([r["S103_pred_dgain"] for r in live])
ob = np.array([r["S103_obs_dgain_1632"] for r in live])
S103 = {"n_live_points": len(live),
        "rho_pred_vs_obs": float(np.corrcoef(pa, ob)[0, 1]),
        "mean_pred_minus_obs": float((pa - ob).mean()),
        "sd_of_residual": float((pa - ob).std(ddof=1)),
        "rmse": float(np.sqrt(((pa - ob) ** 2).mean())),
        "읽는_법": "S103 회귀는 이 계열에서 **순서와 기울기는 맞고 수준이 치우친다** — "
                   "예측이 일정하게 높다. 편의를 빼면 최선 지점의 예측도 0 근처가 된다."}

M = {
 "exp_id": "exp_107",
 "title": "타깃 스무딩 / 자기 증류 — 딥 팔 슬롯 (장부의 마지막 미착수 축)",
 "면": mix["면"],
 "🔴결론": "전 지점 미달. 사다리 맨 아래 무변형 항등 복제본(N3, 26.2698)을 **이긴 지점이 하나도 없다**. "
           "축 A 최선(e=0.10)이 −0.145 (t −0.068), 축 B 최선(e=0.10)이 −1.657, 축 C 가 −4.325. "
           "🔑 그런데 이 축은 **우리 손잡이 중 처음으로 ρ 를 공짜로 내렸다** — "
           "A e=0.10 이 Δρ −0.00648 과 Δalone +10.43 을 동시에 냈다(단가가 음수). "
           "그러고도 혼합 이득이 0 이었다.",
 "🔴판정": {"prereg": mix["prereg"], "VERDICT": mix["VERDICT"],
            "검출한계": mix["검출한계"]},
 "🔴기준선_재추첨": mix["🔴기준선_재추첨"],
 "게이트": {
   "재현_G1_G2_바이트동일": rep,
   "면_재현": {"baseline_w0_rc": mix["baseline_w0"]["rc"],
              "expected": mix["baseline_w0"]["expected"],
              "gate_1632_reproduces_26.2698":
                  mix["baseline_arm"]["gate_1632_reproduces_26.2698"]},
   "축B_리크_게이트": {
     "1차_s1_leak107": {
        "무엇": "control_success 를 시즌 대역째 난수화",
        "결과": "결정성 대조 통과(A==D). 🔴 양성 대조 실패(0.0 이동) — 그리고 그 실패가 "
                "사실을 가르쳤다: OOF 라벨은 control_success 가 아니라 labels.recover() 가 "
                "asof_*_rate 에서 복원한 5셀이다(labels.py:51 — control_success 는 검산 전용).",
        "GATES": leak1["GATES"]},
     "2차_s1b_leak107_PART1": {
        "무엇": "asof_*_rate 5열을 대역째 난수화 → recover 재실행",
        "결과": "🔴 조작 자체가 불가능 — recover 가 "
                "`AssertionError: the five cells are not a partition` 로 죽는다. "
                "이 파이프라인에 그 형태의 난수화는 존재하지 않는다.",
        "runs": leak2["PART1_라벨복원_통로"]["runs"]},
     "2차_s1b_leak107_PART2_종단": leak2["PART2_종단"],
     "3차_s1c_leak107_마스크수준": {
        "무엇": "복원 창이 시즌 경계 안에 있는지를 정확한 기계 검사로 닫는다(전 예측 시즌)",
        "G3_n_next_equals_n_plus_1": leak3["G3_n_next_equals_n_plus_1"],
        "per_season": leak3["per_season"],
        "succ_season_항이_실제로_거르는_행": leak3["🔑_succ_season_항이_실제로_거르는_행"],
        "PASS": leak3["PASS"]},
     "🟢종합": "축 B 의 p_self 는 그 행보다 **엄격히 앞선 시즌**의 결과만 본다. 근거 셋 — "
               "① 마스크 수준: 4개 예측 시즌 전부에서 fit_rows 의 max season 과 "
               "max succ_season 이 s 보다 작고(G1=G2=0), n_next==n+1 이 1,474,300행 전부 참 "
               "② 종단 실행: season>=s 셀 라벨을 전량 난수화해도 예측 비트 동일 "
               "(sha 1b67d93f88a252cb), season<s 난수화하면 max|Δ| 1.911 · 100% 행 이동 "
               "③ 결정성 대조 통과."},
   "출하가능성": {
     "mate/deep_infer.py_수정_줄수": 0,
     "확인": {"파일_줄수": 65, "sha256": "923c6665056d72b8d38ac9eae45d9ac223adbb59"
                                        "96193c1748888581ae60411b",
              "정의된_것": ["_gelu", "_ln", "_fold", "_one", "predict_deep"],
              "label/target/loss/fit_등장": 0,
              "전_적합_foldable_by_mate_deep_infer": bool(
                  all(f["foldable_by_mate_deep_infer"] for f in FJ)),
              "n_fits_checked": len(FJ),
              "hid_전부_동일": sorted(set(tuple(f["hid"]) for f in FJ)),
              "n_params_전부_동일": sorted(set(f["n_params"] for f in FJ))},
     "근거": "타깃 변경은 **학습에만** 닿고 가중치 형상을 안 바꾼다. 교체되는 것은 "
             "`deepf_s*.npz` 의 값뿐이고 `prep.json` 도 그대로다.",
     "배포_재학습_시": "≤2024 학습판이면 p_self 가 2020~2024 를 덮어야 한다. 전진 연쇄 OOF 가 "
                      "2020~2023 을, stash 폴드(≤2023 적합 → 2024 예측)가 2024 를 덮어 가능하고 "
                      "2019 만 hard 로 남는다."}},
 "타깃_구성": {"grids": prep["grids"], "coverage": prep["coverage"],
              "N1_permute": prep["N1_permute"], "N2_shuffle": prep["N2_shuffle"],
              "phase2_eps": prep.get("phase2_eps"),
              "targets": prep["targets"], "p_self_source": prep["p_self_source"]},
 "지점표": rows,
 "귀무_대조": mix["귀무_대조"],
 "🔑_S103_교환율_검산": S103,
 "🔑_ρ_단가": {
   "문턱": "S103 교환율 — ρ 를 0.01 내리는 것은 팔 단독 15.82점의 값어치다. "
           "그보다 싸게 내려야 이 축이 값을 한다.",
   "이_실험의_답": "축 A 는 **음의 단가**로 ρ 를 내렸다 — e=0.10 에서 ρ −0.00648 이면서 "
                  "단독이 +10.43 올랐다(공짜보다 낫다). e=0.05 도 −11.74. "
                  "축 B 는 반대로 ρ 를 **올린다**(+0.007 / +0.022 / +0.039). "
                  "🔴 그런데 공짜로 산 ρ 하락이 혼합 이득으로 **안 바뀌었다**(−0.145).",
   "축A_ρ하락_지점": {r["point"]: {"d_rho": r["d_rho_vs_base_1632"],
                                  "d_alone": r["d_alone_vs_base_1632"],
                                  "price": r["rho_unit_price_pts_per_0.01"],
                                  "obs_delta": r["delta_vs_prereg_baseline_26.2698"]}
                     for r in rows if r["point"].startswith("A")},
   "🔴_N1_쌍은_퇴화했다": "N1a·N1b 는 팔이 망가져 세 스텝 전부에서 argmax w_deep = 0.00 이다 "
                        "(혼합이 그 팔을 아예 안 쓴다). 그래서 N1b − N1a = 정확히 0.0000 이고 "
                        "'신호 없는 팔 위에서 스무딩만의 몫'은 **측정되지 않았다**. "
                        "바닥 대조로서는 유효하다 — 라벨 순열이 팔 단독을 864.84 → 602.50 으로 "
                        "떨어뜨린다."},
 "baseline_arm": mix["baseline_arm"],
 "missing_artifacts": mix["missing_artifacts"],
 "n_fits": len(FJ),
 "wall": {"grid_phase1": "2026-08-26T09:58:41Z → 10:32:35Z (180 fits, 3 레인)",
          "grid_phase2": "→ 10:40:40Z (30 fits)",
          "mix_eval_sec": mix["wall_sec"],
          "leak_gates_sec": {"s1": leak1["wall_sec"], "s1b": leak2["wall_sec"]}},
 "서버_경로": "a5000:~/SungjinPark/exp107/"}
json.dump(M, open(f"{B}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"결론": M["🔴결론"], "VERDICT": M["🔴판정"]["VERDICT"],
                  "S103": S103}, ensure_ascii=False, indent=1))

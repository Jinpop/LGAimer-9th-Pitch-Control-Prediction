"""exp_113 — metrics.json 조립."""
import glob
import json
import os

H = "/home/<user>/SungjinPark"
B = f"{H}/exp113"
J = lambda p: json.load(open(p, encoding="utf-8"))

f1, f2, f3, f3b, f4 = (J(f"{B}/out/{n}.json") for n in ("f1_feats", "f2_gates",
                                                        "f3_screen", "f3b_nulls",
                                                        "f4_eval"))
allf = [J(p) for p in glob.glob(f"{B}/out/fits/*.json")]
fits = [x for x in allf if not x.get("skipped")]
skipped = [x["tag"] for x in allf if x.get("skipped")]
wall = sum(x["fit_wall_sec"] for x in fits)
gate_fold = [x["deployed_fold_gate"] for x in fits if x.get("deployed_fold_gate")]

M = {
    "exp_id": "exp_113",
    "title": "투수 in-season 분해 확장 → 딥 팔 입력 (94 → 94+13열)",
    "면": f4["면"],
    "사전등록": f4["prereg"],
    "🔴결론": None,           # 아래에서 채운다
    "🔴판정": f4["VERDICT"],
    "게이트": {
        "행독립성_규칙2-4": {k: f2[k] for k in
                             ("G_replace", "G_degen", "G_position", "G_single_row")},
        "시간축": {k: f2[k] for k in ("T_label_randomize_all",
                                      "T_neg_randomize_future", "T_pos_randomize_past")},
        "ALL_PASS": f2["ALL_PASS"],
        "94열_재현": {
            "무엇": "EXTRA 없이 seed42·1632 재적합 → exp_097 z_F1632_f64o_s42 와 비트 동일",
            "bitwise": True,
            "sha16": "b1da6808c7f16dea"},
        "배포_fold_비트동일": {"n_checked": len(gate_fold),
                               "max": max([max(g.values()) for g in gate_fold] or [None])},
        "면_재현": f4["G1_surface"],
        "기준선_94열_묶음": f4["G2_base94_bundles"]},
    "열_인벤토리": {
        "prep.json_NUM": 94, "prep.json_CAT": 5,
        "출하_투수_in-season_블록": ["ins_p_n"] + [f"ins_p_{r}{s}"
                                                  for r in ("success", "reverse", "middle",
                                                            "ball", "strike")
                                                  for s in ("", "_dev")],
        "출하_타자_in-season_블록": ["ins_b_n", "ins_b_success", "ins_b_success_dev",
                                     "ins_b_middle", "ins_b_middle_dev"],
        "미출하_원열": "asof_pitcher_pitchmix_n — test 에 있는데 NUM 94열에 없다",
        "후보_14열": f1["cols"], "생존_13열": f3["SEL_PR"],
        "탈락": f3["dropped_stage12"]},
    "열_선별": {c: {k: v[k] for k in ("group", "max_abs_rho", "vs",
                                      "span_resid_frac_2024", "KEEP_stage12")}
                for c, v in f3["cols"].items()},
    "🔴_귀무_결함_발견": {
        "무엇": "(season, pitcher_id) 안 순열은 귀무가 아니다 — 라벨 상관이 거의 그대로 남고 "
                "때로 원본보다 강해진다",
        "실측_2024_corr_원본→순열": {
            "ins_p_ball": [-0.04431, -0.04410], "ins_p_strike": [0.03281, 0.03327],
            "sp_pit_bhand_all": [0.02891, 0.02891],
            "r2_lg_form": [0.02126, 0.02662],
            "asof_pitcher_prev1_game_success_rate": [0.03607, 0.03640]},
        "season_안_순열은_죽인다": "전부 |corr| <= 0.0043",
        "기전": "① 투수-시즌 집계가 신호의 담지자라 그 안에서 섞어도 안 없어진다 "
                "② 누적 asof 열은 시즌 뒷부분 값이 그 투수의 시즌 전체 수준에 더 가까워 "
                "섞으면 덜 시끄러운(그리고 미래를 엿본) 추정치가 된다",
        "영향": "1차 N1 은 귀무가 아니라 강한 팔이었다 — 정직면 이득 39.25 로 94열 기준선 "
                "26.27 을 크게 웃돈다. 판정에서 뺐고 N1s(season 안 순열)로 다시 지었다.",
        "⚠️_다른_실험에도": "exp_103b 의 null3s·rand3 이 (season, batter_id) 안 순열로 같은 계열이다",
        "새_귀무_라벨상관": {t: v["max_abs_label_corr"] for t, v in f3b["blocks"].items()}},
    "판본": f4["variants"],
    "적합간_sd": f4["sigma_fit"],
    "판정_표": f4["verdict"],
    "n_fits": len(fits),
    "n_cancelled": len(skipped),
    "cancelled_reason": "결함 귀무(1차 N1_*·N2_*·N2b_PR)의 추가 스텝 지점 — 판정에서 뺐다",
    "fit_wall_sec_total": round(wall, 1),
    "서버_경로": {"work": f"{B}/work", "preds": f"{B}/out/preds",
                  "blocks": f"{B}/work/blk", "src": f"{B}/src"},
}
json.dump(M, open(f"{B}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items()
                  if k not in ("판본", "열_선별")}, ensure_ascii=False, indent=1)[:6000])

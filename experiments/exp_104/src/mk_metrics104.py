"""exp_104 — 산출물 metrics.json 조립. out/ 의 원자료를 읽어 요약만 만든다(새 계산 없음)."""
import json, os, sys

H = "/home/<user>/SungjinPark"
B = f"{H}/exp104"
A = json.load(open(f"{B}/out/sA_mix104.json"))
JM = json.load(open(f"{B}/out/jobsA_meta.json"))
BEV = json.load(open(f"{B}/out/sB_eval.json")) if os.path.exists(
    f"{B}/out/sB_eval.json") else None
CREP = json.load(open(f"{B}/out/sC_replicate.json")) if os.path.exists(
    f"{B}/out/sC_replicate.json") else None

DEF = {}          # 결함면 값 (exp_091/092 metrics.json 에서 옮긴 표)
for f, key in ((f"{H}/experiments/exp_091/metrics.json", "axis_curves"),
               (f"{H}/experiments/exp_092/metrics.json", "axis_curves")):
    if os.path.exists(f):
        for c in json.load(open(f))[key]:
            DEF[c["point"]] = {"gain_def": c.get("abd_gain_at_argmax"),
                               "w_def": c.get("abd_argmax_w"),
                               "alone_def": c.get("alone_recentred_pts")}
DEFLOCAL = json.load(open(f"{B}/out/defect_table.json")) if os.path.exists(
    f"{B}/out/defect_table.json") else {}
DEF.update(DEFLOCAL)

rows = []
for nm, v in A["표"].items():
    d = DEF.get(nm, {})
    gd = d.get("gain_def")
    rows.append({
        "point": nm, "axis": v.get("axis"), "desc": v.get("desc"),
        "gain_def_결함면": gd, "gain_fix_고친면": v["gain"],
        "확대배율": (v["gain"] / gd) if gd else None,
        "own_argmax_step": v["own_argmax_step"], "own_argmax_w": v["own_argmax_w"],
        "delta_vs_26.2698": v["delta_vs_prereg_baseline"],
        "t_paired_boot": v["vs_base94_1632_paired"]["t"],
        "t_combined_1sigma": v.get("t_combined_1sigma"),
        "alone_recentred": v.get("alone_recentred"),
        "rho_vs_z_mate_진단": v.get("rho_vs_z_mate"),
        "PASS": v["PASS"]})
rows.sort(key=lambda r: -(r["delta_vs_26.2698"]))

M = {
    "exp_id": "exp_104",
    "title": JM.get("stage") and "새 화면으로 팔 공간 재훑기 — 결함면 기각 재판독 + TabPFN 재개",
    "date": "2026-08-26",
    "surface": {"면": A["면"], "기준선_사전등록": A["prereg"]["baseline_gain"],
                "baseline_w0_rc": A["baseline_w0"]["rc"],
                "게이트_26.2698_재현": A["baseline_arm"]["gate_1632_reproduces_26.2698"],
                "기준선_스텝별": {k: {"argmax": vv["argmax"],
                                      "alone": vv["alone_recentred"]}
                                  for k, vv in A["baseline_arm"]["per_step"].items()}},
    "prereg": A["prereg"],
    "축A_기각_재판독": {"VERDICT": A["VERDICT"], "표": rows,
                        "S104_개수맞춤_귀무대조": A.get("S104_개수맞춤_귀무대조"),
                        "귀무가족_퍼짐": A.get("🔴귀무가족_퍼짐_무작위47열"),
                        "적합간_sd": A["적합간_sd_S178"], "검출한계": A["검출한계"],
                        "n_points": len(rows),
                        "missing_artifacts": A["missing_artifacts"]},
    "축A_후속_복제": CREP,
    "축B_TabPFN": BEV and {"VERDICT": BEV["VERDICT"], "표": BEV["표"],
                           "시간관문": BEV["시간관문"],
                           "gate": BEV["gate_2arm_baseline"]},
    "변경변인": JM,
}
json.dump(M, open(sys.argv[1], "w"), ensure_ascii=False, indent=1)
print("wrote", sys.argv[1], "points:", len(rows))

"""exp_096 — metrics.json 조립 + 남은 진단 두 개.

추가로 재는 것:
  · rho(z_deep, z_mate)  — rank 팔의 rho 를 딥 팔 자신의 rho 와 같은 면에서 비교하려면 필요하다.
  · S103 회귀(gain ~ rho + alone; rho계수 −308.92, alone계수 +0.19533)로 rank 의 이득이
    '품질·상관 교환율'로 설명되는 크기인지 대조한다.  회귀는 국소 근사라 부호·자릿수만 본다.
"""
import glob, hashlib, json, os, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B90, B96 = f"{H}/exp090", f"{H}/exp096"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
S5 = [42, 777, 2024, 101, 7]
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
LV = sum(W4[t] * lg(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * lg(V["p_q"]))
          + BO["w_ebm"] * lg(V["p_ebm"]) + BO["beta_ins"] * V["off"])
DZ = np.mean([np.load(f"{B90}/out/preds/z_K1632_oot23_s{s}.npy") for s in S5], 0)
rho = lambda a, b: float(np.corrcoef(a, b)[0, 1])


def rd(p):
    return json.load(open(p)) if os.path.exists(p) else None


M = {"exp_id": "exp_096",
     "title": "새 팔 추가 배터리 — 내부 슬롯 3자유도 (아키텍처 계열 + 시퀀스 팔)",
     "surface": ("정직면 OOT — 전 팔 season<=2023 학습 -> season 2024 253,507행 채점. "
                 "exp_089/090/095 하니스 승계."),
     "mix": "z_inner(w,w2) = (1-w-w2)*z_mate + w*z_deep + w2*z_new",
     "prereg_threshold": {
         "reference_point": "(w=0.36, w2=0)", "reference_gain": 13.714309769419287,
         "rule": "argmax 이득 − 13.714310 >= +2.5  AND  짝지은 투수블록 t >= 2.0",
         "source": "experiments/exp_096/config.yaml (발사 전 확정, 결과 보고 안 바꿨다 — S8)"},
     "🔴_no_LB_conversion": ("정직면 크기를 LB 로 환산하지 않는다 (S54 19.4배 · S78 54배 · "
                             "S111 0.478). 순위·부호·상대 비교만 쓴다."),
     "stage0_feasibility": rd(f"{B96}/out/s0_feasibility.json"),
     "prep": rd(f"{B96}/out/prep096.json"),
     "seqprep": rd(f"{B96}/out/seqprep096.json")}

ARMS = {}
for f in ("s1_arch_partial.json", "s1_attn.json", "s2_seq.json",
          "s1c_ranksteps.json", "s1d_ranksteps_low.json", "s1e_ctrl.json"):
    d = rd(f"{B96}/out/{f}")
    if d:
        M.setdefault("gates_per_eval", {})[f] = d["gates"]
        for a, v in d["arms"].items():
            ARMS[a] = {k: vv for k, vv in v.items()
                       if k not in ("grid", "substitution_curve_w0")}
            ARMS[a]["_grid_n"] = len(v.get("grid", []))
M["arms"] = ARMS
M["s1b_rank_robustness"] = rd(f"{B96}/out/s1b_rank.json")
M["foldb_independent"] = rd(f"{B96}/out/foldb_eval.json")
M["foldb_addendum"] = rd(f"{B96}/out/foldb_addendum.json")
M["foldb_prep"] = rd(f"{B96}/out/foldb_prep.json")
M["prep_f64_control"] = rd(f"{B96}/out/prep_f64.json")
if M["s1b_rank_robustness"]:
    for k in ("wide_grid", "substitution_w0", "three_seed_subsets"):
        if k in M["s1b_rank_robustness"]:
            M["s1b_rank_robustness"][k].pop("curve", None)
            M["s1b_rank_robustness"][k].pop("per_subset", None)

M["deep_reference_arm"] = {
    "tag": "exp_090 K1632_oot23 5시드 평균 (= 이 실험의 z_deep)",
    "rho_vs_z_mate": rho(DZ, Z_MATE), "z_sd": float(DZ.std())}

# ── S103 교환율 대조 ─────────────────────────────────────────────────
S103 = {"rho_coef": -308.92, "alone_coef": 0.19533,
        "🔴_caveat": "S103 회귀는 딥 슬롯 안 국소 근사다. 자릿수·부호만 본다."}
if "rank" in ARMS:
    dr = ARMS["rank"]["DIAG_rho_vs_z_mate"] - M["deep_reference_arm"]["rho_vs_z_mate"]
    da = (ARMS["rank"]["DIAG_alone_score_seedmean_rc"]
          - M["s1b_rank_robustness"]["deep_reference"]["alone_seedmean_rc"])
    S103.update({
        "rank_minus_deep_rho_vs_mate": dr,
        "rank_minus_deep_alone": da,
        "S103_predicted_gain_delta": S103["rho_coef"] * dr + S103["alone_coef"] * da,
        "observed_residual_vs_2arm_ref":
            ARMS["rank"]["residual_vs_prereg_ref"]})
M["S103_crosscheck"] = S103

M["verdict_table"] = {
    a: {"config_grid_argmax": [v["argmax_fixed_alpha"]["w"],
                               v["argmax_fixed_alpha"]["w2"]],
        "gain": v["argmax_fixed_alpha"]["gain_fixed_alpha"],
        "residual_vs_ref": v["residual_vs_prereg_ref"],
        "paired_t": v["paired_boot_vs_ref_w0.36"]["t"],
        "PASS": v["VERDICT"]["PASS"],
        "alone_seedmean": v["DIAG_alone_score_seedmean_rc"],
        "rho_vs_mate": v["DIAG_rho_vs_z_mate"],
        "rho_vs_deep": v["DIAG_rho_vs_z_deep"],
        "n_params": v["n_params"], "n_train": v["n_train"],
        "fit_wall_sec_mean": v["fit_wall_sec_mean"]}
    for a, v in ARMS.items()}
M["ANY_PASS"] = bool(any(v["PASS"] for v in M["verdict_table"].values()))
M["server_paths"] = {
    "work": f"{B96}", "preds": f"{B96}/out/preds/",
    "rank_input_matrix": f"{B96}/work/X_rank_oot23.npy",
    "seq_tables": f"{B96}/work/seq_table*.npy"}
json.dump(M, open(f"{B96}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"verdict_table": M["verdict_table"], "ANY_PASS": M["ANY_PASS"],
                  "deep_ref": M["deep_reference_arm"], "S103": M["S103_crosscheck"]},
                 ensure_ascii=False, indent=1))

# ── 🔴 재실행 안전: 손으로 얹은 판정 블록을 여기서 다시 붙인다 ──────────
# (laneH 가 attn 뒤에 mk_metrics 를 다시 돌린다. 이 줄이 없으면 아래 블록들이 지워진다.)
import subprocess as _sp
M["s1f_rebaselined"] = rd(f"{B96}/out/s1f_rebase.json")
json.dump(M, open(f"{B96}/out/metrics.json", "w"), ensure_ascii=False, indent=1)
for _s in ("attrib.py", "final_note.py", "final2.py"):
    _sp.run(["/home/<user>/SungjinPark/venv/bin/python", f"{B96}/src/{_s}"], check=False)

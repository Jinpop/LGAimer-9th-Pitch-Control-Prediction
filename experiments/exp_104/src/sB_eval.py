"""exp_104 축B 평가 — TabPFN 판본들을 축A 와 **같은 정직면 혼합 화면**에서 읽는다.

혼합: z_inner(w, w2) = (1-w-w2)·z_mate + w·z_deep + w2·z_arm     (exp_101 gateC 와 같은 3자유도)
  · B1(열 부분집합)  : z_arm = logit(p_tab)               — 전 행
  · B2(부분 행 전문가): z_arm = z_mate 를 기본으로 두고 **고른 행에서만** logit(p_tab) 로 교체
    → w2 가 그 행에서만 움직인다. 나머지 행은 정확히 z_mate 이므로 항등이다.
  · 대조로 exp_101 의 전 행 판본(ctx20k 단일 · 배깅5)도 같은 화면에서 다시 읽는다.

🔴 사전 등록(축A 와 동일, 발사 전 확정 S8): 이득 − 26.2698 ≥ +2.5 AND 짝지은 t ≥ 2.0.
🔴 ρ 는 진단으로만(S83). 🔴 시간 관문은 별도 산출물(sB1/sB2 json)의 L4 환산으로 판정한다.
"""
import glob, json, os, sys, time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np

H = "/home/<user>/SungjinPark"
B, B101 = f"{H}/exp104", f"{H}/exp101"
sys.path.insert(0, f"{B}/src")
import surface104 as S                                        # noqa: E402

t0 = time.time()
G1 = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.01)]
G2 = [round(x, 4) for x in np.arange(0.0, 0.6001, 0.02)]


def gain(zm_w, w2, arm):
    return S.sc(S.root((1 - zm_w - w2) * S.Z_MATE + zm_w * S.Z_DEEP + w2 * arm),
                "abd") - S.BASE_ABD


# 2팔 기준선 재현 게이트
cur = [{"w": w, "gain": gain(w, 0.0, S.Z_MATE)} for w in G1]
b2 = max(cur, key=lambda c: c["gain"])
R = {"exp_id": "exp_104", "stage": "sB_eval",
     "면": "축A 와 같은 정직면 (surface104.py)",
     "gate_2arm_baseline": {"argmax": b2, "expected": S.PREREG_BASELINE,
                            "abs_diff": abs(b2["gain"] - S.PREREG_BASELINE),
                            "PASS": abs(b2["gain"] - S.PREREG_BASELINE) < 0.01},
     "prereg": {"rule": "gain - 26.2698 >= +2.5 AND paired pitcher-block t >= 2.0",
                "source": "exp_104 config.yaml — 발사 전 확정 (S8)"},
     "표": {}}
assert R["gate_2arm_baseline"]["PASS"], b2
W_REF = b2["w"]
P_REF = S.prob(S.root((1 - W_REF) * S.Z_MATE + W_REF * S.Z_DEEP), "abd")

ARMS = {}
for p in sorted(glob.glob(f"{B}/work/p_tabB1_top*.npy")):
    nm = "B1_" + os.path.basename(p)[len("p_tabB1_"):-4]
    ARMS[nm] = {"z": S.lg(np.load(p)), "sel": None, "src": p}
for p in sorted(glob.glob(f"{B}/work/z_tabB2_*.npy")):
    tag = os.path.basename(p)[len("z_tabB2_"):-4]
    sel = np.load(f"{B}/work/sel_tabB2_{tag}.npy")
    z = S.Z_MATE.copy()
    z[sel] = np.load(p)
    ARMS["B2_" + tag] = {"z": z, "sel": sel, "src": p}
for p, nm in ((f"{B101}/work/p_tab_ctx20k_e1.npy", "REF_exp101_ctx20k_e1"),
              (f"{B101}/work/p_tab_bag5.npy", "REF_exp101_bag5")):
    if os.path.exists(p):
        ARMS[nm] = {"z": S.lg(np.load(p)), "sel": None, "src": p}

for nm, d in ARMS.items():
    arm = d["z"]
    best = None
    for w in G2:
        for w2 in G2:
            if w + w2 > 0.60:
                continue
            g = gain(w, w2, arm)
            if best is None or g > best["gain"]:
                best = {"w_deep": w, "w_tab": w2, "gain": g}
    zb = S.root((1 - best["w_deep"] - best["w_tab"]) * S.Z_MATE
                + best["w_deep"] * S.Z_DEEP + best["w_tab"] * arm)
    pb = S.prob(zb, "abd")
    c = S.boot_t((pb - S.y) ** 2 - (P_REF - S.y) ** 2)
    sel = d["sel"]
    row = {"src": d["src"], "argmax": best,
           "delta_vs_prereg_baseline": best["gain"] - S.PREREG_BASELINE,
           "vs_2arm_paired": c,
           "alone_recentred_full_frame": S.sc(arm, "rc"),
           "z_sd": float(arm.std()),
           "rho_vs_z_mate": float(np.corrcoef(arm, S.Z_MATE)[0, 1]),
           "rho_vs_z_deep": float(np.corrcoef(arm, S.Z_DEEP)[0, 1]),
           "PASS_size": bool(best["gain"] - S.PREREG_BASELINE >= S.PREREG_DELTA),
           "PASS_t": bool(c["t"] is not None and c["t"] >= S.PREREG_T)}
    if sel is not None:
        row["n_selected_rows"] = int(sel.sum())
        row["share_selected"] = float(sel.mean())
        row["identity_off_selection_max_abs"] = float(
            np.abs(arm[~sel] - S.Z_MATE[~sel]).max())
    row["PASS"] = bool(row["PASS_size"] and row["PASS_t"])
    R["표"][nm] = row
    print("%-24s w_deep %.2f w_tab %.2f gain %+9.4f (d %+8.4f) t %+6.3f "
          "alone %9.2f rho_mate %.4f %s"
          % (nm, best["w_deep"], best["w_tab"], best["gain"],
             row["delta_vs_prereg_baseline"], c["t"] or float("nan"),
             row["alone_recentred_full_frame"], row["rho_vs_z_mate"],
             "PASS" if row["PASS"] else ""), flush=True)

# 시간 관문 요약 — sB1/sB2 산출물에서 그대로 옮긴다
TIME = {}
for p in sorted(glob.glob(f"{B}/out/sB1_cols.json") + glob.glob(f"{B}/out/sB2_rows_*.json")):
    d = json.load(open(p))
    for r in d["rows"]:
        key = f"B1_top{r['topk']}" if "topk" in r and "tag" not in r else f"B2_{r['tag']}"
        TIME[key] = {k: r[k] for k in r if k in
                     ("t_total_sec", "a5000_sec_245789rows", "a5000_sec_245789frame",
                      "L4_conservative_sec", "fits_L4_budget_484s", "n_query",
                      "share_of_2024")}
R["시간관문"] = {"L4_factor": 2.94, "budget_sec": 484.0,
                "exp_101_기준": {"ctx20k_e1_a5000_sec": 207.4,
                                 "L4_conservative_sec": round(207.4 * 2.94, 1),
                                 "bag5_a5000_sec": 1059.0},
                "측정": TIME}
qual = [k for k, v in R["표"].items() if v["PASS"]]
R["VERDICT"] = {"qualifying_signal": qual,
                "PASS_signal": bool(qual),
                "PASS_time": {k: v.get("fits_L4_budget_484s") for k, v in TIME.items()},
                "PASS_both": [k for k in qual
                              if TIME.get(k, {}).get("fits_L4_budget_484s")]}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B}/out/sB_eval.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({"gate": R["gate_2arm_baseline"], "VERDICT": R["VERDICT"],
                  "시간관문": R["시간관문"]}, ensure_ascii=False, indent=1))

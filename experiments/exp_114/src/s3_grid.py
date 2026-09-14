"""exp_114 stage3 — (스텝 x w_deep) 2차원 격자, 서로소 5시드 묶음 5개.

🔴 문턱은 s2_sigma.json 에서 읽는다. 여기서 재정의하지 않는다 (S8).
🔴 앵커 둘을 **함께** 보고한다 — 배포 CE 는 '6에폭 = 2160 배포스텝' 이라
   OOT 로 옮기는 관례가 둘이다:
     A_step  = OOT 2160 스텝 (스텝 정합; config 가 지정한 기준선)
     A_epoch = OOT 1632 스텝 (에폭 정합; exp_090 -> CE 가 실제로 쓴 관례,
               exp_097 sB_indep 의 deploy_steps_oot_equiv)
   판정은 config 대로 A_step 으로 하고, A_epoch 값을 나란히 적는다.
"""
import json
import sys

sys.path.insert(0, "/home/<user>/SungjinPark/exp114/src")
import numpy as np  # noqa: E402
from surf114 import (B, BUNDLE, CKPTS, GRID, LB_FLOOR, TRANSFER,  # noqa: E402
                     BASE_EXPECT, REF_GAIN_1632_B0, Surface, zbundle, bootstrapper)

PRE = json.load(open(f"{B}/out/s2_sigma.json"))["PREREG"]
A_STEP, A_EPOCH, AW = 2160, 1632, 0.40
S = Surface()
boot = bootstrapper(S.pit)
R = {"exp_id": "exp_114", "stage": "s3_grid", "PREREG": PRE,
     "G1_surface": {"w0_rc": S.base_rc, "expected": BASE_EXPECT,
                    "PASS": bool(abs(S.base_rc - BASE_EXPECT) < 1e-9)}}
assert R["G1_surface"]["PASS"]

tagf = lambda s: f"M_s{s}"
G = {}          # G[b][k][w]
for b in range(5):
    G[b] = {}
    for k in CKPTS:
        zd = zbundle(tagf, b, k)
        cur, best = S.curve(zd)
        G[b][k] = cur
        if b == 0 and k in (1632, 2176):
            print("b0", k, best, flush=True)
    print("bundle", b, "done", flush=True)

# G2 — exp_097 재현: 묶음0 @1632 혼합 argmax
b0_1632 = max(G[0][1632].items(), key=lambda kv: kv[1])
R["G2_exp097_replicate"] = {
    "bundle0_step1632_argmax_w": b0_1632[0], "gain": b0_1632[1],
    "expected_exp097": REF_GAIN_1632_B0,
    "abs_diff": abs(b0_1632[1] - REF_GAIN_1632_B0),
    "PASS": bool(abs(b0_1632[1] - REF_GAIN_1632_B0) < 1e-9)}
assert R["G2_exp097_replicate"]["PASS"], b0_1632

mean_g = {k: {w: float(np.mean([G[b][k][w] for b in range(5)])) for w in GRID}
          for k in CKPTS}
sd_g = {k: {w: float(np.std([G[b][k][w] for b in range(5)], ddof=1)) for w in GRID}
        for k in CKPTS}
R["grid_mean_gain"] = mean_g
R["grid_sd_over_bundles"] = sd_g
R["per_bundle_argmax"] = {
    str(b): {"step": (lambda t: t[0])(max(((k, max(G[b][k].values())) for k in CKPTS),
                                          key=lambda t: t[1])),
             "gain": max(max(G[b][k].values()) for k in CKPTS)} for b in range(5)}
R["step_profile_at_w"] = {
    str(w): {str(k): mean_g[k][w] for k in CKPTS} for w in (0.20, 0.30, 0.40, 0.50)}


def judge(anchor_k, name):
    anc = {b: G[b][anchor_k][AW] for b in range(5)}
    rows = {}
    for k in CKPTS:
        for w in GRID:
            d = [G[b][k][w] - anc[b] for b in range(5)]
            m = float(np.mean(d))
            sdd = float(np.std(d, ddof=1))
            rows[(k, w)] = (m, sdd, d)
    kb, wb = max(rows, key=lambda t: rows[t][0])
    m, sdd, d = rows[(kb, wb)]
    sig_fit = max(sdd / np.sqrt(5), PRE["sigma_floor_for_mean"])
    zc = np.mean([(1 - wb) * S.Z_MATE + wb * zbundle(tagf, b, kb) for b in range(5)], 0)
    za = np.mean([(1 - AW) * S.Z_MATE + AW * zbundle(tagf, b, anchor_k) for b in range(5)], 0)
    bt = boot((S.prob(S.root(zc), "abd") - S.y) ** 2
              - (S.prob(S.root(za), "abd") - S.y) ** 2)
    sig = float(np.hypot(sig_fit, bt["boot_se_pts"]))
    # sd 맞춤 (S25/S33): 후보·앵커 **둘 다** 기준선 z0 의 sd 로 맞춘 뒤 차를 낸다
    sdm = (S.sc(S.sd_match(S.root(zc)), "abd")
           - S.sc(S.sd_match(S.root(za)), "abd"))
    return {"anchor": {"steps": anchor_k, "w": AW,
                       "gain_per_bundle": {str(b): anc[b] for b in range(5)},
                       "mean": float(np.mean(list(anc.values())))},
            "best": {"steps": kb, "w": wb},
            "delta_mean_pts": m, "delta_per_bundle": d,
            "delta_sd_over_bundles_ddof1": sdd,
            "sigma_fit_used": float(sig_fit),
            "boot_se_pts": bt["boot_se_pts"], "boot_ci95_pts": bt["ci95_pts"],
            "sigma_combined": sig, "t_combined": m / sig,
            "delta_sd_matched_pts": sdm,
            "lb_equivalent": m * TRANSFER,
            "PASS": bool(m >= PRE["min_pts_from_LB"] and m / sig >= PRE["min_t"]),
            "n_bundles_positive": int(sum(1 for x in d if x > 0))}


R["verdict_A_step_2160"] = judge(A_STEP, "step")
R["verdict_A_epoch_1632"] = judge(A_EPOCH, "epoch")
R["VERDICT"] = {
    "판정_기준선": "A_step = OOT 2160 (config 지정)",
    "PASS": R["verdict_A_step_2160"]["PASS"],
    "delta_pts": R["verdict_A_step_2160"]["delta_mean_pts"],
    "t": R["verdict_A_step_2160"]["t_combined"],
    "lb": R["verdict_A_step_2160"]["lb_equivalent"],
    "🔴_관례_의존": {
        "A_step_2160_delta": R["verdict_A_step_2160"]["delta_mean_pts"],
        "A_epoch_1632_delta": R["verdict_A_epoch_1632"]["delta_mean_pts"],
        "왜": "배포 CE 는 6에폭(=2160 배포스텝)이다. OOT 는 에폭당 272스텝이라 "
              "6에폭 = 1632. 어느 쪽이 배포의 정직면 상인지가 이 실험의 실질 질문이다 "
              "— s4_scale 이 그것을 잰다."}}
json.dump(R, open(f"{B}/out/s3_grid.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items()
                  if k not in ("grid_mean_gain", "grid_sd_over_bundles")},
                 ensure_ascii=False, indent=1)[:4000])

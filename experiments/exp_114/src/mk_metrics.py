"""exp_114 — metrics.json 조립."""
import glob
import json
import os

B = "/home/<user>/SungjinPark/exp114"
O = f"{B}/out"
g = lambda n: json.load(open(f"{O}/{n}.json")) if os.path.exists(f"{O}/{n}.json") else None

gate = json.load(open(f"{O}/fits/GATE.json"))
s2, s3, s4 = g("s2_sigma"), g("s3_grid"), g("s4_scale")
s5, s6, s7 = g("s5_ship"), g("s6_plaw"), g("s7_deploy")
gj = g("gates114")
fits = {}
for p in sorted(glob.glob(f"{O}/fits/*.json")):
    d = json.load(open(p))
    fits[d["tag"]] = {"seed": d["seed"], "frac": d["frac"], "n_train": d["n_train"],
                      "steps_per_epoch": d["steps_per_epoch"],
                      "fit_wall_sec": d["fit_wall_sec"],
                      "ck_sha": {c["steps"]: c["z_sha256"][:16] for c in d["ckpts"]}}

M = {
 "exp_id": "exp_114",
 "title": "딥 학습량 x w_deep — 고친 정직면 2차원 격자 + 스텝/에폭 관례 판정",
 "면": "exp_097 sB1 = exp_113 f4 (f64 표준화 OOT 입력)",
 "🔴_전제_정정": {
   "config_가_적은_것": "고친 면 학습량 지점은 exp_097 의 1632 하나뿐",
   "실제": "exp_097 stageB1 이 10스텝 x w 전 격자를 이미 돌렸다 (out/sB_indep.json surfaces.fix)",
   "exp_097_결론": {"best_point_fix": {"steps": 1632, "w": 0.43, "gain": 26.269817382706947},
                    "deployed_point_fix": {"steps": 1632, "w": 0.40, "gain": 26.15139685310237},
                    "delta": 0.1184205296045775, "boot_t": 0.19143050505446071,
                    "PREREG_PASS": False},
   "이_실험이_더한_것": ["묶음 5개로 σ 직접 측정 (exp_097 은 묶음 1개, 부트스트랩 σ 만)",
                        "체크포인트 23지점으로 스텝 축을 조밀하게",
                        "스텝 정합 대 에폭 정합 관례를 학습행수 축척으로 판정 (s4_scale)"]},
 "gates": gj,
 "prereg": s2["PREREG"] if s2 else None,
 "sigma": {k: v for k, v in (s2 or {}).items() if k in ("anchor", "seed_only_null")},
 "grid": {k: v for k, v in (s3 or {}).items()
          if k in ("G2_exp097_replicate", "per_bundle_argmax", "step_profile_at_w",
                   "verdict_A_step_2160", "verdict_A_epoch_1632", "VERDICT")},
 "scaling_law": {k: v for k, v in (s4 or {}).items() if k != "by_frac"},
 "scaling_by_frac": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("profile")
                         and not kk.startswith("gain_profile")}
                     for k, v in ((s4 or {}).get("by_frac") or {}).items()},
 "ship_fixed_w040": {k: v for k, v in (s5 or {}).items() if k != "candidates_fixed_w0.40"},
 "ship_candidates": (s5 or {}).get("candidates_fixed_w0.40"),
 "p_law_robust": s6,
 "deploy": s7,
 "fits": fits,
 "n_fits": len(fits),
 "total_fit_wall_sec": sum(f["fit_wall_sec"] for f in fits.values()),
}
json.dump(M, open(f"{B}/metrics.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in M.items() if k not in ("fits", "grid")},
                 ensure_ascii=False, indent=1)[:3000])

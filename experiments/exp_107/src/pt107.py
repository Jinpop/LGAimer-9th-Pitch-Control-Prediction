import json, sys
d=json.load(open("/home/<user>/SungjinPark/exp107/out/s2_mix107.json"))
T=d["表"] if "表" in d else d["표"]
b=d["baseline_arm"]
print("BASELINE per step:")
for k,v in b["per_step"].items():
    a=v["argmax"]
    print("  k=%5s gain %+9.4f w %.2f alone %9.2f rho %.5f" % (k,a["gain"],a["w"],v["alone_recentred"],v["rho_vs_z_mate"]))
print("  own_argmax_step", b["own_argmax_step"], "gain", round(b["own_argmax_gain"],4))
print()
hdr="%-14s %5s %5s %9s %8s %7s %9s %8s %9s %8s %9s" % ("point","step","w","gain","delta","t_boot","alone","rho","d_rho","d_alone","price/0.01")
print(hdr); print("-"*len(hdr))
for nm,v in sorted(T.items(), key=lambda kv:-kv[1]["delta_vs_prereg_baseline"]):
    s=v.get("S103_교환율") or {}
    pr=s.get("rho_unit_price_pts_per_0.01")
    print("%-14s %5d %5.2f %9.4f %+8.3f %+7.3f %9.2f %8.5f %+9.5f %+8.2f %9s" % (
      nm, v["own_argmax_step"], v["own_argmax_w"], v["gain"], v["delta_vs_prereg_baseline"],
      v["vs_base94_1632_paired"]["t"] or 0, v["alone_recentred"], v["rho_vs_z_mate"],
      s.get("d_rho_vs_base_1632") or 0, s.get("d_alone_vs_base_1632") or 0,
      ("%.2f"%pr) if pr is not None else "n/a"))
print()
print("VERDICT qualifying:", d["VERDICT"]["qualifying"], "PASS", d["VERDICT"]["PASS"], "missing", len(d["missing_artifacts"]))

import json
d=json.load(open("/home/<user>/SungjinPark/exp107/out/s2_mix107.json"))
N=d["귀무_대조"]["표"]; T=d["표"]
print("%-26s %9s %9s %9s %8s" % ("pair","a_gain","b_gain","a-b","t"))
print("-"*66)
for k,v in N.items():
    print("%-26s %9.4f %9.4f %+9.4f %+8.3f" % (k, v["a_gain"], v["b_gain"], v["a_minus_b"], v["paired"]["t"] or 0))
print()
print("S103 exchange-rate: predicted vs observed  (all at step 1632, vs base94)")
print("%-14s %10s %10s %10s" % ("point","pred_A","pred_B","observed"))
for nm,v in sorted(T.items(), key=lambda kv:-kv[1]["delta_vs_prereg_baseline"]):
    s=v.get("S103_교환율")
    if not s: continue
    print("%-14s %10.3f %10.3f %10.3f" % (nm,
        s["pred_dgain_defectfit_-308.92rho_+0.18533alone"],
        s["pred_dgain_fixedfit_-289.78rho_+0.186alone"],
        s["obs_dgain_1632"]))
print()
print("detect limits:", json.dumps(d["검출한계"], ensure_ascii=False))

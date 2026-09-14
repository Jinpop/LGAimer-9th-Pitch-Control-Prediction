import glob, json, os
import numpy as np
H="/home/<user>/SungjinPark"; B=f"{H}/exp107"
# 1) foldable / shipping
fs=glob.glob(f"{B}/out/fit_*.json")
fold=[json.load(open(f))["foldable_by_mate_deep_infer"] for f in fs]
hid=set(tuple(json.load(open(f))["hid"]) for f in fs)
nps=set(json.load(open(f))["n_params"] for f in fs)
print("n_fits", len(fs), "all_foldable", all(fold), "hid", hid, "n_params", nps)
# 2) S103 residuals
d=json.load(open(f"{B}/out/s2_mix107.json")); T=d["表"] if "表" in d else d["표"]
rows=[(k,v["S103_교환율"]["pred_dgain_defectfit_-308.92rho_+0.18533alone"],
       v["S103_교환율"]["pred_dgain_fixedfit_-289.78rho_+0.186alone"],
       v["S103_교환율"]["obs_dgain_1632"], v["own_argmax_w"])
      for k,v in T.items() if v.get("S103_교환율")]
live=[r for r in rows if r[0] not in ("N3_identity","N1a_perm_e0","N1b_perm_eA")]
pa=np.array([r[1] for r in live]); ob=np.array([r[3] for r in live])
print("S103 live points n=%d  mean(pred-obs)=%.3f  rho(pred,obs)=%.4f  rmse=%.3f"
      % (len(live), float((pa-ob).mean()), float(np.corrcoef(pa,ob)[0,1]),
         float(np.sqrt(((pa-ob)**2).mean()))))
small=[r for r in live if abs(r[3])<10]; big=[r for r in live if abs(r[3])>=10]
for nm,g in (("|obs|<10",small),("|obs|>=10",big)):
    if g:
        p=np.array([r[1] for r in g]); o=np.array([r[3] for r in g])
        print("   %-10s n=%d mean(pred-obs)=%+7.3f" % (nm,len(g),float((p-o).mean())))
# 3) N1 degeneracy note
for nm in ("N1a_perm_e0","N1b_perm_eA"):
    v=T[nm]; ps=v["per_step"]
    print(nm, "per-step argmax w/gain:", {k:(round(x["argmax"]["w"],2), round(x["argmax"]["gain"],4)) for k,x in ps.items()})

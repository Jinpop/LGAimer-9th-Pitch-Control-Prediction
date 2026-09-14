"""exp_097 stageA: a HARD upper bound on what fixing a float32 site could buy.

Brier change of a perturbation dp is
    mean((p+dp-y)^2) - mean((p-y)^2) = mean(dp^2) + 2*mean(dp*(p-y))
and |mean(dp*(p-y))| <= sqrt(mean(dp^2) * mean((p-y)^2))  (Cauchy-Schwarz).
So WITHOUT the labels, and for ANY label vector with the competition reference
Brier, the score move is bounded by K*(mean(dp^2) + 2*sqrt(mean(dp^2)*REF)).
That bound holds even if the rounding noise were perfectly aligned with the
residual, which it is not.
"""
import json, numpy as np, pandas as pd, os
B="/home/<user>/SungjinPark/exp097"; REF=0.2498069; K=100000.0/REF
base=pd.read_csv(f"{B}/out/sub_V0.csv",float_precision="round_trip")
p0=base["control_success"].to_numpy(np.float64)
R={"REF_brier":REF,"K":K,"note":"bound is |score move| <= K*(m2 + 2*sqrt(m2*REF))"}
for v in ("VD","V1","V2","V3","V4"):
    f=f"{B}/out/sub_{v}.csv"
    if not os.path.exists(f): R[v]="MISSING"; continue
    p=pd.read_csv(f,float_precision="round_trip")["control_success"].to_numpy(np.float64)
    dp=p-p0; m2=float((dp**2).mean())
    R[v]={"mean_dp2":m2,"K_mean_dp2":K*m2,
          "max_abs_score_move_bound_pts":K*(m2+2.0*np.sqrt(m2*REF)),
          "max_abs_dp":float(np.abs(dp).max())}
    print("%-3s mean(dp^2)=%.4e  K*mean(dp^2)=%.4e pts  |score move| <= %.6f pts"
          %(v,m2,K*m2,R[v]["max_abs_score_move_bound_pts"]))
json.dump(R,open(f"{B}/out/sA_bound.json","w"),indent=1)
print("WROTE",f"{B}/out/sA_bound.json")

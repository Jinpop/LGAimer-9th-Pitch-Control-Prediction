import numpy as np, pandas as pd
H = "/home/<user>/SungjinPark"
z = np.load(f"{H}/exp083/out/cap_aa_cal.npz", allow_pickle=True)
NM = list(z["names"])
S = [42,7,2024,1,13,101,777,2025,31337,3,17,77,123,555,999,2019,8191,65537]
def pick(b,s,f):
    return z[f"v{NM.index(f'{b}|{b}_seed_{s}_{f}'):03d}"].astype(np.float64)
stk = np.zeros(253507)
for s in S: stk += pick("R", s, "stacker.cbm")
stk /= len(S)
stkF = np.zeros(253507)
for s in [42,7,2024]: stkF += pick("F", s, "stacker.cbm")
stkF /= 3
gt = pd.read_csv(f"{H}/build_AC/work/tr2024/test.csv", encoding="utf-8-sig",
                 usecols=["game_type"])["game_type"].astype(str).to_numpy()
mix = np.where(gt == "F", 0.5*stk + 0.5*stkF, stk)
v = np.load(f"{H}/build_AY/work/pcal_aa.npy").astype(np.float64)
print("pcal_aa range", v.min(), v.max())
for lab, mine in (("R_stage2",stk), ("mix_stage2",mix), ("p_off",z["p_off"]), ("p_final",z["p_final"])):
    d = np.abs(mine - v)
    print(f"pcal_aa vs {lab:<11} ndiff {int(np.count_nonzero(d)):>7} max {d.max():.3e}")
for a in ("S2","S1","T70"):
    w = np.load(f"{H}/build_AY/work/pcal_{a}.npy").astype(np.float64)
    d = np.abs(z[f"smooth_{a}"] - w)
    print(f"pcal_{a} vs smooth_{a} ndiff {int(np.count_nonzero(d)):>7} max {d.max():.3e}")

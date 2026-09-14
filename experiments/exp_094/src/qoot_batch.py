"""여러 시드의 OOT-2024 Q 팔을 한 프로세스에서 굽는다 (feat3 재적재 1회).
env: QOOF QDEVICE QGPUDEV QTC SEEDS OUTDIR
"""
import os, json, time
import numpy as np
import pandas as pd
import catboost as cb
import feats as FT

REF_BRIER = 0.2498069
P6 = ["p6_curve","p6_cutter","p6_fastball","p6_offspeed","p6_sinker","p6_slider",
      "idea_matchup_pitch","i3_anom_curve","i3_anom_cutter","i3_anom_fastball",
      "i3_anom_offspeed","i3_anom_sinker","i3_anom_slider","i3_anom_l1"]
CATF = ["top_bottom", "game_type", "base_state"]
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
ITER_STACK = 915
SEEDS = [int(x) for x in os.environ["SEEDS"].split(",")]
OOFD = os.environ["QOOF"]; DEVICE = os.environ.get("QDEVICE", "CPU")
GPUDEV = os.environ.get("QGPUDEV", "0"); TC = int(os.environ.get("QTC", "2"))
OUTDIR = os.environ["OUTDIR"]; os.makedirs(OUTDIR, exist_ok=True)
CBP = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, loss_function="Logloss",
           bootstrap_type="Bernoulli", subsample=0.8, boosting_type="Plain",
           one_hot_max_size=10, has_time=True, thread_count=TC, verbose=False,
           allow_writing_files=False)
if DEVICE == "GPU":
    CBP.update(task_type="GPU", devices=GPUDEV)

t0 = time.time()
df = pd.read_parquet("feat3.parquet")
base = [c for c in df.columns if c != FT.TGT and not c.startswith("ins_")
        and c not in P6 and not any(c.startswith(v) for v in FT.FAMILIES.values())]
FEAT = base + [c for c in df.columns if c.startswith("ins_")] + P6
for c in CATF:
    df[c] = df[c].astype(str)
y = df[FT.TGT].to_numpy(np.float64)
ssn = df[FT.SEASON].to_numpy()
keepmask = ~((ssn <= 2022) & (df["game_type"].to_numpy() == "F"))
SF = FEAT + [f"p_{c}" for c in CELLS]
tr = keepmask & (ssn < 2024); te = ssn == 2024
print(f"[batch] train {int(tr.sum()):,} test {int(te.sum()):,} dev {DEVICE}:{GPUDEV} "
      f"{time.time()-t0:.0f}s", flush=True)
for sd in SEEDS:
    outp = f"{OUTDIR}/p_q_oot2024_s{sd}.npy"
    if os.path.exists(outp):
        print(f"  s{sd} 이미 있음, 건너뜀", flush=True); continue
    Z = np.load(f"{OOFD}/oof_p6_s{sd}.npz")
    for c in CELLS:
        df[f"p_{c}"] = Z[c]
    t1 = time.time()
    m = cb.CatBoostClassifier(iterations=ITER_STACK, random_seed=sd, **CBP)
    m.fit(df.loc[tr, SF], y[tr], cat_features=CATF)
    p = m.predict_proba(df.loc[te, SF])[:, 1]
    del m
    np.save(outp, p)
    b = float(np.mean((p - y[te]) ** 2))
    print(f"  s{sd} fit {time.time()-t1:.1f}s  brier {b:.9f}  "
          f"score {100000.0*(1.0-b/REF_BRIER):.5f}", flush=True)
print(f"[done] {time.time()-t0:.0f}s", flush=True)

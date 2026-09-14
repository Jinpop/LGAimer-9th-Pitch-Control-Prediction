"""A/B단 검증 — 최종 셀만 재적합해 출하 seed_42 와 예측 대조.

env:
  TC      thread_count (default 2)
  CELLS   쉼표구분 셀 이름 (default is_both)
  SEED    random_seed (default 42)
  DEVICE  CPU | GPU
  GPUDEV  GPU devices 문자열 (default "0")
  VOUT    출력 디렉터리
qdeploy3.py 의 FEAT/마스크/CBP 를 그대로 복사한다 (수정 금지 구역).
"""
import os, sys, time, json, hashlib
import numpy as np
import pandas as pd
import catboost as cb
import feats as FT

TC     = int(os.environ.get("TC", "2"))
SEED   = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get("DEVICE", "CPU")
GPUDEV = os.environ.get("GPUDEV", "0")
VOUT   = os.environ.get("VOUT", "verify/run")
WANT   = os.environ.get("CELLS", "is_both").split(",")
REF    = os.environ.get("REF", "/home/<user>/SungjinPark/build_CECF/pkg/CF/mate/model_q/seed_42")

P6 = ["p6_curve","p6_cutter","p6_fastball","p6_offspeed","p6_sinker","p6_slider",
      "idea_matchup_pitch","i3_anom_curve","i3_anom_cutter","i3_anom_fastball",
      "i3_anom_offspeed","i3_anom_sinker","i3_anom_slider","i3_anom_l1"]
CATF = ["top_bottom", "game_type", "base_state"]
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
ITER_CELL = {"is_mid_only": 708, "is_rev_only": 1652, "is_both": 551, "is_wayoff_true": 1101}
CBP = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, loss_function="Logloss",
           bootstrap_type="Bernoulli", subsample=0.8, boosting_type="Plain",
           one_hot_max_size=10, has_time=True, thread_count=TC, verbose=False,
           allow_writing_files=False)
if DEVICE == "GPU":
    CBP.update(task_type="GPU", devices=GPUDEV)

os.makedirs(VOUT, exist_ok=True)
t0 = time.time()
df = pd.read_parquet("feat3.parquet")
lab = pd.read_parquet("qlab.parquet")
base = [c for c in df.columns if c != FT.TGT and not c.startswith("ins_")
        and c not in P6 and not any(c.startswith(v) for v in FT.FAMILIES.values())]
ins = [c for c in df.columns if c.startswith("ins_")]
FEAT = base + ins + P6
for c in CATF:
    df[c] = df[c].astype(str)
ssn = df[FT.SEASON].to_numpy()
keepmask = ~((ssn <= 2022) & (df["game_type"].to_numpy() == "F"))
valid = lab["valid"].to_numpy().astype(bool)
fit = keepmask & valid
fh = hashlib.sha256("|".join(FEAT).encode()).hexdigest()[:16]
rep = {"n_feat": len(FEAT), "feat_hash": fh, "feat_hash_expect": "d387caeb73979555",
       "n_rows_df": int(len(df)), "cell_fit_rows": int(fit.sum()),
       "cell_fit_rows_expect": 1369186,
       "stacker_fit_rows": int(keepmask.sum()), "stacker_fit_rows_expect": 1369784,
       "device": DEVICE, "gpudev": GPUDEV, "thread_count": TC, "seed": SEED,
       "ref": REF, "cells": {}}
print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, ensure_ascii=False), flush=True)
assert len(FEAT) == 77 and fh == "d387caeb73979555", "FEAT mismatch"

Xf = df.loc[fit, FEAT]
Xall = df[FEAT]
for c in WANT:
    t = time.time()
    m = cb.CatBoostClassifier(iterations=ITER_CELL[c], random_seed=SEED, **CBP)
    m.fit(Xf, lab.loc[fit, c].to_numpy(), cat_features=CATF)
    el = time.time() - t
    m.save_model(os.path.join(VOUT, f"cell_{c}.cbm"))
    pa = m.predict_proba(Xall)[:, 1]
    np.save(os.path.join(VOUT, f"p_{c}.npy"), pa)
    ent = {"fit_sec": round(el, 2), "trees": int(m.tree_count_)}
    rp = os.path.join(REF, f"cell_{c}.cbm")
    if os.path.exists(rp):
        r = cb.CatBoostClassifier(); r.load_model(rp)
        pb = r.predict_proba(Xall)[:, 1]
        d = np.abs(pa - pb)
        ent.update(max_abs_dp=float(d.max()), mean_abs_dp=float(d.mean()),
                   n_diff=int((d > 0).sum()),
                   corr=float(np.corrcoef(pa, pb)[0, 1]))
    rep["cells"][c] = ent
    print(f"[cell] {c} {json.dumps(ent)}  total {time.time()-t0:.0f}s", flush=True)
    del m

json.dump(rep, open(os.path.join(VOUT, "report.json"), "w"), ensure_ascii=False, indent=1)
print("[done]", json.dumps(rep["cells"]), flush=True)

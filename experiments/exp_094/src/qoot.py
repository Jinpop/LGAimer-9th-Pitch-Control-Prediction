"""정직면용 OOT-2024 Q 팔 — 셀도 스태커도 2024 를 안 본다.

qrun.py 가 남긴 OOF 캐시(oof_p6_s{seed}.npz)를 쓴다. 그 캐시의 2024 열은
season<2024 로 학습한 셀이 낸 값이므로 **out-of-time** 이다.
여기서 스태커만 season<2024 행으로 추가 적합해 2024 를 예측한다 → exp_089 의
p_q_oot2024 와 같은 성격의 팔. 그 면의 Q 단독 기준값은 859.79 다.

env: QOOF(캐시 디렉터리) QSEED QDEVICE QGPUDEV QTC OUTJ OUTNPY
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
SEED = int(os.environ["QSEED"])
OOFD = os.environ["QOOF"]
DEVICE = os.environ.get("QDEVICE", "CPU")
GPUDEV = os.environ.get("QGPUDEV", "0")
TC = int(os.environ.get("QTC", "2"))
TAG = os.environ.get("TAG", f"{DEVICE}_s{SEED}")
CBP = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, loss_function="Logloss",
           bootstrap_type="Bernoulli", subsample=0.8, boosting_type="Plain",
           one_hot_max_size=10, has_time=True, thread_count=TC, verbose=False,
           allow_writing_files=False, random_seed=SEED)
if DEVICE == "GPU":
    CBP.update(task_type="GPU", devices=GPUDEV)

t0 = time.time()
df = pd.read_parquet("feat3.parquet")
base = [c for c in df.columns if c != FT.TGT and not c.startswith("ins_")
        and c not in P6 and not any(c.startswith(v) for v in FT.FAMILIES.values())]
ins = [c for c in df.columns if c.startswith("ins_")]
FEAT = base + ins + P6
for c in CATF:
    df[c] = df[c].astype(str)
y = df[FT.TGT].to_numpy(np.float64)
ssn = df[FT.SEASON].to_numpy()
keepmask = ~((ssn <= 2022) & (df["game_type"].to_numpy() == "F"))
Z = np.load(f"{OOFD}/oof_p6_s{SEED}.npz")
for c in CELLS:
    df[f"p_{c}"] = Z[c]
SF = FEAT + [f"p_{c}" for c in CELLS]
tr = keepmask & (ssn < 2024)
te = ssn == 2024
print(f"[oot] seed {SEED} {DEVICE}:{GPUDEV} train {int(tr.sum()):,} test {int(te.sum()):,} "
      f"{time.time()-t0:.0f}s", flush=True)
t1 = time.time()
m = cb.CatBoostClassifier(iterations=ITER_STACK, **CBP)
m.fit(df.loc[tr, SF], y[tr], cat_features=CATF)
fit_sec = time.time() - t1
p = m.predict_proba(df.loc[te, SF])[:, 1]
b = float(np.mean((p - y[te]) ** 2))
R = {"tag": TAG, "seed": SEED, "device": DEVICE, "gpudev": GPUDEV,
     "n_train": int(tr.sum()), "n_test": int(te.sum()), "stacker_fit_sec": round(fit_sec, 2),
     "brier_oot2024": b, "score_oot2024": 100000.0 * (1.0 - b / REF_BRIER),
     "p_mean": float(p.mean()), "p_sd": float(p.std())}
np.save(os.environ["OUTNPY"], p)
json.dump(R, open(os.environ["OUTJ"], "w"), ensure_ascii=False, indent=1)
print("[done]", json.dumps(R, ensure_ascii=False), flush=True)

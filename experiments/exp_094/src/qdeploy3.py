"""배포용 Q_ins 학습 — meta.json 레시피 그대로, 피처만 47 -> 63 (base47 + ins16).

산출: model_q_ins/seed_{42,7,2024}/{cell_*.cbm, stacker.cbm} + meta_ins.json

γ 에 대하여
  q_infer 의 γ 는 스태커 확률 로짓에 더해지는 전역 상수다. 최종 블렌드는
      z = 0.85(0.5·lg(p_v) + 0.5·(lg(p_q_pre) + γ)) + 0.15·z_e + d_adj
  이므로 γ 를 δ 만큼 바꾸면 z 가 정확히 0.425·δ 만큼 평행이동하고, 이는 d_adj 재유도로
  **정확히 흡수된다**. 따라서 γ 는 기존값을 그대로 두고 d_adj 만 다시 푼다.
  (γ 를 새로 풀려면 fold_2024 3시드 예측이 더 필요한데, 그 비용이 이득 0 인 축이다.)

⚠️ qlab.parquet 는 train 전용. 배포 zip 에 넣지 않는다.
"""
import os, sys, json, time, shutil
import numpy as np
import pandas as pd
import catboost as cb
import feats as FT

SEEDS = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["42", "7", "2024"])]
OUT = "model_q_p6"
P6 = ["p6_curve","p6_cutter","p6_fastball","p6_offspeed","p6_sinker","p6_slider",
      "idea_matchup_pitch","i3_anom_curve","i3_anom_cutter","i3_anom_fastball",
      "i3_anom_offspeed","i3_anom_sinker","i3_anom_slider","i3_anom_l1"]
CATF = ["top_bottom", "game_type", "base_state"]
CELLS = ["is_mid_only", "is_rev_only", "is_both", "is_wayoff_true"]
# 배포 iteration (meta.json n_trees) — 레시피 그대로, 단일 변인 유지
ITER_CELL = {"is_mid_only": 708, "is_rev_only": 1652, "is_both": 551, "is_wayoff_true": 1101}
ITER_STACK = 915   # v19: fold Q단독 argmax 700 x 팀 환산비 1.307
CBP = dict(depth=6, learning_rate=0.03, l2_leaf_reg=10, loss_function="Logloss",
           bootstrap_type="Bernoulli", subsample=0.8, boosting_type="Plain",
           one_hot_max_size=10, has_time=True, thread_count=2, verbose=False,
           allow_writing_files=False)
OOF_SEASONS = [2020, 2021, 2022, 2023, 2024]

t0 = time.time()
df = pd.read_parquet("feat3.parquet")
lab = pd.read_parquet("qlab.parquet")
base = [c for c in df.columns if c != FT.TGT and not c.startswith("ins_")
        and c not in P6 and not any(c.startswith(v) for v in FT.FAMILIES.values())]
ins = [c for c in df.columns if c.startswith("ins_")]
FEAT = base + ins + P6
for c in CATF:
    df[c] = df[c].astype(str)
y = df[FT.TGT].to_numpy(np.float64)
ssn = df[FT.SEASON].to_numpy()
keepmask = ~((ssn <= 2022) & (df["game_type"].to_numpy() == "F"))
valid = lab["valid"].to_numpy().astype(bool)
os.makedirs(OUT, exist_ok=True)
print(f"[load] feats {len(FEAT)} (base {len(base)} + ins {len(ins)}) "
      f"| mask {keepmask.mean():.4f} {time.time()-t0:.0f}s", flush=True)

for sd in SEEDS:
    d = os.path.join(OUT, f"seed_{sd}")
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "stacker.cbm")):
        print(f"  seed {sd}: 이미 있음, 건너뜀")
        continue
    have_cells = all(os.path.exists(os.path.join(d, f"cell_{c}.cbm")) for c in CELLS)
    P = dict(CBP, random_seed=sd)
    cache = f"oofstack/oof_p6_s{sd}.npz"
    if os.path.exists(cache):                   # A/B 에서 만든 전방연쇄 OOF 재사용
        Z = np.load(cache)
        stack = {c: Z[c] for c in CELLS}
        print(f"   s{sd} OOF: 캐시 재사용 {time.time()-t0:.0f}s", flush=True)
        OOF_LOOP = []
    else:
        stack = {c: np.full(len(df), np.nan) for c in CELLS}
        OOF_LOOP = OOF_SEASONS
    for s in OOF_LOOP:                          # 전방연쇄 OOF (스태커 학습열)
        fit = keepmask & valid & (ssn < s)
        pred = ssn == s
        Xf, Xp = df.loc[fit, FEAT], df.loc[pred, FEAT]
        for c in CELLS:
            m = cb.CatBoostClassifier(iterations=ITER_CELL[c], **P)
            m.fit(Xf, lab.loc[fit, c].to_numpy(), cat_features=CATF)
            stack[c][pred] = m.predict_proba(Xp)[:, 1]
            del m
        print(f"   s{sd} OOF {s}: fit {int(fit.sum()):,} {time.time()-t0:.0f}s", flush=True)

    fit = keepmask & valid                      # 최종 셀: <=2024 전부
    if have_cells:
        print(f"   s{sd} final cells: 기존 파일 재사용 {time.time()-t0:.0f}s", flush=True)
    else:
        Xf = df.loc[fit, FEAT]
        for c in CELLS:
            m = cb.CatBoostClassifier(iterations=ITER_CELL[c], **P)
            m.fit(Xf, lab.loc[fit, c].to_numpy(), cat_features=CATF)
            m.save_model(os.path.join(d, f"cell_{c}.cbm"))
            del m
        print(f"   s{sd} final cells: fit {int(fit.sum()):,} {time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"oofstack/oof_p6_s{sd}.npz", **{c: stack[c] for c in CELLS})

    SF = FEAT + [f"p_{c}" for c in CELLS]
    for c in CELLS:
        df[f"p_{c}"] = stack[c]
    # meta.json stack_column_construction: "the earliest season keeps NaN",
    # nan_pct_in_stacker_training_rows = 15.45  ->  2019 행을 NaN 인 채로 포함한다.
    # (1차 빌드는 이 행들을 뺐다가 스태커 학습행이 1,158,157 로 팀 기록 1,369,784 와
    #  어긋나 재빌드했다. 셀 학습행 1,369,186 은 팀 기록과 정확히 일치했다.)
    strain = keepmask
    m = cb.CatBoostClassifier(iterations=ITER_STACK, **P)
    m.fit(df.loc[strain, SF], y[strain], cat_features=CATF)
    m.save_model(os.path.join(d, "stacker.cbm"))
    del m
    print(f"   ★ s{sd} stacker: fit {int(strain.sum()):,} {time.time()-t0:.0f}s", flush=True)

meta = json.load(open("v16/model_q/meta.json"))
meta["base_features"] = base
meta["features"] = FEAT + [f"p_{c}" for c in CELLS]
meta["n_base_features"] = len(FEAT)
meta["n_features"] = len(FEAT) + 4
meta["cell_models"] = {c: f"cell_{c}.cbm" for c in CELLS}
meta["_v19_change"] = ("v18 대비: 셀·스태커 입력 피처 63 -> 77 (p6 계열 14 추가), 스태커 트리 784 -> 915. "
                       "아키텍처·하이퍼파라미터·iteration·마스크·OOF 규약·시드·오프셋·γ 전부 무변경. "
                       "p6 계열은 pipeline.apply_features(test, model/tables) 의 타겟 무관 부분집합이다 "
                       "(트랙맨 유래 구종모델 + 카운트별 평균 구종확률). 행 독립.")
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False, indent=1)
print(f"[done] {OUT} {time.time()-t0:.0f}s")

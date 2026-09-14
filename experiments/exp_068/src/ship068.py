"""exp_068 출하 검증 — k=8 배포 npz 가 출하 경로에서 그대로 도는가.

S1 npz/meta 왕복 — 출하 `mate/nn_infer.py::predict_nn` (사본이 아니라 **그 파일**)로
   배포 가중치를 읽어 학습 torch 예측과 대조. exp_067 실측 1.192e-07 수준이면 정상.
S2 범주 어휘·dtype 정합 (대역 프레임 + 실제 대회 표본 5행 둘 다 — 대역 단독 판정 금지).
S3 실제 대회 표본 5행 전방향 전파.
S4 추론 시간 — 🔴 경합 없는 단독 실행.

🔴 판정하지 않는다. 이 스크립트는 형식·시간만 잰다.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

PKG = os.path.expanduser("~/SungjinPark/build_BL/pkg_BL3")
MATE = f"{PKG}/mate"
MDIR = f"{MATE}/model"
TDIR = f"{MDIR}/tables"
B67 = os.path.expanduser("~/SungjinPark/exp067")     # 읽기 전용 (대역 프레임 캐시)
B68 = os.path.expanduser("~/SungjinPark/exp068")
OUT = f"{B68}/out"
SHIP = f"{OUT}/ship"
SEEDS = [42, 777, 2024, 101]
K = 8
CATS = ["base_state", "game_type", "pitcher_team_id", "batter_team_id",
        "cnt_state"]
sys.path.insert(0, MATE)
import nn_infer                     # noqa: E402  출하 모듈 그 자체
import pipeline as P                # noqa: E402
import splits                       # noqa: E402
import inseason                     # noqa: E402


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lg(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def build_real(test):
    feats = P.apply_features(test, TDIR)
    ratios = P.apply_ratio_features(test, feats)
    df = pd.concat([test.reset_index(drop=True), feats.reset_index(drop=True),
                    ratios.reset_index(drop=True)], axis=1)
    df["cnt_state"] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)
    S = splits.apply(df, splits.load(f"{MDIR}/deploy_splits.npz"))
    for c in S.columns:
        df[c] = S[c].values
    I = inseason.apply(df, inseason.load(f"{MDIR}/anchor.npz"))
    for c in I.columns:
        df[c] = I[c].values
    return df


R = {"stage": "ship068", "k": K}
Mm = json.load(open(f"{SHIP}/nn_sp42_meta.json"))
DENSE = Mm["DENSE"]
R["meta_shape"] = {"n_CATS": len(Mm["CATS"]), "n_DENSE": len(DENSE),
                   "n_mu": len(Mm["mu"]), "n_sd": len(Mm["sd"]),
                   "levels_sizes": {c: len(Mm["levels"][c]) for c in Mm["CATS"]}}
R["meta_CATS_matches"] = (Mm["CATS"] == CATS)

# ── S1 — npz 왕복: 출하 numpy 추론 vs 학습 torch 예측 (2024 행) ───────────────
fr = pd.read_parquet(os.path.expanduser(
    "~/SungjinPark/exp064/cache/frame99_ship.parquet"))
SP6 = [c for c in DENSE if c.startswith("sp_")]
fr[SP6] = fr[SP6].fillna(0.0)                 # nnlib.load_frame 과 동일
df24 = fr.loc[fr["season"].to_numpy() == 2024].reset_index(drop=True)
del fr
R["n_2024"] = len(df24)

s1, ps = {}, []
for sd in SEEDS:
    p_np = nn_infer.predict_nn(df24, f"{SHIP}/nn_sp{sd}.npz",
                               f"{SHIP}/nn_sp{sd}_meta.json")
    p_t = np.load(f"{OUT}/preds/D_k{K}_s{sd}.npy")
    ps.append(p_np)
    s1[f"seed_{sd}"] = {"max_abs_diff_numpy_vs_torch": float(np.max(np.abs(p_np - p_t))),
                        "mean_abs_diff": float(np.mean(np.abs(p_np - p_t))),
                        "n": int(len(p_np))}
R["S1_npz_roundtrip"] = {
    "desc": "출하 mate/nn_infer.predict_nn(k=8 배포 npz) vs 학습 torch 예측, 2024행",
    "tol": 1e-6, "per_seed": s1,
    "exp067_reference": 1.1920928955078125e-07,
    "pass": all(v["max_abs_diff_numpy_vs_torch"] < 1e-6 for v in s1.values())}
p_np_av = 1.0 / (1.0 + np.exp(-np.mean([lg(p) for p in ps], axis=0)))
p_t_av = np.load(f"{OUT}/preds/D_k{K}_ens.npy")
R["S1_ens_max_abs_diff"] = float(np.max(np.abs(p_np_av - p_t_av)))

# ── S2 — 범주 어휘 dtype 정합 (대역 + 실제 5행) ──────────────────────────────
band = pd.read_parquet(f"{B67}/cache/test_frame99.parquet")
band["cnt_state"] = (band.balls_before.astype(int) * 3
                     + band.strikes_before.astype(int))
real = build_real(pd.read_csv(os.path.expanduser("~/SungjinPark/data/test.csv"),
                              encoding="utf-8-sig"))
s2 = {}
for c in CATS:
    lv = set(Mm["levels"][c])
    s2[c] = {"n_levels_train": len(lv),
             "dtype_band": str(band[c].dtype) if c in band.columns else "MISSING",
             "dtype_real": str(real[c].dtype),
             "band_unmapped_rows": (int((~band[c].astype(str).isin(lv)).sum())
                                    if c in band.columns else None),
             "real_unmapped_rows": int((~real[c].astype(str).isin(lv)).sum())}
R["S2_cat_vocab"] = s2
R["S2_pass"] = all(v["band_unmapped_rows"] == 0 and v["real_unmapped_rows"] == 0
                   for v in s2.values())

# ── S3 — 실제 대회 표본 5행 전방향 전파 ──────────────────────────────────────
p_real = {}
for sd in SEEDS:
    p_real[f"seed_{sd}"] = nn_infer.predict_nn(
        real, f"{SHIP}/nn_sp{sd}.npz", f"{SHIP}/nn_sp{sd}_meta.json").tolist()
R["S3_real5"] = {"n": len(real), "row_ids": real["row_id"].tolist(),
                 "preds": p_real, "ins_p_n": real["ins_p_n"].tolist(),
                 "all_finite": bool(np.isfinite(
                     np.array(list(p_real.values()))).all()),
                 "🔴 대역_단독_판정_금지": "실제 표본 5행에서 ins_p_n 이 0 이 아닌 행이 "
                                          "있어야 시즌-내 폼 계열이 살아 있음이 확인된다."}

# ── S4 — 추론 시간 (🔴 경합 없는 단독 실행) ─────────────────────────────────
t0 = time.time()
acc = [lg(nn_infer.predict_nn(band, f"{SHIP}/nn_sp{sd}.npz",
                              f"{SHIP}/nn_sp{sd}_meta.json")) for sd in SEEDS]
p_band = 1.0 / (1.0 + np.exp(-np.mean(acc, axis=0)))
t_nn = time.time() - t0
R["S4_inference"] = {
    "n_rows": int(len(band)), "n_seeds": len(SEEDS),
    "nn_forward_total_s": round(t_nn, 2),
    "nn_forward_per_seed_s": round(t_nn / len(SEEDS), 2),
    "exp067_reference_s": 8.6,
    "note": "프레임은 v6 팔이 이미 만든 것을 재사용하므로 피처 생성 추가비용 0.",
    "pred_mean": float(p_band.mean()), "pred_sd": float(p_band.std())}
np.save(f"{OUT}/preds/BAND_k{K}_ens.npy", p_band)

# 대역 프레임 위 exp_067(25완주) vs exp_068(k=8) 예측 차이 — 서술용
b67 = np.load(f"{B67}/out/preds/BAND_deploy_seedavg.npy")
R["band_contrast_vs_exp067"] = {
    "🔴": "이건 두 후보의 **예측 차이**이지 점수 차가 아니다(S9).",
    "mean_abs_prob_diff": float(np.mean(np.abs(b67 - p_band))),
    "max_abs_prob_diff": float(np.max(np.abs(b67 - p_band))),
    "pearson_prob": float(np.corrcoef(b67, p_band)[0, 1]),
    "mean_exp067": float(b67.mean()), "mean_exp068_k8": float(p_band.mean()),
    "sd_exp067": float(b67.std()), "sd_exp068_k8": float(p_band.std())}

files = []
for sd in SEEDS:
    for f in (f"nn_sp{sd}.npz", f"nn_sp{sd}_meta.json"):
        p = f"{SHIP}/{f}"
        files.append({"name": f, "bytes": os.path.getsize(p), "sha256": sha(p)})
R["files"] = {"n": len(files), "total_bytes": sum(f["bytes"] for f in files),
              "list": files}
R["source_sha"] = {"mate/nn_infer.py": sha(f"{MATE}/nn_infer.py"),
                   "mate/pipeline.py": sha(f"{MATE}/pipeline.py"),
                   "mate/splits.py": sha(f"{MATE}/splits.py"),
                   "mate/inseason.py": sha(f"{MATE}/inseason.py"),
                   "mate/model/features_sp.txt": sha(f"{MDIR}/features_sp.txt")}
json.dump(R, open(f"{OUT}/ship068.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps({k: v for k, v in R.items() if k not in ("S3_real5", "files")},
                 indent=1, ensure_ascii=False))

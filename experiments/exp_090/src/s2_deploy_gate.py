"""exp_090 stage 2 게이트 — 배포 npz 가 팀원 `mate/deep_infer.py` 로 **수정 없이** 돌아가는가.

사용: s2_deploy_gate.py <TAGPREFIX>        (예: D1797 → out/npz/deepf_D1797_le2024_s{seed}.npz)

검사 5개 (하나라도 실패하면 PASS=False):
  G1 키·형상·dtype 이 팀원 배포 아티팩트(`deepf_s42.npz`) 와 **전부 같다** (17키)
  G2 우리 `prep.json` 의 NUM·CAT·levels·mu·sd 가 팀원 것과 **완전 동일** (입력 변환이 안 바뀐다)
  G3 팀원 `predict_deep` 이 우리 prep+npz 로 종단 실행 rc=0, 출력 [0,1]·NaN 0·유한
  G4 그 출력이 우리 torch 경로의 5시드 로짓평균과 일치 (max|Δ| < 1e-4)
  G5 규칙 2-4 — 행 순열을 넣었다 빼면 예측이 원래대로 (행 독립성)
"""
import json, os, sys, time, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B90 = f"{H}/exp090"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
E60 = f"{H}/exp060_nn"
sys.path.insert(0, PKG)
import deep_infer as DI                                    # noqa: E402

PRE = sys.argv[1]
S5 = [42, 777, 2024, 101, 7]
t0 = time.time()
MATE_PREP = f"{PKG}/model/deep/prep.json"
OUR_PREP = f"{B90}/out/deploy_{PRE}/prep.json"
NPZ = [f"{B90}/out/deploy_{PRE}/deepf_s{s}.npz" for s in S5]
G = {"tag": PRE, "npz": NPZ, "prep": OUR_PREP}

# ── G1 키·형상·dtype ───────────────────────────────────────────────────
mate = np.load(f"{PKG}/model/deep/deepf_s42.npz")
ref = {k: (list(mate[k].shape), str(mate[k].dtype)) for k in mate.files}
g1 = {}
for p in NPZ:
    d = np.load(p)
    got = {k: (list(d[k].shape), str(d[k].dtype)) for k in d.files}
    g1[os.path.basename(p)] = {"n_keys": len(got), "identical_to_mate": got == ref,
                               "missing": sorted(set(ref) - set(got)),
                               "extra": sorted(set(got) - set(ref)),
                               "mismatch": {k: {"ours": got[k], "mate": ref[k]}
                                            for k in sorted(set(ref) & set(got))
                                            if got[k] != ref[k]}}
G["G1_keys_shapes_dtypes"] = {"mate_reference": ref, "per_file": g1,
                              "PASS": all(v["identical_to_mate"] for v in g1.values())}

# ── G2 prep 입력 변환 동일성 ───────────────────────────────────────────
Pm, Po = json.load(open(MATE_PREP)), json.load(open(OUR_PREP))
same = {k: (Pm[k] == Po[k]) for k in ("NUM", "CAT", "levels", "mu", "sd", "D", "batch")}
G["G2_prep_input_transform"] = {
    "identical_fields": same, "PASS": all(same.values()),
    "changed_fields": {k: {"mate": Pm.get(k), "ours": Po.get(k)}
                       for k in sorted(set(Pm) | set(Po))
                       if Pm.get(k) != Po.get(k)}}

# ── 프레임 (배포 프레임 대역: season==2024 전체 253,507행) ─────────────────
NUMc, CATc = Po["NUM"], Po["CAT"]
df = pd.read_parquet(f"{E60}/cache/frame99.parquet",
                     columns=list(dict.fromkeys(NUMc + CATc + ["row_id", "season"])))
tr = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id", "season"])
assert np.array_equal(df.row_id.to_numpy().astype(str),
                      tr.row_id.to_numpy().astype(str))
m = tr.season.to_numpy() == 2024
dfv = df.loc[m].reset_index(drop=True)
assert len(dfv) == 253507

# ── G3 종단 실행 ───────────────────────────────────────────────────────
rc, err = 0, None
try:
    p = DI.predict_deep(dfv, OUR_PREP, NPZ)
except Exception as e:                                     # noqa: BLE001
    rc, err, p = 1, repr(e), None
G["G3_end_to_end"] = {"rc": rc, "error": err}
if rc == 0:
    G["G3_end_to_end"].update({
        "n": int(len(p)), "n_nan": int(np.isnan(p).sum()),
        "n_nonfinite": int((~np.isfinite(p)).sum()),
        "min": float(p.min()), "max": float(p.max()), "mean": float(p.mean()),
        "in_unit_interval": bool(p.min() >= 0.0 and p.max() <= 1.0),
        "PASS": bool(len(p) == 253507 and np.isfinite(p).all()
                     and p.min() >= 0.0 and p.max() <= 1.0)})
else:
    G["G3_end_to_end"]["PASS"] = False

# ── G4 torch 경로와 대조 ───────────────────────────────────────────────
if rc == 0:
    zt = np.mean([np.load(f"{B90}/out/preds/z_{PRE}_le2024_s{s}.npy") for s in S5], 0)
    zn = np.log(p / (1 - p))
    md = float(np.abs(zt - zn).max())
    G["G4_vs_torch_path"] = {"max_abs_logit_diff": md,
                             "corr": float(np.corrcoef(zt, zn)[0, 1]),
                             "PASS": bool(md < 1e-4)}
else:
    G["G4_vs_torch_path"] = {"PASS": False}

# ── G5 행 독립성 (규칙 2-4) ────────────────────────────────────────────
if rc == 0:
    rng = np.random.default_rng(90090)
    perm = rng.permutation(len(dfv))
    p2 = DI.predict_deep(dfv.iloc[perm].reset_index(drop=True), OUR_PREP, NPZ)
    back = np.empty_like(p2); back[perm] = p2
    G["G5_row_independence"] = {
        "max_abs_diff": float(np.abs(back - p).max()),
        "n_rows_moving": int((np.abs(back - p) > 0).sum()),
        "PASS": bool(np.abs(back - p).max() < 1e-12)}
else:
    G["G5_row_independence"] = {"PASS": False}

G["ALL_PASS"] = all(G[k]["PASS"] for k in
                    ("G1_keys_shapes_dtypes", "G2_prep_input_transform",
                     "G3_end_to_end", "G4_vs_torch_path", "G5_row_independence"))
G["npz_sha256"] = {os.path.basename(q): hashlib.sha256(open(q, "rb").read()).hexdigest()
                   for q in NPZ}
G["wall_sec"] = round(time.time() - t0, 1)
json.dump(G, open(f"{B90}/out/gate_{PRE}.json", "w"), ensure_ascii=False, indent=1)
pr = dict(G)
pr["G1_keys_shapes_dtypes"] = {k: v for k, v in G["G1_keys_shapes_dtypes"].items()
                               if k != "mate_reference"}
print(json.dumps(pr, ensure_ascii=False, indent=1))

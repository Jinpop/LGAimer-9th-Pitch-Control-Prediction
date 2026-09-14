"""exp_113 — 열 선별 + 판본 블록 제작. 🔴 라벨을 한 번도 안 본다.

사전 등록 선별 규칙 (결과를 보기 전에 확정 — S8):
  ① 기존 94열과의 **max|ρ| <= 0.90**            (S174 의 준중복 함정)
  ② 기존 94열의 **선형 스팬 밖 분산 비율 1−R² >= 0.10**
     🔑 왜 ρ 만으로는 부족한가: 딥 팔의 첫 은닉층은 수치 입력에 대해 정확히 선형이다
        (arch097.TokenMLP: t = x·w + b → concat → Linear). 기존 94열의 **선형결합**은
        표현력을 한 톨도 안 늘린다. 개별 ρ 가 낮아도 94열의 조합으로 재구성되면 무의미하다.
        OLS 는 ≤2023 행에서 적합하고 R² 는 2024 행에서 잰다 (표본 밖).
  ③ 살아남은 것들끼리 |ρ| > 0.95 면 앞에 열거된 쪽만 남긴다.
  판정은 **post-fill 표준화 값**으로 한다 — 그것이 신경망이 실제로 보는 벡터다.

판본 블록:
  P     그룹 P 생존열 (구종 믹스 in-season)
  R     그룹 R 생존열 (비선형 재조합)
  PR    둘 다
  N1_*  **같은 개수의 무작위 열** = 기존 94열 중 무작위 k개를 (season, pitcher_id) 안에서 순열
        🔴 이것이 기준선이다. 무변형 대비가 아니라 이것 대비로 읽는다 (S183).
  N2_*  같은 열을 (season, pitcher_id) 안에서 순열 — 형태·결측 패턴 보존, 정보만 제거
  N2b_PR 같은 열을 pitcher_id 안에서만 순열 (시즌을 가로지른다 — 더 무른 귀무, S182)
  🔴 귀무를 둘 이상 쓰고 판정이 뒤집히는지 본다 (S179).

표준화: 배포 `mate/deep_infer.py` 규약 그대로 — (x−mu)/sd 후 nan→0.
        mu·sd 는 **학습 행(≤2023, trm_f64o)에서만**, float64 누산 (S134).
        (season, pitcher_id) 안 순열은 ≤2023 의 열 다중집합을 보존하므로 mu·sd 가 불변이다.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp113"
W102 = f"{H}/exp102/work"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
OUT = f"{B}/work/blk"
os.makedirs(OUT, exist_ok=True)

RHO_MAX = 0.90
SPAN_MIN = 0.10
SEED_N1 = 20260827

NUM94 = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))["NUM"]
CJ = json.load(open(f"{B}/work/pfeat_cols.json"))
COLS, GP, GR = CJ["cols"], CJ["group_P"], CJ["group_R"]
F = np.load(f"{B}/work/pfeat.npy").astype(np.float64)

X94 = np.asarray(np.load(f"{W102}/X_f64o.npy", mmap_mode="r"), np.float64)
trm = np.load(f"{W102}/trm_f64o.npy")
vam = np.load(f"{W102}/valid_mask.npy")
N = X94.shape[0]
assert F.shape[0] == N and X94.shape[1] == 94

meta = pd.read_csv(f"{H}/data/train.csv", usecols=["season", "pitcher_id"],
                   encoding="utf-8-sig")
season = meta["season"].to_numpy(np.int32)
pid = meta["pitcher_id"].to_numpy(np.int64)
assert (season[trm] <= 2023).all() and (season[vam] == 2024).all()


def standardize(A):
    mu = np.nanmean(A[trm], axis=0)
    sd = np.nanstd(A[trm], axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    with np.errstate(invalid="ignore"):
        Z = (A - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), mu, sd


Zc, MU, SD = standardize(F)

# ── ① max|rho| vs 94 ─────────────────────────────────────────────
Xs = (X94 - X94[trm].mean(0)) / np.where(X94[trm].std(0) > 1e-12, X94[trm].std(0), 1.0)
rep = {"exp_id": "exp_113", "stage": "f3_screen",
       "prereg_rule": {"max_abs_rho_le": RHO_MAX, "span_resid_ge": SPAN_MIN,
                       "within_cand_dup_rho": 0.95,
                       "판정면": "post-fill 표준화 값 (신경망이 실제로 보는 벡터)"},
       "cols": {}}


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    da, db = a.std(), b.std()
    return float((a * b).mean() / (da * db)) if da > 1e-15 and db > 1e-15 else 0.0


# ── ② 선형 스팬 (OLS on <=2023, R^2 on 2024) ────────────────────
A = np.hstack([Xs[trm], np.ones((int(trm.sum()), 1))])
G = A.T @ A + 1e-8 * np.eye(95)
Av = np.hstack([Xs[vam], np.ones((int(vam.sum()), 1))])
for j, c in enumerate(COLS):
    yv = Zc[:, j]
    beta = np.linalg.solve(G, A.T @ yv[trm])
    res = yv[vam] - Av @ beta
    tot = float(((yv[vam] - yv[vam].mean()) ** 2).mean())
    span = float((res ** 2).mean() / tot) if tot > 1e-15 else 0.0
    rhos = np.array([corr(yv, Xs[:, k]) for k in range(94)])
    k = int(np.argmax(np.abs(rhos)))
    rep["cols"][c] = {"group": "P" if c in GP else "R",
                      "max_abs_rho": float(abs(rhos[k])), "vs": NUM94[k],
                      "span_resid_frac_2024": span,
                      "sd_train_raw": float(SD[j]), "mu_train_raw": float(MU[j]),
                      "nan_frac_2024": float(np.isnan(F[vam, j]).mean()),
                      "PASS_rho": bool(abs(rhos[k]) <= RHO_MAX),
                      "PASS_span": bool(span >= SPAN_MIN)}
    rep["cols"][c]["KEEP_stage12"] = bool(rep["cols"][c]["PASS_rho"]
                                          and rep["cols"][c]["PASS_span"])

# ── ③ 후보끼리 준중복 제거 ───────────────────────────────────────
cand = [c for c in COLS if rep["cols"][c]["KEEP_stage12"]]
kept, dropped = [], {}
for c in cand:
    j = COLS.index(c)
    bad = None
    for k in kept:
        r = corr(Zc[:, j], Zc[:, COLS.index(k)])
        if abs(r) > 0.95:
            bad = {"vs": k, "rho": float(r)}
            break
    if bad:
        dropped[c] = bad
    else:
        kept.append(c)
rep["dropped_stage12"] = {c: {"max_abs_rho": rep["cols"][c]["max_abs_rho"],
                              "vs": rep["cols"][c]["vs"],
                              "span_resid_frac_2024": rep["cols"][c]["span_resid_frac_2024"]}
                          for c in COLS if not rep["cols"][c]["KEEP_stage12"]}
rep["dropped_stage3_dup"] = dropped
SEL_P = [c for c in kept if c in GP]
SEL_R = [c for c in kept if c in GR]
rep["SEL_P"], rep["SEL_R"], rep["SEL_PR"] = SEL_P, SEL_R, kept

# ── 순열 벡터 ────────────────────────────────────────────────────
rs = np.random.RandomState(SEED_N1)


def group_perm(keys):
    p = np.arange(N)
    o = np.lexsort((np.arange(N), keys))
    ks = keys[o]
    bnd = np.concatenate([[0], np.where(np.diff(ks) != 0)[0] + 1, [N]])
    for i in range(len(bnd) - 1):
        seg = o[bnd[i]:bnd[i + 1]]
        p[seg] = rs.permutation(seg)
    return p


key_sp = pid.astype(np.int64) * 10000 + season
PERM_SP = group_perm(key_sp)          # (season, pitcher_id) 안
PERM_P = group_perm(pid)              # pitcher_id 안 (시즌 가로지름)

# ── 블록 저장 ────────────────────────────────────────────────────
rep["blocks"] = {}


def save_blk(tag, Araw, note, colnames):
    Z, mu, sd = standardize(Araw)
    Z = Z.astype(np.float32)
    np.save(f"{OUT}/{tag}.npy", Z)
    rep["blocks"][tag] = {"k": int(Z.shape[1]), "cols": colnames, "note": note,
                          "mu_train": [float(x) for x in mu],
                          "sd_train": [float(x) for x in sd],
                          "post_fill_sd_2024": [float(Z[vam, j].std())
                                                for j in range(Z.shape[1])],
                          "nan_frac_2024_pre_fill": [float(np.isnan(Araw[vam, j]).mean())
                                                     for j in range(Z.shape[1])]}


def raw(cols):
    return F[:, [COLS.index(c) for c in cols]]


N1_IDX = {}
for tag, cols in (("P", SEL_P), ("R", SEL_R), ("PR", kept)):
    if not cols:
        continue
    save_blk(tag, raw(cols), "실제 열", cols)
    save_blk(f"N2_{tag}", raw(cols)[PERM_SP],
             "귀무 N2 — 같은 열을 (season, pitcher_id) 안에서 순열", cols)
    idx = sorted(rs.choice(94, len(cols), replace=False).tolist())
    N1_IDX[tag] = idx
    save_blk(f"N1_{tag}", X94[:, idx][PERM_SP],
             "귀무 N1(기준선) — 기존 94열 중 무작위 k개를 (season, pitcher_id) 안에서 순열",
             [NUM94[i] for i in idx])
save_blk("N2b_PR", raw(kept)[PERM_P],
         "귀무 N2b — pitcher_id 안에서만 순열 (시즌을 가로지른다, 더 무른 귀무)", kept)
rep["N1_random_column_index"] = N1_IDX
rep["perm_seed"] = SEED_N1
json.dump(rep, open(f"{B}/out/f3_screen.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != "blocks"},
                 ensure_ascii=False, indent=1))
print(json.dumps({t: {"k": v["k"], "cols": v["cols"]} for t, v in rep["blocks"].items()},
                 ensure_ascii=False, indent=1))

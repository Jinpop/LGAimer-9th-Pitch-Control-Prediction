"""exp_113 — 🔴 귀무 재제작. 1차 귀무(N1·N2)가 귀무가 아니었다.

발견 (실측, 재학습 0):
  `(season, pitcher_id)` **안** 순열은 그 열의 라벨 상관을 거의 그대로 남긴다 —
  2024 행 corr(열, control_success) 원본 → 순열:
     ins_p_ball      −0.04431 → −0.04410   (99.5% 잔존)
     ins_p_strike    +0.03281 → +0.03327   (101%)
     sp_pit_bhand_all +0.02891 → +0.02891  (투수-시즌 안 상수라 정확히 동일)
     r2_lg_form      +0.02126 → +0.02662   (🔴 원본보다 **강해진다**)
  기전 둘: ① 투수-시즌 집계가 신호의 담지자라 같은 투수-시즌 안에서 섞어도 안 없어진다
          ② 누적 asof 열은 시즌 뒷부분 값이 그 투수의 **시즌 전체** 수준에 더 가까워,
             섞으면 오히려 덜 시끄러운(그리고 미래를 엿본) 추정치가 된다.
  반면 **season 안** 순열은 라벨 상관을 죽인다 (전부 |corr| <= 0.0043).

  그래서 1차 N1(기존 열 무작위 k개를 (season,pitcher) 안 순열)은 귀무가 아니라
  **강한 팔**이었다 — 정직면 이득 39.25 로 94열 기준선 26.27 을 크게 웃돌았다.
  ⚠️ 이 결함은 exp_103b 의 `null3s`·`rand3`((season, batter_id) 안 순열)와 같은 계열이다.

고침 — 귀무 셋을 다시 짓는다:
  N1s_*  기존 94열 중 무작위 k개를 **season 안** 순열   ← 🔴 새 기준선 (주변분포 일치, 연관 0)
  N1g_PR **iid N(0,1) k열**                              ← 구조가 다른 둘째 귀무 (토큰 수 대가만)
  N2s_*  후보 열을 **season 안** 순열                    ← 주변분포·시즌별 결측 패턴 보존

🔴 사전 등록 문턱은 안 건드린다 (S8) — 바꾸는 것은 귀무의 **정의**이고, 그 이유는
   결과가 아니라 위 라벨 상관 실측이다. 1차 귀무 수치는 버리지 않고 전부 보고한다.
"""
import json
import os

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp113"
W102 = f"{H}/exp102/work"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
OUT = f"{B}/work/blk"
SEED = 20260827

NUM94 = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))["NUM"]
S = json.load(open(f"{B}/out/f3_screen.json"))
CJ = json.load(open(f"{B}/work/pfeat_cols.json"))
COLS = CJ["cols"]
F = np.load(f"{B}/work/pfeat.npy").astype(np.float64)
X94 = np.asarray(np.load(f"{W102}/X_f64o.npy", mmap_mode="r"), np.float64)
trm = np.load(f"{W102}/trm_f64o.npy")
vam = np.load(f"{W102}/valid_mask.npy")
N = X94.shape[0]
meta = pd.read_csv(f"{H}/data/train.csv", usecols=["season", "pitcher_id"],
                   encoding="utf-8-sig")
season = meta["season"].to_numpy(np.int32)
y = np.load(f"{W102}/y.npy").astype(np.float64)


def standardize(A):
    mu = np.nanmean(A[trm], axis=0)
    sd = np.nanstd(A[trm], axis=0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    with np.errstate(invalid="ignore"):
        Z = (A - mu) / sd
    return np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0), mu, sd


rs = np.random.RandomState(SEED)


def gperm(keys):
    p = np.arange(N)
    o = np.lexsort((np.arange(N), keys))
    ks = keys[o]
    b = np.concatenate([[0], np.where(np.diff(ks) != 0)[0] + 1, [N]])
    for i in range(len(b) - 1):
        seg = o[b[i]:b[i + 1]]
        p[seg] = rs.permutation(seg)
    return p


PERM_S = gperm(season.astype(np.int64))

rep = {"exp_id": "exp_113", "stage": "f3b_nulls", "perm_seed": SEED,
       "왜": "(season, pitcher_id) 안 순열이 라벨 상관을 보존한다 — 귀무가 아니었다",
       "blocks": {}}
yv = y[vam] - y[vam].mean()


def labcorr(v):
    q = v[vam] - v[vam].mean()
    s = q.std()
    return float((q * yv).mean() / (s * yv.std())) if s > 1e-12 else 0.0


def save(tag, Araw, note, colnames):
    Z, mu, sd = standardize(Araw)
    np.save(f"{OUT}/{tag}.npy", Z.astype(np.float32))
    rep["blocks"][tag] = {"k": int(Z.shape[1]), "cols": colnames, "note": note,
                          "label_corr_2024": [labcorr(Z[:, j]) for j in range(Z.shape[1])],
                          "max_abs_label_corr": float(max(abs(labcorr(Z[:, j]))
                                                          for j in range(Z.shape[1])))}


SETS = {"P": S["SEL_P"], "R": S["SEL_R"], "PR": S["SEL_PR"]}
idx_new = {}
for tag, cols in SETS.items():
    k = len(cols)
    raw = F[:, [COLS.index(c) for c in cols]]
    save(f"N2s_{tag}", raw[PERM_S],
         "귀무 N2s — 후보 열을 season 안에서 순열", cols)
    idx = sorted(rs.choice(94, k, replace=False).tolist())
    idx_new[tag] = idx
    save(f"N1s_{tag}", X94[:, idx][PERM_S],
         "귀무 N1s(새 기준선) — 기존 94열 중 무작위 k개를 season 안에서 순열",
         [NUM94[i] for i in idx])
save("N1g_PR", rs.randn(N, len(S["SEL_PR"])),
     "귀무 N1g — iid N(0,1) 13열 (토큰 수 대가만)",
     [f"gauss{i}" for i in range(len(S["SEL_PR"]))])

# 참고: 실제 후보 열과 1차 귀무의 라벨 상관도 같이 남긴다
for tag, cols in SETS.items():
    Z, _, _ = standardize(F[:, [COLS.index(c) for c in cols]])
    rep.setdefault("real_label_corr_2024", {})[tag] = {
        c: labcorr(Z[:, j]) for j, c in enumerate(cols)}
rep["N1s_random_column_index"] = idx_new
json.dump(rep, open(f"{B}/out/f3b_nulls.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(rep, ensure_ascii=False, indent=1))

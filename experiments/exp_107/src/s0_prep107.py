"""exp_107 s0 — **타깃 벡터를 전부 여기서 만든다.** 학습 코드는 통로 하나(TGT107)만 쓴다.

축 A 라벨 스무딩       t = (1-e)·y + e·0.5                      (학습 행 전부)
축 B 자기 증류         t = (1-e)·y + e·p_self                   (커버된 행만, 나머지는 hard y)
축 C 혼합              t = (1-eA-eB)·y + eA·0.5 + eB·p_self     (커버 밖은 A 만)
귀무 N1a               (season, pitcher_id) 안에서 순열한 라벨, 스무딩 없음  ← 바닥 대조의 짝
귀무 N1b               같은 순열 라벨 + 같은 eA 스무딩
      🔴 N1 은 **기댓값이 같은 귀무가 아니다**(라벨 정보를 통째로 없앤다). 바닥 대조이고,
         쓸모는 **쌍(N1b − N1a)** 이다 — 신호가 없는 팔 위에서 스무딩만의 몫을 준다.
귀무 N2                p_self 를 커버 행 안에서 무작위 치환      ← B 의 진짜 귀무(타깃 잡음만 남는다)
귀무 N3                y 그대로 (무변형 항등 복제본, S135)       ← 사다리 맨 아래

p_self 출처: exp_092 `work/offF_all.npy` = **시즌 전진 연쇄 OOF**(S106) 의 로짓.
             p_succ = 1 − Σ(실패 4셀), 시즌 s 행은 `season < s` 로만 적합한 모델이 예측했다.
             커버리지 구멍은 2019(연쇄의 첫 시즌) — 그 행은 hard label 로 둔다.
🔴 valid(2024) 타깃은 어떤 축에서도 안 바뀐다. 점수는 언제나 참 라벨로 잰다.
"""
import hashlib, json, os
import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/exp107"
W = f"{H}/exp097/work"
W92 = f"{H}/exp092/work"
os.makedirs(f"{B}/work", exist_ok=True)
os.makedirs(f"{B}/out", exist_ok=True)

EA = [0.02, 0.05, 0.10, 0.20]          # 축 A 격자 (config)
EB = [0.10, 0.20, 0.40]                # 축 B 격자
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

y = np.load(f"{W}/y.npy")                       # float32, 1,475,092
trm = np.load(f"{W}/trm_f64o.npy")
vam = np.load(f"{W}/valid_mask.npy")
assert y.dtype == np.float32 and len(y) == 1475092
trm23 = np.load(f"{H}/exp089/work/trm_oot23.npy")
assert np.array_equal(trm, trm23), "🔴 f64o 학습 마스크 != oot23 (p_self 게이트의 전제)"

off = np.load(f"{W92}/offF_all.npy")            # float64 로짓, 미커버는 NaN
assert len(off) == len(y)
cov = np.isfinite(off)
p_self = np.full(len(y), np.nan)
p_self[cov] = 1.0 / (1.0 + np.exp(-off[cov]))

tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "pitcher_id", "control_success"])
assert np.array_equal(tr.control_success.to_numpy(np.float32), y), "🔴 y 정렬 불일치"
season = tr.season.to_numpy(np.int64)
pit = tr.pitcher_id.to_numpy()

yd = y.astype(np.float64)
COVTR = cov & trm
REP = {
    "exp_id": "exp_107", "stage": "s0_prep107",
    "p_self_source": {
        "path": f"{W92}/offF_all.npy",
        "what": "exp_092 s0_offF — oof_new/stack_fold2024_seed{1,13,42,101}.npz 4시드 평균, "
                "p_succ = 1 - sum(4 실패셀), offset = logit(clip(p_succ,1e-6,1-1e-6))",
        "off_sha16": sha(np.nan_to_num(off)),
        "p_self_train_mean": float(np.nanmean(p_self[COVTR])),
        "p_self_train_sd": float(np.nanstd(p_self[COVTR]))},
    "coverage": {
        "n_train_rows": int(trm.sum()),
        "n_train_rows_covered": int(COVTR.sum()),
        "covered_share": float(COVTR.sum() / trm.sum()),
        "uncovered_share_🔴": float(1.0 - COVTR.sum() / trm.sum()),
        "per_season": {str(s): {"n": int(((season == s) & trm).sum()),
                                "covered_share": float(cov[(season == s) & trm].mean())}
                       for s in sorted(set(season[trm].tolist()))},
        "🔴": "미커버 행(연쇄 첫 시즌 2019)은 hard label 로 둔다 — 축 B 의 e 는 커버 행에만 닿는다",
        "valid_untouched": True},
    "grids": {"A_eps": EA, "B_eps": EB},
    "targets": {}}


def emit(name, t, note):
    t = np.asarray(t, np.float64)
    assert np.isfinite(t).all() and t.min() >= 0.0 and t.max() <= 1.0
    t32 = t.astype(np.float32)
    ch = (t32 != y) & trm
    np.save(f"{B}/work/tgt_{name}.npy", t32)
    REP["targets"][name] = {
        "note": note,
        "n_train_rows_changed": int(ch.sum()),
        "share_train_rows_changed": float(ch.sum() / trm.sum()),
        "mean_abs_shift_train": float(np.abs(t32[trm].astype(np.float64) - yd[trm]).mean()),
        "target_mean_train": float(t32[trm].astype(np.float64).mean()),
        "hard_mean_train": float(yd[trm].mean()),
        "n_distinct_values_train": int(len(np.unique(t32[trm]))),
        "sha16_train_slice": sha(t32[trm])}
    print(f"{name:14s} changed {ch.sum():>9,} ({ch.sum()/trm.sum():6.2%})  "
          f"mean|Δ| {REP['targets'][name]['mean_abs_shift_train']:.6f}", flush=True)


# ── N3 항등 (사다리 맨 아래, S135) ────────────────────────────────────────
emit("N3_identity", yd, "무변형 — y 그대로. train107 기본 경로와 바이트 동일해야 한다")

# ── 축 A ─────────────────────────────────────────────────────────────────
for e in EA:
    emit(f"A{e:g}".replace(".", "p"), (1 - e) * yd + e * 0.5,
         f"라벨 스무딩 e={e} — 전 학습 행")

# ── 축 B ─────────────────────────────────────────────────────────────────
ps = np.where(cov, np.nan_to_num(p_self), 0.0)
for e in EB:
    t = yd.copy()
    t[COVTR] = (1 - e) * yd[COVTR] + e * ps[COVTR]
    emit(f"B{e:g}".replace(".", "p"), t,
         f"자기 증류 e={e} — 커버 행만 (시즌 전진 연쇄 OOF p_self), 미커버는 hard")

# ── 귀무 N2: p_self 를 커버 행 안에서 무작위 치환 ─────────────────────────
rng = np.random.default_rng(20260826)
idx = np.flatnonzero(COVTR)
perm = rng.permutation(idx)
ps_sh = ps.copy()
ps_sh[idx] = ps[perm]
REP["N2_shuffle"] = {"n_permuted": int(len(idx)),
                     "rho_shuffled_vs_orig": float(np.corrcoef(ps[idx], ps_sh[idx])[0, 1]),
                     "mean_preserved": bool(abs(ps[idx].mean() - ps_sh[idx].mean()) < 1e-12)}
for e in EB:
    t = yd.copy()
    t[COVTR] = (1 - e) * yd[COVTR] + e * ps_sh[COVTR]
    emit(f"N2_B{e:g}".replace(".", "p"), t,
         f"귀무 N2 — p_self 를 커버 행 안에서 무작위 치환, e={e}. 타깃 잡음의 크기는 같고 정보는 0")

# ── 귀무 N1: (season, pitcher_id) 안 라벨 순열 ────────────────────────────
rng2 = np.random.default_rng(20260827)
key = pd.MultiIndex.from_arrays([season, pit])
yp = yd.copy()
dfp = pd.DataFrame({"g": key.factorize()[0], "y": yd, "tr": trm})
gi = dfp.g.to_numpy()
tri = np.flatnonzero(trm)
o = np.argsort(gi[tri], kind="stable")
tri_s = tri[o]
gs = gi[tri_s]
bnd = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
ends = np.r_[bnd[1:], len(gs)]
newy = yd[tri_s].copy()
for a, b in zip(bnd, ends):
    if b - a > 1:
        newy[a:b] = newy[a:b][rng2.permutation(b - a)]
yp[tri_s] = newy
REP["N1_permute"] = {
    "n_groups_(season,pitcher)": int(len(bnd)),
    "n_train_rows_permuted": int(len(tri_s)),
    "share_labels_actually_moved": float((yp[trm] != yd[trm]).mean()),
    "group_mean_preserved_max_abs_err": float(
        np.abs(pd.Series(yp[tri_s]).groupby(gs).mean()
               - pd.Series(yd[tri_s]).groupby(gs).mean()).max()),
    "🔴": "이것은 기댓값이 같은 귀무가 아니라 **바닥 대조**다. 쓸모는 쌍 (N1b − N1a) 이고, "
          "그것이 '신호 없는 팔 위에서 스무딩만의 몫' 이다"}
emit("N1a_perm_e0", yp, "귀무 N1a — (season,pitcher) 안 라벨 순열, 스무딩 없음")

REP["wrote"] = sorted(REP["targets"])
json.dump(REP, open(f"{B}/out/s0_prep107.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: REP[k] for k in ("coverage", "N1_permute", "N2_shuffle")},
                 ensure_ascii=False, indent=1))
np.save(f"{B}/work/_yperm.npy", yp.astype(np.float32))
np.save(f"{B}/work/_pself.npy", np.nan_to_num(p_self).astype(np.float64))
np.save(f"{B}/work/_cov.npy", cov)
print("PREP_DONE")

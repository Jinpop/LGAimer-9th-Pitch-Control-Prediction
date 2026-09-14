"""exp_083 캡처 2 — 파트너 v6 팔 + EBM 의 per-row 중간값 포획.  학습 0건.

v6_infer.run() 의 프레임 구축·추론 산술을 그대로 복제한 뒤, 배포 코드가
★이미 계산하고 버리는★ 값만 추가로 꺼낸다.
  B  EBM 항별 기여      ebm_core.score_numpy 의 `z += ts[...]` 항 (가산모델)
  C  Leaf 경로          nn_infer.predict_leaf 의 잎 인덱스 / LR 부분합 / OOV 수
  D  NN 은닉 64         nn_infer.predict_nn 의 마지막 은닉층 (lin(z,8) 뒤 relu)
  E  시드 산포          lgb 3 · cat 4 · nn 4 · leaf 3 시드의 로짓 분산

게이트: 복제한 4팔 블렌드가 저장된 v6 벡터와 일치하는가 (산술 동일성 증명).
🔴 규칙 2-4: 전부 행 i 자신의 값.  행 방향 집계 없음.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import sys, json, pickle, time                                      # noqa: E402
import numpy as np                                                  # noqa: E402
import pandas as pd                                                 # noqa: E402

WORK = sys.argv[1]
OUTNPZ = sys.argv[2]
os.chdir(WORK)
sys.path.insert(0, WORK)
T0 = time.time()

import lightgbm as lgb                                              # noqa: E402
from catboost import CatBoostClassifier                             # noqa: E402
from scipy.special import logit, expit                              # noqa: E402
import pipeline as P                                                # noqa: E402
import nn_infer                                                     # noqa: E402
import splits, inseason                                             # noqa: E402
import ebm_core as EC                                               # noqa: E402

MODEL_DIR, TEST_DIR = "./model", "./data"
TBL_DIR = os.path.join(MODEL_DIR, "tables")
CAT_COLS = ['game_type', 'base_state', 'pitcher_team_id', 'batter_team_id',
            'idea_matchup_pitch']
LGB_SEEDS, CAT_SEEDS, NN_SEEDS, LEAF_SEEDS = [42, 777, 2024], [42, 777, 2024, 101], \
    [42, 777, 2024, 101], [42, 777, 2024]
W_LGB, W_CAT, W_NN, W_LEAF = 0.05, 0.68, 0.15, 0.12


def lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


# ══════════════════════ 프레임 (v6_infer.run 83-124 그대로) ══════════════════
test = pd.read_csv(os.path.join(TEST_DIR, "test.csv"), encoding="utf-8-sig")
feats = P.apply_features(test, TBL_DIR)
ratios = P.apply_ratio_features(test, feats)
df = pd.concat([test.reset_index(drop=True), feats.reset_index(drop=True),
                ratios.reset_index(drop=True)], axis=1)
df['cnt_state'] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)
with open(os.path.join(MODEL_DIR, "features.txt")) as f:
    FE = [c for c in f.read().split("\n") if c]
sp_path = os.path.join(MODEL_DIR, "deploy_splits.npz")
FE_SP = FE
if os.path.exists(sp_path):
    S = splits.apply(df, splits.load(sp_path))
    for c in S.columns:
        df[c] = S[c].values
    with open(os.path.join(MODEL_DIR, "features_sp.txt")) as f:
        FE_SP = [c for c in f.read().split("\n") if c]
anchor_path = os.path.join(MODEL_DIR, "anchor.npz")
if os.path.exists(anchor_path):
    I = inseason.apply(df, inseason.load(anchor_path))
    for c in I.columns:
        df[c] = I[c].values
n = len(df)
print(f"[frame] rows={n} FE={len(FE)} FE_SP={len(FE_SP)}", flush=True)
store = {"row_id": test["row_id"].to_numpy().astype(str)}

# ══════════════════════ E: LightGBM 시드별 ═══════════════════════════════════
X = df[FE_SP].copy()
_lvp = os.path.join(MODEL_DIR, "lgb_cats.json")
_lv = json.load(open(_lvp)) if os.path.exists(_lvp) else None
for c in CAT_COLS:
    if c in X.columns:
        X[c] = (pd.Categorical(X[c].astype(str), categories=_lv[c]) if _lv
                else X[c].astype('category'))
LGB = np.stack([lg(lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_{s}.txt")).predict(X))
                for s in LGB_SEEDS])
pl = LGB.mean(axis=0)
store["seed_lgb_sd"] = LGB.std(axis=0, ddof=1)
print(f"[lgb] {LGB.shape}", flush=True)

# ══════════════════════ E: CatBoost 시드별 ═══════════════════════════════════
Xc = df[FE_SP].copy()
for c in CAT_COLS:
    if c in Xc.columns:
        Xc[c] = Xc[c].astype(str)
CAT = []
for s in CAT_SEEDS:
    m = CatBoostClassifier()
    m.load_model(os.path.join(MODEL_DIR, f"cat_{s}.cbm"))
    CAT.append(lg(m.predict_proba(Xc)[:, 1]))
    del m
CAT = np.stack(CAT)
pc = CAT.mean(axis=0)
store["seed_cat_sd"] = CAT.std(axis=0, ddof=1)
print(f"[cat] {CAT.shape}", flush=True)

# ══════════════════════ D: NN 은닉 64 + E: NN 시드별 ═════════════════════════


def nn_forward_capture(df_, npz_path, meta_path, chunk=100000):
    """nn_infer.predict_nn 과 ★같은 산술★.  마지막 은닉층을 추가로 반환."""
    W = np.load(npz_path)
    M = json.load(open(meta_path))
    CATS, DENSE = M['CATS'], M['DENSE']
    mu = np.array(M['mu'], dtype=np.float32)
    sd = np.array(M['sd'], dtype=np.float32)
    codes = np.zeros((len(df_), len(CATS)), dtype=np.int64)
    for i, c in enumerate(CATS):
        lut = {v: j + 1 for j, v in enumerate(M['levels'][c])}
        codes[:, i] = df_[c].astype(str).map(lut).fillna(0).astype(np.int64).values
    Xd = df_[DENSE].astype('float32').values
    Xd = (Xd - mu) / sd
    Xd = np.nan_to_num(Xd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    embs = [W[f'emb{i}'] for i in range(len(CATS))]

    def lin(z, i):
        return z @ W[f'mlp_{i}_weight'].T + W[f'mlp_{i}_bias']

    def bn(z, i):
        g = W[f'mlp_{i}_weight']; b = W[f'mlp_{i}_bias']
        mm = W[f'mlp_{i}_running_mean']; vv = W[f'mlp_{i}_running_var']
        return (z - mm) / np.sqrt(vv + 1e-5) * g + b

    out = np.empty(len(df_), dtype=np.float64)
    H = None
    for s in range(0, len(df_), chunk):
        e = min(s + chunk, len(df_))
        z = np.concatenate([embs[i][codes[s:e, i]] for i in range(len(CATS))]
                           + [Xd[s:e]], axis=1)
        z = lin(z, 0); z = bn(z, 1); z = np.maximum(z, 0.0)
        z = lin(z, 4); z = bn(z, 5); z = np.maximum(z, 0.0)
        z = lin(z, 8); z = np.maximum(z, 0.0)          # ← 마지막 은닉 (64차원)
        if H is None:
            H = np.empty((len(df_), z.shape[1]), dtype=np.float32)
        H[s:e] = z.astype(np.float32)
        z = lin(z, 10)
        out[s:e] = 1.0 / (1.0 + np.exp(-z.ravel()))
    return out, H


NN, HSUM = [], None
for s in NN_SEEDS:
    p_, H_ = nn_forward_capture(df, os.path.join(MODEL_DIR, f"nn_a{s}.npz"),
                                os.path.join(MODEL_DIR, f"nn_a{s}_meta.json"))
    ref = nn_infer.predict_nn(df, os.path.join(MODEL_DIR, f"nn_a{s}.npz"),
                              os.path.join(MODEL_DIR, f"nn_a{s}_meta.json"))
    assert np.array_equal(p_, ref), f"NN 복제 불일치 seed {s}"
    NN.append(lg(p_))
    HSUM = H_.astype(np.float64) if HSUM is None else HSUM + H_.astype(np.float64)
    del H_
NN = np.stack(NN)
pn = NN.mean(axis=0)
store["seed_nn_sd"] = NN.std(axis=0, ddof=1)
HBAR = HSUM / float(len(NN_SEEDS))
del HSUM
# 주성분 — 회전은 학습행이 아니라 이 프레임에서 잡히므로 '방향 선택'일 뿐이고,
# 계수는 어차피 LB/정직면이 정한다.  ⚠️ 이 회전 자체는 행 방향 집계다 →
# 출하하려면 회전 행렬을 train 에서 동결해야 한다 (report 에 명시).
_m = HBAR.mean(axis=0)
Hc = HBAR - _m
_U, _S_, _Vt = np.linalg.svd(Hc, full_matrices=False)
store["nn_hidden_pc"] = (_U[:, :6] * _S_[:6]).astype(np.float64)
store["nn_hidden_evr"] = (_S_ ** 2 / (_S_ ** 2).sum())[:12]
store["nn_hidden_dim"] = np.array([HBAR.shape[1]])
store["nn_hidden_l2"] = np.linalg.norm(HBAR, axis=1)
store["nn_hidden_nact"] = (HBAR > 0).sum(axis=1).astype(np.float64)
del Hc, _U, _Vt
print(f"[nn] {NN.shape} hidden {HBAR.shape} evr1-3 {store['nn_hidden_evr'][:3]}", flush=True)
del HBAR

# ══════════════════════ C: Leaf 경로 + E: leaf 시드별 ════════════════════════


def leaf_capture(df_, FE_, tree_path, pkl_path, cat_cols, chunk=200000):
    booster = lgb.Booster(model_file=tree_path)
    Pk = pickle.load(open(pkl_path, 'rb'))
    cats = Pk['cats']; coef = Pk['coef'].ravel(); b0 = float(Pk['intercept'][0])
    offs = np.cumsum([0] + [len(c) for c in cats])
    idx = [{v: j for j, v in enumerate(c)} for c in cats]
    Xl = df_[FE_].copy()
    for c in cat_cols:
        if c in Xl.columns:
            Xl[c] = Xl[c].astype('category')
    out = np.empty(len(df_), np.float64)
    oov = np.zeros(len(df_), np.float64)
    pos = np.zeros(len(df_), np.float64)   # 양의 coef 합
    neg = np.zeros(len(df_), np.float64)
    absum = np.zeros(len(df_), np.float64)
    ntree = None
    for s in range(0, len(df_), chunk):
        e = min(s + chunk, len(df_))
        L = booster.predict(Xl.iloc[s:e], pred_leaf=True).astype(np.int32)
        ntree = L.shape[1]
        z = np.full(e - s, b0, np.float64)
        for j in range(L.shape[1]):
            mm = idx[j]
            col = np.array([mm.get(int(v), -1) for v in L[:, j]])
            ok = col >= 0
            cv = np.zeros(e - s, np.float64)
            cv[ok] = coef[offs[j] + col[ok]]
            z += cv
            oov[s:e] += (~ok)
            pos[s:e] += np.maximum(cv, 0.0)
            neg[s:e] += np.minimum(cv, 0.0)
            absum[s:e] += np.abs(cv)
        out[s:e] = 1.0 / (1.0 + np.exp(-z))
    return out, oov, pos, neg, absum, ntree


LEAF, OOV, POS, NEG, ABS = [], [], [], [], []
NT = None
for s in LEAF_SEEDS:
    p_, o_, pp_, nn_, ab_, NT = leaf_capture(df, FE, os.path.join(MODEL_DIR, f"leaftree_{s}.txt"),
                                             os.path.join(MODEL_DIR, f"leaflr_{s}.pkl"), CAT_COLS)
    ref = nn_infer.predict_leaf(df, FE, os.path.join(MODEL_DIR, f"leaftree_{s}.txt"),
                                os.path.join(MODEL_DIR, f"leaflr_{s}.pkl"), CAT_COLS)
    assert np.array_equal(p_, ref), f"leaf 복제 불일치 seed {s}"
    LEAF.append(lg(p_)); OOV.append(o_); POS.append(pp_); NEG.append(nn_); ABS.append(ab_)
LEAF = np.stack(LEAF)
pf = LEAF.mean(axis=0)
store["seed_leaf_sd"] = LEAF.std(axis=0, ddof=1)
store["leaf_oov"] = np.stack(OOV).mean(axis=0)
store["leaf_pos"] = np.stack(POS).mean(axis=0)
store["leaf_neg"] = np.stack(NEG).mean(axis=0)
store["leaf_abs"] = np.stack(ABS).mean(axis=0)
store["leaf_ntree"] = np.array([NT])
print(f"[leaf] {LEAF.shape} ntree={NT} oov mean {store['leaf_oov'].mean():.4f}", flush=True)

# ══════════════════════ B: EBM 항별 기여 ═════════════════════════════════════
EBM_DIR = os.path.join(MODEL_DIR, "ebm6")
if not os.path.isdir(EBM_DIR):
    EBM_DIR = MODEL_DIR
EM = json.load(open(os.path.join(EBM_DIR, "ebm6_meta.json")))
Xmat = df[EM["features"]].to_numpy(dtype=object)


def score_terms(Xm, spec):
    """ebm_core.score_numpy 와 ★같은 산술★, 항별 기여를 함께 반환."""
    nn_ = Xm.shape[0]
    z = np.full(nn_, spec["intercept"], np.float64)
    bins, ftypes = spec["bins"], spec["ftypes"]
    C = np.empty((nn_, len(spec["terms"])), np.float32)
    for ti, (fts, ts) in enumerate(spec["terms"]):
        k = len(fts)
        if k == 1:
            bi = EC._bin_index(Xm[:, fts[0]], fts[0], k, bins, ftypes)
            c = ts[np.clip(bi, 0, ts.shape[0] - 1)]
        else:
            b0 = np.clip(EC._bin_index(Xm[:, fts[0]], fts[0], k, bins, ftypes), 0, ts.shape[0] - 1)
            b1 = np.clip(EC._bin_index(Xm[:, fts[1]], fts[1], k, bins, ftypes), 0, ts.shape[1] - 1)
            c = ts[b0, b1]
        z += c
        C[:, ti] = c.astype(np.float32)
    return 1.0 / (1.0 + np.exp(-z)), C


EBMC = None
elg = []
tnames = None
for s in EM["seeds"]:
    spec = EC.load_ebm_npz(os.path.join(EBM_DIR, f"ebm6_seed{s}.npz"))
    p_, C_ = score_terms(Xmat, spec)
    ref = EC.score_numpy(Xmat, spec)
    assert np.array_equal(p_, ref), f"EBM 복제 불일치 seed {s}"
    elg.append(lg(p_))
    EBMC = C_.astype(np.float64) if EBMC is None else EBMC + C_.astype(np.float64)
    if tnames is None:
        tnames = ["+".join(EM["features"][i] for i in fts) for fts, _ in spec["terms"]]
    del C_
EBMC /= float(len(EM["seeds"]))
elg = np.stack(elg)
store["seed_ebm_sd"] = elg.std(axis=0, ddof=1) if elg.shape[0] > 1 else np.zeros(n)
_sd = EBMC.std(axis=0)
_top = np.argsort(-_sd)[:12]
store["ebm_term_top_idx"] = _top.astype(np.float64)
store["ebm_term_top_sd"] = _sd[_top]
store["ebm_term_names"] = np.asarray([tnames[i] for i in _top], dtype=object)
store["ebm_term_top"] = EBMC[:, _top]
store["ebm_n_terms"] = np.array([EBMC.shape[1]])
store["ebm_logit"] = elg.mean(axis=0)
print(f"[ebm] terms={EBMC.shape[1]} seeds={elg.shape[0]} top sd {_sd[_top][:4]}", flush=True)
del EBMC

# ══════════════════════ 게이트 — 4팔 블렌드 재현 ═════════════════════════════
p_v6 = expit(W_LGB * pl + W_CAT * pc + W_NN * pn + W_LEAF * pf)
store["p_v6"] = p_v6
store["lg_lgb"], store["lg_cat"], store["lg_nn"], store["lg_leaf"] = pl, pc, pn, pf
REFV = "/home/<user>/SungjinPark/exp078/work/tr2024_v6_pv.npy"
if os.path.exists(REFV) and n == 253507:
    r = np.load(REFV).astype(np.float64)
    d = np.abs(p_v6 - r)
    store["gate_v6_ndiff"] = np.array([int(np.count_nonzero(d))])
    store["gate_v6_max"] = np.array([float(d.max())])
    print(f"[GATE v6] ndiff={int(np.count_nonzero(d))} max={d.max():.3e}", flush=True)
np.savez_compressed(OUTNPZ, **store)
print(f"[cap_mate] wall={(time.time()-T0)/60:.2f}min -> {OUTNPZ}", flush=True)

# script.py
import json
import os
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from scipy.special import logit, expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
import nn_infer
import splits
import inseason

ID_COL = "row_id"
TARGET_COL = "control_success"

CAT_COLS = ['game_type', 'base_state', 'pitcher_team_id', 'batter_team_id',
            'idea_matchup_pitch']
LGB_SEEDS = [42, 777, 2024]
CAT_SEEDS = [42, 777, 2024, 101]
NN_SEEDS = [42, 777, 2024, 101]
LEAF_SEEDS = [42, 777, 2024]

# fold_2024 검증에서 최적화된 블렌딩 비중 (로짓 공간)
W_LGB, W_CAT, W_NN, W_LEAF = 0.05, 0.68, 0.15, 0.12


def load_test(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음")
    return df


def load_sample_submission(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if list(df.columns[:2]) != [ID_COL, TARGET_COL]:
        raise ValueError(f"sample_submission 컬럼 불일치: {list(df.columns)}")
    return df


def lg(p):
    return logit(np.clip(p, 1e-6, 1 - 1e-6))


def apply_shift(p, d):
    """logit 공간 평행이동 — 각 행이 독립적으로 계산된다.

    이동량 d는 학습 데이터에서 미리 산출한 상수(model/shift.json)이며
    평가 데이터의 라벨도, 다른 행의 값도 참조하지 않는다.
    트리 모델은 학습 시즌 범위 밖(2025)을 외삽하지 못해 확률 수준이 밀리는데,
    Brier는 편향에 제곱으로 벌점을 주므로 전역 보정이 필요하다.
    """
    return expit(lg(p) + d)


def merge_predictions(sub, ids, preds):
    pred_map = dict(zip(ids, preds))
    values, n_missing = [], 0
    for rid, cur in zip(sub[ID_COL], sub[TARGET_COL]):
        v = pred_map.get(rid)
        if v is None:
            n_missing += 1
            values.append(cur)
        else:
            values.append(v)
    if n_missing:
        print(f" 경고: 예측이 없어 placeholder 유지한 row_id {n_missing}건")
    sub[TARGET_COL] = values
    return sub


def run(model_dir="./model", test_dir="./data"):
    TEST_DIR = test_dir
    MODEL_DIR = model_dir
    TBL_DIR = os.path.join(MODEL_DIR, "tables")
    OUT_PATH = "./output/submission.csv"

    print("Load test data...")
    test = load_test(os.path.join(TEST_DIR, "test.csv"))
    sub = load_sample_submission(os.path.join(TEST_DIR, "sample_submission.csv"))
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    feats = P.apply_features(test, TBL_DIR)
    ratios = P.apply_ratio_features(test, feats)
    df = pd.concat([test.reset_index(drop=True),
                    feats.reset_index(drop=True),
                    ratios.reset_index(drop=True)], axis=1)
    df['cnt_state'] = df.balls_before.astype(int) * 3 + df.strikes_before.astype(int)
    with open(os.path.join(MODEL_DIR, "features.txt")) as f:
        FE = [c for c in f.read().split("\n") if c]

    # 개인 스플릿(SP) — 동결표 룩업. 각 행이 자기 pitcher_id/batter_id 와
    # 자기 카운트만 참조하므로 행 독립(규칙 4항). 미지 주체는 편차 0.0.
    sp_path = os.path.join(MODEL_DIR, "deploy_splits.npz")
    FE_SP = FE
    if os.path.exists(sp_path):
        T = splits.load(sp_path)
        S = splits.apply(df, T)
        for c in S.columns:
            df[c] = S[c].values
        with open(os.path.join(MODEL_DIR, "features_sp.txt")) as f:
            FE_SP = [c for c in f.read().split("\n") if c]
        print(f" SP columns attached ({len(FE_SP) - len(FE)} new)")

    # in-season 분해(ins_*) — asof_* 를 '직전시즌까지 커리어' 와 '당시즌 누적' 으로
    # 분리한다. 앵커는 train(<=2024) 유래 동결표이고 각 행은 자기 asof_* 값과
    # 자기 pitcher_id/batter_id 만 참조한다. 다른 평가행·전체분포 미참조(규칙 4항).
    anchor_path = os.path.join(MODEL_DIR, "anchor.npz")
    if os.path.exists(anchor_path):
        I = inseason.apply(df, inseason.load(anchor_path))
        for c in I.columns:
            df[c] = I[c].values
        print(f" ins columns attached ({len(I.columns)} new)")

    missing = [c for c in set(FE) | set(FE_SP) if c not in df.columns]
    if missing:
        raise ValueError(f"누락 피처: {missing}")
    print(f" features={len(FE)}  features_sp={len(FE_SP)}")

    ids = test[ID_COL].tolist()

    print("Inference LightGBM...")
    X = df[FE_SP].copy()
    # 카테고리 수준을 학습 시점 목록으로 고정 — 등장집합 차이로 코드가 밀리는
    # 사고(exp_007)를 원천 차단한다.
    _lvp = os.path.join(MODEL_DIR, "lgb_cats.json")
    _lv = json.load(open(_lvp)) if os.path.exists(_lvp) else None
    for c in CAT_COLS:
        if c in X.columns:
            X[c] = (pd.Categorical(X[c].astype(str), categories=_lv[c]) if _lv
                    else X[c].astype('category'))
    pl = np.mean([lg(lgb.Booster(model_file=os.path.join(MODEL_DIR, f"lgb_{s}.txt")).predict(X))
                  for s in LGB_SEEDS], axis=0)
    print(f" lgb models={len(LGB_SEEDS)}")

    print("Inference CatBoost...")
    Xc = df[FE_SP].copy()
    for c in CAT_COLS:
        if c in Xc.columns:
            Xc[c] = Xc[c].astype(str)
    cps = []
    for s in CAT_SEEDS:
        m = CatBoostClassifier()
        m.load_model(os.path.join(MODEL_DIR, f"cat_{s}.cbm"))
        cps.append(lg(m.predict_proba(Xc)[:, 1]))
    pc = np.mean(cps, axis=0)
    print(f" cat models={len(CAT_SEEDS)}")

    print("Inference NeuralNet (numpy)...")
    pn = np.mean([lg(nn_infer.predict_nn(df,
                                         os.path.join(MODEL_DIR, f"nn_a{s}.npz"),
                                         os.path.join(MODEL_DIR, f"nn_a{s}_meta.json")))
                  for s in NN_SEEDS], axis=0)
    print(f" nn  models={len(NN_SEEDS)}")

    print("Inference GBDT+LR...")
    pf = np.mean([lg(nn_infer.predict_leaf(df, FE,
                                           os.path.join(MODEL_DIR, f"leaftree_{s}.txt"),
                                           os.path.join(MODEL_DIR, f"leaflr_{s}.pkl"),
                                           CAT_COLS))
                  for s in LEAF_SEEDS], axis=0)
    print(f" leaf models={len(LEAF_SEEDS)}")

    import numpy as _np, os as _os
    _D = "/home/<user>/SungjinPark/exp078/work"
    _os.makedirs(_D, exist_ok=True)
    _np.save(_os.path.join(_D, "tr2024_v6_pl.npy"), _np.asarray(pl, _np.float64))
    _np.save(_os.path.join(_D, "tr2024_v6_pc.npy"), _np.asarray(pc, _np.float64))
    _np.save(_os.path.join(_D, "tr2024_v6_pn.npy"), _np.asarray(pn, _np.float64))
    _np.save(_os.path.join(_D, "tr2024_v6_pf.npy"), _np.asarray(pf, _np.float64))
    _np.save(_os.path.join(_D, "tr2024_v6_ids.npy"), _np.asarray(list(ids)))
    print("[CAP4] pl/pc/pn/pf saved", flush=True)
    preds = expit(W_LGB * pl + W_CAT * pc + W_NN * pn + W_LEAF * pf)
    print(f" blend rows={len(preds)}")

    # 블렌딩 빌드: 자체 shift(0.4785 겨냥)를 적용하지 않는다.
    # 근거 — 팀원 단일변인 실측(후보 P vs N): 외삽 0.4785 겨냥이 관측 0.4861
    # 겨냥 대비 LB -20.12. 수준 보정은 결합 후 script.py가 한 번만 수행한다.
    return ids, preds


if __name__ == "__main__":
    run()

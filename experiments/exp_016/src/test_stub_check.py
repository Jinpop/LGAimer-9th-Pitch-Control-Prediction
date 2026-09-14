"""exp_016 — rule 2-4 demonstration + leak-boundary evidence.

Two things are checked against the ACTUAL saved fold_2024 models:

  (A) rule 2-4 (독립 추론): the composed P(success) for a test row must not
      depend on which other rows are in the batch.  Batch prediction, one-row-
      at-a-time prediction and shuffled-batch prediction must agree exactly.

  (B) leak boundary: the feature list every saved model was fitted on must
      contain none of the recovered `is_*` columns.  Read straight out of the
      model files, not out of the training script's own bookkeeping.
"""

import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from labels import FAIL_CELLS, RECOVERED_COLS, RECOVERED_PREFIXES  # noqa: E402

DATA = os.path.expanduser("~/SungjinPark/data")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_016")
CAT_COLS = ["top_bottom", "game_type", "base_state"]
EPS = 1e-6


def main():
    models = {name: lgb.Booster(model_file=os.path.join(
        OUT_DIR, f"model_{name}_fold_2024.txt"))
        for name in ["B0b"] + FAIL_CELLS}

    out = {"models": list(models)}

    # ---- (B) leak boundary, read from the model files -------------------- #
    feat_check = {}
    for name, bst in models.items():
        fn = bst.feature_name()
        bad = [c for c in fn if c in RECOVERED_COLS
               or c.startswith(RECOVERED_PREFIXES)]
        feat_check[name] = {"n_features": len(fn), "recovered_cols_present": bad}
        assert not bad, (name, bad)
    out["leak_boundary_feature_lists_from_model_files"] = feat_check
    out["all_models_share_one_feature_list"] = len(
        {tuple(b.feature_name()) for b in models.values()}) == 1

    # ---- (A) rule 2-4 ----------------------------------------------------- #
    test = pd.read_csv(os.path.join(DATA, "test.csv"), encoding="utf-8-sig")
    feats = models["B0b"].feature_name()
    assert set(feats).issubset(test.columns), set(feats) - set(test.columns)
    out["all_model_features_exist_in_test_csv"] = True

    # same category vocabulary the models were fitted with
    tr_cat = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig",
                         usecols=CAT_COLS)
    X = test[feats].copy()
    for c in CAT_COLS:
        X[c] = pd.Categorical(X[c], categories=sorted(tr_cat[c].unique()))

    def compose(frame):
        ps = models["B0b"].predict(frame)
        cells = {c: models[c].predict(frame) for c in FAIL_CELLS}
        s = sum(cells.values())
        return {
            "B0b": ps,
            "B2c": np.clip(1.0 - s, EPS, 1 - EPS),
            "B2n": np.clip(ps / (ps + s), EPS, 1 - EPS),
        }

    batch = compose(X)
    single = {k: np.zeros(len(X)) for k in batch}
    for i in range(len(X)):
        one = compose(X.iloc[[i]])
        for k in one:
            single[k][i] = one[k][0]

    order = np.array([3, 0, 4, 1, 2])
    shuf = compose(X.iloc[order])
    inv = np.argsort(order)

    out["rule_2_4"] = {
        k: {
            "batch": [round(float(v), 12) for v in batch[k]],
            "max_abs_diff_batch_vs_one_row_at_a_time": float(
                np.max(np.abs(batch[k] - single[k]))),
            "max_abs_diff_batch_vs_shuffled_batch": float(
                np.max(np.abs(batch[k] - shuf[k][inv]))),
        }
        for k in batch
    }
    out["rule_2_4_passed"] = all(
        v["max_abs_diff_batch_vs_one_row_at_a_time"] == 0.0
        and v["max_abs_diff_batch_vs_shuffled_batch"] == 0.0
        for v in out["rule_2_4"].values())

    with open(os.path.join(OUT_DIR, "test_stub_check.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    assert out["rule_2_4_passed"]


if __name__ == "__main__":
    main()

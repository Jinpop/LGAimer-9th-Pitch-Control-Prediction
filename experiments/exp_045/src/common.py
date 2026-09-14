"""exp_045 — shared definitions for the NN (entity-embedding MLP) arm.

Fold definitions, masks and the segment-metric function are INHERITED from
exp_004 / exp_016 / exp_030 by reading those files, not by re-typing numbers
from memory:

  * exp_004/src/train.py  ->  FOLDS = [(name, train_until, valid_year)] and the
    F-league training exclusion mask (season <= 2022 & game_type == "F"), plus
    the valid mask (season == valid_year, F rows KEPT) and seg_metrics().
  * exp_016/src/train.py  ->  EXPECTED_2024 = {"valid_all": 253507,
    "valid_R": 223497}.
  * exp_030/src/b3_cat.py ->  UNIT_SPEC row counts (n_train / n_valid per unit),
    which are the byte-equal restatement of the exp_004 masks.

All three are asserted at runtime (see assert_fold_shapes) so a silent drift in
the data file or in the mask logic stops the run instead of producing a number.

========================= LEAK BOUNDARY (NN-specific) =========================
Three preprocessing surfaces exist in this experiment that did NOT exist in the
GBDT experiments.  All three are fitted on the TRAIN slice of the fold only:

  (a) QuantileTransformer  -- fit on train rows only, transform applied to both.
  (b) entity vocabularies  -- built from train rows only; index 0 is a learned
      out-of-vocabulary bucket that unseen valid/test IDs fall into.  It is a
      learned "unknown" category, not a sentinel fill.
  (c) missing-value fill   -- a separate {col}_isna flag column is created for
      every continuous column that has a NaN in the TRAIN slice, and only then
      is the value slot filled.  In the quantile-normal output space the train
      median maps to exactly 0.0, so the fill value 0.0 IS the train median.
      No sentinel (0/-1/mean of everything) is ever used.

Nothing here reads the valid slice, the test file, or the label.
==============================================================================
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

REF_BRIER = 0.2498069          # 2025-test conversion reference (CLAUDE.md)

REPO_SRC = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.dirname(os.path.dirname(REPO_SRC))     # experiments/
DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")

TARGET = "control_success"
ROW_ID = "row_id"

# --- column groups ---------------------------------------------------------
# 49 columns total = row_id + target + 47 features.
EMB_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id",
            "base_state"]
# config.yaml model.embeddings.small_binaries: kept as raw 0/1 numerics, NOT
# passed through the quantile transformer.
BIN_COLS = ["top_bottom", "game_type", "pitcher_hand", "batter_hand",
            "runner_on_1b", "runner_on_2b", "runner_on_3b"]
BIN_MAP = {"top_bottom": {"B": 0.0, "T": 1.0},
           "game_type": {"F": 0.0, "R": 1.0},
           "pitcher_hand": {1: 0.0, 2: 1.0},
           "batter_hand": {1: 0.0, 2: 1.0},
           "runner_on_1b": {0: 0.0, 1: 1.0},
           "runner_on_2b": {0: 0.0, 1: 1.0},
           "runner_on_3b": {0: 0.0, 1: 1.0}}

EMB_CARD = {"pitcher_id": 792, "batter_id": 830, "pitcher_team_id": 13,
            "batter_team_id": 13, "base_state": 8}
EMB_DIM = {"pitcher_id": 16, "batter_id": 16, "pitcher_team_id": 4,
           "batter_team_id": 4, "base_state": 4}


def score_of(brier):
    """LB-scale conversion.  100000 * (1 - Brier / 0.2498069)."""
    return 100000.0 * (1.0 - float(brier) / REF_BRIER)


def score_delta(dbrier):
    """Score-point movement for a Brier delta (positive dbrier -> negative)."""
    return -100000.0 * float(dbrier) / REF_BRIER


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# --- inherited constants, parsed from the source of truth ------------------
def _read(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def inherit_from_exp004(exp004_train_py):
    """Parse FOLDS and the F-exclusion season cap out of exp_004/src/train.py."""
    txt = _read(exp004_train_py)
    folds = re.findall(r'\("(fold_\d{4})",\s*(\d{4}),\s*(\d{4})\)', txt)
    assert len(folds) == 3, folds
    folds = [(n, int(a), int(b)) for n, a, b in folds]
    m = re.search(r"F_EXCLUDE_MAX_SEASON\s*=\s*(\d{4})", txt)
    assert m, "F_EXCLUDE_MAX_SEASON not found in exp_004/src/train.py"
    return folds, int(m.group(1))


def inherit_expected_2024(exp016_train_py):
    txt = _read(exp016_train_py)
    m = re.search(r'EXPECTED_2024\s*=\s*\{"valid_all":\s*(\d+),\s*"valid_R":\s*(\d+)\}',
                  txt)
    assert m, "EXPECTED_2024 not found in exp_016/src/train.py"
    return {"valid_all": int(m.group(1)), "valid_R": int(m.group(2))}


def inherit_unit_spec(exp030_b3_py):
    """UNIT_SPEC {year: {train_upto, n_train, n_valid[, n_valid_R]}}."""
    txt = _read(exp030_b3_py)
    blk = re.search(r"UNIT_SPEC\s*=\s*\{(.*?)\n\}", txt, re.S)
    assert blk, "UNIT_SPEC not found in exp_030/src/b3_cat.py"
    spec = {}
    for yr, body in re.findall(r"(\d{4}):\s*\{(.*?)\}", blk.group(1), re.S):
        d = {}
        for k, v in re.findall(r'"(\w+)":\s*(\d+)', body):
            d[k] = int(v)
        spec[int(yr)] = d
    assert set(spec) == {2022, 2023, 2024}, spec
    return spec


# Ancestor files, pinned by content hash.  The local repo and the A5000 server
# lay these out differently (server exp_004 has train.py at the top level, and
# no exp_030/src), so each is looked up over a candidate list and then verified
# against its pinned sha -- the sha, not the path, is what makes it the ancestor.
ANCESTORS = {
    "exp_004_train": {
        "sha": "906fd6116007752f1811fa07983d356ab3841d549e01d081dda6991730a73ca0",
        "candidates": ["exp_004/src/train.py", "exp_004/train.py"]},
    "exp_016_train": {
        "sha": "4f24deb71d9eef3d626900f34a71aecd2d6acbae646eb9c92c51189a38a3ece1",
        "candidates": ["exp_016/src/train.py", "exp_016/train.py"]},
    "b3_cat": {   # exp_030/src/b3_cat.py and exp_033/src/b3_cat.py are byte-equal
        "sha": "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca",
        "candidates": ["exp_030/src/b3_cat.py", "exp_033/src/b3_cat.py"]},
}
BUNDLED = os.path.join(REPO_SRC, "ancestors")


def resolve_ancestor(key, exp_root=None):
    root = exp_root or EXP_ROOT
    spec = ANCESTORS[key]
    cands = [os.path.join(root, c) for c in spec["candidates"]]
    cands.append(os.path.join(BUNDLED, os.path.basename(spec["candidates"][0])))
    cands.append(os.path.join(BUNDLED, key + ".py"))
    tried = []
    for p in cands:
        tried.append(p)
        if os.path.exists(p):
            got = sha256_file(p)
            if got == spec["sha"]:
                return p, got
            tried[-1] += f" (sha {got[:12]} != pinned {spec['sha'][:12]})"
    raise AssertionError(f"ancestor {key} not resolvable; tried: {tried}")


def load_inherited(exp_root=None):
    """Locate the three ancestor files and parse them.  Fails loudly."""
    p4, s4 = resolve_ancestor("exp_004_train", exp_root)
    p16, s16 = resolve_ancestor("exp_016_train", exp_root)
    p30, s30 = resolve_ancestor("b3_cat", exp_root)
    folds, f_cap = inherit_from_exp004(p4)
    out = {
        "folds": folds,
        "f_exclude_max_season": f_cap,
        "expected_2024": inherit_expected_2024(p16),
        "unit_spec": inherit_unit_spec(p30),
        "ancestor_paths": {"exp_004_train": p4, "exp_016_train": p16,
                           "b3_cat": p30},
        "ancestor_sha256": {"exp_004_train": s4, "exp_016_train": s16,
                            "b3_cat": s30},
    }
    return out


# --- metrics (behaviourally identical to exp_004 seg_metrics) --------------
def seg_metrics(y, p, full=True):
    from sklearn.metrics import log_loss, roc_auc_score
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    bss = float(1.0 - brier / denom) if denom > 0 else None
    out = {"n": n,
           "brier": brier,
           "brier6": round(brier, 6),
           "score": round(score_of(brier), 5),
           "bss": round(bss, 6) if bss is not None else None,
           "base_rate": round(r, 6),
           "pred_mean": round(float(np.mean(p)), 6)}
    if full and 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def masks_for_fold(df, train_until, valid_year, f_exclude_max_season):
    """exp_004 masks, verbatim in behaviour.

    train: season <= train_until, MINUS (season <= 2022 & game_type == 'F')
    valid: season == valid_year, F rows KEPT (valid mask untouched by exp_004)
    """
    f_excl = (df["season"] <= f_exclude_max_season) & (df["game_type"] == "F")
    tr = (df["season"] <= train_until) & ~f_excl
    va = (df["season"] == valid_year)
    return tr.to_numpy(), va.to_numpy()


def assert_fold_shapes(inh, valid_year, n_train, n_valid, n_valid_r=None):
    spec = inh["unit_spec"][valid_year]
    assert n_train == spec["n_train"], (valid_year, n_train, spec["n_train"])
    assert n_valid == spec["n_valid"], (valid_year, n_valid, spec["n_valid"])
    if valid_year == 2024:
        e = inh["expected_2024"]
        assert n_valid == e["valid_all"], (n_valid, e["valid_all"])
        if n_valid_r is not None:
            assert n_valid_r == e["valid_R"], (n_valid_r, e["valid_R"])
    return True


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False, default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))

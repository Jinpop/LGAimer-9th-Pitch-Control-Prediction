"""exp_003 — exp_001 baseline + pitcher static TrackMan profile join (fold-wise as-of).

Spec: experiments/exp_003/config.yaml (2026-08-05, campaign 2)
Single changed variable vs exp_001: profile features added. All LGBM params,
seed, folds, categorical handling identical to exp_001 — EXCEPT num_threads=8
(shared server, exp_004~006 may run in parallel). To isolate the thread effect,
this script also re-runs the exp_001 feature set on fold_2024 with num_threads=8
(reference only, reported separately).

Leak rule (config.features.notes): for a fold validating year Y, the profile is
aggregated from trackman_history rows with season <= Y-1 ONLY (as-of cutoff).
The 2024-end profile (exp_002 artifact) is NOT used for fold validation — it
would inject valid-year physics. Logic mirrors exp_002/src/02_build_profile.py
with the cutoff parameterized; "s2024_*" generalizes to "recent_*" (= season ==
cutoff). Join key pitcher_id via exp_002 out/pitcher_id_map.csv; unmapped
pitchers (0.6% of rows) keep NaN — no sentinel filling (project rule).

Current-row TrackMan physics are never used: profiles are historical aggregates
over seasons strictly before the valid year.

Deterministic: seed 42, aggregation only (no randomness in profile build).
Runtime: A5000 CPU, lightgbm 4.7.0 / pandas 2.0.3 / numpy 1.26.4 / Python 3.11.
"""

import hashlib
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

SEED = 42
DATA_DIR = os.path.expanduser("~/SungjinPark/data")
DATA_PATH = os.path.join(DATA_DIR, "train.csv")
TM_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
PMAP_PATH = os.path.expanduser("~/SungjinPark/experiments/exp_002/out/pitcher_id_map.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_003")

TARGET = "control_success"
DROP_COLS = ["row_id", TARGET]
CAT_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_id",
    "batter_id",
]
FOLDS = [
    ("fold_2022", 2021, 2022),
    ("fold_2023", 2022, 2023),
    ("fold_2024", 2023, 2024),
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": SEED,
    "num_threads": 8,  # shared server (exp_004~006 parallel); exp_001 used 24
    "verbose": -1,
}
NUM_BOOST_ROUND = 5000
EARLY_STOPPING_ROUNDS = 200

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
STD_COLS = ("rel_speed", "spin_rate", "rel_height", "rel_side",
            "induced_vert_break", "horz_break")


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def seg_metrics(y, p):
    """Brier / BSS(r = actual valid rate) / AUC / logloss / means for a segment."""
    n = int(len(y))
    if n == 0:
        return {"n": 0}
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    brier = float(np.mean((p - y) ** 2))
    r = float(np.mean(y))
    denom = r * (1.0 - r)
    bss = float(1.0 - brier / denom) if denom > 0 else None
    out = {
        "n": n,
        "brier": round(brier, 6),
        "bss": round(bss, 6) if bss is not None else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def sub_profile(df, prefix):
    """Aggregate one scope (career or recent season) — exp_002/02_build_profile.py logic."""
    g = df.groupby("pitcher_trackman_id")
    out = pd.DataFrame(index=g.size().index)
    out[f"{prefix}_n_pitches"] = g.size()
    out[f"{prefix}_n_games"] = g["trackman_game_id"].nunique()
    for c in PHYS:
        out[f"{prefix}_{c}_mean"] = g[c].mean()
        if c in STD_COLS:
            out[f"{prefix}_{c}_std"] = g[c].std()
    out[f"{prefix}_rel_speed_p90"] = g["rel_speed"].quantile(0.9)
    n = g.size()
    for pg in ("fastball", "breaking", "offspeed"):
        sub = df[df["pitch_type_group"] == pg].groupby("pitcher_trackman_id")
        out[f"{prefix}_{pg}_share"] = (sub.size() / n).fillna(0.0)
        out[f"{prefix}_{pg}_rel_speed_mean"] = sub["rel_speed"].mean()
        out[f"{prefix}_{pg}_spin_rate_mean"] = sub["spin_rate"].mean()
        out[f"{prefix}_{pg}_ivb_mean"] = sub["induced_vert_break"].mean()
        out[f"{prefix}_{pg}_hb_mean"] = sub["horz_break"].mean()
    return out


def build_asof_profile(tm, pmap, cutoff):
    """As-of profile: trackman seasons <= cutoff only. career_* = all <= cutoff,
    recent_* = season == cutoff, last_season_gap = cutoff - last active season.
    Indexed by train pitcher_id. NaN kept for missing (no sentinel)."""
    sub = tm[tm["season"] <= cutoff]
    assert len(sub) > 0 and int(sub["season"].max()) <= cutoff, "as-of cutoff violated"
    career = sub_profile(sub, "career")
    recent = sub_profile(sub[sub["season"] == cutoff], "recent")
    prof = career.join(recent, how="left")
    prof["last_season_gap"] = cutoff - sub.groupby("pitcher_trackman_id")["season"].max()
    prof = prof.reset_index()
    prof = pmap[["pitcher_id", "pitcher_trackman_id"]].merge(
        prof, on="pitcher_trackman_id", how="inner")
    assert prof["pitcher_id"].is_unique
    return prof.drop(columns=["pitcher_trackman_id"]).set_index("pitcher_id")


def run_fold(df, pid, feat_cols, fold_name, train_until, valid_year, prof=None):
    """Train one rolling-origin fold. prof: as-of profile DataFrame or None."""
    ft = time.time()
    tr_mask = (df["season"] <= train_until).to_numpy()
    va_mask = (df["season"] == valid_year).to_numpy()

    def make_X(mask):
        X = df.loc[mask, feat_cols]
        if prof is not None:
            blk = prof.reindex(pid[mask])
            blk.index = X.index
            X = pd.concat([X, blk], axis=1)
        return X

    X_tr, X_va = make_X(tr_mask), make_X(va_mask)
    y_tr = df.loc[tr_mask, TARGET].to_numpy()
    y_va = df.loc[va_mask, TARGET].to_numpy()
    all_cols = list(X_tr.columns)
    print(f"[{time.strftime('%H:%M:%S')}] {fold_name}: train {X_tr.shape[0]:,} "
          f"(<= {train_until}) -> valid {X_va.shape[0]:,} ({valid_year}), "
          f"{len(all_cols)} feats", flush=True)

    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=CAT_COLS)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr, categorical_feature=CAT_COLS)
    booster = lgb.train(
        LGB_PARAMS, dtr, num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dva], valid_names=["valid"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=True),
                   lgb.log_evaluation(100)],
    )
    train_sec = time.time() - ft
    ti = time.time()
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    infer_sec = time.time() - ti

    is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
    seen_pitchers = set(pid[tr_mask])
    cold = ~np.isin(pid[va_mask], list(seen_pitchers))

    out = {
        "train_until": train_until,
        "valid_year": valid_year,
        "n_train": int(tr_mask.sum()),
        "n_features": len(all_cols),
        "best_iteration": int(booster.best_iteration),
        "train_time_min": round(train_sec / 60, 2),
        "inference_time_sec": round(infer_sec, 2),
        "valid_all": seg_metrics(y_va, p_va),
        "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
        "cold_start_all": seg_metrics(y_va[cold], p_va[cold]),
        "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
        "seen_R": seg_metrics(y_va[~cold & is_r], p_va[~cold & is_r]),
        "n_cold_start_pitchers": int(pd.Series(pid[va_mask][cold]).nunique()),
    }

    if prof is not None:
        # profile availability split (valid-R): rows whose pitcher has a profile
        has_prof = X_va["career_n_pitches"].notna().to_numpy()
        out["valid_R_has_profile"] = seg_metrics(y_va[is_r & has_prof], p_va[is_r & has_prof])
        out["valid_R_no_profile"] = seg_metrics(y_va[is_r & ~has_prof], p_va[is_r & ~has_prof])
        out["cold_start_R_has_profile"] = seg_metrics(
            y_va[cold & is_r & has_prof], p_va[cold & is_r & has_prof])
        out["cold_start_R_no_profile"] = seg_metrics(
            y_va[cold & is_r & ~has_prof], p_va[cold & is_r & ~has_prof])
        out["valid_profile_row_coverage"] = round(float(has_prof.mean()), 4)

    # feature importance (gain): overall top 30 + all profile features ranked
    imp = sorted(zip(all_cols, booster.feature_importance("gain")), key=lambda t: -t[1])
    out["feature_importance_gain_top30"] = [[f, round(float(g), 1)] for f, g in imp[:30]]
    if prof is not None:
        prof_cols = set(prof.columns)
        out["profile_feature_importance"] = [
            [rank + 1, f, round(float(g), 1)]
            for rank, (f, g) in enumerate(imp) if f in prof_cols and g > 0
        ][:40]
        total_gain = sum(g for _, g in imp)
        prof_gain = sum(g for f, g in imp if f in prof_cols)
        out["profile_gain_share"] = round(float(prof_gain / total_gain), 4) if total_gain else 0.0

    booster.save_model(os.path.join(OUT_DIR, f"model_{fold_name}.txt"),
                       num_iteration=booster.best_iteration)
    print(json.dumps({fold_name: {k: v for k, v in out.items()
                                  if not k.startswith("feature_importance")}},
                     indent=2), flush=True)
    return out


def main():
    np.random.seed(SEED)
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {DATA_PATH}", flush=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    print(f"train loaded: {df.shape}", flush=True)

    pmap = pd.read_csv(PMAP_PATH)
    tm = pd.read_csv(
        TM_PATH, encoding="utf-8-sig",
        usecols=["season", "trackman_game_id", "pitcher_trackman_id",
                 "pitch_type_group"] + PHYS)
    assert int(tm["season"].max()) <= 2024, "trackman contains 2025 — violation"
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
    print(f"trackman rows for mapped pitchers: {len(tm):,}", flush=True)

    feat_cols = [c for c in df.columns if c not in DROP_COLS]
    pid = df["pitcher_id"].to_numpy()  # numeric copy for profile join / cold-start
    for c in CAT_COLS:  # identical to exp_001: label mapping over full file, no target info
        df[c] = df[c].astype("category")

    metrics = {
        "exp_id": "exp_003",
        "base_exp": "exp_001",
        "seed": SEED,
        "lgb_params": LGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUND,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "n_base_features": len(feat_cols),
        "categorical": CAT_COLS,
        "profile": {
            "source": "trackman_history via exp_002 pitcher_id_map (685 pitchers)",
            "asof_rule": "fold valid year Y -> profile from seasons <= Y-1 only",
            "prefixes": ["career_* (all seasons <= cutoff)",
                         "recent_* (season == cutoff)", "last_season_gap"],
            "missing": "NaN kept (no sentinel)",
        },
        "sha256": {
            "train.csv": sha256_file(DATA_PATH),
            "trackman_history.csv": sha256_file(TM_PATH),
            "pitcher_id_map.csv": sha256_file(PMAP_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "lightgbm": lgb.__version__,
        },
    }

    total_infer = 0.0
    for fold_name, train_until, valid_year in FOLDS:
        cutoff = valid_year - 1
        assert cutoff == train_until  # rolling-origin: profile horizon == train horizon
        tp = time.time()
        prof = build_asof_profile(tm, pmap, cutoff)
        print(f"[{time.strftime('%H:%M:%S')}] as-of profile cutoff={cutoff}: "
              f"{prof.shape[0]} pitchers x {prof.shape[1]} feats "
              f"({time.time()-tp:.1f}s)", flush=True)
        metrics[fold_name] = run_fold(df, pid, feat_cols, fold_name,
                                      train_until, valid_year, prof=prof)
        metrics[fold_name]["profile_cutoff"] = cutoff
        metrics[fold_name]["n_profile_pitchers"] = int(prof.shape[0])
        total_infer += metrics[fold_name]["inference_time_sec"]

    # reference: exp_001 feature set, fold_2024, num_threads=8 (thread-effect isolation)
    print(f"[{time.strftime('%H:%M:%S')}] reference rerun: exp_001 features, "
          f"fold_2024, num_threads=8", flush=True)
    metrics["reference_exp001_threads8_fold_2024"] = run_fold(
        df, pid, feat_cols, "exp001ref_fold_2024", 2023, 2024, prof=None)

    metrics["train_time_min"] = round((time.time() - t0) / 60, 2)
    metrics["inference_time_sec"] = round(total_infer, 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()

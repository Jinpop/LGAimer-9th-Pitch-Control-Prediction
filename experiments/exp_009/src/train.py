"""exp_009 — does our TrackMan pitcher profile still pay off on the STRONG base?

Spec: experiments/exp_009/config.yaml (2026-08-05, campaign 4).

Context (exp_008): on our split (train <= 2023 -> valid 2024) the organizers'
47-feature RandomForest beats our whole LGBM lineage. RF + our exp_004 mask
("R2", valid_R Brier 0.248455) is therefore the new base. This run asks the one
open question about our own feature work: does the exp_003 TrackMan pitcher
profile (65 as-of features) add anything on top of a base that is already
better than everything we built?

A second, equally important question rides along. The organizers' pipeline
imputes every numeric column with SimpleImputer(median). Roughly 19% of valid
rows have no as-of profile (unmapped pitcher / no TrackMan history before the
cutoff), so a naive join silently turns "unknown" into "median pitcher" — a
sentinel, which the project rules forbid. We therefore measure the missingness
encoding explicitly rather than assuming it away.

Arms (every arm carries the exp_004 mask; RF hyperparameters are the
organizers' verbatim: n_estimators=100, max_depth=10, min_samples_leaf=200,
random_state=42). Each arrow changes EXACTLY ONE thing:

  P0   47 base features                          <- matched control (= exp_008 R2)
  P0b  P0 but max_features=10                    <- confound control: P1 raises
                                                    the sqrt(n_features) default
                                                    from 6 to 10 all by itself
  P1   P0 + 65 as-of profile features            <- +profile (median-imputed)
  P2   P1 + has_profile flag                     <- +missingness flag
  P3   P2 + has_recent_profile flag              <- +second missingness flag
  P4   P3 but profile columns bypass the imputer <- median sentinel -> native NaN
       and reach the forest as NaN                  (sklearn >= 1.4 RF splits on
                                                     missing values natively)

Primary metric: fold_2024 valid_R Brier (project rule). Control = P0.

Leakage control
  - Profile is rebuilt per fold with an as-of cutoff: a fold validating year Y
    aggregates TrackMan rows with season <= Y-1 only, asserted in code. The
    exp_002 end-of-2024 artifact is never used for fold validation.
  - Encoder categories, imputer medians and the forest are fit on training rows
    only; valid rows only ever go through .transform / .predict_proba.
  - No current-row TrackMan quantity is used: the 47 base features are exactly
    the test.csv columns, and the profile is a historical aggregate over seasons
    strictly before the valid year.
  - Nothing about the test set is read.

Environment: A5000, ~/SungjinPark/venv (Python 3.11.15 / numpy 1.26.4 /
pandas 2.0.3 / scikit-learn 1.8.0) — matches the evaluation server.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
N_JOBS = 6  # shared box, 3 other experiments may run in parallel (orchestrator).
# n_jobs only affects wall time, not the fitted forest: trees are seeded from
# random_state and grown independently.

DATA_DIR = os.path.expanduser("~/SungjinPark/data")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TM_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
PMAP_PATH = os.path.expanduser(
    "~/SungjinPark/experiments/exp_002/out/pitcher_id_map.csv")
OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_009")

TARGET = "control_success"
CAT_COLS = ["top_bottom", "game_type", "base_state"]  # organizers' 3 categoricals

RF_PARAMS = dict(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=200,
    random_state=42,
)

FOLDS = {
    "fold_2022": (2021, 2022),
    "fold_2023": (2022, 2023),
    "fold_2024": (2023, 2024),
}
ALL3 = ["fold_2022", "fold_2023", "fold_2024"]
LAST2 = ["fold_2023", "fold_2024"]

# exp_001 / exp_007 / exp_008 valid-row counts for fold_2024 (identity check).
EXPECTED_2024 = {"valid_all": 253507, "valid_R": 223497}

# --- profile construction (exp_003/src/train.py logic, cutoff parameterised) --
PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]
STD_COLS = ("rel_speed", "spin_rate", "rel_height", "rel_side",
            "induced_vert_break", "horz_break")

FLAG_PROFILE = "has_profile"
FLAG_RECENT = "has_recent_profile"


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def lb_score(brier, r):
    """Organizers' score formula: max(0, 100000 * (1 - brier / (r(1-r))))."""
    denom = r * (1.0 - r)
    if denom <= 0:
        return None
    return float(max(0.0, 100000.0 * (1.0 - brier / denom)))


def seg_metrics(y, p):
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
        "score": round(lb_score(brier, r), 3) if denom > 0 else None,
        "base_rate": round(r, 6),
        "pred_mean": round(float(np.mean(p)), 6),
    }
    if 0.0 < r < 1.0:
        out["auc"] = round(float(roc_auc_score(y, p)), 6)
        out["logloss"] = round(float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15))), 6)
    return out


def sub_profile(df, prefix):
    """Aggregate one scope (career-to-cutoff or the cutoff season)."""
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
    """As-of profile: TrackMan seasons <= cutoff only. Indexed by train pitcher_id.
    NaN kept for missing (no sentinel at this stage)."""
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


def build_model(cat_cols, imputed_num_cols, raw_num_cols, max_features, n_jobs=N_JOBS):
    """Organizers' pipeline, with an optional passthrough branch that lets the
    profile columns reach the forest as NaN instead of a median sentinel."""
    steps = [
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
         cat_cols),
        ("num", SimpleImputer(strategy="median"), imputed_num_cols),
    ]
    if raw_num_cols:
        steps.append(("raw", "passthrough", raw_num_cols))
    rf_kwargs = dict(RF_PARAMS)
    if max_features is not None:
        rf_kwargs["max_features"] = max_features
    return Pipeline([
        ("pre", ColumnTransformer(steps)),
        ("clf", RandomForestClassifier(n_jobs=n_jobs, **rf_kwargs)),
    ])


def fit_arm(arm, fold_name, X_tr, y_tr, X_va, y_va, meta, cols, raw_cols, max_features):
    """Fit one (arm, fold) and score it. `cols` = full ordered column list of X."""
    # CAT_COLS order (not `cols` order) so P0 is bit-identical to exp_008 R2:
    # column order feeds the forest's per-split feature subsampling.
    cat_cols = [c for c in CAT_COLS if c in cols]
    raw_cols = [c for c in raw_cols if c in cols]
    imputed_cols = [c for c in cols if c not in cat_cols and c not in raw_cols]
    model = build_model(cat_cols, imputed_cols, raw_cols, max_features)

    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_sec = time.time() - t0
    t1 = time.time()
    p_va = model.predict_proba(X_va)[:, 1]
    infer_sec = time.time() - t1

    # sklearn 1.8: the forest exposes no max_features_; the fitted trees do.
    eff_max_features = int(model.named_steps["clf"].estimators_[0].max_features_)
    print(f"[{time.strftime('%H:%M:%S')}] {arm}/{fold_name}: "
          f"{len(cols)} feats (cat {len(cat_cols)} / imputed {len(imputed_cols)} / "
          f"raw-NaN {len(raw_cols)}), max_features={eff_max_features}, "
          f"fit {train_sec/60:.2f} min", flush=True)

    is_r, cold, has_prof, has_rec = meta
    out = {
        "n_features": len(cols),
        "n_raw_nan_features": len(raw_cols),
        "max_features_effective": eff_max_features,
        "train_time_min": round(train_sec / 60, 2),
        "inference_time_sec": round(infer_sec, 2),
        "valid_all": seg_metrics(y_va, p_va),
        "valid_R": seg_metrics(y_va[is_r], p_va[is_r]),
        "valid_F": seg_metrics(y_va[~is_r], p_va[~is_r]),
        "cold_start_R": seg_metrics(y_va[cold & is_r], p_va[cold & is_r]),
        "seen_R": seg_metrics(y_va[~cold & is_r], p_va[~cold & is_r]),
        "valid_R_has_profile": seg_metrics(y_va[is_r & has_prof], p_va[is_r & has_prof]),
        "valid_R_no_profile": seg_metrics(y_va[is_r & ~has_prof], p_va[is_r & ~has_prof]),
        "valid_R_has_recent": seg_metrics(y_va[is_r & has_rec], p_va[is_r & has_rec]),
    }

    # impurity importance, in ColumnTransformer output order
    names = cat_cols + imputed_cols + raw_cols
    imp = sorted(zip(names, model.named_steps["clf"].feature_importances_),
                 key=lambda t: -t[1])
    out["importance_top25"] = [[f, round(float(g), 6)] for f, g in imp[:25]]
    prof_like = set(n for n in names
                    if n.startswith(("career_", "recent_"))
                    or n in (
                        "last_season_gap", FLAG_PROFILE, FLAG_RECENT))
    if prof_like:
        out["profile_importance_share"] = round(
            float(sum(g for f, g in imp if f in prof_like)), 6)
        out["profile_importance_ranked"] = [
            [rank + 1, f, round(float(g), 6)]
            for rank, (f, g) in enumerate(imp) if f in prof_like and g > 0][:40]

    if fold_name == "fold_2024":
        assert out["valid_all"]["n"] == EXPECTED_2024["valid_all"], out["valid_all"]["n"]
        assert out["valid_R"]["n"] == EXPECTED_2024["valid_R"], out["valid_R"]["n"]
        np.save(os.path.join(OUT_DIR, f"pred_{arm}_fold_2024.npy"),
                p_va.astype(np.float32))

    vr = out["valid_R"]
    print(f"    -> valid_R Brier {vr['brier']:.6f} | BSS {vr['bss']:.6f} | "
          f"score {vr['score']} | AUC {vr['auc']:.4f} | "
          f"pred_mean {vr['pred_mean']:.4f} (r {vr['base_rate']:.4f})", flush=True)
    return out


# arm -> (use_profile, use_flag_profile, use_flag_recent, raw_nan_profile,
#         max_features_override, folds)
ARMS = {
    "P0":  dict(profile=False, flag=False, flag_recent=False, raw=False,
                max_features=None, folds=ALL3,
                spec="47 base features (matched control, = exp_008 R2)"),
    "P0b": dict(profile=False, flag=False, flag_recent=False, raw=False,
                max_features=10, folds=LAST2,
                spec="P0 with max_features=10 (isolates the sqrt-default shift P1 causes)"),
    "P1":  dict(profile=True, flag=False, flag_recent=False, raw=False,
                max_features=None, folds=ALL3,
                spec="P0 + 65 as-of profile features, median-imputed (sentinel for the ~19% missing)"),
    "P2":  dict(profile=True, flag=True, flag_recent=False, raw=False,
                max_features=None, folds=ALL3,
                spec="P1 + has_profile flag"),
    "P3":  dict(profile=True, flag=True, flag_recent=True, raw=False,
                max_features=None, folds=LAST2,
                spec="P2 + has_recent_profile flag"),
    "P4":  dict(profile=True, flag=True, flag_recent=True, raw=True,
                max_features=None, folds=LAST2,
                spec="P3 but profile columns bypass the imputer (native NaN splits)"),
}


def main():
    np.random.seed(SEED)
    t_start = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {TRAIN_PATH}", flush=True)
    df = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    assert df["row_id"].is_unique
    print(f"train loaded: {df.shape}", flush=True)

    base_features = [c for c in pd.read_csv(TEST_PATH, encoding="utf-8-sig",
                                            nrows=0).columns if c != "row_id"]
    assert TARGET not in base_features
    missing = [c for c in base_features if c not in df.columns]
    assert not missing, missing

    pmap = pd.read_csv(PMAP_PATH)
    tm = pd.read_csv(TM_PATH, encoding="utf-8-sig",
                     usecols=["season", "trackman_game_id", "pitcher_trackman_id",
                              "pitch_type_group"] + PHYS)
    assert int(tm["season"].max()) <= 2024, "trackman contains 2025 — violation"
    tm = tm[tm["pitcher_trackman_id"].isin(set(pmap["pitcher_trackman_id"]))]
    print(f"base features: {len(base_features)} | trackman rows (mapped): {len(tm):,}",
          flush=True)

    pid = df["pitcher_id"].to_numpy()
    metrics = {
        "exp_id": "exp_009",
        "base_exp": "exp_008 R2 (RF + exp_004 mask, valid_R Brier 0.248455)",
        "hypothesis": ("the exp_003 TrackMan pitcher profile still improves "
                       "fold_2024 valid_R Brier on the strong RF base; "
                       "has_profile preserves the missingness the median "
                       "imputer would otherwise turn into a sentinel"),
        "seed": SEED,
        "n_jobs": N_JOBS,
        "rf_params": RF_PARAMS,
        "mask": "training rows with (season <= 2022 & game_type == 'F') dropped (exp_004)",
        "arms": {k: v["spec"] for k, v in ARMS.items()},
        "profile": {
            "source": "trackman_history via exp_002 pitcher_id_map",
            "asof_rule": "fold validating year Y aggregates seasons <= Y-1 only",
            "prefixes": ["career_* (all seasons <= cutoff)",
                         "recent_* (season == cutoff)", "last_season_gap"],
            "missing_policy": ("P1-P3: SimpleImputer(median) (organizers' pipeline) "
                               "+ explicit flag column; P4: native NaN"),
        },
        "sha256": {
            "train.csv": sha256_file(TRAIN_PATH),
            "trackman_history.csv": sha256_file(TM_PATH),
            "pitcher_id_map.csv": sha256_file(PMAP_PATH),
            "train.py": sha256_file(os.path.abspath(__file__)),
        },
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
        "runs": {a: {"spec": ARMS[a]["spec"]} for a in ARMS},
    }

    total_infer = 0.0
    for fold_name in ALL3:
        train_until, valid_year = FOLDS[fold_name]
        cutoff = valid_year - 1
        assert cutoff == train_until, "rolling-origin: profile horizon == train horizon"

        tp = time.time()
        prof = build_asof_profile(tm, pmap, cutoff)
        prof_cols = list(prof.columns)
        print(f"\n[{time.strftime('%H:%M:%S')}] === {fold_name} === as-of cutoff "
              f"{cutoff}: {prof.shape[0]} pitchers x {len(prof_cols)} feats "
              f"({time.time()-tp:.1f}s)", flush=True)

        tr_mask = ((df["season"] <= train_until)
                   & ~((df["season"] <= 2022) & (df["game_type"] == "F"))).to_numpy()
        va_mask = (df["season"] == valid_year).to_numpy()

        def make_X(mask):
            X = df.loc[mask, base_features].copy()
            blk = prof.reindex(pid[mask])
            blk.index = X.index
            X = pd.concat([X, blk], axis=1)
            X[FLAG_PROFILE] = X["career_n_pitches"].notna().astype("int8")
            X[FLAG_RECENT] = X["recent_n_pitches"].notna().astype("int8")
            return X

        X_tr_full, X_va_full = make_X(tr_mask), make_X(va_mask)
        y_tr = df.loc[tr_mask, TARGET].to_numpy()
        y_va = df.loc[va_mask, TARGET].to_numpy()

        is_r = (df.loc[va_mask, "game_type"] == "R").to_numpy()
        seen = set(pid[tr_mask])
        cold = ~np.isin(pid[va_mask], list(seen))
        has_prof = X_va_full[FLAG_PROFILE].to_numpy().astype(bool)
        has_rec = X_va_full[FLAG_RECENT].to_numpy().astype(bool)
        meta = (is_r, cold, has_prof, has_rec)

        cov = {
            "n_train": int(tr_mask.sum()),
            "n_valid": int(va_mask.sum()),
            "profile_cutoff": cutoff,
            "n_profile_pitchers": int(prof.shape[0]),
            "n_profile_features": len(prof_cols),
            "train_profile_row_coverage": round(
                float(X_tr_full[FLAG_PROFILE].mean()), 4),
            "valid_profile_row_coverage": round(float(has_prof.mean()), 4),
            "valid_recent_profile_row_coverage": round(float(has_rec.mean()), 4),
            "valid_R_profile_row_coverage": round(float(has_prof[is_r].mean()), 4),
        }
        metrics.setdefault("coverage", {})[fold_name] = cov
        print(json.dumps(cov, indent=2), flush=True)

        for arm, cfg in ARMS.items():
            if fold_name not in cfg["folds"]:
                continue
            cols = list(base_features)
            if cfg["profile"]:
                cols += prof_cols
            if cfg["flag"]:
                cols.append(FLAG_PROFILE)
            if cfg["flag_recent"]:
                cols.append(FLAG_RECENT)
            raw_cols = prof_cols if cfg["raw"] else []
            res = fit_arm(arm, fold_name, X_tr_full[cols], y_tr, X_va_full[cols],
                          y_va, meta, cols, raw_cols, cfg["max_features"])
            res.update({"train_until": train_until, "valid_year": valid_year,
                        "n_train": cov["n_train"]})
            metrics["runs"][arm][fold_name] = res
            total_infer += res["inference_time_sec"]
            with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)

        del X_tr_full, X_va_full

    # ---------------- summary: matched single-variable deltas ----------------
    ref = {"exp_008_R2_fold_2024_valid_R_brier": 0.248455,
           "exp_008_R1_fold_2024_valid_R_brier": 0.248487,
           "exp_007_V1_fold_2024_valid_R_brier": 0.249061,
           "target_1061_points_brier": 0.24724}

    def vr(arm, fold, key="brier"):
        r = metrics["runs"].get(arm, {}).get(fold)
        return r["valid_R"][key] if r else None

    summary = {"references": ref, "fold_2024_valid_R_brier": {
        a: vr(a, "fold_2024") for a in ARMS}}
    p0 = vr("P0", "fold_2024")
    if p0 is not None:
        summary["control_reproduces_exp008_R2"] = round(
            p0 - ref["exp_008_R2_fold_2024_valid_R_brier"], 8)
        summary["delta_vs_P0_fold_2024_valid_R"] = {
            a: (None if vr(a, "fold_2024") is None
                else round(vr(a, "fold_2024") - p0, 8)) for a in ARMS if a != "P0"}
    chain = [("P0->P0b", "P0", "P0b"), ("P0->P1", "P0", "P1"),
             ("P1->P2", "P1", "P2"), ("P2->P3", "P2", "P3"), ("P3->P4", "P3", "P4")]
    summary["single_variable_chain_fold_2024_valid_R"] = {
        name: (None if vr(a, "fold_2024") is None or vr(b, "fold_2024") is None
               else round(vr(b, "fold_2024") - vr(a, "fold_2024"), 8))
        for name, a, b in chain}
    summary["fold_2024_valid_R_score"] = {a: vr(a, "fold_2024", "score") for a in ARMS}
    summary["fold_2024_valid_R_auc"] = {a: vr(a, "fold_2024", "auc") for a in ARMS}
    summary["fold_2023_valid_R_brier"] = {a: vr(a, "fold_2023") for a in ARMS}
    summary["fold_2022_valid_R_brier"] = {a: vr(a, "fold_2022") for a in ARMS}
    metrics["summary"] = summary

    # top-level standard shape = P2 (the arm the spec names: profile + flag)
    for fold_name in ALL3:
        r = metrics["runs"]["P2"].get(fold_name)
        if r:
            metrics[fold_name] = {
                "brier": r["valid_R"]["brier"], "bss": r["valid_R"]["bss"],
                "auc": r["valid_R"]["auc"], "score": r["valid_R"]["score"],
                "gauge": "valid_R", "arm": "P2"}
    metrics["train_time_min"] = round((time.time() - t_start) / 60, 2)
    metrics["inference_time_sec"] = round(total_infer, 2)
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print("\n" + json.dumps(summary, indent=2), flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE total {metrics['train_time_min']} min",
          flush=True)


if __name__ == "__main__":
    main()

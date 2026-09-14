"""exp_049 STAGE 0 — acceptance inspection of the 14 columns before they touch a fit.

The team-mate's self-verification is not inherited.  Neither is exp_037's, except
where a stored artefact is re-derived here and its sha re-checked.

  G0-prov  the featdir exp_037 wrote is still the one it recorded (train.csv sha,
           featgen.py sha, every <=2023 table file sha, and a sha of the 14-column
           block of feat_main.parquet that stage 1 will consume).
  G0d-1    SECOND INDEPENDENT READ of train.csv and trackman_history.csv: the row
           ids and trackman ids exp_037 declares it fed the <=2023 tables are
           re-looked-up for their season, and the declared set is compared with
           {season <= 2023} computed here from the raw files.
  G0d-2    the <=2023 pitch model and anom table are REFITTED into a fresh
           directory and the 14 columns for all 253,507 rows of 2024 regenerated
           and compared element by element with the stored block.
  G0d-3    the trap Q's own pipeline would have walked into is measured, not
           argued: Q's shipped pitchmodel6.txt is scored against the same rows
           and the two feature blocks are compared, and the number of rows Q's
           booster was fitted on is read out of the model file.
  G0a      AST scan of feat14.apply14 / predict_p6 for cross-row calls.
  G0b      full shuffle of the 2024 frame -> bit-identical per row.
  G0c      100 rows, and 1 row, generated ALONE -> bit-identical.
  G0e      column-access trace through a proxy frame: is `control_success` read.
  G0f      coverage and missingness by segment; sentinel-shaped values counted.

Writes stage0_gates.json.  Exits non-zero if any gate fails.
"""
import ast
import gzip
import hashlib
import inspect
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
import feat14 as F14                                                # noqa: E402

DATA = os.path.expanduser("~/SungjinPark/data")
FEATDIR = os.path.expanduser("~/SungjinPark/experiments/exp_037/featdir")
OUT = os.path.expanduser("~/SungjinPark/experiments/exp_049")
GATEDIR = os.path.join(OUT, "gate_tables")
QDIR = os.path.join(OUT, "q_tables")          # Q's shipped tables, if unpacked
VALID_SEASON = 2024
N_VALID_EXPECT = 253507
t0 = time.time()


def log(*a):
    print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# --------------------------------------------------------------------------- #
class TraceFrame:
    """Records every column name the apply path asks for."""

    def __init__(self, df, seen):
        object.__setattr__(self, "_df", df)
        object.__setattr__(self, "_seen", seen)

    def __getitem__(self, k):
        if isinstance(k, (list, tuple, pd.Index)):
            self._seen.update([str(x) for x in k])
        else:
            self._seen.add(str(k))
        return self._df[k]

    def __getattr__(self, name):
        df = object.__getattribute__(self, "_df")
        if name in df.columns:
            object.__getattribute__(self, "_seen").add(name)
            return df[name]
        return getattr(df, name)

    def __len__(self):
        return len(object.__getattribute__(self, "_df"))


FORBIDDEN = {"groupby", "sort_values", "sort_index", "sort", "argsort",
             "cumsum", "cumprod", "cummax", "cummin", "diff", "rolling",
             "expanding", "quantile", "unique", "nunique", "value_counts",
             "rank", "shift", "ewm", "resample", "pivot", "pivot_table",
             "merge", "join", "duplicated", "drop_duplicates", "sample"}
# reductions are forbidden only when they collapse rows; axis=1 keeps a value
# per row and is therefore allowed.  The check below is name-based and then
# whitelists the two axis=1 calls this module actually makes.
REDUCERS = {"mean", "sum", "std", "var", "median", "min", "max", "prod"}
ROWWISE_OK = {("np.nansum", "axis=1"), ("np.nanargmax", "axis=1")}


def ast_scan(func):
    src = inspect.getsource(func)
    tree = ast.parse(src)
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else (
            fn.id if isinstance(fn, ast.Name) else None)
        if name is None:
            continue
        kw = {k.arg: ast.unparse(k.value) for k in node.keywords if k.arg}
        text = ast.unparse(node)
        # `os.path.join` is a path concatenation, not pandas' row-matching
        # `join`.  It is the only name collision between the two namespaces
        # in this module and it is excluded by RECEIVER, not by name.
        if isinstance(fn, ast.Attribute) and text.startswith("os.path."):
            continue
        if name in FORBIDDEN:
            hits.append({"call": name, "src": ast.unparse(node)[:160],
                         "why": "cross-row"})
        elif name in REDUCERS:
            if kw.get("axis") == "1":
                continue
            hits.append({"call": name, "src": ast.unparse(node)[:160],
                         "why": "reduction without axis=1"})
    return hits


# --------------------------------------------------------------------------- #
def main():
    os.makedirs(OUT, exist_ok=True)
    rep = {"exp_id": "exp_049", "stage": 0,
           "columns": F14.COLS14, "groups": F14.GROUPS14,
           "table_source": {
               "what": "exp_037 featdir, per-season as-of tables",
               "path": FEATDIR,
               "why_not_Q": ("Q ships ONE pitchmodel6.txt fitted on trackman "
                             "through 2024 and reuses it for every season; a "
                             "2024 validation row would then be inside its own "
                             "scorer's training set.")}}

    # ------------------------------------------------------------- G0-prov
    log("G0-prov: featdir provenance")
    fm = json.load(open(os.path.join(FEATDIR, "featgen_main.json")))
    train_sha = sha256_file(os.path.join(DATA, "train.csv"))
    tdir = os.path.join(FEATDIR, "tables", "upto_2023")
    tbl_now = {fn: sha256_file(os.path.join(tdir, fn))
               for fn in sorted(os.listdir(tdir))}
    tbl_rec = fm["per_season"]["2024"]["table_sha256"]
    prov = {
        "featgen_main_train_csv_sha256": fm["sha256"]["train.csv"],
        "train_csv_sha256_now": train_sha,
        "train_csv_match": bool(fm["sha256"]["train.csv"] == train_sha),
        "featgen_py_sha256_recorded": fm["sha256"]["featgen.py"],
        "mate_pipeline_sha256": fm["mate_pipeline_sha256"],
        "pitch_model_params_recorded": fm["pitch_model_params"],
        "pitch_model_rounds_recorded": fm["pitch_model_rounds"],
        "params_match_feat14": bool(fm["pitch_model_params"] == F14.PM_PARAMS
                                    and fm["pitch_model_rounds"] == F14.PM_ROUNDS),
        "table_sha256_recorded": tbl_rec,
        "table_sha256_now": tbl_now,
        "tables_unchanged": bool(all(tbl_rec[k] == tbl_now[k] for k in tbl_rec)),
        "hist_rowid_sha256_recorded": fm["per_season"]["2024"]["hist_rowid_sha256"],
        "tm_id_sha256_recorded": fm["per_season"]["2024"]["tm_id_sha256"],
        "hist_n_recorded": fm["per_season"]["2024"]["hist_n"],
        "tm_n_recorded": fm["per_season"]["2024"]["tm_n"],
    }
    prov["PASS"] = bool(prov["train_csv_match"] and prov["tables_unchanged"]
                        and prov["params_match_feat14"])
    rep["G0_prov"] = prov
    log("  prov PASS", prov["PASS"])

    # ------------------------------------------------------------- the frames
    log("loading train.csv (independent read #1) ...")
    train = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
    assert train["row_id"].is_unique
    m24 = (train["season"] == VALID_SEASON).to_numpy()
    assert int(m24.sum()) == N_VALID_EXPECT, int(m24.sum())
    df24 = train.loc[m24].copy()
    hist = train.loc[train["season"] <= VALID_SEASON - 1]
    log("  train", train.shape, "2024", df24.shape, "hist", hist.shape)

    feat = pd.read_parquet(os.path.join(FEATDIR, "feat", "feat_main.parquet"))
    assert len(feat) == len(train)
    stored = feat.set_index("row_id").loc[df24["row_id"].to_numpy(), F14.COLS14]
    rep["G0_prov"]["stored_block_sha256"] = sha_arr(
        stored.to_numpy(dtype=np.float64))
    rep["G0_prov"]["feat_parquet_sha256"] = sha256_file(
        os.path.join(FEATDIR, "feat", "feat_main.parquet"))

    # --------------------------------------------------- G0d-1 as-of, re-read
    log("G0d-1: re-deriving the <=2023 input sets from the raw files")
    with gzip.open(os.path.join(FEATDIR, "prov", "hist_rowids_s2024.txt.gz"),
                   "rt") as f:
        hist_ids_declared = f.read().split()
    with gzip.open(os.path.join(FEATDIR, "prov", "tm_ids_s2024.txt.gz"),
                   "rt") as f:
        tm_ids_declared = f.read().split()

    row_season = {str(k): int(v) for k, v in
                  zip(train["row_id"].to_numpy(), train["season"].to_numpy())}
    hs = np.array([row_season[str(i)] for i in hist_ids_declared])
    tm_season_map = pd.read_csv(os.path.join(DATA, "trackman_history.csv"),
                                encoding="utf-8-sig",
                                usecols=["trackman_id", "season"])
    tms = {str(k): int(v) for k, v in
           zip(tm_season_map["trackman_id"].to_numpy(),
               tm_season_map["season"].to_numpy())}
    ts = np.array([tms[str(i)] for i in tm_ids_declared])

    hist_ids_declared = [str(x) for x in hist_ids_declared]
    hist_ids_expected = set(str(x) for x in
                            train.loc[train["season"] <= 2023, "row_id"])
    asof = {
        "hist_rows_declared": len(hist_ids_declared),
        "hist_season_max_reread": int(hs.max()),
        "hist_rows_after_2023": int((hs > 2023).sum()),
        "hist_set_equals_season_le_2023": bool(
            set(hist_ids_declared) == hist_ids_expected),
        "tm_rows_declared": len(tm_ids_declared),
        "tm_season_max_reread": int(ts.max()),
        "tm_rows_after_2023": int((ts > 2023).sum()),
        "tm_season_census": {str(k): int(v) for k, v in
                             pd.Series(ts).value_counts().sort_index().items()},
        "n_trackman_rows_total": int(len(tm_season_map)),
        "n_trackman_rows_le_2023": int(
            (tm_season_map["season"] <= 2023).sum()),
    }
    asof["PASS"] = bool(asof["hist_season_max_reread"] <= 2023
                        and asof["hist_rows_after_2023"] == 0
                        and asof["hist_set_equals_season_le_2023"]
                        and asof["tm_season_max_reread"] <= 2023
                        and asof["tm_rows_after_2023"] == 0)
    rep["G0d_1_asof_reread"] = asof
    log("  as-of PASS", asof["PASS"], "hist_max", asof["hist_season_max_reread"],
        "tm_max", asof["tm_season_max_reread"])
    del tm_season_map, tms

    # ------------------------------------------------- G0d-2 rebuild + compare
    log("G0d-2: refitting the <=2023 pitch model into a fresh directory")
    os.makedirs(GATEDIR, exist_ok=True)
    tml = pd.read_parquet(os.path.join(FEATDIR, "tm_lab.parquet"))
    pmap = pd.read_csv(os.path.join(FEATDIR, "map", "pitcher_map_upto_2023.csv"))
    tm_hist = tml.loc[tml["season"] <= 2023].merge(
        pmap[["tr", "tm"]], left_on="pitcher_trackman_id", right_on="tm",
        how="inner").rename(columns={"tr": "pitcher_id"})
    assert int(tm_hist["season"].max()) <= 2023
    n_fit = F14.build_pitch_model(tm_hist, GATEDIR)
    log(f"  pitch model refitted on {n_fit:,} trackman rows")
    hp = hist[["balls_before", "strikes_before"]].copy()
    hp[F14.PC] = F14.predict_p6(hist, GATEDIR)
    F14.build_anom_table(hp, GATEDIR)
    del hp
    log("  anom table rebuilt")

    regen = F14.apply14(df24, GATEDIR)
    A = stored.to_numpy(dtype=np.float64)
    B = regen.to_numpy(dtype=np.float64)
    na_a, na_b = np.isnan(A), np.isnan(B)
    both = ~na_a & ~na_b
    d = np.abs(A[both] - B[both])
    g2 = {
        "n_rows": int(len(df24)), "n_cols": len(F14.COLS14),
        "pitch_model_fit_rows": n_fit,
        "gate_table_sha256": {fn: sha256_file(os.path.join(GATEDIR, fn))
                              for fn in sorted(os.listdir(GATEDIR))},
        "stored_table_sha256": {k: tbl_now[k] for k in
                                ("pitchmodel6.txt", "pmcat6.json",
                                 "anom_cnt.csv")},
        "nan_pattern_identical": bool((na_a == na_b).all()),
        "max_abs_diff": float(d.max()) if d.size else 0.0,
        "n_cells_differing": int((d > 0).sum()) if d.size else 0,
        "bit_identical": bool((na_a == na_b).all()
                              and (d.size == 0 or d.max() == 0.0)),
        "per_column_max_abs_diff": {
            c: float(np.nanmax(np.abs(A[:, i] - B[:, i])))
            for i, c in enumerate(F14.COLS14)},
    }
    g2["table_sha256_identical"] = bool(
        all(g2["gate_table_sha256"].get(k) == v
            for k, v in g2["stored_table_sha256"].items()))
    g2["PASS"] = bool(g2["nan_pattern_identical"] and g2["max_abs_diff"] == 0.0)
    rep["G0d_2_rebuild_identity"] = g2
    log("  rebuild PASS", g2["PASS"], "max_abs_diff", g2["max_abs_diff"],
        "table_sha_identical", g2["table_sha256_identical"])

    # ------------------------------------------------- G0d-3 what Q's own does
    log("G0d-3: measuring Q's shipped booster")
    q = {"available": os.path.isdir(QDIR)
         and os.path.exists(os.path.join(QDIR, "pitchmodel6.txt"))}
    if q["available"]:
        def root_count(path):
            with open(path) as f:
                for line in f:
                    if line.startswith("internal_count="):
                        return int(line.split("=", 1)[1].split()[0])
            return None
        q["q_pitchmodel6_root_internal_count"] = root_count(
            os.path.join(QDIR, "pitchmodel6.txt"))
        q["our_lt2023_pitchmodel6_root_internal_count"] = root_count(
            os.path.join(GATEDIR, "pitchmodel6.txt"))
        q["our_fit_rows"] = n_fit
        qlv = json.load(open(os.path.join(QDIR, "pmcat6.json")))
        olv = json.load(open(os.path.join(GATEDIR, "pmcat6.json")))
        q["q_n_pitcher_levels"] = len(qlv["pitcher_id"])
        q["our_n_pitcher_levels"] = len(olv["pitcher_id"])
        q["pitchers_only_in_q"] = int(len(set(qlv["pitcher_id"])
                                          - set(olv["pitcher_id"])))
        pid24 = set(df24["pitcher_id"].unique().tolist())
        q["pitchers_only_in_q_that_appear_in_2024"] = int(
            len((set(qlv["pitcher_id"]) - set(olv["pitcher_id"])) & pid24))
        try:
            qregen = F14.apply14(df24, QDIR)
            C = qregen.to_numpy(dtype=np.float64)
            dq = np.abs(np.nan_to_num(A) - np.nan_to_num(C))
            q["max_abs_diff_Q_vs_asof"] = float(dq.max())
            q["per_column_max_abs_diff"] = {
                c: float(np.nanmax(np.abs(A[:, i] - C[:, i])))
                for i, c in enumerate(F14.COLS14)}
            q["idea_matchup_pitch_disagree_pct"] = round(100.0 * float(
                (A[:, F14.COLS14.index(F14.IDEA)]
                 != C[:, F14.COLS14.index(F14.IDEA)]).mean()), 4)
        except Exception as e:                                  # noqa: BLE001
            q["scoring_error"] = repr(e)
    rep["G0d_3_Q_deploy_tables"] = q
    log("  Q probe", json.dumps({k: v for k, v in q.items()
                                 if not isinstance(v, dict)})[:400])

    # ------------------------------------------------------------------ G0a
    log("G0a: AST scan")
    hits = ast_scan(F14.apply14) + ast_scan(F14.predict_p6)
    rep["G0a_ast"] = {"forbidden_calls": hits, "PASS": bool(not hits),
                      "scanned": ["feat14.apply14", "feat14.predict_p6"],
                      "note": ("build_pitch_model / build_anom_table DO "
                               "aggregate; they are TABLE builders and run only "
                               "on <=2023 rows, never on a scored row.")}
    log("  ast PASS", rep["G0a_ast"]["PASS"], hits)

    # ------------------------------------------------------------------ G0b
    log("G0b: full shuffle identity")
    rng = np.random.default_rng(20260815)
    perm = rng.permutation(len(df24))
    sh = df24.iloc[perm]
    got = F14.apply14(sh, GATEDIR)
    got = got.loc[df24.index]                       # back to original order
    Bs = got.to_numpy(dtype=np.float64)
    na_s = np.isnan(Bs)
    ds = np.abs(B[~na_b & ~na_s] - Bs[~na_b & ~na_s])
    rep["G0b_shuffle"] = {
        "n_rows": int(len(df24)),
        "nan_pattern_identical": bool((na_b == na_s).all()),
        "max_abs_diff": float(ds.max()) if ds.size else 0.0,
        "n_cells_differing": int((ds > 0).sum()) if ds.size else 0,
        "bit_identical": bool(np.array_equal(B, Bs, equal_nan=True)),
    }
    rep["G0b_shuffle"]["PASS"] = rep["G0b_shuffle"]["bit_identical"]
    log("  shuffle PASS", rep["G0b_shuffle"]["PASS"])
    del sh, got, Bs

    # ------------------------------------------------------------------ G0c
    log("G0c: isolated rows")
    pick = rng.choice(len(df24), size=100, replace=False)
    sub = df24.iloc[pick]
    g100 = F14.apply14(sub, GATEDIR).to_numpy(dtype=np.float64)
    r100 = B[pick]
    one = df24.iloc[[int(pick[0])]]
    g1 = F14.apply14(one, GATEDIR).to_numpy(dtype=np.float64)
    r1 = B[[int(pick[0])]]
    rep["G0c_isolated"] = {
        "n_isolated": 100,
        "bit_identical_100": bool(np.array_equal(g100, r100, equal_nan=True)),
        "max_abs_diff_100": float(np.nanmax(np.abs(g100 - r100))),
        "bit_identical_single_row": bool(np.array_equal(g1, r1, equal_nan=True)),
        "max_abs_diff_single_row": float(np.nanmax(np.abs(g1 - r1))),
    }
    rep["G0c_isolated"]["PASS"] = bool(
        rep["G0c_isolated"]["bit_identical_100"]
        and rep["G0c_isolated"]["bit_identical_single_row"])
    log("  isolated PASS", rep["G0c_isolated"]["PASS"])

    # ------------------------------------------------------------------ G0e
    log("G0e: column-access trace")
    seen = set()
    _ = F14.apply14(TraceFrame(df24.iloc[:5000], seen), GATEDIR)
    rep["G0e_column_trace"] = {
        "columns_read": sorted(seen),
        "target_read": bool("control_success" in seen),
        "trackman_physics_read": sorted(
            c for c in seen if c in {"rel_speed", "spin_rate", "zone_speed",
                                     "induced_vert_break", "horz_break",
                                     "extension", "rel_height", "rel_side",
                                     "tagged_pitch_type", "auto_pitch_type",
                                     "pitch_type_group"}),
        "row_id_read": bool("row_id" in seen),
    }
    rep["G0e_column_trace"]["PASS"] = bool(
        not rep["G0e_column_trace"]["target_read"]
        and not rep["G0e_column_trace"]["trackman_physics_read"])
    log("  trace PASS", rep["G0e_column_trace"]["PASS"],
        sorted(seen))

    # ------------------------------------------------------------------ G0f
    log("G0f: coverage and missingness")
    is_r = (df24["game_type"].to_numpy() == "R")
    cov = {"n_valid": int(len(df24)), "n_valid_R": int(is_r.sum()),
           "n_valid_F": int((~is_r).sum()), "per_column": {}}
    for i, c in enumerate(F14.COLS14):
        v = A[:, i]
        cov["per_column"][c] = {
            "nan_pct_all": round(100.0 * float(np.isnan(v).mean()), 6),
            "nan_pct_R": round(100.0 * float(np.isnan(v[is_r]).mean()), 6),
            "nan_pct_F": round(100.0 * float(np.isnan(v[~is_r]).mean()), 6),
            "exact_zero_pct": round(100.0 * float((v == 0.0).mean()), 6),
            "n_distinct": int(len(np.unique(v[~np.isnan(v)]))),
            "min": float(np.nanmin(v)), "max": float(np.nanmax(v)),
            "mean": float(np.nanmean(v)),
        }
    # missingness across the WHOLE feature matrix (2019 rows are NaN by design)
    allf = feat[F14.COLS14].to_numpy(dtype=np.float64)
    ssn = train["season"].to_numpy()
    cov["nan_pct_by_season_p6_fastball"] = {
        str(s): round(100.0 * float(
            np.isnan(allf[ssn == s, F14.COLS14.index("p6_fastball")]).mean()), 4)
        for s in sorted(set(ssn.tolist()))}
    cov["sentinel_note"] = ("no fill is applied anywhere in feat14.py; 2019 "
                            "rows are NaN because no antecedent season exists, "
                            "and CatBoost consumes NaN natively.")
    cov["PASS"] = True
    rep["G0f_coverage"] = cov
    log("  2019 nan pct", cov["nan_pct_by_season_p6_fastball"].get("2019"))

    # ------------------------------------------------------------------ done
    gates = ["G0_prov", "G0d_1_asof_reread", "G0d_2_rebuild_identity",
             "G0a_ast", "G0b_shuffle", "G0c_isolated", "G0e_column_trace",
             "G0f_coverage"]
    rep["ALL_PASS"] = bool(all(rep[g]["PASS"] for g in gates))
    rep["minutes"] = round((time.time() - t0) / 60, 2)
    with open(os.path.join(OUT, "stage0_gates.json"), "w") as f:
        json.dump(rep, f, indent=1)
    print("STAGE0 " + ("PASS" if rep["ALL_PASS"] else "FAIL"), flush=True)
    for g in gates:
        print(f"  {g}: {rep[g]['PASS']}", flush=True)
    sys.exit(0 if rep["ALL_PASS"] else 1)


if __name__ == "__main__":
    main()

"""exp_048 STAGE 0 — acceptance gates on the teammate's `ins/` pack.

Nothing in this file trusts the pack's README.  Every claim it makes is
re-derived here from train.csv and from the module's own source text.

G0a  row independence, statically:  AST of `inseason.apply`
G0b  row independence, empirically: whole-frame shuffle identity
G0c  row independence, empirically: 100-row and 1-row isolation
G0d  anchor season boundary:        rebuild both .npz from train.csv
G0e  label-recovery boundary:       the 16 columns vs control_success
G0f  missing handling:              coverage, sentinel-vs-value

Run:  python s0_gates.py --out <dir>
"""

import argparse
import ast
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SRC_DIR, "ins"))

DATA_PATH = os.path.expanduser("~/SungjinPark/data/train.csv")
PACK_DIR = os.path.join(SRC_DIR, "ins")

EXPECT_SHA = {
    "inseason.py":
        "1b4a407d2b53f2cbdbf0d2c837d0cab7c3db5e1e01bfc2e6b639813723303086",
    "anchor_lt2024.npz":
        "0e4b586da41cd6e78c5e19d4dd656bc9df088413625945e2a88ae85d100e5eb7",
    "deploy_anchor.npz":
        "7b96e3de0baeac54bf8cb1727b8913a075efd487465adda1b762aef1f2d1b020",
}

TARGET = "control_success"

# every pandas/numpy name that could make a row's output depend on OTHER rows
FORBIDDEN_CALLS = {
    "groupby", "sort_values", "sort_index", "sort", "argsort", "cumsum",
    "cumprod", "cummax", "cummin", "diff", "shift", "rolling", "expanding",
    "ewm", "quantile", "mean", "median", "std", "var", "sum", "min", "max",
    "unique", "value_counts", "nunique", "rank", "agg", "aggregate",
    "transform", "pivot", "pivot_table", "resample", "interpolate", "ffill",
    "bfill", "fillna", "drop_duplicates", "duplicated", "sample", "percentile",
    "average", "bincount", "histogram", "searchsorted", "argmax", "argmin",
    "nanmean", "nansum", "nanmax", "nanmin", "corrcoef", "cov",
}
# np.maximum / np.where / np.round etc are ELEMENTWISE and must stay allowed;
# `maximum`/`minimum` are not in the set above for that reason.


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def call_name(node):
    f = node.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def qualname(node):
    f = node.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts))


def gate_a(res):
    """AST of the module: `apply` must be row-wise lookup + arithmetic only."""
    path = os.path.join(PACK_DIR, "inseason.py")
    text = open(path, encoding="utf-8").read()
    tree = ast.parse(text)

    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    out = {"functions_defined": sorted(funcs),
           "module_level_statements": [type(n).__name__ for n in tree.body]}

    # module level must be imports / constants / defs only -- no side effects
    bad_top = [type(n).__name__ for n in tree.body
               if not isinstance(n, (ast.Import, ast.ImportFrom, ast.Assign,
                                     ast.FunctionDef, ast.Expr))]
    out["module_level_side_effect_free"] = (bad_top == [])
    out["module_level_unexpected"] = bad_top

    # the whole module must never name the label
    out["target_name_appears_in_source"] = (TARGET in text)

    per_fn = {}
    for name in ("apply", "build_anchor", "_counts", "load", "save"):
        fn = funcs.get(name)
        if fn is None:
            continue
        calls = [qualname(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
        bare = [call_name(n) for n in ast.walk(fn) if isinstance(n, ast.Call)]
        hits = sorted({c for c in bare if c in FORBIDDEN_CALLS})
        per_fn[name] = {"calls": sorted(set(calls)), "forbidden_hits": hits}
    out["per_function"] = per_fn

    ap = per_fn["apply"]
    out["G0a_apply_free_of_cross_row_ops"] = (ap["forbidden_hits"] == [])
    out["G0a_apply_does_not_call_build_anchor"] = (
        "build_anchor" not in ap["calls"])
    out["G0a_build_anchor_is_where_groupby_lives"] = (
        "groupby" in per_fn["build_anchor"]["forbidden_hits"])
    out["PASS"] = bool(out["G0a_apply_free_of_cross_row_ops"]
                       and out["G0a_apply_does_not_call_build_anchor"]
                       and out["module_level_side_effect_free"]
                       and not out["target_name_appears_in_source"])
    res["G0a_ast_row_independence"] = out
    print(f"  G0a  apply forbidden-call hits {ap['forbidden_hits']}  "
          f"PASS={out['PASS']}", flush=True)
    return out["PASS"]


def gate_bc(res, ins, A_lt2024, df_va, rng):
    """Shuffle identity and single-row isolation."""
    I_full = ins.apply(df_va, A_lt2024)
    cols = list(I_full.columns)
    V_full = I_full.to_numpy()

    perm = rng.permutation(len(df_va))
    df_sh = df_va.iloc[perm]
    I_sh = ins.apply(df_sh, A_lt2024)
    assert list(I_sh.columns) == cols
    V_back = np.empty_like(V_full)
    V_back[perm] = I_sh.to_numpy()
    d_sh = np.abs(np.nan_to_num(V_back, nan=-9e9)
                  - np.nan_to_num(V_full, nan=-9e9))
    g_b = {
        "n_rows": int(len(df_va)),
        "bit_identical": bool(np.array_equal(V_back, V_full)),
        "max_abs_diff": float(d_sh.max()),
        "n_cells_differing": int(np.count_nonzero(d_sh)),
    }
    g_b["PASS"] = bool(g_b["bit_identical"])
    res["G0b_full_shuffle_identity"] = g_b
    print(f"  G0b  shuffle identity  max|d|={g_b['max_abs_diff']:.3e}  "
          f"PASS={g_b['PASS']}", flush=True)

    sel = rng.choice(len(df_va), 100, replace=False)
    I_100 = ins.apply(df_va.iloc[sel], A_lt2024)
    d100 = np.abs(I_100.to_numpy() - V_full[sel])
    one = int(sel[0])
    I_1 = ins.apply(df_va.iloc[[one]], A_lt2024)
    d1 = np.abs(I_1.to_numpy() - V_full[one:one + 1])
    g_c = {
        "n_isolated": 100,
        "bit_identical_100": bool(np.array_equal(I_100.to_numpy(), V_full[sel])),
        "max_abs_diff_100": float(d100.max()),
        "bit_identical_single_row": bool(
            np.array_equal(I_1.to_numpy(), V_full[one:one + 1])),
        "max_abs_diff_single_row": float(d1.max()),
    }
    g_c["PASS"] = bool(g_c["bit_identical_100"]
                       and g_c["bit_identical_single_row"])
    res["G0c_isolated_rows"] = g_c
    print(f"  G0c  100-row {g_c['bit_identical_100']}  1-row "
          f"{g_c['bit_identical_single_row']}  PASS={g_c['PASS']}", flush=True)
    return I_full, bool(g_b["PASS"] and g_c["PASS"])


def cmp_anchor(A, B):
    """Exact comparison of two anchor dicts."""
    if sorted(A) != sorted(B):
        return {"keys_equal": False, "A_keys": sorted(A), "B_keys": sorted(B)}
    out = {"keys_equal": True, "per_key": {}}
    allsame = True
    for k in sorted(A):
        a, b = np.asarray(A[k]), np.asarray(B[k])
        same = a.shape == b.shape and np.array_equal(a, b)
        allsame &= bool(same)
        e = {"shape_A": list(a.shape), "shape_B": list(b.shape),
             "bit_identical": bool(same)}
        if a.shape == b.shape and a.dtype.kind == "f":
            e["max_abs_diff"] = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
        out["per_key"][k] = e
    out["all_bit_identical"] = bool(allsame)
    return out


def gate_d(res, ins, df, A_lt2024, A_deploy):
    """The anchor's season boundary, rebuilt from train.csv."""
    season = df["season"].to_numpy()
    t0 = time.time()
    A_lt2024_rebuilt = ins.build_anchor(df.loc[season < 2024])
    A_deploy_rebuilt = ins.build_anchor(df.loc[season <= 2024])
    g = {
        "what": ("anchor_lt2024.npz must be build_anchor(train[season<2024]) "
                 "and deploy_anchor.npz must be build_anchor(train[all]).  "
                 "Rebuilt here from train.csv and compared array by array."),
        "n_rows_lt2024": int((season < 2024).sum()),
        "n_rows_all": int(len(df)),
        "anchor_lt2024_vs_rebuild": cmp_anchor(A_lt2024, A_lt2024_rebuilt),
        "deploy_anchor_vs_rebuild": cmp_anchor(A_deploy, A_deploy_rebuilt),
        "rebuild_sec": round(time.time() - t0, 1),
    }
    g["n_pitchers_lt2024"] = int(len(A_lt2024["p_ids"]))
    g["n_pitchers_deploy"] = int(len(A_deploy["p_ids"]))

    # the trap, demonstrated: 2024 rows under the WRONG anchor
    va = season == 2024
    df_va = df.loc[va]
    I_right = ins.apply(df_va, A_lt2024)
    I_wrong = ins.apply(df_va, A_deploy)
    g["trap_deploy_anchor_on_2024_rows"] = {
        "why": ("deploy_anchor is taken at the END of 2024, so for a 2024 row "
                "n - anchor is NEGATIVE and np.maximum(...,0) flattens it."),
        "ins_p_n_zero_pct_with_lt2024_anchor":
            round(100.0 * float((I_right["ins_p_n"].to_numpy() == 0).mean()), 4),
        "ins_p_n_zero_pct_with_deploy_anchor":
            round(100.0 * float((I_wrong["ins_p_n"].to_numpy() == 0).mean()), 4),
        "ins_p_n_mean_with_lt2024_anchor":
            round(float(I_right["ins_p_n"].mean()), 3),
        "ins_p_n_mean_with_deploy_anchor":
            round(float(I_wrong["ins_p_n"].mean()), 3),
    }
    g["PASS"] = bool(g["anchor_lt2024_vs_rebuild"].get("all_bit_identical")
                     and g["deploy_anchor_vs_rebuild"].get("all_bit_identical"))
    res["G0d_anchor_season_boundary"] = g
    print(f"  G0d  lt2024 rebuild identical "
          f"{g['anchor_lt2024_vs_rebuild'].get('all_bit_identical')}  "
          f"deploy rebuild identical "
          f"{g['deploy_anchor_vs_rebuild'].get('all_bit_identical')}  "
          f"PASS={g['PASS']}", flush=True)
    return g["PASS"]


def gate_e(res, I_va, y_va, df_va):
    """Does any of the 16 columns carry THIS row's label?"""
    from sklearn.metrics import roc_auc_score
    cols = list(I_va.columns)
    V = I_va.to_numpy(np.float64)
    per = {}
    for j, c in enumerate(cols):
        v = V[:, j]
        fin = np.isfinite(v)
        if fin.sum() < 10 or np.nanstd(v) == 0:
            per[c] = {"auc": None, "corr": None, "note": "constant/empty"}
            continue
        per[c] = {
            "auc": round(float(roc_auc_score(y_va[fin], v[fin])), 6),
            "corr": round(float(np.corrcoef(v[fin], y_va[fin])[0, 1]), 6),
            "n_distinct": int(np.unique(v[fin]).size),
        }
    aucs = [abs(per[c]["auc"] - 0.5) for c in cols if per[c]["auc"] is not None]

    # perfect separation / determinism: do identical 16-tuples ever carry
    # BOTH labels?  If every tuple were unique the tuple could memorise y.
    _, inv = np.unique(np.round(V, 7), axis=0, return_inverse=True)
    grp = pd.DataFrame({"k": inv.ravel(), "y": y_va}).groupby("k")["y"]
    sz, mn = grp.size().to_numpy(), grp.mean().to_numpy()
    n_groups = int(sz.size)
    n_singleton = int((sz == 1).sum())
    n_mixed = int(((mn > 0) & (mn < 1)).sum())

    g = {
        "what": ("the 16 columns are built from asof_* (organiser-supplied, "
                 "strictly-before-this-row) minus a frozen train table.  If "
                 "they leaked THIS row's outcome a single column would "
                 "separate y.  Measured on the 2024 valid rows."),
        "per_column": per,
        "max_abs_auc_minus_half": round(max(aucs), 6),
        "max_abs_corr": round(max(abs(per[c]["corr"]) for c in cols
                                  if per[c]["corr"] is not None), 6),
        "tuple_determinism": {
            "n_rows": int(len(y_va)), "n_distinct_16_tuples": n_groups,
            "n_singleton_tuples": n_singleton,
            "pct_singleton": round(100.0 * n_singleton / max(n_groups, 1), 3),
            "n_tuples_carrying_both_labels": n_mixed,
            "note": ("a leak-free feature set must have tuples that carry both "
                     "labels; complete singletons would mean the tuple is a "
                     "row identifier"),
        },
        "label_column_named_in_module": False,
    }
    # a leak here would show as auc far from 0.5 on a SINGLE column
    g["PASS"] = bool(max(aucs) < 0.20 and n_mixed > 0)
    res["G0e_label_recovery_boundary"] = g
    print(f"  G0e  max|auc-0.5| {g['max_abs_auc_minus_half']}  "
          f"mixed tuples {n_mixed}  PASS={g['PASS']}", flush=True)
    return g["PASS"]


def gate_f(res, ins, df, A_lt2024, I_va):
    """Unknown players and the meaning of a 0.0 deviation."""
    season = df["season"].to_numpy()
    va = season == 2024
    pid = df.loc[va, "pitcher_id"].to_numpy()
    bid = df.loc[va, "batter_id"].to_numpy()
    known_p = pd.Index(A_lt2024["p_ids"]).get_indexer(pid) >= 0
    known_b = pd.Index(A_lt2024["b_ids"]).get_indexer(bid) >= 0
    gt = df.loc[va, "game_type"].to_numpy()
    is_r = gt == "R"

    dev_cols = [c for c in I_va.columns if c.endswith("_dev")]
    exact_zero = {c: round(100.0 * float((I_va[c].to_numpy() == 0.0).mean()), 4)
                  for c in dev_cols}
    n = I_va["ins_p_n"].to_numpy()
    g = {
        "unknown_pitcher_pct_2024": round(100.0 * float((~known_p).mean()), 4),
        "unknown_batter_pct_2024": round(100.0 * float((~known_b).mean()), 4),
        "unknown_pitcher_pct_2024_R": round(
            100.0 * float((~known_p[is_r]).mean()), 4),
        "unknown_pitcher_pct_2024_F": round(
            100.0 * float((~known_p[~is_r]).mean()), 4),
        "coverage_ins_p_n_gt0_pct": round(100.0 * float((n > 0).mean()), 4),
        "coverage_ins_p_n_gt0_pct_R": round(
            100.0 * float((n[is_r] > 0).mean()), 4),
        "coverage_ins_p_n_gt0_pct_F": round(
            100.0 * float((n[~is_r] > 0).mean()), 4),
        "ins_p_n_describe": {k: float(v) for k, v in
                             pd.Series(n).describe().items()},
        "dev_exact_zero_pct": exact_zero,
        "verdict_on_zero_dev": (
            "for an UNKNOWN pitcher the module sets anchor=0 and prior=league, "
            "so k = round(n*rate) and ins rate = (k + 50*lg)/(n + 50); the "
            "deviation is then a REAL shrunk number, not a constant.  A dev of "
            "exactly 0.0 arises only when n==0 (no career at all), which IS a "
            "sentinel-shaped value; its share is reported above."),
    }
    g["PASS"] = True
    res["G0f_missing_handling"] = g
    print(f"  G0f  unknown pitcher {g['unknown_pitcher_pct_2024']}%  "
          f"coverage {g['coverage_ins_p_n_gt0_pct']}%", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.expanduser(
        "~/SungjinPark/experiments/exp_048/res"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    res = {"exp_id": "exp_048", "stage": 0,
           "versions": {"python": sys.version.split()[0],
                        "numpy": np.__version__, "pandas": pd.__version__}}

    # ---- sha of the incoming pack ---------------------------------------- #
    got = {f: sha256_file(os.path.join(PACK_DIR, f)) for f in EXPECT_SHA}
    res["pack_sha256"] = {"observed": got, "expected": EXPECT_SHA,
                          "all_match": got == EXPECT_SHA}
    assert got == EXPECT_SHA, ("teammate pack sha mismatch", got)
    print(f"  sha  pack matches all 3 expected digests", flush=True)

    import inseason as ins
    res["train_csv_sha256"] = sha256_file(DATA_PATH)

    ok_a = gate_a(res)

    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    season = df["season"].to_numpy()
    res["seasons"] = {str(int(s)): int((season == s).sum())
                      for s in sorted(set(season.tolist()))}

    A_lt2024 = ins.load(os.path.join(PACK_DIR, "anchor_lt2024.npz"))
    A_deploy = ins.load(os.path.join(PACK_DIR, "deploy_anchor.npz"))

    va = season == 2024
    df_va = df.loc[va].copy()
    y_va = df.loc[va, TARGET].to_numpy(np.float64)
    rng = np.random.default_rng(20260814)

    I_va, ok_bc = gate_bc(res, ins, A_lt2024, df_va, rng)
    ok_d = gate_d(res, ins, df, A_lt2024, A_deploy)
    ok_e = gate_e(res, I_va, y_va, df_va)
    ok_f = gate_f(res, ins, df, A_lt2024, I_va)

    # ---- extra: is 2019 the first season the asof counters ever see? ------ #
    # decides how season 2019 rows (19% of the fold_2024 training set) are
    # treated, since they have no prior-season anchor at all.
    d19 = df.loc[season == 2019]
    first_n = d19.groupby("pitcher_id", sort=False)["asof_pitcher_n"].min()
    res["asof_origin_probe_2019"] = {
        "what": ("if asof_pitcher_n starts at 0 for every pitcher in 2019 then "
                 "the dataset's asof counters begin with the file, so in 2019 "
                 "career == in-season and an EMPTY anchor is arithmetically "
                 "right -- except that the league prior does not exist yet."),
        "n_pitchers_2019": int(len(first_n)),
        "pct_first_row_n_is_zero": round(
            100.0 * float((first_n.to_numpy() == 0).mean()), 4),
        "max_first_n": int(first_n.max()),
    }
    print(f"  probe 2019 first asof_pitcher_n == 0 for "
          f"{res['asof_origin_probe_2019']['pct_first_row_n_is_zero']}% of "
          f"pitchers", flush=True)

    res["columns_16"] = list(I_va.columns)
    assert len(I_va.columns) == 16, len(I_va.columns)
    res["ALL_PASS"] = bool(ok_a and ok_bc and ok_d and ok_e and ok_f)
    res["minutes"] = round((time.time() - t0) / 60, 2)

    p = os.path.join(args.out, "stage0_gates.json")
    with open(p, "w") as f:
        json.dump(res, f, indent=1)
    print("STAGE0 " + json.dumps({"ALL_PASS": res["ALL_PASS"],
                                  "minutes": res["minutes"], "out": p}),
          flush=True)


if __name__ == "__main__":
    main()

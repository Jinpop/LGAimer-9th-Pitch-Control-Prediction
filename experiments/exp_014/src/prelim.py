"""exp_014 preliminary check — run BEFORE any training.

Three questions the config asks (config.yaml: preliminary_check):
  1. Do fail_shape / residual actually vary across pitchers, or are they
     near-constant (in which case the whole idea is dead on arrival)?
  2. What IS `residual = 1 - (success + reverse + middle)`?  The EDA found the
     three-category identity sums to ~0.90; the remaining ~0.10 must be
     something.  Trace it against ball_rate / strike_rate and check whether
     ball + strike has its own identity.
  3. Does fail_shape carry signal — i.e. is the empirical success rate
     monotone (or at least non-flat) across fail_shape bins?

No model is trained here.  Output is printed and dumped to prelim.json.
"""

import json
import os

import numpy as np
import pandas as pd

DATA = os.environ.get(
    "EXP014_DATA", os.path.expanduser("~/SungjinPark/data/train.csv")
)
OUT = os.environ.get("EXP014_OUT", os.path.dirname(os.path.abspath(__file__)))

TARGET = "control_success"
S = "asof_pitcher_success_rate"
R = "asof_pitcher_reverse_rate"
M = "asof_pitcher_middle_rate"
B = "asof_pitcher_ball_rate"
K = "asof_pitcher_strike_rate"


def q(x, name):
    x = pd.Series(x).dropna()
    if len(x) == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(len(x)),
        "mean": round(float(x.mean()), 6),
        "std": round(float(x.std()), 6),
        "min": round(float(x.min()), 6),
        "p01": round(float(x.quantile(0.01)), 6),
        "p05": round(float(x.quantile(0.05)), 6),
        "p25": round(float(x.quantile(0.25)), 6),
        "p50": round(float(x.quantile(0.50)), 6),
        "p75": round(float(x.quantile(0.75)), 6),
        "p95": round(float(x.quantile(0.95)), 6),
        "p99": round(float(x.quantile(0.99)), 6),
        "max": round(float(x.max()), 6),
        "iqr": round(float(x.quantile(0.75) - x.quantile(0.25)), 6),
    }


def main():
    out = {}
    print(f"loading {DATA}", flush=True)
    df = pd.read_csv(DATA, encoding="utf-8-sig")
    print(f"loaded {df.shape}", flush=True)

    # ---------------------------------------------------------------- #
    # career-end (last row per pitcher) view = "per pitcher" question
    # ---------------------------------------------------------------- #
    last = df.groupby("pitcher_id", sort=False).tail(1).copy()
    last["career_n"] = last["asof_pitcher_n"]
    big = last[last["career_n"] >= 500].copy()
    print(f"pitchers total={len(last)}, with >=500 career pitches={len(big)}",
          flush=True)

    for d in (last, big, df):
        d["fail_shape"] = d[R] / (d[R] + d[M])
        d["residual"] = 1.0 - (d[S] + d[R] + d[M])
        d["triple_sum"] = d[S] + d[R] + d[M]
        d["bk_sum"] = d[B] + d[K]
        d["one_minus_bk"] = 1.0 - d["bk_sum"]

    # ---- Q1: dispersion across pitchers ------------------------------ #
    q1 = {
        "n_pitchers": int(len(last)),
        "n_pitchers_ge500": int(len(big)),
        "career_end_all": {
            k: q(last[k], k)
            for k in ["fail_shape", "residual", "triple_sum", S, R, M, B, K,
                      "bk_sum"]
        },
        "career_end_ge500": {
            k: q(big[k], k)
            for k in ["fail_shape", "residual", "triple_sum", S, R, M, B, K,
                      "bk_sum"]
        },
        "row_level_all_rows": {
            k: q(df[k], k) for k in ["fail_shape", "residual", "triple_sum"]
        },
    }
    # how much of fail_shape variance is BETWEEN pitchers vs within?
    rowfs = df[["pitcher_id", "fail_shape", "residual"]].dropna()
    for col in ("fail_shape", "residual"):
        gm = rowfs.groupby("pitcher_id")[col].mean()
        gn = rowfs.groupby("pitcher_id")[col].size()
        grand = rowfs[col].mean()
        between = float((gn * (gm - grand) ** 2).sum() / gn.sum())
        total = float(rowfs[col].var(ddof=0))
        q1[f"{col}_variance_decomposition"] = {
            "total_var_rowlevel": round(total, 8),
            "between_pitcher_var": round(between, 8),
            "between_share_pct": round(100.0 * between / total, 2) if total else None,
        }
    # comparison anchor: is fail_shape spread big relative to success_rate spread?
    q1["spread_anchor"] = {
        "note": "CV = std/mean at career end (>=500 pitches). If fail_shape CV is "
                "comparable to success_rate CV, the feature really does separate "
                "pitchers; if it is an order of magnitude smaller it is a constant.",
        "cv_success_rate": round(float(big[S].std() / big[S].mean()), 4),
        "cv_fail_shape": round(float(big["fail_shape"].std() / big["fail_shape"].mean()), 4),
        "cv_residual": round(float(big["residual"].std() / abs(big["residual"].mean())), 4),
    }
    out["Q1_dispersion"] = q1

    # ---- Q2: what is residual? --------------------------------------- #
    cc = big[[S, R, M, B, K, "residual", "triple_sum", "bk_sum", "one_minus_bk",
              "career_n"]].dropna()
    corr = cc.corr(method="pearson")
    q2 = {
        "note": "career-end values, pitchers with >=500 pitches",
        "n": int(len(cc)),
        "corr_matrix_vs_residual": {
            k: round(float(corr.loc["residual", k]), 4)
            for k in [S, R, M, B, K, "bk_sum", "one_minus_bk", "career_n"]
        },
        "bk_sum_stats": q(cc["bk_sum"], "ball+strike"),
        "triple_sum_stats": q(cc["triple_sum"], "success+reverse+middle"),
        "corr_residual_vs_one_minus_bk": round(
            float(np.corrcoef(cc["residual"], cc["one_minus_bk"])[0, 1]), 4),
        "mean_abs_diff_residual_minus_one_minus_bk": round(
            float(np.abs(cc["residual"] - cc["one_minus_bk"]).mean()), 6),
    }
    # OLS of residual on ball/strike (career level)
    Xd = np.column_stack([np.ones(len(cc)), cc[B].to_numpy(), cc[K].to_numpy()])
    beta, *_ = np.linalg.lstsq(Xd, cc["residual"].to_numpy(), rcond=None)
    pred = Xd @ beta
    ss_res = float(((cc["residual"] - pred) ** 2).sum())
    ss_tot = float(((cc["residual"] - cc["residual"].mean()) ** 2).sum())
    q2["ols_residual_on_ball_strike"] = {
        "intercept": round(float(beta[0]), 5),
        "coef_ball": round(float(beta[1]), 5),
        "coef_strike": round(float(beta[2]), 5),
        "r2": round(1.0 - ss_res / ss_tot, 4),
    }
    # also: does residual look like a fixed constant + noise?
    q2["residual_is_constant_test"] = {
        "std_over_mean": round(float(cc["residual"].std() / cc["residual"].mean()), 4),
        "p95_minus_p05": round(
            float(cc["residual"].quantile(0.95) - cc["residual"].quantile(0.05)), 6),
    }
    # exact-identity hunt: does S+R+M+something = 1?
    for name, expr in [
        ("S+R+M+B", cc[S] + cc[R] + cc[M] + cc[B]),
        ("S+R+M+K", cc[S] + cc[R] + cc[M] + cc[K]),
        ("S+B", cc[S] + cc[B]),
        ("S+K", cc[S] + cc[K]),
        ("R+M+B", cc[R] + cc[M] + cc[B]),
        ("R+M+K", cc[R] + cc[M] + cc[K]),
        ("B+K+M", cc[B] + cc[K] + cc[M]),
        ("B+K+R", cc[B] + cc[K] + cc[R]),
    ]:
        q2.setdefault("identity_hunt", {})[name] = {
            "mean": round(float(expr.mean()), 5),
            "std": round(float(expr.std()), 5),
        }
    out["Q2_residual_identity"] = q2

    # ---- Q3: does fail_shape carry signal? --------------------------- #
    d = df[["fail_shape", "residual", TARGET, "asof_pitcher_n", "season",
            S, M]].copy()
    ok = d["fail_shape"].notna()
    q3 = {
        "n_rows_with_fail_shape": int(ok.sum()),
        "n_rows_nan_fail_shape": int((~ok).sum()),
        "corr_fail_shape_target": round(
            float(np.corrcoef(d.loc[ok, "fail_shape"], d.loc[ok, TARGET])[0, 1]), 5),
        "corr_residual_target": round(
            float(np.corrcoef(d.loc[d["residual"].notna(), "residual"],
                              d.loc[d["residual"].notna(), TARGET])[0, 1]), 5),
        "corr_success_rate_target_ANCHOR": round(
            float(np.corrcoef(d.loc[d[S].notna(), S],
                              d.loc[d[S].notna(), TARGET])[0, 1]), 5),
    }
    # decile table, overall
    dd = d[ok].copy()
    dd["bin"] = pd.qcut(dd["fail_shape"], 10, labels=False, duplicates="drop")
    tab = dd.groupby("bin").agg(
        n=(TARGET, "size"), success=(TARGET, "mean"),
        fs_lo=("fail_shape", "min"), fs_hi=("fail_shape", "max"),
        mean_success_rate=(S, "mean"),
    ).reset_index()
    q3["fail_shape_decile_table"] = [
        {"bin": int(r.bin), "n": int(r.n), "fs_range": [round(r.fs_lo, 4), round(r.fs_hi, 4)],
         "empirical_success": round(float(r.success), 5),
         "mean_asof_success_rate": round(float(r.mean_success_rate), 5)}
        for r in tab.itertuples()
    ]
    # CONTROLLED: within narrow success_rate strata, does fail_shape still move
    # the empirical success rate?  This is the real question — fail_shape is
    # useless if it only proxies success_rate.
    dd["s_stratum"] = pd.qcut(dd[S], 5, labels=False, duplicates="drop")
    dd["fs_q"] = dd.groupby("s_stratum")["fail_shape"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    ct = dd.groupby(["s_stratum", "fs_q"]).agg(
        n=(TARGET, "size"), success=(TARGET, "mean")).reset_index()
    q3["controlled_table_success_stratum_x_failshape_quintile"] = [
        {"s_stratum": int(r.s_stratum), "fs_q": int(r.fs_q), "n": int(r.n),
         "empirical_success": round(float(r.success), 5)}
        for r in ct.itertuples()
    ]
    spread = ct.groupby("s_stratum").apply(
        lambda g: g.loc[g.fs_q == g.fs_q.max(), "success"].iloc[0]
        - g.loc[g.fs_q == g.fs_q.min(), "success"].iloc[0])
    q3["controlled_spread_top_minus_bottom_fs_quintile_pp"] = {
        str(int(k)): round(float(v) * 100, 3) for k, v in spread.items()
    }
    # same, for residual
    dr = d[d["residual"].notna()].copy()
    dr["s_stratum"] = pd.qcut(dr[S], 5, labels=False, duplicates="drop")
    dr["r_q"] = dr.groupby("s_stratum")["residual"].transform(
        lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"))
    ctr = dr.groupby(["s_stratum", "r_q"]).agg(
        n=(TARGET, "size"), success=(TARGET, "mean")).reset_index()
    spread_r = ctr.groupby("s_stratum").apply(
        lambda g: g.loc[g.r_q == g.r_q.max(), "success"].iloc[0]
        - g.loc[g.r_q == g.r_q.min(), "success"].iloc[0])
    q3["controlled_spread_residual_quintile_pp"] = {
        str(int(k)): round(float(v) * 100, 3) for k, v in spread_r.items()
    }
    # by asof_pitcher_n bucket — ratios are noise at low n
    bins = [-1, 0, 100, 500, 2000, 10000, 10**9]
    labels = ["n=0", "1-100", "101-500", "501-2000", "2001-10000", ">10000"]
    dd["nb"] = pd.cut(dd["asof_pitcher_n"], bins=bins, labels=labels)
    q3["fail_shape_std_by_n_bucket"] = {
        str(k): {"n": int(v["size"]), "std": round(float(v["std"]), 5),
                 "mean": round(float(v["mean"]), 5)}
        for k, v in dd.groupby("nb", observed=False)["fail_shape"]
        .agg(["size", "std", "mean"]).iterrows()
    }
    out["Q3_signal"] = q3

    # ---- valid-2024 row counts per n-bucket (for the later breakdown) -- #
    v = df[df["season"] == 2024]
    vb = pd.cut(v["asof_pitcher_n"], bins=bins, labels=labels)
    out["fold2024_rows_per_n_bucket"] = {
        str(k): int(n) for k, n in vb.value_counts().sort_index().items()
    }

    with open(os.path.join(OUT, "prelim.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

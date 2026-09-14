"""exp_047 — derive src/b3_cat_x.py from exp_046's src/b3_cat_form.py by an
EXPLICIT, AUDITABLE set of textual replacements.

Same discipline as exp_046's patch_b3.py (which derived that file from
exp_030's b3_cat.py): retyping a 500-line pipeline is how silent extra
variables get in.  Each replacement is one registered change, the script
refuses to run if an anchor is missing or ambiguous, and a unified diff is
written next to the output so an auditor reads the change set.

PARENT: experiments/exp_046/src/b3_cat_form.py
        sha256 8749a819d8bfe21d69183ee233a1dbd4d0663d0062b892a0af06e8c5a811c955

Registered changes
------------------
  1. bookkeeping: OUT_DIR / exp_id  exp_046 -> exp_047
  2. import form_decomp2 (which itself sha-gates and imports exp_046's
     form_decomp — the exp_046 appliers are reused verbatim, not retyped)
  3. CLI: --arm, --axes, --k-axes, --traj, --k-traj, --k-dev, --stack-ext
  4. the exp_047 feature block: one EB column per selected axis (+ the batter
     coverage flag when a batter axis is used) and/or the trajectory/deviation
     pair.  With no axes and no --traj the block is a no-op and the frame is
     exp_046 A1's 51 columns EXACTLY -> that is the identity anchor.
  5. stage-2 stacker trained past its policy iteration so the validation curve
     has an interior extremum.  `iter_used` is UNCHANGED (still the policy
     value); only the curve is extended, so fold_2024 still selects nothing.
  6. file stems / stage-1 json name carry the arm tag

⚠️ UNAVOIDABLE CO-VARIABLE, registered in advance (config changed_vs_control):
   changing the column count re-derives every cell and stacker iteration from
   that arm's own fold_2023 stage-1 argmin.  No arm's result can be attributed
   to the columns alone; it is "these columns + the iterations they imply".
"""

import difflib
import hashlib
import os
import subprocess
import sys

SRC = os.path.expanduser("~/SungjinPark/experiments/exp_046/src/b3_cat_form.py")
SRC_SHA = "8749a819d8bfe21d69183ee233a1dbd4d0663d0062b892a0af06e8c5a811c955"
DST = os.path.expanduser("~/SungjinPark/experiments/exp_047/src/b3_cat_x.py")

X_BLOCK = '''    # ---------------------------------------------------------------- exp_047
    # F2 axes (one EB-shrunk in-season deviation column each) and/or F3's
    # trajectory/deviation split of the success axis.  Every table below is
    # built ONLY from train rows of seasons <= S-1 and is applied ROW-WISE;
    # src/form_decomp2.py carries the rule 2-4 argument and src/ast_gate.py
    # proves mechanically (from the parsed syntax tree, not a regex) that the
    # applier functions contain no groupby/sort/shift/cumsum/factorize/unique.
    res_x = {"arm": ARM, "axes": list(AXES_SEL), "k_axes": dict(K_AXES),
             "traj": bool(USE_TRAJ), "k_traj": K_TRAJ, "k_dev": K_DEV}
    x_cols = []
    if AXES_SEL or USE_TRAJ:
        assert USE_FORM, "every exp_047 arm is a superset of exp_046 A1"
        _ssn2 = df["season"].to_numpy()
        _ent_code = {}
        for _e in ("pitcher_id", "batter_id"):
            _c2, _u2 = pd.factorize(df[_e], sort=True)
            _ent_code[_e] = (_c2.astype(np.int64), len(_u2))
        _ent_has = {}
        for _tag in AXES_SEL:
            _sp = f2.AXES[_tag]
            _ec, _ne = _ent_code[_sp["entity"]]
            _nc = df[_sp["n"]].to_numpy(np.float64)
            _rc = df[_sp["rate"]].to_numpy(np.float64)
            _tb = f2.build_pre_boundary_tables(_ssn2, _ec, _nc, _rc, _ne)
            _z = (np.zeros(_ne), np.zeros(_ne))
            _col = np.full(len(df), np.nan)
            _hcol = np.zeros(len(df), dtype=np.int8)
            for _s in np.sort(np.unique(_ssn2)):
                _m = _ssn2 == _s
                _t = _tb.get(int(_s) - 1, _z)
                _o = f2.apply_decomposition(_nc[_m], _rc[_m], _ec[_m], _t[0],
                                            _t[1], int(K_AXES[_tag]))
                _col[_m] = _o["d_form_eb"]
                _hcol[_m] = _o["has_form"]
                assert not ((_o["S_season"] < -1e-9)
                            | (_o["S_season"] > _o["n_season"] + 1e-9)
                            | (_o["n_season"] < -1e-9)).any(), (
                    f"G1 broke for axis {_tag} at season {_s}")
            df[f2.axis_col(_tag)] = _col
            x_cols.append(f2.axis_col(_tag))
            _ent_has[_sp["entity"]] = _hcol
        # the pitcher entity already has exp_046's `has_form`; a batter axis
        # needs its own defined-ness flag (a NaN must never be back-filled)
        if "batter_id" in _ent_has:
            df["has_form_bat"] = _ent_has["batter_id"]
            x_cols.append("has_form_bat")
    if USE_TRAJ:
        _pc2, _nu2 = _ent_code["pitcher_id"]
        _tabs46 = fd.build_boundary_tables(df, _pc2, _nu2)
        _z46 = (np.zeros(_nu2), np.zeros(_nu2))
        _sc = {c: np.full(len(df), np.nan) for c in
               ("n_end", "S_end", "n_season", "S_season", "d_form")}
        _fn2 = df[fd.N_COL].to_numpy(np.float64)
        _fr2 = df[fd.RATE_COL].to_numpy(np.float64)
        for _s in np.sort(np.unique(_ssn2)):
            _m = _ssn2 == _s
            _t = _tabs46.get(int(_s) - 1, _z46)
            _o = fd.apply_decomposition(_fn2[_m], _fr2[_m], _pc2[_m], _t[0],
                                        _t[1], K_FORM)
            for _c in _sc:
                _sc[_c][_m] = _o[_c]
        _ytr = df[TARGET].to_numpy(np.float64)
        _tt = f2.build_traj_table(_ssn2, _pc2, _nu2, _sc["n_end"], _sc["S_end"],
                                  _sc["n_season"], _sc["S_season"], _ytr)
        _zt = (np.full(_nu2, np.nan), np.zeros(_nu2))
        _tj = np.full(len(df), np.nan)
        _dv = np.full(len(df), np.nan)
        _ht = np.zeros(len(df), dtype=np.int8)
        for _s in np.sort(np.unique(_ssn2)):
            _m = _ssn2 == _s
            _t = _tt.get(int(_s) - 1, _zt)
            _o = f2.apply_traj(_pc2[_m], _t[0], _t[1], _sc["d_form"][_m],
                               _sc["n_season"][_m], K_TRAJ, K_DEV)
            _tj[_m] = _o["traj_eb"]
            _dv[_m] = _o["dev_eb"]
            _ht[_m] = _o["has_traj"]
        df["traj_eb"] = _tj
        df["dev_eb"] = _dv
        df["has_traj"] = _ht
        x_cols += ["traj_eb", "dev_eb", "has_traj"]
    if x_cols:
        base_feats = base_feats + x_cols
        res_x["new_columns"] = list(x_cols)
        res_x["nan_pct"] = {c: round(100.0 * float(df[c].isna().mean()), 4)
                            for c in x_cols}
    assert len(base_feats) == (51 if USE_FORM else 47) + len(x_cols), (
        len(base_feats), len(x_cols))
'''

STACK_OLD = '''    k = int(stack_round)
    t0 = time.time()
    pr, params, note = fit_catboost(C5, SEED, Fs_tr, y_all[tr_full], Fs_va,
                                    feats, cat_native, k)
    fit_min = (time.time() - t0) / 60
    grid = ref.curve_grid(k)
'''
STACK_NEW = '''    k = int(stack_round)
    # exp_047 registered change 5: exp_046 A1 reported
    # `curve_argmin_is_interior: false` on ALL THREE fold_2024 seeds, i.e. the
    # stacker curve was still falling at the policy iteration and +45.17 was a
    # LOWER bound.  Stage 2 now trains the stacker STACK_EXT times past the
    # policy iteration so the argmin can be located.  `iter_used` is unchanged
    # -- it is still the fold_2023-derived policy value -- so fold_2024's own
    # curve still selects nothing.  The policy point is forced into the grid so
    # it is evaluated at exactly k rather than at a neighbouring grid node.
    k_fit = k if STAGE == 1 else int(k * STACK_EXT)
    t0 = time.time()
    pr, params, note = fit_catboost(C5, SEED, Fs_tr, y_all[tr_full], Fs_va,
                                    feats, cat_native, k_fit)
    fit_min = (time.time() - t0) / 60
    grid = sorted(set(ref.curve_grid(k_fit) + [k]))
'''

RES_OLD = '''        "fit_min": round(fit_min, 3),
        "sec_per_round": round(60.0 * fit_min / k, 5),
        "margin_const": float(const),
        "params": {kk: (vv if isinstance(vv, (int, float, str, bool, type(None)))
                        else str(vv)) for kk, vv in params.items()},
'''
RES_NEW = '''        "fit_min": round(fit_min, 3),
        "sec_per_round": round(60.0 * fit_min / k_fit, 5),
        "stack_fit_rounds": int(k_fit), "stack_ext": float(STACK_EXT),
        "curve_argmin_is_interior_extended": bool(a_min < k_fit),
        "brier_at_extended_argmin_ORACLE": float(f"{a_val:.10g}"),
        "policy_cost_vs_extended_min_pts": float(
            f"{(cv[k_use] - a_val) * -100000.0 / REF_BRIER:.6g}"),
        "margin_const": float(const),
        "params": {kk: (vv if isinstance(vv, (int, float, str, bool, type(None)))
                        else str(vv)) for kk, vv in params.items()},
'''

REPLACEMENTS = [
    # 1. bookkeeping ------------------------------------------------------- #
    ('OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_046")',
     'OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_047")'),
    ('"exp_id": "exp_046", "unit": V,',
     '"exp_id": "exp_047", "unit": V,'),

    # 2. import ------------------------------------------------------------ #
    ('import form_decomp as fd                                            # noqa: E402',
     'import form_decomp as fd                                            # noqa: E402\n'
     'import form_decomp2 as f2                                           # noqa: E402'),

    # 3. CLI --------------------------------------------------------------- #
    ('    ap.add_argument("--k-form", type=int, default=100,\n'
     '                    help="EB shrink k for d_form_eb; chosen on blocks "\n'
     '                         "2022+2023 in stage1_diag.json, never on fold_2024")',
     '    ap.add_argument("--k-form", type=int, default=100,\n'
     '                    help="EB shrink k for d_form_eb; chosen on blocks "\n'
     '                         "2022+2023 in stage1_diag.json, never on fold_2024")\n'
     '    ap.add_argument("--arm", default="F1", help="file-stem tag")\n'
     '    ap.add_argument("--axes", default="",\n'
     '                    help="comma list of exp_047 axis tags from "\n'
     '                         "stage1_selection.json (empty = exp_046 A1)")\n'
     '    ap.add_argument("--k-axes", default="",\n'
     '                    help="comma list tag=k, block-selected in stage 1")\n'
     '    ap.add_argument("--traj", action="store_true",\n'
     '                    help="add the trajectory/deviation split (axis B)")\n'
     '    ap.add_argument("--k-traj", type=int, default=50)\n'
     '    ap.add_argument("--k-dev", type=int, default=25)\n'
     '    ap.add_argument("--stack-ext", type=float, default=6.0,\n'
     '                    help="stage 2 trains the stacker this many times past "\n'
     '                         "its policy iteration; iter_used is unaffected")'),
    ('    USE_FORM, K_FORM = args.form, args.k_form',
     '    USE_FORM, K_FORM = args.form, args.k_form\n'
     '    ARM = args.arm\n'
     '    AXES_SEL = [t for t in args.axes.split(",") if t]\n'
     '    K_AXES = {p.split("=")[0]: int(p.split("=")[1])\n'
     '              for p in args.k_axes.split(",") if p}\n'
     '    assert set(K_AXES) == set(AXES_SEL), (K_AXES, AXES_SEL)\n'
     '    assert all(t in f2.AXES for t in AXES_SEL), AXES_SEL\n'
     '    USE_TRAJ, K_TRAJ, K_DEV = args.traj, args.k_traj, args.k_dev\n'
     '    STACK_EXT = args.stack_ext'),

    # 4. the exp_047 feature block ----------------------------------------- #
    ('    assert len(base_feats) == (51 if USE_FORM else 47), len(base_feats)\n',
     X_BLOCK),

    # 5. stacker trained past the policy iteration ------------------------- #
    (STACK_OLD, STACK_NEW),
    (RES_OLD, RES_NEW),

    # 6. file naming ------------------------------------------------------- #
    ('    res["form"] = res_form\n'
     '    stem = (f"C5B3form_u{V}_seed{SEED}_st{STAGE}" if USE_FORM\n'
     '            else f"C5B3_u{V}_seed{SEED}_st{STAGE}")',
     '    res["form"] = res_form\n'
     '    res["x"] = res_x\n'
     '    stem = f"{ARM}_u{V}_seed{SEED}_st{STAGE}"'),
    ('        name = "stage1_A1.json" if USE_FORM else "stage1_noform.json"',
     '        name = f"stage1_{ARM}.json"'),
]


def sha256_file(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    got = sha256_file(SRC)
    assert got == SRC_SHA, ("parent b3_cat_form.py changed", got, SRC_SHA)
    src = open(SRC).read()
    out = src
    for i, (a, b) in enumerate(REPLACEMENTS, 1):
        cnt = out.count(a)
        if cnt != 1:
            sys.exit(f"replacement {i} matched {cnt} times, expected 1:\n{a[:200]}")
        out = out.replace(a, b)
    with open(DST, "w") as f:
        f.write(out)
    diff = "".join(difflib.unified_diff(src.splitlines(True),
                                        out.splitlines(True),
                                        "exp_046/src/b3_cat_form.py",
                                        "exp_047/src/b3_cat_x.py"))
    with open(DST + ".diff", "w") as f:
        f.write(diff)
    add = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    rem = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"wrote {DST}\n  +{add} / -{rem} lines, {len(REPLACEMENTS)} registered changes")
    subprocess.run([sys.executable, "-c",
                    f"import ast;ast.parse(open({DST!r}).read())"], check=True)
    print("syntax OK, parent sha", got)


if __name__ == "__main__":
    main()

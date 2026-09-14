"""exp_046 — derive src/b3_cat_form.py from exp_030's src/b3_cat.py by an
EXPLICIT, AUDITABLE set of textual replacements.

Why a patcher instead of a hand-written copy: the whole point of arm A1 is
that the ONLY difference from the exp_030 C5_B3 control is four extra feature
columns.  Retyping a 500-line file is how silent extra variables get in.  Each
replacement below is one registered change; the script refuses to run if any
anchor text is missing or ambiguous, and it writes a unified diff next to the
output so an auditor reads the change set rather than the whole file.

Registered changes (matches config.yaml changed_vs_control.A1):
  1. output directory / exp_id  exp_030 -> exp_046   (bookkeeping only)
  2. --form / --k-form CLI switch.  --no-form must reproduce exp_030 BYTE FOR
     BYTE; that is the identity anchor this experiment leans on.
  3. the four in-season-form columns are appended to base_feats (47 -> 51)
  4. the stacker column-count assert follows base_feats (51 -> 55)
  5. stage-1 result filename stage1.json -> stage1_A1.json so it cannot
     collide with the stage-1 DIAGNOSTIC json
  6. prediction/curve/res file stems get a `form` marker when the columns are on
"""

import difflib
import os
import subprocess
import sys

SRC = os.path.expanduser("~/SungjinPark/experiments/exp_030/src/b3_cat.py")
DST = os.path.expanduser("~/SungjinPark/experiments/exp_046/src/b3_cat_form.py")

FORM_BLOCK = '''
    # ---------------------------------------------------------------- exp_046
    # A1: in-season form decomposition.  See src/form_decomp.py for the rule
    # 2-4 boundary; the short version is that the frozen per-pitcher table for
    # a row of season S is built ONLY from rows of seasons <= S-1, so every
    # column below is strictly as-of and contains no information from the
    # validation season.  Row-wise application never touches another row.
    res_form = {"enabled": bool(USE_FORM), "k_form": int(K_FORM)}
    if USE_FORM:
        _pc, _uq = pd.factorize(df["pitcher_id"], sort=True)
        _pc = _pc.astype(np.int64)
        _tabs = fd.build_boundary_tables(df, _pc, len(_uq))
        _zero = (np.zeros(len(_uq)), np.zeros(len(_uq)))
        _ssn = df["season"].to_numpy()
        _fn = df[fd.N_COL].to_numpy(dtype=np.float64)
        _fr = df[fd.RATE_COL].to_numpy(dtype=np.float64)
        _buf = {c: np.full(len(df), np.nan) for c in fd.NEW_COLS}
        for _s in np.sort(np.unique(_ssn)):
            _m = _ssn == _s
            _t = _tabs.get(int(_s) - 1, _zero)
            _o = fd.apply_decomposition(_fn[_m], _fr[_m], _pc[_m],
                                        _t[0], _t[1], K_FORM)
            for _c in fd.NEW_COLS:
                _buf[_c][_m] = _o[_c]
            # G1 re-checked at run time on the rows this unit will actually
            # use -- a broken decomposition must stop the fit, not be reported
            # after it.
            assert not ((_o["S_season"] < -1e-9)
                        | (_o["S_season"] > _o["n_season"] + 1e-9)
                        | (_o["n_season"] < -1e-9)).any(), f"G1 broke at {_s}"
        for _c in fd.NEW_COLS:
            df[_c] = _buf[_c]
        base_feats = base_feats + list(fd.NEW_COLS)
        res_form["new_columns"] = list(fd.NEW_COLS)
        res_form["nan_pct"] = {c: round(100.0 * float(np.isnan(_buf[c]).mean()), 4)
                               for c in fd.NEW_COLS}
        res_form["boundaries_built"] = sorted(_tabs.keys())
    assert len(base_feats) == (51 if USE_FORM else 47), len(base_feats)
'''

REPLACEMENTS = [
    # 1. bookkeeping ------------------------------------------------------- #
    ('OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")',
     'OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_046")'),
    ('"exp_id": "exp_030", "unit": V,',
     '"exp_id": "exp_046", "unit": V,'),

    # 2. imports + CLI switch --------------------------------------------- #
    ('import exp016_ref as ref                                            # noqa: E402',
     'import exp016_ref as ref                                            # noqa: E402\n'
     'import form_decomp as fd                                            # noqa: E402'),
    ('    ap.add_argument("--out", default=OUT_DIR)',
     '    ap.add_argument("--out", default=OUT_DIR)\n'
     '    ap.add_argument("--form", dest="form", action="store_true", default=True)\n'
     '    ap.add_argument("--no-form", dest="form", action="store_false",\n'
     '                    help="identity anchor: must reproduce exp_030 byte for byte")\n'
     '    ap.add_argument("--k-form", type=int, default=100,\n'
     '                    help="EB shrink k for d_form_eb; chosen on blocks "\n'
     '                         "2022+2023 in stage1_diag.json, never on fold_2024")'),
    ('    V, SEED, STAGE = args.unit, args.seed, args.stage',
     '    V, SEED, STAGE = args.unit, args.seed, args.stage\n'
     '    USE_FORM, K_FORM = args.form, args.k_form'),

    # 3. the four columns -------------------------------------------------- #
    ('    assert len(base_feats) == 47, len(base_feats)\n',
     '    assert len(base_feats) == 47, len(base_feats)\n' + FORM_BLOCK),

    # 4. stacker width follows base_feats ---------------------------------- #
    ('    assert len(feats) == 51, len(feats)',
     '    assert len(feats) == len(base_feats) + 4, len(feats)'),

    # 5/6. file naming ----------------------------------------------------- #
    ('    stem = f"C5B3_u{V}_seed{SEED}_st{STAGE}"',
     '    res["form"] = res_form\n'
     '    stem = (f"C5B3form_u{V}_seed{SEED}_st{STAGE}" if USE_FORM\n'
     '            else f"C5B3_u{V}_seed{SEED}_st{STAGE}")'),
    ('        with open(os.path.join(out_dir, "stage1.json"), "w") as f:',
     '        name = "stage1_A1.json" if USE_FORM else "stage1_noform.json"\n'
     '        with open(os.path.join(out_dir, name), "w") as f:'),
]


def main():
    src = open(SRC).read()
    out = src
    for i, (a, b) in enumerate(REPLACEMENTS, 1):
        cnt = out.count(a)
        if cnt != 1:
            sys.exit(f"replacement {i} matched {cnt} times, expected 1:\n{a[:120]}")
        out = out.replace(a, b)
    with open(DST, "w") as f:
        f.write(out)
    diff = "".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                        "exp_030/src/b3_cat.py",
                                        "exp_046/src/b3_cat_form.py"))
    with open(DST + ".diff", "w") as f:
        f.write(diff)
    added = sum(1 for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++"))
    removed = sum(1 for l in diff.splitlines() if l.startswith("-") and not l.startswith("---"))
    print(f"wrote {DST}\n  +{added} / -{removed} lines, {len(REPLACEMENTS)} registered changes")
    subprocess.run([sys.executable, "-c", f"import ast;ast.parse(open({DST!r}).read())"],
                   check=True)
    print("syntax OK")


if __name__ == "__main__":
    main()

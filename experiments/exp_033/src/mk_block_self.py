"""Derive exp_031/src/block_self.py from exp_030/src/b3_cat.py by an EXPLICIT,
enumerated set of textual substitutions, then write the unified diff next to it
so the ledger of what changed is machine-produced, not remembered."""
import difflib
import hashlib
import os

SRC = "<PROJECT_ROOT>/experiments/exp_031/src/b3_cat.py"
DST = "<PROJECT_ROOT>/experiments/exp_031/src/block_self.py"
DIFF = "<PROJECT_ROOT>/experiments/exp_031/src/block_self.diff"
B3_SHA = "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7bfa76ca"

src = open(SRC).read()
assert hashlib.sha256(src.encode()).hexdigest() == B3_SHA

DOC_ADD = '''
exp_031 DERIVATION  (this file is NOT exp_030's b3_cat.py; it is that file with
------------------   ONE behavioural change, enumerated below and diffed in
                     block_self.diff, which sits beside it in src/)

  STAGE 3 = "self early stop".  Identical in every line to the STAGE 1 code
  path -- every cell model and the stacker is trained to --cap and reported /
  used at its OWN validation argmin -- but WITHOUT stage 1's
  `assert V == BASE_UNIT and SEED == 42`, so any (unit, seed) may stop on its
  own curve.

  WHY.  exp_030 built the beta supply blocks under STAGE 2, whose iteration
  count is stage-1's fold_2023 argmin scaled by the training-row ratio.  On
  block 2022 that rule produced stacker iter 282 while the block's own curve
  bottoms at 110 (seed 42) / 130 (seed 7); the resulting predictions pay
  2.56e-04 to 3.16e-04 Brier (-102.4 to -126.5 score points) versus their own
  minimum.  beta is fitted ON those block predictions, so a beta fitted on them
  is fitted on a damaged supply.

  IS THIS VALIDATION CONTAMINATION?  No.  The supply blocks are 2022 and 2023.
  fold_2024 -- the only adoption criterion -- is untouched by this file; its
  predictions are exp_030's, reused byte-for-byte.  Letting a block stop on its
  own curve is what exp_024 already did for the LightGBM supply blocks.

  Everything else -- masks, pools, row counts, the C5 spec imported from
  exp_027, the leak asserts, the per-cell early stopping, the OOF
  forward-chaining -- is byte-identical to b3_cat.py, whose sha256 is asserted
  at run time from the copy carried in this same src/ directory.
'''

subs = [
    # 1. docstring: record the derivation
    ('Usage:\n    python b3_cat.py --unit 2023 --seed 42 --stage 1 --cap 1650',
     DOC_ADD + '\nUsage:\n    python block_self.py --unit 2022 --seed 42 --stage 3 --cap 1650'),
    ('    python b3_cat.py --unit 2024 --seed 7  --stage 2 --base-json ../stage1.json\n',
     ''),
    # 2. output directory
    ('OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_030")',
     'OUT_DIR = os.path.expanduser("~/SungjinPark/experiments/exp_031")\n'
     'B3_CAT = "b3_cat.py"          # exp_030\'s file, carried byte-identical\n'
     'B3_CAT_SHA = "0984c57e5af66600f9fca5532a984ddd6d4c22996c9e0105d3e798bc7'
     'bfa76ca"'),
    # 3. stage choices
    ('ap.add_argument("--stage", type=int, required=True, choices=[1, 2])',
     'ap.add_argument("--stage", type=int, required=True, choices=[1, 2, 3])'),
    # 4. provenance assert
    ('    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA\n',
     '    assert sha256_file(os.path.join(SRC_DIR, "exp016_ref.py")) == REF016_SHA\n'
     '    assert sha256_file(os.path.join(SRC_DIR, B3_CAT)) == B3_CAT_SHA, \\\n'
     '        "exp_030 b3_cat.py, from which this file is derived, has drifted"\n'),
    # 5. exp_id
    ('        "exp_id": "exp_030", "unit": V,', '        "exp_id": "exp_031", "unit": V,'),
    # 6. the one behavioural change: stage 3 takes the stage-1 branch
    ('''    if STAGE == 1:
        assert V == BASE_UNIT and SEED == 42, "stage 1 is unit 2023 seed 42"
        cell_round = {c: args.cap for c in FAIL_CELLS}
        stack_round = args.cap
        res["iter_source"] = ("stage 1 — trained to cap, reported at own "
                              "fold_2023 valid argmin")''',
     '''    SELF_STOP = STAGE in (1, 3)
    res["self_early_stop"] = bool(SELF_STOP)
    if SELF_STOP:
        if STAGE == 1:
            assert V == BASE_UNIT and SEED == 42, "stage 1 is unit 2023 seed 42"
        cell_round = {c: args.cap for c in FAIL_CELLS}
        stack_round = args.cap
        res["iter_source"] = (
            f"stage {STAGE} — every cell model and the stacker trained to cap "
            f"{args.cap} and used at its OWN valid-{vy} argmin; no scaling "
            f"policy, no other unit's curve")'''),
    # 7. cell k_use / argmin_at_cap
    ('        k_use = a_min if STAGE == 1 else k\n        cell_argmin[c] = a_min',
     '        k_use = a_min if SELF_STOP else k\n        cell_argmin[c] = a_min'),
    ('            "argmin_at_cap": bool(STAGE == 1 and a_min >= k),\n            "iter_used": int(k_use),\n            "own_mode_at_used"',
     '            "argmin_at_cap": bool(SELF_STOP and a_min >= k),\n            "iter_used": int(k_use),\n            "own_mode_at_used"'),
    # 8. stacker k_use / argmin_at_cap
    ('    k_use = a_min if STAGE == 1 else k\n    p = ref.sigmoid(marg[k_use])',
     '    k_use = a_min if SELF_STOP else k\n    p = ref.sigmoid(marg[k_use])'),
    ('        "argmin_at_cap": bool(STAGE == 1 and a_min >= k),\n        "curve_argmin_is_interior"',
     '        "argmin_at_cap": bool(SELF_STOP and a_min >= k),\n        "curve_argmin_is_interior"'),
    # 9. the cap-hit warning applies to stage 3 too
    ('    if STAGE == 1 and res["any_argmin_at_cap"]:\n        print("!! STAGE 1 INVALID',
     '    if SELF_STOP and res["any_argmin_at_cap"]:\n        print("!! UNIT INVALID'),
]

out = src
for old, new in subs:
    assert out.count(old) == 1, ("substitution is not unique/absent", old[:70],
                                 out.count(old))
    out = out.replace(old, new)

with open(DST, "w") as f:
    f.write(out)
d = "".join(difflib.unified_diff(src.splitlines(True), out.splitlines(True),
                                 "exp_030/src/b3_cat.py",
                                 "exp_031/src/block_self.py"))
with open(DIFF, "w") as f:
    f.write(d)
print(d)
print("n_substitutions", len(subs))
print("block_self.py sha", hashlib.sha256(out.encode()).hexdigest())
os.system(f"cd {os.path.dirname(DST)} && python3 -c 'import ast,sys;ast.parse(open(\"block_self.py\").read())' && echo 'AST OK'")

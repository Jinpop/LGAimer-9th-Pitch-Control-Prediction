"""exp_031 — turn metrics.json into the exact tables report.md quotes.

Run LOCALLY after the artefacts are pulled back.  Every number that appears in
report.md is printed here, so `grep`-ing the report against this output is a
mechanical check that the report quotes nothing that is not in the artefacts.
"""
import json
import os
import sys

E = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(E, "metrics.json")))
K = M["k_grid_declared"]


def f(x, n=5):
    return f"{x:.{n}f}".rstrip("0").rstrip(".") if isinstance(x, float) else str(x)


print("=" * 78)
print("A. STAGE A — block repair, before/after, and the family verdict")
print("=" * 78)
A = M["stage_A_block_repair"]
for V, d in A["blocks"].items():
    print(f"\nblock {V}")
    for s in ("42", "7"):
        print(f"  seed {s}: stage2(exp_030) {d['C5_B3_stage2_exp030_by_seed'][s]:.14f}"
              f"  ->  stage3 {d['C5_B3_stage3_by_seed'][s]:.14f}"
              f"  ({d['C5_B3_stage3_score_by_seed'][s]})"
              f"   repair {d['repair_score_gain_by_seed'][s]:+} pts"
              f"   stack_iter {d['stack_iter_used'][s]}")
    print(f"  cell_iter_used: {d['cell_iter_used']}")
    print(f"  C5_B3 mean {d['C5_B3_stage3_mean']:.14f} ({d['C5_B3_stage3_mean_score']})"
          f"   LGBM_B3 mean(42,7) {d['LGBM_B3_mean_over_seeds_42_7']:.14f} "
          f"({d['LGBM_B3_mean_score_over_seeds_42_7']})")
    print(f"  C5 - LGBM = {d['C5_minus_LGBM_score_points']:+} pts   "
          f"C5 wins: {d['C5_wins_this_block']}")
print(f"\nVERDICT: {A['verdict']}")

print("\n" + "=" * 78)
print("B. GATES")
print("=" * 78)
g0 = M["gate0_base_reproduction"]
for s, v in g0["checks"].items():
    print(f"  G0 seed {s:>4}: expected {v['expected_from_exp030']:.14f}  got "
          f"{v['got']:.14f}  |diff| {v['abs_diff']:.3e}  bit-identical "
          f"{v['bit_identical']}  pass {v['passed']}")
g1 = M["gate1_block_bit_identity"]
print(f"  G1 block 2023 s42 bit-identical to exp_030 stage 1: {g1['bit_identical']}")
print(f"     exp_031 {g1['exp031_sha256'][:24]}...  exp_030 {g1['exp030_sha256'][:24]}...")
print(f"  G2 k0_self == k0_league: {M['gate2_k0_self_equals_k0_league']}")
print(f"  G3 d==0.0 on miss rows, all k: {all(M['gate3_miss_rows_zero_all_k'].values())}"
      f"  ({len(M['gate3_miss_rows_zero_all_k'])} k values: "
      f"{sorted(int(x) for x in M['gate3_miss_rows_zero_all_k'])})")
li = M["gate4_label_isolation"]
print(f"  G4 label isolation: {li['n_vectors_checked']} vectors constant inside "
      f"{li['n_lookup_groups']} lookup groups: "
      f"{all(li['check_b_constant_within_season_pitcher_hand'].values())}")
print(f"  G5 as-of independent re-read: "
      f"{M['gate5_asof_independent']['frame_season_matches_file']}, "
      f"{len(M['gate5_asof_independent']['per_season'])} seasons")

print("\n" + "=" * 78)
print("C. THE 18-CELL SURFACE (new base) beside exp_029's old-base surface")
print("=" * 78)
S = M["surfaces"]
side = M["old_base_surface_side_by_side"]
print(f"{'k':>6} {'tgt':<7} {'beta':>9} | {'NEW block':>13} {'score':>10} | "
      f"{'NEW fold':>13} {'score':>10} | {'OLD block':>10} {'OLD fold':>10}")
for k in K:
    for tg in ("self", "league"):
        nm = f"S{k}_{tg}"
        c = side["cells"][nm]
        mark = ""
        if nm == M["selection"]["selected_cell"]:
            mark = "  <= BLOCK ARGMIN (selected)"
        if nm == M["selection"]["fold_2024_argmin_ORACLE_DIAGNOSTIC"]:
            mark += "  <= fold oracle"
        print(f"{k:>6} {tg:<7} {c['new_beta']:>9} | {c['new_block_brier']:.14f} "
              f"{c['new_block_score']:>10} | {c['new_fold_brier']:.14f} "
              f"{c['new_fold_score']:>10} | {c['old_block_score']:>10} "
              f"{c['old_fold_score']:>10}{mark}")
print(f"\nreserve cells (computed for enclosure only, NOT part of the declared grid):")
for nm, v in S["reserve_block_brier"].items():
    print(f"  {nm:<16} block {v:.14f}   fold {S['reserve_fold_brier'][nm]:.14f}")
print(f"\nspearman rank corr of the two BLOCK surfaces over "
      f"{side['n_shared_cells']} shared cells: {side['spearman_rank_corr_block_surfaces']:.4f}")
print(f"old base block argmin {side['old_base_block_argmin']}   old base fold argmin "
      f"{side['old_base_fold_argmin']}")
print(f"new base block argmin {side['new_base_block_argmin']}   new base fold argmin "
      f"{side['new_base_fold_argmin_DIAGNOSTIC']}   argmin moved: {side['argmin_moved']}")

print("\n" + "=" * 78)
print("D. SELECTION, ENCLOSURE, self-vs-league")
print("=" * 78)
sel = M["selection"]
for k2, v in sel.items():
    if k2 != "selected_on":
        print(f"  {k2}: {v}")
print()
enc = M["enclosure"]
for tag in ("declared_self", "declared_league", "widened_self", "widened_league"):
    e = enc[tag]
    print(f"  {tag:<17} argmin k={e['argmin_k']:<6} enclosed={e['enclosed_both_sides']}"
          f"  lower-neighbour {e['score_drop_to_lower_neighbour']} pts worse, "
          f"upper {e['score_drop_to_upper_neighbour']} pts worse")
print(f"  both declared targets interior: {enc['declared_argmin_is_interior_both_targets']}")
cr = M["self_vs_league_crossing"]["block_surface"]
print(f"\n  self - league on the BLOCK surface (negative = self wins), score pts:")
print("   " + "  ".join(f"k{k}:{cr['diff_by_k_score_self_minus_league'][str(k)]:+.3f}"
                        for k in K))
print(f"  self wins {cr['n_k_where_self_wins']} k, league wins "
      f"{cr['n_k_where_league_wins']} k, any crossing: {cr['any_crossing']}")

print("\n" + "=" * 78)
print("E. BETA CURVE — quadratic test, clip, p distribution")
print("=" * 78)
bc = M["beta_curve"]
print(f"  beta_hat = {bc['beta_hat']}")
print(f"  {'beta':>10} {'block Brier':>18} {'nclip':>6} | {'fold mean Brier':>18} "
      f"{'score':>10} {'nclip':>6}")
for b in bc["grid"]:
    bb, fb = bc["blocks"][str(b)], bc["fold_2024"][str(b)]
    print(f"  {b:>10} {bb['brier']:.14f} {bb['n_clip']:>6} | "
          f"{fb['mean_brier']:.14f} {fb['mean_score']:>10} {fb['n_clip']:>6}")
q = bc["quadratic_test"]
for lab in ("blocks", "fold_2024"):
    print(f"  {lab}: max|residual| off a 3-point quadratic = "
          f"{q[lab]['max_abs_residual_on_other_points']:.3e} "
          f"(rel {q[lab]['max_rel_residual']:.3e}), exact to 1e-12: "
          f"{q[lab]['is_exactly_quadratic_to_1e-12']}, analytic argmin beta "
          f"{q[lab]['analytic_argmin_beta']:.6f}, clip rows over the whole curve "
          f"{q[lab]['total_clip_rows_over_curve']}")
pd_ = M["p_distribution_fold_2024"]
print(f"\n  p distribution of the NEW base (fold_2024):")
for s in ("42", "7", "2024"):
    d = pd_[s]
    print(f"    seed {s:>4}: mean {d['mean']:.6f} sd {d['sd']:.6f} range "
          f"[{d['min']:.6f}, {d['max']:.6f}]  p(1-p) mean {d['var_p1mp_mean']:.6f} "
          f"sd {d['var_p1mp_sd']:.6f} CV {d['var_p1mp_cv']:.6f}")
print(f"    3-seed mean sd {pd_['3seed_mean_of_sd']:.6f}, mean p(1-p) CV "
      f"{pd_['3seed_mean_of_var_cv']:.6f}")
print(f"    OLD base for contrast: sd {pd_['old_base_reference']['sd']}, range "
      f"{pd_['old_base_reference']['range']}")

print("\n" + "=" * 78)
print("F. ADOPTION (3 seeds, fold_2024 valid_all)")
print("=" * 78)
ad = M["adoption"]
b = M["base_no_offset_fold_2024"]
print(f"  cell {ad['selected_cell']}   beta {ad['beta_applied']}")
for i, s in enumerate(ad["seeds"]):
    print(f"  seed {s:>4}: control {ad['control_brier_by_seed'][i]:.14f}  candidate "
          f"{ad['candidate_brier_by_seed'][i]:.14f}  delta "
          f"{ad['paired_delta_by_seed'][i]:+.7e}  = "
          f"{ad['paired_delta_score_by_seed'][i]:+} pts")
print(f"  mean control  {ad['control_mean_brier']:.14f} ({ad['control_mean_score']})")
print(f"  mean candidate {ad['candidate_mean_brier']:.14f} ({ad['candidate_mean_score']})")
print(f"  mean delta {ad['mean_delta_brier']:+.7e} = {ad['mean_delta_score_points']:+} pts")
print(f"  control seed std (DENOMINATOR) {ad['control_seed_std_DENOMINATOR']:.6e}"
      f"  -> THRESHOLD = 3x = {ad['threshold_abs_mean_delta_THIS_IS_THE_THRESHOLD']:.6e}")
print(f"  candidate seed std {ad['candidate_seed_std']:.6e}   paired-delta seed std "
      f"{ad['paired_delta_seed_std_REFERENCE_ONLY']:.6e} (reference only)")
print(f"  MULTIPLE OF THE THRESHOLD  = {ad['MULTIPLE_OF_THE_THRESHOLD']:.4f}")
print(f"  MULTIPLE OF THE CONTROL STD = {ad['MULTIPLE_OF_THE_CONTROL_STD']:.4f}")
print(f"  signs agree {ad['signs_agree']}  improves {ad['improves']}  magnitude "
      f"{ad['passes_magnitude']}  ->  ADOPTED {ad['ADOPTED']}")
sm = M["seed_matched_beta_sensitivity"]
for nm, v in sm.items():
    print(f"  sensitivity, seed-matched beta on {nm}: mean {v['mean_brier']:.14f} "
          f"({v['mean_score']}), vs pooled {v['vs_pooled_score']:+} pts")

print("\n" + "=" * 78)
print("G. RULERS — what the offset bought on each base")
print("=" * 78)
r = M["rulers"]
for k2, v in r.items():
    print(f"  {k2}: {v}")
print(f"  fold_2024 segments of the selected cell (3-seed mean):")
fc = M["fold_2024_cells"][ad["selected_cell"]]
for s in ("42", "7", "2024"):
    ps = fc["per_seed"][s]
    print(f"    seed {s:>4}: valid_R {ps['valid_R']['brier_full']:.14f} "
          f"({ps['valid_R']['score_vs_fixed_ref']})  valid_F "
          f"{ps['valid_F']['brier_full']:.14f} ({ps['valid_F']['score_vs_fixed_ref']})"
          f"  clip {ps['correction_shift']['n_clip']}  max|shift| "
          f"{ps['correction_shift']['max_abs']:.6f}  raw range "
          f"[{ps['correction_shift']['raw_min']:.6f}, "
          f"{ps['correction_shift']['raw_max']:.6f}]")
print(f"  total clip rows on the selected cell over 3 seeds: {fc['n_clip_total']}")
print(f"\n  compute_time_min (grid) {M['compute_time_min']}")

"""exp_064 — 회수한 metrics.json 을 보고용 표로 찍는다 (로컬 실행)."""
import json
import sys

p = sys.argv[1] if len(sys.argv) > 1 else (
    "<PROJECT_ROOT>/experiments/exp_064/metrics.json")
M = json.load(open(p))

print("== GATES ==")
for k, v in M["gates"].items():
    print(f"  {k:44s} {v}")

print("\n== fold_2024 (주지표, valid_all 253,507) ==")
f = M["fold_2024"]
print(f"  brier {f['brier']:.10f}  score {f['score']:.5f}  bss {f['bss']:.7f} "
      f"auc {f['auc']:.6f}")
print(f"  valid_R brier {f['valid_R']['brier']:.10f} (n {f['valid_R']['n']}) "
      f"| valid_F brier {f['valid_F']['brier']:.10f} (n {f['valid_F']['n']})")

print("\n== 분할 A: 대조군 대비 ==")
A = M["stage_A"]
print(f"  ship p6 99열 seedavg   {A['nn99ship']['seedavg_logit_score']:.5f}")
print(f"  exp_061 우리 p6 99열   {A['control_exp061_99col_ours_p6']['score']:.5f} "
      f"(delta {A['D_vs_exp061_99col']:+.5f})")
print(f"  exp_063 85열           {A['control_exp063_85col']['score']:.5f} "
      f"(delta {A['D_vs_exp063_85col']:+.5f})")
print(f"  per-seed {['%.5f' % s for s in A['nn99ship']['per_seed_score']]} "
      f"mean {A['nn99ship']['per_seed_score_mean']:.5f} "
      f"sd {A['nn99ship']['per_seed_score_sd']:.5f}")
print(f"  best_epochs {A['nn99ship']['best_epochs']}")

print("\n== 분할 B (valid 2023 R) ==")
B = M["stage_B"]
print(f"  ship p6 99열 seedavg   {B['nn99ship']['seedavg_logit_score']:.5f}")
print(f"  exp_061 우리 p6 99열   {B['control_exp061_99col_ours_p6']['score']:.5f} "
      f"(delta {B['D_vs_exp061_99col']:+.5f})")
print(f"  exp_063 85열           {B['control_exp063_85col']['score']:.5f} "
      f"(delta {B['D_vs_exp063_85col']:+.5f})")
print(f"  per-seed {['%.5f' % s for s in B['nn99ship']['per_seed_score']]} "
      f"mean {B['nn99ship']['per_seed_score_mean']:.5f} "
      f"sd {B['nn99ship']['per_seed_score_sd']:.5f}")

print("\n== 시드별 짝지은 차 ==")
for sp in ("A", "B"):
    pb = M["paired_by_seed"][sp]
    print(f"  분할 {sp}: vs exp_061 99열 mean {pb['delta_vs_ours99']['mean']:+.5f} "
          f"sd {pb['delta_vs_ours99']['sd']:.5f} neg {pb['delta_vs_ours99']['n_negative']}/4"
          f" | vs exp_063 85열 mean {pb['delta_vs_85']['mean']:+.5f} "
          f"sd {pb['delta_vs_85']['sd']:.5f} neg {pb['delta_vs_85']['n_negative']}/4")
    for r in pb["per_seed"]:
        print(f"     seed {r['seed']:5d} ship {r['score_ship99']:9.4f} "
              f"ours99 {r['score_ours99']:9.4f} 85 {r['score_85']:9.4f} "
              f"| d99 {r['delta_vs_ours99']:+9.4f} d85 {r['delta_vs_85']:+9.4f}")

if "correlations" in M:
    C = M["correlations"]
    print("\n== 상관 (로짓 공간, fold_2024) ==")
    print("  NN 간:", json.dumps(C["corr_between_NNs"], indent=None))
    print(f"  {'대상':14s} {'ship':>9s} {'ours99':>9s} {'85':>9s} {'ship-99':>9s} "
          f"{'ship-85':>9s}")
    for k, v in C["corr_three_way"].items():
        print(f"  {k:14s} {v['nn_ship']:9.4f} {v['nn99_ours_p6']:9.4f} "
              f"{v['nn85_dropped']:9.4f} {v['ship_minus_99']:+9.4f} "
              f"{v['ship_minus_85']:+9.4f}")
    print("  기준 재현:", json.dumps(C["reference_check"], ensure_ascii=False,
                                 indent=None)[:400])
    print("\n  out-of-time Brier:")
    for k, v in C["brier_out_of_time"].items():
        print(f"    {k:22s} brier {v['brier']:.10f}  score {v['score']:9.5f}")

print("\n== 시간 ==")
print(f"  train_time_min {M['train_time_min']:.2f}")
print(f"  repro {M['repro_gate']['wall_min']:.2f} | A {M['stage_A']['wall_min']:.2f}"
      f" | B {M['stage_B']['wall_min']:.2f}")

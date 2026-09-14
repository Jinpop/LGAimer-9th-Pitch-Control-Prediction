#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp_115 3단 — bat_phand vs pit_ball3 두 칸 비교표."""
import json
W = "/home/<user>/SungjinPark/build_pb3_probe/work"
m = json.load(open(f"{W}/measure30.json", encoding="utf-8"))     # pit_ball3
c = json.load(open(f"{W}/solve30.json", encoding="utf-8"))       # bat_phand (후보 CZ)
AX = ("sp_pit_split", "beta_ins_off", "ctrl_asof_psr", "count_offsets_CS3")

qb = c["amplitude_on_cal_frame"]["K_mean_dp2_pts"]
cb = sum(c["cross_terms_all_axes"][k]["cross_2K_mean_pts"] for k in AX)
qp = m["ceiling"]["q_own_curvature"]
cp = m["ceiling"]["cross_sum"]
A1 = m["ceiling"]["A_sibling_from_LB"] * 0.5

LAM = [(2.47 / 4.315, "0.5724  bat f24 t비"),
       (0.414 / 0.515, "0.8039  bat 연도전이비"),
       (1.0, "1.0000  두 칸 L 동일 가정"),
       (0.254 / 0.515, "0.4932  pb3 연도전이비"),
       (0.22 / 4.315, "0.0510  pb3 f24 t비")]

rows = {"q_own_curvature": {"bat_phand": qb, "pit_ball3": qp},
        "cross_sum": {"bat_phand": cb, "pit_ball3": cp},
        "cost": {"bat_phand": qb + cb, "pit_ball3": qp + cp},
        "rho": {k: {"bat_phand": c["cross_terms_all_axes"][k]["rho_all"],
                    "pit_ball3": m["cross_terms_all_axes"][k]["rho_all"]}
                for k in AX + ("failtype_b_direction",)},
        "A_sibling_half_step_at_lambda1": A1}
tab = {}
for lam, nm in LAM:
    A = A1 * lam
    tab[nm] = {
        "lambda": lam, "A": A,
        "bat_gain_half_step": {"favourable": A - (qb + cb), "opposite": -A - (qb + cb)},
        "pb3_gain_half_step": {"favourable": A - (qp + cp), "opposite": -A - (qp + cp)},
        "bat_ceiling": {"favourable": (A - cb) ** 2 / (4 * qb),
                        "opposite": (A + cb) ** 2 / (4 * qb)},
        "pb3_ceiling": {"favourable": (A - cp) ** 2 / (4 * qp),
                        "opposite": (A + cp) ** 2 / (4 * qp)}}
out = {"summary": rows, "table": tab,
       "amplitude_convention": {
           "bat_phand": {"ratio": 0.2495992978818122, "pct_dev_from_0.25": -0.16},
           "pit_ball3": {"ratio": m["amplitude_convention_check"]["ours_over_sibling"],
                         "pct_dev_from_0.25": m["amplitude_convention_check"]["pct_dev"]}},
       "frames": json.load(open(f"{W}/frames20.json", encoding="utf-8")),
       "determinism": m["determinism"]}
json.dump(out, open(f"{W}/compare40.json", "w"), ensure_ascii=False, indent=1)

print(f"{'':34s}{'bat_phand':>12s}{'pit_ball3':>12s}")
print(f"{'q  self curvature':34s}{qb:12.4f}{qp:12.4f}")
print(f"{'cross sum (all axes)':34s}{cb:+12.4f}{cp:+12.4f}")
print(f"{'cost = q + cross':34s}{qb + cb:12.4f}{qp + cp:12.4f}")
for k in AX + ("failtype_b_direction",):
    print(f"{'  rho ' + k:34s}{rows['rho'][k]['bat_phand']:+12.6f}"
          f"{rows['rho'][k]['pit_ball3']:+12.6f}")
print()
print(f"{'lambda':34s}{'bat ceil fav/opp':>24s}{'pb3 ceil fav/opp':>24s}")
for lam, nm in LAM:
    t = tab[nm]
    print(f"{nm:34s}{t['bat_ceiling']['favourable']:11.3f}/"
          f"{t['bat_ceiling']['opposite']:11.3f}"
          f"{t['pb3_ceiling']['favourable']:12.3f}/{t['pb3_ceiling']['opposite']:10.3f}")
print()
print("half-step gain (fav / opp):")
for lam, nm in LAM:
    t = tab[nm]
    print(f"  {nm:32s} bat {t['bat_gain_half_step']['favourable']:+7.3f} /"
          f" {t['bat_gain_half_step']['opposite']:+7.3f}    "
          f"pb3 {t['pb3_gain_half_step']['favourable']:+7.3f} /"
          f" {t['pb3_gain_half_step']['opposite']:+7.3f}")
print("COMPARE_OK")

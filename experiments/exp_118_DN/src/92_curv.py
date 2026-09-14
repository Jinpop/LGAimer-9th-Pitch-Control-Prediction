#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S255 — 새 축의 자기곡률을 ★라벨 0개 · LB 슬롯 0개★ 로 실측한다.

    Q(u) ≡ K*u^2*G = (K/2) * [ mean((p+ - p0)^2) + mean((p- - p0)^2) ]

p0 = 부모 DM 예측, p+ = u=+0.008 판본, p- = u=-0.008 판본.  추론 3회면 끝난다.
선형항이 정확히 상쇄되므로 라벨이 필요 없다.  🔴 이 값은 ★크기의 상한★ 이지 부호가 아니다.

🔴 이것은 대회 test 가 아니라 ★게이트 프레임★ 에서 잰 것이다(S209 — 프레임 명시 필수).
   cov·band 는 둘 다 train season==2024 행 245,789개의 season 을 2025 로 덮은
   대역(스탠드인) 프레임이다.  배포 test 자체가 아니다.
"""
import json

import numpy as np
import pandas as pd

H = "/home/<user>/SungjinPark"
B = f"{H}/build_DN"
RUNS = f"{B}/runs"
K = 400309.19882517256
U = 0.008
FRAMES = ("cov", "band")


def rd(tag):
    d = pd.read_csv(f"{RUNS}/{tag}.csv", encoding="utf-8-sig",
                    float_precision="round_trip")
    return d.set_index("row_id")["control_success"]


out = {"K": K, "u_probe": U,
       "formula": "Q = (K/2)*[mean((p+ - p0)^2) + mean((p- - p0)^2)];  Q = K*u^2*G",
       "🔴_frame_disclosure": (
         "🔴 대회 test 가 아니라 게이트 프레임에서 쟀다(S209).  cov·band 는 train "
         "season==2024 의 245,789행을 골라 season 을 2025 로 덮은 대역(스탠드인) "
         "프레임이고, 배포 test 의 분포·미지 주체율을 대표하지 않는다(S32).  이 두 "
         "프레임에서 이 열의 결측은 82/245,789 (0.033%) 로 train-2024 원본(88/253,507, "
         "0.035%)과 같은 자릿수다."),
       "🔴_what_this_does_not_give": (
         "부호를 주지 않는다.  gain(u) = 2*u*K*L - Q(u) 에서 이 측정은 Q 만 준다.  L 은 "
         "★이 기반에서 산 LB 점★ 에서만 온다(S262·S217)."),
       "frames": {}}

for fr in FRAMES:
    p0 = rd(f"par__{fr}")
    pp = rd(f"dnp2__{fr}").reindex(p0.index)   # 출하판 DNP (u = +0.008)
    pm = rd(f"dn__{fr}").reindex(p0.index)     # 출하판 DN  (u = -0.008)
    a0 = p0.to_numpy(np.float64)
    ap = pp.to_numpy(np.float64)
    am = pm.to_numpy(np.float64)
    np.save(f"{B}/work/p0_{fr}.npy", a0)
    np.save(f"{B}/work/pplus_{fr}.npy", ap)
    np.save(f"{B}/work/pminus_{fr}.npy", am)
    dp = ap - a0
    dm = am - a0
    q = 0.5 * K * (float((dp ** 2).mean()) + float((dm ** 2).mean()))
    asym = ap + am - 2.0 * a0
    out["frames"][fr] = {
      "n": int(len(a0)),
      "Q_pts": q,
      "K_times_G": q / (U * U),
      "G": q / (K * U * U),
      "half_plus_pts": float(0.5 * K * (dp ** 2).mean()),
      "half_minus_pts": float(0.5 * K * (dm ** 2).mean()),
      "asymmetry_of_the_two_halves":
        float(0.5 * K * (dp ** 2).mean() - 0.5 * K * (dm ** 2).mean()),
      "mean_p0": float(a0.mean()), "mean_pplus": float(ap.mean()),
      "mean_pminus": float(am.mean()),
      "mean_antisymmetry_pplus_plus_pminus_minus_2p0": float(asym.mean()),
      "rms_antisymmetry": float(np.sqrt((asym ** 2).mean())),
      "max_abs_dplus": float(np.abs(dp).max()),
      "max_abs_dminus": float(np.abs(dm).max()),
      "rows_moved_plus": int(np.count_nonzero(dp)),
      "rows_moved_minus": int(np.count_nonzero(dm)),
      "🔴_quadratic_model_check": (
        "2차식 모형이 맞다면 두 반쪽이 거의 같고(asymmetry_of_the_two_halves ≈ 0) "
        "반대칭 항의 평균이 곡률 항의 2배로 잡힌다.  두 반쪽 차이가 Q 의 몇 %인지 "
        "보고 판단한다."),
    }
    out["frames"][fr]["halves_ratio"] = (
      out["frames"][fr]["half_plus_pts"] / out["frames"][fr]["half_minus_pts"])

# 축 자체의 손익분기 — Q 하나로 계산되는 것만 적는다 (L 은 안 빌린다)
for fr in FRAMES:
    q = out["frames"][fr]["Q_pts"]
    out["frames"][fr]["🔴_break_even_L"] = {
      "condition": "gain(u) = 2*u*K*L - Q(u) > 0  <=>  |K*L| > Q/(2*|u|)",
      "K_times_L_needed_pts_per_unit_u": q / (2.0 * U),
      "meaning": (f"u = -0.008 에서 이 축이 양수가 되려면 이 기반의 |K*L| 이 "
                  f"{q / (2.0 * U):.1f} pts/단위u 보다 커야 한다.  🔴 우리는 이 축의 "
                  "L 을 모른다 — 그것이 이번 제출이 사는 것이다."),
      "vertex_if_L_were_known": "u* = L/G = (K*L)/(K*G);  max gain = (K*L)^2/(K*G)",
    }

# 🔴 산문만 고친 재빌드가 예측을 한 비트도 안 바꿨음을 기계로 확인한다.
#    1차 배치의 dnp__* 는 candidate="DN_PROBE_PLUS" 산문을 달고 돌았고, 2차 배치의
#    dnp2__* 는 candidate="DNP" 로 다시 쓴 산문을 달고 돌았다.  u/center/scale/코드가
#    같으므로 출력 CSV sha256 이 같아야 한다.
import csv as _csv
_man = {r["tag"]: r for r in _csv.DictReader(open(f"{B}/runs/manifest.tsv"), delimiter="\t")}
out["🔴_prose_only_rebuild_is_inert"] = {
 fr: {"first_batch_dnp": _man.get(f"dnp__{fr}", {}).get("out_sha256"),
      "shipping_dnp2": _man.get(f"dnp2__{fr}", {}).get("out_sha256"),
      "identical": _man.get(f"dnp__{fr}", {}).get("out_sha256")
                   == _man.get(f"dnp2__{fr}", {}).get("out_sha256")}
 for fr in FRAMES}

json.dump(out, open(f"{B}/work/curvature.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))

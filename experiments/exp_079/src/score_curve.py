"""exp_079 stage1 분석 — 트리 수별 정직면 점수 곡선.

원점은 팀원 배포본이 아니라 **우리가 같은 코드로 적합한 354그루**다(형제 이동 제거).
곡선의 각 점은 같은 1600그루 적합을 ntree_end 로 자른 것이고, stage0b G0e 가
'자른 것 == 짧은 적합'을 비트 동일로 확인했다.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"

import itertools    # noqa: E402
import json         # noqa: E402
import time         # noqa: E402

import numpy as np  # noqa: E402

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp079"
C77 = f"{H}/exp077/cache"
sys.path.insert(0, f"{OUT}/src")
import surface079 as S              # noqa: E402

SEEDS = [42, 777, 2024, 101]
ITERS = 1600
T0 = time.time()
R = {"exp_id": "exp_079", "part": "stage1_curve",
     "surface": "OOT 2024 (<=2023 학습 -> 2024 예측 253,507행), BO 상수, "
                "전역 재중심화, exp_078 stage4 와 동일 산술",
     "seeds": SEEDS, "iterations_fit": ITERS, "threads": 6}

meta = {s: json.load(open(f"{OUT}/cache/oot_s{s}_i{ITERS}_t6.json"))
        for s in SEEDS}
GRID = meta[SEEDS[0]]["grid"]
assert all(meta[s]["grid"] == GRID for s in SEEDS)
R["grid"] = GRID
R["fit_records"] = {str(s): {k: meta[s][k] for k in
                             ("tree_count", "learn_best_error", "model_sha256",
                              "n_train", "sec_fit", "sec_total")}
                    for s in SEEDS}

LGT = {}      # LGT[s][k] = 로짓 벡터
for s in SEEDS:
    LGT[s] = {}
    for k in GRID:
        LGT[s][k] = S.lg(np.load(f"{OUT}/cache/p_oot_s{s}_i{ITERS}_t6_k{k}.npy"))

# ── G0e-full  실행길이 불변, 전체 규모 ──────────────────────────────
# exp_077 은 같은 마스크·같은 하이퍼·같은 스레드로 iterations=354 를 독립 적합했다.
gf = {}
for s in SEEDS:
    a = np.load(f"{OUT}/cache/p_oot_s{s}_i{ITERS}_t6_k354.npy")
    b = np.load(f"{C77}/_cat_oot_s{s}_t6.npy")
    gf[str(s)] = {"max_abs_diff": float(np.abs(a - b).max()),
                  "n_diff_rows": int((a != b).sum()),
                  "bitwise_equal": bool(np.array_equal(a, b))}
R["G0e_full_runlength_invariance_vs_exp077"] = {
    "per_seed": gf,
    "all_bitwise_equal": bool(all(v["bitwise_equal"] for v in gf.values())),
    "해석": "우리 1600그루 적합을 354 에서 자른 것 == exp_077 의 독립 354그루 "
            "적합.  참이면 곡선의 원점이 exp_077 산출물과 문자 그대로 같다."}

# ── 곡선 ────────────────────────────────────────────────────────────
def L4(k, seeds=SEEDS):
    return np.mean([LGT[s][k] for s in seeds], axis=0)


base_partner, base_partner_b = S.surface_pts(S.lg(S.V["p_cat"]))
arm_partner, _ = S.arm_pts(S.lg(S.V["p_cat"]))
R["partner_reference"] = {
    "blend_pts": base_partner, "blend_brier": base_partner_b,
    "arm_pts": arm_partner,
    "expected_exp078_base": 928.501461058684,
    "identity_abs_diff": abs(base_partner - 928.501461058684)}

CUR = {}
for k in GRID:
    bp, bb = S.surface_pts(L4(k))
    ap, ab = S.arm_pts(L4(k))
    CUR[k] = {"blend_pts": bp, "blend_brier": bb, "arm_pts": ap, "arm_brier": ab}
origin = CUR[354]["blend_pts"]
for k in GRID:
    CUR[k]["blend_delta_vs_354"] = CUR[k]["blend_pts"] - origin
    CUR[k]["arm_delta_vs_354"] = CUR[k]["arm_pts"] - CUR[354]["arm_pts"]
R["curve_4seed"] = {str(k): CUR[k] for k in GRID}
R["origin_our_354_blend_pts"] = origin
R["origin_our_354_arm_pts"] = CUR[354]["arm_pts"]
R["origin_vs_partner_gap_pts"] = origin - base_partner

kb = max(GRID, key=lambda k: CUR[k]["blend_pts"])
ka = max(GRID, key=lambda k: CUR[k]["arm_pts"])
R["argmax"] = {
    "blend_argmax_trees": kb, "blend_gain_pts": CUR[kb]["blend_delta_vs_354"],
    "arm_argmax_trees": ka, "arm_gain_pts": CUR[ka]["arm_delta_vs_354"],
    "arm_to_blend_dilution": (CUR[kb]["blend_delta_vs_354"]
                              / CUR[ka]["arm_delta_vs_354"])
    if CUR[ka]["arm_delta_vs_354"] else None}

best = CUR[kb]["blend_pts"]
for tol in (0.10, 0.25, 1.00):
    ks = [k for k in GRID if best - CUR[k]["blend_pts"] <= tol]
    R.setdefault("flat_region", {})[f"within_{tol:.2f}pt_of_max"] = {
        "trees": ks, "min": min(ks), "max": max(ks), "n": len(ks)}

# 국소 2차 보정 argmax (격자 3점)
i = GRID.index(kb)
if 0 < i < len(GRID) - 1:
    x = np.array(GRID[i - 1:i + 2], float)
    yv = np.array([CUR[k]["blend_pts"] for k in GRID[i - 1:i + 2]])
    c = np.polyfit(x, yv, 2)
    R["argmax"]["quadratic_refined_trees"] = (float(-c[1] / (2 * c[0]))
                                              if c[0] < 0 else None)
    R["argmax"]["quadratic_curvature"] = float(c[0])

# ── 척도 대 정보 분리 ───────────────────────────────────────────────
# 트리를 늘리면 팔 로짓의 산포가 커진다.  블렌드 가중은 354그루 팔에 맞춰
# 고정돼 있으므로 이득의 일부는 '정보'가 아니라 '척도'일 수 있다.  척도를
# 354 에 맞춘 대조를 함께 재서 둘을 가른다.  척도분은 alpha 로도 살 수 있는
# 것이고 exp_078 이 alpha 의 꼭지점이 거의 소진됐다고 실측했다.
l354 = L4(354)
mu0, sd0 = float(l354.mean()), float(l354.std())
SC = {}
for k in GRID:
    lk = L4(k)
    mu, sd = float(lk.mean()), float(lk.std())
    lm = mu0 + (lk - mu) * (sd0 / sd)
    bp, _ = S.surface_pts(lm)
    SC[str(k)] = {"logit_mean": mu, "logit_sd": sd, "sd_ratio_vs_354": sd / sd0,
                  "corr_with_354": float(np.corrcoef(lk, l354)[0, 1]),
                  "scale_matched_blend_pts": bp,
                  "scale_matched_delta_vs_354": bp - origin}
R["scale_vs_information"] = SC
R["scale_vs_information_note"] = (
    "scale_matched_delta 는 팔 로짓의 sd 를 354 에 맞춘 뒤의 이득이다. "
    "raw Δ 와 거의 같으면 이득은 정보이고, 크게 줄면 척도(=alpha 로도 살 수 "
    "있는 것)다.")

# ── 시드별 곡선 (부호 일치) ─────────────────────────────────────────
PS = {}
for s in SEEDS:
    d = {}
    for k in GRID:
        bp, _ = S.surface_pts(LGT[s][k])
        d[str(k)] = bp
    o = d["354"]
    PS[str(s)] = {"blend_pts": d,
                  "delta_vs_354": {k: v - o for k, v in d.items()},
                  "argmax_trees": int(max(GRID, key=lambda k: d[str(k)])),
                  "gain_at_4seed_argmax": d[str(kb)] - o,
                  "max_gain": max(v - o for v in d.values())}
R["per_seed_curve"] = PS
R["per_seed_sign_agreement_at_4seed_argmax"] = {
    "n_positive": int(sum(PS[str(s)]["gain_at_4seed_argmax"] > 0 for s in SEEDS)),
    "n_seeds": len(SEEDS),
    "values": {str(s): PS[str(s)]["gain_at_4seed_argmax"] for s in SEEDS}}

# ── 3시드 부분집합 (원점 선택에 대한 민감도) ────────────────────────
sub = []
for cmb in itertools.combinations(SEEDS, 3):
    a0, _ = S.surface_pts(L4(354, cmb))
    a1, _ = S.surface_pts(L4(kb, cmb))
    sub.append(a1 - a0)
R["subset3_delta_at_argmax"] = {
    "values": sub, "mean": float(np.mean(sub)),
    "sd": float(np.std(sub, ddof=1)), "min": float(min(sub)),
    "max": float(max(sub)), "n_positive": int(sum(v > 0 for v in sub))}

# ── 전이 투영 ───────────────────────────────────────────────────────
g = CUR[kb]["blend_delta_vs_354"]
R["transfer_projection"] = {
    "honest_gain_pts": g,
    "measured_band_low_0.3929": g * 0.3929,
    "measured_band_high_0.5463": g * 0.5463,
    "🔴": "S1·S2 — 정직면은 LB 눈금이 아니고 전이비는 축마다 다르며 부호까지 "
          "갈린 이력이 있다. 단일 배율을 믿지 말 것."}

# ── 사전 등록 판정 ──────────────────────────────────────────────────
if kb > 354 and g >= 8.0:
    verdict, why = "A", "argmax > 354 이고 정직면 이득 >= +8점 → 후보 빌드 권고"
elif kb > 354 and g >= 3.0:
    verdict, why = "B", "argmax > 354 이나 이득 +3~8 → 빌드 권고, 기대치 하향"
else:
    verdict, why = "C", "argmax <= 354 또는 이득 < +3 → 축 종료"
R["preregistered_verdict"] = {
    "verdict": verdict, "reason": why,
    "rule": "A: argmax>354 & gain>=+8 / B: argmax>354 & +3<=gain<+8 / "
            "C: argmax<=354 or gain<+3  (config 사전 등록, 결과 보고 바꾸지 않음)",
    "blend_argmax_trees": kb, "blend_gain_pts": g}

R["wall_min"] = (time.time() - T0) / 60.0
json.dump(R, open(f"{OUT}/out/stage1_curve.json", "w"), indent=1,
          ensure_ascii=False)
print("=== G0e-full ===", json.dumps(
    R["G0e_full_runlength_invariance_vs_exp077"]["per_seed"], ensure_ascii=False),
    flush=True)
print(f"\n원점(우리 354) {origin:.5f}점 · 팀원 납품 {base_partner:.5f}점 "
      f"· 차 {origin - base_partner:+.5f}", flush=True)
print("\n트리수   블렌드점수      Δvs354      팔점수      팔Δ", flush=True)
for k in GRID:
    c = CUR[k]
    print(f"{k:5d}  {c['blend_pts']:12.5f}  {c['blend_delta_vs_354']:+9.5f}  "
          f"{c['arm_pts']:11.5f}  {c['arm_delta_vs_354']:+9.5f}", flush=True)
print(f"\nargmax {kb}그루 · 이득 {g:+.5f}점 · 판정 {verdict}", flush=True)
print(f"평탄(±0.25점) {R['flat_region']['within_0.25pt_of_max']['trees']}",
      flush=True)
print(f"척도맞춤 Δ@argmax {SC[str(kb)]['scale_matched_delta_vs_354']:+.5f}점 "
      f"(raw {g:+.5f}) · sd비 {SC[str(kb)]['sd_ratio_vs_354']:.5f}", flush=True)
print(f"시드별 부호 {R['per_seed_sign_agreement_at_4seed_argmax']['values']}",
      flush=True)
print(f"\n기록 {OUT}/out/stage1_curve.json · {R['wall_min']:.2f}분", flush=True)

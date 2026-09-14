"""exp_102 stage C — an A5000 -> L4 conversion factor for GPU candidates.

S122 gives the CPU ruler (our 6 threads / 1.60 = scoring server).  There is no GPU
ruler in the ledger, and the scoring server has an L4 while our box has A5000s.

🔴 L4 CANNOT BE MEASURED HERE.  Every L4 number below is an ESTIMATE.

Method — do not trust datasheet peaks, measure our own two limbs first:
  P1  achieved COMPUTE peak on A5000  (big square fp32 GEMM)
  P2  achieved BANDWIDTH peak on A5000 (large stream copy)
then place each real workload between them:
  W1  shipped deep-arm forward, folded form (deep_infer._one): 94->256->128->64->1
  W2  deep-arm training step (TokenMLP fwd+bwd, batch 4096) — the GPU work we
      actually do today (exp_097/102 fits, 12.9 s / 1632 steps)
A workload's achieved-fraction of each limb says which limb binds it, and the
A5000->L4 factor is the ratio on that limb.
"""
import json
import os
import sys
import time

import numpy as np
import torch

H = "/home/<user>/SungjinPark"
B102 = f"{H}/exp102"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(2)
sys.path.insert(0, f"{B102}/src")
from arch102 import TokenMLP  # noqa: E402

DEV = os.environ.get("SC_DEV", "cuda:1")
dev = torch.device(DEV)
torch.cuda.set_device(dev)          # 🔴 synchronize() 는 현재 장치만 기다린다

SPECS = {
    "RTX_A5000": {"fp32_tflops": 27.8, "bf16_tensor_tflops": 111.1,
                  "mem_bw_gb_s": 768.0, "vram_gb": 24, "sm": 64, "tdp_w": 230,
                  "mem": "GDDR6 384-bit"},
    "L4": {"fp32_tflops": 30.3, "bf16_tensor_tflops": 121.0,
           "mem_bw_gb_s": 300.0, "vram_gb": 24, "sm": 58, "tdp_w": 72,
           "mem": "GDDR6 192-bit"},
}
for v in SPECS.values():
    v["ridge_flop_per_byte_fp32"] = v["fp32_tflops"] * 1e12 / (v["mem_bw_gb_s"] * 1e9)

R = {"exp_id": "exp_102", "stage": "sC_l4", "device_measured": DEV,
     "🔴": "L4 실측 불가 — L4 수치는 전부 추정이다.",
     "specs": SPECS,
     "datasheet_ratio_L4_over_A5000": {
         "fp32_tflops": SPECS["L4"]["fp32_tflops"] / SPECS["RTX_A5000"]["fp32_tflops"],
         "bf16_tensor_tflops": (SPECS["L4"]["bf16_tensor_tflops"]
                                / SPECS["RTX_A5000"]["bf16_tensor_tflops"]),
         "mem_bw": SPECS["L4"]["mem_bw_gb_s"] / SPECS["RTX_A5000"]["mem_bw_gb_s"]}}


def timeit(fn, n_warm=5, n_it=30):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize(dev)
    t = time.perf_counter()
    for _ in range(n_it):
        fn()
    torch.cuda.synchronize(dev)
    return (time.perf_counter() - t) / n_it


# ── P1: achieved compute peak (big square GEMM) ──────────────────────────
n = 8192
A = torch.randn(n, n, device=dev)
Bm = torch.randn(n, n, device=dev)
dt = timeit(lambda: torch.mm(A, Bm), 3, 10)
P1 = 2 * n ** 3 / dt / 1e12
del A, Bm
torch.cuda.empty_cache()

# ── P2: achieved bandwidth peak (stream copy) ────────────────────────────
m = 1 << 27                                     # 128M fp32 = 512 MB
src = torch.empty(m, device=dev)
dst = torch.empty(m, device=dev)
dt = timeit(lambda: dst.copy_(src), 3, 20)
P2 = 2 * m * 4 / dt / 1e9                       # read + write
del src, dst
torch.cuda.empty_cache()

R["a5000_achieved_peaks"] = {
    "compute_tflops_fp32_gemm8192": P1,
    "fraction_of_datasheet_fp32": P1 / SPECS["RTX_A5000"]["fp32_tflops"],
    "bandwidth_gb_s_stream_copy": P2,
    "fraction_of_datasheet_bw": P2 / (SPECS["RTX_A5000"]["mem_bw_gb_s"]),
    "achieved_ridge_flop_per_byte": P1 * 1e12 / (P2 * 1e9)}
print("P1 compute peak  %7.2f TFLOP/s (%.0f%% of datasheet)"
      % (P1, 100 * P1 / SPECS["RTX_A5000"]["fp32_tflops"]))
print("P2 bandwidth peak %6.1f GB/s   (%.0f%% of datasheet)"
      % (P2, 100 * P2 / SPECS["RTX_A5000"]["mem_bw_gb_s"]))

# ── W1: shipped folded deep-arm forward (deep_infer._one arithmetic) ─────
sch = json.load(open(f"{B102}/work/schema_f64o.json"))
CARDS = sch["cards"]
NUMC = 94
GEMM = [(NUMC, 256), (256, 128), (128, 64), (64, 1)]
flops_row = sum(2 * a * b for a, b in GEMM)
bytes_row = 4 * sum(a + b for a, b in GEMM)
Wl = [torch.randn(a, b, device=dev) for a, b in GEMM]
bl = [torch.randn(b, device=dev) for _, b in GEMM]
Tl = [torch.randn(c, 256, device=dev) for c in CARDS]


def fwd_folded(X, C):
    z = X @ Wl[0] + bl[0]
    for j in range(len(CARDS)):
        z = z + Tl[j][C[:, j]]
    z = torch.nn.functional.gelu(torch.nn.functional.layer_norm(z, (256,)))
    z = torch.nn.functional.gelu(
        torch.nn.functional.layer_norm(z @ Wl[1] + bl[1], (128,)))
    z = torch.nn.functional.gelu(z @ Wl[2] + bl[2])
    return z @ Wl[3] + bl[3]


w1 = []
for bs in (4096, 16384, 65536, 131072):
    X = torch.randn(bs, NUMC, device=dev)
    C = torch.zeros(bs, len(CARDS), dtype=torch.long, device=dev)
    with torch.no_grad():
        dt = timeit(lambda: fwd_folded(X, C), 5, 30)
    w1.append({"batch": bs, "sec_per_call": dt, "rows_per_sec": bs / dt,
               "achieved_tflops": bs * flops_row / dt / 1e12,
               "achieved_gb_s": bs * bytes_row / dt / 1e9})
    print("W1 folded fwd batch %6d  %7.3f ms  %.3e rows/s  %.3f TF  %6.1f GB/s"
          % (bs, dt * 1e3, bs / dt, w1[-1]["achieved_tflops"],
             w1[-1]["achieved_gb_s"]))
    del X, C
    torch.cuda.empty_cache()
best1 = max(w1, key=lambda r: r["rows_per_sec"])
R["W1_shipped_folded_forward"] = {
    "gemms": GEMM, "flops_per_row": flops_row,
    "activation_bytes_per_row_fp32": bytes_row,
    "arithmetic_intensity": flops_row / bytes_row,
    "curve": w1, "best": best1,
    "frac_of_achieved_compute": best1["achieved_tflops"] / P1,
    "frac_of_achieved_bandwidth": best1["achieved_gb_s"] / P2,
    "sec_for_245789_rows": 245789 / best1["rows_per_sec"]}

# ── W2: the GPU work we actually run today (training step) ───────────────
X = np.load(f"{B102}/work/X_f64o.npy", mmap_mode="r")
Cc = np.load(f"{B102}/work/codes_f64o.npy", mmap_mode="r")
Xt = torch.from_numpy(np.ascontiguousarray(X[:4096])).to(dev)
Ct = torch.from_numpy(np.ascontiguousarray(Cc[:4096]).astype(np.int64)).to(dev)
yt = torch.randint(0, 2, (4096,), device=dev).double()
model = TokenMLP(Xt.shape[1], CARDS, p_drop=0.0, init="ftt").to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
lossf = torch.nn.BCEWithLogitsLoss()


def step():
    opt.zero_grad(set_to_none=True)
    lossf(model(Ct, Xt), yt).backward()
    opt.step()


dt2 = timeit(step, 5, 30)
R["W2_training_step"] = {
    "batch": 4096, "sec_per_step": dt2,
    "steps_per_sec": 1 / dt2,
    "implied_1632_step_fit_sec": 1632 * dt2,
    "measured_fit_wall_sec_exp102": 12.9,
    "note": "실측 fit 12.9 초에는 데이터 적재·검증 추론이 포함돼 순수 스텝보다 크다."}
print("W2 train step batch 4096: %6.3f ms -> 1632 steps = %.1f s"
      % (dt2 * 1e3, 1632 * dt2))

# ── verdict ──────────────────────────────────────────────────────────────
fc = R["W1_shipped_folded_forward"]["frac_of_achieved_compute"]
fb = R["W1_shipped_folded_forward"]["frac_of_achieved_bandwidth"]
bw_ratio = SPECS["L4"]["mem_bw_gb_s"] / SPECS["RTX_A5000"]["mem_bw_gb_s"]
fp_ratio = SPECS["L4"]["fp32_tflops"] / SPECS["RTX_A5000"]["fp32_tflops"]
LATENCY_BOUND = max(fc, fb) < 0.25
R["bottleneck"] = {
    "W1_frac_of_achieved_compute": fc, "W1_frac_of_achieved_bandwidth": fb,
    "latency_or_occupancy_bound": bool(LATENCY_BOUND),
    "reading": ("두 한계 어느 쪽에도 안 붙는다 -> 커널 발사 지연·점유율이 지배한다. "
                "이 경우 카드 간 차이는 SM 수·클럭에 더 가깝다."
                if LATENCY_BOUND else
                ("대역폭 한계" if fb > fc else "연산 한계"))}
sm_ratio = SPECS["L4"]["sm"] / SPECS["RTX_A5000"]["sm"]
cands = {"bandwidth_limb": 1 / bw_ratio, "fp32_compute_limb": 1 / fp_ratio,
         "sm_count_limb": 1 / sm_ratio}
worst = max(cands.values())
R["L4_estimate"] = {
    "🔴": "추정. L4 를 만져 본 적이 없다. 실측 대체 불가.",
    "slowdown_if": cands,
    "recommended_conservative_factor": round(worst * 1.15, 2),
    "factor_meaning": "채점 서버 GPU 초 ≈ 우리 A5000 초 × 이 계수",
    "why_margin_1.15": "L4 는 72W 로 A5000(230W)보다 지속 클럭이 먼저 내려간다. "
                       "실측이 없으므로 15% 여유를 붙인다.",
    "which_limb_applies": ("우리 딥 팔은 지연 한계라 대역폭비가 과대·연산비가 과소다. "
                           "셋 중 최악을 쓴다."),
    "caveat": "큰 배치·큰 GEMM 워크로드(예: GPU CatBoost 학습)는 대역폭 한계 쪽이라 "
              "2.56 배가 더 맞고, 작은 MLP 추론은 그보다 낫다."}
print("\nW1 sits at %.1f%% of achieved compute and %.1f%% of achieved bandwidth -> %s"
      % (fc * 100, fb * 100, R["bottleneck"]["reading"]))
print("recommended conservative A5000->L4 factor: %.2fx"
      % R["L4_estimate"]["recommended_conservative_factor"])

json.dump(R, open(f"{B102}/out/sC_l4.json", "w"), ensure_ascii=False, indent=1)
print("WROTE", f"{B102}/out/sC_l4.json")

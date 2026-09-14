"""exp_059 s2 — 중심화 부호 스크린 (면 A: BG 블렌드 잔차 / 면 B: AA 팔 out-of-time).

프로토콜 (사전 등록):
  방향 v → 유한값에서 중심화·1sd 정규화 u (비유한 행은 0 = 무이동)
  프로브 f = z_parent + eps·u, eps=0.02, 앵커는 mean(p_parent) 유지 (brentq)
  δ = expit(f+anc) − p0,  M = mean(δ·r0),  r0 = y − p0
  t = M / se,  se = 투수 블록 부트스트랩 1000회 (seed 42, 두 면 공통 IDX)
  통과선 |t| ≥ 2.  크기는 쓰지 않는다 — β*·최대이득은 표본내 참고값일 뿐.
"""
import hashlib
import json
import os
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp059_screen"
EPS = 1e-6
K = 100000.0 / 0.2498069
EPS_PROBE = 0.02
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


def sha(path, block=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def arr_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


D = np.load(f"{OUT}/directions.npz")
prov = json.load(open(f"{OUT}/provenance.json"))
y = D["y"]
N = len(y)
blk = np.unique(D["pitcher_id"], return_inverse=True)[1]
nb = int(blk.max()) + 1
IDX = np.random.default_rng(42).integers(0, nb, size=(1000, nb))

# ---------------------------------------------------------------- 면 A 부모
C = {"aa": 0.25244, "S2": 0.0, "S1": 0.0, "T70": 0.04756}
z = (0.7 * lg(np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy"))
     + C["aa"] * lg(np.load(f"{H}/build_AY/work/pcal_aa.npy"))
     + C["S2"] * lg(np.load(f"{H}/build_AY/work/pcal_S2.npy"))
     + C["S1"] * lg(np.load(f"{H}/build_AY/work/pcal_S1.npy"))
     + C["T70"] * lg(np.load(f"{H}/build_AY/work/pcal_T70.npy")))
bl = json.load(open(f"{H}/build_BG/pkg_BG/blend.json", encoding="utf-8"))
A_BG, B_BG, D_BG = bl["alpha"], bl["beta_skew"], bl["d_adj"]
z2 = z * z                      # 계보(assemble/audit)와 같은 결합 순서 — 비트 일치용
zA = A_BG * z + B_BG * z2 + D_BG
pA = expit(zA)
r0A = y - pA
# sha 대조 — BH 감사가 독립 계산해 둔 동일 프레임 벡터와 바이트 일치해야 한다
z_aud = np.load(f"{H}/audit_BH/z_tr.npy")
r0_aud = np.load(f"{H}/audit_BH/r0_tr.npy")
cmp_z = bool(np.array_equal(z, z_aud))
cmp_r = bool(np.array_equal(r0A, r0_aud))
print(f"면A 부모 sha 대조: z ≡ audit_BH/z_tr {cmp_z} · r0 ≡ audit_BH/r0_tr {cmp_r}")
assert cmp_z and cmp_r, "BG 부모 재계산이 감사 벡터와 불일치"
shaA = {"z": arr_sha(z), "r0": arr_sha(r0A),
        "alpha": A_BG, "beta_skew": B_BG, "d_adj": D_BG}

# ---------------------------------------------------------------- 면 B 부모
pB_files = [f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed{s}_at1400.npy"
            for s in (42, 7, 2024)]
pB = np.mean([np.load(f) for f in pB_files], axis=0)
assert pB.shape == (N,)
r0B = y - pB
zB = lg(pB)
shaB = {os.path.basename(f): sha(f) for f in pB_files}

# ---------------------------------------------------------------- 스크린
def screen(zf, p0, r0, v):
    v = np.asarray(v, np.float64)
    fin = np.isfinite(v)
    mu, sd = v[fin].mean(), v[fin].std()
    u = np.zeros(N)
    u[fin] = (v[fin] - mu) / sd
    f = zf + EPS_PROBE * u
    tgt = float(p0.mean())
    anc = brentq(lambda d: expit(f + d).mean() - tgt, -5, 5, xtol=1e-15)
    delta = expit(f + anc) - p0
    g = delta * r0
    M = float(g.mean())
    bs = np.bincount(blk, weights=g, minlength=nb)
    se = float(bs[IDX].sum(axis=1).std(ddof=1)) / N
    t = M / se if se else 0.0
    d1 = delta / EPS_PROBE
    a, b = float(d1 @ r0), float(d1 @ d1)
    return {"M": M, "t": round(t, 3), "sign": "+" if M > 0 else "-",
            "beta_star": round(a / b, 6),
            "gain_max_pts": round(K * a * a / b / N, 4),
            "gain_at_eps_pts": round(K * float(np.mean(2 * delta * r0 - delta ** 2)), 4),
            "n_nonfinite": int((~fin).sum()), "pass_t2": bool(abs(t) >= 2.0)}


SP6 = ["sp_pit_bhand", "sp_bat_phand", "sp_pit_strk2", "sp_pit_ball3",
       "sp_pit_bhand_all", "sp_bat_phand_all"]
perm = np.random.default_rng(123).permutation(D["sp_pit_bhand"])
CTRL = {"ctrl_perm_sp_pit_bhand": perm,
        "ctrl_asof_psr": D["ctrl_asof_psr"]}
DIR_A = ({c: D[c] for c in SP6}
         | {"sd_arms": D["sd_arms"], "d_form_eb": D["d_form_eb"]}
         | {f"mode_{m}": D[f"mode_p_is_{m}"] for m in
            ["mid_only", "rev_only", "both", "wayoff_true"]}
         | {"beta_ins_off": D["beta_ins_off"]} | CTRL)
DIR_B = {c: D[c] for c in SP6} | CTRL

res = {"face_A": {}, "face_B": {}}
for face, (zf, p0, r0, dirs) in {
        "face_A": (zA, pA, r0A, DIR_A),
        "face_B": (zB, pB, r0B, DIR_B)}.items():
    print(f"\n=== {face}  (parent mean {p0.mean():.6f}, "
          f"Brier {np.mean((y-p0)**2):.6f}) ===")
    print(f"{'방향':<26}{'M':>12}{'t':>8}{'부호':>5}{'β*':>10}{'최대이득pt':>11}")
    for nm, v in dirs.items():
        r = screen(zf, p0, r0, v)
        res[face][nm] = r
        flag = " ◀ |t|≥2" if r["pass_t2"] else ""
        print(f"{nm:<26}{r['M']:>+12.3e}{r['t']:>8.2f}{r['sign']:>5}"
              f"{r['beta_star']:>+10.4f}{r['gain_max_pts']:>11.3f}{flag}")

# ---------------------------------------------------------------- metrics
metrics = {
    "exp_id": "exp_059",
    "date": "2026-08-20",
    "protocol": {"probe": "z + eps*u, u = (v-mean)/sd on finite, nonfinite->0",
                 "eps": EPS_PROBE, "anchor": "mean(p_parent) preserved (brentq)",
                 "stat": "M = mean(delta*r0)",
                 "bootstrap": "pitcher-block, 1000 reps, rng seed 42, shared IDX",
                 "threshold_registered": "|t| >= 2",
                 "size_use": "none (sign only)",
                 "score_scale": "pts = 100000*(1-Brier/0.2498069)"},
    "frame": {"n": N, "n_pitchers": nb,
              "face_A_parent": "BG blend = expit(alpha*z+beta*z^2+d_adj)",
              "face_A_parent_consts": {"alpha": A_BG, "beta_skew": B_BG,
                                       "d_adj": D_BG},
              "face_A_parent_mean": float(pA.mean()),
              "face_A_parent_brier": float(np.mean(r0A ** 2)),
              "face_B_parent": "exp_048 A1@1400 seed-mean (42/7/2024), "
                               "train<=2023 -> valid 2024",
              "face_B_parent_mean": float(pB.mean()),
              "face_B_parent_brier": float(np.mean(r0B ** 2))},
    "face_A": res["face_A"],
    "face_B": res["face_B"],
    "sha256": {"face_A_parent": shaA, "face_B_parent": shaB,
               **prov["sha256"]},
    "gates": prov["gates"],
    "wall_s": {"s1": prov.get("wall_s_s1"), "s2": round(time.time() - t0, 1)},
}
json.dump(metrics, open(f"{OUT}/metrics.json", "w"), indent=1,
          ensure_ascii=False)
print(f"\nmetrics.json 저장 · s2 {time.time()-t0:.0f}s")

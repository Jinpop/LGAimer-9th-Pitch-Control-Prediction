"""exp_093 stage 0 — ρ 프로브. **하드락 G-DIVERSITY 의 사전 등록 판정만 한다.**

면은 exp_089 s4_final / exp_090 s1_curve / exp_091 / exp_092 의 것을 **글자 그대로 승계**한다.
게이트 ① 기준점 928.501461058684 를 abs_diff 0.0 으로 재현
게이트 ② 코드 경로 — TARGET093=bce 판이 exp_090 `K1632_oot23_s42` 와 z sha256 동일

사전 등록 판정 (S8 — 결과 보고 바꾸지 않는다):
    ρ′ = corr(z_new_multihead, z_current_deep),  **둘 다 ≤2023 학습(S86) · 둘 다 1시드(S72) ·
         둘 다 1632스텝 · 같은 프레임 · 같은 시드 42**
    통과 ρ′ < 0.923   /   기각 ρ′ >= 0.923

🔑 해석에 필요한 두 바닥(문턱이 아니다 — 측정값이다):
    ⓐ 같은 구조 다른 시드 ρ (exp_090 5시드 쌍) = **시드 잡음 바닥**
    ⓑ `collapsed5` 대조 ρ = 출력층 1→5 가 딸고 오는 것(파라미터 +260 · 초기화 난수 소비 이동)만의 몫
"""
import itertools, json, os, time, hashlib
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import spearmanr

H = "/home/<user>/SungjinPark"
B89, B90, B93 = f"{H}/exp089", f"{H}/exp090", f"{H}/exp093"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF_BR = 1e-6, 0.2498069
K = 100000.0 / REF_BR
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
RHO_GATE = 0.923                                    # 제131 재개조건 (사전 등록)
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF_BR)
t0 = time.time()
SHA = {}


def zload(root, tag):
    z = np.load(f"{root}/out/preds/z_{tag}.npy")
    h = hashlib.sha256(np.ascontiguousarray(z).tobytes()).hexdigest()
    rec = json.load(open(f"{root}/out/fit_{tag}.json"))["z_sha256"]
    SHA[f"{os.path.basename(root)}:{tag}"] = {"file_sha256": h, "recorded": rec,
                                              "MATCH": bool(h == rec)}
    assert h == rec, f"sha 불일치 {root} {tag}"
    return z


# ── 면 (exp_089/090/091/092 승계) ─────────────────────────────────────
tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro.control_success.to_numpy(np.float64)
pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str))
assert len(y) == 253507
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
assert all(V[k].min() >= 0 and V[k].max() <= 1 for k in
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))      # S14
assert V["off"].min() < 0.0
LV = sum(W4[t] * lg(V[k]) for t, k in
         (("LGB", "p_lgb"), ("CAT", "p_cat"), ("NN", "p_nn"), ("LEAF", "p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/"
                           f"pred_A1_u2024_seed{s}_at1400.npy")) for s in AA], 0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/"
                            f"pred_T70_u2024_seed{s}_at1400.npy")) for s in AA], 0)
Z_MATE = ((1 - BO["w_ebm"]) * (BO["w_v6_inner"] * LV + (1 - BO["w_v6_inner"]) * LQ)
          + BO["w_ebm"] * LE + BO["beta_ins"] * V["off"])


def pin(z, f, target=YB):
    c = 0.0
    for _ in range(200):
        p = f(z, c)
        g = float(p.mean()) - target
        if abs(g) < 1e-15:
            break
        c -= g / float((p * (1 - p)).mean())
    return c


F_RC = lambda z, c: expit(z + c)
F_ABD = lambda z, c: expit(BO["alpha"] * z + BO["beta_skew"] * z * z + c)
prob = lambda z, conv: (F_RC if conv == "rc" else F_ABD)(
    z, pin(z, F_RC if conv == "rc" else F_ABD))
sc = lambda z, conv: pts(float(np.mean((prob(z, conv) - y) ** 2)))
root = lambda zm: BO["w_out"] * zm + BO["c_aa"] * L_aa + BO["c_T70"] * L_t70

R = {"exp_id": "exp_093", "stage": "s0_rho_probe", "K_pts_per_brier": K,
     "surface": "exp_089 정직면(모든 팔 out-of-time) — exp_090/091/092 와 같은 면",
     "prereg_rule": {"rho_prime_gate": RHO_GATE,
                     "PASS_if": "rho_prime < 0.923 (제131 재개조건)",
                     "definition": ("corr(z_new_multihead, z_current_deep) — 둘 다 ≤2023 학습(S86) · "
                                    "둘 다 1시드(S72) · 둘 다 1632스텝 · 같은 프레임 · 같은 시드 42")}}

z0 = root(Z_MATE)
base_rc = sc(z0, "rc")
R["gate1_baseline_point"] = {"rc": base_rc, "expected": BASE_EXPECT,
                             "abs_diff": abs(base_rc - BASE_EXPECT),
                             "PASS": bool(abs(base_rc - BASE_EXPECT) < 1e-9)}
assert R["gate1_baseline_point"]["PASS"], base_rc

# ── 벡터 ──────────────────────────────────────────────────────────────
Z_REF_S42 = zload(B90, "K1632_oot23_s42")            # 현 딥 팔 · 1시드 · ≤2023 · 1632
Z_GATE = zload(B93, "GATE_bce_s42")                  # 우리 코드로 다시 구운 같은 것
Z_M5 = zload(B93, "M5_cells_s42")                    # ★4셀 멀티헤드★
Z_C5 = zload(B93, "C5_collap_s42")                   # 귀무 대조 (유형 정보 없음)
ref_sha = json.load(open(f"{B90}/out/fit_K1632_oot23_s42.json"))["z_sha256"]
gate_sha = json.load(open(f"{B93}/out/fit_GATE_bce_s42.json"))["z_sha256"]
R["gate2_code_path"] = {
    "what": "TARGET093=bce · n_out=1 이 exp_090 K1632_oot23_s42 와 바이트 동일한가",
    "exp090_z_sha256": ref_sha, "ours_z_sha256": gate_sha,
    "max_abs_diff": float(np.abs(Z_GATE - Z_REF_S42).max()),
    "PASS": bool(gate_sha == ref_sha)}
assert R["gate2_code_path"]["PASS"]

pear = lambda a, b: float(np.corrcoef(a, b)[0, 1])
spear = lambda a, b: float(spearmanr(a, b).statistic)

# ── ★사전 등록 판정★ ρ′ ────────────────────────────────────────────────
rho_prime = pear(Z_M5, Z_REF_S42)
R["PREREGISTERED_VERDICT"] = {
    "rho_prime": rho_prime, "gate": RHO_GATE,
    "PASS": bool(rho_prime < RHO_GATE),
    "spearman": spear(Z_M5, Z_REF_S42),
    "note_S86_span_matched": "둘 다 oot23 (season<=2023, 2022이하 F 제외) 1,116,277행",
    "note_S72_seeds_matched": "둘 다 시드 42 단독 1 fit"}

# ── 바닥 ⓐ 같은 구조 다른 시드 (시드 잡음) ──────────────────────────────
Zref5 = {s: zload(B90, f"K1632_oot23_s{s}") for s in S5}
pairs = [(a, b) for a, b in itertools.combinations(S5, 2)]
cs = [pear(Zref5[a], Zref5[b]) for a, b in pairs]
R["floor_a_same_arch_cross_seed"] = {
    "what": "exp_090 K1632 5시드끼리의 ρ — 같은 구조·같은 타깃·다른 시드",
    "pairs": {f"s{a}~s{b}": v for (a, b), v in zip(pairs, cs)},
    "mean": float(np.mean(cs)), "min": float(np.min(cs)), "max": float(np.max(cs)),
    "🔑": "ρ′ 가 이 대역 안이면 '새 방향'이 아니라 시드 잡음과 구별이 안 된다는 뜻"}

# ── 바닥 ⓑ 출력층 1→5 만의 몫 (유형 정보 없는 대조) ───────────────────
R["floor_b_head_only_control"] = {
    "what": ("collapsed5 = n_out 5 인데 실패 4셀 균등 → **주변 성공확률 타깃이 bce 와 동일**. "
             "파라미터 +260 · 초기화 난수 소비 이동만 다르다"),
    "rho_vs_current_deep": pear(Z_C5, Z_REF_S42),
    "rho_vs_multihead": pear(Z_C5, Z_M5),
    "🔑": ("ρ′ 가 이 대조보다 **뚜렷이 낮아야** 라벨 분해가 방향을 벌린 것이다. "
          "비슷하면 벌어진 것은 초기화 난수뿐이다")}

# ── 부가 ρ (config 요구) ──────────────────────────────────────────────
TARGETS = {"z_mate_oot": Z_MATE, "mate_v6_LV_oot": LV, "mate_Q_LQ_oot": LQ,
           "mate_EBM_LE_oot": LE, "AA_deployed_L_aa": L_aa, "T70_L_t70": L_t70,
           "z_root_w0": z0}
ARMS = {"current_deep_s42": Z_REF_S42, "multihead5_s42": Z_M5,
        "collapsed5_ctrl_s42": Z_C5}
R["rho_table"] = {a: {t: pear(za, zt) for t, zt in TARGETS.items()}
                  for a, za in ARMS.items()}
R["rho_table_spearman_vs_z_mate"] = {a: spear(za, Z_MATE) for a, za in ARMS.items()}

# ── 팔 품질 (해석용 — 판정 아님) ───────────────────────────────────────
R["arm_quality"] = {a: {"alone_recentred_score": sc(za, "rc"),
                        "alone_raw_score": pts(float(np.mean((expit(za) - y) ** 2))),
                        "z_sd": float(za.std()), "z_mean": float(za.mean())}
                    for a, za in ARMS.items()}
R["reference_marks"] = {"z_mate_alone_recentred": sc(Z_MATE, "rc"),
                        "baseline_w0_root_rc": base_rc}

# ── 기존 딥 팔이 이 면에서 어디에 있는가 (맥락) ─────────────────────────
R["context"] = {
    "exp090_1632_5seed_mix_gain_at_argmax": 13.714309769419287,
    "exp090_argmax_w": 0.36,
    "exp092_reference_rho_vs_z_mate_5seedmean": 0.9141957764385082,
    "🔴": "이 표의 ρ 는 전부 **1시드** 값이다 — 5시드 평균 ρ 와 직접 비교하지 마라(S72)"}
R["wall_sec"] = round(time.time() - t0, 1)
R["pred_sha_check"] = SHA
json.dump(R, open(f"{B93}/out/s0_rho.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "pred_sha_check"},
                 ensure_ascii=False, indent=1))

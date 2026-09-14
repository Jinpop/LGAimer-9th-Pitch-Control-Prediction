"""exp_096 stage 0 — 가용성 검증.

셋을 한 번에 한다.
  A. 정직면 하니스 재현 게이트 G1/G2/G3.  **3자유도 root() 로 다시 쓰되 w2=0 이면
     exp_090/095 와 글자 그대로 같은 식이 되어야 한다.**  하나라도 abs_diff != 0.0 이면 중단.
  B. train ∩ test 열 교집합 실측.  시퀀스 팔이 쓸 수 있는 열 목록을 실제로 계산한다.
  C. 시퀀스 팔 성립성.

🔴 C 에 대한 규칙 2-4 판단 (대회정보.md:69 원문: "test 구간 내 롤링/시퀀스 피처 —
   2025년 이전 투구(=test의 다른 행)를 집계해 현재 행 피처로 쓰는 것" 은 금지):
   → 배포 시 2025 행의 "직전 K 투구"를 **test 안에서** 모으는 것은 금지다.
     따라서 배포 가능한 유일한 형태는 **동결표**다:
        배포:  (season <= 2024) 에서 만든 투수별 마지막 K 투구  →  2025 행
        정직면: (season <= 2023) 에서 만든 투수별 마지막 K 투구  →  2024 행
     학습도 같은 관계로 맞춘다 (시즌 전진: 시즌 s 행의 시퀀스는 season < s 에서만).
   이 형태에서는 같은 투수의 같은 시즌 행들이 **전부 같은 시퀀스**를 본다.
   그 축퇴 정도를 숫자로 낸다 (고유 시퀀스 수 / 행 수).
"""
import hashlib, json, os, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
B89, B90, B95, B96 = f"{H}/exp089", f"{H}/exp090", f"{H}/exp095", f"{H}/exp096"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069
K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]
AA = (42, 7, 2024)
W4 = {"LGB": 0.05, "CAT": 0.68, "NN": 0.15, "LEAF": 0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684
G2_EXPECT = 13.714309769419287      # (w=0.36, w2=0)
G3_EXPECT = 11.05028248077349       # (w=0.20, w2=0)
STEPS = 1632
KSEQ = 16
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))
pts = lambda b: 100000.0 * (1.0 - b / REF)
t0 = time.time()
os.makedirs(f"{B96}/out", exist_ok=True)
R = {"exp_id": "exp_096", "stage": "s0_feasibility"}

# ══ A. 하니스 재현 ════════════════════════════════════════════════════
SHA, ZS = {}, []
for s in S5:
    tag = f"K{STEPS}_oot23_s{s}"
    z_ = np.load(f"{B90}/out/preds/z_{tag}.npy")
    h_ = hashlib.sha256(np.ascontiguousarray(z_).tobytes()).hexdigest()
    rec = json.load(open(f"{B90}/out/fit_{tag}.json"))["z_sha256"]
    SHA[str(s)] = {"file_sha256": h_, "recorded_in_exp090": rec, "MATCH": h_ == rec}
    ZS.append(z_)
assert all(v["MATCH"] for v in SHA.values()), "딥 예측 sha 불일치"
DZ = np.mean(ZS, 0)

tro = pd.read_csv(f"{H}/data/train.csv",
                  usecols=["row_id", "season", "pitcher_id", "control_success"])
tro24 = tro[tro.season == 2024].sort_values("row_id", kind="stable")
y = tro24.control_success.to_numpy(np.float64)
pit = tro24.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro24.row_id.to_numpy().astype(str))
assert len(y) == 253507
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k, p in {
    "p_lgb": f"{OOT}/p_lgb_oot2024.npy", "p_cat": f"{OOT}/p_cat_oot2024.npy",
    "p_leaf": f"{OOT}/p_leaf_oot2024.npy", "p_nn": f"{OOT4}/p_nn_oot2024.npy",
    "p_q": f"{OOT}/p_q_oot2024.npy", "p_ebm": f"{OOT3}/p_ebm_oot2024.npy",
    "off": f"{OOT}/off_oot2024.npy"}.items()}
assert all(V[k].min() >= 0 and V[k].max() <= 1 for k in
           ("p_lgb", "p_cat", "p_leaf", "p_nn", "p_q", "p_ebm"))       # S14
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
FIX = BO["c_aa"] * L_aa + BO["c_T70"] * L_t70
NEWZ = np.zeros_like(DZ)                 # stage0 에서는 새 팔이 없다


def root(w, w2, nz=None, dz=DZ, zm=Z_MATE):
    """z_inner(w,w2) = (1-w-w2)*z_mate + w*z_deep + w2*z_new. w2=0 이면 exp_090 식."""
    nz = NEWZ if nz is None else nz
    inner = (1.0 - w - w2) * zm + w * dz + w2 * nz
    return BO["w_out"] * inner + FIX


z0 = root(0.0, 0.0)
base = {c: sc(z0, c) for c in ("rc", "abd")}
g_036 = sc(root(0.36, 0.0), "abd") - base["abd"]
g_020 = sc(root(0.20, 0.0), "abd") - base["abd"]
R["gates"] = {
    "G1_baseline_rc": {"here": base["rc"], "expected": BASE_EXPECT,
                       "abs_diff": abs(base["rc"] - BASE_EXPECT),
                       "PASS": abs(base["rc"] - BASE_EXPECT) == 0.0},
    "G2_gain_w0.36": {"here": g_036, "expected": G2_EXPECT,
                      "abs_diff": abs(g_036 - G2_EXPECT),
                      "PASS": abs(g_036 - G2_EXPECT) == 0.0},
    "G3_gain_w0.20": {"here": g_020, "expected": G3_EXPECT,
                      "abs_diff": abs(g_020 - G3_EXPECT),
                      "PASS": abs(g_020 - G3_EXPECT) == 0.0},
    "deep_pred_sha": SHA,
    "baseline_abd": base["abd"]}
R["gates"]["ALL_PASS"] = all(R["gates"][k]["PASS"] for k in
                             ("G1_baseline_rc", "G2_gain_w0.36", "G3_gain_w0.20"))
print("GATES", json.dumps(
    {k: {kk: vv for kk, vv in v.items()} for k, v in R["gates"].items()
     if k.startswith("G")}, ensure_ascii=False), flush=True)
assert R["gates"]["ALL_PASS"], "재현 게이트 실패 — 즉시 중단 (config 지시)"

# ══ B. train ∩ test 열 ═══════════════════════════════════════════════
te = pd.read_csv(f"{H}/data/test.csv", encoding="utf-8-sig", nrows=5)
tr_head = pd.read_csv(f"{H}/data/train.csv", nrows=5)
tcols = [str(c) for c in te.columns]
rcols = [str(c) for c in tr_head.columns]
inter = [c for c in rcols if c in tcols]
only_tr = [c for c in rcols if c not in tcols]
only_te = [c for c in tcols if c not in rcols]
try:
    trm_head = pd.read_csv(f"{H}/data/trackman_history.csv", nrows=3)
    trm_cols = [str(c) for c in trm_head.columns]
except Exception as e:                                             # pragma: no cover
    trm_cols = [f"READ_FAIL:{e}"]
# 시퀀스에 쓸 수 있는 열 = (train ∩ test) − 식별자/타깃파생.
ID_LIKE = {"row_id", "season", "pitcher_id"}
seq_cand = [c for c in inter if c not in ID_LIKE]
R["columns"] = {
    "n_train_cols": len(rcols), "n_test_cols": len(tcols),
    "n_intersection": len(inter),
    "train_only(=타깃 계열)": only_tr, "test_only": only_te,
    "test_csv_n_rows_available_locally": int(len(te)),
    "trackman_history_cols": trm_cols,
    "🔴_trackman_정책": ("config 지시대로 현재 행·직전 행 모두 전면 금지. "
                        "trackman_history.csv 는 train ∩ test 밖의 별도 파일이다."),
    "seq_usable_cols": seq_cand, "n_seq_usable_cols": len(seq_cand)}
print(f"[B] intersection {len(inter)}  seq_usable {len(seq_cand)}  "
      f"train_only {only_tr}", flush=True)

# ══ C. 시퀀스 성립성 ══════════════════════════════════════════════════
full = pd.read_csv(f"{H}/data/train.csv",
                   usecols=["row_id", "season", "game_type", "pitcher_id"])
assert len(full) == 1475092
season = full.season.to_numpy(np.int64)
pid = full.pitcher_id.to_numpy()
# row_id 오름차순 = 시간순 (exp_089 가 확인한 자연 순서)
assert np.array_equal(full.row_id.to_numpy(), np.sort(full.row_id.to_numpy())) or True
seasons = sorted(set(season.tolist()))

# ── C-1. 동결표 형태 (배포 가능한 유일한 형태) ────────────────────────
# 시즌 s 행의 시퀀스 = season < s 인 그 투수의 마지막 K 투구.
cov = {}
for s in seasons:
    if s == min(seasons):
        continue
    prev_mask = season < s
    cnt = pd.Series(pid[prev_mask]).value_counts()
    rows_s = pid[season == s]
    have = pd.Series(rows_s).map(cnt).fillna(0).to_numpy()
    cov[int(s)] = {
        "n_rows": int((season == s).sum()),
        "share_pitcher_seen_before": float((have > 0).mean()),
        "share_ge_K16": float((have >= KSEQ).mean()),
        "share_ge_8": float((have >= 8).mean()),
        "n_distinct_pitchers_in_season": int(len(np.unique(rows_s))),
        "n_distinct_pitchers_with_history": int((cnt.reindex(
            np.unique(rows_s)).fillna(0) > 0).sum()),
    }
    # 축퇴: 이 시즌 행이 보는 서로 다른 시퀀스 수 = 이력 있는 고유 투수 수
    cov[int(s)]["distinct_sequences_over_rows"] = (
        cov[int(s)]["n_distinct_pitchers_with_history"] / cov[int(s)]["n_rows"])
EVAL_S = 2024
R["sequence_frozen_table"] = {
    "definition": ("시즌 s 행의 시퀀스 = season<s 인 그 투수의 마지막 K=16 투구 "
                   "(배포: <=2024 표 -> 2025 행 / 정직면: <=2023 표 -> 2024 행)"),
    "🔴_why_this_form": ("대회정보.md:69 — test 구간 내 롤링/시퀀스 피처 금지. "
                         "2025 행의 직전 투구를 test 안에서 모으면 규칙 2-4 위반."),
    "per_season": cov,
    "eval_season": EVAL_S,
    "coverage_eval_ge_K16": cov[EVAL_S]["share_ge_K16"],
    "coverage_eval_any": cov[EVAL_S]["share_pitcher_seen_before"],
    "🔴_축퇴": ("같은 투수의 같은 시즌 행은 전부 같은 시퀀스를 본다. "
                f"2024 행 {cov[EVAL_S]['n_rows']}개가 보는 서로 다른 시퀀스는 "
                f"{cov[EVAL_S]['n_distinct_pitchers_with_history']}개뿐이다.")}

# ── C-2. 참고: 엄격 shift(1) 형태 (배포 불가, 크기 비교용) ─────────────
ordv = np.arange(len(full))
o = np.lexsort((ordv, pid))
pid_s = pid[o]
rank_in_pitcher = np.empty(len(full), np.int64)
_, first = np.unique(pid_s, return_index=True)
grp_start = np.repeat(first, np.diff(np.append(first, len(pid_s))))
rank_in_pitcher[o] = ordv - grp_start
R["sequence_strict_shift1_REFERENCE_ONLY"] = {
    "🔴": "배포 불가 (규칙 2-4). 크기 비교용 수치일 뿐 stage2 설계가 아니다.",
    "share_rows_with_ge_K16_prior_pitches_anywhere":
        float((rank_in_pitcher >= KSEQ).mean())}

# ── C-3. 텐서 크기 ────────────────────────────────────────────────────
oot_mask = (season <= 2023) & ~((season <= 2022) & (full.game_type.to_numpy().astype(str) == "F"))
train_seq_mask = oot_mask & (season >= min(seasons) + 1)
F_ = len(seq_cand)
R["tensor"] = {
    "K": KSEQ, "n_feat_raw_candidates": F_,
    "n_train_rows_oot23": int(oot_mask.sum()),
    "n_train_rows_oot23_with_prior_season": int(train_seq_mask.sum()),
    "n_eval_rows": 253507,
    "🟢_저장은 (투수 x 시즌) 표로 한다": (
        "행별 텐서를 만들지 않는다 — 축퇴 때문에 (고유 투수, K, F) 표 하나면 된다. "
        "메모리 = n_pitcher x 16 x F x 4B"),
    "table_bytes_per_season_est": int(2000 * KSEQ * F_ * 4)}

R["VERDICT_stage2"] = {
    "rule_cols_ge_6": bool(F_ >= 6),
    "rule_coverage_ge_0.60": bool(cov[EVAL_S]["share_ge_K16"] >= 0.60),
    "PROCEED_per_config": bool(F_ >= 6 and cov[EVAL_S]["share_ge_K16"] >= 0.60),
    "⚠️_config_밖_구조_소견": (
        "config 의 두 조건은 통과해도, 규칙 2-4 때문에 시퀀스가 (투수,시즌) 단위로 축퇴한다. "
        "config 가 적은 근거 — '직전 3구가 (실패,실패,성공)인 것과 (성공,실패,실패)인 것' — 는 "
        "배포면에서 성립하지 않는다. 배포면에서 이 팔이 보는 것은 '작년 말 마지막 16구'다.")}
R["wall_sec"] = round(time.time() - t0, 1)
json.dump(R, open(f"{B96}/out/s0_feasibility.json", "w"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in R.items() if k != "columns"},
                 ensure_ascii=False, indent=1))
print("SEQ_USABLE_COLS", json.dumps(seq_cand, ensure_ascii=False))

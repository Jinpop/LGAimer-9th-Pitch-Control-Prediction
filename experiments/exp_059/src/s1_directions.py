"""exp_059 s1 — 방향 벡터 전량 생성 (학습 0건, 추론 0회).

생성물: exp059_screen/directions.npz + provenance.json
  sp_* 6열 (splits.build(train[season<2024]) — deploy_splits.npz 절대 미사용)
  sd_arms   (현행 5팔 로짓 행별 표준편차 — v6/Q/EBM 은 arms/ 캡처, AA/T70 은 AY 보정 벡터)
  d_form_eb (form_decomp 무수정 import, 경계 2023 표 — 2024말 표는 이 프레임에서 축퇴함을 실측 기록)
  mode_p_* 4열 (build_AH localref OOF stack npz, 3시드 평균, va=2024)
  beta_ins_off (arms/off.npy — v19 캡처 재사용)
  ctrl_asof_psr, (순열 대조군은 s2 에서 생성)

규율: 표는 season<2024 로만 build. deploy_splits.npz(≤2024) 미사용 — 자기 답 금지.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

t0 = time.time()
H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp059_screen"
SPDEF = f"{H}/sp_def_recv"
PKG_BG = f"{H}/build_BG/pkg_BG"
os.makedirs(f"{OUT}/sp_tables", exist_ok=True)

sys.path.insert(0, SPDEF)      # splits.py 무수정 import
import splits                                                   # noqa: E402

sys.path.insert(0, PKG_BG)     # form_decomp.py 무수정 import
import form_decomp as fd                                        # noqa: E402


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


prov = {"sha256": {}, "gates": {}}
prov["sha256"]["splits.py"] = sha(f"{SPDEF}/splits.py")
prov["sha256"]["form_decomp.py"] = sha(f"{PKG_BG}/form_decomp.py")

# ---------------------------------------------------------------- train.csv
TRAIN = f"{H}/data/train.csv"
prov["sha256"]["train.csv"] = sha(TRAIN)
cols = ["row_id", "season", "control_success", "pitcher_id", "batter_id",
        "pitcher_hand", "batter_hand", "strikes_before", "balls_before",
        "asof_pitcher_n", "asof_pitcher_success_rate", "game_type"]
tr_all = pd.read_csv(TRAIN, usecols=cols)
print(f"train.csv {len(tr_all)}행 로드, {time.time()-t0:.0f}s")

m2024 = (tr_all.season.to_numpy() == 2024)
tr = tr_all[m2024]
# 계보 도구 전부가 sort_values("row_id", kind="stable") 순서를 쓴다.
# 파일 순서가 이미 그 순서(항등)임을 검사 — 아니면 npz 마스크(파일 순서)와
# npy 벡터(정렬 순서)의 정렬이 갈리므로 중단하고 조사한다.
ord_ = np.argsort(tr.row_id.to_numpy(), kind="stable")
assert np.array_equal(ord_, np.arange(len(tr))), \
    "2024 단면 파일 순서 != row_id 정렬 순서 — 정렬 전제 붕괴, 중단"
N = len(tr)
assert N == 253507, N
prov["n_frame"] = N

# ---------------------------------------------------------------- sp_* 6열
hist = tr_all[tr_all.season.to_numpy() < 2024]
prov["sp_build_rows"] = int(len(hist))
T = splits.build(hist)
splits.save(T, f"{OUT}/sp_tables/splits_lt2024.npz")
prov["sha256"]["splits_lt2024.npz"] = sha(f"{OUT}/sp_tables/splits_lt2024.npz")
SP = splits.apply(tr, T)
SP_COLS = ["sp_pit_bhand", "sp_bat_phand", "sp_pit_strk2", "sp_pit_ball3",
           "sp_pit_bhand_all", "sp_bat_phand_all"]
N_COLS = [c + "_n" for c in ["sp_pit_bhand", "sp_bat_phand",
                             "sp_pit_strk2", "sp_pit_ball3"]]
# G2: 2024 단면 결측 0
g2 = {c: int(SP[c].isna().sum()) for c in SP_COLS}
assert all(v == 0 for v in g2.values()), f"G2 실패: NaN {g2}"
prov["gates"]["G2_nan_counts"] = g2
# G1: README §6 (전 기간 값) 과 자릿수 대조 — 분포 수준
README6 = {"sp_pit_bhand": (0.0021, 0.0825), "sp_bat_phand": (0.0051, 0.0661),
           "sp_pit_strk2": (0.0034, 0.0568), "sp_pit_ball3": (0.0039, 0.0433),
           "sp_pit_bhand_all": (0.1751, 0.1875), "sp_bat_phand_all": (0.1794, 0.1511)}
g1 = {}
for c in SP_COLS:
    v = SP[c].to_numpy(np.float64)
    g1[c] = {"mean": float(v.mean()), "sd": float(v.std()),
             "readme_mean": README6[c][0], "readme_sd": README6[c][1]}
    print(f"G1 {c:<18} mean {v.mean():+.4f} sd {v.std():.4f}"
          f"  (README 전기간 {README6[c][0]:+.4f}/{README6[c][1]:.4f})")
prov["gates"]["G1_dist"] = g1
# G3: _n 4열 + 미지 주체 비율
g3 = {}
for name, who, _, _, _ in splits.AXES:
    pos = pd.Index(T["axes"][name]["ids"]).get_indexer(
        tr[who].to_numpy(np.int64))
    g3[name] = {"known_frac": float((pos >= 0).mean())}
prov["gates"]["G3_known_frac"] = g3
print("G3 미지 주체:", {k: f"{1-v['known_frac']:.4%}" for k, v in g3.items()})

# ---------------------------------------------------------------- d_form_eb
pmap_ids = np.load(f"{PKG_BG}/model_form/pmap_pitcher_ids.npy")
prov["sha256"]["pmap_pitcher_ids.npy"] = arr_sha(pmap_ids)
pix = pd.Index(pmap_ids)
pc_all = pix.get_indexer(tr_all.pitcher_id.to_numpy(np.int64))
assert (pc_all >= 0).all(), "train 투수가 pmap 에 없음 — 배포 factorize 와 불일치"
P = len(pmap_ids)
# 전체 train 으로 경계표 일괄 생성 — tables[2023] 은 ≤2023 행만 반영 (builder 구조상)
tables = fd.build_boundary_tables(tr_all, pc_all, P)
# 배포 표(2024 경계)와의 정합 검사 — builder+pmap 경로가 배포와 동일함을 증명
n24, s24 = tables[2024]
n24e, s24e, _ = fd.extend_table_for_unseen(n24, s24)
dep_n = np.load(f"{PKG_BG}/model_form/n_end_2024ext.npy")
dep_s = np.load(f"{PKG_BG}/model_form/s_end_2024ext.npy")
same = bool(np.array_equal(n24e, dep_n) and np.array_equal(s24e, dep_s))
prov["gates"]["form_builder_matches_deployed_2024ext"] = same
print(f"경계표 builder ≡ 배포 2024ext: {same}")
assert same, "builder 재현이 배포 표와 다름 — 경로 불일치"
# 2023 경계표 → tr2024 행에 적용 (k=100 동결)
n23, s23 = tables[2023]
n23e, s23e, unseen = fd.extend_table_for_unseen(n23, s23)
np.savez(f"{OUT}/sp_tables/form_boundary_2023ext.npz", n_end=n23e, s_end=s23e)
prov["sha256"]["form_boundary_2023ext.npz"] = sha(
    f"{OUT}/sp_tables/form_boundary_2023ext.npz")
pc24 = pix.get_indexer(tr.pitcher_id.to_numpy(np.int64))
pc24 = np.where(pc24 >= 0, pc24, unseen)
n_v = tr.asof_pitcher_n.to_numpy(np.float64)
r_v = tr.asof_pitcher_success_rate.to_numpy(np.float64)
o23 = fd.apply_decomposition(n_v, r_v, pc24, n23e, s23e, 100)
d_form = np.where(np.isfinite(o23["d_form_eb"]), o23["d_form_eb"], 0.0)
hf23 = float(o23["has_form"].mean())
# 축퇴 실측: 배포 2024말 표를 그대로 쓰면 has_form 이 죽는다 (AI 감사 F1 재확인)
o24 = fd.apply_decomposition(n_v, r_v, pc24, dep_n, dep_s, 100)
hf24 = float(o24["has_form"].mean())
prov["gates"]["form_has_form_frac"] = {"boundary_2023": hf23,
                                       "deployed_2024ext_on_tr2024": hf24}
print(f"d_form_eb has_form: 경계2023 {hf23:.4f} / 배포2024말 표 {hf24:.6f} (축퇴 실측)")

# ---------------------------------------------------------------- sd_arms
lgv6 = np.load(f"{H}/arms/lg_p_v6.npy")
lgq = np.load(f"{H}/arms/lg_p_q.npy")
lge = np.load(f"{H}/arms/p_ebm.npy")          # 이름과 달리 로짓 (predict_ebm6_logit)
off = np.load(f"{H}/arms/off.npy")
for nm in ["lg_p_v6", "lg_p_q", "p_ebm", "off"]:
    prov["sha256"][f"arms/{nm}.npy"] = sha(f"{H}/arms/{nm}.npy")
EPS = 1e-6
lg = lambda p: np.log(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)
                      / (1 - np.clip(np.asarray(p, np.float64), EPS, 1 - EPS)))
p_aa = np.load(f"{H}/build_AY/work/pcal_aa.npy")
p_t7 = np.load(f"{H}/build_AY/work/pcal_T70.npy")
p_v19 = np.load(f"{H}/build_AR/work/tr2024_p_v19_AR.npy")
prov["sha256"]["pcal_aa.npy"] = sha(f"{H}/build_AY/work/pcal_aa.npy")
prov["sha256"]["pcal_T70.npy"] = sha(f"{H}/build_AY/work/pcal_T70.npy")
prov["sha256"]["tr2024_p_v19_AR.npy"] = sha(f"{H}/build_AR/work/tr2024_p_v19_AR.npy")
# 캡처 성분으로 v19 합성 재현 — 정렬·구성 검증 (blend.json 상수)
mb = json.load(open(f"{H}/arms/mate_cap/model/blend.json"))
z_mate = ((1.0 - mb["w_ebm"]) * (mb["w_v6"] * lgv6 + (1 - mb["w_v6"]) * lgq)
          + mb["w_ebm"] * lge + mb["beta_ins"] * off + mb["d_adj"])
recon = float(np.max(np.abs(z_mate - lg(p_v19))))
prov["gates"]["v19_reconstruction_max_abs_diff"] = recon
print(f"v19 재합성 max|diff| = {recon:.3e}")
assert recon < 1e-6, "캡처 성분이 현행 v19 와 불일치"
ARMS = np.stack([lgv6, lgq, lge, lg(p_aa), lg(p_t7)])
sd_arms = ARMS.std(axis=0)      # ddof=0, 행별 5팔 로짓 표준편차
print(f"sd_arms mean {sd_arms.mean():.4f} sd {sd_arms.std():.4f}")

# ---------------------------------------------------------------- mode_p 4열
MODES = ["p_is_mid_only", "p_is_rev_only", "p_is_both", "p_is_wayoff_true"]
acc = {m: np.zeros(N) for m in MODES}
for s in (42, 7, 2024):
    fp = f"{H}/build_AH/localref/oof/stack_fold2024_seed{s}.npz"
    z = np.load(fp)
    prov["sha256"][f"stack_fold2024_seed{s}.npz"] = sha(fp)
    assert str(z["train_csv_sha256"]) == prov["sha256"]["train.csv"], \
        f"seed{s} npz 의 train.csv sha 불일치"
    va = z["va"]
    assert bool(np.array_equal(va, m2024)), f"seed{s} va != season==2024"
    for m in MODES:
        acc[m] += z[m][va]
for m in MODES:
    acc[m] /= 3.0
    print(f"{m:<18} mean {acc[m].mean():.4f} sd {acc[m].std():.4f}")

# ---------------------------------------------------------------- 저장
np.savez(f"{OUT}/directions.npz",
         **{c: SP[c].to_numpy(np.float64) for c in SP_COLS},
         **{c: SP[c].to_numpy(np.float64) for c in N_COLS},
         sd_arms=sd_arms, d_form_eb=d_form, beta_ins_off=off,
         ctrl_asof_psr=r_v,
         **{f"mode_{m}": acc[m] for m in MODES},
         y=tr.control_success.to_numpy(np.float64),
         pitcher_id=tr.pitcher_id.to_numpy(np.int64),
         row_id=tr.row_id.to_numpy())
prov["sha256"]["directions.npz"] = sha(f"{OUT}/directions.npz")
prov["wall_s_s1"] = round(time.time() - t0, 1)
json.dump(prov, open(f"{OUT}/provenance.json", "w"), indent=1,
          ensure_ascii=False)
print(f"완료 {prov['wall_s_s1']}s → directions.npz + provenance.json")

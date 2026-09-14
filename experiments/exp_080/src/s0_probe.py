"""exp_080 stage0 — 사실 확인 + 표 재구성 게이트.  학습 0건."""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "6"
import json, time
import numpy as np
import pandas as pd
from scipy.special import expit, logit

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp080/out"
os.makedirs(OUT, exist_ok=True)
FRAME = f"{H}/build_AC/work/tr2024"
EPS = 1e-6
R = {}
t0 = time.time()


def lg(p):
    return logit(np.clip(np.asarray(p, np.float64), EPS, 1 - EPS))


# ── S14 범위 확인 ────────────────────────────────────────────────────
V = {}
for nm, p in [("arms/lg_p_q", f"{H}/arms/lg_p_q.npy"),
              ("arms/lg_p_v6", f"{H}/arms/lg_p_v6.npy"),
              ("arms/p_ebm", f"{H}/arms/p_ebm.npy"),
              ("arms/off", f"{H}/arms/off.npy"),
              ("oot/p_q", f"{H}/exp069/oot/p_q_oot2024.npy"),
              ("oot/p_cat", f"{H}/exp069/oot/p_cat_oot2024.npy"),
              ("oot/p_nn", f"{H}/exp071/out/p_nn_oot2024.npy"),
              ("oot/p_ebm", f"{H}/exp070/oot3/p_ebm_oot2024.npy"),
              ("build_AY/pcal_aa", f"{H}/build_AY/work/pcal_aa.npy")]:
    v = np.load(p).astype(np.float64)
    V[nm] = v
    R.setdefault("S14_ranges", {})[nm] = {
        "min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
        "sd": float(v.std()), "n": int(v.size),
        "is_probability_range": bool(v.min() >= 0.0 and v.max() <= 1.0)}
    print(f"  {nm:<22} min {v.min():+.6f} max {v.max():+.6f} mean {v.mean():+.6f}"
          f"  prob_range={bool(v.min()>=0 and v.max()<=1)}", flush=True)

# seed 파일 존재 확인
sd = sorted([f for f in os.listdir(f"{H}/exp069/oot") if f.startswith("p_q_oot2024_seed")])
R["oot_q_seed_files"] = sd
if sd:
    sv = [np.load(f"{H}/exp069/oot/{f}").astype(np.float64) for f in sd]
    mn_prob = np.mean(sv, axis=0)
    mn_logit = expit(np.mean([lg(x) for x in sv], axis=0))
    R["oot_q_assembly"] = {
        "n_seeds": len(sd),
        "max_abs_diff_vs_prob_mean": float(np.abs(V["oot/p_q"] - mn_prob).max()),
        "max_abs_diff_vs_logit_mean": float(np.abs(V["oot/p_q"] - mn_logit).max())}
    print("  oot p_q 조립:", R["oot_q_assembly"], flush=True)

# ── 라벨·프레임 ──────────────────────────────────────────────────────
tr = pd.read_csv(f"{H}/data/train.csv",
                 usecols=["row_id", "season", "game_type", "control_success",
                          "pitcher_id", "batter_hand", "pitcher_hand"])
tr24 = tr[tr.season == 2024].sort_values("row_id", kind="stable")
y24 = tr24.control_success.to_numpy(np.float64)
rate24 = float(y24.mean())
R["rate_2024"] = rate24
R["oot_q_mean_vs_rate2024"] = {"mean_p_q_oot": float(V["oot/p_q"].mean()),
                               "rate_2024": rate24,
                               "diff": float(V["oot/p_q"].mean() - rate24)}
print(f"  2024 성공률 {rate24:.9f} · mean(p_q_oot) {V['oot/p_q'].mean():.9f}", flush=True)

fr = pd.read_csv(f"{FRAME}/test.csv", encoding="utf-8-sig",
                 usecols=["row_id", "season", "game_type", "pitcher_id",
                          "batter_hand", "pitcher_hand"])
R["deploy_frame"] = {"n": int(len(fr)), "season_vals": sorted(fr.season.unique().tolist()),
                     "game_type_counts": fr.game_type.value_counts().to_dict(),
                     "row_id_matches_tr2024": bool(np.array_equal(
                         fr.row_id.to_numpy().astype(str),
                         tr24.row_id.to_numpy().astype(str)))}
print("  배포 대역 프레임:", R["deploy_frame"], flush=True)

# ── model_q 표 재구성 게이트 ─────────────────────────────────────────
MQ = json.load(open(f"{H}/build_BO/pkg_BO/mate/model_q/meta.json"))
OFS, BKO = MQ["offset"], MQ["backoff"]
BETA_Q, EBK, GAMMA_Q = OFS["beta"], OFS["eb_k"], MQ["logit_calibration"]["gamma"]
R["q_constants"] = {"beta": BETA_Q, "eb_k": EBK, "gamma": GAMMA_Q,
                    "gamma_unrounded": MQ["logit_calibration"]["gamma_unrounded"],
                    "key_cols": OFS["key_cols"], "n_lookup": len(OFS["lookup_d"]),
                    "league_dev": BKO["league_dev"], "n_by_cell": BKO["n_by_cell"]}


def build_tables(maxseason):
    """d_s(k) 표와 league_dev 를 train.csv 에서 재구성.

    마스크: exp_004 — season<=2022 & game_type=='F' 는 학습에서 제외.
    d_s = (s_cell - n_cell*overall) / (n_cell + k),  overall = 그 투수의 무조건 성공률.
    L(ph,bh) = P(succ|ph,bh) - P(succ|ph).
    """
    m = (tr.season <= maxseason) & ~((tr.season <= 2022) & (tr.game_type == "F"))
    d = tr[m & tr.control_success.notna()].copy()
    d["pitcher_id"] = d.pitcher_id.astype("Int64")
    ov = d.groupby("pitcher_id").control_success.mean()
    g = d.groupby(["pitcher_id", "batter_hand"]).control_success.agg(["sum", "count"])
    g = g.reset_index()
    g["overall"] = g.pitcher_id.map(ov)
    num = g["sum"].to_numpy(np.float64) - g["count"].to_numpy(np.float64) * g.overall.to_numpy(np.float64)
    keys = [f"{int(a)}|{int(b)}" for a, b in zip(g.pitcher_id, g.batter_hand)]
    NUM = dict(zip(keys, num))
    NC = dict(zip(keys, g["count"].to_numpy(np.int64)))
    # league_dev
    lg_ph = d.groupby("pitcher_hand").control_success.mean()
    lg_pb = d.groupby(["pitcher_hand", "batter_hand"]).control_success.agg(["mean", "count"])
    LD, NB = {}, {}
    for (ph, bh), row in lg_pb.iterrows():
        LD[f"{int(ph)}|{int(bh)}"] = float(row["mean"] - lg_ph.loc[ph])
        NB[f"{int(ph)}|{int(bh)}"] = int(row["count"])
    return NUM, NC, LD, NB, int(len(d))


NUM24, NC24, LD24, NB24, nfit24 = build_tables(2024)
dd_ref, nn_ref = OFS["lookup_d"], OFS["lookup_n"]
common = set(NUM24) & set(dd_ref)
mism_n = [k for k in common if NC24[k] != nn_ref[k]]
dv = np.array([NUM24[k] / (NC24[k] + EBK) - dd_ref[k] for k in common])
R["table_rebuild_gate_le2024"] = {
    "n_cells_rebuilt": len(NUM24), "n_cells_meta": len(dd_ref),
    "n_common": len(common), "n_only_in_rebuild": len(set(NUM24) - set(dd_ref)),
    "n_only_in_meta": len(set(dd_ref) - set(NUM24)),
    "n_count_mismatch": len(mism_n),
    "max_abs_diff_d": float(np.abs(dv).max()) if len(dv) else None,
    "n_fit_rows_rebuilt": nfit24, "n_fit_rows_meta": MQ["n_fit_rows_cells"],
    "league_dev_rebuilt": LD24, "league_dev_meta": BKO["league_dev"],
    "league_dev_max_abs_diff": max(abs(LD24[k] - BKO["league_dev"][k])
                                   for k in BKO["league_dev"]) if set(BKO["league_dev"]) <= set(LD24) else None,
    "n_by_cell_rebuilt": NB24, "n_by_cell_meta": BKO["n_by_cell"]}
print("\n=== 표 재구성 게이트(≤2024) ===", flush=True)
print(json.dumps({k: v for k, v in R["table_rebuild_gate_le2024"].items()
                  if not isinstance(v, dict)}, ensure_ascii=False, indent=1), flush=True)
print("  league_dev 재구성:", {k: round(v, 9) for k, v in LD24.items()}, flush=True)
print("  league_dev 메타  :", {k: round(v, 9) for k, v in BKO["league_dev"].items()}, flush=True)
print("  n_by_cell 재구성 :", NB24, flush=True)
print("  n_by_cell 메타   :", BKO["n_by_cell"], flush=True)

json.dump(R, open(f"{OUT}/s0_probe.json", "w"), indent=1, ensure_ascii=False)
print(f"\n기록 {OUT}/s0_probe.json · {time.time()-t0:.1f}s", flush=True)

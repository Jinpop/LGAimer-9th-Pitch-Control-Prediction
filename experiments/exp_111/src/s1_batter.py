# exp_111 — 타자 키 잔차 오프셋표를 ★현 기반(CS3)★ 캘프레임에서 잰다.
# read-only: 재학습 0 · 패키지 실행 0 · GPU 0 · 2스레드.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import glob, json, time
import numpy as np, pandas as pd

H = "/home/<user>/SungjinPark"
OUT = f"{H}/exp111"
os.makedirs(OUT, exist_ok=True)
REF, EPS, K_PERS = 0.2498069, 1e-6, 400
K_PTS = 100000.0 / REF
BETA = {"R": 0.597044, "F": 0.492041}
ALPHA = 1.07911
rng = np.random.default_rng(42)
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)

# ───────────────────────── stage A : 정렬 · 기반 잔차 ─────────────────────────
USE = ["row_id","season","game_type","control_success","pitcher_id","batter_id",
       "batter_hand","pitcher_hand","batter_team_id","pitcher_team_id",
       "balls_before","strikes_before","inning","outs_before","base_state","li",
       "score_diff_pitcher_team","game_month","asof_batter_n","asof_pitcher_n",
       "asof_pitcher_success_rate","asof_batter_success_rate",
       "asof_pitcher_breaking_rate","asof_pitcher_prev3_game_success_rate"]
tr = pd.read_csv(f"{H}/data/train.csv", encoding="utf-8-sig", usecols=USE)
log("train", len(tr))

ids = pd.read_csv(f"{H}/build_AC/work/tr2024/test.csv", encoding="utf-8-sig",
                  usecols=["row_id"])["row_id"].to_numpy()
cur24 = tr[tr.season == 2024].reset_index(drop=True)
assert len(cur24) == len(ids) == 253507, (len(cur24), len(ids))
SAME_ORDER = bool((cur24.row_id.to_numpy() == ids).all())
if not SAME_ORDER:
    cur24 = cur24.set_index("row_id").reindex(ids).reset_index()
log("calframe order == train[2024] order ?", SAME_ORDER)

p_cs3 = np.load(f"{H}/build_CU/work/calp_PAR.npy")
assert p_cs3.shape == (253507,)
y24 = cur24.control_success.to_numpy(np.float64)

def anchored(p):
    """전역 평균 재착륙(d_adj 자리)까지 넣은 기반과 그 잔차."""
    q = np.clip(p, EPS, 1-EPS)
    g = float(np.mean(y24 - q))
    b = np.clip(q + g, EPS, 1-EPS)
    return b, y24 - b, g

# 현 기반 CS3
base_cs3, r_cs3, g_cs3 = anchored(p_cs3)
# 참조 기반 : 2026-08-16 스크린이 쓴 67열 AA 팔 단독 (하니스 양성대조용)
past24 = tr[tr.season <= 2023]
def deployed_offset(cur, past):
    ov = past.groupby("pitcher_id").control_success.mean()
    cell = past.groupby(["pitcher_id","batter_hand"]).control_success.agg(["sum","count"])
    s_map, n_map = cell["sum"].to_dict(), cell["count"].to_dict()
    key = list(zip(cur.pitcher_id, cur.batter_hand))
    hit = np.array([k in n_map for k in key])
    o = np.nan_to_num(cur.pitcher_id.map(ov).to_numpy())
    s_ = np.array([s_map.get(k,0.0) for k in key], float)
    n_ = np.array([n_map.get(k,0)  for k in key], float)
    d_s = (s_ - n_*o)/(n_ + K_PERS)
    f = past.groupby(["pitcher_hand","batter_hand"]).control_success.mean()
    c = past.groupby("pitcher_hand").control_success.mean()
    L = np.nan_to_num(np.array([f.get((a,b),np.nan)-c.get(a,np.nan)
                                for a,b in zip(cur.pitcher_hand, cur.batter_hand)]))
    return np.where(hit, d_s, L)
off24  = deployed_offset(cur24, past24)
beta24 = np.array([BETA.get(g,0.59072) for g in cur24.game_type])
p67 = np.mean([np.load(x) for x in sorted(glob.glob(
        f"{H}/experiments/exp_048/pred/pred_A1_u2024_seed*_at1400.npy"))], axis=0)
base67, r67, g67 = anchored(np.clip(p67 + beta24*off24, EPS, 1-EPS))

def score(b): return 100000.0*(1.0 - np.mean((b-y24)**2)/REF)
A = {"calframe_order_equals_train2024": SAME_ORDER,
     "n_rows": int(len(y24)),
     "cs3": {"calp_mean": float(p_cs3.mean()), "anchor_shift_g": g_cs3,
             "brier": float(np.mean(r_cs3**2)), "score_pts": float(score(base_cs3))},
     "ref67": {"anchor_shift_g": g67, "brier": float(np.mean(r67**2)),
               "score_pts": float(score(base67))}}
log("CS3 calframe score", A["cs3"]["score_pts"], "| 67col ref", A["ref67"]["score_pts"])

# ───────────────────── 전파계수 c : 세 방식 ─────────────────────
w_sp = np.load(f"{H}/exp109/work/w_unit.npy")
pCR = np.load(f"{H}/exp109/work/calp_CR.npy"); pCT = np.load(f"{H}/exp109/work/calp_CT.npy")
c_emp = (pCT - pCR) / (-0.014 * w_sp)                     # 실측 할선 (pre-alpha 로짓축)
c_ana = ALPHA * p_cs3 * (1.0 - p_cs3)                     # 해석식
C = {"empirical_secant": {"mean": float(c_emp.mean()), "sd": float(c_emp.std()),
                          "K_mean_c2": float(K_PTS*np.mean(c_emp**2))},
     "analytic_alpha_p1p": {"mean": float(c_ana.mean()), "sd": float(c_ana.std()),
                            "K_mean_c2": float(K_PTS*np.mean(c_ana**2))},
     "exp_109_reference": {"c_mean": 0.26870501915988315, "c_sd": 0.006521888316361284,
                           "K_mean_c2": 28920.30698468467},
     "K_G_unit_sp_axis_exp109_measured": 28954.695452108943}
C["analytic_vs_empirical_mean_ratio"] = float(c_ana.mean()/c_emp.mean())
log("c_emp mean/sd", C["empirical_secant"]["mean"], C["empirical_secant"]["sd"],
    "| c_ana", C["analytic_alpha_p1p"]["mean"])
json.dump({"stageA": A, "propagation_c": C}, open(f"{OUT}/sA.json","w"), ensure_ascii=False, indent=1)

# ───────────────────── stage B : 표 재료 (<=2023) ─────────────────────
PRED = {s: f"{H}/experiments/exp_030/pred/pred_C5B3_u{s}_seed*_st2.npy" for s in (2022,2023)}
tabs, prov = [], {}
for s in (2022, 2023):
    fs = sorted(glob.glob(PRED[s])); prov[str(s)] = [os.path.basename(x) for x in fs]
    p = np.mean([np.load(x) for x in fs], axis=0)
    cur = tr[tr.season == s].copy(); past = tr[tr.season <= s-1]
    off = deployed_offset(cur, past)
    beta = np.array([BETA.get(g,0.59072) for g in cur.game_type])
    r = cur.control_success.to_numpy(float) - np.clip(p + beta*off, EPS, 1-EPS)
    cur["resid"] = r - r.mean()
    tabs.append(cur[["batter_id","pitcher_hand","batter_team_id","resid"]])
TAB = pd.concat(tabs, ignore_index=True)
log("table material rows", len(TAB), prov)

AXES = {"batter_id":                (["batter_id"],),
        "batter_id_x_pitcher_hand": (["batter_id","pitcher_hand"],),
        "batter_team":              (["batter_team_id"],)}

def group_codes(frame, cols):
    if len(cols) == 1:
        return frame[cols[0]].to_numpy()
    return (frame[cols[0]].to_numpy().astype(np.int64)*10 +
            frame[cols[1]].to_numpy().astype(np.int64))

def eb_k(codes, resid):
    """사전 등록 적률법 EB 수축상수. 표 재료만 쓴다 — 2024 는 한 값도 안 들어간다."""
    uid, idx = np.unique(codes, return_inverse=True)
    G, N = len(uid), len(resid)
    n = np.bincount(idx, minlength=G).astype(np.float64)
    s = np.bincount(idx, weights=resid, minlength=G)
    m = s/n
    ss = np.bincount(idx, weights=resid**2, minlength=G)
    SSW = float(ss.sum() - (s**2/n).sum())
    s2w = SSW/(N-G)
    mbar = float(resid.mean())
    Q = float(np.sum(n*(m-mbar)**2))
    denom = N - float(np.sum(n**2))/N
    s2b = (Q - (G-1)*s2w)/denom
    k = float(s2w/s2b) if s2b > 0 else float("inf")
    return k, dict(G=int(G), N=int(N), sigma2_within=float(s2w),
                   sigma2_between=float(s2b), k=k, Q=Q, denom=float(denom))

def table_dir(tab_codes, tab_resid, row_codes, k):
    uid, idx = np.unique(tab_codes, return_inverse=True)
    n = np.bincount(idx, minlength=len(uid)).astype(np.float64)
    s = np.bincount(idx, weights=tab_resid, minlength=len(uid))
    val = s/(n+k)                       # = n*m/(n+k)
    pos = pd.Index(uid).get_indexer(row_codes)
    hit = pos >= 0
    raw = np.where(hit, val[np.maximum(pos,0)], 0.0)
    return raw, hit

def vertex(delta, resid):
    d = delta - delta.mean()
    G = float(np.mean(d*d)); L = float(np.mean(d*resid))
    if G <= 0: return dict(K_L=0.0, K_G=0.0, vertex_pts=0.0, u_star=0.0)
    return dict(K_L=K_PTS*L, K_G=K_PTS*G, vertex_pts=K_PTS*L*L/G, u_star=L/G)

def boot_t(delta, resid, blocks, nb_draw=1000):
    d = delta - delta.mean(); prod = d*resid
    uidb, bidx = np.unique(blocks, return_inverse=True); nb = len(uidb)
    bs = np.bincount(bidx, weights=prod, minlength=nb)
    draws = rng.integers(0, nb, size=(nb_draw, nb))
    se = float(bs[draws].sum(axis=1).std(ddof=1))/len(resid)
    L = float(prod.mean())
    return (L/se if se > 0 else 0.0), se

bat24 = cur24.batter_id.to_numpy()
SHRINK_GRID = [20, 60, 200, 600, 2000]
B = {"table_material": {"rows": int(len(TAB)), "preds": prov,
     "🔴_limit": "잔차원이 exp_030 51열 옛 팔이다. 현 블렌드의 <=2023 예측은 존재하지 않는다."}}
DIRS = {}
for name, (cols,) in AXES.items():
    tc = group_codes(TAB, cols); rc = group_codes(cur24, cols)
    k, kd = eb_k(tc, TAB.resid.to_numpy(np.float64))
    raw, hit = table_dir(tc, TAB.resid.to_numpy(np.float64), rc, k)
    f = float(hit.mean())
    sd = float(raw.std()); ctr = float(raw.mean())
    w_std = (raw - ctr)/sd
    rec = {"cols": cols, "prereg_k": k, "k_estimator": kd,
           "coverage_f": f, "n_groups_table": kd["G"],
           "n_groups_seen_in_2024": int(len(np.unique(rc))),
           "raw_center": ctr, "raw_sd": sd}
    for cname, cvec in (("a_prob_direct", np.ones_like(raw)),
                        ("b_empirical_secant", c_emp),
                        ("c_analytic", c_ana)):
        vr  = vertex(cvec*w_std, r_cs3)          # 현 기반
        vr67= vertex(cvec*w_std, r67)            # 참조 기반
        rec[cname] = {"cs3": vr, "ref67": vr67}
    t, se = boot_t(c_emp*w_std, r_cs3, bat24)
    rec["t_of_L_cs3_batter_block_boot"] = t
    rec["se_L_cs3"] = se
    t67, _ = boot_t(c_emp*w_std, r67, bat24)
    rec["t_of_L_ref67"] = t67
    # 5점 격자 민감도 (선택 아님)
    rec["k_grid_sensitivity_vertex_pts_cs3"] = {}
    for kk in SHRINK_GRID:
        rr, _ = table_dir(tc, TAB.resid.to_numpy(np.float64), rc, kk)
        s2 = rr.std()
        rec["k_grid_sensitivity_vertex_pts_cs3"][str(kk)] = (
            vertex(c_emp*(rr-rr.mean())/s2, r_cs3)["vertex_pts"] if s2>0 else 0.0)
    # S201 연속형 전파 검산
    dd = c_emp*w_std; dd = dd - dd.mean()
    rec["S201_check"] = {"K_G_measured": K_PTS*float(np.mean(dd*dd)),
        "K_mean_c2_times_Var_w": float(K_PTS*np.mean(c_emp**2)*np.var(w_std)),
        "Var_w": float(np.var(w_std))}
    rec["S201_check"]["pct_dev"] = 100.0*(rec["S201_check"]["K_G_measured"]/
                                          rec["S201_check"]["K_mean_c2_times_Var_w"] - 1.0)
    rec["S57_indicator_law"] = {"applicable": False,
        "why": "셀 지시자가 아니다 — 타자 표는 행마다 다른 연속값이고 커버리지 f 가 ~1 이라 f(1-f)~0.",
        "f": f, "A_if_forced": (rec["S201_check"]["K_G_measured"]/(f*(1-f))) if 0<f<1 else None}
    DIRS[name] = (w_std, raw, hit, rc, tc, k)
    B[name] = rec
    log(name, "k=%.1f f=%.4f vertex(cs3,b)=%.3f ref67=%.3f t=%.2f" %
        (k, f, rec["b_empirical_secant"]["cs3"]["vertex_pts"],
         rec["b_empirical_secant"]["ref67"]["vertex_pts"], t))


# ── 하니스 양성대조 : e110_ax22.py 관례 그대로(c=1, k 를 2024 표본 내 5격자에서 선택) 재현 ──
REPRO = {}
for name, (cols,) in AXES.items():
    tc = group_codes(TAB, cols); rc = group_codes(cur24, cols)
    best = None
    for kk in SHRINK_GRID:
        rr, _ = table_dir(tc, TAB.resid.to_numpy(np.float64), rc, kk)
        b = np.mean((np.clip(base67+rr, EPS, 1-EPS)-y24)**2)
        if best is None or b < best[0]: best = (b, kk, rr)
    _, kk, rr = best
    v = vertex(rr, r67)
    d1 = (base67-y24)**2 - (np.clip(base67+rr, EPS, 1-EPS)-y24)**2
    REPRO[name] = {"k_insample": kk, "gain_at_1": K_PTS*float(d1.mean()),
                   "K_L": v["K_L"], "K_G": v["K_G"], "vertex_pts": v["vertex_pts"],
                   "s_star": v["u_star"]}
    log("REPRO", name, REPRO[name])
EXPECT = {"batter_id": 6.975, "batter_id_x_pitcher_hand": 6.743, "batter_team": 6.625}
REPRO["🔴_expected_exp110"] = EXPECT
REPRO["🔴_max_abs_dev"] = max(abs(REPRO[n]["vertex_pts"]-EXPECT[n]) for n in EXPECT)
B["harness_positive_control_repro_exp110"] = REPRO
log("REPRO max dev", REPRO["🔴_max_abs_dev"])

# 3형제 동시해 (S44 고윳값 절단)
def joint(vs, resid):
    V = np.array([v - v.mean() for v in vs])
    L = K_PTS*(V @ resid)/len(resid); G = K_PTS*(V @ V.T)/len(resid)
    w, Q = np.linalg.eigh(G); keep = w > w.max()*1e-10
    u = Q[:,keep] @ (Q[:,keep].T @ L / w[keep])
    return float(2*L@u - u@G@u), int(keep.sum()), np.corrcoef(V).tolist()
vs_cs3 = [c_emp*DIRS[n][0] for n in AXES]
jg, rk, rho3 = joint(vs_cs3, r_cs3)
jg67, rk67, _ = joint(vs_cs3, r67)
B["joint_trio"] = {"cs3_joint_vertex_pts": jg, "rank": rk,
                   "naive_sum_cs3": sum(B[n]["b_empirical_secant"]["cs3"]["vertex_pts"] for n in AXES),
                   "ref67_joint_vertex_pts": jg67,
                   "pairwise_rho": rho3, "order": list(AXES)}
log("joint trio cs3", jg, "ref67", jg67)
json.dump(B, open(f"{OUT}/sB.json","w"), ensure_ascii=False, indent=1)
np.save(f"{OUT}/dirs.npy", np.array([DIRS[n][0] for n in AXES]))
print("STAGE_B_OK")

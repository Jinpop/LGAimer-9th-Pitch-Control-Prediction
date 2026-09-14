# exp_110 stage2 — vintage stability of table-riding offset directions.
# Read-only. No retraining. 2 threads. Mirrors exp_109 stage2c exactly:
#   build table on hist=<=2023 and hist=<=2024, apply BOTH to the SAME 2024 rows,
#   report rho(w_le2023, w_le2024), sd ratio, and the fraction of rows that go to 0.
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import json, numpy as np, pandas as pd

USE = ["season","pitcher_id","batter_id","batter_hand","pitcher_hand",
       "strikes_before","balls_before","control_success",
       "asof_pitcher_n","asof_pitcher_success_rate",
       "asof_batter_n","asof_batter_success_rate"]
df = pd.read_csv(os.path.expanduser("~/SungjinPark/data/train.csv"),
                 encoding="utf-8-sig", usecols=USE)
print("train rows", len(df), flush=True)
row = df[df.season == 2024].reset_index(drop=True)
print("2024 rows", len(row), flush=True)
H23 = df[df.season <= 2023]
H24 = df[df.season <= 2024]
out = {}

def stat(name, a, b, note=""):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    sa, sb = a[m].std(), b[m].std()
    r = float(np.corrcoef(a[m], b[m])[0,1]) if sa > 0 and sb > 0 else float("nan")
    out[name] = {"rho_le2023_vs_le2024": r,
                 "sd_le2023": float(sa), "sd_le2024": float(sb),
                 "sd_ratio_23_over_24": float(sa/sb) if sb else None,
                 "var_ratio": float((sa/sb)**2) if sb else None,
                 "frac_zero_le2023": float(np.mean(a[m]==0.0)),
                 "frac_zero_le2024": float(np.mean(b[m]==0.0)),
                 "n": int(m.sum()), "note": note}
    print(f"{name:28s} rho={r:+.4f}  sd23={sa:.5f} sd24={sb:.5f}  "
          f"varR={(sa/sb)**2:.4f}  z23={np.mean(a[m]==0.0):.4f}", flush=True)

# ---------- A/B/C: personal cell table (lookup_d): key (pitcher_id, batter_hand)
def personal(H, k=400.0):
    y = H.control_success.to_numpy(np.float64)
    pid = H.pitcher_id.to_numpy(np.int64); bh = H.batter_hand.to_numpy(np.int64)
    # pitcher unconditional rate ("self" target), unshrunk
    ids, idx = np.unique(pid, return_inverse=True)
    pn = np.bincount(idx, minlength=len(ids)).astype(np.float64)
    ps = np.bincount(idx, weights=y, minlength=len(ids))
    o = ps / np.maximum(pn, 1.0)
    key = pid.astype(np.int64) * 10 + bh
    kids, kidx = np.unique(key, return_inverse=True)
    cn = np.bincount(kidx, minlength=len(kids)).astype(np.float64)
    cs = np.bincount(kidx, weights=y, minlength=len(kids))
    # each cell's pitcher's own overall
    cell_pid = kids // 10
    po = pd.Series(o, index=ids).reindex(cell_pid).to_numpy()
    d_s = (cs - cn * po) / (cn + k)
    return dict(zip(kids, zip(d_s, cn)))

rkey = row.pitcher_id.to_numpy(np.int64) * 10 + row.batter_hand.to_numpy(np.int64)
for tag, H in (("23", H23), ("24", H24)):
    T = personal(H)
    ds = np.array([T.get(kk, (0.0, 0.0))[0] for kk in rkey])
    nc = np.array([T.get(kk, (0.0, 0.0))[1] for kk in rkey])
    hit = np.array([kk in T for kk in rkey], np.float64)
    globals()["ds_"+tag], globals()["nc_"+tag], globals()["hit_"+tag] = ds, nc, hit
    print(f"  personal table <= {'2023' if tag=='23' else '2024'}: "
          f"cells={len(T)} hit_rate={hit.mean():.6f}", flush=True)

stat("L12c_shrunk_dev_s400", ds_23, ds_24, "q_infer lookup_d d_s(k=400,self) — 라벨 EB 수축")
stat("L11_personal_miss_flag", 1.0-hit_23, 1.0-hit_24, "적중 여부 표시자")
stat("L12_personal_n_cell", nc_23, nc_24, "셀 표본수")
stat("L12b_personal_log1p_n", np.log1p(nc_23), np.log1p(nc_24), "로그 셀 표본수")

# ---------- D: league_dev backoff L(ph,bh)
def leaguedev(H):
    y = H.control_success.to_numpy(np.float64)
    ph = H.pitcher_hand.to_numpy(np.int64); bh = H.batter_hand.to_numpy(np.int64)
    r = {}
    for p in (1,2):
        mp = ph == p
        base = y[mp].mean()
        for b in (1,2):
            m2 = mp & (bh == b)
            r[f"{p}|{b}"] = float(y[m2].mean() - base)
    return r
out["league_dev_cells"] = {"le2023": leaguedev(H23), "le2024": leaguedev(H24)}
print("league_dev 23:", out["league_dev_cells"]["le2023"], flush=True)
print("league_dev 24:", out["league_dev_cells"]["le2024"], flush=True)

# ---------- E: sp_pit_split (POSITIVE CONTROL — exp_109 got rho 0.660)
EPS = 1e-6
def _lg(p):
    p = np.clip(np.asarray(p, np.float64), EPS, 1-EPS); return np.log(p/(1-p))
def splits_axis(H, R, who_col, cellf, nc_, k=75.0):
    y = H.control_success.to_numpy(np.float64); league = float(y.mean())
    w = H[who_col].to_numpy(np.int64)
    ids, idx = np.unique(w, return_inverse=True); P = len(ids)
    n = np.bincount(idx, minlength=P).astype(np.float64)
    s = np.bincount(idx, weights=y, minlength=P)
    p_all = (s + k*league)/(n + k)
    code = cellf(H)
    ln = np.bincount(code, minlength=nc_).astype(np.float64)
    ls = np.bincount(code, weights=y, minlength=nc_)
    lrate = (ls + k*league)/(ln + k)
    lgc = _lg(lrate) - _lg(league)
    flat = idx*nc_ + code
    cn = np.bincount(flat, minlength=P*nc_).astype(np.float64).reshape(P, nc_)
    cs = np.bincount(flat, weights=y, minlength=P*nc_).reshape(P, nc_)
    rate = (cs + k*p_all[:,None])/(cn + k)
    D = (_lg(rate) - _lg(p_all)[:,None]) - lgc[None,:]
    pos = pd.Index(ids).get_indexer(R[who_col].to_numpy(np.int64))
    rc = cellf(R)
    v = np.where(pos >= 0, D[np.maximum(pos,0), rc], 0.0)
    return v
f_bh = lambda d: d.batter_hand.to_numpy(np.int64) - 1
f_s2 = lambda d: (d.strikes_before.to_numpy(np.int64) >= 2).astype(np.int64)
SD1, SD2 = 0.08766134165019425, 0.05495482212319005   # exp_109 stage1 shipped
for tag, H in (("23", H23), ("24", H24)):
    a = splits_axis(H, row, "pitcher_id", f_bh, 2)
    b = splits_axis(H, row, "pitcher_id", f_s2, 2)
    globals()["sp_"+tag] = a/SD1 + b/SD2
stat("sp_pit_split_CONTROL", sp_23, sp_24, "exp_109 stage2c 실측 rho=0.6595 · Var비 0.9037")

# ---------- F: inseason anchor (beta_ins_off)
PR = ['success','reverse','middle','ball','strike']
def anchor(H):
    n = H.asof_pitcher_n.to_numpy(np.float64)
    c = np.round(n * np.nan_to_num(H.asof_pitcher_success_rate.to_numpy(np.float64)))
    t = pd.DataFrame({"pid": H.pitcher_id.to_numpy(), "n": n, "c": c})
    p = t.loc[t.groupby("pid")["n"].idxmax()].set_index("pid")
    lg = float(np.nansum(t.c)/max(np.nansum(t.n),1))
    return p, lg
def ins_off(H, R, K=50.0):
    p, lg = anchor(H)
    n = R.asof_pitcher_n.to_numpy(np.float64)
    pos = pd.Index(p.index.to_numpy()).get_indexer(R.pitcher_id.to_numpy())
    kn = pos >= 0; safe = np.where(kn, pos, 0)
    an = np.where(kn, p["n"].to_numpy()[safe], 0.0)
    ac = np.where(kn, p["c"].to_numpy()[safe], 0.0)
    ins_n = np.maximum(n - an, 0.0)
    prior = np.where(an > 0, ac/np.maximum(an,1.0), lg)
    k = np.maximum(np.round(n*np.nan_to_num(R.asof_pitcher_success_rate.to_numpy(np.float64))) - ac, 0.0)
    cur = (k + K*prior)/(ins_n + K)
    dev = cur - prior
    cc = np.clip(cur, .02, .98); pp = np.clip(cc - dev, .02, .98)
    return np.nan_to_num(np.log(cc/(1-cc)) - np.log(pp/(1-pp))), cur
o23, c23 = ins_off(H23, row); o24, c24 = ins_off(H24, row)
stat("beta_ins_off_AXIS1", o23, o24, "script.py axis_offsets[0] — inseason 동결 앵커(카운트, 무라벨)")
stat("ins_p_success", c23, c24, "in-season 성공률 자체")

# ---------- G: supp_anc_suc (exp_076 frozen career anchor rate)
def anc_rate(H, R):
    p, lg = anchor(H)
    pos = pd.Index(p.index.to_numpy()).get_indexer(R.pitcher_id.to_numpy())
    kn = pos >= 0; safe = np.where(kn, pos, 0)
    an = np.where(kn, p["n"].to_numpy()[safe], np.nan)
    ac = np.where(kn, p["c"].to_numpy()[safe], np.nan)
    return np.where(an > 0, ac/np.maximum(an,1.0), np.nan)
stat("supp_anc_suc", anc_rate(H23,row), anc_rate(H24,row), "exp_076 동결 커리어 성공률 앵커")

# ---------- H: ctrl_asof_psr (NO table — pure own column). rho must be exactly 1.
psr = row.asof_pitcher_success_rate.to_numpy(np.float64)
stat("ctrl_asof_psr_NOTABLE", psr, psr, "표를 안 탄다 — 행 자기 열. rho=1 은 정의상")

json.dump(out, open(os.path.expanduser("~/SungjinPark/exp110_vint.json"),"w"),
          ensure_ascii=False, indent=1)
print("WROTE ~/SungjinPark/exp110_vint.json")

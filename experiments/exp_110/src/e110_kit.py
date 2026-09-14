# exp_110 stage5 — use the CT solver pack.
#  (a) prove row alignment by decomposing SHIFT into its three axis rawvectors
#  (b) K*G and vertex for the 8 post-hoc constants the pack scores instantly
#  (c) is ctrl_asof_psr at its vertex?  (coordinator's question)
import os, json, sys
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from solve import predict, solve_dadj, AIM, Z, N, A, BS, D0, FB, FC, W0
K_PTS = 100000.0 / 0.2498069
ROOT = "<PROJECT_ROOT>"

# ---------- rows: train season==2024, row_id order ----------
tr = pd.read_csv(f"{ROOT}/data/train.csv", encoding="utf-8-sig")
r24 = tr[tr.season == 2024].sort_values("row_id", kind="stable").reset_index(drop=True)
print("frame rows", len(r24), "pack rows", N, flush=True)
y = r24.control_success.to_numpy(np.float64)

# ---------- (a) decompose SHIFT ----------
def unit(raw, c, s):
    u = np.zeros(len(raw)); f = np.isfinite(raw); u[f] = (raw[f] - c) / s; return u
psr = r24.asof_pitcher_success_rate.to_numpy(np.float64)
U_PSR = unit(psr, 0.5113970221638665, 0.054384800312060846)

# ins offset with the DEPLOYED (<=2024) anchor, exactly as the package computes it
PRc = ['success','reverse','middle','ball','strike']
def ins_off(H, R, KK=50.0):
    n = H.asof_pitcher_n.to_numpy(np.float64)
    c = np.round(n*np.nan_to_num(H.asof_pitcher_success_rate.to_numpy(np.float64)))
    t = pd.DataFrame({"pid": H.pitcher_id.to_numpy(), "n": n, "c": c})
    p = t.loc[t.groupby("pid")["n"].idxmax()].set_index("pid")
    lgm = float(np.nansum(t.c)/max(np.nansum(t.n),1))
    nn = R.asof_pitcher_n.to_numpy(np.float64)
    pos = pd.Index(p.index.to_numpy()).get_indexer(R.pitcher_id.to_numpy())
    kn = pos >= 0; safe = np.where(kn, pos, 0)
    an = np.where(kn, p["n"].to_numpy()[safe], 0.0)
    ac = np.where(kn, p["c"].to_numpy()[safe], 0.0)
    ins_n = np.maximum(nn-an, 0.0)
    prior = np.where(an > 0, ac/np.maximum(an,1.0), lgm)
    k = np.maximum(np.round(nn*np.nan_to_num(R.asof_pitcher_success_rate.to_numpy(np.float64)))-ac, 0.0)
    cur = (k + KK*prior)/(ins_n + KK); dev = cur - prior
    cc = np.clip(cur,.02,.98); pp = np.clip(cc-dev,.02,.98)
    return np.nan_to_num(np.log(cc/(1-cc)) - np.log(pp/(1-pp)))
U_INS = unit(ins_off(tr[tr.season <= 2024], r24), -0.13044477259429785, 0.2250062102818995)

# sp_pit_split with the DEPLOYED (<=2024) splits table
EPSs = 1e-6
_lg = lambda p: np.log(np.clip(p,EPSs,1-EPSs)/(1-np.clip(p,EPSs,1-EPSs)))
def sp_ax(H, R, cellf, nc_, k=75.0):
    yy = H.control_success.to_numpy(np.float64); league = float(yy.mean())
    w = H.pitcher_id.to_numpy(np.int64); ids, idx = np.unique(w, return_inverse=True); P=len(ids)
    n = np.bincount(idx,minlength=P).astype(float); s = np.bincount(idx,weights=yy,minlength=P)
    p_all = (s+k*league)/(n+k); code = cellf(H)
    ln = np.bincount(code,minlength=nc_).astype(float); ls = np.bincount(code,weights=yy,minlength=nc_)
    lgc = _lg((ls+k*league)/(ln+k)) - _lg(league)
    flat = idx*nc_+code
    cn = np.bincount(flat,minlength=P*nc_).astype(float).reshape(P,nc_)
    cs = np.bincount(flat,weights=yy,minlength=P*nc_).reshape(P,nc_)
    D = (_lg((cs+k*p_all[:,None])/(cn+k)) - _lg(p_all)[:,None]) - lgc[None,:]
    pos = pd.Index(ids).get_indexer(R.pitcher_id.to_numpy(np.int64))
    return np.where(pos>=0, D[np.maximum(pos,0), cellf(R)], 0.0)
H24 = tr[tr.season <= 2024]
sp_raw = (sp_ax(H24, r24, lambda d: d.batter_hand.to_numpy(np.int64)-1, 2)/0.08766134165019425
          + sp_ax(H24, r24, lambda d: (d.strikes_before.to_numpy(np.int64)>=2).astype(np.int64), 2)/0.05495482212319005)
U_SP = unit(sp_raw, -0.12989747735282217, 1.4450786086160488)

X = np.column_stack([U_INS, U_PSR, U_SP])
coef, *_ = np.linalg.lstsq(X, Z['shift'], rcond=None)
resid = Z['shift'] - X @ coef
print("\n★ALIGNMENT GATE★  lstsq(SHIFT ~ [u_ins, u_psr, u_sp])")
print("  coef      =", [float(f"{c:.9f}") for c in coef])
print("  shipped u = [-0.006534, +0.006544, -0.014]")
print(f"  residual  max|r| = {np.abs(resid).max():.3e}   rms = {resid.std():.3e}"
      f"   ||SHIFT|| = {Z['shift'].std():.3e}", flush=True)
OK = np.abs(resid).max() < 1e-6
print("  VERDICT:", "PASS — 행 정렬·세 축 분해 확인" if OK else "FAIL — 정렬 미확인, 아래 수치 쓰지 마라")

out = {"alignment_gate": {"coef": coef.tolist(),
       "shipped_u": [-0.006534, 0.006544, -0.014],
       "resid_max_abs": float(np.abs(resid).max()), "PASS": bool(OK)}}

# ---------- (b) K*G and vertex for the 8 instant constants ----------
def score(**kw):
    d = solve_dadj(**kw); p = predict(d_adj=d, **kw); return p
p0 = predict(); b0 = float(np.mean((p0-y)**2))
print(f"\n부모 CT 캘프레임 Brier = {b0:.9f}  (점수 {K_PTS*(0.2498069-b0)/1.0*0+100000*(1-b0/0.2498069):.5f})")
print(f"\n{'constant':16s}{'h':>10s}{'K*L':>11s}{'K*G':>11s}{'vertex':>9s}{'step*':>11s}{'resid':>9s}")
CONST = [("w_v6", W0, 0.01), ("alpha", A, 0.01), ("beta_skew", BS, 0.01),
         ("ft_b", FB, 0.005), ("ft_c", FC, 0.005)]
for name, v0, h in CONST:
    gs = []
    for dv in (-h, 0.0, h):
        kw = {name: v0+dv}
        p = score(**kw); gs.append(100000*(1-np.mean((p-y)**2)/0.2498069))
    gm, g0, gp = gs
    KL = (gp-gm)/(4*h); KG = -(gp-2*g0+gm)/(h*h)
    vtx = KL*KL/KG if KG > 0 else float('nan')
    step = KL/KG if KG > 0 else float('nan')
    print(f"{name:16s}{h:10.4f}{KL:11.2f}{KG:11.1f}{vtx:9.4f}{step:+11.6f}"
          f"{(vtx - (2*KL*0-KG*0)):9.4f}")
    out.setdefault("instant_constants", {})[name] = dict(
        h=h, K_L=float(KL), K_G=float(KG), vertex_gain_pts=float(vtx),
        step_to_vertex=float(step), current_value=float(v0))

# ---------- (c) is ctrl_asof_psr at its vertex? ----------
print("\n★ctrl_asof_psr 축 (axis_offsets[1], test.csv 원열, 표 안 탐)★")
def score_axis(du, vec):
    sh = Z['shift'] + du*vec
    m = W0*np.log(np.clip(Z['p_partner'],1e-15,1-1e-15)/(1-np.clip(Z['p_partner'],1e-15,1-1e-15)))
    return sh
def predict_shift(shift_new, d_adj):
    lg_ = lambda p: np.log(np.clip(p,1e-15,1-1e-15)/(1-np.clip(p,1e-15,1-1e-15)))
    r = 1.0-W0; c_aa = r*(0.25244/0.30); c_T70 = r*(0.04756/0.30)
    m = W0*lg_(Z['p_partner']) + c_aa*lg_(Z['p_aa']) + c_T70*lg_(Z['p_t70']) + shift_new
    off = FB*(Z['lg_mr']-0) + FC*(Z['lg_wo']-0)
    from solve import MU_MR, MU_WO
    off = FB*(Z['lg_mr']-MU_MR) + FC*(Z['lg_wo']-MU_WO) + Z['aux']
    from scipy.special import expit
    return expit(A*m + BS*m*m + d_adj + off)
from scipy.optimize import brentq
def sc_axis(du, vec):
    sh = Z['shift'] + du*vec
    d = brentq(lambda dd: predict_shift(sh, dd).mean()-AIM, -5, 5, xtol=1e-16,
               rtol=4*np.finfo(float).eps)
    p = predict_shift(sh, d)
    return 100000*(1-np.mean((p-y)**2)/0.2498069)
for label, vec, cur in (("ctrl_asof_psr", U_PSR, 0.006544),
                        ("beta_ins_off", U_INS, -0.006534),
                        ("sp_pit_split", U_SP, -0.014)):
    h = 0.002
    gm, g0, gp = sc_axis(-h, vec), sc_axis(0.0, vec), sc_axis(h, vec)
    KL = (gp-gm)/(4*h); KG = -(gp-2*g0+gm)/(h*h)
    step = KL/KG if KG > 0 else float('nan')
    vtx_from_here = KL*KL/KG if KG > 0 else float('nan')
    print(f"  {label:15s} K*L={KL:9.2f}  K*G={KG:9.1f}  현재에서 꼭지점까지 {step:+.6f}"
          f"  남은 이득 {vtx_from_here:7.4f}점   (u_now={cur})")
    out.setdefault("axis_offsets", {})[label] = dict(
        K_L=float(KL), K_G=float(KG), step_from_current_to_vertex=float(step),
        residual_gain_pts=float(vtx_from_here), u_now=cur, probe_h=h)

out["frame_caveat"] = ("팩 프레임 = train season==2024 253,507행 · 표는 <=2024 "
   "(정렬 게이트의 u_ins/u_sp 계수가 그것을 증명한다). S32 자기적합 프레임이라 "
   "다섯 팔 전부에 표본 내다 — L 은 팔이 이미 외운 방향으로 눌린다. K*G 는 신뢰(S147).")
out["aim_point"] = AIM
json.dump(out, open(f"{ROOT}/experiments/exp_110/out/exp110_kit.json","w"),
          ensure_ascii=False, indent=1)
print("\nWROTE experiments/exp_110/out/exp110_kit.json")

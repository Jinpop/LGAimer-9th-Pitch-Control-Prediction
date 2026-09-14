"""exp_101 관문C — 3자유도 격자.  z_inner(w,w2) = (1-w-w2)*z_mate + w*z_deep + w2*z_tabpfn

면·상수·핀·부트스트랩은 exp_097 src/sB1_rebase.py 를 그대로 옮겼다(독립 재작성 아님 — 같은 면임을 보장하려고).
기준선 = 고친 면 2팔 최적 26.269817382706947 (1632스텝, w=0.43).
사전 등록: argmax 이득 - 26.2698 >= +2.5  AND  기준점 대비 짝지은 투수블록 부트스트랩 t >= 2.0.
"""
import json, os, itertools, argparse
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "4"
import numpy as np, pandas as pd
from scipy.special import expit, logit

ap = argparse.ArgumentParser(); ap.add_argument("--ptab", required=True); ap.add_argument("--tag", default="c")
a = ap.parse_args()
H = "/home/<user>/SungjinPark"; B97 = f"{H}/exp097"; B = f"{H}/exp101"
OOT, OOT3, OOT4 = f"{H}/exp069/oot", f"{H}/exp070/oot3", f"{H}/exp071/out"
EPS, REF = 1e-6, 0.2498069; K = 100000.0 / REF
S5 = [42, 777, 2024, 101, 7]; AA = (42, 7, 2024)
W4 = {"LGB":0.05,"CAT":0.68,"NN":0.15,"LEAF":0.12}
BO = dict(alpha=1.07911, beta_skew=-0.059151, c_aa=0.25244, c_T70=0.04756,
          w_out=0.7, w_v6_inner=0.6668397489, w_ebm=-0.1, beta_ins=0.03)
BASE_EXPECT = 928.501461058684; REF_GAIN = 26.269817382706947
lg = lambda p: logit(np.clip(np.asarray(p, np.float64), EPS, 1-EPS))
pts = lambda b: 100000.0*(1.0-b/REF)

tro_all = pd.read_csv(f"{H}/data/train.csv", usecols=["row_id","season","pitcher_id","control_success"])
vam_file = (tro_all.season.to_numpy() == 2024)
vam = np.load(f"{H}/exp089/work/valid_mask.npy")
assert np.array_equal(vam_file, vam), "valid mask 불일치"
tro24 = tro_all[vam_file]
perm = np.argsort(tro24.row_id.to_numpy(), kind="stable")   # 파일순 -> 면의 row_id 정렬순
tro = tro24.iloc[perm]
y = tro.control_success.to_numpy(np.float64); pit = tro.pitcher_id.to_numpy()
rid = np.asarray(np.load(f"{OOT}/row_id_oot2024.npy", allow_pickle=True)).astype(str)
assert np.array_equal(rid, tro.row_id.to_numpy().astype(str)); assert len(y) == 253507
YB = float(y.mean())
V = {k: np.load(p).astype(np.float64) for k,p in {
    "p_lgb":f"{OOT}/p_lgb_oot2024.npy","p_cat":f"{OOT}/p_cat_oot2024.npy",
    "p_leaf":f"{OOT}/p_leaf_oot2024.npy","p_nn":f"{OOT4}/p_nn_oot2024.npy",
    "p_q":f"{OOT}/p_q_oot2024.npy","p_ebm":f"{OOT3}/p_ebm_oot2024.npy",
    "off":f"{OOT}/off_oot2024.npy"}.items()}
LV = sum(W4[t]*lg(V[k]) for t,k in (("LGB","p_lgb"),("CAT","p_cat"),("NN","p_nn"),("LEAF","p_leaf")))
LQ, LE = lg(V["p_q"]), lg(V["p_ebm"])
L_aa = np.mean([lg(np.load(f"{H}/experiments/exp_049/pred/pred_A1_u2024_seed{s}_at1400.npy")) for s in AA],0)
L_t70 = np.mean([lg(np.load(f"{H}/experiments/exp_055/pred/pred_T70_u2024_seed{s}_at1400.npy")) for s in AA],0)
Z_MATE = ((1-BO["w_ebm"])*(BO["w_v6_inner"]*LV+(1-BO["w_v6_inner"])*LQ) + BO["w_ebm"]*LE + BO["beta_ins"]*V["off"])
Z_DEEP = np.mean([np.load(f"{B97}/out/preds/z_F1632_f64o_s{s}.npy") for s in S5],0)
P_TAB = np.load(a.ptab)                       # 파일순(valid) 확률
assert P_TAB.shape[0] == 253507, P_TAB.shape
Z_TAB = lg(P_TAB)[perm]

def pin(z,f,target=YB):
    c=0.0
    for _ in range(200):
        p=f(z,c); g=float(p.mean())-target
        if abs(g)<1e-15: break
        c-=g/float((p*(1-p)).mean())
    return c
F_RC = lambda z,c: expit(z+c)
F_ABD = lambda z,c: expit(BO["alpha"]*z+BO["beta_skew"]*z*z+c)
prob = lambda z,conv: (F_RC if conv=="rc" else F_ABD)(z, pin(z, F_RC if conv=="rc" else F_ABD))
sc = lambda z,conv: pts(float(np.mean((prob(z,conv)-y)**2)))
root = lambda zm: BO["w_out"]*zm + BO["c_aa"]*L_aa + BO["c_T70"]*L_t70
def sd_match(zn,zr): return ((zn-zn.mean())/zn.std())*zr.std()+zr.mean()

rng = np.random.default_rng(20260824)
uq, inv = np.unique(pit, return_inverse=True); nb = len(uq)
order = np.argsort(inv, kind="stable"); st = np.searchsorted(inv[order], np.arange(nb))
en = np.append(st[1:], len(order)); BSIZE = (en-st).astype(np.float64)
PICKS = np.stack([rng.integers(0,nb,nb) for _ in range(1000)])
def boot_t(d):
    bs_=np.array([d[order[st[j]:en[j]]].sum() for j in range(nb)])
    bs=bs_[PICKS].sum(1)/BSIZE[PICKS].sum(1)
    m=float(d.mean()); se=float(bs.std(ddof=1))
    return {"gain_pts":-m*K,"boot_se_brier":se,"t":(-m/se) if se>0 else None,
            "ci95_pts":[float(-np.percentile(bs,97.5)*K),float(-np.percentile(bs,2.5)*K)]}

z0 = root(Z_MATE); base = sc(z0,"abd")
assert abs(sc(z0,"rc")-BASE_EXPECT) < 1e-9, sc(z0,"rc")
R = {"exp_id":"exp_101","stage":"gateC","ptab":a.ptab,"n_rows":int(len(y)),
     "baseline_w0_rc":sc(z0,"rc"),"REF_GAIN_2arm":REF_GAIN,
     "prereg":{"rule":"argmax gain - 26.2698 >= +2.5 AND boot t vs ref point >= 2.0",
               "source":"experiments/exp_101/config.yaml (발사 전 확정, S8)"}}
R["diag"] = {
 "tabpfn_alone_recentred": sc(Z_TAB,"rc"), "deep_alone_recentred": sc(Z_DEEP,"rc"),
 "mate_alone_recentred": sc(Z_MATE,"rc"),
 "rho_tab_mate": float(np.corrcoef(Z_TAB,Z_MATE)[0,1]),
 "rho_tab_deep": float(np.corrcoef(Z_TAB,Z_DEEP)[0,1]),
 "rho_deep_mate": float(np.corrcoef(Z_DEEP,Z_MATE)[0,1]),
 "z_sd":{"tab":float(Z_TAB.std()),"deep":float(Z_DEEP.std()),"mate":float(Z_MATE.std())},
 "S115_seed_band":[0.92756,0.91064,0.93932]}
G = [round(x,3) for x in np.arange(0.0,0.601,0.01)]
G2 = [round(x,3) for x in np.arange(0.0,0.401,0.01)]
best = {"gain":-1e9}; cur = []
for w in G:
    for w2 in G2:
        if w+w2 > 0.85: continue
        g = sc(root((1-w-w2)*Z_MATE + w*Z_DEEP + w2*Z_TAB),"abd") - base
        cur.append((w,w2,g))
        if g > best["gain"]: best = {"w":w,"w2":w2,"gain":g}
R["grid_argmax"] = best
R["two_arm_check"] = {"w":0.43,"w2":0.0,
   "gain": sc(root(0.57*Z_MATE+0.43*Z_DEEP),"abd")-base}
R["best_w2_at_w043"] = max([(w2, sc(root((1-0.43-w2)*Z_MATE+0.43*Z_DEEP+w2*Z_TAB),"abd")-base)
                            for w2 in G2], key=lambda t:t[1])
z_ref = root(0.57*Z_MATE + 0.43*Z_DEEP)
p_ref = prob(z_ref,"abd")
z_b = root((1-best["w"]-best["w2"])*Z_MATE + best["w"]*Z_DEEP + best["w2"]*Z_TAB)
p_b = prob(z_b,"abd")
R["boot_vs_ref"] = boot_t((p_b-y)**2 - (p_ref-y)**2)
R["gain_sd_matched"] = sc(sd_match(z_b,z_ref),"abd") - base
d1 = best["gain"] - REF_GAIN
R["prereg"]["delta_vs_ref"] = d1
R["prereg"]["t"] = R["boot_vs_ref"]["t"]
R["prereg"]["PASS_size"] = bool(d1 >= 2.5)
R["prereg"]["PASS_t"] = bool((R["boot_vs_ref"]["t"] or 0) >= 2.0)
R["prereg"]["PASS"] = bool(R["prereg"]["PASS_size"] and R["prereg"]["PASS_t"])
R["curve_w2_at_argmax_w"] = [[w2,g] for (w,w2,g) in cur if abs(w-best["w"])<1e-9]
json.dump(R, open(f"{B}/out/gateC_{a.tag}.json","w"), ensure_ascii=False, indent=1)
print(json.dumps({k:v for k,v in R.items() if k!="curve_w2_at_argmax_w"}, ensure_ascii=False, indent=1))

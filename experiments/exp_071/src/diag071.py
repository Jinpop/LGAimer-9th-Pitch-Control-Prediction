"""왜 NN 이 나빠질수록 4팔 블렌드가 좋아지는가 — 정보인가 로짓 배율인가."""
import os
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[v]="6"
import json,numpy as np,pandas as pd
from scipy.special import expit,logit
H="/home/<user>/SungjinPark"; E=f"{H}/exp071"; OOT=f"{H}/exp069/oot"; OUT=f"{E}/out"
EPS=1e-6; REF=0.2498069
lg=lambda p: logit(np.clip(np.asarray(p,np.float64),EPS,1-EPS))
conv=lambda b:100000.0*(1.0-b/REF)
tr=pd.read_csv(f"{H}/data/train.csv",usecols=["row_id","season","control_success"])
tr=tr[tr.season==2024].sort_values("row_id",kind="stable").reset_index(drop=True)
y=tr.control_success.to_numpy(np.float64); ybar=float(y.mean())
def rc(z,c0=0.0):
    for _ in range(40):
        p=expit(z+c0); mu=float(p.mean())
        if abs(mu-ybar)<1e-14: break
        d=(mu-ybar)/float((p*(1-p)).mean()); c0-=d
        if abs(d)<1e-14: break
    return c0
def sz(z): 
    c=rc(z); return conv(float(np.mean((expit(z+c)-y)**2)))
V={k:np.load(f"{OOT}/{f}").astype(np.float64) for k,f in
   [("p_lgb","p_lgb_oot2024.npy"),("p_cat","p_cat_oot2024.npy"),
    ("p_leaf","p_leaf_oot2024.npy"),("p_v6_3arm","p_v6_3arm_renorm_oot2024.npy")]}
L3=lg(V["p_v6_3arm"])
EP={s:np.load(f"{E}/cache/p24_epochs_seed{s}.npy").astype(np.float64) for s in (42,777,2024,101)}
R={"로짓_표준편차":{"v6_3arm":float(L3.std()),"p_cat":float(lg(V["p_cat"]).std()),
                    "p_lgb":float(lg(V["p_lgb"]).std()),"p_leaf":float(lg(V["p_leaf"]).std())}}
print("① 3종판 로짓을 단순히 a 배로 늘리기만 했을 때 (NN 정보 0)")
sc={}
for a in [1.0,1.05,1.10,1.15,1.20,1.25,1.30,1.40,1.50]:
    sc[f"{a:.2f}"]=sz(a*L3); print(f"   a={a:.2f}  {sc[f'{a:.2f}']:8.3f}")
R["3종판_로짓배율_단독"]=sc
print("② k 별 NN 로짓 sd 와 4팔 블렌드 로짓 sd")
rows={}
for kk in [2,4,6,8,10,12,14,20,25]:
    pk=expit(np.mean([lg(EP[s][kk-1]) for s in (42,777,2024,101)],0))
    Ln=lg(pk); z4=0.05*lg(V["p_lgb"])+0.68*lg(V["p_cat"])+0.15*Ln+0.12*lg(V["p_leaf"])
    # 배율만 맞춘 3종판과 같은 sd 로 만들었을 때의 점수 (정보 없는 대조군)
    a=float(z4.std()/L3.std())
    rows[str(kk)]={"nn_logit_sd":float(Ln.std()),"blend_logit_sd":float(z4.std()),
                   "blend_점수":sz(z4),"같은sd_3종판배율_점수":sz(a*L3),"a":a,
                   "NN정보_순증":sz(z4)-sz(a*L3)}
    print(f"   k={kk:>2} nn_sd {Ln.std():6.3f} blend_sd {z4.std():6.3f} (3arm {L3.std():.3f}) "
          f"blend {sz(z4):8.3f}  같은배율3종판 {sz(a*L3):8.3f}  순증 {sz(z4)-sz(a*L3):+7.3f}")
R["k별_배율분해"]=rows
R["해설"]=("'같은sd_3종판배율_점수' 는 NN 을 넣지 않고 3종판 로짓만 같은 표준편차로 늘린 대조군이다. "
           "blend 점수에서 이것을 뺀 '순증' 이 NN 이 실제로 넣은 정보다.")
json.dump(R,open(f"{OUT}/diag_scale.json","w"),indent=1,ensure_ascii=False)
print("DONE_DIAG")

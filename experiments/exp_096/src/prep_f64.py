"""exp_096 — 🔴 감사 후속: float64 표준화 판본 X 를 만든다.

감사 §5(b): exp_089 `prep089.py:132-133` 이 float32 배열에 111만 행 nanmean/nanstd 를 걸어
`season` 열의 mu 2040.479(참 2021.057) · sd 19.616(참 1.4238) 로 무너진다.
순위변환은 열별 단조 사상에 불변이라 **A3 는 이 결함을 조용히 고치고 표준화 팔들은 안 고친다.**
따라서 선언 변인("표준화 -> 순위변환")과 "float32 결함 수리"가 한 덩어리다.

이 파일이 그 둘을 가른다: 같은 프레임·같은 마스크로 **float64 누산** 표준화 X 를 만든다.
분위표와 마찬가지로 mu/sd 는 학습 구간(trm_oot23) 행에서만 계산한다.
"""
import json, os, time
for _v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
    os.environ[_v]="4"
import numpy as np, pandas as pd
H="/home/<user>/SungjinPark"; B=f"{H}/exp096"; W89=f"{H}/exp089/work"
E60=f"{H}/exp060_nn"; PKG=f"{H}/audit_deep/pkg/DEEP/mate"
t0=time.time()
P=json.load(open(f"{PKG}/model/deep/prep.json",encoding="utf-8"))
NUM=P["NUM"]; CAT=P["CAT"]
df=pd.read_parquet(f"{E60}/cache/frame99.parquet",columns=list(dict.fromkeys(NUM+["row_id"])))
tr=pd.read_csv(f"{H}/data/train.csv",usecols=["row_id"])
assert np.array_equal(df.row_id.to_numpy().astype(str),tr.row_id.to_numpy().astype(str))
Xraw=df[NUM].astype("float32").to_numpy()
trm=np.load(f"{W89}/trm_oot23.npy"); vam=np.load(f"{W89}/valid_mask.npy")
assert int((trm & vam).sum())==0
mu64=np.nanmean(Xraw[trm].astype(np.float64),axis=0)
sd64=np.nanstd(Xraw[trm].astype(np.float64),axis=0)
sd64=np.where(sd64<1e-6,1.0,sd64)
mu32=np.nanmean(Xraw[trm],axis=0).astype(np.float32)
sd32=np.nanstd(Xraw[trm],axis=0).astype(np.float32)
sd32=np.where(sd32<np.float32(1e-6),np.float32(1.0),sd32)
with np.errstate(invalid="ignore"):
    X=(Xraw-mu64.astype(np.float32))/sd64.astype(np.float32)
X=np.nan_to_num(X,nan=0.0,posinf=0.0,neginf=0.0).astype(np.float32)
np.save(f"{B}/work/X_f64_oot23.npy",X)
Xold=np.load(f"{W89}/X_oot23.npy",mmap_mode="r")
d=np.abs(np.asarray(Xold[vam],np.float64)-X[vam].astype(np.float64))
worst=np.argsort(-d.max(0))[:6]
M={"exp_id":"exp_096","stage":"prep_f64",
   "🔴_why":"감사 §5(b) — float32 누산 표준화 결함과 순위변환을 가르기 위한 대조 입력",
   "mu_sd_source":"trm_oot23 행만 (2024 행 0)",
   "n_train_rows":int(trm.sum()),
   "max_abs_diff_vs_exp089_X_on_valid":float(d.max()),
   "n_cols_changed":int((d.max(0)>1e-6).sum()),
   "worst_cols":[{"col":NUM[int(i)],"max_abs_diff_valid":float(d[:,i].max()),
                  "mu_f32":float(mu32[i]),"mu_f64":float(mu64[i]),
                  "sd_f32":float(sd32[i]),"sd_f64":float(sd64[i])} for i in worst],
   "wall_sec":round(time.time()-t0,1)}
json.dump(M,open(f"{B}/out/prep_f64.json","w"),ensure_ascii=False,indent=1)
print(json.dumps(M,ensure_ascii=False,indent=1))

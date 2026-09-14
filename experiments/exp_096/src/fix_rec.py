"""exp_096 — 감사 지적 기록 정정 두 건 (계산은 안 바뀌고 기록만 바로잡는다)."""
import json, sys
sys.path.insert(0,"/home/<user>/SungjinPark/exp096/src")
from arch096 import TokenMLP, FTAttn, n_params
H="/home/<user>/SungjinPark"
C=json.load(open(f"{H}/exp089/work/schema_oot23.json"))["cards"]
p=f"{H}/exp096/out/prep096.json"; d=json.load(open(p))
a192=n_params(FTAttn(94,C,D=192,blocks=3,heads=8,mult=2))
d["param_match"]["🔴_정정_감사지적"]=(
  "이 블록의 원래 값은 ATT_D=128 기본값으로 계산된 것이고 **실제 실행은 ATT_D=192**다. "
  "run_laneA.sh 가 ATT_D=192 ATT_BLK=3 ATT_HEAD=8 ATT_MULT=2 를 export 하고, "
  "대조군은 run_laneB.sh 의 mlpP35(TokenMLP D=35)다. 아래가 실제 실행값이다.")
d["param_match"]["ACTUAL_A1_attn_params_D192_blk3"]=a192
d["param_match"]["ACTUAL_A0_ctrl_arm"]="mlpP35 (TokenMLP D=35)"
d["param_match"]["ACTUAL_A0_ctrl_params"]=n_params(TokenMLP(94,C,D=35))
d["param_match"]["ACTUAL_rel_gap"]=abs(n_params(TokenMLP(94,C,D=35))-a192)/a192
d["param_match"]["🔴_A1_결과"]=("A1(attn) 은 **fit 0건** — 학습 6에폭은 끝나는데 "
  "예측 청크 32768 에서 CUDA OOM 으로 죽는다(logs/laneA.log). "
  "고침은 attn 일 때 예측 청크를 8192 이하로 내리는 것 하나뿐이다. "
  "따라서 파라미터수 맞춤 대조군 mlpP35 는 대조할 상대가 없다.")
json.dump(d,open(p,"w"),ensure_ascii=False,indent=1)
print(json.dumps(d["param_match"],ensure_ascii=False,indent=1))

"""exp_106 s9 — 🔴 실행하지 않고 건너뛴다 (s10 이 대체).

원래 목적: s2 의 7축 해가 국소 정체인지 확인하려고 출발점 8개로 다시 올리는 것.
(근거: 8축 적합이 7축의 상위집합인데 더 **낮은** +12.19 를 냈다.)

건너뛰는 이유: s2 산출물을 본 뒤 **s10 이 같은 일을 더 잘 한다**는 것이 분명해졌다 —
s10 의 `wide` 판이 c_T70 경계를 -2.0 까지 넓히고 각 족마다 출발점 5개로 두 번씩 올린다.
s9 를 그대로 두면 같은 것을 20분 더 계산한다. 원본은 `s9_multistart_ORIGINAL.py.bak`.

⚠️ 이것은 사전 등록 문턱(S8) 변경이 아니다 — 판정은 run106 에서 이미 계산됐고
   이 파일은 진단 하나를 다른 진단으로 대체할 뿐이다.
"""
import json
json.dump({"exp_id": "exp_106", "stage": "s9_multistart",
           "SKIPPED": True,
           "사유": "s10 의 wide 판이 더 넓은 경계 + 족별 다중출발로 대체한다. "
                   "원본 코드는 src/s9_multistart_ORIGINAL.py.bak 에 보존.",
           "대체": "s10_shipvalid.json fits.wide"},
          open("/home/<user>/SungjinPark/exp106/out/s9_multistart.json", "w"),
          ensure_ascii=False, indent=1)
print("s9 skipped — superseded by s10 fits.wide")

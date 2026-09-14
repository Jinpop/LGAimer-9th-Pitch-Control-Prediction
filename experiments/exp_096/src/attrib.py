"""exp_096 — 귀속 판정 블록. 감사 §5(b)(c) 대조가 끝난 뒤 metrics.json 에 못박는다."""
import json
p="/home/<user>/SungjinPark/exp096/out/metrics.json"
M=json.load(open(p))
vt=M["verdict_table"]
M["🔴_ATTRIBUTION"]={
 "질문":"사전 등록 문턱을 통과한 이득(+7.29)은 무엇이 낸 것인가.",
 "대조_사다리":[
  {"arm":"mlp96","바꾼것":"없음 (exp_089 X 그대로, train096 하니스)",
   "rho_vs_deep":vt["mlp96"]["rho_vs_deep"],"alone":vt["mlp96"]["alone_seedmean"],
   "resid":vt["mlp96"]["residual_vs_ref"],
   "읽기":"exp_090 딥 팔을 **정확히 재현**한다 (resid 0.0, argmax (0.36,0), rho 1.0). "
          "→ train096.py 하니스가 train090.py 와 같은 함수임이 증명됐다."},
  {"arm":"ctrl35","바꾼것":"모델 용량 (D 96->35)","rho_vs_deep":vt["ctrl35"]["rho_vs_deep"],
   "alone":vt["ctrl35"]["alone_seedmean"],"resid":vt["ctrl35"]["residual_vs_ref"],
   "읽기":"거의 0. 용량만 바꾼 두 번째 팔은 값이 없다."},
  {"arm":"cos","바꾼것":"LR 일정 (상수 -> cosine)","rho_vs_deep":vt["cos"]["rho_vs_deep"],
   "alone":vt["cos"]["alone_seedmean"],"resid":vt["cos"]["residual_vs_ref"],
   "읽기":"거의 0. S98 이 지목한 손잡이는 이 면에서 값이 없다."},
  {"arm":"mlp64","바꾼것":"🔴 표준화 누산 정밀도만 (float32 -> float64). 순위변환 없음.",
   "rho_vs_deep":vt["mlp64"]["rho_vs_deep"],"alone":vt["mlp64"]["alone_seedmean"],
   "resid":vt["mlp64"]["residual_vs_ref"],"t":vt["mlp64"]["paired_t"],
   "읽기":"**PASS**. 단독 점수는 딥과 사실상 같은데(+2.27) 잔여가 +8.37 로 rank 보다 크다."},
  {"arm":"rank","바꾼것":"입력 분위수-정규 변환 (+ float32 결함도 함께 사라진다)",
   "rho_vs_deep":vt["rank"]["rho_vs_deep"],"alone":vt["rank"]["alone_seedmean"],
   "resid":vt["rank"]["residual_vs_ref"],"t":vt["rank"]["paired_t"],
   "읽기":"PASS. 단독 +24.96 로 더 좋지만 잔여는 mlp64 보다 **작다**."}],
 "🔴_결론":(
  "**이 이득의 원인은 '순위변환'이 아니다.** 순위변환을 전혀 쓰지 않고 표준화 누산 정밀도만 "
  "float32->float64 로 바꾼 mlp64 가 같은 크기의 이득(+8.37)을 낸다. rank 는 열별 단조 사상에 "
  "불변이라 그 결함을 **덤으로** 고치고 있었다(감사 §5(b)). "
  "즉 사다리가 말하는 것은 '입력 파이프라인을 조금만 흔들면 딥 팔과 탈상관된 두 번째 팔이 "
  "생기고, 그 탈상관 자체가 이 면에서 +8 을 낸다'이다. "
  "용량(ctrl35)·LR 일정(cos)을 흔드는 것으로는 탈상관이 안 생긴다(rho_deep 0.991·0.984)."),
 "S103_정합성":{
  "mlp64_delta_rho_vs_mate":vt["mlp64"]["rho_vs_mate"]-M["deep_reference_arm"]["rho_vs_z_mate"],
  "mlp64_delta_alone":vt["mlp64"]["alone_seedmean"]-862.5652996988365,
  "mlp64_S103_predicted":(-308.92)*(vt["mlp64"]["rho_vs_mate"]-M["deep_reference_arm"]["rho_vs_z_mate"])
                          +0.19533*(vt["mlp64"]["alone_seedmean"]-862.5652996988365),
  "mlp64_observed":vt["mlp64"]["residual_vs_ref"],
  "rank_S103_predicted":M["S103_crosscheck"]["S103_predicted_gain_delta"],
  "rank_observed":M["S103_crosscheck"]["observed_residual_vs_2arm_ref"],
  "읽기":"둘 다 S103 교환율이 예측하는 자릿수 안이다. 새 기전이 아니라 알려진 교환율의 실현이다."},
 "🔴_남은_경고":[
  "독립 fold(<=2022 -> 2023)에서 rank 는 표준화 팔보다 **나쁘다** (척도 자유 −6.85, t −1.13). "
  "그 fold 는 신호가 훨씬 약한 환경이라(AUC 0.531 vs 0.553) 반증은 아니나 확인도 아니다.",
  "argmax 는 2024 라벨로 고른 점이라 상향 편의가 있다. 인접 점 (0.20,0.20) t 3.11 · "
  "(0.24,0.20) t 2.83 로 argmax 한 점에만 얹혀 있지는 않다.",
  "스텝 축 진단: rank 이득이 816<1224<1632<2448<3264 순으로 **단조 감소**한다. 봉우리를 "
  "아래로 못 묶었다. 사전 등록 정본은 딥과 스텝을 맞춘 1632 다.",
  "A1(attn)은 fit 0건 — 예측 청크 32768 에서 CUDA OOM. 아키텍처 계열 질문은 **미측정**이다.",
  "팀 원장 분류로는 '팔 추가'(전이 0.574)와 '팔 손질'(전이 ~0)의 경계에 있다 — 슬롯은 새로 "
  "생기지만 내용은 같은 팔의 흔든 사본이다. 전이비를 0.574 로 가정하지 마라."]}
json.dump(M,open(p,"w"),ensure_ascii=False,indent=1)
print(json.dumps(M["🔴_ATTRIBUTION"]["🔴_결론"],ensure_ascii=False))
print("attribution written")

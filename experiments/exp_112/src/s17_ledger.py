import json
O='out/'
led={
 'blend.json /failtype_offset/c':{
  'LB_points':0,'base':None,'residual_computed':'아니오',
  'ledger':'exp_082/metrics.json /constants/c_shipped = 0.0 · /stage2_honest_surface/c_lg_wo (정직면 단독 스크린)',
  'verdict':'★측정하고 버렸다★ — 단독으로는 옳은 판정. (b,c) 동시 해는 미실행'},
 'blend.json /beta_f':{
  'LB_points':2,'base':'AF(부모 AC, 2026-08-13) 와 AI(부모 AH, 2026-08-14) — 둘 다 약 40후보 전',
  'ledger':'CV_LB.md:151 (AF β_f 0.19, +27.9504851031) · CV_LB.md:148 (AI β_f 0.19→0.0, +17.6298337629)',
  '🔴':'AI 는 ★묶인 쌍★ 이다 — `ins_* 16열` 과 `β_f→0` 을 같이 바꿨다. β_f→0 자신의 몫은 ★분리된 적이 없다★.',
  'rationale_recorded':'CV_LB.md:344 «약한 형태로 싣고 있던 신호를 강한 형태로 바꿨다» — ins_* 16열이 같은 시즌 내 폼 신호의 강한 형태라 약한 형태를 껐다',
  'residual_computed':'아니오',
  'verdict':'의도적으로 0 을 둔 것이 맞다(중복 제거). 다만 ① 귀속이 묶여 있고 ② 잔여 미계산 ③ 우리 캘프레임에서 측정 불가(has_form 0/253,507). 🔴 감사관 cov 프레임은 has_form 을 245,634/245,789 로 살렸다(CV_LB.md:324) — 재려면 그 프레임이 필요하다'},
 'mate/model/blend.json /beta_ins':{
  'LB_points':1,'base':'AR (2026-08-17) — 딥 팔 이전',
  'ledger':'CV_LB.md:934 — AT 가 0.03→0.06 으로 ★−1.0314113487★. 축 기울기 −34.3804/단위. 🔴 AS(0.00)는 ★제출되지 않았다★ — 1094.9667375 는 선형 ★예측★ 이다',
  'residual_computed':'아니오',
  '🔴_S138':'그 축은 mate 내부라 진폭이 (1−w_deep) 배로 움직인다. AR 시절 w_deep=0(배율 1.0) → 지금 0.40(배율 0.60). 같은 beta_ins 값이 지금은 0.6배 효과이므로 ★꼭지점이 beta_ins 단위로 1.667배 바깥으로 이동했다★ — AR 시절 «줄이는 게 낫다» 가 지금 그대로 성립하지 않는다',
  'verdict':'방향 근거가 낡았고 재추론 없이 못 잰다. 방어 가능한 숫자 없음'},
 'blend.json /d_form_eb_k = 100':{'LB_points':0,'verdict':'★죽은 메타데이터★ — 어떤 .py 도 이 키를 안 읽는다. form_offset.py 가 K=100 을 하드코딩하고, beta_f=0 이라 그 항 전체가 죽어 있다. 조정 가능한 상수가 아니다'},
 'mate/model_q/meta.json offset.eb_k = 400':{'LB_points':0,'verdict':'★PRINT-ONLY + 가드★ — q_infer.py 514(대입)·536(assert>0)·620(print) 뿐. 수축은 lookup_d 표에 이미 구워져 있어 이 숫자를 바꿔도 예측이 한 비트도 안 변한다. 조정 가능한 상수가 아니다'},
 'model/meta.json offset.eb_k = 400':{'LB_points':0,'verdict':'★PRINT-ONLY + 가드★ — aa_infer.py 1084·1136·1216 뿐. 위와 같다'},
 'blend.json /beta_skew = −0.059151':{'LB_points':'프로브 1 + 꼭지점 착륙 (BC 세대)','base':'BC — 딥 팔 이전',
  'ledger':'blend.json /beta_skew_derivation (Q·G·probe_sizing·vertex_machinery)',
  'residual_computed':'그 세대 기준으로만','verdict':'기반 다름. 현 기반 LB 점 0개. 캘프레임 K·G 82.72 — LB +1.4 를 얻으려면 |K·L| ≥ 10.76 이 필요'},
 'blend.json /axis_offsets[1] ctrl_asof_psr':{'LB_points':1,'base':'BJ/BL 세대',
  'ledger':'blend.json /axis_offsets[1] {probe_u −0.006087, probe_gain_pts −3.0727297301, L_i_back_solved +0.0004138189513845949, G_ii 0.07119965607894004}',
  'residual_pts':0.0152,'verdict':'사실상 닫힘. 🟢 그 세대 G_ii 를 K 배한 값 28,502.4 가 내 현 기반 캘프레임 28,500.5 와 ★0.007%★ 일치 — 기하가 안 변했다'},
 'blend.json /axis_offsets[0] beta_ins_off':{'LB_points':1,'base':'BI/BL 세대',
  'ledger':'blend.json /axis_offsets[0] {probe_u −0.006033, probe_gain_pts +0.9596987465, L_i_back_solved −0.0004138284137996067, G_ii 0.07132053755550466}',
  'residual_pts':0.0134,'verdict':'사실상 닫힘'},
 'blend.json /failtype_offset/b':{'LB_points':1,'base':'BO (팀원, 2026-08-19 21:46)',
  'ledger':'CV_LB.md:1048 — «실패유형 오프셋 b = −0.05», BO 1115.9913864857 → 1118.6680789834, +2.6766924977. 🔴 그 걸음은 −0.05 이고 현 출하 −0.036451 은 CA 의 5축 동시 해가 나중에 옮긴 값이다(blend.json /CA_one_dimensional_LB_solves/b)',
  'residual_pts':0.0089,'verdict':'단독으로는 닫힘. (b,c) 평면으로 보면 아니다'},
}
json.dump(led,open(O+'s17_ledger.json','w'),ensure_ascii=False,indent=1)
m=json.load(open('metrics.json')); m['ledger_original_lines']=led
json.dump(m,open('metrics.json','w'),ensure_ascii=False,indent=1)
print('ok')

import json
O='out/'
J=lambda f: json.load(open(O+f))
m=J('s10_master.json'); L=J('s9_Lverdict.json'); st=J('s7_step.json')
br=J('s6_bracket.json'); an=J('s3_anchors.json'); sc=J('s8_screen.json'); cr=J('s11_cross.json')
out={
 'exp_id':'exp_112','date':'2026-08-27',
 'type':'read-only 측정 · 재학습 0 · zip 0 · 슬롯 0 · GPU 0 · 서버 CPU: 열 추출 1회(수 초)',
 'question':'현 패키지 CW 의 조정 가능한 상수 중 아직 꼭지점에 착륙하지 않은 것이 무엇이고 각각 LB 몇 점인가',
 '🔴_ANSWER':{
   'n_live_adjustable_scalars_or_tables':21,
   'n_print_only_or_dead_that_looked_adjustable':8,
   'n_never_bought_on_any_LB':6,
   '★top3★':[
     {'rank':1,'constant':'blend.json /failtype_offset/c','value_now':0.0,
      'LB_expected':'★측정 불가 — 방향 미지★. K·G = 3385.74 는 확정. 크기는 |K·L| 을 몰라 상한을 못 낸다',
      'action':'프로브 1발 (|u| ≈ 0.0075, 2차 비용 0.19점) — 착륙이 아니라 측정'},
     {'rank':2,'constant':'blend.json /beta_f','value_now':0.0,
      'LB_expected':'★우리 프레임으로는 원리적으로 측정 불가★ (has_form 0/253,507)',
      'action':'배포에서만 발동하는 축. 프로브하려면 실제 대회 표본이나 앵커 뒤 프레임이 필요'},
     {'rank':3,'constant':'model/meta.json /branch_mix (F 행 0.5/0.5) · /offset/beta_by_segment',
      'value_now':[0.5,[0.597044,0.492041]],
      'LB_expected':'LB 점 0개. meta 가 기록한 로컬 대가는 valid_all −2.768점(branch_mix)',
      'action':'AA 팔 재추론이 필요 — 이번 실험 범위 밖'}],
   '🔴_what_this_experiment_could_NOT_do':'이 프레임에서 L 을 못 읽는다(아래 L_verdict). 따라서 "얼마짜리인가" 에 숫자로 답할 수 있는 미착륙 축은 ★0개★ 다. G 는 전부 냈다.'},
 'L_verdict':L,
 'master_table':m,
 'calframe_signal_screen':sc,
 'failtype_c':{
   'K_G_calframe':sc['failtype_c (출하 0.0)']['K_G'],
   'K_L_calframe_UNTRUSTED':sc['failtype_c (출하 0.0)']['K_L_cal'],
   't_calframe':sc['failtype_c (출하 0.0)']['t'],
   'K_G_bc_cross_with_shipped_b':br['K_G_bc'],
   'rho_with_shipped_b':-0.6352555590530388,
   'rho_with_other_live_axes':cr['ft_c_vs_others'],
   'bracket_IF_the_b_anchored_ratio_holds':br['failtype_c_bracket'],
   'minimax_step_IF_that_bracket_holds':st,
   '🔴_caveat':'이 브래킷은 캘프레임 L 에 b 의 전이비를 곱한 것이다. 같은 프레임이 다른 두 축의 부호를 틀렸으므로 ★근거로 쓰지 마라★. 브래킷은 프로브 크기를 고르는 데만 쓴다',
   'probe_design':{'u':-0.0075,'quadratic_cost_pts':0.1904,
     'linear_readout_if_abs_KL_100':1.5,
     'note':'부호가 미지이므로 걸음의 부호도 미지다. 순위는 제출 이력의 최대값이라 프로브에 하방이 없다',
     'ship_cost':'blend.json 숫자 1개 + d_adj 종속 재유도. 코드 0줄, 부스터 호출 0'}},
 'anchors':an,
 'gates':{
   'K_G_check_1_ctrl_asof_psr':{'calframe':sc['axis ctrl_asof_psr (출하 +0.006544)']['K_G'],
      'package_recorded':400309.19882517256*0.07119965607894004,'pct':0.0068},
   'K_G_check_2_count':{'calframe':an['anchor_count_CS3']['K_G_calframe'],'package_recorded':2.30510,'pct':0.53},
   'ledger_reproduction_count_residual':{'mine':an['anchor_count_CS3']['residual_LB_pts'],'S215':0.250},
   'probe_gain_recompute_from_recorded_L_G':{'ctrl_asof_psr':[-3.0727297300999994,-3.0727297301],
      'beta_ins_off':[0.9596987465000002,0.9596987465]},
   'm_logit_mix_recovery_max_resid':2.1094237467877974e-15,
   'offline_CW_is_shipped_vector':'logit 차가 선언 d_adj 걸음과 5.62e-16 안에서 상수'},
}
json.dump(out,open('metrics.json','w'),ensure_ascii=False,indent=1)
print('metrics.json written', len(json.dumps(out)))

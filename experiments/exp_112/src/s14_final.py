import json
O='out/'; J=lambda f: json.load(open(O+f))
m=J('s10_master.json'); L=J('s9_Lverdict.json'); sc=J('s8_screen.json')
an=J('s3_anchors.json'); fx=J('s2b_joint_fixed.json'); hy=J('s13_hyp.json'); cr=J('s11_cross.json')
out={
 'exp_id':'exp_112','date':'2026-08-27',
 'type':'read-only 측정 · 재학습 0 · zip 0 · 슬롯 0 · GPU 0 · 서버 CPU: 열 추출 1회(수 초)',
 'question':'현 패키지 CW 의 조정 가능한 상수 중 아직 꼭지점에 착륙하지 않은 것이 무엇이고 각각 LB 몇 점인가',
 'package':{'zip':'submissions/submit_CW_batresid_flip.zip','parent':'CS3 (LB 1151.6566018102)','CW_own_LB':'미측정'},
 'frame':'우리 규약 캘프레임 build_AC/work/tr2024 (train season==2024, 253,507행). 성분 벡터는 CW 출하 빌드 산출물. 솔버팩(ct_vec.npz) 미사용 (S210)',

 '🔴🔴_HEADLINE':{
  'n_live_adjustable':21,
  'n_print_only_or_dead_that_look_adjustable':8,
  'n_never_bought_on_any_LB':6,
  'n_unlanded_with_a_NUMBER_we_can_defend':0,
  'why':'이 프레임은 K·G 를 정확히 주지만 K·L(방향·크기)은 못 준다 — LB 로 대조 가능한 축 3개 중 2개의 부호가 반대다. 따라서 "얼마짜리"에 방어 가능한 숫자를 붙일 수 있는 미착륙 축은 0개이고, 답은 "무엇을 프로브할 가치가 있는가"로 바뀐다.',
  '★top3★':[
   {'rank':1,'constant':'blend.json /failtype_offset/c','now':0.0,
    'K_G':hy['K_G_c'],'K_G_cross_with_shipped_b':hy['K_G_bc'],
    'expected_LB':'가설에 따라 +3.30 ~ −11.87. 중심 가설(형제 b 의 전이비)에서 최대이득 +3.297',
    'verdict':'★프로브 가치 있음★ — 유일하게 (a) LB 점 0개 (b) 코드 변경 0줄 (c) 부스터 호출 0 (d) 가설 분리가 검출한계보다 큼(2.19점) 을 모두 만족한다',
    'probe':hy['★probe_recommendation★']},
   {'rank':2,'constant':'blend.json /beta_f','now':0.0,
    'K_G':'측정 불가','expected_LB':'미지 — 상한도 못 낸다',
    'verdict':'🔴 우리 캘프레임에서 ★원리적으로 측정 불가★. has_form = 0/253,507행 (동결 경계가 ≤2024년 말이라 2024 행은 전부 n_season ≤ 0). 배포(2025)에서만 발동한다'},
   {'rank':3,'constant':'model/meta.json /branch_mix w=0.5 (F 행) · /offset/beta_by_segment',
    'now':[0.5,[0.597044,0.492041]],'expected_LB':'미지',
    'verdict':'LB 점 0개. meta 자신이 fold_2024 argmin 은 w=1.0 이고 안 맞춘 대가가 valid_all −2.768점이라 적는다. 🔴 AA 팔 재추론이 필요해 이번 범위 밖'}]},

 '🔴_L_VERDICT':L,
 'positive_control_failtype_b':fx,
 'failtype_c':{**hy,'rho_with_other_live_axes':cr['ft_c_vs_others'],
   't_calframe':sc['failtype_c (출하 0.0)']['t'],
   '🔴_G_LEDGER':'형제 b 의 LB 점은 걸음 −0.05 (CV_LB.md:1048, 부모 BO 1115.9913864857 → 1118.6680789834). 현 출하 −0.036451 은 CA 의 5축 동시 해가 나중에 옮긴 값이다.'},
 'calframe_signal_screen':sc,
 'anchors':an,
 'master_table':m,
 'gates':{
  'G1_reproduce_CA_b_solve_12_digits':fx['gate_reproduce_CA_b_solve'],
  'G2_K_G_vs_package_ctrl_asof_psr':{'mine':sc['axis ctrl_asof_psr (출하 +0.006544)']['K_G'],
     'package':400309.19882517256*0.07119965607894004,'pct_diff':0.0068},
  'G3_K_G_vs_package_count':{'mine':an['anchor_count_CS3']['K_G_calframe'],'package':2.30510,'pct_diff':0.53},
  'G4_ledger_reproduction_count_residual':{'mine':an['anchor_count_CS3']['residual_LB_pts'],'S215_recorded':0.250},
  'G5_probe_gain_recompute_from_recorded_L_and_G':{
     'ctrl_asof_psr':[-3.0727297300999994,-3.0727297301],
     'beta_ins_off':[0.9596987465000002,0.9596987465]},
  'G6_m_logit_mix_recovery_max_resid':2.1094237467877974e-15,
  'G7_offline_CW_is_the_shipped_vector':'logit 차가 선언 d_adj 걸음과 5.62e-16 안에서 상수'},
 'limits':[
  '캘프레임 L 은 판정 근거가 아니다 (본 실험이 그것을 증명했다).',
  'aux_offsets[0] nact 의 raw 를 이 방에 보유하지 않아 m(logit_mix) 역산에 |m| ≤ 0.0139 의 미포함 항이 있다 (m 의 sd 0.196 의 7%).',
  'mate 내부 상수(w_ebm·beta_ins·Q beta·league_dev AA 사본)와 AA 팔 내부 상수(branch_mix·beta_by_segment)는 재추론 없이 K·G 를 못 잰다 — 이번 실험은 런 0 규율로 안 쟀다.',
  'b 의 LB 점은 부모 BO 세대다. 현 기반 캘프레임 G(1971.02)와 그 세대 G(1874.10)가 5.2% 다르다 — 두 G 로 모두 풀어 표에 실었다.']}
json.dump(out,open('metrics.json','w'),ensure_ascii=False,indent=1)
print('ok',len(json.dumps(out)))

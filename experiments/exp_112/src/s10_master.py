import json
O='out/'
sc=json.load(open(O+'s8_screen.json')); L=json.load(open(O+'s9_Lverdict.json'))
st=json.load(open(O+'s7_step.json')); an=json.load(open(O+'s3_anchors.json'))
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
K=100000.0/0.2498069
def g(k,f): return sc[k][f]
M=[]
def add(**kw): M.append(kw)
add(idx=1,constant='blend.json /failtype_offset/c', file='blend.json', value=0.0,
    live='산술 O — script.py 가 ft_c*(lg_wo − mu_wo) 를 최종 블렌드 로짓에 가산',
    kind='순수 오프셋 (파트너 Q 실패유형 way-off 로짓, 표 조회 없음)',
    K_G_calframe=g('failtype_c (출하 0.0)','K_G'),
    K_L_calframe_UNTRUSTED=g('failtype_c (출하 0.0)','K_L_cal'), t_calframe=g('failtype_c (출하 0.0)','t'),
    LB_points_on_this_axis=0, base_of_those_points=None,
    why_not_bought='측정 없음 — 이 축은 한 번도 제출된 적이 없다',
    ship_cost='blend.json 숫자 1개 + d_adj 종속 재유도.  코드 변경 0줄, 부스터 호출 증가 0')
add(idx=2,constant='blend.json /beta_f', file='blend.json', value=0.0,
    live='산술 O — 단 0.0 이면 폼 항 전체가 정확히 죽는다 (preds = p_blend + 0.0*d_eff)',
    kind='확률공간 가산 오프셋 (동결 폼 분해)',
    K_G_calframe=0.0, K_L_calframe_UNTRUSTED=0.0, t_calframe=None,
    LB_points_on_this_axis=None, base_of_those_points=None,
    why_not_bought='🔴 캘프레임에서 ★구조적으로 측정 불가★ — has_form = 0/253,507행. 동결 경계가 ≤2024년 말이라 2024 행은 전부 n_season ≤ 0. S32 계열',
    ship_cost='blend.json 숫자 1개')
add(idx=3,constant='blend.json /alpha', file='blend.json', value=bl['alpha'],
    live='산술 O — expit(alpha*m + ...)', kind='척도 (가중 계열)',
    K_G_calframe=g('alpha (출하 1.07911)','K_G'), K_L_calframe_UNTRUSTED=g('alpha (출하 1.07911)','K_L_cal'),
    t_calframe=g('alpha (출하 1.07911)','t'), LB_points_on_this_axis='다수 (alpha_derivation 블록)',
    why_not_bought='닫힘 — exp_078 잔여 +0.2546 · exp_106 sd 맞춤 후 정확히 0.0000 (S191). 🔴 캘프레임 L 이 양수인 것은 표본 안 암기가 척도 파라미터에 만드는 전형적 편향이라 근거가 아니다')
add(idx=4,constant='blend.json /beta_skew', file='blend.json', value=bl['beta_skew'],
    live='산술 O', kind='2차항 (모양 상수)',
    K_G_calframe=g('beta_skew (출하 −0.059151)','K_G'), K_L_calframe_UNTRUSTED=g('beta_skew (출하 −0.059151)','K_L_cal'),
    t_calframe=g('beta_skew (출하 −0.059151)','t'), LB_points_on_this_axis='beta_skew_derivation 블록 (BC 세대 프로브 + 꼭지점)',
    why_not_bought='기반 다름 — 꼭지점이 BC 세대 LB 점에서 풀렸다. 현 기반 LB 점 0개')
add(idx=5,constant='blend.json /axis_offsets[1] ctrl_asof_psr .u', file='blend.json', value=bl['axis_offsets'][1]['u'],
    live='산술 O (alpha 앞)', kind='순수 오프셋 (test.csv 원열)',
    K_G_calframe=g('axis ctrl_asof_psr (출하 +0.006544)','K_G'),
    K_L_calframe_UNTRUSTED=g('axis ctrl_asof_psr (출하 +0.006544)','K_L_cal'),
    t_calframe=g('axis ctrl_asof_psr (출하 +0.006544)','t'),
    LB_points_on_this_axis=1, base_of_those_points='BJ/BL 세대 (프로브 u=−0.006087, gain −3.0727297301)',
    residual_pts=0.0152,
    why_not_bought='사실상 닫힘 — 기록된 단일축 꼭지점의 112.6% 에 서 있고 잔여 +0.015. 🔴 기반이 다르나 K·G 가 그 세대 기록과 0.007% 일치해 기하는 안 변했다')
add(idx=6,constant='blend.json /axis_offsets[0] beta_ins_off .u', file='blend.json', value=bl['axis_offsets'][0]['u'],
    live='산술 O (alpha 앞)', kind='순수 오프셋 (동결 앵커 대비 in-season 레벨)',
    LB_points_on_this_axis=1, base_of_those_points='BI/BL 세대 (프로브 u=−0.006033, gain +0.9596987465)',
    residual_pts=0.0134,
    why_not_bought='사실상 닫힘 — 기록 꼭지점의 112.6%, 잔여 +0.013')
add(idx=7,constant='blend.json /axis_offsets[2] sp_pit_split .u', file='blend.json', value=bl['axis_offsets'][2]['u'],
    live='산술 O (alpha 앞)', kind='표 조회 · EB 수축 (deploy_splits.npz)',
    LB_points_on_this_axis=1, base_of_those_points='CR→CT (현 기반 직전)',
    residual_pts=0.042, why_not_bought='닫힘 — exp_109 가 잔여 +0.042 (브래킷 0.042~0.26) 로 종결')
add(idx=8,constant='blend.json /count_offsets/theta (12칸)', file='blend.json', value='표',
    live='산술 O', kind='순수 오프셋 (test.csv 원열 balls/strikes)',
    K_G_calframe=g('count_offsets 표 (출하 u=1)','K_G'), t_calframe=g('count_offsets 표 (출하 u=1)','t'),
    LB_points_on_this_axis=1, base_of_those_points='CT→CS3 (현 기반)', residual_pts=an['anchor_count_CS3']['residual_LB_pts'],
    why_not_bought='닫힘 — S215 기록 잔여 +0.250, 본 실험 독립 재현 +0.2473')
add(idx=9,constant='blend.json /bat_offsets/theta (526칸+miss)', file='blend.json', value='표',
    live='산술 O', kind='라벨 잔차 EB 수축 표 (batter_id 키)',
    K_G_calframe=g('bat_offsets 표 (출하 u=1)','K_G'), t_calframe=g('bat_offsets 표 (출하 u=1)','t'),
    LB_points_on_this_axis=1, base_of_those_points='CS3→CV (부호 +, LB −3.9905355919)',
    why_not_bought='측정 진행 중 — CW 가 부호를 뒤집어 0.65배로 다시 실었고 아직 LB 미측정')
add(idx=10,constant='blend.json /aux_offsets[0] v6_nn_hidden_nact .u', file='blend.json', value=bl['aux_offsets'][0]['u'],
    live='산술 O', kind='순수 오프셋', LB_points_on_this_axis=1, base_of_those_points='CA2 (프로브)',
    why_not_bought='닫힘 — S78 잔여 +0.0369, 합성 방향 최대 +0.04124')
add(idx=11,constant='blend.json /aux_offsets[1] aa_seed_sd18 .u', file='blend.json', value=0.0,
    live='산술 O — 배선돼 있고 u=0 이라 항이 정확히 0', kind='순수 오프셋',
    LB_points_on_this_axis=1, base_of_those_points='CA2 (프로브, 동시 해가 0 을 줬다)',
    why_not_bought='닫힘 — S78 과 같은 절, 잔여 +0.04 이하')
add(idx=12,constant='blend.json /w_v6 · /c_aa · /smooth_weights.T70', file='blend.json',
    value=[bl['w_v6'],bl['c_aa'],bl['smooth_weights']['T70']],
    live='산술 O', kind='가중 축 (전이비 0.1137)', LB_points_on_this_axis='없음(현 기반)',
    why_not_bought='크기 미달 — exp_106 이 교차적합으로 측정: 전량 걸음 표본 밖 −1.418, 절반 걸음 +2.355(t 4.97) → LB 환산 ★+0.27★ (원장 0.247 로 봐도 +0.58)')
add(idx=13,constant='mate/model/blend.json /w_deep', file='mate/model/blend.json', value=0.4,
    live='산술 O', kind='딥 계열 (전이비 0.348 — 계열 닫힘)', LB_points_on_this_axis=1, base_of_those_points='CF→CG',
    why_not_bought='닫힘 — 우리 정직면 argmax 0.43, 이동 이득 +0.1184(t 0.191) → LB +0.0412 (S145)')
add(idx=14,constant='mate/model/blend.json /w_v6 (내부)', file='mate/model/blend.json', value=0.6668397489,
    live='산술 O', kind='가중 축', LB_points_on_this_axis=1, base_of_those_points='BN→BO',
    why_not_bought='기반 다름 + 크기 미달 — exp_106 이 mate 내부 3축 합 +1.434 vs 귀무 기대 +1.17 로 측정')
add(idx=15,constant='mate/model/blend.json /w_ebm', file='mate/model/blend.json', value=-0.1,
    live='산술 O', kind='가중 축', LB_points_on_this_axis=0,
    why_not_bought='측정 없음 — 팀원이 정한 값. 🔴 우리가 옮기려면 파트너 팔 재추론(약 5분/판)이 필요하고, exp_106 이 이 계열 전체를 표본 밖 −1.4 로 닫았다')
add(idx=16,constant='mate/model/blend.json /beta_ins', file='mate/model/blend.json', value=0.03,
    live='산술 O', kind='순수 오프셋 (파트너 내부, in-season 레벨)', LB_points_on_this_axis='후보 AS·AT 가 0.00/0.06 를 쟀다 — 장부 대조 필요',
    why_not_bought='장부 대조 결과에 따름')
add(idx=17,constant='mate/model_q/meta.json /offset/beta', file='mate/model_q/meta.json', value=0.59072,
    live='산술 O — q_infer stage 3', kind='표 조회 · EB 수축 계열', LB_points_on_this_axis=0,
    why_not_bought='측정 없음 — 파트너 팔 재추론 필요. 🔴 이 값을 옮기면 aa_infer/q_infer 의 기록-대조 assert 6계열 9자리를 함께 고쳐야 한다 (S35)')
add(idx=18,constant='mate/model_q/meta.json /backoff/league_dev 4셀', file='mate/model_q/meta.json',
    value='4셀', live='산술 O', kind='표 조회 백오프',
    LB_points_on_this_axis='4셀 각각 1~3개 (BS4·BW·BX·CJ·CN·CQ·CR)',
    why_not_bought='닫힘 — exp_099 가 3셀 정산 완료, CR 이 마지막 1|2 를 착륙시켰다')
add(idx=19,constant='model/meta.json /offset/beta_by_segment (R .597044 · F .492041)', file='model/meta.json',
    value=[0.597044,0.492041], live='산술 O — aa_infer stage 3', kind='표 조회 · EB 수축 계열',
    LB_points_on_this_axis=0,
    why_not_bought='측정 없음 — AA 팔 재추론 필요. 후보 W 가 2022+2023 공급 블록에서 적합했고 그 뒤 한 번도 LB 로 안 옮겼다')
add(idx=20,constant='model/meta.json /branch_mix (F 행 0.5/0.5)', file='model/meta.json', value=0.5,
    live='산술 O — stage 2c', kind='가중 축 (AA 팔 내부)', LB_points_on_this_axis=0,
    why_not_bought='측정 없음 — 🔴 meta 자신이 fold_2024 argmin 은 w=1.0 이고 안 맞춘 대가가 valid_all −2.768점이라 기록한다. 로컬 값이고 LB 로 산 적 없다')
add(idx=21,constant='model/meta.json /backoff/league_dev 4셀 (AA 사본)', file='model/meta.json',
    value='4셀', live='산술 O', kind='표 조회 백오프', LB_points_on_this_axis=0,
    why_not_bought='측정 없음 — S66 이 이중 적재를 지적했으나 AA 사본은 한 번도 안 옮겼다. 실효 가중이 c_aa 0.25244 라 Q 사본보다 작다')
DEAD=[
 dict(key='blend.json /target_rate = 0.48589279261815', verdict='★PRINT-ONLY★ — f-string 1회, 산술 경로 0 (S209 재확인)'),
 dict(key='blend.json /d_form_eb_k = 100', verdict='★죽은 메타데이터★ — script.py 도 form_offset.py 도 이 키를 읽지 않는다. form_offset.py 가 K=100 을 하드코딩하고, 게다가 beta_f=0.0 이라 그 항 전체가 죽어 있다'),
 dict(key='mate/model/blend.json /target_rate = 0.4785', verdict='★PRINT-ONLY★'),
 dict(key='model/meta.json /offset/eb_k = 400', verdict='★PRINT-ONLY + 가드★ — aa_infer.py 3곳(대입·assert>0·print)뿐. 수축은 이미 lookup_d 표에 구워져 있어 이 숫자를 바꿔도 예측이 한 비트도 안 변한다'),
 dict(key='mate/model_q/meta.json /offset/eb_k = 400', verdict='★PRINT-ONLY + 가드★ — q_infer.py 3곳(514/536/620)뿐. 위와 같은 이유'),
 dict(key='mate/model_q/meta.json /logit_calibration/gamma = −0.029941', verdict='★죽은 코드★ — q_infer.py:694 가 "블렌딩 빌드: stage 4(gamma)를 적용하지 않는다" 로 preds = p_off 를 반환한다. 읽히고 assert 되지만 적용되지 않는다'),
 dict(key='model/meta.json /logit_calibration/gamma_by_segment (R −0.031877 · F −0.003383)', verdict='★블렌드 경로에서 죽은 코드★ — aa_infer 는 gamma 적용본을 _aa_ret[3] 로 내지만 script.py 는 _aa_ret[2](gamma 이전)를 쓴다. gamma 판본은 anchor_mode="AA" 항등 앵커에서만 쓰인다'),
 dict(key='blend.json /smooth_weights/S1 = S2 = 0.0', verdict='조정 불가 — 지분 0 인 평활 팔은 벡터 자체를 안 싣는다. 0 이 아니게 하려면 그 팔들을 새로 출하해야 한다'),
]
json.dump({'live_adjustable':M,'not_adjustable_or_dead':DEAD},open(O+'s10_master.json','w'),ensure_ascii=False,indent=1)
print(len(M),'live rows,',len(DEAD),'dead/print-only rows')

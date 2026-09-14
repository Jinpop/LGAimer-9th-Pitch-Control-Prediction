import json,numpy as np
O='out/'; K=100000.0/0.2498069
sc=json.load(open(O+'s8_screen.json')); an=json.load(open(O+'s3_anchors.json'))
PK='<SCRATCH>/cw/pkg/blend.json'
bl=json.load(open(PK,encoding='utf-8'))
a1=bl['axis_offsets'][1]; a0=bl['axis_offsets'][0]
rows=[]
# ① failtype b — LB 점 1개(팀원), G 는 캘프레임
KGb=sc['failtype_b (출하 −0.036451)']['K_G']; b=-0.03645076320272027
KLb=(2.6766924977+KGb*b*b)/(2*b)
rows.append(dict(axis='failtype_b',KL_cal=sc['failtype_b (출하 −0.036451)']['K_L_cal'],
  t=sc['failtype_b (출하 −0.036451)']['t'],KL_LB=KLb,
  LB_source='LB 이득 +2.6766924977 + 캘프레임 G 로 역산',sign_agrees=True))
# ② ctrl_asof_psr — 패키지가 LB 유도 L 을 직접 기록해 뒀다
KL1=K*a1['L_i_back_solved']; KG1=K*a1['G_ii']
chk=2*KL1*a1['probe_u']-KG1*a1['probe_u']**2
rows.append(dict(axis='axis ctrl_asof_psr',KL_cal=sc['axis ctrl_asof_psr (출하 +0.006544)']['K_L_cal'],
  t=sc['axis ctrl_asof_psr (출하 +0.006544)']['t'],KL_LB=KL1,
  LB_source=f"blend.json axis_offsets[1].L_i_back_solved (BL 세대 LB 프로브 u={a1['probe_u']}, gain {a1['probe_gain_pts']})",
  recomputed_probe_gain=chk, recorded_probe_gain=a1['probe_gain_pts'], sign_agrees=False))
KL0=K*a0['L_i_back_solved']; KG0=K*a0['G_ii']
rows.append(dict(axis='axis beta_ins_off',KL_cal=None,t=None,KL_LB=KL0,
  LB_source=f"blend.json axis_offsets[0].L_i_back_solved (LB 프로브 u={a0['probe_u']}, gain {a0['probe_gain_pts']})",
  recomputed_probe_gain=2*KL0*a0['probe_u']-KG0*a0['probe_u']**2,
  recorded_probe_gain=a0['probe_gain_pts'],sign_agrees=None))
# ③ count CS3
rows.append(dict(axis='count_offsets (CS3)',KL_cal=sc['count_offsets 표 (출하 u=1)']['K_L_cal'],
  t=sc['count_offsets 표 (출하 u=1)']['t'],KL_LB=an['anchor_count_CS3']['K_L_implied_by_LB'],
  LB_source='LB 이득 +0.7866627052 (부모 CT) + 캘프레임 G 로 역산',sign_agrees=False))
out={'🔴🔴_VERDICT':'캘프레임은 L 을 읽지 못한다. LB 로 대조 가능한 축 3개 중 ★2개의 부호가 반대★ 이고, 그중 하나(ctrl_asof_psr)는 |t| 3.83 으로 잡음이 아니다. 따라서 이 프레임의 K·L 은 어떤 축의 방향 근거도 되지 못한다.',
     '🟢_G_는_반대다':'같은 프레임의 K·G 는 독립 기록과 두 번 맞았다 — ctrl_asof_psr 28,500.5 vs 패키지 28,502.4 (0.007%) · count 2.2928 vs 패키지 2.30510 (0.5%).',
     'cross_check_rows':rows,
     'K':K}
json.dump(out,open(O+'s9_Lverdict.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))

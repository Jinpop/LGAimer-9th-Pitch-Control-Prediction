import json, numpy as np
O='out/'
K = 100000.0/0.2498069
p=np.load(O+'offline_CW.npy'); y=np.load(O+'labels.npy')
lg_mr=np.load(O+'ftmr_CW_0.npy'); lg_wo=np.load(O+'ftwo_CW_0.npy')
mu_mr,mu_wo=-0.4543162230525402,-2.009182627386164
s=p*(1-p); r=p-y
def pin(v):
    return s*(v - (s*v).sum()/s.sum())
db,dc = pin(lg_mr-mu_mr), pin(lg_wo-mu_wo)
KG=np.array([[K*(db*db).mean(),K*(db*dc).mean()],[K*(db*dc).mean(),K*(dc*dc).mean()]])
KLcal=np.array([-K*(db*r).mean(), -K*(dc*r).mean()])

b_ship=-0.03645076320272027
LB_gain_b=2.6766924977          # 팀원 exp_082 실측 (S54 · CV_LB)
# ---- b 를 양성 대조로: G 는 캘프레임을 믿고(S147) L 만 LB 로 눈금 ----
KG_bb=KG[0,0]
KL_b_LB=(LB_gain_b + KG_bb*b_ship**2)/(2*b_ship)
ratio=KL_b_LB/KLcal[0]
KL_LB=KLcal*ratio               # 같은 계열이므로 같은 비율을 c 에 이식

res={}
res['K_G_matrix_calframe']=KG.tolist()
res['rho_pinned']=float(KG[0,1]/np.sqrt(KG[0,0]*KG[1,1]))
res['K_L_calframe']=KLcal.tolist()
res['positive_control_b']={
 'shipped_step':b_ship,
 'LB_gain_observed_pts':LB_gain_b,
 'source':'팀원 exp_082 / 후보 teammate-b — 옛 기반(딥 팔 이전)의 LB 점 1개',
 'K_G_bb_calframe':KG_bb,
 'quadratic_cost_at_step_pts':KG_bb*b_ship**2,
 'K_L_implied_by_LB':KL_b_LB,
 'K_L_calframe':float(KLcal[0]),
 '★transfer_ratio_L_cal_to_LB★':float(ratio),
 'calframe_predicted_gain_at_step_pts':float(2*KLcal[0]*b_ship-KG_bb*b_ship**2),
 'calframe_overreads_gain_by':float((2*KLcal[0]*b_ship-KG_bb*b_ship**2)/LB_gain_b),
 'u_star_LB':float(KL_b_LB/KG_bb),
 'shipped_pct_of_vertex':float(100*b_ship/(KL_b_LB/KG_bb)),
 'max_gain_LB_pts':float(KL_b_LB**2/KG_bb),
 'residual_b_pts':float(KL_b_LB**2/KG_bb-LB_gain_b),
}
# ---- c 단독 ----
res['c_alone_LB_scaled']={
 'K_G':float(KG[1,1]),'K_L':float(KL_LB[1]),
 'u_star':float(KL_LB[1]/KG[1,1]),
 '★max_gain_LB_pts★':float(KL_LB[1]**2/KG[1,1]),
}
# ---- b·c 동시 (b 를 현 자리에서 다시 풀 수 있다고 가정) ----
u_j=np.linalg.solve(KG,KL_LB)
gj=float(2*KL_LB@u_j-u_j@KG@u_j)
gb=float(KL_b_LB**2/KG[0,0])
res['joint_bc_LB_scaled']={'u_star':u_j.tolist(),'max_gain_pts':gj,
  'b_only_max_pts':gb,'marginal_of_adding_c_pts':gj-gb}
# ---- c 만 열고 b 는 출하값 고정 (실제 후보의 모양) ----
KL_c_cond=KL_LB[1]-KG[0,1]*b_ship
res['c_conditional_on_shipped_b']={
 'K_L_c_conditional':float(KL_c_cond),
 'u_star':float(KL_c_cond/KG[1,1]),
 '★max_gain_LB_pts★':float(KL_c_cond**2/KG[1,1]),
 'note':'b 를 −0.03645 에 고정한 채 c 만 여는 경우.  교차항 −K·G_bc·b 가 c 의 기울기를 바꾼다'}
print(json.dumps(res,ensure_ascii=False,indent=1))
json.dump(res,open(O+'s2_joint.json','w'),ensure_ascii=False,indent=1)

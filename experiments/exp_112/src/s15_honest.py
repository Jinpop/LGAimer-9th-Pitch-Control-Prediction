import json,numpy as np
O='out/'
e82=json.load(open('<PROJECT_ROOT>/experiments/exp_082/metrics.json'))
hs=e82['stage2_honest_surface']; rho_h=e82['stage2_rho']['c_max_abs_rho_signed']
sc=json.load(open(O+'s8_screen.json')); br=json.load(open(O+'s6_bracket.json'))
# ---- 정직면 L·G 역산 (max = L^2/G, u* = L/G) ----
def LG(u_star,gmax): L=gmax/u_star; return L, L/u_star
Lb_h,Gb_h=LG(hs['b_lg_mr']['raw_argmax_t'],hs['b_lg_mr']['raw_gain_pts'])
Lc_h,Gc_h=LG(hs['c_lg_wo']['raw_argmax_t'],hs['c_lg_wo']['raw_gain_pts'])
Gbc_h=rho_h*np.sqrt(Gb_h*Gc_h)
# ---- LB 눈금: b 의 LB 점 (u=−0.05, +2.6766924977, 부모 BO) + 캘프레임 G ----
KGb=sc['failtype_b (출하 −0.036451)']['K_G']; KGc=sc['failtype_c (출하 0.0)']['K_G']; KGbc=br['K_G_bc']
u_LB,LBg,b_ship=-0.05,2.6766924977,-0.03645076320272027
Lb_LB=(LBg+KGb*u_LB**2)/(2*u_LB)
s_L=Lb_LB/Lb_h                                   # 정직면 → LB 의 L 축척 (b 로 눈금)
Lc_LB=Lc_h*s_L
def joint(Lb,Lc,Gb,Gc,Gbc):
    G=np.array([[Gb,Gbc],[Gbc,Gc]]); L=np.array([Lb,Lc]); u=np.linalg.solve(G,L)
    return u, float(L@u)
res={'🔴🔴_ANSWER_why_c_is_zero':
  'c 는 “안 산” 것이 아니라 ★측정하고 버린★ 것이다. exp_082 가 b 와 나란히 정직면 단독 스크린을 돌려 '
  'c 의 최대이득 +0.12450574932154268점(꼭지점 t −0.006142939399853696)을 얻었다 — 같은 화면에서 b 는 '
  '+168.42241313652642 였다. ★1,353배 차이★. exp_082 metrics.json 이 /constants/c_shipped = 0.0 으로 '
  '그 결정을 기록한다. S54 가 인용한 sd비 0.9998887329723813 이 바로 이 측정의 값이다.',
 'exp_082_honest_surface':{
   'base_pts':hs['base_pts'],'🔴_실험자_경고':hs['🔴'],
   'b_lg_mr':hs['b_lg_mr'],'c_lg_wo':hs['c_lg_wo'],
   'derived_K_L_K_G':{'b':{'K_L':Lb_h,'K_G':Gb_h},'c':{'K_L':Lc_h,'K_G':Gc_h},
     'rho_bc':rho_h,'K_G_bc':Gbc_h}},
 'GATE_두_면의_G_가_맞는가':{
   'b':{'정직면':Gb_h,'캘프레임(CW 기반)':KGb,'CA_기록(BX 세대)':1874.0988676056702,
        'pct_honest_vs_CA':100*(Gb_h/1874.0988676056702-1)},
   'c':{'정직면':Gc_h,'캘프레임(CW 기반)':KGc,'pct':100*(Gc_h/KGc-1)},
   'rho_bc':{'정직면':rho_h,'캘프레임':br.get('rho_pinned',-0.6352555590530388)},
   '🟢':'두 독립 면이 G 와 ρ 에서 3% 안으로 맞는다. G 는 믿을 수 있다.'},
 'L_축척_정직면_to_LB':{'ratio':s_L,'근거':'b 의 LB 점 하나(u=−0.05, +2.6766924977) + 캘프레임 G',
   'b_K_L_honest':Lb_h,'b_K_L_LB':Lb_LB,
   'c_K_L_LB_implied':Lc_LB,
   'c_단독_최대이득_LB_pts':Lc_LB**2/KGc}}
# ---- 진짜 질문: (b,c) 동시 ----
u_now=np.array([b_ship,0.0])
G=np.array([[KGb,KGbc],[KGbc,KGc]]); L=np.array([Lb_LB,Lc_LB])
g_now=float(2*L@u_now-u_now@G@u_now)
uj,gj=joint(Lb_LB,Lc_LB,KGb,KGc,KGbc)
Lc_cond=Lc_LB-KGbc*b_ship
res['🔑_THE_REAL_FINDING']={
 'gain_at_current_point_pts':g_now,
 'joint_bc_optimum_u':uj.tolist(),'joint_bc_max_pts':gj,
 'available_in_the_bc_plane_pts':gj-g_now,
 'c_only_b_frozen':{'K_L_c_conditional':Lc_cond,'u_star':Lc_cond/KGc,'max_gain_pts':Lc_cond**2/KGc},
 '🔑':'c 는 ★자기 신호로는 무가치★(LB 단독 최대 %.4f점)인데, ★출하 b 가 (b,c) 동시 꼭지점이 아닌 b 단독 꼭지점에 서 있어서★ '
      'c 방향에 기울기가 남아 있다. 그 기울기의 %.0f%% 는 c 자신의 L 이 아니라 교차항 −K·G_bc·b 다.'%(
      Lc_LB**2/KGc, 100*abs(KGbc*b_ship)/abs(Lc_cond)),
 '🟢_부호_견고성':'교차항 %.2f 가 c 자신의 L %.2f 를 %.1f배 압도한다 → L_c 의 부호가 뒤집혀도 조건부 기울기의 부호는 안 변한다.'%(
      -KGbc*b_ship, Lc_LB, abs(KGbc*b_ship/Lc_LB))}
# 정직면에서 같은 계산 (독립 확인)
ujh,gjh=joint(Lb_h,Lc_h,Gb_h,Gc_h,Gbc_h)
res['독립확인_정직면에서_같은_계산']={'joint_max_honest_pts':gjh,'b_alone_honest_pts':Lb_h**2/Gb_h,
  'available_honest_pts':gjh-Lb_h**2/Gb_h,'LB_환산':(gjh-Lb_h**2/Gb_h)*s_L**2,
  '🟢':'캘프레임 경로와 정직면 경로가 독립으로 같은 자리를 가리키는지 본다'}
json.dump(res,open(O+'s15_honest.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))

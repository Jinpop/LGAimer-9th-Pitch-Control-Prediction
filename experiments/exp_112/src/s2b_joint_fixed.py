import json,numpy as np
O='out/'; K=100000.0/0.2498069
sc=json.load(open(O+'s8_screen.json')); br=json.load(open(O+'s6_bracket.json'))
KGb=sc['failtype_b (출하 −0.036451)']['K_G']; KGc=sc['failtype_c (출하 0.0)']['K_G']; KGbc=br['K_G_bc']
KLb_cal=sc['failtype_b (출하 −0.036451)']['K_L_cal']; KLc_cal=sc['failtype_c (출하 0.0)']['K_L_cal']
u_LB=-0.05; LB=2.6766924977; b_ship=-0.03645076320272027
KG_CA=1874.0988676056702                     # blend.json CA_one_dimensional_LB_solves.b.K_G (BX 세대 프레임)
res={'🔴_G_LEDGER_정정':'CV_LB.md:1048 원문 — 그 LB 점의 걸음은 b = −0.05 이고 현 출하값 −0.036451 이 아니다. CA 가 5축 동시 해에서 나중에 옮겼다.',
 'gate_reproduce_CA_b_solve':{'my_2KL_with_CA_G':2*((LB+KG_CA*u_LB**2)/(2*u_LB)),
   'CA_recorded_2KL':-147.23879333428016,
   'my_vertex_with_CA_G':(LB+KG_CA*u_LB**2)/(2*u_LB)/KG_CA,'CA_recorded_vertex':-0.03928255757456141}}
for tag,KG in (('현_기반_캘프레임_G',KGb),('CA_가_쓴_BX세대_G',KG_CA)):
    KL=(LB+KG*u_LB**2)/(2*u_LB)
    res[tag]={'K_G_b':KG,'K_L_b_from_LB':KL,'u_star':KL/KG,'max_gain_pts':KL*KL/KG,
      'gain_at_shipped_b':2*KL*b_ship-KG*b_ship**2,
      'residual_b_pts':KL*KL/KG-(2*KL*b_ship-KG*b_ship**2),
      'shipped_pct_of_vertex':100*b_ship/(KL/KG),
      'transfer_ratio_L_cal_to_LB':KL/KLb_cal}
r=res['현_기반_캘프레임_G']['transfer_ratio_L_cal_to_LB']
band={'중심 %.5f'%r:r,'절반 %.5f'%(r/2):r/2,'두배 %.5f'%(2*r):2*r,'L_c=0':0.0}
c={}
for k,rat in band.items():
    KLc=KLc_cal*rat; cond=KLc-KGbc*b_ship
    c[k]=dict(ratio=rat,K_L_c=KLc,alone_max_pts=KLc*KLc/KGc,
      cond_on_shipped_b_max_pts=cond*cond/KGc,u_star_cond=cond/KGc)
grid=np.linspace(-0.06,0.005,1301)
def gain(u,KLc): 
    L=KLc-KGbc*b_ship; return 2*L*u-KGc*u*u
w=np.array([min(gain(u,c[k]['K_L_c']) for k in c) for u in grid]); i=int(w.argmax())
res['failtype_c_bracket_corrected']=c
res['failtype_c_minimax']={'step':float(grid[i]),'worst_case_gain_pts':float(w[i]),
  'gain_by_scenario':{k:float(gain(grid[i],c[k]['K_L_c'])) for k in c}}
res['🔴_still_untrusted']='위 브래킷은 캘프레임 L 에 의존한다. 같은 프레임이 ctrl_asof_psr(|t| 3.83)·count 의 부호를 틀렸으므로 ★판정 근거가 아니다★. 프로브 크기 고르는 데만 쓴다.'
json.dump(res,open(O+'s2b_joint_fixed.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))

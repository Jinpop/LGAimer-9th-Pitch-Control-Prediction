import json,numpy as np
O='out/'
d=json.load(open(O+'s6_bracket.json'))
KGc=d['failtype_c']['K_G']; KGbc=d['K_G_bc']; b=-0.03645076320272027
scen={k:v['K_L_c'] for k,v in d['failtype_c_bracket'].items()}
def gain(u,KLc): 
    Lc=KLc-KGbc*b
    return 2*Lc*u-KGc*u*u
grid=np.linspace(-0.06,0.005,1301)
worst=np.array([min(gain(u,KL) for KL in scen.values()) for u in grid])
i=int(worst.argmax())
res={'K_G_c':KGc,'K_G_bc':KGbc,'b_shipped':b,
 'scenarios_K_L_c':scen,
 'per_scenario_vertex_and_max':{k:dict(u_star=(v-KGbc*b)/KGc,max_pts=(v-KGbc*b)**2/KGc) for k,v in scen.items()},
 '★minimax_step★':float(grid[i]),'★minimax_worst_case_gain_pts★':float(worst[i]),
 'minimax_gain_by_scenario':{k:float(gain(grid[i],v)) for k,v in scen.items()},
 'central_vertex_0.65_shrink':{'step':0.65*(scen['앵커_b_비율_0.1809']-KGbc*b)/KGc,
   'gain_by_scenario':{k:float(gain(0.65*(scen['앵커_b_비율_0.1809']-KGbc*b)/KGc,v)) for k,v in scen.items()}},
 'note':'u=0 이 현 출하값. gain 은 현 패키지 대비 LB 점 증분. b 는 −0.036451 에 고정한 모형(축 하나만 여는 후보의 모양).'}
json.dump(res,open(O+'s7_step.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(res,ensure_ascii=False,indent=1))

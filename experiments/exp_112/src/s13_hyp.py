import json,numpy as np
O='out/'
sc=json.load(open(O+'s8_screen.json')); br=json.load(open(O+'s6_bracket.json'))
KGc=sc['failtype_c (출하 0.0)']['K_G']; KGbc=br['K_G_bc']; b=-0.03645076320272027
KLc_cal=sc['failtype_c (출하 0.0)']['K_L_cal']
H={'① b 계열 전이 (+0.18936) — 같은 파일·같은 계열, LB 점 1개':0.18935876969068458,
   '② ctrl_asof_psr 형 부호반전 (−0.41328)':165.65553288744022/-400.8241719988514,
   '③ count 형 부호반전 (−1.51239)':1.5397416180211632/-1.0180818316058955,
   '④ L_c = 0 (신호 없음)':0.0}
def gain(u,KLc): return 2*(KLc-KGbc*b)*u-KGc*u*u
rows={}
for k,r in H.items():
    KLc=KLc_cal*r; cond=KLc-KGbc*b
    rows[k]=dict(ratio=r,K_L_c=KLc,K_L_c_conditional=cond,u_star=cond/KGc,
                 max_gain_pts=cond*cond/KGc,
                 gain_at_probe_minus_0p0075=gain(-0.0075,KLc),
                 gain_at_minus_0p01765=gain(-0.01765,KLc))
sep=rows['① b 계열 전이 (+0.18936) — 같은 파일·같은 계열, LB 점 1개']['gain_at_probe_minus_0p0075']-\
    rows['② ctrl_asof_psr 형 부호반전 (−0.41328)']['gain_at_probe_minus_0p0075']
out={'K_G_c':KGc,'K_G_bc':KGbc,'b_shipped':b,'K_L_c_calframe_raw':KLc_cal,
 'hypotheses':rows,
 '★probe_recommendation★':{'u':-0.0075,'quadratic_cost_pts':KGc*0.0075**2,
   'hypothesis_separation_pts':sep,
   'why_this_size':'가설 ①과 ②가 %.2f점 갈린다 — 팀 원장 검출한계 +1.4 를 넘는다. 더 키우면 하방만 커진다(S42)'%sep,
   'ship_cost':'blend.json 숫자 1개 + d_adj 종속 재유도. 코드 0줄 · 부스터 호출 0 · A단'},
 '🔴_no_floor':'“L_c=0 이어도 +1.06” 은 성립하지 않는다. ②형 반전이면 같은 걸음이 %.2f점을 잃는다. 방향은 사야 한다.'%rows['② ctrl_asof_psr 형 부호반전 (−0.41328)']['gain_at_minus_0p01765']}
json.dump(out,open(O+'s13_hyp.json','w'),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))

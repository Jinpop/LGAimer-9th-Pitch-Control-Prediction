import json,numpy as np
O='out/'; J=lambda f: json.load(open(O+f))
hs=J('s15_honest.json'); L=J('s9_Lverdict.json'); sc=J('s8_screen.json')
an=J('s3_anchors.json'); fx=J('s2b_joint_fixed.json'); cr=J('s11_cross.json'); m=J('s10_master.json')
KGc=sc['failtype_c (출하 0.0)']['K_G']; F=hs['🔑_THE_REAL_FINDING']
us=F['c_only_b_frozen']['u_star']; M=F['c_only_b_frozen']['max_gain_pts']
shr={f'{f:.2f}':{'u':f*us,'expected_gain_pts':M*(2*f-f*f)} for f in (0.5,0.65,0.8,1.0)}
out=J('../metrics.json') if False else json.load(open('metrics.json'))
out['🔴🔴_HEADLINE']={
 'n_live_adjustable':21,'n_print_only_or_dead_that_look_adjustable':8,
 'n_never_bought_on_any_LB':6,
 '★답★':'아직 꼭지점에 착륙하지 않은 축은 ★1개★ 다 — 그리고 그것은 상수 하나가 아니라 ★(failtype b, failtype c) 평면★ 이다. '
        '나머지 조정 가능한 상수는 전부 (a) 잔여가 0.25점 미만으로 닫혔거나 (b) 계열이 닫혔거나 (c) 재추론 없이는 못 재거나 (d) 애초에 산술에 안 들어간다.',
 '★top3★':[
  {'rank':1,'axis':'blend.json /failtype_offset/c  (그리고 그것이 여는 (b,c) 평면)',
   'now':0.0,
   '★LB_expected★':{'c 만 옮김(b 동결)':F['c_only_b_frozen']['max_gain_pts'],
                    '(b,c) 동시 재착륙':F['available_in_the_bc_plane_pts'],
                    '독립_정직면_경로_확인':hs['독립확인_정직면에서_같은_계산']['LB_환산']},
   'u_star_c_only':us,'shrunk_steps':shr,
   'why_not_bought_yet':'exp_082 가 c 를 ★단독으로★ 스크린해 무가치(정직면 +0.1245, LB 환산 +0.0022)로 판정하고 0 에 뒀다. '
     '옳은 판정이었다. 그러나 ★(b,c) 동시 해는 아무도 풀지 않았다★ — c 의 값은 자기 신호가 아니라 '
     '출하 b 가 단독 꼭지점에 서 있어서 남은 교차 기울기다.',
   'sign_robustness':F['🟢_부호_견고성'],
   'ship_cost':'blend.json 숫자 1개(+선택적으로 b 도 1개) + d_adj 종속 재유도. 코드 0줄 · 부스터 호출 0 · A단'},
  {'rank':2,'axis':'blend.json /beta_f','now':0.0,
   '★LB_expected★':'미지 — 상한도 못 낸다',
   'why_not_bought_yet':'🔴 우리 캘프레임에서 ★원리적으로 측정 불가★: has_form = 0/253,507행. '
     '동결 경계가 ≤2024년 말이라 2024 행은 전부 n_season ≤ 0 이다. 배포(2025)에서만 발동한다.'},
  {'rank':3,'axis':'model/meta.json /branch_mix (F 행 0.5/0.5) · /offset/beta_by_segment (R .597044 F .492041)',
   'now':[0.5,[0.597044,0.492041]],'★LB_expected★':'미지 (LB 점 0개)',
   'why_not_bought_yet':'meta 자신이 fold_2024 argmin 은 w=1.0 이고 안 맞춘 대가가 valid_all −2.768점이라 적는다. '
     'AA 팔 재추론이 필요해 이번 범위 밖.'}]}
out['🔴🔴_failtype_c_ANSWER']=hs
out['🔴_L_VERDICT']=L
out['positive_control_failtype_b']=fx
out['failtype_c_cross_rho']=cr['ft_c_vs_others']
out['🔴_그리고_이_실험이_스스로_증명한_것']= \
 '캘프레임 L 을 믿었다면 c 의 조건부 최대이득을 +3.30 으로 보고했을 것이다. 정직면 + 형제 b 의 LB 점으로 다시 풀면 +1.16 이다. '\
 '2.8배 과대였고, 같은 프레임이 ctrl_asof_psr(|t| 3.83)·count 의 ★부호★ 를 틀렸다. ★캘프레임은 G 전용이다.★'
json.dump(out,open('metrics.json','w'),ensure_ascii=False,indent=1)
print(json.dumps({'shrunk':shr,'c_only_max':M,'plane':F['available_in_the_bc_plane_pts']},ensure_ascii=False,indent=1))

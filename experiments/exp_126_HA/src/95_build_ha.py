#!/usr/bin/env python3
"""후보 HA — 부모 FT2 에 HistGB 팔 + 2차 스태커를 한 항으로 추가한다.

근거(전부 로컬 측정, LB 적합 0):
  exp_122/123  5개 모형족 13설계 정직면 스크린 -> HistGB 가 최고
  exp_124      결합 규칙 3종 교차적합 -> 2차 스태커가 전역 스칼라의 1.57배
  exp_126      2차 스태커 x 5팔 교차적합 -> hgb +3.053 (rf/xgb 는 음수)
  전이 0.574(팀 원장 '팔 추가') -> LB 추정 +1.75
출하 진폭은 정직면 theta 의 0.65 배다 (S216: 정직면이 최적 걸음을 2.3배 과대평가,
S294: tau ~ 0.61). 2차식이라 최적의 65% 지점에서 이득의 88% 를 지킨다.

항등 앵커: theta=[0,0,0] 이면 _hgb_shift 가 정확히 0.0 이고 x+0.0 은 IEEE-754 에서
정확하므로 부모 출력과 바이트 동일해야 한다.
"""
import json, os, shutil, subprocess, sys, zipfile, hashlib
ROOT='<PROJECT_ROOT>'
PAR=f'{ROOT}/submissions/submit_FT2_failtypeC_probe.zip'
W=f'{ROOT}/experiments/exp_126_HA/work'; OUT=f'{ROOT}/submissions/submit_HA_hgbstack.zip'
THETA_HONEST=[-0.053706, 0.138554, -0.002965]
SHRINK=0.65
THETA=[round(t*SHRINK,9) for t in THETA_HONEST]
IDENTITY = os.environ.get('HA_IDENTITY')=='1'
if IDENTITY:
    THETA=[0.0,0.0,0.0]; OUT=f'{ROOT}/experiments/exp_126_HA/work/anchor.zip'
print('theta 정직면',THETA_HONEST,'-> 출하',THETA,'(수축',SHRINK,')')

shutil.rmtree(W,ignore_errors=True); os.makedirs(W)
with zipfile.ZipFile(PAR) as z: z.extractall(f'{W}/pkg')
P=f'{W}/pkg'
# --- 팔 자산 반입
os.makedirs(f'{P}/hgb',exist_ok=True)
for f in ['hgb_le2024.joblib','hgb_meta.json']:
    src=f'{ROOT}/experiments/exp_126_HA/assets/{f}'
    assert os.path.exists(src), f'자산 없음: {src}'
    shutil.copy2(src,f'{P}/hgb/{f}')
meta=json.load(open(f'{P}/hgb/hgb_meta.json'))
assert meta['sklearn']=='1.8.0', meta['sklearn']

# --- script.py 수정: 삽입점 하나, 치환 하나
sp=f'{P}/script.py'; s=open(sp).read()
OLDBLK = """        p_blend = expit(alpha * logit_mix
                        + beta_skew * logit_mix * logit_mix + d_adj
                        + ft_b * (ft_lg_mr - ft_mu_mr)
                        + ft_c * (ft_lg_wo - ft_mu_wo)
                        + _aux_shift
                        + _cnt_shift)"""
assert s.count(OLDBLK)==1, '삽입점이 유일하지 않다'
NEWBLK = '''        # ---------------------------------------------- HA: HistGB stacker ----
        # 새 팔 하나(sklearn HistGradientBoosting, 원본 test 47열, <=2024 학습)를
        # 2차 스태커로 최종 블렌드 로짓에 더한다.  z 는 그 팔의 로짓에서 ★train 에서
        # 얼린 중심 상수★ 를 뺀 값이다 - test 통계를 쓰지 않으므로 규칙 2-4 를 지킨다.
        # 범주 사전·결측 대체값도 train 에서 확정돼 meta 에 실려 있고, 따라서 이 항은
        # test.csv 의 ★그 행 하나★ 만으로 결정된다.
        # theta == [0,0,0] 이면 세 항이 정확히 0.0 이고 x+0.0 은 IEEE-754 에서 정확하므로
        # 부모와 바이트 동일하다 (항등 앵커).
        _Zpost = (alpha * logit_mix
                  + beta_skew * logit_mix * logit_mix + d_adj
                  + ft_b * (ft_lg_mr - ft_mu_mr)
                  + ft_c * (ft_lg_wo - ft_mu_wo)
                  + _aux_shift
                  + _cnt_shift)
        _hgb_shift = np.zeros(n, dtype=np.float64)
        _hg_th = [float(v) for v in bl.get("hgb_stacker", {}).get("theta", [0.0, 0.0, 0.0])]
        assert len(_hg_th) == 3, "hgb_stacker.theta 는 3개여야 한다"
        if any(v != 0.0 for v in _hg_th):
            import joblib as _jl
            _hgd = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hgb")
            _hgm = json.load(open(os.path.join(_hgd, "hgb_meta.json"), encoding="utf-8"))
            assert _hgm["sklearn"] == "1.8.0", "hgb 모델의 sklearn 빈티지가 다르다"
            _hgb = _jl.load(os.path.join(_hgd, "hgb_le2024.joblib"))
            _hdf = pd.read_csv(os.path.join(TEST_DIR, "test.csv"),
                               encoding="utf-8-sig", usecols=_hgm["features"])
            assert len(_hdf) == n, "hgb 프레임 행 수가 두 팔과 다르다"
            _hX = _hdf[_hgm["features"]].copy()
            for _c in _hgm["cats"]:
                _hX[_c] = pd.Index(_hgm["vocab"][_c]).get_indexer(_hX[_c].astype(str))
            _hX = _hX.astype(np.float32).fillna(float(_hgm["na_fill"]))
            _hp = _hgb.predict_proba(_hX.values)[:, 1]
            _hz = (np.log(np.clip(_hp, 1e-9, 1.0 - 1e-9)
                          / (1.0 - np.clip(_hp, 1e-9, 1.0 - 1e-9)))
                   - float(_hgm["z_center_frozen"]))
            _hgb_shift = (_hg_th[0] * _hz
                          + _hg_th[1] * _hz * _Zpost
                          + _hg_th[2] * _hz * _hz)
            print(f"HA HistGB 스태커: theta={_hg_th!r} "
                  f"z_center={_hgm['z_center_frozen']!r} "
                  f"|z| max={float(np.abs(_hz).max()):.6f} "
                  f"이동!=0 인 행={int(np.count_nonzero(_hgb_shift))}/{n} "
                  f"|shift| max={float(np.abs(_hgb_shift).max()):.8f}")
        else:
            print("HA HistGB 스태커: theta 전부 0.0 - 부모와 바이트 동일 (항등 앵커)")
        p_blend = expit(_Zpost + _hgb_shift)'''
s=s.replace(OLDBLK,NEWBLK); open(sp,'w').write(s)

# --- blend.json 에 계수와 출처를 사실대로 적는다
bj=f'{P}/blend.json'; b=json.load(open(bj))
b['hgb_stacker']={'theta':THETA,'theta_honest_face':THETA_HONEST,'shrink':SHRINK,
 'arm':'sklearn HistGradientBoostingClassifier, 원본 test 47열, season<=2024 학습',
 'form':'p_blend = expit(Zpost + t0*z + t1*z*Zpost + t2*z*z), z = logit(hgb) - z_center_frozen',
 'derived_from':('exp_122/123 이 5개 모형족 13설계를 정직면 혼합으로 스크린해 HistGB 를 골랐고, '
   'exp_124 가 결합 규칙 3종을 투수 id 2겹 교차적합으로 비교해 2차 스태커를 골랐으며, '
   'exp_126 이 그 규칙을 5팔에 씌워 hgb 교차적합 +3.053점을 얻었다. '
   '출하 진폭은 그 정직면 theta 의 0.65 배다(S216 정직면 과대평가 2.3배 · S294 tau~0.61). '
   'LB 로 적합한 양은 0개이고, 이 축의 LB 측정은 이번이 처음이다.'),
 'cross_fitted':True,'fold':'pitcher_id 해시 2겹',
 'gain_honest_cv_pts':3.0534439187411895,'transfer_used':0.574,'lb_estimate_pts':1.7526768093574425}
json.dump(b,open(bj,'w'),ensure_ascii=False,indent=1)

# --- 재압축 (부모 파일 순서 유지)
if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(PAR) as pz: names=pz.namelist()
extra=[f'hgb/{x}' for x in ['hgb_le2024.joblib','hgb_meta.json']]
with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
    for nm in names+extra:
        p=os.path.join(P,nm)
        if os.path.isdir(p): continue
        z.write(p,nm)
print('zip',OUT, os.path.getsize(OUT)/1e6,'MB')
print('sha256', hashlib.sha256(open(OUT,'rb').read()).hexdigest())

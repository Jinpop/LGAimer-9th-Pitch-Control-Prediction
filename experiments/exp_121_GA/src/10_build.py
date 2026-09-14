#!/usr/bin/env python3
"""후보 GA — `ctrl_asof_batn` (asof_batter_n 선형 사후 축).

계열 근거: 원열 ★선형★ 축은 LB 3승 1패 (prev5mid 승 · middle_rate tau=0.617 ·
success_rate tau=2.22 · batmid 패).  면 잔차로 ★모양까지★ 적합한 축(EA, tau=-0.10)
과는 다른 계열이다 — EA 는 그쪽을 죽였지 이쪽을 죽이지 않았다.

🔴 S301: 산문 수정은 패키지와 이 스크립트 둘 다에 건다.
"""
import json, zipfile, hashlib, os, sys

ROOT='<PROJECT_ROOT>'
PARENT=f'{ROOT}/submissions/submit_FT2_failtypeC_probe.zip'
STAGE=sys.argv[1] if len(sys.argv)>1 else 'probe'      # probe | ship
DADJ_DELTA=float(sys.argv[2]) if len(sys.argv)>2 else 0.0
SHIP=f'{ROOT}/submissions/submit_GA_batn{"" if STAGE=="ship" else "_v1"}.zip'

NAME='ctrl_asof_batn'; COL='asof_batter_n'
CENTER=5402.548777745782; SCALE=4011.2322072820784
U_STAR=-0.010194619869518508; A=0.5
U=A*U_STAR
CEIL=2.5433204265966314; K_L=-249.47672979951457

z=zipfile.ZipFile(PARENT)
b=json.loads(z.read('blend.json')); s=z.read('script.py').decode('utf-8')
d0=b['d_adj']

def g(a,tau): return CEIL*(2*a*tau-a*a)
b['axis_offsets'].append({
 "name":NAME,"u":U,"center":CENTER,"scale":SCALE,"fitted_at_inference":False,
 "frozen_from":("TRAIN season-2024 프레임 253,507행. center = 그 열의 평균, scale = 모분산 sd. "
   "동결 스칼라이고 test 통계가 아니다. (실측: 결측 0행, 0값 88행 = 타자 첫 등장, 범위 0~13,927)"),
 "raw_definition":("test.csv 컬럼 asof_batter_n, 변환 없음. 그 행 자신의 컬럼 하나다. "
   "축 2 `ctrl_asof_psr` 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 읽으므로 "
   "test.csv 재파싱 횟수는 부모와 동일하다. 행 대응도 축 2 와 같은 _pos_asof 를 재사용한다."),
 "nan_policy":("raw 가 비유한이면 그 행의 이 축 이동은 정확히 0.0 이다. 센티넬·평균 대입 없음. "
   "train-2024 프레임에는 결측이 0행이다 (n=0 은 결측이 아니라 '이력 없음' 이라는 실값이라 "
   "표준화 후 -1.347 로 정상 이동한다)."),
 "applied_where":("기존 9축과 완전히 같은 자리 — script.py 의 _shift 누적에 들어가고 "
   "logit_mix = logit_mix + _shift 로 더해진다. alpha/beta_skew 변환 앞."),
 "derivation":{
  "면":"exp111 out-of-time 정직면 (팔 season<=2023 학습 -> 2024행 253,507). 리더보드 점수 0회 사용.",
  "부모 8축 잔차화":{"K_L":K_L,"ceiling":CEIL,"u_star":U_STAR,
    "투수 반분할 교차적합 실현":2.511737142657627,
    "최대 rho(부모 축)":0.12688849428728202},
  "★왜 이 축인가★":("부모 9축 중 ★가장 직교하다★(최대 rho 0.127). 그리고 교차적합 실현(2.512)이 "
    "표본 안 ceiling(2.543)과 거의 같아 과적합이 없다."),
  "★계열 구분★":("원열 ★선형★ 축은 LB 3승 1패다 — prev5mid 승 · middle_rate tau=0.617 · "
    "success_rate tau=2.22 · batmid 패(S294).  후보 EA(2026-08-30, LB -2.4971, tau=-0.10)가 "
    "죽인 것은 ★면 잔차로 모양까지 적합한 축★ 이고 이 계열이 아니다.  "
    "이 축은 방향이 데이터로 고정돼 있고 부호·크기만 정직면에서 온다 — 면 의존도가 한 겹 낮다."),
  "진폭":{"a":A,"u":U,
    "근거":"tau 중심 0.617 의 0.81배 수축(S216·S294). 손익분기 tau = a/2 = 0.25.",
    "지불표":{f"tau={t}":round(g(A,t),3) for t in [-0.3,0,0.25,0.617,1.0,2.22]}},
  "d_adj":{"stage":STAGE,"적용한 d_adj 이동":DADJ_DELTA,
    "감도(실측)":0.24749,
    "note":("stage=probe 판본은 d_adj 를 안 옮기고 cov 프레임에서 평균 표류를 실측하는 용도다. "
            "stage=ship 판본만 제출한다.")}},
})
b['candidate']='GA' if STAGE=='ship' else 'GA_v1'
b['d_adj']=d0+DADJ_DELTA
b['CZ_d_adj_provenance_inherited']=b.get('CZ_d_adj_provenance_inherited',b['derived_from'])
b['derived_from']=("TRAIN 유래 상수다. 🔴 이 문장은 후보 GA 가 새로 썼다 (조상 원문은 "
 "`CZ_d_adj_provenance_inherited`).  "
 f"출하 d_adj = {d0+DADJ_DELTA!r}, 부모 {d0!r} 에서 {DADJ_DELTA!r} 이동.  "
 "이동 이유는 하나뿐이다 — 이 후보가 더한 사후 축 `ctrl_asof_batn` 이 만드는 평균 예측 표류를 되돌리는 것.  "
 "방법은 해석적 보정 Δd = −Δmean / mean(p(1−p)) 이고 감도 0.24749 는 실측치다.  "
 "조준은 cov 대역 프레임(245,789행)에서 ★부모 자신의 평균★ 을 복원하는 것이며 test 통계가 아니다.")
b['changed_vs_FT2']=(f"사후 축 1개 추가 (10/10 `{NAME}`, u={U}) + d_adj 종속 보정. "
 "부스터·표·팔·기존 축 9개의 u·alpha·beta_skew·w_v6·failtype 전부 부모 그대로.")
b['identity_anchor_GA']={"how":f"`{NAME}` 의 u 를 0.0 으로, d_adj 를 {d0!r} 로 되돌리면 부모 출력이 바이트 재현된다.",
 "files_to_touch":["blend.json"]}

def rep(old,new):
    global s
    assert s.count(old)==1, ('앵커 %d회'%s.count(old), old[:70])
    s=s.replace(old,new)
rep('    assert len(_axl) == 9, "EA는 축 오프셋을 정확히 9개 쓴다 (ctrl_rev_band 추가)"'
    if 'ctrl_rev_band' in s else
    '    assert len(_axl) == 8, "DN은 축 오프셋을 정확히 8개 쓴다 (ctrl_asof_midtail 추가)"',
    '    assert len(_axl) == 9, "GA는 축 오프셋을 정확히 9개 쓴다 (ctrl_asof_batn 추가)"')
rep('''                        "ctrl_asof_batmid", "ctrl_asof_midtail"], \\''',
    '''                        "ctrl_asof_batmid", "ctrl_asof_midtail",
                        "ctrl_asof_batn"], \\''')
rep('''                                    "asof_batter_middle_rate"],
                           encoding="utf-8-sig")''',
    '''                                    "asof_batter_middle_rate",
                                    "asof_batter_n"],
                           encoding="utf-8-sig")''')
rep('''        _col_batmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)''',
    '''        _col_batmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)
        _col_batn = _tdf["asof_batter_n"].to_numpy(np.float64)''')
rep('''        # --- apply all seven, row by row ----------------------------------''',
    '''        # --- axis 10/10 `ctrl_asof_batn` -- 후보 GA -----------------------
        #       raw = 이 행 자신의 `asof_batter_n` (타자의 as-of 누적 투구 수).
        #       공식 입력 컬럼이고 변환이 없다.  ★행 독립★: 그 행 자신의 값 하나를
        #       두 동결 스칼라(center/scale)로 표준화할 뿐이다.
        #       n = 0 은 결측이 아니라 '이력 없음' 이라는 실값이라 정상 이동한다.
        #       근거: 부모 9축 중 가장 직교(최대 rho 0.127)하고 정직면 교차적합
        #       실현(2.512)이 표본 안 ceiling(2.543)과 거의 같다.
        _raw_batn = np.where(_pos_asof >= 0,
                             _col_batn[np.maximum(_pos_asof, 0)], np.nan)
        print(f"  asof_batter_n 결측={int(np.isnan(_col_batn).sum())}행"
              f"  n=0 행={int(np.count_nonzero(_col_batn == 0))}"
              f"  행 대응 실패={int((_pos_asof < 0).sum())}행")
        # --- apply all ten, row by row ------------------------------------''')
rep('''                   "ctrl_asof_midtail": _raw_mt}''',
    '''                   "ctrl_asof_midtail": _raw_mt,
                   "ctrl_asof_batn": _raw_batn}''')
rep('''        #    축 7 `asof_batter_middle_rate`.''',
    '''        #    축 7 `asof_batter_middle_rate`,
        #    축 10 `asof_batter_n`.''')
compile(s,'script.py','exec')
txt=json.dumps(b,ensure_ascii=False,indent=1)
with zipfile.ZipFile(SHIP,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as zo:
    for i in z.infolist():
        zo.writestr(i, txt.encode('utf-8') if i.filename=='blend.json'
                    else s.encode('utf-8') if i.filename=='script.py' else z.read(i.filename))
z.close()
zp2=zipfile.ZipFile(PARENT); zs=zipfile.ZipFile(SHIP)   # 쓰기 뒤에는 핸들을 새로 연다
names=[n for n in zp2.namelist() if not n.endswith('/')]
diff=[n for n in names if hashlib.sha256(zp2.read(n)).hexdigest()!=hashlib.sha256(zs.read(n)).hexdigest()]
print(f'[{STAGE}] u={U!r}  d_adj {d0!r} -> {b["d_adj"]!r}')
print('  지불표:',{f'tau={t}':round(g(A,t),3) for t in [-0.3,0,0.25,0.617,1.0,2.22]})
print('  다른 항목:',diff,' 동일',len(names)-len(diff))
print(' ',os.path.basename(SHIP),os.path.getsize(SHIP),hashlib.sha256(open(SHIP,'rb').read()).hexdigest())

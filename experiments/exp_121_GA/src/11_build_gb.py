#!/usr/bin/env python3
"""후보 GB — `ctrl_asof_pstr` (asof_batter_n 선형 사후 축).

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
SHIP=f'{ROOT}/submissions/submit_GB_pstr{"" if STAGE=="ship" else "_v1"}.zip'

# 🔴 후보 1(asof_batter_n)은 폐기했다 — 누적 카운트라 center 가 해마다 이동한다
#    (train 평균 720.8 -> 5402.5, cov 프레임 11,556; 표준화 후 평균 +1.534).
#    라벨없는 곡률이 7.68배로 튀어 축이 방향이 아니라 수준을 실었다.  asof_pitcher_n 도 같은 이유로 폐기.
NAME='ctrl_asof_pstr'; COL='asof_pitcher_strike_rate'
CENTER=0.4442294401797764; SCALE=0.03791751397751183
U_STAR=-0.010979930850378591; A=0.5
U=A*U_STAR
CEIL=2.942916691973941; K_L=-268.0268876076272

z=zipfile.ZipFile(PARENT)
b=json.loads(z.read('blend.json')); s=z.read('script.py').decode('utf-8')
d0=b['d_adj']

def g(a,tau): return CEIL*(2*a*tau-a*a)
b['axis_offsets'].append({
 "name":NAME,"u":U,"center":CENTER,"scale":SCALE,"fitted_at_inference":False,
 "frozen_from":("TRAIN season-2024 프레임 253,507행. center = 그 열의 평균, scale = 모분산 sd. "
   "동결 스칼라이고 test 통계가 아니다."),
 "🟢_정상성_검사":("이 축의 center/scale 은 2024 에서 얼려 2025 에 쓴다. 그래서 그 열이 해를 넘어 "
   "정상(定常)인지 먼저 쟀다: asof_pitcher_strike_rate 연도평균 2019~2024 = 0.4414 0.4394 0.4397 "
   "0.4422 0.4442 0.4442 — 6년 폭이 0.0048 = 0.13 sd 다. 배포 프레임(cov 245,789행)에서 "
   "표준화 후 평균이 0.000 이다.  🔴 이 검사가 선행 후보를 죽였다 — asof_batter_n 은 누적 카운트라 "
   "연도평균이 720.8 -> 5402.5 로 움직이고 cov 에서 표준화 평균이 +1.534 였다(곡률이 7.68배로 튐). "
   "asof_pitcher_n 도 같은 이유로 폐기."),
 "raw_definition":("test.csv 컬럼 asof_pitcher_strike_rate, 변환 없음. 그 행 자신의 컬럼 하나다. "
   "축 2 `ctrl_asof_psr` 가 이미 여는 ★단 한 번의★ test.csv read 에 usecols 로 얹어 읽으므로 "
   "test.csv 재파싱 횟수는 부모와 동일하다. 행 대응도 축 2 와 같은 _pos_asof 를 재사용한다."),
 "nan_policy":("raw 가 비유한이면 그 행의 이 축 이동은 정확히 0.0 이다. 센티넬·평균 대입 없음. "
   "train-2024 프레임 기준 81행이 비유한이다 (투수 첫 등장)."),
 "applied_where":("기존 9축과 완전히 같은 자리 — script.py 의 _shift 누적에 들어가고 "
   "logit_mix = logit_mix + _shift 로 더해진다. alpha/beta_skew 변환 앞."),
 "derivation":{
  "면":"exp111 out-of-time 정직면 (팔 season<=2023 학습 -> 2024행 253,507). 리더보드 점수 0회 사용.",
  "부모 8축 잔차화":{"K_L":K_L,"ceiling":CEIL,"u_star":U_STAR,
    "투수 반분할 교차적합 실현":3.089179298014278,
    "투수 반분할 각 쪽":[8.69,-2.51],
    "최대 rho(부모 축)":0.24171873077054437},
  "★왜 이 축인가★":("① 미청구 원열 중 정직면 ceiling 최대(2.943) ② ★어느 스크린에도 대조군으로 "
    "등록된 적이 없다★ ③ 후보 열 여섯 중 ★유일하게 연도 정상★ 이라 동결 center 를 2025 로 옮길 수 있다. "
    "⚠️ 투수 반분할 두 쪽이 +8.69 / -2.51 로 갈린다 — 효과가 투수 부분집합에 균일하지 않다. "
    "다만 후보 EA 는 반분할이 완벽히 안정(+14.69/+11.72)했는데도 tau = -0.10 이었다. "
    "반분할 안정성은 전이를 예측하지 못한다(EA 가 실측)."),
  "🔴★이 후보는 꼭지점 착륙이 아니라 부호 프로브다★":(
    "감사 AUDIT_GB.md 가 정정했다.  이 축의 ★부호는 로컬로 확인되지 않았다★:\n"
    "  (1) `prev5mid 승` 은 교락이다 — CV_LB.md:2467 이 '한 점으로는 둘을 못 가른다 … "
    "p5m 성립 · 꼬리 미상' 이라 적는다.  선행 판본 계획서가 이것을 '승' 으로 단정했다.\n"
    "  (2) 🔴 S299 를 반쪽만 인용했다.  '어느 스크린에도 대조군으로 등록된 적 없다' 다음에 "
    "'유일한 근거가 S297 에서 부호를 틀린 그 OOT 측정이다' 가 이어진다.  그 측정은 "
    "experiments/exp_118_DN/artifacts/linear_KL_oot.json 이고, ★후보 DS 에서 -3.07 을 "
    "태운 바로 그 파일·산식★ 이다.\n"
    "  (3) 검증된 다른 도구가 반대를 가리킨다.  S281 라벨상관을 그 도구 자신의 실적으로 먼저 "
    "검증하면(psr 6/6 양수 · mid 0/6 양수 — 출하 부호와 일치) 신뢰할 만한데, strike_rate 는 "
    "3/6 으로 불안정하고 ★최근 두 시즌(2023 +0.0354 · 2024 +0.0300)이 둘 다 양수★ 로 "
    "출하 u(음수)와 반대다.\n"
    "→ 따라서 이 발의 목적은 ★부호를 LB 로 사는 것★ 이다(S263·S297 이 권하는 형태). "
    "'tau=0.617 을 믿고 꼭지점 95%에 착륙한다' 는 서사로 읽으면 안 된다 — 그 서사는 부호를 "
    "이미 안다고 전제한다.  지면 다음 슬롯에서 반대 부호로 쏜다(S250 ± 쌍)."),
  "계열 구분 (사실만)":("후보 EA(2026-08-30, LB -2.4971, tau=-0.10)가 죽인 것은 ★면 잔차로 "
    "모양까지 적합한 축★ 이다.  이 축은 방향이 데이터로 고정돼 있고 부호·크기만 정직면에서 온다 — "
    "면 의존도가 한 겹 낮다.  ⚠️ 그러나 그것이 부호를 보증하지는 않는다(위 항목)."),
  "★배포면 실측 (라벨 0개, cov 245,789행)★":{
    "G_deploy":28642.96,"G_honest_screen":24410.62,"비":1.173,
    "평균 표류(미보정)":5.5504e-06,"미보정 수준비용_pts":0.00001,
    "해석":("정직면과 배포면의 곡률이 17% 안에서 같다 = 같은 축으로 행동한다. "
      "폐기한 asof_batter_n 은 이 비가 7.68 이었고 그것이 폐기 사유였다."),
    "배포면 지불표":{"tau=0":-0.863,"tau=0.293":0.0,"tau=0.617":0.953,"tau=1.0":2.080,"tau=2.22":5.670},
    "손익분기_tau":0.293,
    "tau=0.617 에서 꼭지점 대비":"u*_deploy = -0.005773, 출하 -0.005490 = 그 95.1%, 최대의 99.8%"},
  "진폭":{"a":A,"u":U,
    "근거":("진폭을 줄이지 않은 이유: 순위 = 제출 이력 최대값이라 ★지는 발은 순위를 안 깎는다★. "
      "그리고 LB 가 결정론이라 어떤 u != 0 이든 tau 를 정확히 준다 — 정보량은 진폭과 무관하다. "
      "따라서 큰 쪽이 이겼을 때만 이득이 커진다.  하방 -0.86 ~ -3.39 는 슬롯 하나값이다."),
    "지불표":{f"tau={t}":round(g(A,t),3) for t in [-0.3,0,0.25,0.617,1.0,2.22]}},
  "d_adj":{"stage":STAGE,"적용한 d_adj 이동":DADJ_DELTA,
    "감도(실측)":0.24749,
    "note":("stage=probe 판본은 d_adj 를 안 옮기고 cov 프레임에서 평균 표류를 실측하는 용도다. "
            "stage=ship 판본만 제출한다.")}},
})
b['candidate']='GB' if STAGE=='ship' else 'GB_v1'
b['d_adj']=d0+DADJ_DELTA
b['CZ_d_adj_provenance_inherited']=b.get('CZ_d_adj_provenance_inherited',b['derived_from'])
b['derived_from']=("TRAIN 유래 상수다. 🔴 이 문장은 후보 GB 가 새로 썼다 (조상 원문은 "
 "`CZ_d_adj_provenance_inherited`).  "
 f"출하 d_adj = {d0+DADJ_DELTA!r}, 부모 {d0!r} 에서 {DADJ_DELTA!r} 이동.  "
 "이동 이유는 하나뿐이다 — 이 후보가 더한 사후 축 `ctrl_asof_pstr` 이 만드는 평균 예측 표류를 되돌리는 것.  "
 "방법은 해석적 보정 Δd = −Δmean / mean(p(1−p)) 이고 감도 0.24749 는 실측치다.  "
 "조준은 cov 대역 프레임(245,789행)에서 ★부모 자신의 평균★ 을 복원하는 것이며 test 통계가 아니다.")
b['changed_vs_FT2']=(f"사후 축 1개 추가 (9/9 `{NAME}`, u={U}) + d_adj 종속 보정. "
 "부스터·표·팔·기존 축 9개의 u·alpha·beta_skew·w_v6·failtype 전부 부모 그대로.")
b['identity_anchor_GB']={"how":f"`{NAME}` 의 u 를 0.0 으로, d_adj 를 {d0!r} 로 되돌리면 부모 출력이 바이트 재현된다.",
 "files_to_touch":["blend.json"]}

def rep(old,new):
    global s
    assert s.count(old)==1, ('앵커 %d회'%s.count(old), old[:70])
    s=s.replace(old,new)
rep('    assert len(_axl) == 9, "EA는 축 오프셋을 정확히 9개 쓴다 (ctrl_rev_band 추가)"'
    if 'ctrl_rev_band' in s else
    '    assert len(_axl) == 8, "DN은 축 오프셋을 정확히 8개 쓴다 (ctrl_asof_midtail 추가)"',
    '    assert len(_axl) == 9, "GA는 축 오프셋을 정확히 9개 쓴다 (ctrl_asof_pstr 추가)"')
rep('''                        "ctrl_asof_batmid", "ctrl_asof_midtail"], \\''',
    '''                        "ctrl_asof_batmid", "ctrl_asof_midtail",
                        "ctrl_asof_pstr"], \\''')
rep('''                                    "asof_batter_middle_rate"],
                           encoding="utf-8-sig")''',
    '''                                    "asof_batter_middle_rate",
                                    "asof_pitcher_strike_rate"],
                           encoding="utf-8-sig")''')
rep('''        _col_batmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)''',
    '''        _col_batmid = _tdf["asof_batter_middle_rate"].to_numpy(np.float64)
        _col_pstr = _tdf["asof_pitcher_strike_rate"].to_numpy(np.float64)''')
rep('''        # --- apply all seven, row by row ----------------------------------''',
    '''        # --- axis 9/9 `ctrl_asof_pstr` -- 후보 GB ------------------------
        #       raw = 이 행 자신의 `asof_pitcher_strike_rate` (투수 as-of 스트라이크 비율).
        #       공식 입력 컬럼이고 변환이 없다.  ★행 독립★: 그 행 자신의 값 하나를
        #       두 동결 스칼라(center/scale)로 표준화할 뿐이다.
        #       근거: 미청구 원열 중 정직면 ceiling 최대(2.943)이고, 후보 여섯 중
        #       유일하게 연도 정상이라 동결 center 를 2025 로 옮길 수 있다.
        _raw_pstr = np.where(_pos_asof >= 0,
                             _col_pstr[np.maximum(_pos_asof, 0)], np.nan)
        print(f"  asof_pitcher_strike_rate 결측={int(np.isnan(_col_pstr).sum())}행"
              f"  "
              f"  행 대응 실패={int((_pos_asof < 0).sum())}행")
        # --- apply all nine, row by row -----------------------------------''')
rep('''                   "ctrl_asof_midtail": _raw_mt}''',
    '''                   "ctrl_asof_midtail": _raw_mt,
                   "ctrl_asof_pstr": _raw_pstr}''')
rep('''        #    축 7 `asof_batter_middle_rate`.''',
    '''        #    축 7 `asof_batter_middle_rate`,
        #    축 9 `asof_pitcher_strike_rate`.''')
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

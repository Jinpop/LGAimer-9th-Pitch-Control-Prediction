#!/usr/bin/env python3
"""후보 FT2 생성 스크립트 — failtype_offset.c 프로브.

🔴 S301: 산문 수정은 패키지와 ★이 스크립트★ 둘 다에 걸려 있다.
   감사 AUDIT_FT.md 가 F1~F4 로 잡은 승계 산문 결함을 여기서 고친다 — 다음 후보에서 부활하지 않도록.
"""
import json, zipfile, hashlib, os, sys

ROOT = '<PROJECT_ROOT>'
PARENT = f'{ROOT}/submissions/submit_mid꼬리_bmid보수적.zip'
SHIP   = f'{ROOT}/submissions/submit_FT2_failtypeC_probe.zip'

# ── 상수 ───────────────────────────────────────────────────────────────────
C_SHIP   = -0.0075                 # exp_112 ★probe_recommendation★ 그대로
G_CC     = 3673.27                 # 현 기반 라벨없는 곡률 (cov 3판)
G_BC     = -1970.48
G_BB     = 2338.52
B_SHIP   = -0.03645076320272027    # LB 가 산 값 — 건드리지 않는다
CROSS    = -2 * G_BC * B_SHIP      # = -143.651
DMEAN_AT_C     = -3.217e-04        # 실측: c=-0.019554 에서 평균 예측 표류
C_AT_MEASURE   = -0.019554
DMEAN_PER_DADJ = 0.24752           # 실측 감도 (FT/FTD 두 판 역산)

drift  = DMEAN_AT_C * (C_SHIP / C_AT_MEASURE)
DADJ_D = -drift / DMEAN_PER_DADJ

def gain(c, L_c):                  # gain(c) = 2*L_c*c + CROSS*c - G_CC*c^2
    return 2*L_c*c + CROSS*c - G_CC*c*c
HYP = {"① b 계열 전이 (+0.18936)": -45.84227638065127,
       "② ctrl_asof_psr 형 반전 (−0.41328)": 100.05361710613145,
       "③ count 형 반전 (−1.51239)": 366.13893445110966,
       "④ L_c = 0 (신호 없음)": 0.0}

z = zipfile.ZipFile(PARENT)
b = json.loads(z.read('blend.json'))
d0 = b['d_adj']; d_new = d0 + DADJ_D

# ── F1 (차단) : derived_from 을 FT 세대로 다시 쓴다 ────────────────────────
b['CZ_d_adj_provenance_inherited'] = b['derived_from']
b['derived_from'] = (
 "TRAIN 유래 상수다. 🔴 이 문장은 후보 FT2 가 새로 쓴 것이고 부모 CZ 세대의 원문은 "
 "`CZ_d_adj_provenance_inherited` 에 그대로 보존돼 있다 (감사 AUDIT_FT.md F1 · S240).  "
 f"출하 d_adj = {d_new!r} 이고 부모값 {d0!r} 에서 {DADJ_D!r} 만큼 움직였다.  "
 "움직인 이유는 하나뿐이다 — 이 후보가 옮긴 failtype_offset.c 가 만드는 평균 예측 표류를 되돌리는 것.  "
 "🔴 방법은 부모와 다르다: 부모는 캘프레임(tr2024 253,507행)에서 brentq 근을 풀었고, "
 "이 후보는 ★해석적 보정★ Δd = −Δmean / mean(p(1−p)) 를 쓴다.  "
 "감도 0.24752 는 같은 패키지를 c 를 옮겨 두 번 돌려 역산한 ★실측치★ 다 "
 "(c=-0.019554 에서 미보정 평균 표류 -3.217e-04, d_adj +1.287e-03 적용 후 잔여 -3.145e-06).  "
 "🔴 조준 프레임도 부모와 다르다: 부모의 조준은 캘프레임 평균 0.48591045 였고, "
 "이 후보의 조준은 ★cov 대역 프레임(245,789행)에서 부모 자신의 평균을 복원하는 것★ 이다.  "
 "그 프레임은 train season==2024 행의 season 을 2025 로 덮은 자작 프레임이고 "
 "(exp_073/src/s0_ship.py:92), 조준에 쓴 값은 ★부모 패키지의 출력 평균★ 이지 test 통계가 아니다.  "
 "추론 시점에 읽는 test 통계는 0 개다.")

# ── F2 (차단) : S137/S240 전용 장부를 FT 기준으로 갱신 ─────────────────────
sw = b.get('CZ_S73_prose_sweep', {}).get('S137_S240_value_leaves')
if isinstance(sw, dict):
    b['CZ_S73_prose_sweep']['S137_S240_value_leaves'] = {
      "🔴_CZ_세대_기준_기록이다": ("아래 네 리프는 ★조부 CZ 세대★ 의 스윕 결과이고 후보 FT2 에서는 "
        "전부 낡았다. 지우지 않고 이력으로 남긴다 (감사 AUDIT_FT.md F2 · S234·S243 — "
        "면책 목록에서 키를 빼는 편집이 새 위험 표면이다)."),
      "CZ_generation_record": sw,
      "FT2_현행": {
        "shipped_d_adj": repr(d_new),
        "parent_d_adj": repr(d0),
        "leaves_quoting_parent_d_adj_without_history_marker": [],
        "n": 0,
        "note": ("`derived_from` 은 FT2 가 다시 썼고 그 안의 d_adj 는 ★출하값★ 이다. "
                 "부모값은 `CZ_d_adj_provenance_inherited` 안에만 남아 있고 그 키 이름이 이력 표시다.")}}

# ── F3 (권고) : 승계 identity_anchor 에 이력 표시 ──────────────────────────
if 'identity_anchor' in b:
    b['CZ_generation_history_identity_anchor'] = b.pop('identity_anchor')
    b['CZ_generation_history_identity_anchor']['🔴_이것은_조상_세대_지시문이다'] = (
      "후보 FT2 의 되돌리기 지시문은 `identity_anchor_FT2` 하나뿐이다. 이 블록의 "
      "'it currently reads ...' 같은 괄호는 조상 세대 기준이라 지금은 거짓이다 (감사 F3 · S91·S300).")
b['identity_anchor_FT2'] = {
  "how": (f"blend.json 의 failtype_offset.c 를 0.0 으로, d_adj 를 {d0!r} 로 되돌리면 "
          "부모 출력 CSV 가 바이트 단위로 재현된다. c=0.0 이면 c*(lg_wo-mu_wo) 항이 정확히 0.0 이다."),
  "files_to_touch": ["blend.json"],
  "실행_검증": "선행 판본(c=-0.019554)에서 real5 · cov 두 프레임 sha 가 부모와 일치했다."}

# ── F4 (권고) : failtype_offset 승계 산문이 c 에 대해 낡았음을 명시 ────────
ft = b['failtype_offset']
ft['c_parent_FT2'] = ft['c']; ft['c'] = C_SHIP
ft['🔴_c_offgrid_note'] = (
  "출하 (b, c) 는 ★둘 다 원 16점 격자 밖★ 이다. b = -0.03645076320272027 은 후보 CA 의 5축 동시 해가 "
  "격자 밖으로 옮긴 값이고, c 는 이 후보가 교차항으로 새로 정한 값이다. 따라서 이 블록의 승계 산문 중 "
  "`provenance`(격자 (b∈{0,-0.05,-0.105,-0.15})×(c∈{0,+0.011,+0.0175,+0.03}) 에서 d_off 정렬로 골랐다) · "
  "`cell_route_note`(계수 (b,c) 는 81열 셀 위에서 뽑혔다) · `source`(fold_2024 Δ+11.791056, 8/8) 셋은 "
  "★격자 채택쌍★ 의 기록이지 출하쌍의 기록이 아니다. 지우지 않고 이력으로 남긴다 (감사 AUDIT_FT.md F4).")
ft['c_derivation'] = {
  "★리더보드 점수를 읽지 않았다": "c 의 부호·크기는 (a) 이미 LB 가 산 b (b) 라벨 없는 곡률 에서만 나온다.",
  "면": "부모(mid꼬리)를 cov 프레임 245,789행에서 c=±0.02 · b=+0.02 세 판 돌려 잰 라벨없는 곡률. 라벨 0개.",
  "G_cc": G_CC, "G_bb": G_BB, "G_bc": G_BC, "b_ship": B_SHIP,
  "산식": f"gain(c) = 2*L_c*c + ({CROSS:.3f})*c - {G_CC}*c^2",
  "★왜 이 값인가★": (
    f"exp_112 metrics.json 의 ★probe_recommendation★ u = -0.0075 을 ★그대로★ 쓴다. "
    "그 값의 사전 등록 사유는 '가설 ①과 ②가 2.19점 갈린다 — 검출한계 +1.4 를 넘는다. "
    "더 키우면 하방만 커진다(S42)' 이다.  "
    "🔴 선행 판본(FT)은 이것을 -0.019554(2.6배)로 키웠고 사유를 적지 않았다 — S8 위반이라 되돌렸다.  "
    "그리고 되돌리는 것이 산술로도 우월하다: ①이 참이면 다음 슬롯에서 꼭지점 -0.032033 에 착륙해 "
    "합계 +3.769 로 같은 종착점에 닿는데, 경로의 하방만 작아진다."),
  "🔴 바닥은 없다": (
    "exp_112 의 `🔴_no_floor` 키를 그대로 인용한다: \"'L_c=0 이어도 +1.06' 은 성립하지 않는다. "
    "②형 반전이면 같은 걸음이 -2.48점을 잃는다. 방향은 사야 한다.\"  "
    "🔴 선행 판본 계획서가 '바닥이 +1.404' 라고 적었는데 그것은 인용 원천이 미리 반박해 둔 문장이다(S233)."),
  "지불표": {k: round(gain(C_SHIP, L), 3) for k, L in HYP.items()},
  "가설 ①-② 분리": round(gain(C_SHIP, HYP["① b 계열 전이 (+0.18936)"]) -
                         gain(C_SHIP, HYP["② ctrl_asof_psr 형 반전 (−0.41328)"]), 3),
  "③ 이 무시 못 할 이유": "rho(failtype_c, count_offsets) = +0.3348 (exp_112 rho_with_other_live_axes).",
  "🔴 왜 EA 와 다른가": (
    "후보 EA(2026-08-30, LB -2.4971)는 부호·크기가 전부 우리 정직면에서 왔고 tau = -0.10 으로 측정됐다. "
    "정직면 유래 몫 100% 축은 5/5 로 졌다(합계 -41.14). 이 축은 S232 가 정직면 유래 몫을 2.7% 로 "
    "실측한 축이고, 부호는 LB 가 산 b 와의 교차항에서 온다(교차 -143.65 대 L_c -2.74, 21.8배)."),
  "b 는 안 건드린다": "b 는 LB 가 산 값이고(이득 +2.6766924977) c 교차항의 부호가 거기서 나온다.",
  "d_adj": {"방식": "해석적 보정", "감도(실측)": DMEAN_PER_DADJ,
            "이 걸음의 예상 평균 표류": drift, "적용한 d_adj 이동": DADJ_D,
            "미보정 시 수준 비용": f"K*delta^2 = {400309.19882517256*drift*drift:.4f} pts",
            "🔴 조준 프레임 변경을 변인으로 선언한다": (
              "부모는 캘프레임 aim 0.48591045 를 조준했고 이 후보는 cov 프레임에서 부모 평균을 "
              "복원했다. 크기는 0.04점 수준이다(감사 §5).")}}
# ── F2 확장 : 부모 d_adj 를 이력표시 없이 인용하는 승계 리프 전수 표시 ────
#    (S243 — 면책 목록에서 키를 ★빼는★ 편집이 새 위험 표면이다. 지우지 말고 표시한다.)
_stale = []
def _walk(o, pth=''):
    if isinstance(o, dict):
        for k, v in o.items(): _walk(v, pth + '/' + k)
    elif isinstance(o, list):
        for i, v in enumerate(o): _walk(v, pth + f'[{i}]')
    elif isinstance(o, str) and repr(d0).strip("'") in o:
        if not any(t in pth for t in ('inherited','CZ_generation','CZ_S73','CZ_d_adj',
                                      'identity_anchor_FT2','derived_from','FT2_')):
            _stale.append(pth)
_walk(b)
b['FT2_d_adj_prose_sweep'] = {
  "🔴_읽는 법": ("아래 리프들은 부모 d_adj 값 %s 를 인용하지만 전부 ★조상 세대의 생성 기록★ 이다 — "
    "'우리 세대는 d_adj 를 안 옮겼다' 는 뜻이지 ★출하값★ 을 주장하는 게 아니다.  "
    "후보 FT2 의 출하 d_adj 는 %s 이고 그것을 주장하는 리프는 `derived_from` 하나뿐이다.  "
    "S243 대로 지우지 않고 표시만 한다 — 키를 빼는 편집이 새 위험 표면이기 때문이다.") % (repr(d0), repr(d_new)),
  "조상 세대 기록 리프": _stale,
  "n": len(_stale),
  "출하값을 주장하는 리프": ["/derived_from"],
  "판정 기준": "S48 — '그 숫자를 인용하는가' 가 아니라 '★현 출하값의 성질★ 을 주장하는가' 로 갈랐다."}

b['d_adj'] = d_new
b['candidate'] = 'FT2'
b['changed_vs_mid꼬리'] = (
  f"failtype_offset.c 0.0 -> {C_SHIP} 와 그에 종속된 d_adj 해석적 보정, 그리고 감사 AUDIT_FT.md 가 "
  "지적한 승계 산문 F1~F4 수정. ★script.py 는 한 글자도 안 바뀐다★ — c 는 이미 배선돼 있다. "
  "부스터·표·팔·축 9개·alpha·beta_skew·w_v6·b 전부 부모 그대로.")

txt = json.dumps(b, ensure_ascii=False, indent=1)
with zipfile.ZipFile(SHIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zo:
    for i in z.infolist():
        zo.writestr(i, txt.encode('utf-8') if i.filename == 'blend.json' else z.read(i.filename))
print(f'c = {C_SHIP}   d_adj {d0!r} -> {d_new!r}  (이동 {DADJ_D:+.6e})')
print('지불표:', {k: round(gain(C_SHIP, L), 3) for k, L in HYP.items()})
print('①-② 분리 %.3f pts'%(gain(C_SHIP,HYP["① b 계열 전이 (+0.18936)"])-gain(C_SHIP,HYP["② ctrl_asof_psr 형 반전 (−0.41328)"])))
print('①이 참이면 꼭지점 c* = %.6f 에서 합계 %.3f'%((2*HYP["① b 계열 전이 (+0.18936)"]+CROSS)/(2*G_CC),
      gain((2*HYP["① b 계열 전이 (+0.18936)"]+CROSS)/(2*G_CC), HYP["① b 계열 전이 (+0.18936)"])))
print(SHIP, os.path.getsize(SHIP), hashlib.sha256(open(SHIP,'rb').read()).hexdigest())

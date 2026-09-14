"""exp_067 산출 — **C단 빌드용 추론 코드 초안**. 이 파일은 어디에도 배포되지 않는다.

배경 — 후보 BM 은 출하 `aa_infer.py` 가 필요한 값을 함수 밖으로 안 내놔서 죽었다.
NN 팔은 그 문제가 **없다**. 이유:

  · NN 이 먹는 99열은 이미 `mate/v6_infer.py::run` 안에서 전량 조립된다
    (`mate/model/features_sp.txt` sha 가 우리 학습 피처 목록과 **같은 파일**이다).
  · 그 `df` 는 `mate/nn_infer.py::predict_nn(df, ...)` 의 첫 인자로 **함수 밖으로 나온다**.
  · 우리 top-level `script.py` 는 이미 mate 를 자식 프로세스로 띄우면서
    `v6_infer.run`·`q_infer.run`·`infer_ebm6.predict_ebm6_logit` 을 감싸 값을 회수한다
    (BK 이후 BOOT 문자열). **같은 자리에 하나를 더 감싸면 된다.**

즉 새 피처 경로·새 표·새 조인·새 상수가 하나도 필요 없다.
아래 두 조각이 C단 빌드가 넣을 전부다.
"""

# ══════════════════════════════════════════════════════════════════════════
# 조각 1 — top-level script.py 의 BOOT 문자열에 덧붙일 래퍼
#   🔴 `mate/` 안의 파일은 한 바이트도 바뀌지 않는다(gate d 승계 sha 유지).
#   🔴 npz·meta 는 **부모 패키지 루트의 새 디렉터리**에 둔다. 절대경로를 부모가
#      만들어 문자열에 박아 넣으므로 자식의 cwd(=mate/)와 무관하다.
#   🔴 첫 호출 1회만 계산한다(v6 는 자기 NN 4시드 때문에 predict_nn 을 4번 부른다).
# ══════════════════════════════════════════════════════════════════════════
BOOT_ADDITION = '''
"import nn_infer as _m4\\n"
"_f4 = _m4.predict_nn\\n"
"_ns = []\\n"
"def _w4(_df, *a, **k):\\n"
"    if not _ns:\\n"
"        _ns.append(1)\\n"
"        import numpy as _n\\n"
"        _acc = []\\n"
"        for _s in (42, 777, 2024, 101):\\n"
"            _q = _f4(_df, %(D)r + '/nn_sp%%d.npz' %% _s,\\n"
"                          %(D)r + '/nn_sp%%d_meta.json' %% _s)\\n"
"            _q = _n.clip(_q, 1e-12, 1 - 1e-12)\\n"
"            _acc.append(_n.log(_q / (1 - _q)))\\n"
"        _n.save('_cap_lnn.npy', _n.mean(_acc, axis=0))\\n"
"        _n.save('_cap_nnrid.npy', _n.asarray(_df['row_id'].tolist()))\\n"
"    return _f4(_df, *a, **k)\\n"
"_m4.predict_nn = _w4\\n"
'''  # %(D)s = os.path.abspath("./model_nn_sp") — 부모가 채운다

# ══════════════════════════════════════════════════════════════════════════
# 조각 2 — 부모(top-level script.py)에서 회수·합류
# ══════════════════════════════════════════════════════════════════════════
PARENT_SNIPPET = '''
# 여섯째 팔 회수 — 자식이 mate/ 안에 남긴 두 파일. 행 단위 row_id 조인.
_lnn_p = os.path.join(MATE_DIR, "_cap_lnn.npy")
_rid_p = os.path.join(MATE_DIR, "_cap_nnrid.npy")
logit_nn = None
if os.path.exists(_lnn_p) and os.path.exists(_rid_p):
    _rid = np.load(_rid_p, allow_pickle=True)
    _lnn = np.load(_lnn_p)
    assert len(_rid) == len(_lnn) == len(ids_aa), "NN 팔 길이 불일치"
    # row_id 로 두 팔의 순서를 맞춘다. 행 i 는 행 i 자기 값만 받는다(규칙 2-4).
    _pos = pd.Index([str(x) for x in _rid]).get_indexer(
        [str(x) for x in ids_aa])
    assert (_pos >= 0).all(), "NN 팔에 없는 row_id 가 있다"
    logit_nn = _lnn[_pos]
    assert np.isfinite(logit_nn).all(), "NN 팔 로짓에 비유한값"

# 팔 가중: 기존 규약 그대로 (1 - w_v6) 를 나눠 갖는 전역 스칼라 하나를 더한다.
#   c_aa + c_S2 + c_S1 + c_T70 + c_NN == 1 - w_v6
# 🔴 c_NN 값 자체는 이 실험이 정하지 않는다 (판정 금지). 빌드 계획서 소관.
if logit_nn is not None and c_nn:
    logit_mix = logit_mix + c_nn * logit_nn
'''

# ══════════════════════════════════════════════════════════════════════════
# 이 팔이 요구하는 입력 열 — 전량, 그리고 각 열의 산출 모듈
# ══════════════════════════════════════════════════════════════════════════
INPUT_COLUMNS = {
    "총_모델_입력": 100,
    "features_sp.txt": 99,
    "추가_1열": "cnt_state — `v6_infer.run` 이 자기 줄에서 만든다 "
                "(balls_before*3 + strikes_before). features_sp.txt 에는 없다.",
    "산출_모듈별_귀속(실측)": {
        "test.csv(raw)": 33,
        "mate/pipeline.py::apply_features": 22,
        "mate/pipeline.py::apply_ratio_features": 22,
        "mate/splits.py::apply (deploy_splits.npz)": 6,
        "mate/inseason.py::apply (anchor.npz)": 16,
        "v6_infer.run 자체(cnt_state)": 1,
    },
    "범주_5열": ["base_state", "game_type", "pitcher_team_id",
                 "batter_team_id", "cnt_state"],
    "밀집_95열": "위 99열에서 범주 4열(base_state·game_type·pitcher_team_id·"
                 "batter_team_id)을 뺀 것. cnt_state 는 99열에 없으므로 빼지 않는다.",
    "새로_필요한_표·조인·상수": "없음 — 전부 부모 zip 이 이미 싣고 있는 파일이다.",
}

# ══════════════════════════════════════════════════════════════════════════
# C단 빌드가 챙겨야 할 것 (체크리스트 초안)
# ══════════════════════════════════════════════════════════════════════════
BUILD_CHECKLIST = [
    "새 디렉터리 `model_nn_sp/` 에 npz 4 + meta 4 = 8개 파일. mate/ 는 무변경.",
    "🔴 `model/meta.json` 은 부스터 트리 수를 대조한다 — NN 팔은 부스터를 "
    "건드리지 않으므로 수정 대상이 아니다. 그래도 빌드 때 확인은 할 것.",
    "🔴 blend.json 에 `c_nn` 추가 + `c_aa + c_S2 + c_S1 + c_T70 + c_nn == 1 - w_v6` "
    "assert 를 함께 옮길 것(현행 assert 가 5항 합만 검사한다 — 안 고치면 터진다).",
    "항등 앵커: `c_nn = 0.0` 으로 빌드해 출력 CSV sha 가 부모와 같은지. "
    "같으면 프레임·조인·산술이 부모와 동일함이 한 번에 증명된다.",
    "규칙 2-4: 내용/위치 분리 실험. NN 팔은 numpy 축약이라 팀원 v6 NN 과 같은 "
    "SIMD 블록 경계 잡음(~1e-9)이 나올 수 있다(S13·S15 계열). 100행 단독 검사만으로 "
    "판정하지 말 것.",
    "추론 시간: 프레임 재사용이므로 추가비용은 NN 전방향 전파뿐 (g3_ship.json S4).",
    "🔴 커버리지 게이트를 대역 프레임 단독으로 돌리지 말 것 — 대역에서는 ins_* 16열이 "
    "100% 축퇴한다(exp_067 실측 245,789/245,789). 실제 표본 5행 병행 필수.",
]

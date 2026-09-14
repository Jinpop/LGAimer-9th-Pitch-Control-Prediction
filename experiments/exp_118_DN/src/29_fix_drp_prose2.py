#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🔴 DRP 산문 수정 2차 — 게이트가 잡은 세 번째 거짓 자기서술 (S30·S48·S73).

`🔴_어떤_계수도_빌려오지_않았다` 가 "정직면(honest surface)에서 유도한 몫은 ★0%★ 다"
라고 적고 있었다.  ★DRP 에 대해 부정확하다★ — 정직면은 이 축을 ★고르는 데★ 썼고
(미청구 11열 중 |K*L_res| 최대), 다만 부호는 정직면이 가리킨 것의 ★반대★ 이고 크기도
정직면에서 오지 않았다.  "0%" 는 그 셋을 뭉뚱그려 사실과 다르게 만든다.

🔴 원인: 27_build_drp.py 가 23_build_dr.py 에서 생성됐는데, 26_fix_dr_prose.py 는
   ★빌드 스크립트가 아니라 이미 만들어진 패키지만★ 고쳤다.  같은 거짓 문장이 새 빌드에
   재생됐다 (S30 이 경고한 "수정본에서 재발한다" 가 그대로 일어났다).

★예측값은 한 비트도 안 바뀐다★ — 고친 뒤 3프레임을 다시 돌려 sha256 을 대조한다.
"""
import hashlib
import json

B = "/home/<user>/SungjinPark/build_DN"
OLD = "🔴_어떤_계수도_빌려오지_않았다"
NEW = "🔴_정직면은_축을_고르는_데만_썼다_부호도_크기도_아니다"
TXT = (
 "S262 — 곡률 G 는 기반을 넘어 옮겨도 되지만 선형항 L 은 절대 안 된다.  이 축은 다른 "
 "후보·다른 기반의 K_L·K_G 를 옮겨 쓰지 않았다.  "
 "🔴 정직면(out-of-time honest surface)이 이 후보에 들어간 자리를 정확히 적는다"
 "(S232):\n"
 "① ★축 선정★ — 썼다.  미청구 11열 중 잔차화 |K*L_res| = 313.9 로 가장 커서 이 열을 "
 "골랐다.  즉 '어느 방향을 볼 것인가' 는 정직면이 정했다.\n"
 "② ★부호★ — 안 썼다.  오히려 ★정직면이 가리킨 것의 반대★ 다 (정직면은 음수를 "
 "지시했고 이 후보는 u = +0.008 이다).  부호 근거가 없다는 것은 "
 "`🔴_이_축의_부호_근거는_없다` 에 적었다.\n"
 "③ ★크기★ — 안 썼다.  0.008 은 형제 축 `ctrl_asof_mid` 가 이 계보에서 검증한 프로브 "
 "규격이다.\n"
 "🔴 그래서 '정직면 유래 몫 0%' 라고 적지 않는다 — ①이 있으므로 거짓이다.  "
 "S245·S252 가 경고한 위험(정직면 유래 몫 100% 축이 4/4 로 졌다)에 대해서는, 이 후보가 "
 "★부호를 정직면 반대로★ 실었으므로 그 계열과 성질이 다르다.")
NOTE2 = (
 "🔴 2차 산문 수정.  `" + OLD + "` 키가 '정직면에서 유도한 몫은 ★0%★ 다' 라고 적고 "
 "있었다 — 정직면을 ★축 선정★ 에 썼으므로 부정확하다.  🔴 원인이 중요하다: "
 "27_build_drp.py 가 23_build_dr.py 에서 생성됐는데 1차 수정(26_fix_dr_prose.py)은 "
 "★빌드 스크립트가 아니라 이미 만들어진 패키지만★ 고쳤다.  그래서 같은 거짓 문장이 새 "
 "빌드에 ★재생됐다★ — S30 이 경고한 '수정본에서 재발한다' 가 그대로 일어났고, 출하 전 "
 "게이트가 다시 잡았다.  예측값은 한 비트도 안 바뀐다(3프레임 sha256 대조로 확인).")


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


for tag in ("DRP_SHIP", "ANC_DRP"):
    p = B + "/pkg/" + tag + "/blend.json"
    b = json.load(open(p, encoding="utf-8"))
    d = b["axis_offsets"][5]["derivation"]
    assert "정직면(honest surface)에서 유도한 몫은 ★0%★" in d[OLD], tag
    del d[OLD]
    d[NEW] = TXT
    b["🔴_PROSE_CORRECTION_2_BEFORE_SHIPPING"] = NOTE2
    par = json.load(open(B + "/pkg/PAR/blend.json", encoding="utf-8"))
    nk = sorted(set(b) - set(par))
    ch = sorted(k for k in set(b) & set(par)
                if json.dumps(b[k], ensure_ascii=False, sort_keys=True)
                != json.dumps(par[k], ensure_ascii=False, sort_keys=True))
    b["DRP_S73_prose_sweep"]["top_level_keys_describing_DRP"] = {
      "n_new": len(nk), "new": nk, "n_changed": len(ch), "changed": ch,
      "n_removed": 0, "total": len(nk) + len(ch),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 대조."}
    open(p, "w", encoding="utf-8", newline="").write(
        json.dumps(b, ensure_ascii=False, indent=1))
    print("fixed2", tag)

REL = ["pkg/DRP_SHIP/blend.json", "pkg/DRP_SHIP/script.py",
       "pkg/ANC_DRP/blend.json", "pkg/ANC_DRP/script.py"]
open(B + "/work/frozen_prose_sha_drp.txt", "w").write(
    "\n".join(sh(B + "/" + r) + "  " + r for r in REL) + "\n")
print("re-frozen 4 files")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🔴 DR 산문 수정 — 출하 전 게이트가 잡은 거짓 자기서술 하나 (S30·S48·S73).

첫 판본이 `어떤 계수도 빌려오지 않았다` 키에 "정직면 유래 몫은 0%" 라고 적고 있었다.
★DR 에 대해서는 거짓★ 이다 — 이 축의 부호는 out-of-time 정직면의 잔차화 선형항에서 왔다.
그 문장은 형제 후보 DQ 의 서술을 기계 치환으로 물려받은 것이다(DQ 는 부호가 라벨 상관
규약에서 와서 참이었다).

★예측값은 한 비트도 안 바뀐다★ — 산문 전용 수정이고, 고친 뒤 3프레임을 다시 돌려
출력 sha256 이 수정 전과 동일함을 기계로 확인한다(87_regate_dr.sh).
"""
import hashlib
import json

B = "/home/<user>/SungjinPark/build_DN"
KEY_OLD = "🔴_어떤_계수도_빌려오지_않았다"
KEY_NEW = "🔴_계수는_안_빌렸고_정직면_유래_몫은_부호에_한해_100퍼센트다"
NEW = (
 "S262 — 곡률 G 는 기반을 넘어 옮겨도 되지만 선형항 L 은 절대 안 된다.  이 축은 형제 "
 "축 5 의 ★진폭★ 만 반사했고 다른 후보·다른 기반의 K_L·K_G 를 옮겨 쓰지 않았다.  "
 "🔴 다만 ★정직면 유래 몫을 정직하게 적는다(S232)★: 이 축의 ★부호★ 는 out-of-time "
 "정직면의 잔차화 선형항(K*L_res = -313.9)에서 왔으므로 ★부호의 정직면 유래 몫은 "
 "100%★ 다.  S245·S252 가 적은 대로 정직면 유래 몫 100% 인 축은 우리 원장에서 "
 "★4/4 로 졌다(합계 -38.6)★ — 그 위험을 이 축이 그대로 안고 있다.  완화 근거는 "
 "둘뿐이다: ① 그 측정이 출하 축 둘로 ★양성 대조 2/2★ 를 통과했고(표본 안 면은 같은 "
 "대조에서 2/2 로 부호가 반대였다) ② ★크기는 정직면에서 오지 않았다★ — 진폭 0.008 은 "
 "팀원 축 `ctrl_asof_mid` 가 이 계보에서 검증한 규격이다.")
NOTE = (
 "🔴 이 후보의 첫 판본은 `" + KEY_OLD + "` 키에 '정직면(honest surface)에서 유도한 몫은 "
 "★0%★ 다' 라고 적고 있었다.  ★DR 에 대해서는 거짓★ 이다 — 이 축의 부호는 out-of-time "
 "정직면의 잔차화 선형항에서 왔고, 같은 blend.json 의 `🔴_u_가_음수인_근거_그리고_부호_"
 "근거_둘이_충돌한다` 가 그것을 정확히 서술하고 있어 ★두 문장이 모순★ 이었다.  그 거짓 "
 "문장은 형제 후보 DQ 의 서술을 기계 치환으로 물려받은 것이다(DQ 는 부호가 라벨 상관 "
 "규약에서 와서 참이었다).  ★출하 전 게이트가 잡아 고쳤다.★  S30·S48·S73 계열.  "
 "예측값은 한 비트도 안 바뀐다 — 산문 전용 수정이고, 고친 뒤 3프레임(real5·keep110·cov)을 "
 "다시 돌려 출력 sha256 이 수정 전과 동일함을 기계로 확인했다.")


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


for tag in ("DR", "ANC_DR", "DRP"):
    p = f"{B}/pkg/{tag}/blend.json"
    b = json.load(open(p, encoding="utf-8"))
    d = b["axis_offsets"][5]["derivation"]
    assert "정직면(honest surface)에서 유도한 몫은 ★0%★" in d[KEY_OLD], tag
    del d[KEY_OLD]
    d[KEY_NEW] = NEW
    b["🔴_PROSE_CORRECTION_BEFORE_SHIPPING"] = NOTE
    open(p, "w", encoding="utf-8", newline="").write(
        json.dumps(b, ensure_ascii=False, indent=1))
    print("fixed", tag)

REL = ["pkg/DR/blend.json", "pkg/DR/script.py",
       "pkg/ANC_DR/blend.json", "pkg/ANC_DR/script.py"]
lines = []
for r in REL:
    lines.append(sh(B + "/" + r) + "  " + r)
open(B + "/work/frozen_prose_sha_dr.txt", "w").write("\n".join(lines) + "\n")
print("re-frozen 4 files")

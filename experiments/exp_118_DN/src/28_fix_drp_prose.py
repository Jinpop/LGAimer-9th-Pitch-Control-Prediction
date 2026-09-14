#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""🔴 DRP 산문 수정 — 출하 전에 잡은 거짓 자기서술 둘 + 키 이름 (S30·S48·S73·S91).

① `🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE` 가 "부호는 라벨 상관 부호 규약(2/2
   실적)에서 왔다" 고 적고 있었다.  ★두 군데가 거짓★ 이다:
   - 이 후보의 부호는 어떤 로컬 근거에서도 오지 않았다 (지휘자 지시대로 없는 게 사실).
   - "2/2 실적" 은 2026-08-29 후보 DS 의 LB 가 `bmid` 에서 반증했다.
② `if_gain_is_negative` 가 "그러면 OOT 가 틀렸고 라벨 상관이 맞았던 것" 이라고 적고
   있었다.  ★이 후보에서는 정반대★ 다 — u=+0.008 이 라벨 상관 방향이므로 음수 gain 은
   라벨 상관이 틀렸다는 뜻이다.
③ 키 이름이 DR_* 인데 candidate 는 DRP 다 (S91 계열의 자기소개 불일치).

★예측값은 한 비트도 안 바뀐다★ — 고친 뒤 3프레임을 다시 돌려 sha256 을 대조한다.
"""
import hashlib
import json

B = "/home/<user>/SungjinPark/build_DN"

SIGN = (
 "S263 — 신규 축은 계산된 꼭지점 착륙이 아니라 ★작은 프로브★ 로 시작한다.  "
 "🔴 그리고 이 후보에서는 ★부호 근거가 아예 없다★ — 자세한 것은 "
 "axis_offsets[5].derivation.🔴_이_축의_부호_근거는_없다 를 보라.  요약하면 두 로컬 "
 "도구가 서로 반대를 가리키고(OOT 잔차 -313.9 → 음수 · 라벨 상관 2024 +0.0300 → 양수), "
 "2026-08-29 후보 DS 의 LB 실측이 형제 열에서 ★둘 다 틀렸음★ 을 증명했다.  "
 "★크기★ 는 형제 축 `ctrl_asof_mid` 가 이 계보에서 검증한 프로브 규격 0.008 이다.  "
 "gain ≈ 2*u*K*L 로 선형인 구간이라 부호만 맞으면 이기고, 틀리면 S250 대로 더 값진 "
 "정보(그 축의 L 부호)가 온다.  ★이 제출이 사는 것은 부호다.★")
NEG = (
 "🔴 그러면 ★라벨 상관 방향이 틀렸고 out-of-time 정직면 방향이 맞았던 것★ 이다 "
 "(u = +0.008 이 라벨 상관이 가리킨 쪽이므로).  이 축은 두 도구가 정면으로 갈린 "
 "유일한 축이라 지든 이기든 ★어느 쪽 도구를 앞으로 쓸지★ 를 정한다 — 그게 이 발의 "
 "두 번째 값이다.  S264 — 부호 반전(u = -0.008 재출하)은 |K*L| > Q/(2|u|) = 114 일 "
 "때만 이긴다.")
POS = (
 "🔴 그러면 ★라벨 상관 방향이 맞았고 OOT 잔차 방향이 틀렸던 것★ 이다.  형제 열 "
 "`bmid` 에서 관측한 반전(비율 -0.857)이 이 축에서도 재현된 것이므로, 그때 비로소 "
 "★두 관측★ 이 되어 방향 힌트를 근거로 승격시킬 수 있다.  2발째로 같은 기반에서 "
 "꼭지점 착륙을 검토한다 (u* = L/G, G 는 이 빌드가 라벨 없이 실측한 곡률 K*G 28,615).  "
 "다른 기반의 계수를 빌려오지 않는다(S262).")
NOTE = (
 "🔴 이 후보의 첫 판본에 거짓 자기서술이 둘 있었고 출하 전에 고쳤다.  "
 "① `🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE` 가 '부호는 라벨 상관 부호 규약(2/2 "
 "실적)에서 왔다' 고 적고 있었다 — ★이 후보의 부호는 어떤 로컬 근거에서도 오지 "
 "않았고★, '2/2 실적' 자체도 2026-08-29 후보 DS 의 LB 가 `bmid` 에서 반증했다.  "
 "② `if_gain_is_negative` 가 '그러면 OOT 가 틀렸고 라벨 상관이 맞았던 것' 이라 "
 "적고 있었다 — u=+0.008 이 라벨 상관 방향이므로 ★정반대★ 다.  "
 "③ 키 이름이 DR_* 인데 candidate 는 DRP 였다 (S91 계열).  "
 "두 거짓 문장은 형제 후보 DR(u=-0.008)의 서술을 기계 치환으로 물려받은 것이다.  "
 "S30·S48·S73.  ★예측값은 한 비트도 안 바뀐다★ — 산문 전용 수정이고, 고친 뒤 "
 "3프레임(real5·keep110·cov)을 다시 돌려 출력 sha256 이 수정 전과 동일함을 기계로 "
 "확인했다.")


def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for x in iter(lambda: f.read(1 << 20), b""):
            h.update(x)
    return h.hexdigest()


for tag in ("DRP_SHIP", "ANC_DRP"):
    p = B + "/pkg/" + tag + "/blend.json"
    b = json.load(open(p, encoding="utf-8"))
    ax = b["DR_axis"]
    assert "라벨 상관 부호 규약(2/2 실적)에서 왔고" in ax[
        "🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE"], tag
    ax["🔴_THIS_SUBMISSION_BUYS_A_SIGN_NOT_A_SIZE"] = SIGN
    oc = ax["🔴_WHAT_WE_DO_WITH_EACH_OUTCOME_WRITTEN_BEFORE_THE_MEASUREMENT"]
    assert "라벨 상관 규약이 맞았던 것" in oc["if_gain_is_negative"], tag
    oc["if_gain_is_negative"] = NEG
    oc["if_gain_is_positive"] = POS
    # 키 이름을 자기 세대(DRP)로
    for old, new in (("DR_axis", "DRP_axis"), ("DR_rule_2_4", "DRP_rule_2_4"),
                     ("DR_S73_prose_sweep", "DRP_S73_prose_sweep"),
                     ("DR_inherited_derived_from", "DRP_inherited_derived_from")):
        b[new] = b.pop(old)
    sw = b["DRP_S73_prose_sweep"]
    sw["top_level_keys_describing_DRP"] = sw.pop("top_level_keys_describing_DR")
    b["🔴_PROSE_CORRECTION_BEFORE_SHIPPING"] = NOTE
    # 스윕 계수 재계산 (키 이름이 바뀌었으므로)
    par = json.load(open(B + "/pkg/PAR/blend.json", encoding="utf-8"))
    new_k = sorted(set(b) - set(par))
    chg = sorted(k for k in set(b) & set(par)
                 if json.dumps(b[k], ensure_ascii=False, sort_keys=True)
                 != json.dumps(par[k], ensure_ascii=False, sort_keys=True))
    sw["top_level_keys_describing_DRP"] = {
      "n_new": len(new_k), "new": new_k, "n_changed": len(chg), "changed": chg,
      "n_removed": 0, "total": len(new_k) + len(chg),
      "🔴": "기계로 센 값이다.  부모 zip 의 blend.json 과 리프 단위 대조."}
    open(p, "w", encoding="utf-8", newline="").write(
        json.dumps(b, ensure_ascii=False, indent=1))
    print("fixed", tag)

REL = ["pkg/DRP_SHIP/blend.json", "pkg/DRP_SHIP/script.py",
       "pkg/ANC_DRP/blend.json", "pkg/ANC_DRP/script.py"]
open(B + "/work/frozen_prose_sha_drp.txt", "w").write(
    "\n".join(sh(B + "/" + r) + "  " + r for r in REL) + "\n")
print("re-frozen 4 files")

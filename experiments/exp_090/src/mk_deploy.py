"""exp_090 stage 2 — 배포 npz 5개 + prep.json 을 팀원 패키지와 같은 파일 배치로 모은다.

사용: mk_deploy.py <TAGPREFIX> <TOTAL_STEPS>
전제: train090.py 가 SAVE_NPZ=1 로 `out/npz/deepf_<PRE>_le2024_s<seed>.npz` 를 이미 썼다.
결과: out/deploy_<PRE>/{deepf_s42.npz,...,prep.json}   ← 팀원 `model/deep/` 과 같은 이름 규약

🔴 prep.json 은 팀원 것에서 **입력 변환에 닿는 필드(NUM·CAT·levels·mu·sd·D·batch)를
   글자 그대로** 승계하고, 학습량 기록(epoch/total_steps/provenance)만 우리 값으로 바꾼다.
   그래야 배포 팔 대비 **변인이 학습량 하나**가 된다.
"""
import json, os, shutil, sys
import numpy as np

H = "/home/<user>/SungjinPark"
B90 = f"{H}/exp090"
PKG = f"{H}/audit_deep/pkg/DEEP/mate"
PRE, STEPS = sys.argv[1], int(sys.argv[2])
S5 = [42, 777, 2024, 101, 7]
D = f"{B90}/out/deploy_{PRE}"
os.makedirs(D, exist_ok=True)
for s in S5:
    src = f"{B90}/out/npz/deepf_{PRE}_le2024_s{s}.npz"
    assert os.path.exists(src), src
    shutil.copyfile(src, f"{D}/deepf_s{s}.npz")
P = json.load(open(f"{PKG}/model/deep/prep.json", encoding="utf-8"))
spe = 1475092 // 4096
P["epoch"] = STEPS / spe
P["total_steps"] = STEPS
P["lr_schedule"] = "constant (no scheduler), lr=0.001"
P["provenance"] = (P.get("provenance", "") +
                   f" | exp_090: 학습량만 재선택 (총 옵티마이저 스텝 {STEPS}, "
                   f"steps_per_epoch {spe}). NUM/CAT/levels/mu/sd/D/batch 는 원본 승계.")
json.dump(P, open(f"{D}/prep.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(json.dumps({"dir": D, "files": sorted(os.listdir(D)),
                  "epoch": P["epoch"], "total_steps": STEPS,
                  "sizes": {f: os.path.getsize(f"{D}/{f}")
                            for f in sorted(os.listdir(D))}},
                 ensure_ascii=False, indent=1))

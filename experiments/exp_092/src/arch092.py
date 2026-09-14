"""exp_092 — 팀원 딥 팔(token-MLP) 구조. **arch091(=arch090).py 를 폭·깊이 파라미터화한 것뿐이다.**

🔴 HID=(256,128,64) 일 때 `nn.Sequential` 의 레이어 인덱스가 arch091(=arch090) 과 **글자 그대로 같다**:
     i=0        Linear[0] LayerNorm[1] GELU[2] Dropout[3]
     i=1        Linear[4] LayerNorm[5] GELU[6] Dropout[7]
     i=2(마지막) Linear[8] GELU[9]
     출력       Linear[10]
   그래서 기본값으로 돌리면 exp_090 과 **바이트 동일한 z** 가 나와야 한다 — 재현 게이트가 증명한다.

🔴 **구조를 바꾼 지점은 팀원 `mate/deep_infer.py` 로 출하되지 않는다 — 폭도 깊이도.**
   ① 깊이: `_one` 이 m.1/m.4/m.5/m.8/m.10 을 **하드코딩**한다(깊이 4 는 m.12/m.14 가 된다)
   ② 폭:   `_fold` 가 `W0.reshape(256, 99, 96)` 으로 **첫 은닉 폭 256 을 하드코딩**한다
   ②는 계획 단계에서 "폭은 형상만 달라 먹는다"고 본 것이 **틀렸음**이 실행 중 드러난 것이다
   (B_w05_d0 5 fit 이 `cannot reshape array of size 1216512` 로 죽었다).
"""
import numpy as np
import torch
import torch.nn as nn

D_TOK = 96
HID_BASE = (256, 128, 64)


class TokenMLP(nn.Module):
    """수치 94열을 열마다 D차 토큰으로 펴고 범주 5열 임베딩과 이어 붙인 뒤 MLP."""

    def __init__(self, n_num, cards, D=D_TOK, p_drop=0.1, init="randn",
                 hid=HID_BASE):
        super().__init__()
        self.n_num, self.D, self.cards = n_num, D, list(cards)
        self.hid = tuple(int(h) for h in hid)
        assert len(self.hid) >= 2
        self.w = nn.Parameter(torch.empty(n_num, D))
        self.b = nn.Parameter(torch.empty(n_num, D))
        self.ce = nn.ModuleList([nn.Embedding(c, D) for c in cards])
        n_tok = n_num + len(cards)
        layers, prev = [], n_tok * D
        for i, h in enumerate(self.hid):
            if i < len(self.hid) - 1:          # LN + GELU + Dropout 블록
                layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(),
                           nn.Dropout(p_drop)]
            else:                              # 마지막 은닉: GELU 만
                layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.m = nn.Sequential(*layers)
        self.reset_tokenizer(init)

    def reset_tokenizer(self, init):
        with torch.no_grad():
            if init == "randn":          # w,b ~ N(0,1)  (nn.Embedding 기본과 같은 규약)
                self.w.normal_(0.0, 1.0)
                self.b.normal_(0.0, 1.0)
            elif init == "randn_w":      # w ~ N(0,1), b = 0
                self.w.normal_(0.0, 1.0)
                self.b.zero_()
            elif init == "ones0":        # w = 1, b = 0 (항등 토큰화)
                self.w.fill_(1.0)
                self.b.zero_()
            elif init == "ftt":          # FT-Transformer 수치 토크나이저 관례 U(±1/√D)
                bd = 1.0 / np.sqrt(self.D)
                self.w.uniform_(-bd, bd)
                self.b.uniform_(-bd, bd)
            else:
                raise ValueError(init)
        self.init_kind = init

    def forward(self, xc, xd):
        t = xd.unsqueeze(-1) * self.w + self.b                      # (B, n_num, D)
        e = torch.stack([emb(xc[:, j]) for j, emb in enumerate(self.ce)], 1)
        z = torch.cat([t, e], 1).reshape(xd.shape[0], -1)           # (B, 99*D)
        return self.m(z).squeeze(1)


RAW_KEYS_BASE = ["m.0.weight", "m.0.bias", "w", "b",
                 "m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias",
                 "m.5.weight", "m.5.bias", "m.8.weight", "m.8.bias",
                 "m.10.weight", "m.10.bias"]


def raw_npz(model):
    """`deep_infer._fold` 가 먹는 **원본 아티팩트** 형태. 깊이 3(현행)에서만 키가 일치한다."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    keys = [k for k in sd if k.startswith("m.")] + ["w", "b"]
    out = {k: sd[k] for k in keys}
    for j in range(len(model.cards)):
        out[f"ce.{j}.weight"] = sd[f"ce.{j}.weight"]
    return out


def foldable(model):
    """팀원 `_fold`/`_one` 이 **수정 없이** 먹는 조건.

    🔴 실측(exp_092): `deep_infer._fold` 는 `W0.reshape(256, 99, 96)` 로 **첫 은닉 폭 256 을
       하드코딩**한다. 따라서 폭을 바꾸면 깊이가 3 이어도 안 먹는다 — 폭 축 전체가
       팀원 파일 수정 없이는 출하 불가다(계획 단계에서 "폭은 형상만 달라 먹는다"고 본 것은 틀렸다).
       깊이는 `_one` 이 m.1/m.4/m.5/m.8/m.10 을 하드코딩해 역시 안 먹는다.
    """
    return len(model.hid) == 3 and model.hid[0] == 256

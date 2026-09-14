"""exp_093 — 팀원 딥 팔(token-MLP) 구조. **arch092.py 를 출력 폭만 파라미터화한 것뿐이다.**

🔴 바뀐 것은 **마지막 Linear 의 출력 차원 하나**다 (`n_out`, 기본 1).
   `n_out=1` 이면 `nn.Linear(prev, 1)` 로 arch092(=arch091=arch090) 과 **글자 그대로 같은 모듈 트리**이고,
   같은 순서로 같은 개수의 난수를 뽑는다 → z 가 바이트 동일해야 한다(재현 게이트가 증명한다).

🔴 **`n_out>1` 은 두 가지를 동시에 바꾼다 — 변인 열거에 반드시 적는다**:
   ① 파라미터 수: 마지막 Linear 가 (64→1) 에서 (64→5) 로 = **+260 파라미터**
   ② **초기화 난수 소비**: `reset_tokenizer` 가 `nn.Sequential` **생성 뒤**에 불리므로,
      마지막 Linear 가 난수를 더 뽑으면 그 뒤의 `w`·`b` 토크나이저 초기화가 **다른 난수**를 받는다.
      → 같은 시드라도 토크나이저 초기값이 다르다. 이건 피할 수 없다(순서를 바꾸면 재현 게이트가 깨진다).
   ②의 몫만 따로 재려면 `n_out=5` 로 두고 목적함수를 실패 4셀 균등(=BCE 와 주변확률 동일)으로
   두는 대조(`CTRL_nout5_collapsed`)를 돌린다 — train093.py 의 TARGET093=collapsed5.

🔴 출하 제약(S101) 그대로: 폭 256·깊이 3 을 안 건드린다. 단 `n_out>1` 은 팀원
   `mate/deep_infer.py` 의 `_fold` 가 `m.10.weight` 를 (1, 64) 로 보므로 **그대로는 출하 불가**다.
   출하 경로는 stage1 통과 후에 따로 설계한다(성공 로짓 하나로 접는 사영이 필요).
"""
import numpy as np
import torch
import torch.nn as nn

D_TOK = 96
HID_BASE = (256, 128, 64)


class TokenMLP(nn.Module):
    """수치 94열을 열마다 D차 토큰으로 펴고 범주 5열 임베딩과 이어 붙인 뒤 MLP."""

    def __init__(self, n_num, cards, D=D_TOK, p_drop=0.1, init="randn",
                 hid=HID_BASE, n_out=1):
        super().__init__()
        self.n_num, self.D, self.cards = n_num, D, list(cards)
        self.hid = tuple(int(h) for h in hid)
        self.n_out = int(n_out)
        assert len(self.hid) >= 2 and self.n_out >= 1
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
        layers += [nn.Linear(prev, self.n_out)]
        self.m = nn.Sequential(*layers)
        self.reset_tokenizer(init)

    def reset_tokenizer(self, init):
        with torch.no_grad():
            if init == "randn":          # w,b ~ N(0,1)
                self.w.normal_(0.0, 1.0)
                self.b.normal_(0.0, 1.0)
            elif init == "randn_w":
                self.w.normal_(0.0, 1.0)
                self.b.zero_()
            elif init == "ones0":
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
        o = self.m(z)
        return o.squeeze(1) if self.n_out == 1 else o


def raw_npz(model):
    """`deep_infer._fold` 가 먹는 **원본 아티팩트** 형태. n_out=1·깊이 3 에서만 키가 일치한다."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    keys = [k for k in sd if k.startswith("m.")] + ["w", "b"]
    out = {k: sd[k] for k in keys}
    for j in range(len(model.cards)):
        out[f"ce.{j}.weight"] = sd[f"ce.{j}.weight"]
    return out


def foldable(model):
    """팀원 `_fold`/`_one` 이 **수정 없이** 먹는 조건.
    🔴 exp_093 추가: 출력 폭도 조건이다 — `_fold` 가 `m.10.weight` 를 (1, 64) 로 전제한다."""
    return len(model.hid) == 3 and model.hid[0] == 256 and model.n_out == 1

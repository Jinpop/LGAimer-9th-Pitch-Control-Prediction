"""exp_089 stage 0 — 팀원 딥 팔(token-MLP) 구조 복원.

🔴 복원의 근거는 세 가지뿐이고 전부 읽은 것이다:
  ① `mate/deep_infer.py::_fold` 가 **원본 아티팩트에서 요구하는 키와 형상**
       m.0.weight (256, 99*D) · m.0.bias (256,)
       w (94, D) · b (94, D)        → 수치 토큰  token_i = x_i·w[i] + b[i]
       ce.{j}.weight (card_j, D)    → 범주 토큰  token_{94+j} = ce_j[c_j]
     `B0 = W0.reshape(256, 99, 96)` 이 토큰 배치를 [수치 94 | 범주 5], 토큰폭 96 으로 못박는다.
  ② `mate/deep_infer.py::_one` 의 순전파 순서
       Linear → LN(m.1) → GELU → [Dropout] → Linear(m.4) → LN(m.5) → GELU → [Dropout]
       → Linear(m.8) → GELU → Linear(m.10)
     파라미터를 갖는 인덱스가 0,1,4,5,8,10 이고 2,3,6,7,9 가 안 가지므로
     비어 있는 자리는 GELU/Dropout/GELU/Dropout/GELU 로 유일하게 채워진다.
  ③ `prep.json`  D=96 · lr 1e-3 · epoch 2 · batch 4096 · NUM 94 · CAT 5 · levels · mu · sd

⚠️ **복원되지 않는 것 (probe 로 식별한다)**: Dropout 확률 · 최적화기(Adam/AdamW·wd) ·
   `w`/`b` 초기화. npz 는 **접힌** 아티팩트라 원본 키가 안 들어 있어 형상조차 직접 못 본다.
   → `_fold` 를 우리 파라미터에 태워 나온 A·c0·T 의 형상이 출하 npz 와 키별로 같은지로 검증한다.

⚠️ `prep.json` 의 `ED = 8` 은 이 구조 어디에도 안 들어간다. `_fold` 가 `ce.{j}.weight` 를
   (card_j, 96) 로 곧장 곱하므로 범주 임베딩 폭은 D=96 이지 ED 가 아니다. 미사용 기록으로 본다.
"""
import numpy as np
import torch
import torch.nn as nn

D_TOK = 96
HID = (256, 128, 64)


class TokenMLP(nn.Module):
    """수치 94열을 열마다 D차 토큰으로 펴고 범주 5열 임베딩과 이어 붙인 뒤 MLP."""

    def __init__(self, n_num, cards, D=D_TOK, p_drop=0.1, init="randn"):
        super().__init__()
        self.n_num, self.D, self.cards = n_num, D, list(cards)
        self.w = nn.Parameter(torch.empty(n_num, D))
        self.b = nn.Parameter(torch.empty(n_num, D))
        self.ce = nn.ModuleList([nn.Embedding(c, D) for c in cards])
        n_tok = n_num + len(cards)
        self.m = nn.Sequential(
            nn.Linear(n_tok * D, HID[0]), nn.LayerNorm(HID[0]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(HID[0], HID[1]), nn.LayerNorm(HID[1]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(HID[1], HID[2]), nn.GELU(),
            nn.Linear(HID[2], 1))
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


RAW_KEYS = ["m.0.weight", "m.0.bias", "w", "b",
            "m.1.weight", "m.1.bias", "m.4.weight", "m.4.bias",
            "m.5.weight", "m.5.bias", "m.8.weight", "m.8.bias",
            "m.10.weight", "m.10.bias"]


def raw_npz(model):
    """`deep_infer._fold` 가 먹는 **원본 아티팩트** 형태로 내보낸다."""
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    out = {k: sd[k] for k in RAW_KEYS}
    for j in range(len(model.cards)):
        out[f"ce.{j}.weight"] = sd[f"ce.{j}.weight"]
    return out

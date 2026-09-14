"""exp_096 — 새 팔 아키텍처.

TokenMLP 은 exp_090 arch090.py 를 **그대로 재수입**한다 (복제 게이트가 그 파일에 걸려 있다).
여기서 새로 정의하는 것은 A1 어텐션 팔 하나뿐이다.

A1_attn — FT-Transformer 축소판
  토큰화는 딥 팔과 **같은 규약**(수치 열마다 token_i = x_i·w_i + b_i, 범주는 임베딩).
  달라지는 것은 그 뒤다: 99 토큰을 펴서 한 번에 Linear 로 밟는 대신
  self-attention 블록으로 **토큰 쌍 상호작용을 명시적으로** 만든다.
  pre-LN 을 쓴다 — post-LN 은 warmup 없이 lr 1e-3 에서 자주 발산하는데
  이 실험은 팀원 동결 레시피(상수 lr 1e-3, warmup 없음)를 유지해야 하기 때문이다.
"""
import numpy as np
import torch
import torch.nn as nn


# ── 딥 팔 원본 (exp_090 arch090.py 와 동일) ────────────────────────────
D_TOK = 96
HID = (256, 128, 64)


class TokenMLP(nn.Module):
    def __init__(self, n_num, cards, D=D_TOK, p_drop=0.1, init="randn",
                 hid=HID):
        super().__init__()
        self.n_num, self.D, self.cards = n_num, D, list(cards)
        self.w = nn.Parameter(torch.empty(n_num, D))
        self.b = nn.Parameter(torch.empty(n_num, D))
        self.ce = nn.ModuleList([nn.Embedding(c, D) for c in cards])
        n_tok = n_num + len(cards)
        self.m = nn.Sequential(
            nn.Linear(n_tok * D, hid[0]), nn.LayerNorm(hid[0]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid[0], hid[1]), nn.LayerNorm(hid[1]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid[1], hid[2]), nn.GELU(),
            nn.Linear(hid[2], 1))
        self.reset_tokenizer(init)

    def reset_tokenizer(self, init):
        with torch.no_grad():
            if init == "randn":
                self.w.normal_(0.0, 1.0); self.b.normal_(0.0, 1.0)
            elif init == "randn_w":
                self.w.normal_(0.0, 1.0); self.b.zero_()
            elif init == "ones0":
                self.w.fill_(1.0); self.b.zero_()
            elif init == "ftt":
                bd = 1.0 / np.sqrt(self.D)
                self.w.uniform_(-bd, bd); self.b.uniform_(-bd, bd)
            else:
                raise ValueError(init)
        self.init_kind = init

    def forward(self, xc, xd):
        t = xd.unsqueeze(-1) * self.w + self.b
        e = torch.stack([emb(xc[:, j]) for j, emb in enumerate(self.ce)], 1)
        z = torch.cat([t, e], 1).reshape(xd.shape[0], -1)
        return self.m(z).squeeze(1)


# ── A1 : FT-Transformer 축소판 ────────────────────────────────────────
class Block(nn.Module):
    def __init__(self, D, heads, mult, p_drop):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.att = nn.MultiheadAttention(D, heads, dropout=p_drop,
                                         batch_first=True)
        self.ln2 = nn.LayerNorm(D)
        self.ffn = nn.Sequential(nn.Linear(D, mult * D), nn.GELU(),
                                 nn.Dropout(p_drop), nn.Linear(mult * D, D))
        self.dp = nn.Dropout(p_drop)

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.att(h, h, h, need_weights=False)
        x = x + self.dp(a)
        return x + self.dp(self.ffn(self.ln2(x)))


class FTAttn(nn.Module):
    def __init__(self, n_num, cards, D=128, blocks=3, heads=8, mult=2,
                 p_drop=0.0, init="ftt"):
        super().__init__()
        self.n_num, self.D, self.cards = n_num, D, list(cards)
        self.w = nn.Parameter(torch.empty(n_num, D))
        self.b = nn.Parameter(torch.empty(n_num, D))
        self.ce = nn.ModuleList([nn.Embedding(c, D) for c in cards])
        self.cls = nn.Parameter(torch.zeros(1, 1, D))
        self.blocks = nn.ModuleList([Block(D, heads, mult, p_drop)
                                     for _ in range(blocks)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)
        with torch.no_grad():
            bd = 1.0 / np.sqrt(D)
            if init == "ftt":
                self.w.uniform_(-bd, bd); self.b.uniform_(-bd, bd)
            else:
                self.w.normal_(0.0, 1.0); self.b.normal_(0.0, 1.0)
            self.cls.normal_(0.0, 0.02)
        self.init_kind = init

    def forward(self, xc, xd):
        t = xd.unsqueeze(-1) * self.w + self.b                  # (B, n_num, D)
        e = torch.stack([emb(xc[:, j]) for j, emb in enumerate(self.ce)], 1)
        x = torch.cat([self.cls.expand(xd.shape[0], -1, -1), t, e], 1)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x[:, 0])).squeeze(1)


# ── A4 : 시퀀스 팔 (stage2) ───────────────────────────────────────────
class SeqArm(nn.Module):
    """정적 99열(딥 팔과 같은 토큰화) + 투수 동결 시퀀스(K x Fs) 인과 합성곱 가지.

    🔴 시퀀스는 **동결표**다 — 시즌 s 행은 season<s 에서 만든 그 투수의 마지막 K 투구를 본다.
       (배포: <=2024 표 -> 2025 행). test 안에서 아무것도 모으지 않으므로 규칙 2-4 를 지킨다.
    """

    def __init__(self, n_num, cards, Fs, K=16, D=64, Cs=64, p_drop=0.0,
                 hid=(256, 128, 64)):
        super().__init__()
        self.n_num, self.D, self.cards, self.K, self.Fs = n_num, D, list(cards), K, Fs
        self.w = nn.Parameter(torch.empty(n_num, D))
        self.b = nn.Parameter(torch.empty(n_num, D))
        self.ce = nn.ModuleList([nn.Embedding(c, D) for c in cards])
        bd = 1.0 / np.sqrt(D)
        with torch.no_grad():
            self.w.uniform_(-bd, bd); self.b.uniform_(-bd, bd)
        self.proj = nn.Conv1d(Fs + 1, Cs, 1)                 # +1 = 유효 마스크
        self.c1 = nn.Conv1d(Cs, Cs, 3, padding=1)
        self.c2 = nn.Conv1d(Cs, Cs, 3, padding=2, dilation=2)
        self.gn = nn.GroupNorm(1, Cs)
        n_tok = n_num + len(cards)
        self.m = nn.Sequential(
            nn.Linear(n_tok * D + 2 * Cs, hid[0]), nn.LayerNorm(hid[0]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid[0], hid[1]), nn.LayerNorm(hid[1]), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(hid[1], hid[2]), nn.GELU(),
            nn.Linear(hid[2], 1))

    def forward(self, xc, xd, xs):
        t = xd.unsqueeze(-1) * self.w + self.b
        e = torch.stack([emb(xc[:, j]) for j, emb in enumerate(self.ce)], 1)
        z = torch.cat([t, e], 1).reshape(xd.shape[0], -1)
        s = self.proj(xs)                                    # (B, Cs, K)
        s = torch.nn.functional.gelu(self.gn(self.c1(s)))
        s = torch.nn.functional.gelu(self.c2(s))
        s = torch.cat([s.mean(-1), s[:, :, -1]], 1)          # 평균 풀 + 마지막 투구
        return self.m(torch.cat([z, s], 1)).squeeze(1)


def n_params(m):
    return int(sum(p.numel() for p in m.parameters()))

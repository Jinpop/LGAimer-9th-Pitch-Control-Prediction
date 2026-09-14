"""exp_108 — 딥 팔 구조. **arch102b.py 를 초기화 파라미터화한 것뿐이다.**

🔴 바뀐 것은 `reset_tokenizer` 에 **분기를 더한 것 하나**다. 기존 네 이름
   (`randn` · `randn_w` · `ones0` · `ftt`) 의 본문은 **한 글자도 안 건드렸고**
   `__init__` 의 파라미터 생성·`nn.Sequential` 구성 순서도 그대로다
   → 난수 소비 순서가 같아 `ftt` 는 exp_104/107 과 **바이트 동일한 z** 를 낸다
     (재현 게이트 G1/G2 가 증명한다).

🔴 **초기화가 닿는 곳은 수치 토크나이저 `w`·`b` 뿐이다** — `nn.Linear`(m.*) 와
   `nn.Embedding`(ce.*) 은 파이토치 기본값 그대로다. arch089/102b 이래 같다.
   · nn.Linear   기본: W ~ U(±1/√fan_in),  bias ~ U(±1/√fan_in)
   · nn.Embedding 기본: N(0, 1)

새 초기화 문법 (INIT 문자열):
   u<mult>    w,b ~ U(±mult/√D)        mult=1 이면 `ftt` 와 같은 분포(난수 호출도 같다)
   n<sd>      w,b ~ N(0, sd)           🔴 sd 는 **표준편차**다 (분산 아님)
   nw<sd>     w  ~ N(0, sd), b = 0
   접미 @ce<m>  위에 더해 **범주 임베딩 ce.* 전체에 상수 m 을 곱한다**
              (귀무 N4 — 같은 크기의 스케일 변경을 *토크나이저가 아닌 이웃 블록*에 준다.
               난수를 더 안 쓰고 곱만 하므로 결정적이다.)

D = 96 이므로:
   ftt            = U(±0.10206207)   sd 0.05892557
   n0.10206207    = N(0, 0.10206207) = D^-0.5 를 **표준편차**로 읽은 것 (팀원 문구의 문자 그대로)
   n0.31950791    = N(0, 0.31950791) = D^-0.25 = D^-0.5 를 **분산**으로 읽은 것
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

    # ── 초기화 ───────────────────────────────────────────────────────────
    def reset_tokenizer(self, init):
        spec, ce_mult = init, None
        if "@ce" in init:
            spec, _cm = init.split("@ce", 1)
            ce_mult = float(_cm)
        with torch.no_grad():
            if spec == "randn":          # w,b ~ N(0,1)  (nn.Embedding 기본과 같은 규약)
                self.w.normal_(0.0, 1.0)
                self.b.normal_(0.0, 1.0)
            elif spec == "randn_w":      # w ~ N(0,1), b = 0
                self.w.normal_(0.0, 1.0)
                self.b.zero_()
            elif spec == "ones0":        # w = 1, b = 0 (항등 토큰화)
                self.w.fill_(1.0)
                self.b.zero_()
            elif spec == "ftt":          # FT-Transformer 수치 토크나이저 관례 U(±1/√D)
                bd = 1.0 / np.sqrt(self.D)
                self.w.uniform_(-bd, bd)
                self.b.uniform_(-bd, bd)
            # ── exp_108 이 더한 분기 셋 ─────────────────────────────────
            elif spec.startswith("nw"):  # w ~ N(0, sd), b = 0   (층별 축 C)
                sd = float(spec[2:])
                assert sd > 0
                self.w.normal_(0.0, sd)
                self.b.zero_()
            elif spec.startswith("uw"):  # w ~ U(±mult/√D), b = 0  (층별 축 C)
                mult = float(spec[2:])
                assert mult > 0
                bd = mult / np.sqrt(self.D)
                self.w.uniform_(-bd, bd)
                self.b.zero_()
            elif spec.startswith("uv"):  # w 만 배율, b 는 ftt 그대로 (층별 축 C)
                mult = float(spec[2:])
                assert mult > 0
                bd = 1.0 / np.sqrt(self.D)
                self.w.uniform_(-bd * mult, bd * mult)
                self.b.uniform_(-bd, bd)
            elif spec.startswith("ub"):  # w 는 ftt 그대로, b 만 배율 (층별 축 C)
                mult = float(spec[2:])
                assert mult > 0
                bd = 1.0 / np.sqrt(self.D)
                self.w.uniform_(-bd, bd)
                self.b.uniform_(-bd * mult, bd * mult)
            elif spec.startswith("n"):   # w,b ~ N(0, sd)   sd 는 표준편차
                sd = float(spec[1:])
                assert sd > 0
                self.w.normal_(0.0, sd)
                self.b.normal_(0.0, sd)
            elif spec.startswith("u"):   # w,b ~ U(±mult/√D)
                mult = float(spec[1:])
                assert mult > 0
                bd = mult / np.sqrt(self.D)
                self.w.uniform_(-bd, bd)
                self.b.uniform_(-bd, bd)
            else:
                raise ValueError(init)
            if ce_mult is not None:      # 귀무 N4 — 이웃 블록에 같은 형태의 스케일
                for emb in self.ce:
                    emb.weight.mul_(ce_mult)
        self.init_kind = init
        self.init_stats = {
            "spec": spec, "ce_mult": ce_mult,
            "w_sd": float(self.w.detach().std()),
            "w_absmax": float(self.w.detach().abs().max()),
            "b_sd": float(self.b.detach().std()),
            "ce_sd": [float(e.weight.detach().std()) for e in self.ce],
        }

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

    🔴 초기화는 형상을 안 바꾸므로 이 조건에 영향이 없다 — 그것이 이 축의 출하 근거다.
    """
    return len(model.hid) == 3 and model.hid[0] == 256

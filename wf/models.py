"""Trunks + heads, small enough to try a lot of variants on one L4."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import DYN_IDX, STA_IDX


def cna(cin: int, cout: int, k: int = 3, stride: int = 1, dilation: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, k, stride, padding=dilation * (k // 2), dilation=dilation, bias=False),
        nn.BatchNorm2d(cout),
        nn.GELU(),
    )


class SE(nn.Module):
    def __init__(self, c: int, r: int = 8):
        super().__init__()
        self.fc1 = nn.Conv2d(c, max(8, c // r), 1)
        self.fc2 = nn.Conv2d(max(8, c // r), c, 1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.gelu(self.fc1(w))))


class Res(nn.Module):
    def __init__(self, c: int, dilation: int = 1, drop_path: float = 0.0):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, dilation, dilation, bias=False)
        self.n1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, dilation, dilation, bias=False)
        self.n2 = nn.BatchNorm2d(c)
        self.se = SE(c)
        self.dp = drop_path

    def forward(self, x):
        h = F.gelu(self.n1(self.c1(x)))
        h = self.se(self.n2(self.c2(h)))
        if self.training and self.dp > 0:
            keep = 1.0 - self.dp
            mask = torch.rand(h.shape[0], 1, 1, 1, device=h.device) < keep
            h = h * mask / keep
        return x + h


class LK(nn.Module):
    """ConvNeXt-lite: 7x7 depthwise + inverted bottleneck, gives wide context cheaply."""

    def __init__(self, c: int, exp: int = 2, drop_path: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, 1, 3, groups=c, bias=False)
        self.n = nn.BatchNorm2d(c)
        self.pw1 = nn.Conv2d(c, exp * c, 1)
        self.pw2 = nn.Conv2d(exp * c, c, 1)
        self.gamma = nn.Parameter(torch.ones(c, 1, 1) * 1e-2)
        self.dp = drop_path

    def forward(self, x):
        h = self.n(self.dw(x))
        h = self.pw2(F.gelu(self.pw1(h)))
        if self.training and self.dp > 0:
            keep = 1.0 - self.dp
            h = h * (torch.rand(h.shape[0], 1, 1, 1, device=h.device) < keep) / keep
        return x + self.gamma * h


class FullRes(nn.Module):
    def __init__(self, cin: int, width: int, blocks: list[list[int]], block: str = "res",
                 fusion: str = "concat", drop_path: float = 0.0):
        super().__init__()
        self.fusion = fusion
        if fusion == "gated":
            half = max(16, width // 2)
            self.stem_s = cna(len(STA_IDX), half)
            self.stem_d = cna(len(DYN_IDX), half)
            self.gate = nn.Sequential(nn.Conv2d(width, width, 1), nn.Sigmoid())
        else:
            self.stem = cna(cin, width)
        mk = Res if block == "res" else LK
        self.blocks = nn.ModuleList([mk(width, *([d] if block == "res" else []), drop_path=drop_path)
                                     for d, n in blocks for _ in range(n)])
        self.out_c = width

    def forward(self, x):
        if self.fusion == "gated":
            s, d = x[:, STA_IDX], x[:, DYN_IDX]
            h = torch.cat([self.stem_s(s), self.stem_d(d)], 1)
            h = h * self.gate(h)
        else:
            h = self.stem(x)
        for b in self.blocks:
            h = b(h)
        return h


class ASPP(nn.Module):
    def __init__(self, c: int, dilations=(1, 2, 4, 8)):
        super().__init__()
        self.branches = nn.ModuleList([cna(c, c // 2, 3, dilation=d) for d in dilations])
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, c // 2, 1), nn.GELU())
        self.out = cna(c // 2 * (len(dilations) + 1), c, 1)

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        p = F.interpolate(self.pool(x), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return self.out(torch.cat(feats + [p], 1))


class AG(nn.Module):
    """Attention gate on the skip connection."""

    def __init__(self, f_g: int, f_l: int, f_int: int | None = None):
        super().__init__()
        f_int = f_int or max(8, f_l // 2)
        self.wg = nn.Conv2d(f_g, f_int, 1)
        self.wx = nn.Conv2d(f_l, f_int, 1)
        self.psi = nn.Conv2d(f_int, 1, 1)

    def forward(self, g, x):
        a = torch.sigmoid(self.psi(F.gelu(self.wg(g) + self.wx(x))))
        return x * a


class UNet(nn.Module):
    def __init__(self, cin: int, widths: list[int], depths: list[int], block: str = "res",
                 drop_path: float = 0.0, gated_skip: bool = True):
        super().__init__()
        mk = Res if block == "res" else LK
        self.stem = cna(cin, widths[0])
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i, w in enumerate(widths):
            self.enc.append(nn.Sequential(*[mk(w, 1, drop_path) if block == "res" else mk(w, drop_path=drop_path)
                                            for _ in range(depths[i])]))
            if i + 1 < len(widths):
                self.down.append(cna(w, widths[i + 1], 3, stride=2))
        self.bott = ASPP(widths[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        self.ag = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(cna(widths[i], widths[i - 1], 3))
            self.dec.append(nn.Sequential(*[mk(widths[i - 1], 1, drop_path) if block == "res"
                                            else mk(widths[i - 1], drop_path=drop_path) for _ in range(depths[i - 1])]))
            self.ag.append(AG(widths[i - 1], widths[i - 1]) if gated_skip else None)
        self.out_c = widths[0]

    def forward(self, x):
        h = self.stem(x)
        skips = []
        for i, e in enumerate(self.enc):
            h = e(h)
            if i + 1 < len(self.enc):
                skips.append(h)
                h = self.down[i](h)
        h = self.bott(h)
        for i, (u, d) in enumerate(zip(self.up, self.dec)):
            s = skips[-1 - i]
            h = F.interpolate(u(h), size=s.shape[-2:], mode="bilinear", align_corners=False)
            if self.ag[i] is not None:
                s = self.ag[i](h, s)
            h = d(h + s)
        return h


class HRNet(nn.Module):
    """3 parallel resolutions with repeated fusion; keeps full-res detail for fire fronts."""

    def __init__(self, cin: int, width: int = 32, stages: int = 3, block: str = "res", drop_path: float = 0.0):
        super().__init__()
        mk = Res if block == "res" else LK
        ws = [width, width * 2, width * 4]
        self.stem = nn.Sequential(cna(cin, ws[0]), cna(ws[0], ws[0]))
        self.branch = nn.ModuleList()
        self.fuse = nn.ModuleList()
        self.trans = nn.ModuleList()
        for i in range(1, 3):
            self.trans.append(cna(ws[i - 1], ws[i], 3, stride=2))
        for s in range(stages):
            b = nn.ModuleList([nn.Sequential(*[mk(w, 1, drop_path) if block == "res" else mk(w, drop_path=drop_path)
                                               for _ in range(2)]) for w in ws])
            self.branch.append(b)
            f = nn.ModuleList([nn.ModuleList([nn.Conv2d(o, w, 1, bias=False) for o in ws]) for w in ws])
            self.fuse.append(f)
        self.out_c = ws[0] + ws[1] + ws[2]

    def forward(self, x):
        h = self.stem(x)
        feats = [h]
        for i, t in enumerate(self.trans):
            feats.append(t(feats[-1]))
        for b, f in zip(self.branch, self.fuse):
            hs = [b[i](feats[i]) for i in range(3)]
            new = []
            for i in range(3):
                acc = hs[i]
                for j in range(3):
                    if j == i:
                        continue
                    t = f[i][j](hs[j])
                    if j > i:
                        t = F.interpolate(t, size=hs[i].shape[-2:], mode="bilinear", align_corners=False)
                    elif j < i:
                        t = F.avg_pool2d(t, 1 << (i - j))
                    acc = acc + t
                new.append(acc)
            feats = new
        up = [feats[0]] + [F.interpolate(f, size=feats[0].shape[-2:], mode="bilinear", align_corners=False)
                           for f in feats[1:]]
        return torch.cat(up, 1)


class CAFIM(nn.Module):
    """Cross-attentive fuel/weather gate: alpha for fuel, 1-alpha for weather."""

    def __init__(self, c: int):
        super().__init__()
        self.wf = nn.Conv2d(c, c, 1)
        self.ww = nn.Conv2d(c, c, 1)
        self.g = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, fs, fd):
        a = torch.sigmoid(self.g(torch.cat([self.wf(fs), self.ww(fd)], 1)))
        return torch.cat([a * fs, (1 - a) * fd], 1)


class DualUNet(nn.Module):
    """Two encoders (fuel 3x3, weather 5x5) fused per scale, U-Net decoder."""

    def __init__(self, cin: int, widths=(32, 64, 128), depths=(2, 2, 2), drop_path: float = 0.0):
        super().__init__()
        w = widths
        self.stem_s = cna(len(STA_IDX), w[0], 3)
        self.stem_d = cna(len(DYN_IDX), w[0], 5, dilation=2)

        def block(c, k):
            return nn.Sequential(cna(c, c, k), cna(c, c, k))

        self.es = nn.ModuleList([nn.Sequential(*[block(w[i], 3) for _ in range(depths[i])]) for i in range(3)])
        self.ed = nn.ModuleList([nn.Sequential(*[block(w[i], 5) for _ in range(depths[i])]) for i in range(3)])
        self.fuse = nn.ModuleList([CAFIM(w[i]) for i in range(3)])
        self.ds = nn.ModuleList([cna(w[i], w[i + 1], 3, stride=2) for i in range(2)])
        self.dd = nn.ModuleList([cna(w[i], w[i + 1], 3, stride=2) for i in range(2)])
        self.up = nn.ModuleList([cna(2 * w[2], 2 * w[1]), cna(w[1], 2 * w[0])])
        self.mix = nn.ModuleList([cna(2 * w[1] + 2 * w[1], w[1]), cna(2 * w[0] + 2 * w[0], w[0])])
        self.out_c = w[0]

    def forward(self, x):
        fs, fd = self.stem_s(x[:, STA_IDX]), self.stem_d(x[:, DYN_IDX])
        skips = []
        for i in range(3):
            fs, fd = self.es[i](fs), self.ed[i](fd)
            f = self.fuse[i](fs, fd)
            if i < 2:
                skips.append(f)
                fs, fd = self.ds[i](fs), self.dd[i](fd)
        h = F.interpolate(self.up[0](f), size=skips[1].shape[-2:], mode="bilinear", align_corners=False)
        h = self.mix[0](torch.cat([h, skips[1]], 1))
        h = F.interpolate(self.up[1](h), size=skips[0].shape[-2:], mode="bilinear", align_corners=False)
        return self.mix[1](torch.cat([h, skips[0]], 1))


"""Orthogonal-transform trunks (TD-FusionUNet, ICIP 2026).

Idea borrowed from Luo et al.: a U-Net whose blocks mix globally through a fixed
orthogonal transform (Walsh-Hadamard / DCT) with a learned per-coefficient gain
and soft-threshold. That gives a 64x64 patch long-range mixing for the price of
two small matmuls per channel, instead of attention or a deep dilated stack.
"""

_BASIS: dict[tuple[str, int], torch.Tensor] = {}


def basis(kind: str, n: int) -> torch.Tensor:
    """Orthonormal n x n transform matrix (Walsh-Hadamard or DCT-II)."""
    key = (kind, n)
    if key not in _BASIS:
        if kind == "ht":
            h = np.ones((1, 1), np.float32)
            while h.shape[0] < n:
                h = np.block([[h, h], [h, -h]])
            h /= np.sqrt(n)
        else:
            k = np.arange(n)[:, None]
            i = np.arange(n)[None, :]
            h = np.cos(np.pi * (i + 0.5) * k / n) * np.sqrt(2.0 / n)
            h[0] = np.sqrt(1.0 / n)
        _BASIS[key] = torch.tensor(h, dtype=torch.float32)
    return _BASIS[key]


class Spectral(nn.Module):
    """Transform, per-coefficient gain, soft-threshold, inverse transform."""

    def __init__(self, c: int, size: int, kind: str = "ht"):
        super().__init__()
        self.register_buffer("B", basis(kind, size))
        self.gain = nn.Parameter(torch.ones(1, c, size, size))
        self.thr = nn.Parameter(torch.full((1, c, size, size), -5.0))  # softplus(-5) ~ 0.007

    def forward(self, x):
        z = self.B @ x @ self.B.t()
        z = z * self.gain
        z = torch.sign(z) * F.relu(z.abs() - F.softplus(self.thr))
        return self.B.t() @ z @ self.B


class SpecBlock(nn.Module):
    """Conv block + spectral perceptron, residual. `conv` kind = plain control."""

    def __init__(self, c: int, size: int, kind: str):
        super().__init__()
        self.body = nn.Sequential(cna(c, c), cna(c, c))
        self.spec = None if kind == "conv" else Spectral(c, size, kind)

    def forward(self, x):
        y = self.body(x)
        return x + (y if self.spec is None else self.spec(y))


class BranchEnc(nn.Module):
    def __init__(self, cin: int, width: int, depth: int, kind: str, sizes):
        super().__init__()
        self.stem = cna(cin, width)
        ws = [width * 2 ** i for i in range(len(sizes))]
        self.stages = nn.ModuleList([
            nn.Sequential(*[SpecBlock(ws[i], sizes[i], kind) for _ in range(depth)]) for i in range(len(sizes))])
        self.down = nn.ModuleList([cna(ws[i], ws[i + 1], 3, stride=2) for i in range(len(sizes) - 1)])

    def forward(self, x):
        h, out = self.stem(x), []
        for i, st in enumerate(self.stages):
            h = st(h)
            out.append(h)
            if i < len(self.stages) - 1:
                h = self.down[i](h)
        return out


class Fuse(nn.Module):
    """Merge k branch feature maps: per-branch scalar sum, or a learned pixel gate."""

    def __init__(self, c: int, k: int, mode: str = "gate"):
        super().__init__()
        self.mode = mode
        if mode == "sum":
            self.w = nn.Parameter(torch.ones(k))
        else:
            self.proj = nn.ModuleList([nn.Conv2d(c, c, 1) for _ in range(k)])
            self.g = nn.Sequential(nn.Conv2d(c * k, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, k, 1))

    def forward(self, fs):
        if self.mode == "sum":
            return sum(w * f for w, f in zip(self.w, fs))
        z = torch.stack([p(f) for p, f in zip(self.proj, fs)], 1)
        a = torch.softmax(self.g(torch.cat(fs, 1)), 1).unsqueeze(2)
        return (z * a).sum(1)


class TDUNet(nn.Module):
    """k parallel transform encoders, shared decoder, fused at every scale.

    `dec` widens only the decoder (the trunk stays 1M), `attn` adds SE channel
    attention after each fusion. Both default to the screened winner's shape.
    """

    def __init__(self, cin: int, width: int = 16, branches=("ht", "dct"), depth: int = 1,
                 fusion: str = "gate", sizes=(64, 32, 16), attn: bool = False, dec: int = 1):
        super().__init__()
        w, k, n = width, len(branches), len(sizes)
        ch = [w * 2 ** i for i in range(n)]
        self.enc = nn.ModuleList([BranchEnc(cin, w, depth, b, sizes) for b in branches])
        fuses = []
        for c in ch:
            f = Fuse(c, k, fusion)
            fuses.append(nn.Sequential(f, SE(c)) if attn else f)
        self.fuse = nn.ModuleList(fuses)
        self.up = nn.ModuleList([cna(ch[i + 1] * (dec if i < n - 2 else 1), ch[i] * dec)
                                 for i in range(n - 1)])
        self.mix = nn.ModuleList([cna(ch[i] * dec + ch[i], ch[i] * dec) for i in range(n - 1)])
        self.out_c = w * dec

    def forward(self, x):
        fs = [e(x) for e in self.enc]
        h = self.fuse[-1]([f[-1] for f in fs])
        for i in range(len(fs[0]) - 2, -1, -1):
            h = self.up[i](h)
            h = F.interpolate(h, size=fs[0][i].shape[-2:], mode="bilinear", align_corners=False)
            h = self.mix[i](torch.cat([h, self.fuse[i]([f[i] for f in fs])], 1))
        return h


class Head(nn.Module):
    def __init__(self, c: int, out: int = 1):
        super().__init__()
        self.f = cna(c, max(16, c // 2))
        self.o = nn.Conv2d(max(16, c // 2), out, 1)

    def forward(self, x):
        return self.o(self.f(x))


class Net(nn.Module):
    def __init__(self, cfg: dict, mean: list[float], std: list[float]):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        cfg = dict(cfg)
        trunk = cfg.pop("trunk", "fullres")
        heads = cfg.pop("heads", ["main"])
        cin = cfg.pop("cin", 23)
        self.trunk_name = trunk
        if trunk == "fullres":
            self.trunk = FullRes(cin, cfg.pop("width", 96), cfg.pop("blocks", [[1, 2], [2, 2], [4, 2], [8, 2], [16, 2]]),
                                 cfg.pop("block", "res"), cfg.pop("fusion", "concat"), cfg.pop("drop_path", 0.0))
        elif trunk == "unet":
            self.trunk = UNet(cin, cfg.pop("widths", [48, 96, 192]), cfg.pop("depths", [2, 2, 2]),
                              cfg.pop("block", "res"), cfg.pop("drop_path", 0.0), cfg.pop("gated_skip", True))
        elif trunk == "hrnet":
            self.trunk = HRNet(cin, cfg.pop("width", 32), cfg.pop("stages", 3), cfg.pop("block", "res"),
                               cfg.pop("drop_path", 0.0))
        elif trunk == "dual":
            self.trunk = DualUNet(cin, cfg.pop("widths", (32, 64, 128)), cfg.pop("depths", (2, 2, 2)),
                                  cfg.pop("drop_path", 0.0))
        elif trunk == "td":
            self.trunk = TDUNet(cin, cfg.pop("width", 16), tuple(cfg.pop("branches", ("ht", "dct"))),
                                cfg.pop("depth", 1), cfg.pop("fusion", "gate"),
                                tuple(cfg.pop("sizes", (64, 32, 16))), cfg.pop("attn", False),
                                cfg.pop("dec", 1))
        else:
            raise ValueError(f"unknown trunk {trunk}")
        wide = cfg.pop("area", 0)
        if cfg:
            raise ValueError(f"unused model keys: {sorted(cfg)}")
        c = self.trunk.out_c
        self.heads = nn.ModuleDict({h: Head(c, 4 if h == "dist" else 1) for h in heads})
        # Per-patch fire size, regressed from the decoder's global statistics. The
        # mask's own probability mass is a poor size estimate (corr 0.56 with the
        # GT area), while the area decides ~0.09 Dice at fixed ranking.
        self.area_head = None
        if wide:
            self.area_head = nn.Sequential(nn.Linear(2 * c, wide), nn.GELU(), nn.Linear(wide, 1))
            # Start exactly at the mean GT area: a random last layer can land far
            # from it, which puts the joint training in a useless basin.
            nn.init.zeros_(self.area_head[-1].weight)
            nn.init.constant_(self.area_head[-1].bias, 2.9)  # ~ln 17 px

    def forward(self, x):
        x = (x - self.mean) / self.std
        f = self.trunk(x)
        out = {h: m(f) for h, m in self.heads.items()}
        if self.area_head is not None:
            out["area"] = self.area_head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)).squeeze(1)
        return out


class SMPNet(nn.Module):
    """segmentation_models_pytorch wrapper with the same normalisation + dict output."""

    def __init__(self, arch: str, encoder: str, weights: str | None, mean: list[float], std: list[float],
                 cin: int = 23, heads=("main",), up: int = 1, area: int = 0):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.register_buffer("mean", torch.tensor(mean).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, -1, 1, 1))
        cls = getattr(smp, arch)
        self.model = cls(encoder_name=encoder, encoder_weights=weights, in_channels=cin, classes=len(heads))
        self.head_names = list(heads)
        # Tiny targets in a 64x64 patch: run the encoder on a finer grid, then
        # bring the logits back so the loss and metrics stay at 64x64.
        self.up = int(up)
        self.area_head = self._feat = None
        if area:
            # The decoder's full-res features are the segmentation head's input;
            # grab them there so this works for every smp architecture.
            conv = next(m for m in self.model.segmentation_head.modules() if isinstance(m, nn.Conv2d))
            self.model.segmentation_head.register_forward_pre_hook(
                lambda _m, a: setattr(self, "_feat", a[0]))
            self.area_head = nn.Sequential(nn.Linear(2 * conv.in_channels, area), nn.GELU(),
                                           nn.Linear(area, 1))
            nn.init.zeros_(self.area_head[-1].weight)
            nn.init.constant_(self.area_head[-1].bias, 2.9)

    def forward(self, x):
        x = (x - self.mean) / self.std
        size = x.shape[-2:]
        if self.up > 1:
            x = F.interpolate(x, size=(size[0] * self.up, size[1] * self.up),
                              mode="bilinear", align_corners=False)
        o = self.model(x)
        if self.up > 1:
            o = F.interpolate(o, size=size, mode="bilinear", align_corners=False)
        out = {h: o[:, i:i + 1] for i, h in enumerate(self.head_names)}
        if self.area_head is not None:
            f = self._feat
            out["area"] = self.area_head(torch.cat([f.mean((2, 3)), f.amax((2, 3))], 1)).squeeze(1)
        return out


def build_model(mcfg: dict, mean, std) -> nn.Module:
    mcfg = dict(mcfg)
    if "smp" in mcfg:
        smp = mcfg.pop("smp")
        heads = mcfg.pop("heads", ["main"])
        up = mcfg.pop("up", 1)
        return SMPNet(smp["arch"], smp["encoder"], smp.get("weights", "imagenet"), mean, std,
                      cin=mcfg.pop("cin", 23), heads=heads, up=up, area=mcfg.pop("area", 0))
    return Net(mcfg, mean, std)

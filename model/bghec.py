# -*- coding: utf-8 -*-
"""
BGHEC: Bidirectional Gated Hierarchical Evidence Communication.

Native-resolution cross-level feature propagation without explicit Query/Key/Value.
Top-down: deep semantic context flows downward to guide shallow detail interpretation.
Bottom-up: shallow target details flow upward to compensate deep semantic degradation.
Spatial dynamic gates control injection strength at every spatial position.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  base layers
# ---------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for [B, C, H, W] tensors."""
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
#  BGHEC sub-modules
# ---------------------------------------------------------------------------

class DWConvRefine(nn.Module):
    """Lightweight local alignment: 3x3 depthwise + 1x1 pointwise."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class SpatialDynamicGate(nn.Module):
    """Produces a single-channel spatial gate ∈ (0,1) from [target, source, |target-source|].

    Initial bias = -1.0 → σ(-1) ≈ 0.27, so cross-level injection starts conservatively.
    """
    def __init__(self, dim, hidden=16, bias_init=-1.0):
        super().__init__()
        hidden = max(int(hidden), 8)

        self.reduce = nn.Sequential(
            nn.Conv2d(dim * 3, hidden, 1, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
        )
        self.local = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
        )
        self.out = nn.Conv2d(hidden, 1, 1, bias=True)

        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, float(bias_init))

    def forward(self, target, source):
        cue = torch.cat([target, source, torch.abs(target - source)], dim=1)
        return torch.sigmoid(self.out(self.local(self.reduce(cue))))


class GatedPropagationFuse(nn.Module):
    """Fuses current level with gated source.  Final 1x1 has NO activation
    so the output is a signed delta (can both enhance and suppress)."""
    def __init__(self, dim):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.dw = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.out = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, target, source, gate):
        return self.out(self.dw(self.pre(torch.cat([target, gate * source], dim=1))))


class DetailPreservingDownsample(nn.Module):
    """PixelUnshuffle(2) preserves every sub-pixel, unlike AvgPool which
    discards 3/4 of spatial information — critical for tiny 1-2 px targets."""
    def __init__(self, dim):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.proj = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )

    def forward(self, x):
        return self.proj(self.unshuffle(x))


# ---------------------------------------------------------------------------
#  BGHEC Block  and  full fusion module
# ---------------------------------------------------------------------------

class BGHECBlock(nn.Module):
    """One bidirectional propagation round: Top-down → Bottom-up.

    All features stay at their native resolutions throughout the block.
    """
    def __init__(self, dim=32, gamma_init=0.1, gate_hidden=16, gate_bias_init=-1.0):
        super().__init__()
        num_scales = 4

        # top-down  (deep → shallow)
        self.td_align = nn.ModuleList([DWConvRefine(dim) for _ in range(3)])
        self.td_gate  = nn.ModuleList([SpatialDynamicGate(dim, gate_hidden, gate_bias_init) for _ in range(3)])
        self.td_fuse  = nn.ModuleList([GatedPropagationFuse(dim) for _ in range(3)])

        # bottom-up  (shallow → deep)
        self.downsample = nn.ModuleList([DetailPreservingDownsample(dim) for _ in range(3)])
        self.bu_align   = nn.ModuleList([DWConvRefine(dim) for _ in range(3)])
        self.bu_gate    = nn.ModuleList([SpatialDynamicGate(dim, gate_hidden, gate_bias_init) for _ in range(3)])
        self.bu_fuse    = nn.ModuleList([GatedPropagationFuse(dim) for _ in range(3)])

        self.gamma_td = nn.Parameter(torch.full((3,), float(gamma_init)))
        self.gamma_bu = nn.Parameter(torch.full((3,), float(gamma_init)))

    def forward(self, features, return_logs=False):
        x = list(features)
        td = [None] * 4

        # ---- top-down ----
        td[3] = x[3]
        for i in (2, 1, 0):
            src = F.interpolate(td[i + 1], size=x[i].shape[-2:],
                                mode="bilinear", align_corners=False)
            src = self.td_align[i](src)
            g = self.td_gate[i](x[i], src)
            td[i] = x[i] + self.gamma_td[i] * self.td_fuse[i](x[i], src, g)

        # ---- bottom-up ----
        bu = [None] * 4
        bu[0] = td[0]
        for i in (1, 2, 3):
            src = self.downsample[i - 1](bu[i - 1])
            if src.shape[-2:] != td[i].shape[-2:]:
                src = F.interpolate(src, size=td[i].shape[-2:],
                                    mode="bilinear", align_corners=False)
            src = self.bu_align[i - 1](src)
            g = self.bu_gate[i - 1](td[i], src)
            bu[i] = td[i] + self.gamma_bu[i - 1] * self.bu_fuse[i - 1](td[i], src, g)

        logs = {}
        if return_logs:
            logs = {
                "gamma_td": self.gamma_td.detach(),
                "gamma_bu": self.gamma_bu.detach(),
            }
        return bu, logs


class BGHECFusion(nn.Module):
    """Complete BGHEC fusion module.

    Input : four native-resolution encoder features  E0–E3
            [B,  32, 256, 256], [B,  64, 128, 128],
            [B, 128,  64,  64], [B, 256,  32,  32]

    Output: four cross-level deltas  Δ0–Δ3  (same shapes as the encoder features
            after channel projection). The main network does a single residual:
                G_i = E_i + Δ_i
    """
    def __init__(self, channel_num=(32, 64, 128, 256), common_dim=32,
                 depth=1, gamma_init=0.1, gate_hidden=16,
                 gate_bias_init=-1.0, vis=False):
        super().__init__()
        if len(channel_num) != 4:
            raise ValueError("BGHEC expects four encoder levels")

        self.channel_num = tuple(channel_num)
        self.common_dim = common_dim
        self.vis = vis

        #  1×1 channel projection  C_i → D  (no spatial change)
        self.in_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, common_dim, 1, bias=False),
                LayerNorm2d(common_dim), nn.GELU(),
            ) for c in self.channel_num
        ])

        self.blocks = nn.ModuleList([
            BGHECBlock(dim=common_dim, gamma_init=gamma_init,
                       gate_hidden=gate_hidden, gate_bias_init=gate_bias_init)
            for _ in range(depth)
        ])

        #  D → C_i  for delta extraction
        self.out_proj = nn.ModuleList([
            nn.Conv2d(common_dim, c, 1, bias=False) for c in self.channel_num
        ])
        self.gamma_out = nn.Parameter(torch.full((4,), float(gamma_init)))

        self.gate_logs = []

    def forward(self, e0, e1, e2, e3):
        native = [e0, e1, e2, e3]
        initial = [proj(feat) for proj, feat in zip(self.in_proj, native)]

        features = initial
        self.gate_logs = []
        for block in self.blocks:
            features, logs = block(features, return_logs=self.vis)
            if self.vis:
                self.gate_logs.append(logs)

        #  delta = γ_out × Conv(features_after − features_before)
        deltas = []
        for i in range(4):
            delta = self.gamma_out[i] * self.out_proj[i](features[i] - initial[i])
            deltas.append(delta)

        return deltas[0], deltas[1], deltas[2], deltas[3], self.gate_logs


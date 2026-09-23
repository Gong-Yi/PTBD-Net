# -*- coding: utf-8 -*-
"""
PTBD-Net: Progressive Target-Background Discrimination Network.

Three-stage progressive pipeline:
  1.  DHiF Encoder   — local high-frequency structure discrimination
  2.  BGHEC Fusion    — bidirectional gated hierarchical evidence communication
  3.  RAHMoE Decoder  — response-aware hierarchical mixture of experts

Architecture overview
---------------------
Input [B, 1, 256, 256]
    │
    ▼  DHiF Encoder (L1–L3) + Standard ResBlock (L0, L4)
    │
    ├── E0 [B,  32, 256, 256]  ──┐
    ├── E1 [B,  64, 128, 128]  ──┤
    ├── E2 [B, 128,  64,  64]  ──┤  BGHEC
    ├── E3 [B, 256,  32,  32]  ──┘  ──→  Δ0–Δ3
    │
    │   G_i = E_i + Δ_i   (single residual)
    │
    ▼  CNN Decoder  +  RAHMoE @ d3, d2
    │
    ▼  Deep Supervision (6 outputs in training)
Output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dhif import DHiF
from .bghec import BGHECFusion
from .rahmoe import RAHMoE


# ============================================================================
#  basic CNN blocks
# ============================================================================

class ResBlock(nn.Module):
    """Standard residual block: Conv3x3 → BN → LReLU → Conv3x3 → BN → +residual."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class DHiFResBlock(nn.Module):
    """Residual block whose first 3×3 convolution is replaced by DHiF."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = DHiF(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.act   = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_ch)

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


# ============================================================================
#  decoder components
# ============================================================================

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelCrossAttention(nn.Module):
    """Channel cross-attention for skip-connection gating in the decoder."""
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(Flatten(), nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(Flatten(), nn.Linear(F_g, F_x))
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, g, x):
        avg_x = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        avg_g = F.avg_pool2d(g, (g.size(2), g.size(3)), stride=(g.size(2), g.size(3)))
        scale = torch.sigmoid((self.mlp_x(avg_x) + self.mlp_g(avg_g)) / 2.0)
        scale = scale.unsqueeze(2).unsqueeze(3).expand_as(x)
        return self.relu(x * scale)


def _make_conv_block(in_ch, out_ch, n, activation='ReLU'):
    act = getattr(nn, activation)() if hasattr(nn, activation) else nn.ReLU()
    layers = []
    for i in range(n):
        layers.extend([
            nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), act,
        ])
    return nn.Sequential(*layers)


class UpBlock(nn.Module):
    """Decoder upsample block:  Upsample(×2)  →  CCA gating  →  Concat  →  Conv×2."""
    def __init__(self, in_ch, out_ch, n_conv=2, activation='ReLU'):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2)
        self.coatt = ChannelCrossAttention(F_g=in_ch // 2, F_x=in_ch // 2)
        self.convs = _make_conv_block(in_ch, out_ch, n_conv, activation)

    def forward(self, x, skip):
        up = self.up(x)
        skip_att = self.coatt(g=up, x=skip)
        return self.convs(torch.cat([skip_att, up], dim=1))


# ============================================================================
#  PTBD-Net
# ============================================================================

class PTBDNet(nn.Module):
    """Progressive Target-Background Discrimination Network.

    Parameters
    ----------
    config :  ml_collections.ConfigDict from config.get_config()
    mode   :  'train'  or  'test'
    use_dhif :  enable DHiF in encoder levels 1–3
    """

    def __init__(self, config, n_channels=1, n_classes=1, img_size=256,
                 mode='train', deepsuper=True, use_dhif=True,
                 use_bghec=True, use_rahmoe=True):
        super().__init__()
        self.mode       = mode
        self.deepsuper  = deepsuper
        self.n_channels = n_channels
        self.n_classes  = n_classes
        self.use_bghec  = use_bghec
        self.use_rahmoe = use_rahmoe
        in_ch = config.base_channel                                   # 32

        print(f"PTBD-Net  |  DHiF: {use_dhif}  |  BGHEC: {use_bghec}  |  RAHMoE: {use_rahmoe}")

        # =================  Encoder  =================
        self.pool = nn.MaxPool2d(2, 2)
        encoder_blk = DHiFResBlock if use_dhif else ResBlock

        self.inc = self._make_layer(ResBlock, n_channels, in_ch)                          # L0
        self.enc1 = self._make_layer(encoder_blk, in_ch,      in_ch * 2, 1)               # L1  DHiF
        self.enc2 = self._make_layer(encoder_blk, in_ch * 2,  in_ch * 4, 1)               # L2  DHiF
        self.enc3 = self._make_layer(encoder_blk, in_ch * 4,  in_ch * 8, 1)               # L3  DHiF
        self.enc4 = self._make_layer(ResBlock,    in_ch * 8,  in_ch * 8, 1)               # L4  standard

        # =================  BGHEC  =================
        if use_bghec:
            cfg = config.bghec
            self.bghec = BGHECFusion(
                channel_num=[in_ch, in_ch * 2, in_ch * 4, in_ch * 8],
                common_dim=cfg.common_dim, depth=cfg.depth,
                gamma_init=cfg.gamma_init,
                gate_hidden=max(cfg.common_dim // 2, 16),
                gate_bias_init=cfg.gate_bias_init, vis=False,
            )
        else:
            self.bghec = None

        # =================  Decoder  =================
        self.up4 = UpBlock(in_ch * 16, in_ch * 4, 2)          # 256+256 → 128
        self.up3 = UpBlock(in_ch * 8,  in_ch * 2, 2)          # 128+128 →  64
        self.up2 = UpBlock(in_ch * 4,  in_ch,     2)          #  64+ 64 →  32
        self.up1 = UpBlock(in_ch * 2,  in_ch,     2)          #  32+ 32 →  32
        self.outc = nn.Conv2d(in_ch, n_classes, 1, 1)

        # =================  RAHMoE @ d3, d2  =================
        if use_rahmoe:
            cfg_moe = config.rahmoe
            self.rahmoe3 = RAHMoE(
                dim=in_ch * 2, num_experts=cfg_moe.num_experts,
                temperature=cfg_moe.temperature,
                gamma_shared_init=cfg_moe.gamma_shared,
                gamma_routed_init=cfg_moe.gamma_routed,
                balance_weight=cfg_moe.balance_weight,
                diversity_weight=cfg_moe.diversity_weight,
                entropy_weight=cfg_moe.entropy_weight,
                zloss_weight=cfg_moe.zloss_weight,
                entropy_target=cfg_moe.entropy_target,
            )
            self.rahmoe2 = RAHMoE(
                dim=in_ch, num_experts=cfg_moe.num_experts,
                temperature=cfg_moe.temperature,
                gamma_shared_init=cfg_moe.gamma_shared,
                gamma_routed_init=cfg_moe.gamma_routed,
                balance_weight=cfg_moe.balance_weight,
                diversity_weight=cfg_moe.diversity_weight,
                entropy_weight=cfg_moe.entropy_weight,
                zloss_weight=cfg_moe.zloss_weight,
                entropy_target=cfg_moe.entropy_target,
            )
        else:
            self.rahmoe3 = None
            self.rahmoe2 = None

        # =================  Deep Supervision  =================
        if self.deepsuper:
            self.gt5 = nn.Conv2d(in_ch * 8, 1, 1)               # from bottleneck
            self.gt4 = nn.Conv2d(in_ch * 4, 1, 1)               # from d4
            self.gt3 = nn.Conv2d(in_ch * 2, 1, 1)               # from d3
            self.gt2 = nn.Conv2d(in_ch,     1, 1)               # from d2
            self.fuse_conv = nn.Conv2d(5, 1, 1)                 # fuse 5 predictions

        self.moe_aux_loss = None
        self.gate_logs    = []

    def _make_layer(self, block, in_ch, out_ch, n=1):
        layers = [block(in_ch, out_ch)]
        for _ in range(n - 1):
            layers.append(block(out_ch, out_ch))
        return nn.Sequential(*layers)

    def forward(self, x, return_moe_vis=False):
        # =================  Encoder  =================
        e0 = self.inc(x)                                    # [B,  32, 256, 256]
        e1 = self.enc1(self.pool(e0))                       # [B,  64, 128, 128]
        e2 = self.enc2(self.pool(e1))                       # [B, 128,  64,  64]
        e3 = self.enc3(self.pool(e2))                       # [B, 256,  32,  32]
        b  = self.enc4(self.pool(e3))                       # [B, 256,  16,  16]  bottleneck

        # =================  BGHEC  =================
        if self.use_bghec:
            d0, d1, d2_c, d3_c, _ = self.bghec(e0, e1, e2, e3)
            g0 = e0 + d0
            g1 = e1 + d1
            g2 = e2 + d2_c
            g3 = e3 + d3_c
        else:
            g0, g1, g2, g3 = e0, e1, e2, e3

        # =================  Decoder + RAHMoE  =================
        d4 = self.up4(b, g3)                                # [B, 128,  32,  32]
        d3 = self.up3(d4, g2)                               # [B,  64,  64,  64]
        vis_d3 = None; vis_d2 = None
        if self.use_rahmoe:
            if return_moe_vis: d3, vis_d3 = self.rahmoe3(d3, return_vis=True)
            else: d3 = self.rahmoe3(d3)

        d2 = self.up2(d3, g1)
        if self.use_rahmoe:
            if return_moe_vis: d2, vis_d2 = self.rahmoe2(d2, return_vis=True)
            else: d2 = self.rahmoe2(d2)

        d1 = self.up1(d2, g0)                               # [B,  32, 256, 256]
        out = self.outc(d1)                                 # [B,   1, 256, 256]

        #  collect vis data before aux loss
        moe_vis = {"d3": vis_d3, "d2": vis_d2} if return_moe_vis else None

        # =================  Aux loss  =================
        if self.use_rahmoe:
            self.moe_aux_loss = 0.5 * (self.rahmoe3.aux_loss + self.rahmoe2.aux_loss)
        else:
            self.moe_aux_loss = out.new_zeros(())

        # =================  Deep Supervision  =================
        if self.deepsuper:
            gt_5 = self.gt5(b)
            gt_4 = self.gt4(d4)
            gt_3 = self.gt3(d3)
            gt_2 = self.gt2(d2)

            gt5 = F.interpolate(gt_5, scale_factor=16, mode='bilinear', align_corners=True)
            gt4 = F.interpolate(gt_4, scale_factor=8,  mode='bilinear', align_corners=True)
            gt3 = F.interpolate(gt_3, scale_factor=4,  mode='bilinear', align_corners=True)
            gt2 = F.interpolate(gt_2, scale_factor=2,  mode='bilinear', align_corners=True)
            d0  = self.fuse_conv(torch.cat((gt2, gt3, gt4, gt5, out), 1))

            if self.mode == 'train':
                out_tuple = (torch.sigmoid(gt5), torch.sigmoid(gt4), torch.sigmoid(gt3),
                             torch.sigmoid(gt2), torch.sigmoid(d0),  torch.sigmoid(out))
            else:
                out_tuple = torch.sigmoid(out)
        else:
            out_tuple = torch.sigmoid(out)

        if return_moe_vis:
            return out_tuple, moe_vis
        return out_tuple

    def get_moe_aux_loss(self):
        return self.moe_aux_loss

    def get_gate_logs(self):
        return self.gate_logs

    @torch.no_grad()
    def set_router_temperature(self, value):
        self.rahmoe3.router.set_temperature(value)
        self.rahmoe2.router.set_temperature(value)


# ============================================================================
#  self-test
# ============================================================================

if __name__ == '__main__':
    from .config import get_config
    config = get_config()
    model = PTBDNet(config, mode='train', deepsuper=True, use_dhif=True)

    x = torch.randn(1, 1, 256, 256)
    out = model(x)
    print(f"\nOutput: {type(out).__name__}", end='')
    if isinstance(out, tuple):
        print(f" of {len(out)} tensors")
    else:
        print()
    print(f"Params:  {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")
    print(f"MoE aux: {model.get_moe_aux_loss().item():.6f}")

    loss = sum(o.mean() for o in (out if isinstance(out, tuple) else [out]))
    loss = loss + model.get_moe_aux_loss()
    loss.backward()

    no_grad = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    print(f"No-grad: {len(no_grad)} params" + (" ✓" if len(no_grad) == 0 else f" ⚠ {no_grad[:3]}"))
    print("PTBD-Net self-test passed.\n")


# -*- coding: utf-8 -*-
"""
RAHMoE: Response-Aware Hierarchical Mixture of Experts.

Deployed after decoder fusion stages (d3 and d2), this module applies
per-pixel dense soft routing over 1 shared expert and 4 response-state
experts to handle heterogeneous local response states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=eps)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class ECA2d(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        y = self.pool(x).flatten(2).transpose(1, 2)
        y = torch.sigmoid(self.conv(y))
        return x * y.transpose(1, 2).unsqueeze(-1)


# ---------------------------------------------------------------------------
#  Experts
# ---------------------------------------------------------------------------

class SharedDecoderExpert(nn.Module):
    """Always-active shared expert for stable general decoding."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=False), LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class LowResponseTargetExpert(nn.Module):
    """Enhances weak targets via local contrast gating.

    x − AvgPool7(x) highlights local protrusions even when absolute
    response is low, preventing dim targets from being ignored.
    """
    def __init__(self, dim):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.contrast_gate = nn.Sequential(
            nn.Conv2d(1, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        response = x.abs().mean(dim=1, keepdim=True)
        contrast = response - F.avg_pool2d(response, 7, 1, 3)
        return self.local(x) * (1.0 + self.contrast_gate(contrast))


class CompactTargetRefineExpert(nn.Module):
    """Refines compact candidate responses with dual 3×3 / 5×5 receptive fields."""
    def __init__(self, dim):
        super().__init__()
        self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.dw5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.eca = ECA2d(dim)

    def forward(self, x):
        return self.eca(self.fuse(torch.cat([self.dw3(x), self.dw5(x)], dim=1)))


class HighResponseAmbiguityExpert(nn.Module):
    """Contextual recalibration for strong-but-ambiguous responses.

    Large-kernel (7×7) and dilated (3×3, d=2) context are fused to
    decide whether a high response should keep its own feature or
    rely more on surrounding context.
    """
    def __init__(self, dim):
        super().__init__()
        self.large = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.dilated = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, groups=dim, bias=False),
            LayerNorm2d(dim), nn.GELU(),
            nn.Conv2d(dim, dim, 1, bias=False),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        c1, c2 = self.large(x), self.dilated(x)
        g = self.gate(torch.cat([c1, c2], dim=1))
        return g * x + (1.0 - g) * 0.5 * (c1 + c2)


class StructuredClutterExpert(nn.Module):
    """Suppresses directional structural clutter via 1×7 / 7×1 strip convolutions.

    Cloud edges, horizons, and building contours typically exhibit
    strong directionality that this expert can recognise.
    """
    def __init__(self, dim):
        super().__init__()
        self.h_conv = nn.Conv2d(dim, dim, (1, 7), padding=(0, 3),
                                groups=dim, bias=False)
        self.v_conv = nn.Conv2d(dim, dim, (7, 1), padding=(3, 0),
                                groups=dim, bias=False)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            LayerNorm2d(dim), nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=True), nn.Sigmoid(),
        )

    def forward(self, x):
        s = self.fuse(torch.cat([self.h_conv(x), self.v_conv(x)], dim=1))
        return s * self.gate(s)


# ---------------------------------------------------------------------------
#  Router
# ---------------------------------------------------------------------------

class DenseResponseStateRouter(nn.Module):
    """Per-pixel dense soft router.

    Constructs 6 explicit response-state cues (mean response, local contrast,
    multi-scale context, directional structure) and fuses them with a
    compressed feature descriptor to produce per-expert gates via Softmax.
    """
    NUM_CUES = 6

    def __init__(self, dim, num_experts=4, hidden=32, temperature=1.5):
        super().__init__()
        self.num_experts = num_experts

        self.feat_reduce = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
        )
        self.router = nn.Sequential(
            nn.Conv2d(hidden + self.NUM_CUES, hidden, 3, padding=1, bias=False),
            LayerNorm2d(hidden), nn.GELU(),
            nn.Conv2d(hidden, num_experts, 1, bias=True),
        )
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    @torch.no_grad()
    def set_temperature(self, value):
        self.temperature.fill_(max(0.3, float(value)))

    def build_cues(self, x):
        feat = self.feat_reduce(x)

        r = x.abs().mean(dim=1, keepdim=True)                          # mean response
        c = r - F.avg_pool2d(r, 7, 1, 3)                               # local contrast
        s3 = F.avg_pool2d(r, 3, 1, 1)                                  # 3×3 context
        s5 = F.avg_pool2d(r, 5, 1, 2)                                  # 5×5 context
        s7 = F.avg_pool2d(r, 7, 1, 3)                                  # 7×7 context
        gh = torch.abs(r - F.avg_pool2d(r, (1, 7), 1, (0, 3)))        # horizontal structure
        gv = torch.abs(r - F.avg_pool2d(r, (7, 1), 1, (3, 0)))        # vertical structure

        return torch.cat([feat, r, c, s3, s5, s7, gh + gv], dim=1)

    def forward(self, x):
        prompt = self.build_cues(x)
        logits = self.router(prompt)
        gates = torch.softmax(logits / self.temperature.clamp_min(0.3), dim=1)
        return gates, logits


# ---------------------------------------------------------------------------
#  RAHMoE wrapper
# ---------------------------------------------------------------------------

class RAHMoE(nn.Module):
    """Response-Aware Hierarchical Mixture of Experts.

    Deployed at decoder stages d3 (64×64) and d2 (128×128).
    Output:  F_out = F + γ_shared · Shared(F) + γ_routed · Σ_e π_e ⊙ Expert_e(F)

    where π_e ∈ [0, 1]^{B×1×H×W} are per-pixel soft router weights
    (dense routing — all experts always participate).
    """
    def __init__(self, dim, num_experts=4, router_hidden=32, temperature=1.5,
                 gamma_shared_init=0.1, gamma_routed_init=0.1,
                 balance_weight=1e-3, diversity_weight=1e-4,
                 entropy_weight=1e-4, zloss_weight=1e-5,
                 entropy_target=1.05):
        super().__init__()
        self.num_experts = num_experts

        self.shared = SharedDecoderExpert(dim)
        self.experts = nn.ModuleList([
            LowResponseTargetExpert(dim),
            CompactTargetRefineExpert(dim),
            HighResponseAmbiguityExpert(dim),
            StructuredClutterExpert(dim),
        ])
        self.router = DenseResponseStateRouter(
            dim=dim, num_experts=num_experts,
            hidden=router_hidden, temperature=temperature,
        )

        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.gamma_shared = nn.Parameter(torch.tensor(float(gamma_shared_init)))
        self.gamma_routed = nn.Parameter(torch.tensor(float(gamma_routed_init)))

        self.balance_weight   = balance_weight
        self.diversity_weight = diversity_weight
        self.entropy_weight   = entropy_weight
        self.zloss_weight     = zloss_weight
        self.entropy_target   = entropy_target

        self.aux_loss  = None
        self.route_logs = {}

    def forward(self, x, return_vis=False):
        shared = self.shared(x)

        gates, logits = self.router(x)                                  # [B, E, H, W]
        expert_outputs = [e(x) for e in self.experts]
        expert_stack = torch.stack(expert_outputs, dim=1)               # [B, E, C, H, W]

        weighted = gates[:, :, None] * expert_stack
        routed = weighted.sum(dim=1)
        routed = self.out_proj(routed)

        out = x + self.gamma_shared * shared + self.gamma_routed * routed

        self.aux_loss = self._compute_aux(gates, logits, expert_stack)
        self.route_logs = {
            "gates": gates.detach(),
            "expert_norms": expert_stack.detach().pow(2).mean(dim=(2, 3, 4)).sqrt(),
            "gamma_shared": self.gamma_shared.detach(),
            "gamma_routed": self.gamma_routed.detach(),
        }

        if not return_vis:
            return out

        vis = {
            "input_feature": x.detach(),
            "response_map": x.detach().abs().mean(dim=1, keepdim=True),
            "router_logits": logits.detach(),
            "route_weights": gates.detach(),
            "expert_outputs": expert_stack.detach(),
            "weighted_outputs": weighted.detach(),
            "shared_output": shared.detach(),
            "routed_sum": weighted.sum(dim=1).detach(),
            "routed_projection": routed.detach(),
            "final_output": out.detach(),
        }
        return out, vis

    def _compute_aux(self, gates, logits, experts):
        E = self.num_experts

        #  balance:  prevent collapse to a single expert
        usage = gates.mean(dim=(0, 2, 3))
        balance = (usage - torch.full_like(usage, 1.0 / E)).pow(2).sum()

        #  entropy target:  prevent gates from becoming too uniform or too peaked
        entropy = -(gates * torch.log(gates + 1e-8)).sum(dim=1).mean()
        entropy_loss = (entropy - self.entropy_target).pow(2)

        #  diversity:  encourage experts to produce different outputs
        pooled = F.normalize(experts.mean(dim=(-1, -2)), dim=-1)        # [B, E, C]
        sim = torch.matmul(pooled, pooled.transpose(-1, -2))            # [B, E, E]
        eye = torch.eye(E, device=sim.device, dtype=torch.bool)
        diversity = sim[:, ~eye].pow(2).mean()

        #  z-loss:  stabilise router logit magnitude
        zloss = torch.logsumexp(logits, dim=1).pow(2).mean()

        return (self.balance_weight * balance
                + self.diversity_weight * diversity
                + self.entropy_weight * entropy_loss
                + self.zloss_weight * zloss)

    @torch.no_grad()
    def set_temperature(self, value):
        self.router.set_temperature(value)


# -*- coding: utf-8 -*-
"""DHiF: Dynamic High-frequency Convolution.

Generates location-specific dynamic local filter banks conditioned on
local input patterns, enabling discriminative representations for
target-like and clutter-like high-frequency components.

Reference: "Dynamic High-frequency Convolution for Infrared Small Target Detection"
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class DHiF(nn.Module):
    """Dynamic High-frequency Convolution for encoder hidden layers."""

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, bias=True):
        super().__init__()
        if groups != 1:
            raise NotImplementedError("DHiF requires groups=1")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        if self.kernel_size[0] != self.kernel_size[1]:
            raise ValueError("DHiF requires a square kernel")

        self.k = self.kernel_size[0]
        self.K = self.k * self.k

        self.operator_generator = nn.Sequential(
            nn.Linear(self.K, self.K * self.K),
            nn.Tanh(),
        )
        self.weight = Parameter(torch.empty(out_channels, in_channels, self.k, self.k))
        self.bias = Parameter(torch.empty(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def _output_hw(self, h, w):
        ho = (h + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] + 1
        wo = (w + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] + 1
        return ho, wo

    def forward(self, x):
        b, c, h, w = x.shape
        ho, wo = self._output_hw(h, w)

        patches = F.unfold(x, kernel_size=self.kernel_size, dilation=self.dilation,
                           padding=self.padding, stride=self.stride)
        patches = patches.view(b, c, self.K, ho, wo).permute(0, 1, 3, 4, 2).contiguous()

        descriptor = torch.linalg.vector_norm(patches, ord=2, dim=1)
        operator = self.operator_generator(descriptor).view(b, ho, wo, self.K, self.K)

        dynamic = torch.einsum("bchwk,bhwkl->bchwl", patches, operator)
        augmented = patches + dynamic

        weight = self.weight.view(self.out_channels, self.in_channels, self.K)
        out = torch.einsum("bchwk,ock->bohw", augmented, weight)
        if self.bias is not None:
            out = out + self.bias[None, :, None, None]
        return out


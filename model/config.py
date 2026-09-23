# -*- coding: utf-8 -*-
"""PTBD-Net configuration."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ---- Encoder ----
    config.base_channel = 32       #  first-layer channels
    config.n_classes = 1           #  binary segmentation

    # ---- BGHEC (Bidirectional Gated Hierarchical Evidence Communication) ----
    config.bghec = ml_collections.ConfigDict()
    config.bghec.common_dim = 32
    config.bghec.depth = 1
    config.bghec.gamma_init = 0.1
    config.bghec.gate_bias_init = -1.0

    # ---- RAHMoE (Response-Aware Hierarchical Mixture of Experts) ----
    config.rahmoe = ml_collections.ConfigDict()
    config.rahmoe.num_experts = 4
    config.rahmoe.temperature = 1.5
    config.rahmoe.gamma_shared = 0.1
    config.rahmoe.gamma_routed = 0.1
    config.rahmoe.balance_weight = 1e-3
    config.rahmoe.diversity_weight = 1e-4
    config.rahmoe.entropy_weight = 1e-4
    config.rahmoe.zloss_weight = 1e-5
    config.rahmoe.entropy_target = 1.05

    return config


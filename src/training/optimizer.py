# src/training/optimizer.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
from torch.optim import AdamW


def build_optimizer(model, learning_rate: float, weight_decay: float,
                    use_8bit: bool = False):
    """
    Build optimizer with weight decay on matrix weights only.
    use_8bit=True uses bitsandbytes 8-bit Adam — saves ~1.2 GB VRAM.
    """
    decay_params   = []
    nodecay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2:
            nodecay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params,   "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    decay_count   = sum(p.numel() for p in decay_params)
    nodecay_count = sum(p.numel() for p in nodecay_params)

    print(f"  Optimizer      : {'AdamW 8-bit' if use_8bit else 'AdamW 32-bit'}")
    print(f"  Decay params   : {decay_count:,}")
    print(f"  No-decay params: {nodecay_count:,}")
    print(f"  Learning rate  : {learning_rate}")
    print(f"  Weight decay   : {weight_decay}")

    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.AdamW8bit(
                groups,
                lr    = learning_rate,
                betas = (0.9, 0.95),
                eps   = 1e-8,
            )
            print(f"  8-bit Adam     : enabled — saves ~1.2 GB VRAM")
            return optimizer
        except Exception as e:
            print(f"  8-bit Adam     : failed ({e}) — falling back to 32-bit")

    return AdamW(
        groups,
        lr    = learning_rate,
        betas = (0.9, 0.95),
        eps   = 1e-8,
    )
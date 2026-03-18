# To be implemented
# src/model/transformer.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
from src.model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.

    Applied independently to every token position.
    Expands to 4x hidden dim then projects back.

    This is where the model stores factual knowledge.
    Attention finds relationships — FFN stores facts.

    Linear(n_embd → 4*n_embd) → GELU → Linear(4*n_embd → n_embd) → Dropout
    """

    def __init__(self, n_embd, dropout=0.1, bias=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One full Transformer decoder block.

    Structure:
        x → LayerNorm → Attention → (+x) → LayerNorm → FFN → (+x) → out
                                     ↑                          ↑
                                  residual                   residual

    Two key ideas:
      1. Pre-LayerNorm  — normalize BEFORE each sub-layer (more stable)
      2. Residual connection — add input back after each sub-layer
                               (gradients flow cleanly through depth)
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1, bias=True):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embd)
        self.ln_2 = nn.LayerNorm(n_embd)

        self.attn = CausalSelfAttention(
            n_embd     = n_embd,
            n_head     = n_head,
            block_size = block_size,
            dropout    = dropout,
            bias       = bias,
        )
        self.ffn = FeedForward(
            n_embd  = n_embd,
            dropout = dropout,
            bias    = bias,
        )

    def forward(self, x):
        # Sub-layer 1 — Attention with residual
        x = x + self.attn(self.ln_1(x))

        # Sub-layer 2 — FFN with residual
        x = x + self.ffn(self.ln_2(x))

        return x


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing TransformerBlock...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    N_EMBD     = 512
    N_HEAD     = 8
    BLOCK_SIZE = 512
    N_LAYERS   = 6

    block = TransformerBlock(
        n_embd=N_EMBD, n_head=N_HEAD, block_size=BLOCK_SIZE, dropout=0.1
    ).to(device)

    x = torch.randn(4, 512, N_EMBD).to(device)
    print(f"  Input  shape : {x.shape}")

    with torch.no_grad():
        out = block(x)

    print(f"  Output shape : {out.shape}")
    assert out.shape == x.shape
    print(f"  Shape check  : ✓ PASSED")

    # Residual check — output norm should be close to input norm
    ratio = out.norm().item() / x.norm().item()
    res_ok = 0.5 < ratio < 2.0
    print(f"  Norm ratio   : {ratio:.3f}  (expect 0.5–2.0)")
    print(f"  Residual     : {'✓ PASSED' if res_ok else '✗ FAILED'}")

    # Parameter breakdown
    attn_p = sum(p.numel() for p in block.attn.parameters())
    ffn_p  = sum(p.numel() for p in block.ffn.parameters())
    ln_p   = sum(p.numel() for p in block.ln_1.parameters()) + \
             sum(p.numel() for p in block.ln_2.parameters())
    total  = sum(p.numel() for p in block.parameters())

    print(f"\n  Params per block:")
    print(f"    Attention  : {attn_p:>10,}")
    print(f"    FFN        : {ffn_p:>10,}")
    print(f"    LayerNorms : {ln_p:>10,}")
    print(f"    Total      : {total:>10,}")
    print(f"    x{N_LAYERS} blocks : {total*N_LAYERS:>10,}")

    # Stack test — simulate full GPT depth
    print(f"\n  Testing stack of {N_LAYERS} blocks...")
    stack = nn.Sequential(*[
        TransformerBlock(N_EMBD, N_HEAD, BLOCK_SIZE, dropout=0.1)
        for _ in range(N_LAYERS)
    ]).to(device)

    with torch.no_grad():
        out_stack = stack(x)

    assert out_stack.shape == x.shape
    stack_params = sum(p.numel() for p in stack.parameters())
    print(f"  Stack output : {out_stack.shape}")
    print(f"  Stack params : {stack_params:,}")
    print(f"  Stack check  : ✓ PASSED")

    print(f"\n  TransformerBlock working correctly!")
# To be implemented
# src/model/attention.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention.

    Each token looks at all PREVIOUS tokens and decides
    how much to attend to each one. Done in parallel
    across n_head independent heads — each head learns
    different relationships (syntax, semantics, etc).

    "Causal" = token at position i cannot see position i+1
    This is enforced by the triangular mask.
    """

    def __init__(self, n_embd, n_head, block_size, dropout=0.1, bias=True):
        super().__init__()

        assert n_embd % n_head == 0, \
            f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"

        self.n_embd   = n_embd
        self.n_head   = n_head
        self.head_dim = n_embd // n_head  # dims per head

        # Q, K, V projections in one matrix — GPU efficient
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Dropout
        self.attn_dropout  = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask — lower triangular matrix
        # token i can only attend to positions 0..i
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
                  .view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape   # batch, seq_len, n_embd

        # ── Project to Q, K, V ───────────────────────────────
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        # ── Reshape for multi-head ───────────────────────────
        # (B, T, C) → (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # ── Scaled dot-product attention ─────────────────────
        # Attention(Q,K,V) = softmax(QKᵀ / √d_k) · V
        scale  = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # ── Apply causal mask ────────────────────────────────
        # Future positions → -inf → softmax gives them 0 weight
        scores = scores.masked_fill(
            self.mask[:, :, :T, :T] == 0, float('-inf')
        )

        # ── Softmax + weighted sum ───────────────────────────
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        y = torch.matmul(weights, v)           # (B, n_head, T, head_dim)

        # ── Reassemble heads ─────────────────────────────────
        # (B, n_head, T, head_dim) → (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # ── Output projection ────────────────────────────────
        y = self.resid_dropout(self.c_proj(y))
        return y


# ── Test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing CausalSelfAttention...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    attn = CausalSelfAttention(
        n_embd=512, n_head=8, block_size=512, dropout=0.1
    ).to(device)

    x = torch.randn(4, 512, 512).to(device)
    print(f"  Input  shape : {x.shape}")

    with torch.no_grad():
        out = attn(x)
    print(f"  Output shape : {out.shape}")
    assert out.shape == x.shape
    print(f"  Shape check  : ✓ PASSED")

    # Causal check — eval mode disables dropout for determinism
    attn.eval()
    with torch.no_grad():
        out1 = attn(x)
        x2   = x.clone()
        x2[:, 5:, :] = torch.randn_like(x2[:, 5:, :])
        out2 = attn(x2)
    causal_ok = torch.allclose(out1[:, :5, :], out2[:, :5, :], atol=1e-5)
    print(f"  Causal check : {'✓ PASSED' if causal_ok else '✗ FAILED'}")
    attn.train()

    params = sum(p.numel() for p in attn.parameters())
    print(f"  Params       : {params:,}")
    print(f"\n  Attention working correctly!")
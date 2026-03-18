# src/model/gpt.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from src.model.transformer import TransformerBlock


# ── Config ────────────────────────────────────────────────────
@dataclass
class GPTConfig:
    vocab_size  : int   = 32000
    block_size  : int   = 1024
    n_layer     : int   = 12
    n_head      : int   = 12
    n_embd      : int   = 768
    dropout     : float = 0.1
    bias        : bool  = True


# ── Full GPT Model ────────────────────────────────────────────
class GPT(nn.Module):
    """
    GPT Language Model — built from scratch.

    Architecture:
      Token IDs
          ↓
      Token Embedding  (vocab_size → n_embd)
          +
      Position Embedding (block_size → n_embd)
          ↓
      Dropout
          ↓
      TransformerBlock × n_layer
          ↓
      LayerNorm
          ↓
      Linear head (n_embd → vocab_size)
          ↓
      Logits
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            # Token embedding — each of vocab_size tokens gets a vector
            tok_emb = nn.Embedding(config.vocab_size, config.n_embd),
            # Position embedding — each position 0..block_size gets a vector
            pos_emb = nn.Embedding(config.block_size, config.n_embd),
            drop    = nn.Dropout(config.dropout),
            blocks  = nn.ModuleList([
                TransformerBlock(
                    n_embd     = config.n_embd,
                    n_head     = config.n_head,
                    block_size = config.block_size,
                    dropout    = config.dropout,
                    bias       = config.bias,
                )
                for _ in range(config.n_layer)
            ]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))

        # Output head — projects n_embd → vocab_size
        self.lm_head = nn.Linear(
            config.n_embd, config.vocab_size, bias=False
        )

        # Weight tying — token embedding and output head share weights
        # Tokens similar in embedding space get similar output scores
        self.transformer.tok_emb.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale down residual projections by 1/√n_layer
        # Prevents signal explosion with depth
        for name, param in self.named_parameters():
            if name.endswith("c_proj.weight"):
                nn.init.normal_(
                    param,
                    mean = 0.0,
                    std  = 0.02 / math.sqrt(2 * config.n_layer),
                )

    def _init_weights(self, module):
        """GPT-2 style weight initialization."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx     : token IDs (batch, seq_len)
            targets : next token IDs (batch, seq_len) — optional

        Returns:
            logits : (batch, seq_len, vocab_size)
            loss   : scalar cross-entropy or None
        """
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block_size {self.config.block_size}"

        device = idx.device

        # Token + position embeddings
        tok = self.transformer.tok_emb(idx)
        pos = self.transformer.pos_emb(
            torch.arange(T, device=device)
        )
        x = self.transformer.drop(tok + pos)

        # Transformer blocks
        for block in self.transformer.blocks:
            x = block(x)

        # Final norm + output head
        x      = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # Loss — only computed during training
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index = 0,   # ignore <|pad|> token
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature        = 1.0,
        top_k              = 40,
        repetition_penalty = 1.3,
    ):
        """
        Autoregressive generation with repetition penalty.

        Args:
            idx                : starting token IDs (1, seq_len)
            max_new_tokens     : how many tokens to generate
            temperature        : >1 creative, <1 focused
            top_k              : sample from top k tokens only
            repetition_penalty : >1.0 reduces repeated tokens
                                 1.0 = off
                                 1.3 = moderate (recommended)
                                 1.5 = strong

        Returns:
            idx : (1, seq_len + max_new_tokens)
        """
        self.eval()

        for _ in range(max_new_tokens):

            # Crop to block_size
            idx_cond  = idx[:, -self.config.block_size:]

            # Forward pass
            logits, _ = self(idx_cond)
            logits     = logits[:, -1, :]   # last position only

            # ── Repetition penalty ─────────────────────────────
            # Penalize tokens that have already appeared
            # This prevents the model from looping
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # ── Temperature scaling ───────────────────────────
            logits = logits / temperature

            # ── Top-k filtering ───────────────────────────────
            if top_k is not None:
                v, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1))
                )
                logits[logits < v[:, [-1]]] = float('-inf')

            # ── Sample next token ─────────────────────────────
            probs      = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append and continue
            idx = torch.cat([idx, next_token], dim=1)

        return idx

    def get_num_params(self):
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters())

    def estimate_mfu(self, tokens_per_step, dt_sec):
        """
        Estimate Model FLOPs Utilization.
        RTX 4060 Laptop fp16 peak ≈ 16.2 TFLOPS
        """
        N   = self.get_num_params()
        cfg = self.config
        L   = cfg.n_layer
        H   = cfg.n_head
        Q   = cfg.n_embd // cfg.n_head
        T   = cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_step  = flops_per_token * tokens_per_step
        return flops_per_step / (16.2e12 * dt_sec)


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing 110M GPT model...\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device : {device}")

    config = GPTConfig(
        vocab_size = 32000,
        block_size = 1024,
        n_layer    = 12,
        n_head     = 12,
        n_embd     = 768,
        dropout    = 0.1,
    )

    model = GPT(config).to(device)
    total = model.get_num_params()
    print(f"  Params : {total:,}")

    # Forward pass
    torch.cuda.reset_peak_memory_stats()
    idx     = torch.randint(0, 32000, (2, 1024)).to(device)
    targets = torch.randint(0, 32000, (2, 1024)).to(device)

    from torch.amp import autocast
    with autocast(device_type="cuda", enabled=True):
        logits, loss = model(idx, targets)

    print(f"  Logits : {logits.shape}")
    print(f"  Loss   : {loss.item():.4f}")

    expected = math.log(config.vocab_size)
    loss_ok  = abs(loss.item() - expected) < 1.5
    print(f"  Expected loss  : ~{expected:.4f}")
    print(f"  Loss check     : {'✓ PASSED' if loss_ok else '✗ FAILED'}")

    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM   : {vram:.2f} GB / 8.00 GB")

    # Generation with repetition penalty
    print(f"\n  Testing generation with repetition penalty...")
    prompt    = torch.randint(0, 32000, (1, 8)).to(device)
    generated = model.generate(
        prompt,
        max_new_tokens     = 30,
        temperature        = 0.8,
        top_k              = 40,
        repetition_penalty = 1.3,
    )
    print(f"  Output length  : {generated.shape[1]} tokens")
    print(f"  Gen check      : {'✓ PASSED' if generated.shape[1] == 38 else '✗ FAILED'}")

    print(f"\n  110M GPT ready!")
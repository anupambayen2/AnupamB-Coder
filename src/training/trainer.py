# src/training/trainer.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import math
import time
import json
import torch
from torch.amp import GradScaler, autocast
from datetime import datetime

from paths import CHECKPOINT_BASE, LOG_DIR, checkpoint_dir, stage_log
from src.model.gpt          import GPT, GPTConfig
from src.data.dataset       import build_dataloaders
from src.training.optimizer import build_optimizer


# ── Training config — 110M model ─────────────────────────────
class TrainConfig:

    # ── Model ─────────────────────────────────────────────────
    vocab_size  = 32000
    block_size  = 1024
    n_layer     = 12
    n_head      = 12    # updated from 16
    n_embd      = 768   # updated from 1024
    dropout     = 0.1
    bias        = True

    # ── Training ─────────────────────────────────────────────
    batch_size              = 2
    grad_accum_steps        = 16    # effective batch = 32
    max_steps               = 100000
    warmup_steps            = 2000
    learning_rate           = 3e-4
    min_lr                  = 3e-5
    weight_decay            = 0.1
    grad_clip               = 1.0

    # ── Intervals ────────────────────────────────────────────
    eval_interval           = 500
    eval_steps              = 50
    save_interval           = 1000
    log_interval            = 10

    # ── Hardware ─────────────────────────────────────────────
    device                  = "cuda" if torch.cuda.is_available() else "cpu"
    fp16                    = True
    use_8bit_adam           = True

    # ── Stage ────────────────────────────────────────────────
    stage                   = 1


# ── LR schedule — cosine with linear warmup ──────────────────
def get_lr(step: int, cfg: TrainConfig) -> float:
    # Phase 1 — linear warmup
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    # After max steps — hold at min_lr
    if step > cfg.max_steps:
        return cfg.min_lr
    # Phase 2 — cosine decay
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, cfg: TrainConfig) -> dict:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= cfg.eval_steps:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        with autocast(device_type="cuda", enabled=cfg.fp16):
            _, loss = model(x, y)
        losses.append(loss.item())
    avg_loss   = sum(losses) / len(losses)
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {"val_loss": avg_loss, "perplexity": perplexity}


# ── Checkpoint save ───────────────────────────────────────────
def save_checkpoint(model, optimizer, scaler, step, loss, cfg: TrainConfig):
    ckpt_dir = checkpoint_dir(cfg.stage)
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"step_{step:06d}.pt")
    torch.save({
        "step"        : step,
        "stage"       : cfg.stage,
        "model_state" : model.state_dict(),
        "optim_state" : optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "loss"        : loss,
        "config"      : {
            "vocab_size" : cfg.vocab_size,
            "block_size" : cfg.block_size,
            "n_layer"    : cfg.n_layer,
            "n_head"     : cfg.n_head,
            "n_embd"     : cfg.n_embd,
        },
    }, path)
    print(f"  [ckpt] saved → {path}")
    return path


# ── Checkpoint load ───────────────────────────────────────────
def load_checkpoint(model, optimizer, scaler, path: str, device: str):
    print(f"  [ckpt] loading → {path}")
    ckpt  = torch.load(path, map_location=device)
    state = ckpt["model_state"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    if optimizer and "optim_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("  [ckpt] optimizer state skipped")
    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])
    step = ckpt.get("step", 0)
    loss = ckpt.get("loss", 0.0)
    print(f"  [ckpt] resumed step {step}  loss {loss:.4f}")
    return step, loss


# ── Logger ────────────────────────────────────────────────────
class Logger:
    def __init__(self, stage: int):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path = stage_log(stage)
        self.t0   = time.time()

    def write(self, data: dict):
        data["ts"]  = datetime.now().strftime("%H:%M:%S")
        data["min"] = round((time.time() - self.t0) / 60, 1)
        with open(self.path, "a") as f:
            f.write(json.dumps(data) + "\n")

    # def print_step(self, step, loss, lr, dt_ms, vram_gb=None):
    #     vram_s = f"  vram {vram_gb:.1f}GB" if vram_gb else ""
    #     print(
    #         f"  step {step:>7} | "
    #         f"loss {loss:.4f} | "
    #         f"lr {lr:.2e} | "
    #         f"{dt_ms:5.0f} ms/step"
    #         f"{vram_s}"
    #     )

    def print_step(self, step, loss, lr, dt_ms, vram_gb=None, max_steps=None):
        vram_s    = f"  vram {vram_gb:.1f}GB" if vram_gb else ""
        time_s    = ""
        if max_steps:
            elapsed_min   = (time.time() - self.t0) / 60
            remaining_min = ((max_steps - step) * dt_ms) / 1000 / 60
            if elapsed_min < 60:
                elapsed_s = f"{elapsed_min:.0f}m"
            else:
                elapsed_s = f"{elapsed_min/60:.1f}h"
            if remaining_min < 60:
                remain_s = f"~{remaining_min:.0f}m left"
            else:
                remain_s = f"~{remaining_min/60:.1f}h left"
            time_s = f"  {elapsed_s} elapsed  {remain_s}"
        print(
            f"  step {step:>7} | "
            f"loss {loss:.4f} | "
            f"lr {lr:.2e} | "
            f"{dt_ms:5.0f} ms/step"
            f"{vram_s}"
            f"{time_s}"
        )
    


# ── VRAM helper ───────────────────────────────────────────────
def get_vram_gb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3
    return None


# ── Main train function ───────────────────────────────────────
def train(resume_from=None, stage=1, max_steps=None):
    cfg       = TrainConfig()
    cfg.stage = stage

    if max_steps is not None:
        cfg.max_steps = max_steps

    logger = Logger(stage)

    print(f"\n{'═'*65}")
    print(f"  MINIGPT TRAINING — STAGE {stage}")
    print(f"{'═'*65}")
    print(f"  Device              : {cfg.device}")
    print(f"  Model               : {cfg.n_layer}L · {cfg.n_embd}D · "
          f"{cfg.n_head}H · {cfg.block_size} ctx")
    print(f"  Parameters          : ~110M")
    print(f"  Effective batch     : {cfg.batch_size} × "
          f"{cfg.grad_accum_steps} = "
          f"{cfg.batch_size * cfg.grad_accum_steps}")
    print(f"  Max steps           : {cfg.max_steps:,}")
    print(f"  Warmup steps        : {cfg.warmup_steps:,}")
    print(f"  Learning rate       : {cfg.learning_rate}")
    print(f"  8-bit Adam          : {cfg.use_8bit_adam}")
    print(f"  Mixed precision     : {cfg.fp16}")
    print(f"  Checkpoint dir      : {checkpoint_dir(stage)}")
    print(f"  Log file            : {stage_log(stage)}")
    print(f"{'─'*65}\n")

    # ── Data ─────────────────────────────────────────────────
    train_loader, val_loader, vocab_size = build_dataloaders(
        batch_size  = cfg.batch_size,
        num_workers = 0,
        block_size  = cfg.block_size,
        cache_name  = "stage1_bs1024",
    )

    # ── Model ────────────────────────────────────────────────
    model = GPT(GPTConfig(
        vocab_size = vocab_size,
        block_size = cfg.block_size,
        n_layer    = cfg.n_layer,
        n_head     = cfg.n_head,
        n_embd     = cfg.n_embd,
        dropout    = cfg.dropout,
        bias       = cfg.bias,
    )).to(cfg.device)

    total_params = model.get_num_params()
    print(f"  Model params        : {total_params:,}")

    vram = get_vram_gb()
    if vram:
        print(f"  VRAM after load     : {vram:.2f} GB / 8.00 GB")
        print(f"  VRAM headroom       : {8.0 - vram:.2f} GB\n")

    # ── Optimizer ────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        cfg.learning_rate,
        cfg.weight_decay,
        use_8bit = cfg.use_8bit_adam,
    )
    scaler = GradScaler(enabled=cfg.fp16)

    # ── Resume ───────────────────────────────────────────────
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        start_step, _ = load_checkpoint(
            model, optimizer, scaler,
            resume_from, cfg.device,
        )

    print("  torch.compile       : disabled (Windows)")
    model.train()

    # ── Training loop ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  Stage {stage} — starting from step {start_step}")
    print(f"{'─'*65}\n")

    train_iter    = iter(train_loader)
    best_val_loss = float("inf")
    step          = start_step
    t0            = time.time()
    optimizer.zero_grad()

    while step < cfg.max_steps:

        # LR update
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ── Gradient accumulation ─────────────────────────────
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y       = next(train_iter)

            x, y = x.to(cfg.device), y.to(cfg.device)
            with autocast(device_type="cuda", enabled=cfg.fp16):
                _, loss = model(x, y)
                loss    = loss / cfg.grad_accum_steps
            scaler.scale(loss).backward()
            loss_accum += loss.item()

        # ── Clip + step ───────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step  += 1
        dt_ms  = (time.time() - t0) * 1000
        t0     = time.time()

        # ── Log ───────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            vram = get_vram_gb()
            logger.print_step(step, loss_accum, lr, dt_ms, vram)
            logger.write({
                "step"      : step,
                "stage"     : stage,
                "train_loss": round(loss_accum, 4),
                "lr"        : round(lr, 8),
                "grad_norm" : round(grad_norm.item(), 4),
                "dt_ms"     : round(dt_ms, 1),
                "vram_gb"   : round(vram, 2) if vram else None,
            })

        # ── Eval ──────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg)
            print(f"\n  ── Eval @ step {step} "
                  f"(stage {stage}) {'─'*25}")
            print(f"     val_loss   : {metrics['val_loss']:.4f}")
            print(f"     perplexity : {metrics['perplexity']:.2f}")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_path = os.path.join(
                    checkpoint_dir(stage), "best.pt"
                )
                torch.save({
                    "step"        : step,
                    "stage"       : stage,
                    "model_state" : model.state_dict(),
                    "loss"        : best_val_loss,
                    "config"      : {
                        "vocab_size" : cfg.vocab_size,
                        "block_size" : cfg.block_size,
                        "n_layer"    : cfg.n_layer,
                        "n_head"     : cfg.n_head,
                        "n_embd"     : cfg.n_embd,
                    },
                }, best_path)
                print(f"     new best   → {best_path}")
            print()

            logger.write({
                "step"      : step,
                "stage"     : stage,
                "val_loss"  : round(metrics["val_loss"], 4),
                "perplexity": round(metrics["perplexity"], 2),
            })

        # ── Periodic checkpoint ───────────────────────────────
        if step % cfg.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                step, loss_accum, cfg,
            )

    # ── Done ──────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Stage {stage} complete!")
    print(f"  Total steps    : {step:,}")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Checkpoints    : {checkpoint_dir(stage)}")
    print(f"{'═'*65}\n")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume",    type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--stage",     type=int, default=1,
                        help="Training stage number (1-6)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max steps from config")
    args = parser.parse_args()
    train(
        resume_from = args.resume,
        stage       = args.stage,
        max_steps   = args.max_steps,
    )
    
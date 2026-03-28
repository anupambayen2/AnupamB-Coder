# src/training/trainer.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import math
import time
import json
import glob
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
    n_head      = 12
    n_embd      = 768
    dropout     = 0.1
    bias        = True

    # ── Training defaults (overridden per stage below) ────────
    batch_size              = 2
    grad_accum_steps        = 16
    max_steps               = 100000
    warmup_steps            = 2000
    learning_rate           = 3e-4
    min_lr                  = 3e-5
    weight_decay            = 0.1
    grad_clip               = 0.5     # ← reduced from 1.0

    # ── Intervals ─────────────────────────────────────────────
    eval_interval           = 500
    eval_steps              = 50
    save_interval           = 1000
    log_interval            = 10

    # ── Hardware ──────────────────────────────────────────────
    device                  = "cuda" if torch.cuda.is_available() else "cpu"
    fp16                    = True
    use_8bit_adam           = True

    # ── NaN protection ────────────────────────────────────────
    max_nan_skips           = 10

    # ── Stage ─────────────────────────────────────────────────
    stage                   = 1


# ── LR schedule — cosine with linear warmup ──────────────────
def get_lr(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
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
        if torch.isfinite(loss):
            losses.append(loss.item())

    if not losses:
        model.train()
        return {
            "val_loss"  : float("nan"),
            "perplexity": float("nan"),
        }

    avg_loss   = sum(losses) / len(losses)
    perplexity = math.exp(min(avg_loss, 20))
    model.train()
    return {"val_loss": avg_loss, "perplexity": perplexity}


# ── Checkpoint save ───────────────────────────────────────────
def save_checkpoint(
    model, optimizer, scaler, step, loss, cfg: TrainConfig
):
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

    all_ckpts = sorted(glob.glob(
        os.path.join(ckpt_dir, "step_*.pt")
    ))
    if len(all_ckpts) > 3:
        for old in all_ckpts[:-3]:
            try:
                os.remove(old)
                print(
                    f"  [ckpt] removed old → "
                    f"{os.path.basename(old)}"
                )
            except Exception:
                pass

    return path


# ── Checkpoint load ───────────────────────────────────────────
def load_checkpoint(
    model, optimizer, scaler, path: str, device: str
):
    print(f"  [ckpt] loading → {path}")
    ckpt  = torch.load(path, map_location=device)
    state = ckpt["model_state"]
    state = {
        k.replace("_orig_mod.", ""): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=False)

    if optimizer and "optim_state" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optim_state"])
        except Exception:
            print("  [ckpt] optimizer state skipped — fresh start")

    if scaler and "scaler_state" in ckpt:
        scaler.load_state_dict(ckpt["scaler_state"])

    step      = ckpt.get("step",  0)
    loss      = ckpt.get("loss",  0.0)
    ckpt_stage = ckpt.get("stage", 1)
    print(
        f"  [ckpt] resumed step {step}  "
        f"loss {loss:.4f}  "
        f"stage {ckpt_stage}"
    )
    return step, loss, ckpt_stage


# ── Logger ────────────────────────────────────────────────────
class Logger:
    def __init__(self, stage: int):
        os.makedirs(LOG_DIR, exist_ok=True)
        self.path  = stage_log(stage)
        self.t0    = time.time()
        self.stage = stage

        with open(self.path, "a") as f:
            f.write(json.dumps({
                "event" : "training_start",
                "stage" : stage,
                "ts"    : datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),
            }) + "\n")
        print(f"  Log file            : {self.path}")

    def write(self, data: dict):
        data["ts"]  = datetime.now().strftime("%H:%M:%S")
        data["min"] = round((time.time() - self.t0) / 60, 1)
        try:
            with open(self.path, "a") as f:
                f.write(json.dumps(data) + "\n")
        except Exception as e:
            print(f"  [log error] {e}")

    def print_step(
        self, step, loss, lr, dt_ms,
        vram_gb=None, max_steps=None,
    ):
        vram_s = f"  vram {vram_gb:.1f}GB" if vram_gb else ""
        time_s = ""

        if max_steps:
            elapsed_min   = (time.time() - self.t0) / 60
            remaining_min = (
                (max_steps - step) * dt_ms
            ) / 1000 / 60

            elapsed_s = (
                f"{elapsed_min:.0f}m"
                if elapsed_min < 60
                else f"{elapsed_min/60:.1f}h"
            )
            remain_s = (
                f"~{remaining_min:.0f}m left"
                if remaining_min < 60
                else f"~{remaining_min/60:.1f}h left"
            )
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

    # ── Stage-specific config ──────────────────────────────────
    if stage == 1:
        cfg.learning_rate = 3e-4
        cfg.min_lr        = 3e-5
        cfg.max_steps     = 100_000
        cfg.warmup_steps  = 2_000

    elif stage == 2:
        cfg.learning_rate = 1e-4
        cfg.min_lr        = 1e-5
        cfg.max_steps     = 400_000
        cfg.warmup_steps  = 1_000

    elif stage == 3:
        cfg.learning_rate = 2e-5
        cfg.min_lr        = 2e-6
        cfg.max_steps     = 300_000
        cfg.warmup_steps  = 1_000

    elif stage == 4:
        cfg.learning_rate = 1e-5
        cfg.min_lr        = 1e-6
        cfg.max_steps     = 150_000
        cfg.warmup_steps  = 500

    elif stage == 5:
        cfg.learning_rate = 5e-6
        cfg.min_lr        = 5e-7
        cfg.max_steps     = 80_000
        cfg.warmup_steps  = 200

    elif stage == 6:
        cfg.learning_rate = 2e-6
        cfg.min_lr        = 2e-7
        cfg.max_steps     = 30_000
        cfg.warmup_steps  = 200

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
    print(f"  Min LR              : {cfg.min_lr}")
    print(f"  Grad clip           : {cfg.grad_clip}  ← NaN safe")
    print(f"  8-bit Adam          : {cfg.use_8bit_adam}")
    print(f"  Mixed precision     : {cfg.fp16}")
    print(f"  NaN protection      : enabled")
    print(f"  Checkpoint dir      : {checkpoint_dir(stage)}")
    print(f"{'─'*65}\n")

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, vocab_size = build_dataloaders(
        batch_size  = cfg.batch_size,
        num_workers = 0,
        block_size  = cfg.block_size,
        stage       = stage,
    )

    # ── Model ─────────────────────────────────────────────────
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

    # ── Optimizer ─────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        cfg.learning_rate,
        cfg.weight_decay,
        use_8bit = cfg.use_8bit_adam,
    )
    scaler = GradScaler(enabled=cfg.fp16)

    # ── Resume — smart stage detection ────────────────────────
    start_step = 0

    if resume_from and os.path.exists(resume_from):
        start_step, _, ckpt_stage = load_checkpoint(
            model, optimizer, scaler,
            resume_from, cfg.device,
        )

        if ckpt_stage == stage:
            # ── SAME stage — continue from saved step ─────────
            print(
                f"  Resuming stage {stage} "
                f"from step {start_step}  ✅"
            )
            print(
                f"  Steps remaining : "
                f"{cfg.max_steps - start_step:,} / "
                f"{cfg.max_steps:,}"
            )
        else:
            # ── DIFFERENT stage — reset step counter ──────────
            print(
                f"  Stage change: {ckpt_stage} → {stage}"
            )
            print(
                f"  Resetting step counter for new stage"
            )
            start_step = 0

    print("  torch.compile       : disabled (Windows)")
    model.train()

    # ── Training loop ─────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(
        f"  Stage {stage} — "
        f"step {start_step:,} → {cfg.max_steps:,}"
    )
    print(f"{'─'*65}\n")

    train_iter        = iter(train_loader)
    best_val_loss     = float("inf")
    step              = start_step
    t0                = time.time()
    nan_skip_count    = 0
    total_nan_skipped = 0
    optimizer.zero_grad()

    while step < cfg.max_steps:

        # ── LR update ─────────────────────────────────────────
        lr = get_lr(step, cfg)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # ── Gradient accumulation with NaN protection ─────────
        loss_accum = 0.0
        skip_step  = False

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

            # ── NaN/Inf loss check ─────────────────────────────
            if not torch.isfinite(loss):
                print(
                    f"\n  [NaN] loss={loss.item()} "
                    f"at step {step} — skipping batch"
                )
                optimizer.zero_grad()
                skip_step         = True
                nan_skip_count   += 1
                total_nan_skipped += 1

                if nan_skip_count >= cfg.max_nan_skips:
                    print(
                        f"\n  [EMERGENCY STOP] "
                        f"{nan_skip_count} consecutive NaN!\n"
                        f"  Resume from best checkpoint:\n"
                        f"  python src/training/trainer.py "
                        f"--stage {stage} "
                        f"--resume "
                        f"{checkpoint_dir(stage)}/best.pt"
                    )
                    return
                break

            scaler.scale(loss).backward()
            loss_accum += loss.item()

        if skip_step:
            continue

        # Reset consecutive NaN counter on good batch
        nan_skip_count = 0

        # ── Clip + step ───────────────────────────────────────
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg.grad_clip
        )

        # ── NaN gradient check ────────────────────────────────
        if not torch.isfinite(grad_norm):
            print(
                f"\n  [NaN grad] norm={grad_norm.item()} "
                f"at step {step} — skipping update"
            )
            optimizer.zero_grad()
            scaler.update()
            nan_skip_count   += 1
            total_nan_skipped += 1

            if nan_skip_count >= cfg.max_nan_skips:
                print(
                    f"\n  [EMERGENCY STOP] "
                    f"{nan_skip_count} consecutive NaN grads!\n"
                    f"  Resume from best checkpoint:\n"
                    f"  python src/training/trainer.py "
                    f"--stage {stage} "
                    f"--resume "
                    f"{checkpoint_dir(stage)}/best.pt"
                )
                return
            continue

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        step  += 1
        dt_ms  = (time.time() - t0) * 1000
        t0     = time.time()

        # ── Log ───────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            vram = get_vram_gb()
            logger.print_step(
                step, loss_accum, lr, dt_ms, vram,
                max_steps = cfg.max_steps,
            )
            logger.write({
                "step"        : step,
                "stage"       : stage,
                "train_loss"  : round(loss_accum, 4),
                "lr"          : round(lr, 8),
                "grad_norm"   : round(grad_norm.item(), 4),
                "dt_ms"       : round(dt_ms, 1),
                "vram_gb"     : round(vram, 2) if vram else None,
                "nan_skipped" : total_nan_skipped,
            })

        # ── Eval ──────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            metrics    = evaluate(model, val_loader, cfg)
            val_loss   = metrics["val_loss"]
            perplexity = metrics["perplexity"]

            print(
                f"\n  ── Eval @ step {step} "
                f"(stage {stage}) {'─'*25}"
            )

            if not math.isfinite(val_loss):
                print(
                    f"     val_loss   : NaN ← model corrupted!\n"
                    f"     Stop and resume from best.pt"
                )
                print()
            else:
                print(f"     val_loss   : {val_loss:.4f}")
                print(f"     perplexity : {perplexity:.2f}")
                if total_nan_skipped > 0:
                    print(
                        f"     nan skipped: "
                        f"{total_nan_skipped} total"
                    )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_path     = os.path.join(
                        checkpoint_dir(stage), "best.pt"
                    )
                    os.makedirs(
                        checkpoint_dir(stage), exist_ok=True
                    )
                    torch.save({
                        "step"        : step,
                        "stage"       : stage,
                        "model_state" : model.state_dict(),
                        "loss"        : best_val_loss,
                        "config"      : {
                            "vocab_size": cfg.vocab_size,
                            "block_size": cfg.block_size,
                            "n_layer"   : cfg.n_layer,
                            "n_head"    : cfg.n_head,
                            "n_embd"    : cfg.n_embd,
                        },
                    }, best_path)
                    print(f"     new best   → {best_path}")
                print()

            logger.write({
                "step"       : step,
                "stage"      : stage,
                "val_loss"   : round(val_loss, 4)
                               if math.isfinite(val_loss)
                               else None,
                "perplexity" : round(perplexity, 2)
                               if math.isfinite(perplexity)
                               else None,
                "nan_skipped": total_nan_skipped,
            })

        # ── Periodic checkpoint ───────────────────────────────
        if step % cfg.save_interval == 0:
            save_checkpoint(
                model, optimizer, scaler,
                step, loss_accum, cfg,
            )

    # ── Stage complete ────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  Stage {stage} complete!")
    print(f"  Total steps      : {step:,}")
    print(f"  Best val loss    : {best_val_loss:.4f}")
    print(f"  NaN batches skip : {total_nan_skipped}")
    print(f"  Checkpoints      : {checkpoint_dir(stage)}")
    print(f"{'═'*65}\n")


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description = "Train AnupamB-Coder-110M"
    )
    parser.add_argument(
        "--resume",
        type    = str,
        default = None,
        help    = "Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--stage",
        type    = int,
        default = 1,
        help    = "Training stage (1-6)",
    )
    parser.add_argument(
        "--max_steps",
        type    = int,
        default = None,
        help    = "Override max steps from config",
    )
    args = parser.parse_args()

    train(
        resume_from = args.resume,
        stage       = args.stage,
        max_steps   = args.max_steps,
    )
# trainer_test.py
# ─────────────────────────────────────────────────────────────
# 2.5 hour smoke test for the 110M model
# Runs 5000 steps on existing 535K data
# Verifies: VRAM, loss curve, checkpointing, speed
# Run: python trainer_test.py
# ─────────────────────────────────────────────────────────────

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import math
import time
import torch
from torch.amp import GradScaler, autocast

from paths import LOG_DIR
from src.model.gpt          import GPT, GPTConfig
from src.data.dataset       import build_dataloaders
from src.training.optimizer import build_optimizer


# ── Smoke test config — 110M model ───────────────────────────
class SmokeConfig:

    # ── Model ─────────────────────────────────────────────────
    vocab_size  = 32000
    block_size  = 1024
    n_layer     = 12
    n_head      = 12    # 110M config
    n_embd      = 768   # 110M config
    dropout     = 0.1
    bias        = True

    # ── Short run ─────────────────────────────────────────────
    max_steps        = 5000
    batch_size       = 2
    grad_accum_steps = 16       # effective batch = 32
    warmup_steps     = 200
    learning_rate    = 3e-4
    min_lr           = 3e-5
    weight_decay     = 0.1
    grad_clip        = 1.0

    # ── Frequent checks ───────────────────────────────────────
    eval_interval    = 500
    eval_steps       = 20
    save_interval    = 1000
    log_interval     = 10

    # ── Hardware ─────────────────────────────────────────────
    device           = "cuda" if torch.cuda.is_available() else "cpu"
    fp16             = True
    use_8bit_adam    = True

    # ── Output — separate from real runs ─────────────────────
    checkpoint_dir   = os.path.join(ROOT, "checkpoints", "smoke_test")
    log_path         = os.path.join(ROOT, "logs", "smoke_test.jsonl")


# ── LR schedule ──────────────────────────────────────────────
def get_lr(step: int, cfg: SmokeConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step > cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / (cfg.max_steps - cfg.warmup_steps)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


# ── Evaluation ────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, cfg: SmokeConfig) -> dict:
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


# ── Main smoke test ───────────────────────────────────────────
def run_smoke_test():
    cfg = SmokeConfig()

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print(f"\n{'═'*65}")
    print(f"  110M MODEL SMOKE TEST")
    print(f"{'═'*65}")
    print(f"  Purpose   : verify 110M trains correctly before 500GB run")
    print(f"  Steps     : {cfg.max_steps:,}  (~2.5 hrs)")
    print(f"  Data      : existing 535K Python + SQL examples")
    print(f"  Device    : {cfg.device}")
    print(f"  Model     : {cfg.n_layer}L · {cfg.n_embd}D · "
          f"{cfg.n_head}H · {cfg.block_size} ctx")
    print(f"  Batch     : {cfg.batch_size} × {cfg.grad_accum_steps}"
          f" = {cfg.batch_size * cfg.grad_accum_steps} effective")
    print(f"  Output    : {cfg.checkpoint_dir}")
    print(f"{'─'*65}\n")

    # ── Data ─────────────────────────────────────────────────
    print("  Loading dataset...")
    train_loader, val_loader, vocab_size = build_dataloaders(
        batch_size  = cfg.batch_size,
        num_workers = 0,
        block_size  = cfg.block_size,
        cache_name  = "test_bs1024",   # reuse existing cache
    )

    # ── Model ────────────────────────────────────────────────
    print("  Building 110M model...")
    model = GPT(GPTConfig(
        vocab_size = vocab_size,
        block_size = cfg.block_size,
        n_layer    = cfg.n_layer,
        n_head     = cfg.n_head,
        n_embd     = cfg.n_embd,
        dropout    = cfg.dropout,
        bias       = cfg.bias,
    )).to(cfg.device)

    params = model.get_num_params()
    vram   = torch.cuda.memory_allocated() / 1024**3
    print(f"  Parameters  : {params:,}")
    print(f"  VRAM loaded : {vram:.2f} GB / 8.00 GB\n")

    # ── Optimizer ────────────────────────────────────────────
    optimizer = build_optimizer(
        model,
        cfg.learning_rate,
        cfg.weight_decay,
        use_8bit = cfg.use_8bit_adam,
    )
    scaler = GradScaler(enabled=cfg.fp16)

    model.train()
    optimizer.zero_grad()

    # ── Tracking ─────────────────────────────────────────────
    train_iter    = iter(train_loader)
    best_val_loss = float("inf")
    step          = 0
    t0            = time.time()
    t_start       = time.time()
    losses_log    = []
    vram_log      = []
    speed_log     = []

    print(f"{'─'*65}")
    print(f"  Starting smoke test...")
    print(f"{'─'*65}\n")

    while step < cfg.max_steps:

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

        vram_gb = torch.cuda.memory_allocated() / 1024**3
        losses_log.append(loss_accum)
        vram_log.append(vram_gb)
        speed_log.append(dt_ms)

        # ── Log ───────────────────────────────────────────────
        if step % cfg.log_interval == 0:
            elapsed   = (time.time() - t_start) / 60
            remaining = ((cfg.max_steps - step) * dt_ms) / 1000 / 60
            print(
                f"  step {step:>5} | "
                f"loss {loss_accum:.4f} | "
                f"lr {lr:.2e} | "
                f"{dt_ms:5.0f}ms | "
                f"vram {vram_gb:.1f}GB | "
                f"elapsed {elapsed:.0f}m | "
                f"~{remaining:.0f}m left"
            )

        # ── Eval ──────────────────────────────────────────────
        if step % cfg.eval_interval == 0:
            metrics = evaluate(model, val_loader, cfg)
            print(f"\n  ── Eval @ step {step} {'─'*35}")
            print(f"     val_loss   : {metrics['val_loss']:.4f}")
            print(f"     perplexity : {metrics['perplexity']:.2f}")

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_path = os.path.join(cfg.checkpoint_dir, "best.pt")
                torch.save({
                    "step"        : step,
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

        # ── Periodic checkpoint ───────────────────────────────
        if step % cfg.save_interval == 0:
            ckpt_path = os.path.join(
                cfg.checkpoint_dir, f"step_{step:06d}.pt"
            )
            torch.save({
                "step"        : step,
                "model_state" : model.state_dict(),
                "optim_state" : optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "loss"        : loss_accum,
            }, ckpt_path)
            print(f"  [ckpt] saved → {ckpt_path}")

    # ── Final report ──────────────────────────────────────────
    total_time = (time.time() - t_start) / 60
    avg_speed  = sum(speed_log) / len(speed_log)
    max_vram   = max(vram_log)
    start_loss = losses_log[0]
    end_loss   = losses_log[-1]
    loss_drop  = start_loss - end_loss

    checks = {
        "Loss dropped > 1.0"    : loss_drop > 1.0,
        "No OOM (vram < 7.5GB)" : max_vram < 7.5,
        "Speed < 5000ms/step"   : avg_speed < 5000,
        "Val loss improved"     : best_val_loss < 10.0,
        "Checkpoints saved"     : os.path.exists(
            os.path.join(cfg.checkpoint_dir, "best.pt")
        ),
    }

    print(f"\n{'═'*65}")
    print(f"  SMOKE TEST COMPLETE")
    print(f"{'═'*65}")
    print(f"  Total time     : {total_time:.1f} minutes")
    print(f"  Steps          : {step:,}")
    print(f"  Start loss     : {start_loss:.4f}")
    print(f"  End loss       : {end_loss:.4f}")
    print(f"  Loss drop      : {loss_drop:.4f}")
    print(f"  Best val loss  : {best_val_loss:.4f}")
    print(f"  Max VRAM       : {max_vram:.2f} GB / 8.00 GB")
    print(f"  Avg speed      : {avg_speed:.0f} ms/step")

    est_stage1 = (100000 * avg_speed) / 1000 / 3600
    est_total  = (1070000 * avg_speed) / 1000 / 3600 / 24
    print(f"  Est Stage 1    : ~{est_stage1:.1f} hrs (100K steps)")
    print(f"  Est all stages : ~{est_total:.1f} days (1.07M steps)")

    print(f"\n{'─'*65}")
    print(f"  PASS / FAIL CHECKS:")
    all_passed = True
    for check, passed in checks.items():
        status = "PASSED" if passed else "FAILED"
        mark   = "+" if passed else "!"
        if not passed:
            all_passed = False
        print(f"    [{mark}] {check:<30} : {status}")

    print(f"{'─'*65}")

    if all_passed:
        print(f"\n  ALL CHECKS PASSED")
        print(f"  110M model is ready for 500GB training.")
        print(f"\n  Next steps:")
        print(f"    Terminal 1 — start Stage 1 training:")
        print(f"      python src/training/trainer.py --stage 1")
        print(f"    Terminal 2 — start downloading 500GB data:")
        print(f"      python src/data/downloader_500gb.py")
    else:
        print(f"\n  SOME CHECKS FAILED")
        print(f"  Review the failures above before full training.")

    print(f"{'═'*65}\n")


if __name__ == "__main__":
    run_smoke_test()
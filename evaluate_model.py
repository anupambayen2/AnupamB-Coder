# evaluate_model.py
# AnupamB-Coder-110M  |  Stage 1 Inference & Code Generation Test
# ═══════════════════════════════════════════════════════════════
#
#  Usage:
#    python evaluate_model.py                        # interactive mode
#    python evaluate_model.py --prompt "def sort("   # single prompt
#    python evaluate_model.py --suite                # full test suite
#    python evaluate_model.py --benchmark            # perplexity bench
#
# ═══════════════════════════════════════════════════════════════

import sys
import os
import time
import argparse
import textwrap

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
import numpy as np
from tokenizers import Tokenizer

# ── Paths ──────────────────────────────────────────────────────
CKPT_PATH      = r"E:\mini_gpt\checkpoints\run_001\step_100000.pt"
TOKENIZER_PATH = r"E:\mini_gpt\tokenizer\tokenizer.json"   # adjust if needed

# ── Generation defaults ────────────────────────────────────────
DEFAULT_MAX_NEW   = 256
DEFAULT_TEMP      = 0.8
DEFAULT_TOP_K     = 50
DEFAULT_TOP_P     = 0.95
DEFAULT_REP_PEN   = 1.1     # penalise repeated tokens


# ══════════════════════════════════════════════════════════════
#  Model loader  (reads architecture from checkpoint itself)
# ══════════════════════════════════════════════════════════════

def load_model_and_tokenizer(ckpt_path: str, tokenizer_path: str, device: str):
    print(f"\n{'═'*58}")
    print(f"  Loading Stage 1 checkpoint")
    print(f"{'═'*58}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── Read config from checkpoint ───────────────────────────
    cfg = ckpt.get("config", ckpt.get("model_config", ckpt.get("cfg", {})))
    print(f"\n  Model config from checkpoint:")
    for k, v in cfg.items():
        print(f"    {k:<20} : {v}")

    # ── Import and build your model ───────────────────────────
    # Adjust this import to match your project structure
    try:
        from model import GPT, GPTConfig          # try root-level
    except ImportError:
        try:
            from src.model import GPT, GPTConfig  # try src/
        except ImportError:
            from src.model.gpt import GPT, GPTConfig  # try src/model/

    model_cfg = GPTConfig(**cfg) if cfg else GPTConfig()
    model     = GPT(model_cfg).to(device)

    # Load weights
    state_dict = ckpt.get("model", ckpt.get("model_state_dict", ckpt))
    # Strip "module." prefix if saved with DataParallel
    state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\n  ✓ Model loaded — {n_params:.1f}M parameters")

    # ── Load tokenizer ────────────────────────────────────────
    tokenizer = Tokenizer.from_file(tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"  ✓ Tokenizer loaded — vocab {vocab_size:,}")
    print(f"{'═'*58}\n")

    return model, tokenizer, cfg


# ══════════════════════════════════════════════════════════════
#  Core generation function
# ══════════════════════════════════════════════════════════════

@torch.inference_mode()
def generate(
    model,
    tokenizer,
    prompt      : str,
    max_new     : int   = DEFAULT_MAX_NEW,
    temperature : float = DEFAULT_TEMP,
    top_k       : int   = DEFAULT_TOP_K,
    top_p       : float = DEFAULT_TOP_P,
    rep_penalty : float = DEFAULT_REP_PEN,
    device      : str   = "cuda",
    ctx_len     : int   = 1024,
) -> tuple[str, dict]:
    """
    Generate tokens autoregressively from a prompt.

    Returns:
        generated_text : str   — full output (prompt + completion)
        stats          : dict  — tokens/sec, latency, token counts
    """

    # Encode prompt
    enc       = tokenizer.encode(prompt)
    input_ids = enc.ids

    # Strip BOS if tokenizer added it automatically
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")
    if input_ids and input_ids[0] == bos_id:
        input_ids = input_ids[1:]

    # Prepend BOS manually
    tokens = torch.tensor(
        [bos_id] + input_ids, dtype=torch.long, device=device
    ).unsqueeze(0)   # [1, T]

    prompt_len  = tokens.shape[1]
    generated   = 0
    t0          = time.perf_counter()

    for _ in range(max_new):
        # Crop to context window
        idx_cond = tokens if tokens.shape[1] <= ctx_len \
                   else tokens[:, -ctx_len:]

        logits = model(idx_cond)          # [1, T, vocab]
        logits = logits[:, -1, :]         # last position: [1, vocab]

        # ── Repetition penalty ────────────────────────────────
        if rep_penalty != 1.0:
            for tok_id in set(tokens[0].tolist()):
                if logits[0, tok_id] < 0:
                    logits[0, tok_id] *= rep_penalty
                else:
                    logits[0, tok_id] /= rep_penalty

        # ── Temperature ───────────────────────────────────────
        if temperature != 1.0:
            logits = logits / temperature

        # ── Top-K ─────────────────────────────────────────────
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_k_vals[:, [-1]]] = -float("Inf")

        # ── Top-P (nucleus sampling) ──────────────────────────
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative prob above top_p
            sorted_logits[cumprobs - F.softmax(sorted_logits, dim=-1) > top_p] \
                = -float("Inf")
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs    = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)   # [1, 1]

        tokens    = torch.cat([tokens, next_tok], dim=1)
        generated += 1

        # Stop at EOS
        if next_tok.item() == eos_id:
            break

    elapsed   = time.perf_counter() - t0
    tok_per_s = generated / elapsed if elapsed > 0 else 0

    # Decode full sequence (skip leading BOS)
    out_ids   = tokens[0].tolist()
    if out_ids and out_ids[0] == bos_id:
        out_ids = out_ids[1:]
    output    = tokenizer.decode(out_ids)

    stats = {
        "prompt_tokens"    : prompt_len - 1,   # -1 for BOS
        "generated_tokens" : generated,
        "total_tokens"     : prompt_len - 1 + generated,
        "latency_s"        : round(elapsed, 3),
        "tokens_per_sec"   : round(tok_per_s, 1),
    }

    return output, stats


# ══════════════════════════════════════════════════════════════
#  Benchmark prompts  (code generation test suite)
# ══════════════════════════════════════════════════════════════

TEST_SUITE = [
    {
        "name"   : "Python — basic function",
        "prompt" : "def add_numbers(a, b):",
        "expect" : ["return", "a", "b"],
    },
    {
        "name"   : "Python — list comprehension",
        "prompt" : "def get_even_numbers(nums):\n    # return only even numbers\n    return",
        "expect" : ["for", "if", "%"],
    },
    {
        "name"   : "Python — class definition",
        "prompt" : "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):",
        "expect" : ["append", "self"],
    },
    {
        "name"   : "Python — file handling",
        "prompt" : "def read_file(filepath):\n    with open(",
        "expect" : ["as", "read", "return"],
    },
    {
        "name"   : "Python — recursion",
        "prompt" : "def fibonacci(n):\n    if n <= 1:\n        return n",
        "expect" : ["return", "fibonacci", "n-"],
    },
    {
        "name"   : "Python — error handling",
        "prompt" : "def safe_divide(a, b):\n    try:",
        "expect" : ["except", "ZeroDivisionError", "return"],
    },
    {
        "name"   : "SQL — SELECT query",
        "prompt" : "-- Get all users older than 30\nSELECT",
        "expect" : ["FROM", "WHERE", "age"],
    },
    {
        "name"   : "SQL — JOIN",
        "prompt" : "-- Get orders with customer names\nSELECT orders.id, customers.name\nFROM orders\nJOIN",
        "expect" : ["customers", "ON", "id"],
    },
    {
        "name"   : "Python — sorting",
        "prompt" : "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):",
        "expect" : ["for", "range", "swap"],
    },
    {
        "name"   : "Python — dictionary",
        "prompt" : "def count_words(text):\n    words = text.split()\n    counts = {}",
        "expect" : ["for", "in", "counts"],
    },
]


def run_test_suite(model, tokenizer, device, cfg, max_new=150, temp=0.7):
    """Run all test prompts and print a scorecard."""

    ctx_len = cfg.get("block_size", cfg.get("ctx_len", cfg.get("n_positions", 1024)))

    print(f"\n{'═'*58}")
    print(f"  STAGE 1 CODE GENERATION TEST SUITE")
    print(f"  temp={temp}  top_k=40  top_p=0.95  max_new={max_new}")
    print(f"{'═'*58}\n")

    passed = 0
    results = []

    for i, test in enumerate(TEST_SUITE):
        output, stats = generate(
            model, tokenizer,
            prompt      = test["prompt"],
            max_new     = max_new,
            temperature = temp,
            top_k       = 40,
            top_p       = 0.95,
            rep_penalty = 1.1,
            device      = device,
            ctx_len     = ctx_len,
        )

        # Simple keyword check — did expected tokens appear?
        completion = output[len(test["prompt"]):]
        hits   = sum(1 for kw in test["expect"]
                     if kw.lower() in completion.lower())
        score  = hits / len(test["expect"])
        ok     = score >= 0.5
        passed += int(ok)

        tag = "✓" if ok else "✗"
        results.append((test["name"], ok, score, completion, stats))

        print(f"  {tag} [{i+1:02d}/{len(TEST_SUITE)}] {test['name']}")
        print(f"  {'─'*54}")
        print(f"  PROMPT:\n{textwrap.indent(test['prompt'], '    ')}")
        print(f"\n  COMPLETION:")
        # Print first 10 lines of completion
        lines = completion.split("\n")[:10]
        for line in lines:
            print(f"    {line}")
        if len(completion.split("\n")) > 10:
            print(f"    ... ({len(completion.split(chr(10)))-10} more lines)")
        print(f"\n  Keywords hit: {hits}/{len(test['expect'])}  "
              f"Score: {score:.0%}  "
              f"Speed: {stats['tokens_per_sec']:.0f} tok/s")
        print(f"  {'═'*54}\n")

    # ── Summary scorecard ─────────────────────────────────────
    print(f"\n{'═'*58}")
    print(f"  SCORECARD — Stage 1 @ 100K steps")
    print(f"{'─'*58}")
    print(f"  Tests passed  : {passed}/{len(TEST_SUITE)}")
    print(f"  Pass rate     : {passed/len(TEST_SUITE):.0%}")
    print(f"\n  Per-test results:")
    for name, ok, score, _, stats in results:
        tag = "✓" if ok else "✗"
        print(f"    {tag} {name:<35} {score:.0%}  "
              f"({stats['tokens_per_sec']:.0f} tok/s)")

    # ── Stage readiness verdict ───────────────────────────────
    rate = passed / len(TEST_SUITE)
    print(f"\n{'─'*58}")
    if rate >= 0.8:
        verdict = "🟢  EXCELLENT — Stage 2 ready"
        detail  = "Model generates coherent, syntactically correct code."
    elif rate >= 0.6:
        verdict = "🟡  GOOD — Stage 2 recommended"
        detail  = "Model understands structure. Stage 2 will improve depth."
    elif rate >= 0.4:
        verdict = "🟠  FAIR — Stage 2 needed"
        detail  = "Basic patterns present. Stage 2 training critical."
    else:
        verdict = "🔴  WEAK — check training run before Stage 2"
        detail  = "Low coherence. Verify checkpoint and tokenizer alignment."

    print(f"  {verdict}")
    print(f"  {detail}")
    print(f"{'═'*58}\n")

    return results


# ══════════════════════════════════════════════════════════════
#  Interactive REPL
# ══════════════════════════════════════════════════════════════

def interactive_loop(model, tokenizer, device, cfg):
    ctx_len = cfg.get("block_size", cfg.get("ctx_len", cfg.get("n_positions", 1024)))

    print(f"\n{'═'*58}")
    print(f"  INTERACTIVE CODE GENERATION")
    print(f"  Commands: :q = quit | :temp 0.7 | :top_k 50 | :len 256")
    print(f"  Multi-line: end prompt with \\ then press Enter")
    print(f"{'═'*58}\n")

    temp    = DEFAULT_TEMP
    top_k   = DEFAULT_TOP_K
    top_p   = DEFAULT_TOP_P
    max_new = DEFAULT_MAX_NEW

    while True:
        try:
            lines = []
            while True:
                prefix = ">>> " if not lines else "... "
                line   = input(prefix)
                if line.endswith("\\"):
                    lines.append(line[:-1])
                else:
                    lines.append(line)
                    break
            prompt = "\n".join(lines)
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Exiting.\n")
            break

        if not prompt.strip():
            continue

        # ── Commands ──────────────────────────────────────────
        if prompt.strip() == ":q":
            print("\n  Goodbye!\n")
            break
        if prompt.strip().startswith(":temp"):
            try:
                temp = float(prompt.split()[1])
                print(f"  temperature → {temp}")
            except:
                print("  Usage: :temp 0.8")
            continue
        if prompt.strip().startswith(":top_k"):
            try:
                top_k = int(prompt.split()[1])
                print(f"  top_k → {top_k}")
            except:
                print("  Usage: :top_k 50")
            continue
        if prompt.strip().startswith(":len"):
            try:
                max_new = int(prompt.split()[1])
                print(f"  max_new_tokens → {max_new}")
            except:
                print("  Usage: :len 256")
            continue
        if prompt.strip() == ":settings":
            print(f"  temp={temp}  top_k={top_k}  "
                  f"top_p={top_p}  max_new={max_new}")
            continue

        # ── Generate ──────────────────────────────────────────
        print(f"\n{'─'*58}")
        output, stats = generate(
            model, tokenizer,
            prompt      = prompt,
            max_new     = max_new,
            temperature = temp,
            top_k       = top_k,
            top_p       = top_p,
            rep_penalty = DEFAULT_REP_PEN,
            device      = device,
            ctx_len     = ctx_len,
        )

        # Print only the completion (not the prompt)
        completion = output[len(prompt):]
        print(completion)
        print(f"{'─'*58}")
        print(f"  [{stats['generated_tokens']} tokens | "
              f"{stats['latency_s']}s | "
              f"{stats['tokens_per_sec']} tok/s]\n")


# ══════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="AnupamB-Coder-110M  |  Stage 1 Evaluation"
    )
    parser.add_argument(
        "--ckpt",
        default = CKPT_PATH,
        help    = "Path to checkpoint .pt file",
    )
    parser.add_argument(
        "--tokenizer",
        default = TOKENIZER_PATH,
        help    = "Path to tokenizer.json",
    )
    parser.add_argument(
        "--prompt",
        type    = str,
        default = None,
        help    = "Single prompt — print output and exit",
    )
    parser.add_argument(
        "--suite",
        action  = "store_true",
        help    = "Run the full test suite (10 prompts)",
    )
    parser.add_argument(
        "--temp",
        type    = float,
        default = DEFAULT_TEMP,
        help    = "Sampling temperature (default 0.8)",
    )
    parser.add_argument(
        "--max_new",
        type    = int,
        default = DEFAULT_MAX_NEW,
        help    = "Max new tokens to generate",
    )
    parser.add_argument(
        "--top_k",
        type    = int,
        default = DEFAULT_TOP_K,
    )
    parser.add_argument(
        "--cpu",
        action  = "store_true",
        help    = "Force CPU (default: auto-detect CUDA)",
    )
    args = parser.parse_args()

    device = "cpu" if args.cpu else \
             ("cuda" if torch.cuda.is_available() else "cpu")

    if device == "cuda":
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n  GPU : {gpu}  ({mem:.1f} GB)")

    # Load model
    model, tokenizer, cfg = load_model_and_tokenizer(
        args.ckpt, args.tokenizer, device
    )

    ctx_len = cfg.get("block_size", cfg.get("ctx_len",
              cfg.get("n_positions", 1024)))

    # ── Mode: single prompt ───────────────────────────────────
    if args.prompt:
        output, stats = generate(
            model, tokenizer,
            prompt      = args.prompt,
            max_new     = args.max_new,
            temperature = args.temp,
            top_k       = args.top_k,
            device      = device,
            ctx_len     = ctx_len,
        )
        print(f"\n{'─'*58}")
        print(output)
        print(f"{'─'*58}")
        print(f"  [{stats['generated_tokens']} tokens | "
              f"{stats['latency_s']}s | "
              f"{stats['tokens_per_sec']} tok/s]\n")
        return

    # ── Mode: test suite ──────────────────────────────────────
    if args.suite:
        run_test_suite(
            model, tokenizer, device, cfg,
            max_new = args.max_new,
            temp    = args.temp,
        )
        return

    # ── Mode: interactive REPL (default) ─────────────────────
    interactive_loop(model, tokenizer, device, cfg)


if __name__ == "__main__":
    main()
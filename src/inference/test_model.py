# src/inference/test_model.py
# ─────────────────────────────────────────────────────────────
# Tests model quality after Stage 2 training
# Runs Python and SQL prompts and shows generated output
#
# Run : python src/inference/test_model.py
# Run : python src/inference/test_model.py --checkpoint checkpoints/run_002/best.pt
# ─────────────────────────────────────────────────────────────

import sys
import os
ROOT = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
))
sys.path.insert(0, ROOT)

import torch
import argparse
import math
import time
from src.model.gpt import GPT, GPTConfig

# ── Default checkpoint ────────────────────────────────────────
DEFAULT_CHECKPOINT = os.path.join(
    ROOT, "checkpoints", "run_002", "best.pt"
)

# ── Generation config ─────────────────────────────────────────
MAX_NEW_TOKENS     = 300
TEMPERATURE        = 0.8
TOP_K              = 50
REPETITION_PENALTY = 1.3


# ── Load tokenizer ────────────────────────────────────────────
def load_tokenizer():
    """Load tokenizer from E:\\mini_gpt\\data\\tokenizer\\"""

    # ── HuggingFace tokenizers format ─────────────────────────
    tok_path = os.path.join(ROOT, "data", "tokenizer",
                            "tokenizer.json")

    if os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tok_path)
            print(
                f"  ✓ Tokenizer loaded from "
                f"data/tokenizer/tokenizer.json"
            )
            print(
                f"  ✓ Vocab size : "
                f"{tok.get_vocab_size()}"
            )
            return tok
        except Exception as e:
            print(f"  [!] Tokenizer load failed: {e}")

    raise RuntimeError(
        f"Tokenizer not found at {tok_path}"
    )


def encode(tokenizer, text: str) -> list:
    """Encode text to token ids."""
    try:
        # HuggingFace Tokenizer
        if hasattr(tokenizer, "encode"):
            result = tokenizer.encode(text)
            # HF tokenizers returns Encoding object
            if hasattr(result, "ids"):
                return result.ids
            return list(result)
    except Exception as e:
        print(f"  [encode error] {e}")
    return []


def decode(tokenizer, tokens: list) -> str:
    """Decode token ids to text."""
    try:
        if hasattr(tokenizer, "decode"):
            return tokenizer.decode(tokens)
    except Exception as e:
        print(f"  [decode error] {e}")
    return ""


# ── Load model ────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str):
    print(f"  Loading : {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("config", {})
    step   = ckpt.get("step",   0)
    loss   = ckpt.get("loss",   0.0)
    stage  = ckpt.get("stage",  "?")

    print(f"  Step    : {step:,}")
    print(f"  Stage   : {stage}")
    print(f"  Loss    : {loss:.4f}")

    model = GPT(GPTConfig(
        vocab_size = config.get("vocab_size", 32000),
        block_size = config.get("block_size", 1024),
        n_layer    = config.get("n_layer",    12),
        n_head     = config.get("n_head",     12),
        n_embd     = config.get("n_embd",     768),
        dropout    = 0.0,
        bias       = True,
    )).to(device)

    state = ckpt["model_state"]
    state = {
        k.replace("_orig_mod.", ""): v
        for k, v in state.items()
    }
    model.load_state_dict(state, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params  : {n_params:,}")
    return model, step, loss


# ── Generate ──────────────────────────────────────────────────

@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int   = MAX_NEW_TOKENS,
    temperature: float    = TEMPERATURE,
    top_k: int            = TOP_K,
    repetition_penalty: float = REPETITION_PENALTY,
) -> str:
    tokens = encode(tokenizer, prompt)
    if not tokens:
        return ""

    input_ids = torch.tensor(
        [tokens], dtype=torch.long
    ).to(device)
    generated = list(tokens)

    for _ in range(max_new_tokens):
        # Truncate to block size
        context = input_ids[:, -1024:]

        with torch.amp.autocast(device_type="cuda",
                                enabled=(device == "cuda")):
            logits, _ = model(context)

        logits = logits[:, -1, :]  # last token

        # Repetition penalty
        for tok_id in set(generated[-64:]):
            if logits[0, tok_id] < 0:
                logits[0, tok_id] *= repetition_penalty
            else:
                logits[0, tok_id] /= repetition_penalty

        # Temperature
        logits = logits / temperature

        # Top-k
        if top_k > 0:
            top_k_vals, _ = torch.topk(logits, top_k)
            threshold = top_k_vals[:, -1].unsqueeze(-1)
            logits    = logits.masked_fill(
                logits < threshold, float("-inf")
            )

        probs    = torch.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)

        tok_id = next_tok.item()
        generated.append(tok_id)

        input_ids = torch.cat(
            [input_ids, next_tok], dim=1
        )

        # Stop on common end tokens
        decoded_so_far = decode(tokenizer, generated)
        if decoded_so_far.count("\n\n\n") >= 2:
            break
        if decoded_so_far.endswith("# Example:"):
            break

    full_output = decode(tokenizer, generated)
    # Return only generated part
    return full_output[len(prompt):]


# ── Test prompts ──────────────────────────────────────────────

PYTHON_PROMPTS = [

    (
        "Basic — Hello World",
        "### Instruction\n# Task: Print hello world\n# Solution:\n"
    ),
    (
        "Basic — Add two numbers",
        "### Instruction\n# Task: Write a function to add two numbers\n# Solution:\ndef add("
    ),
    (
        "Intermediate — Binary search",
        "### Instruction\n# Task: Implement binary search\n# Solution:\ndef binary_search(arr, target):\n"
    ),
    (
        "Intermediate — Fibonacci",
        "### Instruction\n# Task: Generate fibonacci sequence\n# Solution:\ndef fibonacci(n):\n"
    ),
    (
        "Advanced — Two Sum",
        "### Instruction\n# Task: Solve two sum problem using hash map\n# Solution:\ndef two_sum(nums, target):\n    seen = {}\n"
    ),
    (
        "Advanced — Quicksort",
        "### Instruction\n# Task: Implement quicksort algorithm\n# Solution:\ndef quicksort(arr):\n"
    ),
    (
        "OOP — Stack class",
        "### Instruction\n# Task: Implement a Stack class\n# Solution:\nclass Stack:\n    def __init__(self):\n"
    ),
    (
        "Decorator pattern",
        "### Instruction\n# Task: Write a timer decorator\n# Solution:\nimport time\nimport functools\n\ndef timer(func):\n"
    ),
]

SQL_PROMPTS = [

    (
        "Basic — SELECT",
        "### SQL Query\n-- Question: Get all users from users table\n-- Answer:\n"
    ),
    (
        "Basic — WHERE",
        "### SQL Query\n-- Question: Find users where age > 30\n-- Answer:\nSELECT * FROM users\n"
    ),
    (
        "Intermediate — GROUP BY",
        "### SQL Query\n-- Question: Count orders per user\n-- Answer:\nSELECT user_id,\n"
    ),
    (
        "Intermediate — JOIN",
        "### SQL Query\n-- Question: Join users with orders\n-- Answer:\nSELECT u.name, o.amount\nFROM users u\n"
    ),
    (
        "Advanced — CTE",
        "### SQL Query\n-- Question: Find top 5 customers by revenue using CTE\n-- Answer:\nWITH ranked AS (\n"
    ),
    (
        "Advanced — Window function",
        "### SQL Query\n-- Question: Calculate running total of sales\n-- Answer:\nSELECT id, amount,\n    SUM(amount) OVER (\n"
    ),
]


# ── Evaluation metrics ────────────────────────────────────────

def score_output(prompt: str, output: str,
                 lang: str) -> dict:
    """Simple quality scoring."""
    scores = {}

    # Length score
    scores["length"]     = min(len(output) / 200, 1.0)

    # Not empty
    scores["not_empty"]  = 1.0 if len(output.strip()) > 10 else 0.0

    # No repetition
    lines  = output.strip().split("\n")
    unique = len(set(lines))
    scores["no_repeat"]  = unique / max(len(lines), 1)

    if lang == "python":
        # Has Python syntax
        has_code = any(kw in output for kw in [
            "return", "def ", "if ", "for ",
            "while ", "print(", ":", "    "
        ])
        scores["has_code"] = 1.0 if has_code else 0.0

        # Indentation
        has_indent = "    " in output or "\t" in output
        scores["indented"] = 1.0 if has_indent else 0.0

    elif lang == "sql":
        # Has SQL keywords
        upper = output.upper()
        has_sql = any(kw in upper for kw in [
            "SELECT", "FROM", "WHERE", "JOIN",
            "GROUP", "ORDER", "HAVING", "WITH"
        ])
        scores["has_sql"] = 1.0 if has_sql else 0.0

        # Proper structure
        has_struct = "FROM" in upper or "WHERE" in upper
        scores["structured"] = 1.0 if has_struct else 0.0

    overall = sum(scores.values()) / len(scores)
    return scores, overall


# ── Main test runner ──────────────────────────────────────────

def run_tests(model, tokenizer, device: str,
              checkpoint_path: str):

    print(f"\n{'═'*65}")
    print(f"  MODEL QUALITY TEST — Stage 2")
    print(f"{'═'*65}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Device     : {device}")
    print(f"  Temperature: {TEMPERATURE}")
    print(f"  Top-k      : {TOP_K}")
    print(f"{'─'*65}\n")

    python_scores = []
    sql_scores    = []

    # ── Python tests ───────────────────────────────────────────
    print(f"{'═'*65}")
    print(f"  PYTHON TESTS ({len(PYTHON_PROMPTS)} prompts)")
    print(f"{'═'*65}")

    for name, prompt in PYTHON_PROMPTS:
        print(f"\n  ── {name} {'─'*(50-len(name))}")
        print(f"  Prompt: {repr(prompt[:60])}...")

        t0     = time.time()
        output = generate(model, tokenizer, prompt, device)
        elapsed = time.time() - t0

        scores, overall = score_output(prompt, output, "python")

        print(f"\n  Generated output:")
        print(f"  {'─'*55}")
        # Show first 400 chars
        display = (prompt + output)[:400]
        for line in display.split("\n"):
            print(f"  {line}")
        if len(prompt + output) > 400:
            print(f"  ... ({len(output)} chars generated)")

        print(f"\n  Quality score : {overall*100:.0f}/100")
        print(f"  Time          : {elapsed:.1f}s")
        print(f"  Scores        : {scores}")

        python_scores.append(overall)

    # ── SQL tests ──────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"  SQL TESTS ({len(SQL_PROMPTS)} prompts)")
    print(f"{'═'*65}")

    for name, prompt in SQL_PROMPTS:
        print(f"\n  ── {name} {'─'*(50-len(name))}")

        t0     = time.time()
        output = generate(model, tokenizer, prompt, device)
        elapsed = time.time() - t0

        scores, overall = score_output(prompt, output, "sql")

        print(f"\n  Generated output:")
        print(f"  {'─'*55}")
        display = (prompt + output)[:400]
        for line in display.split("\n"):
            print(f"  {line}")
        if len(prompt + output) > 400:
            print(f"  ... ({len(output)} chars)")

        print(f"\n  Quality score : {overall*100:.0f}/100")
        print(f"  Time          : {elapsed:.1f}s")

        sql_scores.append(overall)

    # ── Summary ────────────────────────────────────────────────
    avg_python = sum(python_scores) / len(python_scores)
    avg_sql    = sum(sql_scores) / len(sql_scores)
    avg_overall = (avg_python + avg_sql) / 2

    print(f"\n{'═'*65}")
    print(f"  QUALITY SUMMARY")
    print(f"{'═'*65}")
    print(
        f"  Python score  : "
        f"{avg_python*100:.1f}/100  "
        f"({len(python_scores)} tests)"
    )
    print(
        f"  SQL score     : "
        f"{avg_sql*100:.1f}/100  "
        f"({len(sql_scores)} tests)"
    )
    print(f"  Overall score : {avg_overall*100:.1f}/100")
    print(f"\n  Interpretation:")

    if avg_overall >= 0.8:
        print(f"  ✅ EXCELLENT — Model is performing very well!")
        print(f"  ✅ Ready to proceed to Stage 3")
    elif avg_overall >= 0.6:
        print(f"  ✅ GOOD — Model is performing well")
        print(f"  ✅ Ready for Stage 3")
    elif avg_overall >= 0.4:
        print(f"  ⚠️  FAIR — Model has basic capability")
        print(f"  ⚠️  Stage 3 will improve it significantly")
    else:
        print(f"  ❌ POOR — Something may be wrong")
        print(f"  ❌ Check checkpoint and tokenizer")

    print(f"{'═'*65}\n")

    return avg_overall


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test model quality after training"
    )
    parser.add_argument(
        "--checkpoint",
        type    = str,
        default = DEFAULT_CHECKPOINT,
        help    = "Path to checkpoint file",
    )
    parser.add_argument(
        "--temperature",
        type    = float,
        default = TEMPERATURE,
        help    = f"Generation temperature (default: {TEMPERATURE})",
    )
    parser.add_argument(
        "--top_k",
        type    = int,
        default = TOP_K,
        help    = f"Top-k sampling (default: {TOP_K})",
    )
    parser.add_argument(
        "--max_tokens",
        type    = int,
        default = MAX_NEW_TOKENS,
        help    = f"Max new tokens (default: {MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--prompt",
        type    = str,
        default = None,
        help    = "Custom prompt to test",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device : {device}")

    # Load tokenizer and model
    print(f"\n  Loading tokenizer...")
    tokenizer = load_tokenizer()

    print(f"\n  Loading model...")
    model, step, loss = load_model(
        args.checkpoint, device
    )

    # Custom prompt mode
    if args.prompt:
        print(f"\n{'═'*65}")
        print(f"  CUSTOM PROMPT TEST")
        print(f"{'═'*65}")
        print(f"  Prompt: {args.prompt}")
        print(f"{'─'*65}\n")
        output = generate(
            model, tokenizer,
            args.prompt, device,
            max_new_tokens = args.max_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
        )
        print(args.prompt + output)
    else:
        # Full test suite
        run_tests(model, tokenizer, device, args.checkpoint)
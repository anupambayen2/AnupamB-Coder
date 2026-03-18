# src/inference/generator.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import torch
import glob
from tokenizers import Tokenizer
from paths import TOKENIZER_PATH
from src.model.gpt import GPT, GPTConfig


# ── Checkpoint directory ──────────────────────────────────────
# smoke_test  = 5K steps  — quick test
# run_001     = stage 1   — 100K steps on 535K examples
# run_002     = stage 2   — 22GB new data
# run_003     = stage 3   — stack_dedup 60GB
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "smoke_test")


# ── Load model ────────────────────────────────────────────────
def load_model(checkpoint_path: str = None, device: str = None):
    """
    Load trained GPT model from checkpoint.
    Auto-reads config from checkpoint so it works
    across all model sizes — smoke_test, run_001, etc.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-find best checkpoint
    if checkpoint_path is None:
        best = os.path.join(CHECKPOINT_DIR, "best.pt")
        if os.path.exists(best):
            checkpoint_path = best
        else:
            ckpts = sorted(glob.glob(
                os.path.join(CHECKPOINT_DIR, "step_*.pt")
            ))
            if not ckpts:
                raise FileNotFoundError(
                    f"No checkpoints in {CHECKPOINT_DIR}"
                )
            checkpoint_path = ckpts[-1]

    print(f"  Loading   : {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Read architecture from checkpoint
    # Falls back to 110M defaults if not stored
    saved = ckpt.get("config", {})

    config = GPTConfig(
        vocab_size = saved.get("vocab_size", 32000),
        block_size = saved.get("block_size", 1024),
        n_layer    = saved.get("n_layer",    12),
        n_head     = saved.get("n_head",     12),
        n_embd     = saved.get("n_embd",     768),
        dropout    = 0.0,   # always 0 at inference
        bias       = True,
    )

    model = GPT(config)

    # Strip _orig_mod prefix if torch.compile was used
    state = ckpt["model_state"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()

    # Print summary
    stage = ckpt.get("stage", "smoke_test")
    step  = ckpt.get("step",  0)
    loss  = ckpt.get("loss",  0.0)

    print(f"  Parameters: {model.get_num_params():,}")
    print(f"  Stage     : {stage}")
    print(f"  Step      : {step:,}" if isinstance(step, int)
          else f"  Step      : {step}")
    print(f"  Val loss  : {loss:.4f}")
    print(f"  Device    : {device}")
    print(f"  Config    : {config.n_layer}L · "
          f"{config.n_embd}D · "
          f"{config.n_head}H · "
          f"{config.block_size} ctx")

    return model, device


# ── Load tokenizer ────────────────────────────────────────────
def load_tokenizer():
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    return tok


# ── Core generation ───────────────────────────────────────────
@torch.no_grad()
def generate(
    model,
    tokenizer,
    prompt             : str,
    max_tokens         : int   = 250,
    temperature        : float = 0.8,
    top_k              : int   = 40,
    repetition_penalty : float = 1.3,
    device             : str   = "cuda",
) -> str:
    """
    Generate text from a prompt.

    Args:
        model              : loaded GPT model
        tokenizer          : loaded tokenizer
        prompt             : input text
        max_tokens         : max new tokens to generate
        temperature        : 0.1 focused ↔ 1.5 creative
        top_k              : sample from top k tokens only
        repetition_penalty : 1.0 off, 1.3 moderate, 1.5 strong
        device             : cuda or cpu

    Returns:
        full generated text string
    """
    model.eval()

    # Encode
    enc = tokenizer.encode(prompt)
    ids = enc.ids

    # Strip BOS/EOS that post-processor adds automatically
    # Leaving EOS in the prompt causes immediate generation stop
    bos_id = tokenizer.token_to_id("<|bos|>")
    eos_id = tokenizer.token_to_id("<|eos|>")

    if ids and ids[0] == bos_id:
        ids = ids[1:]
    if ids and ids[-1] == eos_id:
        ids = ids[:-1]

    idx = torch.tensor([ids], dtype=torch.long).to(device)

    # Generate
    out = model.generate(
        idx,
        max_new_tokens     = max_tokens,
        temperature        = temperature,
        top_k              = top_k,
        repetition_penalty = repetition_penalty,
    )

    # Decode — stop at first EOS
    all_ids = out[0].tolist()
    if eos_id in all_ids:
        all_ids = all_ids[:all_ids.index(eos_id)]

    return tokenizer.decode(all_ids)


# ── Task helpers ──────────────────────────────────────────────

def generate_python(
    model,
    tokenizer,
    description        : str,
    device             : str,
    temperature        : float = 0.7,
    max_tokens         : int   = 300,
    repetition_penalty : float = 1.3,
) -> str:
    """Generate a Python function from plain English."""
    prompt = (
        f"### Python Function\n"
        f"# Description: {description}\n"
        f"def"
    )
    return generate(
        model, tokenizer, prompt,
        max_tokens         = max_tokens,
        temperature        = temperature,
        top_k              = 40,
        repetition_penalty = repetition_penalty,
        device             = device,
    )


def generate_sql(
    model,
    tokenizer,
    question           : str,
    context            : str,
    device             : str,
    temperature        : float = 0.3,
    max_tokens         : int   = 200,
    repetition_penalty : float = 1.2,
) -> str:
    """Generate SQL from plain English + schema."""
    prompt = (
        f"### SQL Query\n"
        f"-- Question : {question}\n"
        f"-- Context  : {context}\n"
        f"-- Answer   :\n"
    )
    return generate(
        model, tokenizer, prompt,
        max_tokens         = max_tokens,
        temperature        = temperature,
        top_k              = 20,
        repetition_penalty = repetition_penalty,
        device             = device,
    )


def generate_completion(
    model,
    tokenizer,
    code_prefix        : str,
    device             : str,
    temperature        : float = 0.7,
    max_tokens         : int   = 250,
    repetition_penalty : float = 1.3,
) -> str:
    """Complete a partial code snippet."""
    return generate(
        model, tokenizer, code_prefix,
        max_tokens         = max_tokens,
        temperature        = temperature,
        top_k              = 40,
        repetition_penalty = repetition_penalty,
        device             = device,
    )


def generate_code(
    model,
    tokenizer,
    task               : str,
    device             : str,
    temperature        : float = 0.7,
    max_tokens         : int   = 300,
    repetition_penalty : float = 1.3,
) -> str:
    """
    Auto-detect Python vs SQL and generate accordingly.
    """
    sql_keywords = [
        "select", "insert", "update", "delete",
        "table", "database", "query", "sql",
        "join", "where", "group by", "order by",
    ]
    if any(kw in task.lower() for kw in sql_keywords):
        prompt = (
            f"### SQL Query\n"
            f"-- Question : {task}\n"
            f"-- Answer   :\n"
        )
        return generate(
            model, tokenizer, prompt,
            max_tokens         = max_tokens,
            temperature        = temperature,
            top_k              = 20,
            repetition_penalty = repetition_penalty,
            device             = device,
        )
    return generate_python(
        model, tokenizer, task, device,
        temperature        = temperature,
        max_tokens         = max_tokens,
        repetition_penalty = repetition_penalty,
    )


# ── Full test ─────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'═'*65}")
    print(f"  MINIGPT INFERENCE TEST — 110M MODEL")
    print(f"{'═'*65}\n")

    model, device = load_model()
    tokenizer     = load_tokenizer()

    tests = [
        # (label, type, args)
        (
            "Python — factorial recursive",
            "python",
            {
                "description" : "calculate the factorial of a number recursively",
                "temperature" : 0.6,
                "max_tokens"  : 200,
            }
        ),
        (
            "Python — sieve of eratosthenes",
            "python",
            {
                "description" : "find all prime numbers up to n using sieve of eratosthenes",
                "temperature" : 0.5,
                "max_tokens"  : 250,
            }
        ),
        (
            "Python — binary search",
            "python",
            {
                "description" : "implement binary search on a sorted list",
                "temperature" : 0.4,
                "max_tokens"  : 200,
            }
        ),
        (
            "Python — reverse a string",
            "python",
            {
                "description" : "reverse a string",
                "temperature" : 0.5,
                "max_tokens"  : 150,
            }
        ),
        (
            "SQL — basic WHERE",
            "sql",
            {
                "question"    : "find all users older than 30",
                "context"     : "CREATE TABLE users (id INT, name TEXT, age INT)",
                "temperature" : 0.3,
                "max_tokens"  : 150,
            }
        ),
        (
            "SQL — JOIN with aggregation",
            "sql",
            {
                "question"    : "get total sales amount per customer name",
                "context"     : (
                    "CREATE TABLE customers (id INT, name TEXT); "
                    "CREATE TABLE orders (id INT, customer_id INT, amount FLOAT)"
                ),
                "temperature" : 0.3,
                "max_tokens"  : 200,
            }
        ),
        (
            "Code completion — binary search",
            "completion",
            {
                "code_prefix" : (
                    "def binary_search(arr, target):\n"
                    "    left, right = 0, len(arr) - 1\n"
                    "    while left <="
                ),
                "temperature" : 0.4,
                "max_tokens"  : 200,
            }
        ),
        (
            "Code completion — quicksort",
            "completion",
            {
                "code_prefix" : (
                    "def quicksort(arr):\n"
                    "    if len(arr) <= 1:\n"
                    "        return arr\n"
                    "    pivot ="
                ),
                "temperature" : 0.4,
                "max_tokens"  : 200,
            }
        ),
    ]

    for label, kind, kwargs in tests:
        print(f"\n{'─'*65}")
        print(f"  {label}")
        print(f"{'─'*65}")

        if kind == "python":
            print(f"  Description : {kwargs['description']}\n")
            result = generate_python(
                model, tokenizer,
                kwargs["description"], device,
                temperature        = kwargs.get("temperature", 0.7),
                max_tokens         = kwargs.get("max_tokens",  250),
                repetition_penalty = 1.3,
            )
        elif kind == "sql":
            print(f"  Question : {kwargs['question']}")
            print(f"  Context  : {kwargs['context']}\n")
            result = generate_sql(
                model, tokenizer,
                kwargs["question"],
                kwargs["context"],
                device,
                temperature        = kwargs.get("temperature", 0.3),
                max_tokens         = kwargs.get("max_tokens",  200),
                repetition_penalty = 1.2,
            )
        elif kind == "completion":
            print(f"  Prefix :\n{kwargs['code_prefix']}\n")
            result = generate_completion(
                model, tokenizer,
                kwargs["code_prefix"], device,
                temperature        = kwargs.get("temperature", 0.5),
                max_tokens         = kwargs.get("max_tokens",  200),
                repetition_penalty = 1.3,
            )
        else:
            continue

        print(result)

    print(f"\n{'═'*65}")
    print(f"  All tests complete!")
    print(f"{'═'*65}\n")
# src/inference/ui.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import gradio as gr
from src.inference.generator import (
    load_model,
    load_tokenizer,
    generate_python,
    generate_sql,
    generate_completion,
    generate_code,
)

# ── Load model once at startup ────────────────────────────────
print("\n  Loading AnupamB-Coder-110M...")
model, device = load_model()
tokenizer     = load_tokenizer()
print("  Model ready!\n")


# ── Generation functions for Gradio ──────────────────────────

def run_python(description, temperature, max_tokens, rep_penalty):
    if not description.strip():
        return "Please enter a description."
    try:
        return generate_python(
            model, tokenizer,
            description        = description.strip(),
            device             = device,
            temperature        = float(temperature),
            max_tokens         = int(max_tokens),
            repetition_penalty = float(rep_penalty),
        )
    except Exception as e:
        return f"Error: {e}"


def run_sql(question, context, temperature, max_tokens):
    if not question.strip():
        return "Please enter a question."
    try:
        return generate_sql(
            model, tokenizer,
            question    = question.strip(),
            context     = context.strip(),
            device      = device,
            temperature = float(temperature),
            max_tokens  = int(max_tokens),
        )
    except Exception as e:
        return f"Error: {e}"


def run_completion(prefix, temperature, max_tokens, rep_penalty):
    if not prefix.strip():
        return "Please enter a code prefix."
    try:
        return generate_completion(
            model, tokenizer,
            code_prefix        = prefix,
            device             = device,
            temperature        = float(temperature),
            max_tokens         = int(max_tokens),
            repetition_penalty = float(rep_penalty),
        )
    except Exception as e:
        return f"Error: {e}"


def run_free(prompt, temperature, max_tokens, rep_penalty):
    if not prompt.strip():
        return "Please enter a prompt."
    try:
        return generate_code(
            model, tokenizer,
            task               = prompt.strip(),
            device             = device,
            temperature        = float(temperature),
            max_tokens         = int(max_tokens),
            repetition_penalty = float(rep_penalty),
        )
    except Exception as e:
        return f"Error: {e}"


# ── Model info ────────────────────────────────────────────────

MODEL_INFO = """
## AnupamB-Coder-110M

**Built entirely from scratch — no pretrained weights.**

| Property | Value |
|----------|-------|
| Parameters | 110,418,432 |
| Architecture | GPT decoder-only transformer |
| Layers | 12 |
| Attention heads | 12 |
| Embedding dim | 768 |
| Context length | 1024 tokens |
| Vocabulary | 32,000 (custom BPE) |

---

### Stage 1 Training Results
| Metric | Value |
|--------|-------|
| Steps | 100,000 |
| Best val loss | 1.1349 |
| Perplexity | 3.13 |
| Training data | 535K Python + SQL examples |
| GPU | RTX 4060 Laptop 8GB |
| Training time | ~4 days |

---

### Training Roadmap
| Stage | Data | Steps | Status |
|-------|------|-------|--------|
| 1 | 535K examples | 100K | ✅ Done |
| 2 | 5.3M examples 13GB | 350K | 🔄 Next |
| 3 | Problem solving | 100K | ⏳ Planned |
| 4 | SQL mastery | 60K | ⏳ Planned |
| 5 | Instruction FT | 20K | ⏳ Planned |

---

### Links
- GitHub : [anupambayen2/AnupamB-Coder](https://github.com/anupambayen2/AnupamB-Coder)
- HuggingFace : [anupambayen/AnupamB-Coder-110M](https://huggingface.co/anupambayen/AnupamB-Coder-110M)

*Built by Anupam Bayen — Tamil Nadu, India*
"""

# ── Example prompts ───────────────────────────────────────────

PYTHON_EXAMPLES = [
    ["calculate factorial of a number recursively", 0.7, 250, 1.3],
    ["find all prime numbers up to n using sieve of eratosthenes", 0.5, 300, 1.3],
    ["implement binary search on a sorted list", 0.5, 250, 1.3],
    ["reverse a linked list", 0.6, 250, 1.3],
    ["merge two sorted arrays", 0.6, 300, 1.3],
    ["sort a list of dictionaries by a key", 0.5, 200, 1.3],
    ["check if a string is a palindrome", 0.5, 200, 1.3],
    ["implement a stack using a list", 0.6, 250, 1.3],
]

SQL_EXAMPLES = [
    [
        "find all users older than 30",
        "CREATE TABLE users (id INT, name TEXT, age INT)",
        0.3, 150,
    ],
    [
        "get total sales amount per customer name",
        "CREATE TABLE customers (id INT, name TEXT); CREATE TABLE orders (id INT, customer_id INT, amount FLOAT)",
        0.3, 200,
    ],
    [
        "find the top 5 most expensive products",
        "CREATE TABLE products (id INT, name TEXT, price FLOAT, category TEXT)",
        0.3, 150,
    ],
    [
        "count number of orders per month",
        "CREATE TABLE orders (id INT, customer_id INT, amount FLOAT, order_date DATE)",
        0.3, 150,
    ],
]

COMPLETION_EXAMPLES = [
    [
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
        0.5, 250, 1.3,
    ],
    [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
        0.5, 150, 1.3,
    ],
    [
        "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):",
        0.6, 250, 1.3,
    ],
    [
        "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid =",
        0.5, 300, 1.3,
    ],
]


# ── Build Gradio UI ───────────────────────────────────────────

def build_ui():
    with gr.Blocks(
        title = "AnupamB-Coder-110M",
    ) as demo:

        # ── Header ────────────────────────────────────────────
        gr.Markdown("""
# 🤖 AnupamB-Coder-110M
### GPT Language Model — Built From Scratch · No Pretrained Weights
*110M parameters · Custom BPE tokenizer · Trained on Python + SQL · RTX 4060 8GB*
        """)

        # ── Tabs ──────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Python Generator ───────────────────────
            with gr.Tab("🐍 Python Generator"):
                gr.Markdown("### Generate Python functions from plain English")

                with gr.Row():
                    with gr.Column(scale=2):
                        py_desc = gr.Textbox(
                            label       = "Description",
                            placeholder = "e.g. calculate factorial of a number recursively",
                            lines       = 3,
                        )
                        with gr.Row():
                            py_temp    = gr.Slider(
                                0.1, 1.5, value=0.7, step=0.1,
                                label = "Temperature",
                            )
                            py_tokens  = gr.Slider(
                                50, 500, value=250, step=50,
                                label = "Max tokens",
                            )
                            py_penalty = gr.Slider(
                                1.0, 2.0, value=1.3, step=0.1,
                                label = "Repetition penalty",
                            )
                        py_btn = gr.Button(
                            "Generate Python",
                            variant = "primary",
                        )

                    with gr.Column(scale=3):
                        py_output = gr.Code(
                            label    = "Generated Code",
                            language = "python",
                            lines    = 20,
                        )

                gr.Examples(
                    examples        = PYTHON_EXAMPLES,
                    inputs          = [py_desc, py_temp, py_tokens, py_penalty],
                    label           = "Quick examples",
                    examples_per_page = 4,
                )

                py_btn.click(
                    fn      = run_python,
                    inputs  = [py_desc, py_temp, py_tokens, py_penalty],
                    outputs = py_output,
                )

            # ── Tab 2: SQL Generator ──────────────────────────
            with gr.Tab("🗄️ SQL Generator"):
                gr.Markdown("### Generate SQL queries from plain English + schema")

                with gr.Row():
                    with gr.Column(scale=2):
                        sql_question = gr.Textbox(
                            label       = "Question",
                            placeholder = "e.g. find all users older than 30",
                            lines       = 2,
                        )
                        sql_context = gr.Textbox(
                            label       = "Table schema (context)",
                            placeholder = "CREATE TABLE users (id INT, name TEXT, age INT)",
                            lines       = 3,
                        )
                        with gr.Row():
                            sql_temp   = gr.Slider(
                                0.1, 1.0, value=0.3, step=0.1,
                                label = "Temperature",
                            )
                            sql_tokens = gr.Slider(
                                50, 300, value=150, step=50,
                                label = "Max tokens",
                            )
                        sql_btn = gr.Button(
                            "Generate SQL",
                            variant = "primary",
                        )

                    with gr.Column(scale=3):
                        sql_output = gr.Code(
                            label    = "Generated SQL",
                            language = "sql",
                            lines    = 15,
                        )

                gr.Examples(
                    examples        = SQL_EXAMPLES,
                    inputs          = [sql_question, sql_context, sql_temp, sql_tokens],
                    label           = "Quick examples",
                    examples_per_page = 4,
                )

                sql_btn.click(
                    fn      = run_sql,
                    inputs  = [sql_question, sql_context, sql_temp, sql_tokens],
                    outputs = sql_output,
                )

            # ── Tab 3: Code Completion ────────────────────────
            with gr.Tab("⚡ Code Completion"):
                gr.Markdown("### Complete partial code snippets")

                with gr.Row():
                    with gr.Column(scale=2):
                        comp_prefix = gr.Code(
                            label    = "Code prefix",
                            language = "python",
                            lines    = 8,
                            value    = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
                        )
                        with gr.Row():
                            comp_temp    = gr.Slider(
                                0.1, 1.5, value=0.5, step=0.1,
                                label = "Temperature",
                            )
                            comp_tokens  = gr.Slider(
                                50, 500, value=250, step=50,
                                label = "Max tokens",
                            )
                            comp_penalty = gr.Slider(
                                1.0, 2.0, value=1.3, step=0.1,
                                label = "Repetition penalty",
                            )
                        comp_btn = gr.Button(
                            "Complete Code",
                            variant = "primary",
                        )

                    with gr.Column(scale=3):
                        comp_output = gr.Code(
                            label    = "Completed Code",
                            language = "python",
                            lines    = 20,
                        )

                gr.Examples(
                    examples        = COMPLETION_EXAMPLES,
                    inputs          = [comp_prefix, comp_temp, comp_tokens, comp_penalty],
                    label           = "Quick examples",
                    examples_per_page = 4,
                )

                comp_btn.click(
                    fn      = run_completion,
                    inputs  = [comp_prefix, comp_temp, comp_tokens, comp_penalty],
                    outputs = comp_output,
                )

            # ── Tab 4: Free Generation ────────────────────────
            with gr.Tab("✨ Free Generation"):
                gr.Markdown(
                    "### Auto-detects Python vs SQL and generates accordingly"
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        free_prompt = gr.Textbox(
                            label       = "Prompt",
                            placeholder = "e.g. write a function to merge two sorted lists",
                            lines       = 4,
                        )
                        with gr.Row():
                            free_temp    = gr.Slider(
                                0.1, 1.5, value=0.7, step=0.1,
                                label = "Temperature",
                            )
                            free_tokens  = gr.Slider(
                                50, 500, value=300, step=50,
                                label = "Max tokens",
                            )
                            free_penalty = gr.Slider(
                                1.0, 2.0, value=1.3, step=0.1,
                                label = "Repetition penalty",
                            )
                        free_btn = gr.Button(
                            "Generate",
                            variant = "primary",
                        )

                    with gr.Column(scale=3):
                        free_output = gr.Code(
                            label    = "Generated Output",
                            language = "python",
                            lines    = 20,
                        )

                free_btn.click(
                    fn      = run_free,
                    inputs  = [free_prompt, free_temp, free_tokens, free_penalty],
                    outputs = free_output,
                )

            # ── Tab 5: Model Info ─────────────────────────────
            with gr.Tab("ℹ️ Model Info"):
                gr.Markdown(MODEL_INFO)

                gr.Markdown("""
---
### Temperature guide
| Value | Effect |
|-------|--------|
| 0.1 – 0.3 | Very focused — deterministic, safe |
| 0.4 – 0.7 | Balanced — recommended for most tasks |
| 0.8 – 1.2 | Creative — more variety, some risk |
| 1.3 – 1.5 | Experimental — unpredictable |

### Repetition penalty guide
| Value | Effect |
|-------|--------|
| 1.0 | Off — may produce loops |
| 1.2 | Light — reduces mild repetition |
| 1.3 | Recommended — good balance |
| 1.5 | Strong — very diverse output |
                """)

        # ── Footer ────────────────────────────────────────────
        gr.Markdown("""
---
*AnupamB-Coder-110M · Built from scratch by Anupam Bayen · Tamil Nadu, India*
*Stage 1 complete · val_loss 1.1349 · perplexity 3.13 · 100K steps*
*[GitHub](https://github.com/anupambayen2/AnupamB-Coder) · [HuggingFace](https://huggingface.co/anupambayen/AnupamB-Coder-110M)*
        """)

    return demo


# ── Launch ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n  Starting AnupamB-Coder-110M Gradio UI...")
    print("  Open: http://localhost:7860\n")

    demo = build_ui()
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = False,
        show_error  = True,
    )
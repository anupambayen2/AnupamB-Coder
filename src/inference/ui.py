# src/inference/ui.py

import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import time
import gradio as gr
from src.inference.generator import (
    load_model,
    load_tokenizer,
    generate_python,
    generate_sql,
    generate_completion,
)

# ── Load model once at startup ────────────────────────────────
print("\n  Loading MiniGPT...")
model, device = load_model()
tokenizer     = load_tokenizer()
print("  Ready!\n")


# ── Tab 1: Python generator ───────────────────────────────────
def run_python(description, max_tokens, temperature):
    if not description.strip():
        return "Please enter a description."
    t0     = time.time()
    result = generate_python(
        model, tokenizer, description.strip(),
        device,
        temperature = float(temperature),
        max_tokens  = int(max_tokens),
    )
    dt = (time.time() - t0) * 1000
    return f"{result}\n\n# Generated in {dt:.0f}ms"


# ── Tab 2: SQL generator ──────────────────────────────────────
def run_sql(question, context, max_tokens, temperature):
    if not question.strip():
        return "Please enter a question."
    if not context.strip():
        return "Please enter a table schema."
    t0     = time.time()
    result = generate_sql(
        model, tokenizer,
        question.strip(),
        context.strip(),
        device,
        temperature = float(temperature),
        max_tokens  = int(max_tokens),
    )
    dt = (time.time() - t0) * 1000
    return f"{result}\n\n-- Generated in {dt:.0f}ms"


# ── Tab 3: Code completion ────────────────────────────────────
def run_completion(prompt, max_tokens, temperature):
    if not prompt.strip():
        return "Please enter some code to complete."
    t0     = time.time()
    result = generate_completion(
        model, tokenizer,
        prompt.strip(),
        device,
        temperature = float(temperature),
        max_tokens  = int(max_tokens),
    )
    dt = (time.time() - t0) * 1000
    return f"{result}\n\n# Generated in {dt:.0f}ms"


# ── Example data ──────────────────────────────────────────────
python_examples = [
    ["calculate the fibonacci sequence up to n", 200, 0.7],
    ["sort a dictionary by its values", 150, 0.7],
    ["read a csv file and return a list of rows", 200, 0.7],
    ["find the largest number in a list", 100, 0.5],
    ["check if a string is a palindrome", 100, 0.5],
    ["connect to a sqlite database and fetch all rows", 200, 0.7],
]

sql_examples = [
    [
        "find all users older than 25",
        "CREATE TABLE users (id INT, name TEXT, age INT)",
        150, 0.4,
    ],
    [
        "get total sales per product",
        "CREATE TABLE sales (id INT, product TEXT, amount FLOAT)",
        150, 0.4,
    ],
    [
        "find employees with salary above 50000",
        "CREATE TABLE employees (id INT, name TEXT, salary FLOAT, dept TEXT)",
        150, 0.4,
    ],
    [
        "get all orders with customer names",
        "CREATE TABLE orders (id INT, customer_id INT, amount FLOAT); CREATE TABLE customers (id INT, name TEXT)",
        200, 0.4,
    ],
]

completion_examples = [
    ["def merge_sort(arr):", 200, 0.7],
    ["def binary_search(arr, target):", 150, 0.7],
    ["class Stack:\n    def __init__(self):", 200, 0.7],
    ["SELECT * FROM orders WHERE", 100, 0.4],
    ["def connect_to_db(host, port, db_name):", 150, 0.7],
]


# ── Build UI ──────────────────────────────────────────────────
with gr.Blocks(title="MiniGPT — Built from Scratch") as demo:

    # ── Header ────────────────────────────────────────────────
    gr.Markdown("""
    # MiniGPT — Built from Scratch
    **35M parameter GPT** trained on Python + SQL · RTX 4060 8GB · No pretrained weights
    `val_loss: 1.81 · perplexity: 6.13 · 50,000 steps`
    """)

    with gr.Tabs():

        # ── Tab 1: Python ─────────────────────────────────────
        with gr.TabItem("Python Generator"):
            gr.Markdown("### Generate a Python function from a plain English description")

            with gr.Row():
                with gr.Column(scale=2):
                    py_desc = gr.Textbox(
                        label       = "Function description",
                        placeholder = "e.g. calculate the fibonacci sequence up to n",
                        lines       = 3,
                    )
                    py_tokens = gr.Slider(
                        minimum = 50,
                        maximum = 500,
                        value   = 200,
                        step    = 10,
                        label   = "Max tokens",
                    )
                    py_temp = gr.Slider(
                        minimum = 0.1,
                        maximum = 1.5,
                        value   = 0.7,
                        step    = 0.1,
                        label   = "Temperature  (lower = focused, higher = creative)",
                    )
                    py_btn = gr.Button(
                        "Generate Python Function",
                        variant = "primary",
                    )

                with gr.Column(scale=3):
                    py_out = gr.Code(
                        label    = "Generated code",
                        language = "python",
                        lines    = 22,
                    )

            py_btn.click(
                fn      = run_python,
                inputs  = [py_desc, py_tokens, py_temp],
                outputs = py_out,
            )
            py_desc.submit(
                fn      = run_python,
                inputs  = [py_desc, py_tokens, py_temp],
                outputs = py_out,
            )

            gr.Examples(
                examples       = python_examples,
                inputs         = [py_desc, py_tokens, py_temp],
                outputs        = py_out,
                fn             = run_python,
                cache_examples = False,
                label          = "Try these examples — click any row",
            )

        # ── Tab 2: SQL ────────────────────────────────────────
        with gr.TabItem("SQL Generator"):
            gr.Markdown("### Generate SQL queries from plain English + table schema")

            with gr.Row():
                with gr.Column(scale=2):
                    sql_q = gr.Textbox(
                        label       = "Question",
                        placeholder = "e.g. find all users older than 30",
                        lines       = 2,
                    )
                    sql_ctx = gr.Textbox(
                        label       = "Table schema  (CREATE TABLE statements)",
                        placeholder = "CREATE TABLE users (id INT, name TEXT, age INT)",
                        lines       = 4,
                    )
                    sql_tokens = gr.Slider(
                        minimum = 50,
                        maximum = 400,
                        value   = 150,
                        step    = 10,
                        label   = "Max tokens",
                    )
                    sql_temp = gr.Slider(
                        minimum = 0.1,
                        maximum = 1.0,
                        value   = 0.4,
                        step    = 0.1,
                        label   = "Temperature  (keep low for SQL accuracy)",
                    )
                    sql_btn = gr.Button(
                        "Generate SQL Query",
                        variant = "primary",
                    )

                with gr.Column(scale=3):
                    sql_out = gr.Code(
                        label    = "Generated SQL",
                        language = "sql",
                        lines    = 22,
                    )

            sql_btn.click(
                fn      = run_sql,
                inputs  = [sql_q, sql_ctx, sql_tokens, sql_temp],
                outputs = sql_out,
            )
            sql_q.submit(
                fn      = run_sql,
                inputs  = [sql_q, sql_ctx, sql_tokens, sql_temp],
                outputs = sql_out,
            )

            gr.Examples(
                examples       = sql_examples,
                inputs         = [sql_q, sql_ctx, sql_tokens, sql_temp],
                outputs        = sql_out,
                fn             = run_sql,
                cache_examples = False,
                label          = "Try these examples — click any row",
            )

        # ── Tab 3: Code completion ────────────────────────────
        with gr.TabItem("Code Completion"):
            gr.Markdown("### Complete any partial Python or SQL code snippet")

            with gr.Row():
                with gr.Column(scale=2):
                    comp_prompt = gr.Textbox(
                        label       = "Code prefix",
                        placeholder = "def binary_search(arr, target):",
                        lines       = 6,
                    )
                    comp_tokens = gr.Slider(
                        minimum = 50,
                        maximum = 500,
                        value   = 150,
                        step    = 10,
                        label   = "Max tokens",
                    )
                    comp_temp = gr.Slider(
                        minimum = 0.1,
                        maximum = 1.5,
                        value   = 0.7,
                        step    = 0.1,
                        label   = "Temperature",
                    )
                    comp_btn = gr.Button(
                        "Complete Code",
                        variant = "primary",
                    )

                with gr.Column(scale=3):
                    comp_out = gr.Code(
                        label    = "Completed code",
                        language = "python",
                        lines    = 22,
                    )

            comp_btn.click(
                fn      = run_completion,
                inputs  = [comp_prompt, comp_tokens, comp_temp],
                outputs = comp_out,
            )
            comp_prompt.submit(
                fn      = run_completion,
                inputs  = [comp_prompt, comp_tokens, comp_temp],
                outputs = comp_out,
            )

            gr.Examples(
                examples       = completion_examples,
                inputs         = [comp_prompt, comp_tokens, comp_temp],
                outputs        = comp_out,
                fn             = run_completion,
                cache_examples = False,
                label          = "Try these examples — click any row",
            )

        # ── Tab 4: Model info ─────────────────────────────────
        with gr.TabItem("Model Info"):
            gr.Markdown("""
            ## MiniGPT — Technical details

            | Property | Value |
            |----------|-------|
            | Architecture | GPT decoder-only transformer |
            | Parameters | 35,561,472 |
            | Layers | 6 transformer blocks |
            | Attention heads | 8 |
            | Embedding dimension | 512 |
            | Vocabulary size | 32,000 (custom BPE) |
            | Context length | 512 tokens |
            | Training data | 535,323 Python + SQL examples |
            | Training steps | 50,000 |
            | Final val loss | 1.8126 |
            | Perplexity | 6.13 |
            | Hardware | NVIDIA RTX 4060 Laptop 8GB |
            | Training time | ~6 hours |
            | Framework | PyTorch — no pretrained weights |

            ## Capabilities
            - Generate Python functions from plain English descriptions
            - Write SQL queries from natural language + table schema
            - Complete partial code snippets

            ## Planned next stages
            | Stage | Data | Purpose |
            |-------|------|---------|
            | Stage 2 | Wikipedia 6M articles | Adds factual knowledge |
            | Stage 3 | ArXiv abstracts | Adds scientific reasoning |
            | Stage 4 | C4 filtered web | Broad language coverage |
            | Final | Instruction pairs | Follow instructions better |

            ## How to resume training
```bash
            python src/training/trainer.py --resume checkpoints/run_001/best.pt
```
            """)

    # ── Footer ────────────────────────────────────────────────
    gr.Markdown("""
    ---
    Built from scratch with PyTorch · Custom BPE tokenizer · No pretrained weights · RTX 4060 8GB
    """)


# ── Launch ────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name = "0.0.0.0",
        server_port = 7860,
        share       = True,
        show_error  = True,
        theme       = gr.themes.Soft(),
    )
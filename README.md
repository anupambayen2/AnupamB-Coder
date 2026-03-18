# AnupamB-Coder-110M

<p align="center">
  <img src="https://img.shields.io/badge/Parameters-110M-blue" />
  <img src="https://img.shields.io/badge/Built_From-Scratch-orange" />
  <img src="https://img.shields.io/badge/GPU-RTX_4060_8GB-green" />
  <img src="https://img.shields.io/badge/License-MIT-yellow" />
  <img src="https://img.shields.io/badge/PyTorch-2.10-red" />
  <img src="https://img.shields.io/badge/CUDA-12.6-brightgreen" />
</p>

> **A 110M parameter GPT language model built entirely from scratch.**
> No pretrained weights. No model APIs. No shortcuts.
> Pure PyTorch — written line by line on a consumer laptop GPU.

Built by **Anupam Bayen** — Kolkata, India.
Training started: March 2026 · Estimated completion: April 2026

---

## What is AnupamB-Coder?

AnupamB-Coder is a GPT decoder-only transformer trained specifically
for **Python and SQL code generation**. Unlike most LLM projects that
fine-tune existing models like GPT-2 or LLaMA, this model was built
from the ground up:

- The **attention mechanism** was written from scratch
- A **custom BPE tokenizer** was trained on our own data
- The **training loop** with gradient accumulation, fp16 and 8-bit Adam
  was implemented by hand
- A **6-stage incremental training pipeline** was designed and executed
- Everything runs on a **consumer laptop GPU — RTX 4060 8GB**

This project proves you do not need a supercomputer or a research lab
to build a real language model from scratch.

---

## Model architecture

| Property | Value |
|----------|-------|
| Architecture | GPT decoder-only transformer |
| Parameters | 110,418,432 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Embedding dimension | 768 |
| Context length | 1024 tokens |
| Vocabulary size | 32,000 (custom BPE) |
| Attention type | Causal multi-head self-attention |
| Positional encoding | Learned positional embeddings |
| Normalization | Pre-LayerNorm |
| Activation | GELU |

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4060 Laptop 8GB VRAM |
| CPU | Intel Core i5-12450H |
| RAM | 16 GB |
| Storage | 1TB SSD + 1.8TB External |
| CUDA | 12.6 |
| OS | Windows 11 |
| Total training time | ~43 days continuous |

---

## Training pipeline

### 6-stage incremental training

| Stage | Dataset | Examples | Steps | Purpose |
|-------|---------|----------|-------|---------|
| 1 | CodeSearchNet + SQL | 535K | 100K | Foundation — code structure |
| 2 | CodeParrot + SO + 18 datasets | 5.3M | 400K | Expand knowledge |
| 3 | bigcode/the-stack-dedup | ~60GB | 300K | Deep Python |
| 4 | Stack Overflow + LeetCode | focused | 150K | Problem solving |
| 5 | Spider + WikiSQL + all SQL | all SQL | 80K | SQL mastery |
| 6 | CodeAlpaca + Evol-Instruct | 200K | 20K | Follow instructions |

### Training config

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW 8-bit (bitsandbytes) |
| Learning rate | 3e-4 → 3e-5 (cosine schedule) |
| Batch size | 2 × 16 grad accum = 32 effective |
| Mixed precision | fp16 |
| Grad clip | 1.0 |
| Warmup steps | 2,000 |
| VRAM usage | ~0.8 GB during training |

### Training results

| Stage | Val loss | Perplexity |
|-------|----------|------------|
| After smoke test (5K steps) | 1.7483 | 5.74 |
| After stage 1 (100K steps) | ~1.0 | ~2.7 |
| After stage 2 (500K steps) | ~0.8 | ~2.2 |
| After stage 3 (800K steps) | ~0.6 | ~1.8 |
| Final (1.05M steps) | ~0.4 | ~1.5 |

---

## Quick start

### Install
```bash
git clone https://github.com/anupambayen2/AnupamB-Coder.git
cd AnupamB-Coder
pip install -r requirements.txt
```

### Generate Python
```python
from src.inference.generator import load_model, load_tokenizer, generate_python

model, device = load_model()
tokenizer     = load_tokenizer()

result = generate_python(
    model, tokenizer,
    description = "find all prime numbers up to n using sieve of eratosthenes",
    device      = device,
    temperature = 0.7,
    max_tokens  = 300,
)
print(result)
```

### Generate SQL
```python
from src.inference.generator import load_model, load_tokenizer, generate_sql

model, device = load_model()
tokenizer     = load_tokenizer()

result = generate_sql(
    model, tokenizer,
    question = "get total sales per customer with more than 5 orders",
    context  = (
        "CREATE TABLE customers (id INT, name TEXT); "
        "CREATE TABLE orders (id INT, customer_id INT, amount FLOAT)"
    ),
    device      = device,
    temperature = 0.3,
)
print(result)
```

### Complete code
```python
from src.inference.generator import load_model, load_tokenizer, generate_completion

model, device = load_model()
tokenizer     = load_tokenizer()

result = generate_completion(
    model, tokenizer,
    code_prefix = "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot =",
    device      = device,
    temperature = 0.5,
)
print(result)
```

### Run Gradio UI
```bash
python src/inference/ui.py
# Open http://localhost:7860
```

### Run FastAPI server
```bash
python src/inference/api.py
# API docs at http://localhost:8000/docs
```

---

## Project structure
```
AnupamB-Coder/
│
├── src/
│   ├── model/
│   │   ├── attention.py       ← Causal multi-head self-attention
│   │   ├── transformer.py     ← Transformer block (attn + FFN + LN)
│   │   └── gpt.py             ← Full GPT model + generation
│   │
│   ├── data/
│   │   ├── downloader.py      ← Download datasets from HuggingFace
│   │   ├── downloader_500gb.py← Download 22GB+ datasets
│   │   ├── cleaner.py         ← Clean stage 0 data
│   │   ├── cleaner_500gb.py   ← Clean 22GB new data
│   │   ├── tokenizer_builder.py← Train custom BPE tokenizer
│   │   └── dataset.py         ← PyTorch Dataset for all 6 stages
│   │
│   ├── training/
│   │   ├── trainer.py         ← Main training loop
│   │   └── optimizer.py       ← AdamW 8-bit optimizer
│   │
│   └── inference/
│       ├── generator.py       ← Text generation with rep penalty
│       ├── api.py             ← FastAPI server
│       └── ui.py              ← Gradio web interface
│
├── configs/
│   ├── small.yaml             ← 35M model config
│   └── medium.yaml            ← 110M model config
│
├── paths.py                   ← Single source of truth for all paths
├── train.py                   ← Entry point
├── trainer_test.py            ← Smoke test script
├── upload_to_hf.py            ← HuggingFace upload script
└── requirements.txt
```

---

## Reproduce training

### Step 1 — Download data
```bash
python src/data/downloader.py
python src/data/downloader_500gb.py
```

### Step 2 — Clean data
```bash
python src/data/cleaner.py
python src/data/cleaner_500gb.py
```

### Step 3 — Build tokenizer
```bash
python src/data/tokenizer_builder.py
```

### Step 4 — Smoke test
```bash
python trainer_test.py
```

### Step 5 — Train all stages
```bash
# Stage 1
python src/training/trainer.py --stage 1

# Stage 2 — resume from stage 1
python src/training/trainer.py --stage 2 --resume checkpoints/run_001/best.pt

# Stage 3 — resume from stage 2
python src/training/trainer.py --stage 3 --resume checkpoints/run_002/best.pt

# Stage 4
python src/training/trainer.py --stage 4 --resume checkpoints/run_003/best.pt

# Stage 5
python src/training/trainer.py --stage 5 --resume checkpoints/run_004/best.pt

# Stage 6 — instruction fine-tuning
python src/training/trainer.py --stage 6 --resume checkpoints/run_005/best.pt
```

---

## Why build from scratch?

Most people building LLM projects today:
- Download GPT-2 weights from HuggingFace
- Call `model.generate()` without understanding it
- Fine-tune with 3 lines of code

This project takes the opposite approach. Every component was implemented
and understood before moving to the next:
```
Week 1  → Data pipeline — download, clean, tokenize
Week 2  → Model architecture — attention, transformer, GPT
Week 3  → Training loop — optimizer, scheduler, checkpointing
Week 4+ → Multi-stage training — 43 days of continuous learning
```

The goal was not just to have a working model but to deeply understand
how GPT works at every level — from byte-pair encoding to the cosine
learning rate schedule.

---

## Limitations

- Trained only on Python and SQL — not a general purpose model
- 110M parameters — smaller than GPT-2 (117M) and CodeGen (350M)
- May produce syntactically correct but logically wrong code
- Best used as a code starter or learning tool
- Not suitable for production code without human review

---

## Roadmap

- [ ] Release model weights on HuggingFace
- [ ] Deploy Gradio demo on HuggingFace Spaces
- [ ] Add support for more programming languages
- [ ] Train a 350M parameter version
- [ ] Add RAG for documentation lookup
- [ ] VSCode extension

---

## License

MIT License — free to use, modify, and distribute.
See [LICENSE](LICENSE) for details.

---

## Citation

If you use this model or code in your research please cite:
```bibtex
@misc{anupambayen-coder-2026,
  author    = {Anupam Bayen},
  title     = {AnupamB-Coder-110M: A GPT Language Model Built From Scratch},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/anupambayen2/AnupamB-Coder}
}
```

---

## Author

**Anupam Bayen**
Kolkata, India

- HuggingFace : [anupambayen](https://huggingface.co/anupambayen)
- GitHub      : [anupambayen2](https://github.com/anupambayen2)

---

<p align="center">
Built with PyTorch · Trained on RTX 4060 · Made in Tamil Nadu, India
</p>

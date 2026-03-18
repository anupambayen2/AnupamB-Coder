# setup.py
# Run once from E:\mini_gpt to create entire project structure
# python setup.py

import os

# ── Absolute root — everything is relative to this ──────────
ROOT = os.path.dirname(os.path.abspath(__file__))
print(f"Project root: {ROOT}\n")

# ── Folders ──────────────────────────────────────────────────
folders = [
    "src",
    "src/model",
    "src/data",
    "src/training",
    "src/inference",
    "src/utils",
    "data/raw",
    "data/processed",
    "data/processed/cache",
    "data/tokenizer",
    "checkpoints/run_001",
    "logs",
    "experiments",
    "configs",
]

# ── paths.py — single source of truth for ALL paths ─────────
# Every script imports from here — no more hardcoded strings
PATHS_CONTENT = f'''\
# paths.py
# Single source of truth for all file paths.
# Every script in this project imports from here.
# Never hardcode paths anywhere else.

import os

ROOT           = r"{ROOT}"
SRC_DIR        = os.path.join(ROOT, "src")
DATA_DIR       = os.path.join(ROOT, "data")
RAW_DIR        = os.path.join(ROOT, "data", "raw")
PROCESSED_DIR  = os.path.join(ROOT, "data", "processed")
CACHE_DIR      = os.path.join(ROOT, "data", "processed", "cache")
TOKENIZER_DIR  = os.path.join(ROOT, "data", "tokenizer")
TOKENIZER_PATH = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
TOK_CFG_PATH   = os.path.join(ROOT, "data", "tokenizer", "tokenizer_config.json")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints", "run_001")
LOG_DIR        = os.path.join(ROOT, "logs")
LOG_PATH       = os.path.join(ROOT, "logs", "train_log.jsonl")

PYTHON_RAW     = os.path.join(RAW_DIR,       "python_code.jsonl")
SQL_RAW        = os.path.join(RAW_DIR,       "sql_code.jsonl")
PYTHON_CLEAN   = os.path.join(PROCESSED_DIR, "python_clean.jsonl")
SQL_CLEAN      = os.path.join(PROCESSED_DIR, "sql_clean.jsonl")
DATASET_CACHE  = os.path.join(CACHE_DIR,     "full_dataset.npy")

# Ensure all directories exist when this module is imported
for _d in [RAW_DIR, PROCESSED_DIR, CACHE_DIR, TOKENIZER_DIR,
           CHECKPOINT_DIR, LOG_DIR]:
    os.makedirs(_d, exist_ok=True)
'''

# ── __init__.py files — all empty ───────────────────────────
INIT = ""

# ── configs/small.yaml ──────────────────────────────────────
SMALL_YAML = """\
# Small GPT — fits comfortably in 8GB VRAM
model:
  vocab_size  : 32000
  block_size  : 512
  n_layer     : 6
  n_head      : 8
  n_embd      : 512
  dropout     : 0.1
  bias        : true

training:
  batch_size        : 4
  grad_accum_steps  : 8
  learning_rate     : 3.0e-4
  min_lr            : 3.0e-5
  max_steps         : 50000
  warmup_steps      : 1000
  eval_interval     : 500
  save_interval     : 1000
  grad_clip         : 1.0
  weight_decay      : 0.1
  fp16              : true
"""

# ── configs/medium.yaml ─────────────────────────────────────
MEDIUM_YAML = """\
# Medium GPT — use after small model trains successfully
model:
  vocab_size  : 32000
  block_size  : 1024
  n_layer     : 12
  n_head      : 12
  n_embd      : 768
  dropout     : 0.1
  bias        : true

training:
  batch_size        : 2
  grad_accum_steps  : 16
  learning_rate     : 1.0e-4
  min_lr            : 1.0e-5
  max_steps         : 100000
  warmup_steps      : 2000
  eval_interval     : 500
  save_interval     : 1000
  grad_clip         : 1.0
  weight_decay      : 0.1
  fp16              : true
"""

# ── README.md ────────────────────────────────────────────────
README = """\
# MiniGPT — Built from Scratch

GPT language model built from scratch for:
- Python & SQL code assistance
- Document summarization
- Knowledge Q&A

## Hardware
- GPU  : NVIDIA RTX 4060 Laptop 8GB
- CUDA : 12.6
- RAM  : 16GB

## Run
```bash
# Download data
python src/data/downloader.py

# Clean data
python src/data/cleaner.py

# Build tokenizer
python src/data/tokenizer_builder.py

# Train
python train.py
```
"""

# ── .gitignore ───────────────────────────────────────────────
GITIGNORE = """\
data/
checkpoints/
logs/
.env
__pycache__/
*.pyc
*.pyo
.pytest_cache/
.ipynb_checkpoints/
.vscode/
"""

# ── requirements.txt ────────────────────────────────────────
REQUIREMENTS = """\
torch==2.10.0
transformers==5.3.0
datasets==4.7.0
accelerate==1.13.0
bitsandbytes==0.49.2
tokenizers==0.22.2
numpy==2.3.5
pandas==3.0.1
pyarrow==23.0.1
pyyaml>=6.0
tqdm>=4.0
"""

# ── train.py — root entry point ──────────────────────────────
TRAIN_PY = """\
# train.py  —  run as: python train.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.training.trainer import train
if __name__ == "__main__":
    train()
"""

# ── Stub content for source files ───────────────────────────
STUB = "# To be implemented\n"

# ── File map ─────────────────────────────────────────────────
files = {
    # Root
    "paths.py"           : PATHS_CONTENT,
    "train.py"           : TRAIN_PY,
    "requirements.txt"   : REQUIREMENTS,
    ".gitignore"         : GITIGNORE,
    "README.md"          : README,

    # Package inits — all empty
    "src/__init__.py"            : INIT,
    "src/model/__init__.py"      : INIT,
    "src/data/__init__.py"       : INIT,
    "src/training/__init__.py"   : INIT,
    "src/inference/__init__.py"  : INIT,
    "src/utils/__init__.py"      : INIT,

    # Model stubs
    "src/model/config.py"        : STUB,
    "src/model/attention.py"     : STUB,
    "src/model/transformer.py"   : STUB,
    "src/model/gpt.py"           : STUB,

    # Data stubs
    "src/data/downloader.py"     : STUB,
    "src/data/cleaner.py"        : STUB,
    "src/data/tokenizer_builder.py" : STUB,
    "src/data/dataset.py"        : STUB,

    # Training stubs
    "src/training/trainer.py"    : STUB,
    "src/training/optimizer.py"  : STUB,
    "src/training/evaluator.py"  : STUB,

    # Inference stubs
    "src/inference/generator.py" : STUB,
    "src/inference/api.py"       : STUB,
    "src/inference/ui.py"        : STUB,

    # Utils stubs
    "src/utils/logger.py"        : STUB,
    "src/utils/helpers.py"       : STUB,

    # Configs
    "configs/small.yaml"         : SMALL_YAML,
    "configs/medium.yaml"        : MEDIUM_YAML,
}

# ── Create folders ───────────────────────────────────────────
print("Creating folders...\n")
for folder in folders:
    full = os.path.join(ROOT, folder)
    os.makedirs(full, exist_ok=True)
    print(f"  ✓ {folder}/")

# ── Create files ─────────────────────────────────────────────
print("\nCreating files...\n")
for filepath, content in files.items():
    full = os.path.join(ROOT, filepath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  ✓ {filepath}")

# ── Verify paths.py works ────────────────────────────────────
print("\nVerifying paths.py...")
import importlib.util
spec = importlib.util.spec_from_file_location("paths", os.path.join(ROOT, "paths.py"))
paths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paths)
print(f"  ROOT           = {paths.ROOT}")
print(f"  TOKENIZER_PATH = {paths.TOKENIZER_PATH}")
print(f"  CHECKPOINT_DIR = {paths.CHECKPOINT_DIR}")

# ── Summary ──────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  Folders : {len(folders)}")
print(f"  Files   : {len(files)}")
print(f"  Status  : Ready")
print(f"{'═'*55}")
print(f"\nNext step: python src/data/downloader.py")
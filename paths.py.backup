# paths.py
# Single source of truth for ALL paths across both drives.
# E: = project, processed data, cache, checkpoints
# F: = raw downloaded datasets (large files)

import os

# ── Drive roots ───────────────────────────────────────────────
E_ROOT  = r"E:\mini_gpt"
F_ROOT  = r"F:\gpt_rawdata"

# ── Project root ──────────────────────────────────────────────
ROOT    = E_ROOT
SRC_DIR = os.path.join(ROOT, "src")

# ── F: Raw data directories (downloads land here) ─────────────
RAW_ROOT        = F_ROOT
RAW_PYTHON      = os.path.join(F_ROOT, "python")
RAW_SQL         = os.path.join(F_ROOT, "sql")
RAW_STACKOVERFLOW = os.path.join(F_ROOT, "stackoverflow")
RAW_ALGORITHMS  = os.path.join(F_ROOT, "algorithms")
RAW_INSTRUCT    = os.path.join(F_ROOT, "instruct")

# Stage 0 raw files (already downloaded on E:)
PYTHON_RAW      = os.path.join(ROOT, "data", "raw", "python_code.jsonl")
SQL_RAW         = os.path.join(ROOT, "data", "raw", "sql_code.jsonl")

# Stage 2+ raw files (land on F:)
STACK_V2_RAW    = os.path.join(RAW_PYTHON,       "stack_v2.jsonl")
CODEPARROT_RAW  = os.path.join(RAW_PYTHON,       "codeparrot.jsonl")
STACKOVERFLOW_RAW = os.path.join(RAW_STACKOVERFLOW, "stackoverflow.jsonl")
LEETCODE_RAW    = os.path.join(RAW_ALGORITHMS,   "leetcode.jsonl")
SQL_SPIDER_RAW  = os.path.join(RAW_SQL,          "spider.jsonl")
SQL_WIKISQL_RAW = os.path.join(RAW_SQL,          "wikisql.jsonl")
SQL_BIRD_RAW    = os.path.join(RAW_SQL,          "bird.jsonl")
ALPACA_RAW      = os.path.join(RAW_INSTRUCT,     "code_alpaca.jsonl")
EVOL_RAW        = os.path.join(RAW_INSTRUCT,     "evol_instruct.jsonl")

# ── E: Processed data (cleaned JSONL after filtering) ─────────
PROCESSED_DIR   = os.path.join(ROOT, "data", "processed")
PYTHON_CLEAN    = os.path.join(PROCESSED_DIR, "python_clean.jsonl")
SQL_CLEAN       = os.path.join(PROCESSED_DIR, "sql_clean.jsonl")

# Stage 2+ processed files
STACK_V2_CLEAN      = os.path.join(PROCESSED_DIR, "stack_v2_clean.jsonl")
CODEPARROT_CLEAN    = os.path.join(PROCESSED_DIR, "codeparrot_clean.jsonl")
STACKOVERFLOW_CLEAN = os.path.join(PROCESSED_DIR, "stackoverflow_clean.jsonl")
LEETCODE_CLEAN      = os.path.join(PROCESSED_DIR, "leetcode_clean.jsonl")
SQL_ALL_CLEAN       = os.path.join(PROCESSED_DIR, "sql_all_clean.jsonl")
INSTRUCT_CLEAN      = os.path.join(PROCESSED_DIR, "instruct_clean.jsonl")

# ── E: Tokenizer ──────────────────────────────────────────────
TOKENIZER_DIR   = os.path.join(ROOT, "data", "tokenizer")
TOKENIZER_PATH  = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
TOK_CFG_PATH    = os.path.join(ROOT, "data", "tokenizer", "tokenizer_config.json")

# ── E: Cache (tokenized .npy files — fast training load) ──────
CACHE_DIR       = os.path.join(ROOT, "data", "processed", "cache")
DATASET_CACHE   = os.path.join(CACHE_DIR, "stage0_cache.npy")

# Stage-specific caches
def stage_cache(stage: int) -> str:
    return os.path.join(CACHE_DIR, f"stage{stage}_cache.npy")

# ── E: Checkpoints ────────────────────────────────────────────
CHECKPOINT_BASE = os.path.join(ROOT, "checkpoints")

def checkpoint_dir(stage: int) -> str:
    return os.path.join(CHECKPOINT_BASE, f"run_{stage:03d}")

# Convenience shortcuts
CHECKPOINT_DIR  = checkpoint_dir(1)   # current active stage
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best.pt")

# ── E: Logs ───────────────────────────────────────────────────
LOG_DIR  = os.path.join(ROOT, "logs")
LOG_PATH = os.path.join(ROOT, "logs", "train_log.jsonl")

def stage_log(stage: int) -> str:
    return os.path.join(LOG_DIR, f"stage{stage}_log.jsonl")

# ── Create all E: directories on import ───────────────────────
for _d in [
    PROCESSED_DIR,
    CACHE_DIR,
    TOKENIZER_DIR,
    LOG_DIR,
    checkpoint_dir(1),
    checkpoint_dir(2),
    checkpoint_dir(3),
    checkpoint_dir(4),
    checkpoint_dir(5),
    checkpoint_dir(6),
]:
    os.makedirs(_d, exist_ok=True)


# ── Verify F: drive is accessible ─────────────────────────────
def verify_drives():
    print(f"\n  Drive check:")
    print(f"    E: project root  : {'OK' if os.path.exists(ROOT)   else 'MISSING'} — {ROOT}")
    print(f"    F: raw data root : {'OK' if os.path.exists(F_ROOT) else 'MISSING'} — {F_ROOT}")
    for name, path in [
        ("python",        RAW_PYTHON),
        ("sql",           RAW_SQL),
        ("stackoverflow", RAW_STACKOVERFLOW),
        ("algorithms",    RAW_ALGORITHMS),
        ("instruct",      RAW_INSTRUCT),
    ]:
        status = "OK" if os.path.exists(path) else "MISSING"
        print(f"    F:\\{name:<14} : {status}")
    print()


if __name__ == "__main__":
    verify_drives()
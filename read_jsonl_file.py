# read_jsonl.py
# Run: python read_jsonl.py

import json
import os

FILE = r"F:\gpt_rawdata\synthetic\python\chunk_000.jsonl"

# ── Quick stats ───────────────────────────────────────────────
def stats(filepath):
    total  = 0
    sizes  = []
    levels = {"basic": 0, "intermediate": 0,
              "advanced": 0, "expert": 0}

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "")
                total += 1
                sizes.append(len(text))

                for lvl in levels:
                    if lvl in text.lower():
                        levels[lvl] += 1
                        break
            except json.JSONDecodeError:
                continue

    print(f"\n{'═'*55}")
    print(f"  FILE STATS")
    print(f"{'═'*55}")
    print(f"  File       : {os.path.basename(filepath)}")
    print(f"  Size       : {os.path.getsize(filepath)/1024/1024:.1f} MB")
    print(f"  Examples   : {total:,}")
    print(f"  Avg length : {sum(sizes)//len(sizes) if sizes else 0} chars")
    print(f"  Min length : {min(sizes) if sizes else 0} chars")
    print(f"  Max length : {max(sizes) if sizes else 0} chars")
    print(f"\n  Difficulty distribution:")
    for lvl, cnt in levels.items():
        pct = cnt / total * 100 if total else 0
        bar = "█" * int(pct / 3)
        print(f"    {lvl:<14} {cnt:>7,}  {pct:>5.1f}%  {bar}")
    print(f"{'═'*55}\n")


# ── Show first N examples ─────────────────────────────────────
def show_examples(filepath, n=5, level=None):
    print(f"\n{'═'*55}")
    print(f"  FIRST {n} EXAMPLES"
          + (f" — {level} only" if level else ""))
    print(f"{'═'*55}")

    shown = 0
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            if shown >= n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                text = item.get("text", "")

                # Filter by level if requested
                if level and level.lower() not in text.lower():
                    continue

                print(f"\n  ── Example {shown + 1} "
                      f"{'─'*35}")
                print(text[:800])  # First 800 chars
                if len(text) > 800:
                    print(f"  ... ({len(text)} chars total)")
                shown += 1

            except json.JSONDecodeError as e:
                print(f"  [bad line] {e}")

    print(f"\n{'═'*55}\n")


# ── Search for keyword ────────────────────────────────────────
def search(filepath, keyword, n=3):
    print(f"\n  Searching for '{keyword}'...")
    found = 0
    with open(filepath, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if found >= n:
                break
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                if keyword.lower() in text.lower():
                    print(f"\n  ── Match {found+1} (line {i+1}) ──")
                    # Show context around keyword
                    idx   = text.lower().find(keyword.lower())
                    start = max(0, idx - 100)
                    end   = min(len(text), idx + 300)
                    print(f"  ...{text[start:end]}...")
                    found += 1
            except Exception:
                continue
    if found == 0:
        print(f"  No matches found for '{keyword}'")
    print()


# ── Run all ───────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    filepath = sys.argv[1] if len(sys.argv) > 1 else FILE

    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        sys.exit(1)

    # Show stats
    stats(filepath)

    # Show first 3 examples
    show_examples(filepath, n=3)

    # Show 2 examples of each level
    for level in ["basic", "intermediate", "advanced", "expert"]:
        show_examples(filepath, n=2, level=level)

    # Search examples
    search(filepath, "binary_search", n=2)
    search(filepath, "SELECT", n=2)
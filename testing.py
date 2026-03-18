# vram_test.py
import sys, os, torch
sys.path.insert(0, os.path.abspath('.'))
from src.model.gpt import GPT, GPTConfig
from torch.amp import autocast

device = "cuda"
print("Building 85M model...")

cfg = GPTConfig(
    vocab_size = 32000,
    block_size = 1024,
    n_layer    = 12,
    n_head     = 12,
    n_embd     = 768,
    dropout    = 0.1,
)
model = GPT(cfg).to(device)
params = model.get_num_params()
print(f"Parameters : {params:,}")

torch.cuda.reset_peak_memory_stats()
x = torch.randint(0, 32000, (2, 1024)).to(device)
y = torch.randint(0, 32000, (2, 1024)).to(device)

with autocast(device_type="cuda", enabled=True):
    _, loss = model(x, y)
loss.backward()

peak     = torch.cuda.max_memory_allocated() / 1024**3
headroom = 8.0 - peak

print(f"VRAM peak  : {peak:.2f} GB / 8.00 GB")
print(f"Headroom   : {headroom:.2f} GB")

if peak < 6.5:
    print("Status     : SAFE — plenty of headroom")
elif peak < 7.2:
    print("Status     : OK — tight but workable")
elif peak < 7.8:
    print("Status     : WARNING — very tight")
else:
    print("Status     : DANGER — likely OOM")
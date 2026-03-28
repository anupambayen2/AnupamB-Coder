"""
Gradio Code Generation Interface for AnupamB-Coder-110M
Test your model's coding capabilities interactively

Usage:
    cd E:\mini_gpt
    python test_gradio_interface.py
"""

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import gradio as gr
from src.model.gpt import GPT, GPTConfig

# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint"""
    print(f"\n{'='*70}")
    print(f"  LOADING MODEL")
    print(f"{'='*70}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {device}")
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    cfg = ckpt.get("config", {})
    vocab_size = cfg.get("vocab_size", 32000)
    block_size = cfg.get("block_size", 1024)
    n_layer = cfg.get("n_layer", 12)
    n_head = cfg.get("n_head", 12)
    n_embd = cfg.get("n_embd", 768)
    
    print(f"  Vocab size: {vocab_size}")
    print(f"  Block size: {block_size}")
    print(f"  Layers: {n_layer}")
    print(f"  Heads: {n_head}")
    print(f"  Embedding: {n_embd}")
    
    # Create model
    model_config = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.0,  # No dropout for inference
        bias=True,
    )
    
    model = GPT(model_config)
    
    # Load weights
    state_dict = ckpt["model_state"]
    # Remove _orig_mod. prefix if present (from torch.compile)
    state_dict = {
        k.replace("_orig_mod.", ""): v 
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    
    model.to(device)
    model.eval()
    
    # Get training info
    step = ckpt.get("step", "unknown")
    loss = ckpt.get("loss", "unknown")
    stage = ckpt.get("stage", "unknown")
    
    print(f"  Training step: {step}")
    print(f"  Training loss: {loss}")
    print(f"  Training stage: {stage}")
    print(f"{'='*70}\n")
    
    return model, vocab_size, block_size


def load_tokenizer():
    """Load tokenizer"""
    try:
        from tokenizers import Tokenizer
        tok_path = os.path.join(ROOT, "data", "tokenizer", "tokenizer.json")
        if os.path.exists(tok_path):
            print(f"  ✓ Loading tokenizer from: {tok_path}")
            tok = Tokenizer.from_file(str(tok_path))
            print(f"  ✓ Tokenizer loaded (vocab size: {tok.get_vocab_size()})\n")
            return tok
    except Exception as e:
        print(f"  [!] Error loading tokenizer: {e}")
        raise
    
    raise RuntimeError("Tokenizer not found!")


# ══════════════════════════════════════════════════════════════
#  CODE GENERATION
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_code(
    prompt: str,
    model,
    tokenizer,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.95,
    device: str = "cuda"
):
    """Generate code completion from prompt"""
    
    if not prompt.strip():
        return "⚠️ Please enter a prompt!"
    
    # Tokenize prompt
    encoding = tokenizer.encode(prompt)
    input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=device)
    
    if input_ids.shape[1] == 0:
        return "⚠️ Prompt tokenization failed!"
    
    # Generate
    model.eval()
    
    generated_ids = input_ids.clone()
    
    for _ in range(max_tokens):
        # Get logits for last position
        logits, _ = model(generated_ids[:, -1024:])  # Use last 1024 tokens
        logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = -float('Inf')
        
        # Sample
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append to sequence
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # Check for natural stopping (optional - you can add stop tokens)
        # For now, just generate max_tokens
    
    # Decode
    output_ids = generated_ids[0].tolist()
    output_text = tokenizer.decode(output_ids)
    
    return output_text


# ══════════════════════════════════════════════════════════════
#  GRADIO INTERFACE
# ══════════════════════════════════════════════════════════════

def create_interface(model, tokenizer, device="cuda"):
    """Create Gradio interface"""
    
    def generate_wrapper(prompt, max_tokens, temperature, top_k, top_p):
        """Wrapper for Gradio"""
        try:
            output = generate_code(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_k=int(top_k),
                top_p=float(top_p),
                device=device
            )
            return output
        except Exception as e:
            return f"❌ Error: {str(e)}"
    
    # Example prompts
    examples = [
        [
            "def fibonacci(n):\n    \"\"\"Calculate nth fibonacci number\"\"\"\n    ",
            150, 0.7, 50, 0.95
        ],
        [
            "# Binary search implementation\ndef binary_search(arr, target):\n    ",
            200, 0.7, 50, 0.95
        ],
        [
            "class LinkedList:\n    \"\"\"Singly linked list implementation\"\"\"\n    def __init__(self):\n        ",
            200, 0.8, 50, 0.95
        ],
        [
            "# Quick sort algorithm\ndef quick_sort(arr):\n    ",
            200, 0.7, 50, 0.95
        ],
        [
            "import pandas as pd\n\n# Load and clean data\ndef process_data(filename):\n    ",
            200, 0.8, 50, 0.95
        ],
        [
            "-- SQL query to find top 10 customers by revenue\nSELECT ",
            150, 0.7, 50, 0.95
        ],
        [
            "# Flask API endpoint\nfrom flask import Flask, request, jsonify\n\napp = Flask(__name__)\n\n@app.route('/api/predict', methods=['POST'])\ndef predict():\n    ",
            200, 0.8, 50, 0.95
        ],
        [
            "# Calculate factorial recursively\ndef factorial(n):\n    ",
            100, 0.7, 50, 0.95
        ],
    ]
    
    # Create interface
    with gr.Blocks(title="AnupamB-Coder-110M Code Generator") as demo:
        gr.Markdown(
            """
            # 🚀 AnupamB-Coder-110M Code Generator
            
            Test your model's code generation capabilities! Enter a code prompt and watch the model complete it.
            
            **Currently testing:** Stage 2 checkpoint (run_002/best.pt)
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                prompt_input = gr.Textbox(
                    label="Code Prompt",
                    placeholder="Enter your code prompt here...\nExample: def fibonacci(n):",
                    lines=10,
                    value=""
                )
                
                with gr.Row():
                    max_tokens = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=200,
                        step=10,
                        label="Max Tokens to Generate"
                    )
                
                with gr.Row():
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.8,
                        step=0.1,
                        label="Temperature (creativity)"
                    )
                    
                with gr.Row():
                    top_k = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=50,
                        step=1,
                        label="Top-K"
                    )
                    
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top-P (nucleus)"
                    )
                
                generate_btn = gr.Button("🚀 Generate Code", variant="primary")
                clear_btn = gr.ClearButton([prompt_input])
            
            with gr.Column(scale=1):
                output = gr.Textbox(
                    label="Generated Code",
                    lines=20,
                    value=""
                )
        
        gr.Markdown(
            """
            ### 📝 Example Prompts
            Click an example below to load it:
            """
        )
        
        gr.Examples(
            examples=examples,
            inputs=[prompt_input, max_tokens, temperature, top_k, top_p],
            outputs=output,
            fn=generate_wrapper,
            cache_examples=False,
        )
        
        gr.Markdown(
            """
            ---
            ### 💡 Tips:
            - **Temperature**: Lower (0.3-0.7) = more focused, Higher (0.8-1.5) = more creative
            - **Top-K**: Limits vocabulary to top K tokens (50 is good default)
            - **Top-P**: Nucleus sampling threshold (0.95 is good default)
            - **Max Tokens**: How much code to generate (100-300 typical)
            
            ### 🎯 What to Look For:
            - ✅ **Syntax correctness**: Is the code syntactically valid?
            - ✅ **Logic coherence**: Does the logic make sense?
            - ✅ **Comment quality**: Are comments relevant and helpful?
            - ✅ **Code style**: Does it follow Python/SQL conventions?
            - ✅ **Completeness**: Does it finish the task?
            
            ### 📊 Stage 2 Baseline:
            Use this to establish baseline performance, then compare with Stage 3a after training!
            """
        )
        
        # Connect button
        generate_btn.click(
            fn=generate_wrapper,
            inputs=[prompt_input, max_tokens, temperature, top_k, top_p],
            outputs=output
        )
    
    return demo


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model code generation with Gradio")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/run_002/best.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run Gradio on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public share link"
    )
    
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"\n❌ Checkpoint not found: {args.checkpoint}")
        print(f"   Please provide valid checkpoint path\n")
        sys.exit(1)
    
    # Load model and tokenizer
    print("\n" + "="*70)
    print("  AnupamB-Coder-110M Code Generator - Gradio Interface")
    print("="*70 + "\n")
    
    tokenizer = load_tokenizer()
    model, vocab_size, block_size = load_model(args.checkpoint, args.device)
    
    # Create and launch interface
    demo = create_interface(model, tokenizer, args.device)
    
    print(f"\n{'='*70}")
    print(f"  LAUNCHING GRADIO INTERFACE")
    print(f"{'='*70}")
    print(f"  Server will start at: http://localhost:{args.port}")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*70}\n")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True
    )

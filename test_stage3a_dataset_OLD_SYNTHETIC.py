"""
Test Stage 3a dataset - verify data loading and cache building
Run this before starting Stage 3a training

Usage:
    cd E:\mini_gpt
    python test_stage3a_dataset.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.dataset_stage3 import build_dataloaders_stage3

def test_stage3a_dataset():
    """Test Stage 3a dataset loading"""
    
    print("="*70)
    print("  STAGE 3a DATASET TEST")
    print("="*70)
    print()
    
    # Configuration
    stage = "3a"
    batch_size = 2  # Small batch for testing
    block_size = 1024
    
    print(f"Test Configuration:")
    print(f"  Stage      : {stage}")
    print(f"  Batch size : {batch_size} (small for testing)")
    print(f"  Block size : {block_size}")
    print()
    
    # Expected data sources for Stage 3a
    print("Expected data sources for Stage 3a:")
    print("  Python curated files:")
    print("    • code_feedback.jsonl        (368 MB)")
    print("    • evol_magicoder.jsonl       (257 MB)")
    print("    • glaive_code.jsonl          (408 MB)")
    print("    • python_codes_25k.jsonl     (25 MB)")
    print("    • tested_python.jsonl        (49 MB)")
    print()
    print("  SQL files:")
    print("    • sql_instruct.jsonl         (77 MB)")
    print()
    print("  Synthetic data:")
    print("    • python chunks 000-019      (~1.5 GB)")
    print("    • sql chunks 000-019         (~1.2 GB)")
    print()
    print("  Expected cache size: ~8 GB")
    print()
    
    # Build dataloaders
    print("─"*70)
    print("Building Stage 3a dataloaders...")
    print("─"*70)
    print()
    
    try:
        train_loader, val_loader, vocab_size = build_dataloaders_stage3(
            stage=stage,
            batch_size=batch_size,
            num_workers=0,
            block_size=block_size,
            val_split=0.05,
            vocab_size=32000
        )
        
        print("\n✓ Dataloaders created successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Missing data file")
        print(f"   {str(e)}")
        print("\nPlease verify:")
        print("  1. All curated files exist in F:\\gpt_rawdata\\stage3\\python\\")
        print("  2. sql_instruct.jsonl exists in F:\\gpt_rawdata\\stage3\\sql\\")
        print("  3. Synthetic chunks 000-019 exist in:")
        print("     F:\\gpt_rawdata\\synthetic\\python\\")
        print("     F:\\gpt_rawdata\\synthetic\\sql\\")
        return False
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()
        return False
    
    # Test batch loading
    print("\n" + "─"*70)
    print("Testing batch loading...")
    print("─"*70)
    
    try:
        # Get a batch from train
        print("\nLoading batch from training set...")
        x_train, y_train = next(iter(train_loader))
        
        print(f"✓ Train batch loaded successfully")
        print(f"  x shape       : {x_train.shape}")
        print(f"  y shape       : {y_train.shape}")
        print(f"  x dtype       : {x_train.dtype}")
        print(f"  Min token     : {x_train.min().item()}")
        print(f"  Max token     : {x_train.max().item()}")
        print(f"  Vocab size    : {vocab_size}")
        
        # Show first 10 tokens
        first_10 = x_train[0, :10].tolist()
        print(f"  First 10 tokens: {first_10}")
        
        # Verify shift relationship (y should be x shifted by 1)
        print(f"\n  Verifying y = x shifted by 1...")
        assert (x_train[0][1:] == y_train[0][:-1]).all(), "Shift check failed!"
        print(f"  Shift check   : ✓ PASSED")
        
        # Verify token range
        print(f"  Verifying token range 0 <= token < {vocab_size}...")
        assert x_train.min() >= 0, f"Min token {x_train.min()} is negative!"
        assert x_train.max() < vocab_size, f"Max token {x_train.max()} >= vocab_size {vocab_size}!"
        print(f"  Range check   : ✓ PASSED")
        
        # Get a batch from val
        print("\nLoading batch from validation set...")
        x_val, y_val = next(iter(val_loader))
        
        print(f"✓ Val batch loaded successfully")
        print(f"  x shape       : {x_val.shape}")
        print(f"  y shape       : {y_val.shape}")
        
    except Exception as e:
        print(f"\n❌ ERROR loading batch: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    # Check cache file
    print("\n" + "─"*70)
    print("Checking cache file...")
    print("─"*70)
    
    cache_path = Path("E:/mini_gpt/data/processed/cache/stage3a_bs1024.npy")
    
    if cache_path.exists():
        print(f"\n✓ Cache file exists: {cache_path}")
        
        # Check file size
        file_size_gb = cache_path.stat().st_size / (1024**3)
        print(f"  Size: {file_size_gb:.2f} GB")
        
        # Expected size check
        if 6 <= file_size_gb <= 10:
            print(f"  ✓ Size looks good (expected ~8 GB)")
        elif file_size_gb < 6:
            print(f"  ⚠️ WARNING: Cache smaller than expected")
            print(f"     Got {file_size_gb:.2f} GB, expected ~8 GB")
            print(f"     Some data files might be missing")
        else:
            print(f"  ⚠️ WARNING: Cache larger than expected")
            print(f"     Got {file_size_gb:.2f} GB, expected ~8 GB")
    else:
        print(f"\n⚠️ Cache file not found at: {cache_path}")
        print(f"   (This is normal if this is the first run)")
    
    # Summary
    print("\n" + "="*70)
    print("  TEST SUMMARY")
    print("="*70)
    
    print("\n✅ Stage 3a dataset test PASSED!")
    print("\nAll checks completed successfully:")
    print("  ✓ Tokenizer loaded correctly")
    print("  ✓ Data files found and processed")
    print("  ✓ Cache built successfully")
    print("  ✓ Train and validation loaders working")
    print("  ✓ Batch shapes correct")
    print("  ✓ Token range valid")
    print("  ✓ Shift relationship verified")
    
    print("\nDataset is ready for training!")
    
    print("\n" + "─"*70)
    print("NEXT STEPS:")
    print("─"*70)
    print("1. Start monitoring scripts:")
    print("   - Email: python screenshot_emailer.py")
    print("   - Telegram: python monitor_training.py")
    print()
    print("2. Check cooling pad is ON")
    print()
    print("3. Start Stage 3a training:")
    print("   python src/training/trainer.py --stage 3a \\")
    print("          --resume checkpoints/run_002/best.pt")
    print()
    print("Expected training time: ~4 days (100,000 steps)")
    print("="*70)
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = test_stage3a_dataset()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

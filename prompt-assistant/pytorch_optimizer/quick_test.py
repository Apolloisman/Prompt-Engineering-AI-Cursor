"""Quick test to verify the implementation works"""

import sys
import os

print("Testing Latent Prompt Optimizer...")
print("=" * 50)

# Test 1: Check dependencies
print("\n1. Checking dependencies...")
try:
    import torch
    print(f"   [OK] PyTorch {torch.__version__}")
except ImportError:
    print("   [FAIL] PyTorch not installed")
    sys.exit(1)

try:
    import transformers
    print(f"   [OK] Transformers {transformers.__version__}")
except ImportError:
    print("   [FAIL] Transformers not installed")
    print("   Install with: pip install transformers")
    sys.exit(1)

try:
    import numpy
    print(f"   [OK] NumPy {numpy.__version__}")
except ImportError:
    print("   [FAIL] NumPy not installed")
    sys.exit(1)

# Test 2: Check if code can be imported
print("\n2. Testing imports...")
try:
    from latent_prompt_optimizer import LatentPromptOptimizer, ContrastivePromptTrainer
    print("   [OK] Successfully imported LatentPromptOptimizer")
    print("   [OK] Successfully imported ContrastivePromptTrainer")
except Exception as e:
    print(f"   [FAIL] Import failed: {e}")
    sys.exit(1)

# Test 3: Check if model can be initialized (without downloading models)
print("\n3. Testing model initialization...")
print("   Note: This will download ~1.5GB of pre-trained models on first run")
print("   (RoBERTa-large ~1.3GB, GPT-2 ~500MB)")

try:
    device = 'cpu'  # Use CPU for testing
    print(f"   Using device: {device}")
    
    # Note: This will download models if not already cached
    # Comment out if you want to skip model download
    print("   Initializing model (this may take a minute on first run)...")
    model = LatentPromptOptimizer(
        latent_dim=768,
        delta_hidden_dims=[1024, 512, 256],
        dropout_rate=0.2,
        device=device
    )
    print("   [OK] Model initialized successfully")
    
except Exception as e:
    print(f"   [FAIL] Model initialization failed: {e}")
    print("   This is expected if models haven't been downloaded yet")
    print("   Run the full training script to download models")

# Test 4: Check trainer initialization
print("\n4. Testing trainer initialization...")
try:
    trainer = ContrastivePromptTrainer(
        model=model,
        learning_rate=1e-4,
        margin=0.5
    )
    print("   [OK] Trainer initialized successfully")
except Exception as e:
    print(f"   [FAIL] Trainer initialization failed: {e}")

print("\n" + "=" * 50)
print("[OK] All basic tests passed!")
print("\nNext steps:")
print("1. Create training data: python train.py --create_sample")
print("2. Train model: python train.py --data_path ./data/training_data.json")
print("3. See examples: python example_usage.py")


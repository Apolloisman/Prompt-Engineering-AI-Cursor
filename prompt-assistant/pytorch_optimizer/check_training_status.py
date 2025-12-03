"""Check LoRA training status"""
import os
import json
from datetime import datetime

print("="*70)
print("LORA TRAINING STATUS")
print("="*70)

# Check data
data_path = "./data/lora_training_data.json"
if os.path.exists(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    num_samples = len(data['training_pairs'])
    print(f"Training samples: {num_samples}")
    print(f"Batches per epoch (batch_size=8): {num_samples // 8}")
    print(f"Total batches (5 epochs): {5 * (num_samples // 8)}")
else:
    print("Training data not found!")
    num_samples = 0

# Check checkpoints
checkpoint_dir = "./checkpoints/lora_model"
if os.path.exists(checkpoint_dir):
    files = []
    for root, dirs, filenames in os.walk(checkpoint_dir):
        for f in filenames:
            files.append(os.path.join(root, f))
    
    if files:
        print(f"\nCheckpoints found: {len(files)}")
        latest = max(files, key=lambda x: os.path.getmtime(x))
        mtime = datetime.fromtimestamp(os.path.getmtime(latest))
        age = (datetime.now() - mtime).total_seconds() / 60
        print(f"Latest: {os.path.basename(latest)}")
        print(f"Age: {age:.1f} minutes ago")
        
        # Check for final adapter
        if os.path.exists(os.path.join(checkpoint_dir, "final_adapter")):
            print("\n✓ TRAINING COMPLETE! Final adapter found.")
        else:
            print("\n⏳ Training in progress...")
    else:
        print("\n⚠ No checkpoint files found yet")
        print("Training may still be initializing or downloading Flan-T5-base")
else:
    print("\n⚠ Training directory not created yet")

print("="*70)
print("\nExpected training time:")
print("- CPU: 30-60 minutes for 5 epochs")
print("- GPU: 5-15 minutes for 5 epochs")
print("- First run: +5-10 minutes to download Flan-T5-base (~1GB)")
print("="*70)



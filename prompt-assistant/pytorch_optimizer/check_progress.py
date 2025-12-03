"""Check training progress"""
import os
import sys
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

checkpoints_dir = "./checkpoints"
if os.path.exists(checkpoints_dir):
    files = os.listdir(checkpoints_dir)
    phase1_files = sorted([f for f in files if f.startswith('phase1_epoch_')])
    phase2_files = sorted([f for f in files if f.startswith('phase2_epoch_')])
    final_file = 'neuro_final_model.pt' in files
    
    print("=" * 60)
    print("TRAINING PROGRESS")
    print("=" * 60)
    print(f"Phase 1 (Distillation): {len(phase1_files)}/3 epochs")
    print(f"Phase 2 (Contrastive): {len(phase2_files)}/3 epochs")
    print(f"Final Model: {'✓ Yes' if final_file else '✗ No'}")
    
    if phase1_files:
        print(f"\nPhase 1 Checkpoints:")
        for f in phase1_files:
            path = os.path.join(checkpoints_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"  - {f} ({size_mb:.1f} MB) - {mtime.strftime('%H:%M:%S')}")
    
    if phase2_files:
        print(f"\nPhase 2 Checkpoints:")
        for f in phase2_files:
            path = os.path.join(checkpoints_dir, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            print(f"  - {f} ({size_mb:.1f} MB) - {mtime.strftime('%H:%M:%S')}")
    
    if final_file:
        print("\n" + "=" * 60)
        print("✓ TRAINING COMPLETE!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        if len(phase2_files) == 3:
            print("⚠ Phase 2 complete but final model not saved yet.")
        elif len(phase2_files) > 0:
            print(f"⏳ Training Phase 2: {len(phase2_files)}/3 epochs")
        elif len(phase1_files) == 3:
            print("⏳ Phase 1 complete. Starting Phase 2...")
        else:
            print(f"⏳ Training Phase 1: {len(phase1_files)}/3 epochs")
        print("=" * 60)
else:
    print("Checkpoints directory not found. Training may not have started.")



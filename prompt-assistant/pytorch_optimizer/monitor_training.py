"""Monitor training progress with live updates"""
import os
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8')

def get_training_status():
    """Get current training status"""
    checkpoints_dir = "./checkpoints"
    if not os.path.exists(checkpoints_dir):
        return None
    
    files = os.listdir(checkpoints_dir)
    phase1_files = sorted([f for f in files if f.startswith('phase1_epoch_')])
    phase2_files = sorted([f for f in files if f.startswith('phase2_epoch_')])
    final_file = 'neuro_final_model.pt' in files
    
    status = {
        'phase1_count': len(phase1_files),
        'phase2_count': len(phase2_files),
        'final_model': final_file,
        'phase1_files': phase1_files,
        'phase2_files': phase2_files
    }
    
    # Get latest checkpoint times
    if phase1_files:
        latest_phase1 = os.path.join(checkpoints_dir, phase1_files[-1])
        status['phase1_latest_time'] = datetime.fromtimestamp(os.path.getmtime(latest_phase1))
        status['phase1_latest_size'] = os.path.getsize(latest_phase1) / (1024 * 1024)  # MB
    
    if phase2_files:
        latest_phase2 = os.path.join(checkpoints_dir, phase2_files[-1])
        status['phase2_latest_time'] = datetime.fromtimestamp(os.path.getmtime(latest_phase2))
        status['phase2_latest_size'] = os.path.getsize(latest_phase2) / (1024 * 1024)  # MB
    
    return status

def display_status(status, iteration=0):
    """Display training status"""
    os.system('cls' if os.name == 'nt' else 'clear')  # Clear screen
    
    print("=" * 70)
    print(" NEURO-LATENT OPTIMIZER - TRAINING MONITOR")
    print("=" * 70)
    print(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    print(f"Check #: {iteration}")
    print()
    
    # Phase 1 Status
    print("PHASE 1: DISTILLATION TRAINING")
    print("-" * 70)
    print(f"Progress: {status['phase1_count']}/3 epochs")
    if status['phase1_count'] > 0:
        print(f"Latest: {status['phase1_files'][-1]}")
        print(f"Size: {status['phase1_latest_size']:.1f} MB")
        print(f"Time: {status['phase1_latest_time'].strftime('%H:%M:%S')}")
        age = (datetime.now() - status['phase1_latest_time']).total_seconds() / 60
        print(f"Age: {age:.1f} minutes ago")
    print()
    
    # Phase 2 Status
    print("PHASE 2: CONTRASTIVE TRAINING")
    print("-" * 70)
    print(f"Progress: {status['phase2_count']}/3 epochs")
    if status['phase2_count'] > 0:
        print(f"Latest: {status['phase2_files'][-1]}")
        print(f"Size: {status['phase2_latest_size']:.1f} MB")
        print(f"Time: {status['phase2_latest_time'].strftime('%H:%M:%S')}")
        age = (datetime.now() - status['phase2_latest_time']).total_seconds() / 60
        print(f"Age: {age:.1f} minutes ago")
    else:
        print("Status: Waiting for first checkpoint...")
    print()
    
    # Overall Status
    print("OVERALL STATUS")
    print("-" * 70)
    if status['final_model']:
        print("✓ TRAINING COMPLETE! Final model saved.")
    elif status['phase2_count'] == 3:
        print("⏳ Phase 2 complete. Finalizing model...")
    elif status['phase2_count'] > 0:
        print(f"⏳ Phase 2 in progress: {status['phase2_count']}/3 epochs")
        if status['phase2_count'] > 0:
            age = (datetime.now() - status['phase2_latest_time']).total_seconds() / 60
            if age < 5:
                print("  → Training appears active (recent checkpoint)")
            elif age < 15:
                print(f"  → Training may be in progress (checkpoint {age:.1f} min old)")
            else:
                print(f"  ⚠ No recent activity (checkpoint {age:.1f} min old)")
    elif status['phase1_count'] == 3:
        print("⏳ Phase 1 complete. Phase 2 should start soon...")
    else:
        print(f"⏳ Phase 1 in progress: {status['phase1_count']}/3 epochs")
    
    print()
    print("=" * 70)
    print("Press Ctrl+C to stop monitoring")
    print("=" * 70)

def main():
    """Main monitoring loop"""
    print("Starting training monitor...")
    print("Checking every 30 seconds...")
    time.sleep(2)
    
    iteration = 0
    last_phase2_count = 0
    
    try:
        while True:
            iteration += 1
            status = get_training_status()
            
            if status is None:
                print("Error: Checkpoints directory not found")
                time.sleep(30)
                continue
            
            display_status(status, iteration)
            
            # Check for progress
            if status['phase2_count'] > last_phase2_count:
                print(f"\n🎉 PROGRESS! Phase 2 epoch {status['phase2_count']} checkpoint saved!")
                time.sleep(5)
                last_phase2_count = status['phase2_count']
            
            # Check if training is complete
            if status['final_model']:
                print("\n" + "=" * 70)
                print("🎉 TRAINING COMPLETE! 🎉")
                print("=" * 70)
                print("Final model saved: neuro_final_model.pt")
                print("You can now use the trained model!")
                print("=" * 70)
                break
            
            time.sleep(30)  # Check every 30 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")
        print("Training continues in the background.")

if __name__ == '__main__':
    main()



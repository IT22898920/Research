"""
Training Progress Monitor
Monitors the training log and reports epoch completions
"""

import time
import re
from datetime import datetime
from pathlib import Path

LOG_FILE = Path("training_log.txt")
CHECK_INTERVAL = 60  # Check every 60 seconds

print("="*70)
print("         LEAF DISEASE MODEL - TRAINING MONITOR")
print("="*70)
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {LOG_FILE}")
print(f"Checking every {CHECK_INTERVAL} seconds...")
print("="*70)
print()

last_position = 0
current_epoch = 0
current_phase = 1
epoch_times = []

while True:
    try:
        if LOG_FILE.exists():
            with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                # Move to last position
                f.seek(last_position)
                new_content = f.read()
                last_position = f.tell()

                if new_content:
                    lines = new_content.split('\n')

                    for line in lines:
                        # Detect phase changes
                        if "PHASE 1:" in line:
                            current_phase = 1
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] PHASE 1 STARTED - Frozen Base Layers")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)

                        elif "PHASE 2:" in line:
                            current_phase = 2
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] PHASE 2 STARTED - Fine-tuning")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)

                        # Detect epoch completion (look for validation metrics)
                        # Format: 469/469 ... - accuracy: 0.xxxx - loss: 0.xxxx - val_accuracy: 0.xxxx - val_loss: 0.xxxx
                        if re.search(r'469/469.*- val_accuracy:', line):
                            # Extract metrics
                            acc_match = re.search(r'accuracy: ([\d.]+)', line)
                            val_acc_match = re.search(r'val_accuracy: ([\d.]+)', line)
                            loss_match = re.search(r'(?<!val_)loss: ([\d.]+)', line)
                            val_loss_match = re.search(r'val_loss: ([\d.]+)', line)
                            time_match = re.search(r'(\d+)s', line)

                            if acc_match and val_acc_match:
                                current_epoch += 1
                                acc = float(acc_match.group(1))
                                val_acc = float(val_acc_match.group(1))
                                loss = float(loss_match.group(1)) if loss_match else 0
                                val_loss = float(val_loss_match.group(1)) if val_loss_match else 0
                                epoch_time = int(time_match.group(1)) if time_match else 0

                                if epoch_time > 0:
                                    epoch_times.append(epoch_time)

                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "-"*60)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] EPOCH {current_epoch} COMPLETED (Phase {current_phase})")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "-"*60)
                                print(f"  Train Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
                                print(f"  Train Loss:          {loss:.4f}")
                                print(f"  Val Accuracy:        {val_acc:.4f} ({val_acc*100:.2f}%)")
                                print(f"  Val Loss:            {val_loss:.4f}")
                                print(f"  Time:                {epoch_time}s ({epoch_time/60:.1f} min)")

                                if len(epoch_times) > 1:
                                    avg_time = sum(epoch_times) / len(epoch_times)
                                    print(f"  Avg Epoch Time:      {avg_time:.0f}s ({avg_time/60:.1f} min)")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "-"*60)

                        # Detect phase completion
                        if "Phase 1 completed" in line or "Phase 2 completed" in line:
                            time_match = re.search(r'([\d.]+) minutes', line)
                            if time_match:
                                phase_time = float(time_match.group(1))
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] PHASE {current_phase} COMPLETED!")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Total Time: {phase_time:.2f} minutes ({phase_time/60:.2f} hours)")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)

                        # Detect best validation accuracy
                        if "Best validation accuracy:" in line:
                            acc_match = re.search(r'([\d.]+)', line)
                            if acc_match:
                                best_acc = float(acc_match.group(1))
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] Best Val Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")

                        # Detect evaluation start
                        if "MODEL EVALUATION" in line:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] STARTING MODEL EVALUATION ON TEST SET")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)

                        # Detect test accuracy
                        if "Test Accuracy:" in line:
                            acc_match = re.search(r'([\d.]+)', line)
                            if acc_match:
                                test_acc = float(acc_match.group(1))
                                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] TEST ACCURACY: {test_acc:.4f} ({test_acc*100:.2f}%)")
                                print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*60)

                        # Detect training completion
                        if "TRAINING COMPLETE" in line:
                            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] " + "="*70)
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] *** TRAINING COMPLETED SUCCESSFULLY! ***")
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] " + "="*70)
                            print(f"\nMonitor stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            exit(0)

        time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:
        print(f"\n\nMonitor stopped by user at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        break
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(CHECK_INTERVAL)

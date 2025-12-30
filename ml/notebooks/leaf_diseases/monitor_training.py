#!/usr/bin/env python3
"""
Training Progress Monitor
Checks training log every 10 minutes and saves progress updates
"""

import time
import re
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent / "training_output.log"
PROGRESS_FILE = Path(__file__).parent / "training_progress.txt"
CHECK_INTERVAL = 600  # 10 minutes in seconds

def extract_progress(log_content):
    """Extract key progress information from log."""
    progress = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'current_cell': 'Unknown',
        'current_epoch': None,
        'phase': None,
        'loss': None,
        'accuracy': None,
        'val_loss': None,
        'val_accuracy': None,
        'status': 'Running'
    }

    # Check if training completed
    if 'TRAINING COMPLETE' in log_content or '✅' in log_content:
        progress['status'] = 'Completed'

    # Extract current cell being executed
    cell_match = re.search(r'Executing cell (\d+)', log_content)
    if cell_match:
        progress['current_cell'] = f"Cell {cell_match.group(1)}"

    # Extract epoch information
    epoch_matches = re.findall(r'Epoch (\d+)/(\d+)', log_content)
    if epoch_matches:
        current, total = epoch_matches[-1]
        progress['current_epoch'] = f"{current}/{total}"

    # Determine phase
    if 'PHASE 1' in log_content or 'FEATURE EXTRACTION' in log_content:
        progress['phase'] = 'Phase 1: Feature Extraction'
    elif 'PHASE 2' in log_content or 'FINE-TUNING' in log_content:
        progress['phase'] = 'Phase 2: Fine-Tuning'

    # Extract latest metrics
    metric_pattern = r'- loss: ([\d.]+) - accuracy: ([\d.]+) - .*val_loss: ([\d.]+) - val_accuracy: ([\d.]+)'
    metric_matches = re.findall(metric_pattern, log_content)
    if metric_matches:
        loss, acc, val_loss, val_acc = metric_matches[-1]
        progress['loss'] = float(loss)
        progress['accuracy'] = float(acc)
        progress['val_loss'] = float(val_loss)
        progress['val_accuracy'] = float(val_acc)

    return progress

def format_progress_message(progress):
    """Format progress as readable message."""
    lines = []
    lines.append("="*60)
    lines.append(f" TRAINING PROGRESS UPDATE - {progress['timestamp']}")
    lines.append("="*60)
    lines.append(f"Status: {progress['status']}")
    lines.append(f"Current Cell: {progress['current_cell']}")

    if progress['phase']:
        lines.append(f"Phase: {progress['phase']}")

    if progress['current_epoch']:
        lines.append(f"Epoch: {progress['current_epoch']}")

    if progress['accuracy']:
        lines.append("")
        lines.append("Latest Metrics:")
        lines.append(f"  Training   - Loss: {progress['loss']:.4f}, Accuracy: {progress['accuracy']:.4f}")
        lines.append(f"  Validation - Loss: {progress['val_loss']:.4f}, Accuracy: {progress['val_accuracy']:.4f}")

        # Overfitting check
        acc_gap = abs(progress['accuracy'] - progress['val_accuracy'])
        lines.append("")
        if acc_gap < 0.05:
            lines.append(f"  ✅ No overfitting (gap: {acc_gap:.4f})")
        elif acc_gap < 0.10:
            lines.append(f"  ✅ Minimal overfitting (gap: {acc_gap:.4f})")
        else:
            lines.append(f"  ⚠️ Possible overfitting (gap: {acc_gap:.4f})")

    lines.append("="*60)
    lines.append("")

    return "\n".join(lines)

def main():
    """Main monitoring loop."""
    print(f"Training Monitor Started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Checking every {CHECK_INTERVAL // 60} minutes...")
    print(f"Log file: {LOG_FILE}")
    print(f"Progress file: {PROGRESS_FILE}")
    print("")

    iteration = 0

    while True:
        iteration += 1

        try:
            # Read log file
            if LOG_FILE.exists():
                with open(LOG_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                    log_content = f.read()

                # Extract progress
                progress = extract_progress(log_content)

                # Format message
                message = format_progress_message(progress)

                # Save to progress file (append mode)
                with open(PROGRESS_FILE, 'a', encoding='utf-8') as f:
                    f.write(message)

                # Print to console
                print(message)

                # Check if completed
                if progress['status'] == 'Completed':
                    print("🎉 Training completed! Stopping monitor.")
                    break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for log file to be created...")

        except Exception as e:
            print(f"[ERROR] {e}")

        # Wait before next check
        print(f"Next check in {CHECK_INTERVAL // 60} minutes... (Iteration {iteration})")
        print("")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

"""Verify all image files in dataset exist and are valid"""
import os
from pathlib import Path

base_dir = r"D:\SLIIT\Reaserch Project\CoconutHealthMonitor\Research\ml\data\raw\pest_mite\dataset_v3_clean"

for split in ['train', 'validation', 'test']:
    for cls in ['coconut_mite', 'healthy', 'not_coconut']:
        folder = Path(base_dir) / split / cls
        if folder.exists():
            files = list(folder.glob('*'))
            valid = 0
            invalid = 0
            for f in files:
                if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    if f.exists() and f.stat().st_size > 0:
                        valid += 1
                    else:
                        invalid += 1
                        print(f"Invalid: {f}")
            print(f"{split}/{cls}: {valid} valid, {invalid} invalid")

print("\nDone!")

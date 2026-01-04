"""
Clean dataset by removing files with excessively long names that Windows can't handle.
Windows has a 260 character path limit.
"""

import os
from pathlib import Path

# Dataset directory
DATA_DIR = Path('../../data/raw/stage_1')

# Maximum filename length (leaving room for path)
MAX_FILENAME_LENGTH = 150

def clean_long_filenames(data_dir, max_length=150):
    """Remove files with filenames longer than max_length."""
    removed_files = []

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if len(filename) > max_length:
                filepath = os.path.join(root, filename)
                try:
                    os.remove(filepath)
                    removed_files.append(filepath)
                    print(f"Removed: {filename[:80]}... (length: {len(filename)})")
                except Exception as e:
                    print(f"Error removing {filename}: {e}")

    return removed_files

print("="*60)
print(" CLEANING DATASET - REMOVING LONG FILENAMES")
print("="*60)
print(f"Max filename length: {MAX_FILENAME_LENGTH} characters\n")

removed = clean_long_filenames(DATA_DIR, MAX_FILENAME_LENGTH)

print(f"\n{'='*60}")
print(f"CLEANUP COMPLETE!")
print(f"{'='*60}")
print(f"Total files removed: {len(removed)}")

if removed:
    print("\nRemoved files:")
    for f in removed:
        print(f"  - {f}")
else:
    print("\nNo files with long names found!")

print("\nDataset is now clean and ready for training!")

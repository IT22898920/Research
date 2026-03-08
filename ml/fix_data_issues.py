"""
Fix Data Issues Script
- Remove corrupted files
- Fix data leaking (remove duplicates from val/test)
- Create clean dataset v4
"""

import os
import shutil
import hashlib
from pathlib import Path
from PIL import Image
from collections import defaultdict

# Configuration
BASE_DIR = Path("D:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/ml")
SOURCE_DIR = BASE_DIR / "data" / "raw" / "pest_mite" / "dataset_v3_clean"
TARGET_DIR = BASE_DIR / "data" / "raw" / "pest_mite" / "dataset_v4_clean"

CLASSES = ['coconut_mite', 'healthy', 'not_coconut']
SPLITS = ['train', 'validation', 'test']

def get_file_hash(filepath):
    """Get MD5 hash of file content."""
    hasher = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            buf = f.read(65536)
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except:
        return None

def is_valid_image(filepath):
    """Check if image is valid and can be opened."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        # Re-open to actually load (verify doesn't load pixels)
        with Image.open(filepath) as img:
            img.load()
        return True
    except:
        return False

def fix_dataset():
    """Create clean dataset without duplicates and corrupted files."""
    print("=" * 70)
    print("  CREATING CLEAN DATASET v4")
    print("=" * 70)

    # Step 1: Collect all train hashes first (train is priority)
    print("\n[1] Collecting train image hashes (priority)...")
    train_hashes = set()

    for cls in CLASSES:
        train_class_dir = SOURCE_DIR / "train" / cls
        if train_class_dir.exists():
            for img_file in train_class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    file_hash = get_file_hash(img_file)
                    if file_hash:
                        train_hashes.add(file_hash)

    print(f"  Train images: {len(train_hashes)} unique hashes")

    # Step 2: Create target directories
    print("\n[2] Creating target directory structure...")
    if TARGET_DIR.exists():
        shutil.rmtree(TARGET_DIR)

    for split in SPLITS:
        for cls in CLASSES:
            (TARGET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    print(f"  Created: {TARGET_DIR}")

    # Step 3: Copy files with filtering
    print("\n[3] Copying files with filtering...")

    stats = {
        'copied': defaultdict(lambda: defaultdict(int)),
        'corrupted': 0,
        'duplicate_removed': 0,
        'total_processed': 0
    }

    for split in SPLITS:
        split_dir = SOURCE_DIR / split
        seen_hashes_in_split = set()  # Track within-split duplicates too

        for cls in CLASSES:
            class_dir = split_dir / cls
            if not class_dir.exists():
                continue

            file_counter = 1
            for img_file in sorted(class_dir.iterdir()):
                if img_file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    continue

                stats['total_processed'] += 1

                # Check if image is valid
                if not is_valid_image(img_file):
                    stats['corrupted'] += 1
                    print(f"    SKIP (corrupted): {img_file.name[:50]}...")
                    continue

                # Get hash
                file_hash = get_file_hash(img_file)
                if not file_hash:
                    continue

                # For val/test: skip if exists in train (data leak)
                if split in ['validation', 'test']:
                    if file_hash in train_hashes:
                        stats['duplicate_removed'] += 1
                        print(f"    SKIP (in train): {split}/{cls}/{img_file.name[:30]}...")
                        continue

                # Skip within-split duplicates
                if file_hash in seen_hashes_in_split:
                    stats['duplicate_removed'] += 1
                    continue

                seen_hashes_in_split.add(file_hash)

                # Copy with clean filename
                new_name = f"{file_counter:04d}{img_file.suffix.lower()}"
                target_path = TARGET_DIR / split / cls / new_name
                shutil.copy2(img_file, target_path)
                stats['copied'][split][cls] += 1
                file_counter += 1

    # Step 4: Print summary
    print("\n" + "=" * 70)
    print("  CLEAN DATASET CREATED")
    print("=" * 70)

    print(f"\n  [Files Processed]")
    print(f"    Total processed: {stats['total_processed']}")
    print(f"    Corrupted removed: {stats['corrupted']}")
    print(f"    Duplicates removed: {stats['duplicate_removed']}")

    print(f"\n  [Final Counts]")
    print(f"  {'Split':<12} {'coconut_mite':>14} {'healthy':>12} {'not_coconut':>14} {'Total':>10}")
    print("  " + "-" * 66)

    total_all = 0
    for split in SPLITS:
        mite = stats['copied'][split]['coconut_mite']
        healthy = stats['copied'][split]['healthy']
        not_coco = stats['copied'][split]['not_coconut']
        total = mite + healthy + not_coco
        total_all += total
        print(f"  {split:<12} {mite:>14} {healthy:>12} {not_coco:>14} {total:>10}")

    print("  " + "-" * 66)
    print(f"  {'TOTAL':<12} {sum(stats['copied'][s]['coconut_mite'] for s in SPLITS):>14} "
          f"{sum(stats['copied'][s]['healthy'] for s in SPLITS):>12} "
          f"{sum(stats['copied'][s]['not_coconut'] for s in SPLITS):>14} {total_all:>10}")

    print(f"\n  Dataset saved to: {TARGET_DIR}")
    print("=" * 70)

    return stats

if __name__ == "__main__":
    fix_dataset()

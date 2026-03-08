"""
Data Audit Script for Coconut Mite 3-Class Model
Checks for: Data leaking, duplicates, class imbalance, image quality
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
from PIL import Image
import json

# Configuration
BASE_DIR = Path("D:/SLIIT/Reaserch Project/CoconutHealthMonitor/Research/ml")
DATA_DIR = BASE_DIR / "data" / "raw" / "pest_mite" / "dataset_v3_clean"

CLASSES = ['coconut_mite', 'healthy', 'not_coconut']
SPLITS = ['train', 'validation', 'test']

def get_file_hash(filepath):
    """Get MD5 hash of file content to detect duplicates."""
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

def get_image_info(filepath):
    """Get image dimensions and basic info."""
    try:
        with Image.open(filepath) as img:
            return {
                'size': img.size,
                'mode': img.mode,
                'format': img.format
            }
    except Exception as e:
        return {'error': str(e)}

def audit_dataset():
    """Main audit function."""
    print("=" * 70)
    print("  DATA AUDIT - Coconut Mite 3-Class Dataset")
    print("=" * 70)

    # Storage for analysis
    all_hashes = {}  # hash -> [(split, class, filepath), ...]
    class_counts = defaultdict(lambda: defaultdict(int))
    image_sizes = defaultdict(list)
    corrupted_files = []

    # 1. Scan all images
    print("\n[1] Scanning all images...")
    total_images = 0

    for split in SPLITS:
        split_dir = DATA_DIR / split
        if not split_dir.exists():
            print(f"  WARNING: {split} directory not found!")
            continue

        for cls in CLASSES:
            class_dir = split_dir / cls
            if not class_dir.exists():
                print(f"  WARNING: {split}/{cls} directory not found!")
                continue

            for img_file in class_dir.iterdir():
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    total_images += 1
                    class_counts[split][cls] += 1

                    # Get hash for duplicate detection
                    file_hash = get_file_hash(img_file)
                    if file_hash:
                        if file_hash not in all_hashes:
                            all_hashes[file_hash] = []
                        all_hashes[file_hash].append((split, cls, str(img_file)))

                    # Get image info
                    img_info = get_image_info(img_file)
                    if 'error' in img_info:
                        corrupted_files.append(str(img_file))
                    else:
                        image_sizes[f"{split}/{cls}"].append(img_info['size'])

    print(f"  Total images scanned: {total_images}")

    # 2. Class Distribution
    print("\n[2] CLASS DISTRIBUTION")
    print("-" * 70)
    print(f"{'Split':<12} {'coconut_mite':>14} {'healthy':>12} {'not_coconut':>14} {'Total':>10}")
    print("-" * 70)

    for split in SPLITS:
        mite = class_counts[split]['coconut_mite']
        healthy = class_counts[split]['healthy']
        not_coco = class_counts[split]['not_coconut']
        total = mite + healthy + not_coco
        print(f"{split:<12} {mite:>14} {healthy:>12} {not_coco:>14} {total:>10}")

    # Totals
    total_mite = sum(class_counts[s]['coconut_mite'] for s in SPLITS)
    total_healthy = sum(class_counts[s]['healthy'] for s in SPLITS)
    total_not_coco = sum(class_counts[s]['not_coconut'] for s in SPLITS)
    print("-" * 70)
    print(f"{'TOTAL':<12} {total_mite:>14} {total_healthy:>12} {total_not_coco:>14} {total_mite+total_healthy+total_not_coco:>10}")

    # 3. Check for Data Leaking (duplicates across splits)
    print("\n[3] DATA LEAKING CHECK")
    print("-" * 70)

    cross_split_duplicates = []
    within_split_duplicates = []
    cross_class_duplicates = []  # Same image in different classes!

    for file_hash, locations in all_hashes.items():
        if len(locations) > 1:
            splits = set(loc[0] for loc in locations)
            classes = set(loc[1] for loc in locations)

            if len(classes) > 1:
                # CRITICAL: Same image in different classes
                cross_class_duplicates.append(locations)
            elif len(splits) > 1:
                # Data leaking: same image in train AND val/test
                cross_split_duplicates.append(locations)
            else:
                # Duplicate within same split/class (less critical)
                within_split_duplicates.append(locations)

    if cross_class_duplicates:
        print(f"\n  CRITICAL: {len(cross_class_duplicates)} images in MULTIPLE CLASSES!")
        for locs in cross_class_duplicates[:5]:
            print(f"    - {locs}")
    else:
        print("  Cross-class duplicates: 0 (GOOD)")

    if cross_split_duplicates:
        print(f"\n  WARNING: {len(cross_split_duplicates)} images in MULTIPLE SPLITS (DATA LEAK!)")
        for locs in cross_split_duplicates[:5]:
            print(f"    - {locs}")
    else:
        print("  Cross-split duplicates: 0 (GOOD)")

    print(f"  Within-split duplicates: {len(within_split_duplicates)}")

    # 4. Corrupted files
    print("\n[4] CORRUPTED FILES")
    print("-" * 70)
    if corrupted_files:
        print(f"  WARNING: {len(corrupted_files)} corrupted files found!")
        for f in corrupted_files[:10]:
            print(f"    - {f}")
    else:
        print("  Corrupted files: 0 (GOOD)")

    # 5. Validation/Test set size check
    print("\n[5] VALIDATION/TEST SET SIZE CHECK")
    print("-" * 70)

    for split in ['validation', 'test']:
        for cls in CLASSES:
            count = class_counts[split][cls]
            if count < 50:
                print(f"  WARNING: {split}/{cls} has only {count} images (recommend >= 50)")
            elif count < 100:
                print(f"  INFO: {split}/{cls} has {count} images (OK, but 100+ is better)")
            else:
                print(f"  GOOD: {split}/{cls} has {count} images")

    # 6. Class Imbalance Analysis
    print("\n[6] CLASS IMBALANCE ANALYSIS (Training Set)")
    print("-" * 70)

    train_total = sum(class_counts['train'].values())
    for cls in CLASSES:
        count = class_counts['train'][cls]
        pct = (count / train_total * 100) if train_total > 0 else 0
        ratio = count / min(class_counts['train'].values()) if min(class_counts['train'].values()) > 0 else 0
        status = "BALANCED" if 0.8 <= ratio <= 1.25 else "IMBALANCED"
        print(f"  {cls}: {count} ({pct:.1f}%) - Ratio: {ratio:.2f}x - {status}")

    # 7. Summary
    print("\n" + "=" * 70)
    print("  AUDIT SUMMARY")
    print("=" * 70)

    issues = []
    if cross_class_duplicates:
        issues.append(f"CRITICAL: {len(cross_class_duplicates)} cross-class duplicates")
    if cross_split_duplicates:
        issues.append(f"DATA LEAK: {len(cross_split_duplicates)} cross-split duplicates")
    if corrupted_files:
        issues.append(f"CORRUPTED: {len(corrupted_files)} files")

    # Check for small val/test sets
    for split in ['validation', 'test']:
        for cls in CLASSES:
            if class_counts[split][cls] < 30:
                issues.append(f"TOO SMALL: {split}/{cls} has only {class_counts[split][cls]} images")

    if issues:
        print("\n  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No major issues found!")

    # Save audit results
    audit_results = {
        'total_images': total_images,
        'class_counts': dict(class_counts),
        'cross_class_duplicates': len(cross_class_duplicates),
        'cross_split_duplicates': len(cross_split_duplicates),
        'within_split_duplicates': len(within_split_duplicates),
        'corrupted_files': len(corrupted_files),
        'issues': issues
    }

    with open(BASE_DIR / 'data_audit_results.json', 'w') as f:
        json.dump(audit_results, f, indent=2, default=str)

    print(f"\n  Audit results saved to: {BASE_DIR / 'data_audit_results.json'}")

    return audit_results

if __name__ == "__main__":
    audit_dataset()

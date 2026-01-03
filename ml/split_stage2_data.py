"""
Split stage_2 images into train/test/val folders with 80-10-10 ratio
Maintains class distribution across all splits
"""

import os
import shutil
import random
from pathlib import Path

# Configuration
SOURCE_DIR = Path("data/raw/stage_2")
OUTPUT_DIR = Path("data/raw/stage_2_split")
SPLIT_RATIOS = {"train": 0.8, "test": 0.1, "val": 0.1}
RANDOM_SEED = 42

# Classes in stage_2
CLASSES = ["healthy", "leaf die back", "Leaf Rot", "Leaf_Spot"]

def create_folder_structure():
    """Create train/test/val folder structure for each class"""
    print("Creating folder structure...")
    for split in SPLIT_RATIOS.keys():
        for class_name in CLASSES:
            folder_path = OUTPUT_DIR / split / class_name
            folder_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {folder_path}")

def split_class_data(class_name):
    """Split images of a single class into train/test/val"""
    source_path = SOURCE_DIR / class_name

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    all_images = [f for f in source_path.iterdir()
                  if f.is_file() and f.suffix in image_extensions]

    # Shuffle with seed for reproducibility
    random.seed(RANDOM_SEED)
    random.shuffle(all_images)

    total = len(all_images)
    train_count = int(total * SPLIT_RATIOS["train"])
    test_count = int(total * SPLIT_RATIOS["test"])

    # Split indices
    train_images = all_images[:train_count]
    test_images = all_images[train_count:train_count + test_count]
    val_images = all_images[train_count + test_count:]

    # Copy files to respective folders
    splits = {
        "train": train_images,
        "test": test_images,
        "val": val_images
    }

    print(f"\n{class_name}:")
    print(f"  Total images: {total}")

    for split_name, images in splits.items():
        dest_dir = OUTPUT_DIR / split_name / class_name
        for img in images:
            shutil.copy2(img, dest_dir / img.name)
        print(f"  {split_name}: {len(images)} images")

    return {
        "class": class_name,
        "total": total,
        "train": len(train_images),
        "test": len(test_images),
        "val": len(val_images)
    }

def main():
    print("="*60)
    print("Stage 2 Data Split Script")
    print(f"Split Ratio - Train: {SPLIT_RATIOS['train']*100}%, "
          f"Test: {SPLIT_RATIOS['test']*100}%, "
          f"Val: {SPLIT_RATIOS['val']*100}%")
    print("="*60)

    # Create folder structure
    create_folder_structure()

    # Split each class
    print("\nSplitting images...")
    results = []
    for class_name in CLASSES:
        result = split_class_data(class_name)
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Class':<20} {'Total':<8} {'Train':<8} {'Test':<8} {'Val':<8}")
    print("-"*60)

    totals = {"total": 0, "train": 0, "test": 0, "val": 0}
    for r in results:
        print(f"{r['class']:<20} {r['total']:<8} {r['train']:<8} {r['test']:<8} {r['val']:<8}")
        totals["total"] += r["total"]
        totals["train"] += r["train"]
        totals["test"] += r["test"]
        totals["val"] += r["val"]

    print("-"*60)
    print(f"{'TOTAL':<20} {totals['total']:<8} {totals['train']:<8} {totals['test']:<8} {totals['val']:<8}")
    print("="*60)
    print(f"\nOutput directory: {OUTPUT_DIR.absolute()}")
    print("Split completed successfully!")

if __name__ == "__main__":
    main()

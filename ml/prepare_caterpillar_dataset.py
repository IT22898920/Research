"""
Caterpillar Dataset Preparation Script
=======================================
Creates train/validation/test split with augmentation
- 70% train (20x augmented), 15% validation, 15% test
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime

# Configuration
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
AUGMENTATIONS_PER_IMAGE = 20

# Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw', 'pest_caterpillar')

SOURCE_FOLDERS = {
    'caterpillar': os.path.join(DATA_DIR, 'coconut_caterpillar_original'),
    'healthy': os.path.join(DATA_DIR, 'healthy_leaves_original')
}

OUTPUT_DIR = os.path.join(DATA_DIR, 'dataset')


def augment_image(img):
    """Apply random augmentations."""

    if random.random() > 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

    if random.random() > 0.7:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    if random.random() > 0.3:
        angle = random.uniform(-20, 20)
        img = img.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))

    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.75, 1.25))

    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))

    enhancer = ImageEnhance.Color(img)
    img = enhancer.enhance(random.uniform(0.85, 1.15))

    if random.random() > 0.5:
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.4:
        w, h = img.size
        zoom = random.uniform(0.05, 0.20)
        crop_px_w = int(w * zoom)
        crop_px_h = int(h * zoom)
        left = random.randint(0, crop_px_w)
        top = random.randint(0, crop_px_h)
        right = w - random.randint(0, crop_px_w)
        bottom = h - random.randint(0, crop_px_h)
        if right > left and bottom > top:
            img = img.crop((left, top, right, bottom))
            img = img.resize((w, h), Image.BICUBIC)

    if random.random() > 0.90:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 0.8)))

    if random.random() > 0.85:
        img_array = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(2, 5), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    return img


def get_images(folder):
    """Get all image files from folder."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    images = []
    if os.path.exists(folder):
        for f in os.listdir(folder):
            if any(f.endswith(ext) for ext in extensions):
                images.append(os.path.join(folder, f))
    return sorted(images)


def process_class(class_name, source_folder, train_dir, val_dir, test_dir):
    """Process a single class."""

    print(f"\n{'='*50}")
    print(f"  Processing: {class_name.upper()}")
    print(f"{'='*50}")

    images = get_images(source_folder)
    if len(images) == 0:
        print(f"  ERROR: No images found!")
        return None

    print(f"  Found {len(images)} original images")

    random.seed(42)
    shuffled = images.copy()
    random.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * TRAIN_RATIO)
    n_val = int(n_total * VAL_RATIO)

    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:n_train + n_val]
    test_images = shuffled[n_train + n_val:]

    print(f"  Split: train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")

    stats = {'original_train': 0, 'augmented': 0, 'val': 0, 'test': 0}

    # Training set with augmentation
    print(f"\n  Creating training set (with {AUGMENTATIONS_PER_IMAGE}x augmentation)...")
    img_counter = 1

    for idx, img_path in enumerate(train_images, 1):
        try:
            img = Image.open(img_path).convert('RGB')

            # Save original
            img.save(os.path.join(train_dir, f"{img_counter}.jpg"), 'JPEG', quality=95)
            stats['original_train'] += 1
            img_counter += 1

            # Create augmented versions
            for aug_idx in range(AUGMENTATIONS_PER_IMAGE):
                aug_img = augment_image(img.copy())
                aug_img.save(os.path.join(train_dir, f"{img_counter}.jpg"), 'JPEG', quality=92)
                stats['augmented'] += 1
                img_counter += 1

            if idx % 50 == 0:
                print(f"    Processed {idx}/{len(train_images)}...")

        except Exception as e:
            print(f"    Error: {e}")

    # Validation set (no augmentation)
    print(f"  Creating validation set...")
    img_counter = 1
    for img_path in val_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(val_dir, f"{img_counter}.jpg"), 'JPEG', quality=95)
            stats['val'] += 1
            img_counter += 1
        except Exception as e:
            print(f"    Error: {e}")

    # Test set (no augmentation)
    print(f"  Creating test set...")
    img_counter = 1
    for img_path in test_images:
        try:
            img = Image.open(img_path).convert('RGB')
            img.save(os.path.join(test_dir, f"{img_counter}.jpg"), 'JPEG', quality=95)
            stats['test'] += 1
            img_counter += 1
        except Exception as e:
            print(f"    Error: {e}")

    total_train = stats['original_train'] + stats['augmented']
    print(f"\n  {class_name} complete!")
    print(f"    Train: {stats['original_train']} + {stats['augmented']} augmented = {total_train}")
    print(f"    Validation: {stats['val']}")
    print(f"    Test: {stats['test']}")

    return stats


def main():
    print("\n" + "=" * 60)
    print("  CATERPILLAR DATASET PREPARATION")
    print("  70% Train (20x Aug) / 15% Validation / 15% Test")
    print("=" * 60)

    start = datetime.now()

    # Clean and create folders
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)

    for split in ['train', 'validation', 'test']:
        for cls in ['caterpillar', 'healthy']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, cls), exist_ok=True)

    print("\n  Folder structure created!")

    all_stats = {}

    for class_name, source_folder in SOURCE_FOLDERS.items():
        train_dir = os.path.join(OUTPUT_DIR, 'train', class_name)
        val_dir = os.path.join(OUTPUT_DIR, 'validation', class_name)
        test_dir = os.path.join(OUTPUT_DIR, 'test', class_name)

        stats = process_class(class_name, source_folder, train_dir, val_dir, test_dir)
        if stats:
            all_stats[class_name] = stats

    elapsed = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 60)
    print("  DATASET PREPARATION COMPLETE!")
    print("=" * 60)

    print(f"\n  Time: {elapsed:.1f} seconds")
    print(f"\n  Output: {OUTPUT_DIR}")

    total_train = sum(s['original_train'] + s['augmented'] for s in all_stats.values())
    total_val = sum(s['val'] for s in all_stats.values())
    total_test = sum(s['test'] for s in all_stats.values())

    print(f"\n  SUMMARY:")
    print(f"    Train:      {total_train} images")
    print(f"    Validation: {total_val} images")
    print(f"    Test:       {total_test} images")
    print(f"    Total:      {total_train + total_val + total_test} images")


if __name__ == '__main__':
    main()

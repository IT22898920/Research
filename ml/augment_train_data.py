"""
Data Augmentation Script for Stage 2 Train Data
Augments each class to reach target of ~5000 images
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import random

# Configuration
TRAIN_DIR = Path("data/raw/stage_2_split/train")
TARGET_COUNT = 5000
RANDOM_SEED = 42

# Set random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

def augment_image(image):
    """Apply random augmentation to an image"""

    # Random rotation (-30 to 30 degrees)
    if random.random() > 0.5:
        angle = random.uniform(-30, 30)
        image = image.rotate(angle, fillcolor=(0, 0, 0))

    # Random horizontal flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    # Random vertical flip
    if random.random() > 0.5:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # Random brightness adjustment
    if random.random() > 0.5:
        brightness_factor = random.uniform(0.7, 1.3)
        image = Image.fromarray(
            np.clip(np.array(image) * brightness_factor, 0, 255).astype(np.uint8)
        )

    # Random zoom (crop and resize)
    if random.random() > 0.5:
        width, height = image.size
        zoom_factor = random.uniform(0.8, 0.95)

        new_width = int(width * zoom_factor)
        new_height = int(height * zoom_factor)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)

        image = image.crop((left, top, left + new_width, top + new_height))
        image = image.resize((width, height), Image.LANCZOS)

    return image

def get_image_files(class_dir):
    """Get all image files in a directory"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return [f for f in class_dir.iterdir()
            if f.is_file() and f.suffix in image_extensions]

def augment_class(class_name):
    """Augment images in a class to reach target count"""
    class_dir = TRAIN_DIR / class_name

    # Get current images
    original_images = get_image_files(class_dir)
    current_count = len(original_images)

    print(f"\n{class_name}:")
    print(f"  Current images: {current_count}")

    if current_count >= TARGET_COUNT:
        print(f"  Already has {current_count} images (>= {TARGET_COUNT})")
        print(f"  No augmentation needed")
        return current_count

    needed = TARGET_COUNT - current_count
    print(f"  Need to generate: {needed} augmented images")

    # Generate augmented images
    generated = 0
    while generated < needed:
        # Pick a random original image
        source_img_path = random.choice(original_images)

        try:
            # Load and augment
            img = Image.open(source_img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            augmented_img = augment_image(img)

            # Save with new name
            base_name = source_img_path.stem
            new_name = f"{base_name}_aug_{generated}{source_img_path.suffix}"
            new_path = class_dir / new_name

            augmented_img.save(new_path, quality=95)
            generated += 1

            if generated % 100 == 0:
                print(f"  Generated {generated}/{needed} images...")

        except Exception as e:
            print(f"  Error processing {source_img_path.name}: {e}")
            continue

    final_count = len(get_image_files(class_dir))
    print(f"  Final count: {final_count}")
    return final_count

def main():
    print("="*60)
    print("Data Augmentation Script")
    print(f"Target: {TARGET_COUNT} images per class")
    print("="*60)

    # Get all class directories
    classes = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]
    classes.sort()

    print(f"\nFound {len(classes)} classes: {', '.join(classes)}")

    # Augment each class
    results = {}
    for class_name in classes:
        results[class_name] = augment_class(class_name)

    # Summary
    print("\n" + "="*60)
    print("AUGMENTATION SUMMARY")
    print("="*60)
    print(f"{'Class':<20} {'Final Count':<15}")
    print("-"*60)

    total = 0
    for class_name, count in results.items():
        print(f"{class_name:<20} {count:<15}")
        total += count

    print("-"*60)
    print(f"{'TOTAL':<20} {total:<15}")
    print("="*60)
    print("\nAugmentation completed successfully!")

if __name__ == "__main__":
    main()

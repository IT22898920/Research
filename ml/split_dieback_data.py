"""Split leaf die back data into train/val/test"""
import os
import shutil
from sklearn.model_selection import train_test_split

# Paths - use raw strings for Windows
BASE_DIR = r"C:\Users\Tharindu Nandun\Desktop\Research\Research\ml"
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "leaf die back")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "leaf_dieback_v1")

# Remove old if exists
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)

# Class mapping
class_mapping = {'healthy': 'healthy', 'unhealthy': 'leaf_die_back'}

# Create output directories
for split in ['train', 'val', 'test']:
    for class_name in class_mapping.values():
        os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

# Split ratios
VAL_RATIO = 0.15
TEST_RATIO = 0.15

print("=" * 50)
print("SPLITTING LEAF DIE BACK DATA")
print("=" * 50)

results = {}

for orig_class, mapped_class in class_mapping.items():
    class_path = os.path.join(DATA_DIR, orig_class)
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Split
    train_val, test = train_test_split(images, test_size=TEST_RATIO, random_state=42)
    val_ratio_adjusted = VAL_RATIO / (1 - TEST_RATIO)
    train, val = train_test_split(train_val, test_size=val_ratio_adjusted, random_state=42)

    # Copy files
    for img in train:
        shutil.copy2(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, 'train', mapped_class, img))
    for img in val:
        shutil.copy2(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, 'val', mapped_class, img))
    for img in test:
        shutil.copy2(os.path.join(class_path, img), os.path.join(OUTPUT_DIR, 'test', mapped_class, img))

    results[mapped_class] = {'train': len(train), 'val': len(val), 'test': len(test)}
    print(f'{mapped_class}: Train={len(train)}, Val={len(val)}, Test={len(test)}')

print()
total_train = sum(r['train'] for r in results.values())
total_val = sum(r['val'] for r in results.values())
total_test = sum(r['test'] for r in results.values())
print(f'Total: Train={total_train}, Val={total_val}, Test={total_test}')
print(f'\nSaved to: {OUTPUT_DIR}')
print("\nDONE!")

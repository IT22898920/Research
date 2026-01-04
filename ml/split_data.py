import os, shutil, random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

src = Path(r'C:\Users\Tharindu Nandun\Desktop\Research\Research\ml\data\raw\leaf die back')
dst = Path(r'C:\Users\Tharindu Nandun\Desktop\Research\Research\ml\data\processed\leaf_dieback_v1')

if dst.exists():
    shutil.rmtree(dst)

for split in ['train', 'val', 'test']:
    for cls in ['healthy', 'leaf_die_back']:
        (dst / split / cls).mkdir(parents=True, exist_ok=True)

mapping = {'healthy': 'healthy', 'unhealthy': 'leaf_die_back'}

# Target: ~3000 per class, ~2850 in train + ~150 in val/test
TARGET_TRAIN_PER_CLASS = 2850

# Get all images per class
all_imgs = {}
for orig, mapped in mapping.items():
    folder = src / orig
    imgs = [f.name for f in folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    all_imgs[mapped] = {'folder': folder, 'imgs': imgs}
    print(f'{mapped}: {len(imgs)} original images')

# More augmentation functions with ZOOM
def augment_image(img):
    augmented = []
    w, h = img.size

    # Flips
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    augmented.append(img.transpose(Image.FLIP_TOP_BOTTOM))
    augmented.append(img.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM))

    # Rotations
    augmented.append(img.rotate(90, expand=True))
    augmented.append(img.rotate(180))
    augmented.append(img.rotate(270, expand=True))
    augmented.append(img.rotate(15, expand=True, fillcolor=(0,0,0)))
    augmented.append(img.rotate(30, expand=True, fillcolor=(0,0,0)))
    augmented.append(img.rotate(-15, expand=True, fillcolor=(0,0,0)))
    augmented.append(img.rotate(-30, expand=True, fillcolor=(0,0,0)))

    # ZOOM IN (crop center and resize back)
    for zoom_factor in [1.2, 1.4, 1.6]:
        crop_w = int(w / zoom_factor)
        crop_h = int(h / zoom_factor)
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        cropped = img.crop((left, top, left + crop_w, top + crop_h))
        zoomed = cropped.resize((w, h), Image.LANCZOS)
        augmented.append(zoomed)

    # ZOOM OUT (add border and resize)
    for zoom_factor in [0.8, 0.7]:
        new_w = int(w * zoom_factor)
        new_h = int(h * zoom_factor)
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        padded = Image.new('RGB', (w, h), (0, 0, 0))
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        padded.paste(resized, (paste_x, paste_y))
        augmented.append(padded)

    # Brightness variations
    enhancer = ImageEnhance.Brightness(img)
    augmented.append(enhancer.enhance(0.7))
    augmented.append(enhancer.enhance(0.85))
    augmented.append(enhancer.enhance(1.15))
    augmented.append(enhancer.enhance(1.3))

    # Contrast variations
    enhancer = ImageEnhance.Contrast(img)
    augmented.append(enhancer.enhance(0.7))
    augmented.append(enhancer.enhance(0.85))
    augmented.append(enhancer.enhance(1.15))
    augmented.append(enhancer.enhance(1.3))

    # Saturation
    enhancer = ImageEnhance.Color(img)
    augmented.append(enhancer.enhance(0.7))
    augmented.append(enhancer.enhance(1.3))

    # Sharpness
    enhancer = ImageEnhance.Sharpness(img)
    augmented.append(enhancer.enhance(0.5))
    augmented.append(enhancer.enhance(2.0))

    # Filters
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=1)))
    augmented.append(img.filter(ImageFilter.GaussianBlur(radius=2)))
    augmented.append(img.filter(ImageFilter.SHARPEN))
    augmented.append(img.filter(ImageFilter.EDGE_ENHANCE))

    # Mirror + zoom combo
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)
    crop_w = int(w / 1.3)
    crop_h = int(h / 1.3)
    left = (w - crop_w) // 2
    top = (h - crop_h) // 2
    cropped = flipped.crop((left, top, left + crop_w, top + crop_h))
    augmented.append(cropped.resize((w, h), Image.LANCZOS))

    return augmented

random.seed(42)

# First, split all classes
splits = {}
for mapped, data in all_imgs.items():
    imgs = data['imgs'].copy()
    random.shuffle(imgs)

    n = len(imgs)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    splits[mapped] = {
        'train': imgs[:train_end],
        'val': imgs[train_end:val_end],
        'test': imgs[val_end:]
    }

print(f'\nTarget train per class: {TARGET_TRAIN_PER_CLASS}')
print('Creating augmented images...\n')

# Process each class
total_images = 0

for mapped, data in all_imgs.items():
    folder = data['folder']
    train_imgs = splits[mapped]['train']
    val_imgs = splits[mapped]['val']
    test_imgs = splits[mapped]['test']

    current_train = len(train_imgs)
    need_more = TARGET_TRAIN_PER_CLASS - current_train

    print(f'{mapped}: {current_train} original train, need {need_more} augmented')

    # Copy original train images
    copied = {'train': 0, 'val': 0, 'test': 0}

    for i, img in enumerate(train_imgs):
        ext = Path(img).suffix
        src_file = folder / img
        dst_file = dst / 'train' / mapped / f'{mapped}_{i:04d}{ext}'
        try:
            shutil.copy2(str(src_file), str(dst_file))
            copied['train'] += 1
        except:
            continue

    # Augment to reach target
    if need_more > 0:
        aug_idx = 0
        aug_count = 0
        round_num = 0
        while aug_count < need_more:
            round_num += 1
            for img_name in train_imgs:
                if aug_count >= need_more:
                    break
                try:
                    img_path = folder / img_name
                    img = Image.open(str(img_path))
                    img = img.convert('RGB')

                    augs = augment_image(img)
                    random.shuffle(augs)

                    for aug_img in augs:
                        if aug_count >= need_more:
                            break
                        ext = Path(img_name).suffix
                        new_name = f'{mapped}_aug_{aug_idx:04d}{ext}'
                        dst_file = dst / 'train' / mapped / new_name

                        # Resize to consistent size
                        aug_img = aug_img.resize((224, 224), Image.LANCZOS)
                        aug_img.save(str(dst_file), quality=95)
                        aug_idx += 1
                        aug_count += 1
                        copied['train'] += 1

                    img.close()
                except:
                    continue

            if round_num % 2 == 0:
                print(f'  ... {aug_count}/{need_more} augmented')

    # Copy val images
    for i, img in enumerate(val_imgs):
        ext = Path(img).suffix
        src_file = folder / img
        dst_file = dst / 'val' / mapped / f'{mapped}_{i:04d}{ext}'
        try:
            shutil.copy2(str(src_file), str(dst_file))
            copied['val'] += 1
        except:
            continue

    # Copy test images
    for i, img in enumerate(test_imgs):
        ext = Path(img).suffix
        src_file = folder / img
        dst_file = dst / 'test' / mapped / f'{mapped}_{i:04d}{ext}'
        try:
            shutil.copy2(str(src_file), str(dst_file))
            copied['test'] += 1
        except:
            continue

    class_total = copied['train'] + copied['val'] + copied['test']
    print(f'  {mapped}: Train={copied["train"]}, Val={copied["val"]}, Test={copied["test"]} | Total={class_total}')
    total_images += class_total

print(f'\n{"="*50}')
print(f'TOTAL IMAGES: {total_images}')
print(f'{"="*50}')
print('\nDone!')

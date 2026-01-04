import os, shutil, random
from pathlib import Path

src = Path(r'C:\Users\Tharindu Nandun\Desktop\Research\Research\ml\data\raw\stage_1\train\not_cocount')
dst = Path(r'C:\Users\Tharindu Nandun\Desktop\Research\Research\ml\data\processed\leaf_dieback_v1\train\not_cocount')

# Current count
current = len(list(dst.iterdir()))
target = 2850
need = target - current

print(f'Current: {current}, Target: {target}, Need: {need}')

# Get source images
src_imgs = [f.name for f in src.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
random.seed(42)
random.shuffle(src_imgs)

# Copy needed images
copied = 0
idx = current
for img in src_imgs:
    if copied >= need:
        break
    src_file = src / img
    dst_file = dst / f'not_cocount_extra_{idx:04d}{Path(img).suffix}'
    try:
        shutil.copy2(str(src_file), str(dst_file))
        copied += 1
        idx += 1
    except:
        continue

print(f'Copied: {copied}')
print(f'New total: {current + copied}')
print('Done!')

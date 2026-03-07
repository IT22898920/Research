import os
from pathlib import Path

base = Path(r'C:\Users\Tharindu Nandun\Desktop\Research\Research\ml\data\processed\leaf_dieback_v1')

print("Current Dataset Distribution:")
print("="*50)

for split in ['train', 'val', 'test']:
    split_path = base / split
    print(f"\n{split.upper()}:")
    total = 0
    for cls in sorted(os.listdir(split_path)):
        cls_path = split_path / cls
        count = len(list(cls_path.iterdir()))
        total += count
        print(f"  {cls}: {count}")
    print(f"  TOTAL: {total}")

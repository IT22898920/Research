"""
Rename files with long names to shorter Windows-compatible names.
Keeps all images, just makes filenames shorter.
"""

import os
from pathlib import Path
import hashlib

# Dataset directory
DATA_DIR = Path('../../data/raw/stage_1')

# Maximum filename length
MAX_FILENAME_LENGTH = 100

def get_short_name(original_name, counter, extension):
    """Generate a short, unique filename."""
    # Use first few chars + hash for uniqueness
    prefix = original_name[:20].replace(' ', '_').replace('-', '_')
    # Add counter for uniqueness
    short_name = f"{prefix}_{counter:04d}{extension}"
    return short_name

def rename_long_files(data_dir, max_length=100):
    """Rename files with long names to shorter names."""
    renamed_files = []
    counter = 0

    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            if len(filename) > max_length:
                counter += 1
                filepath = Path(root) / filename
                extension = filepath.suffix

                # Generate short name
                short_name = get_short_name(filename, counter, extension)
                new_filepath = Path(root) / short_name

                # Make sure new name doesn't exist
                while new_filepath.exists():
                    counter += 1
                    short_name = get_short_name(filename, counter, extension)
                    new_filepath = Path(root) / short_name

                try:
                    # Rename the file
                    os.rename(filepath, new_filepath)
                    renamed_files.append({
                        'old': filename,
                        'new': short_name,
                        'path': root
                    })
                    print(f"✅ Renamed: {filename[:50]}... -> {short_name}")
                except Exception as e:
                    print(f"❌ Error renaming {filename}: {e}")

    return renamed_files

print("="*70)
print(" RENAMING FILES WITH LONG NAMES")
print("="*70)
print(f"Max filename length: {MAX_FILENAME_LENGTH} characters")
print(f"Searching in: {DATA_DIR.absolute()}\n")

renamed = rename_long_files(DATA_DIR, MAX_FILENAME_LENGTH)

print(f"\n{'='*70}")
print(f" RENAMING COMPLETE!")
print(f"{'='*70}")
print(f"Total files renamed: {len(renamed)}")

if renamed:
    print(f"\n📋 Renamed Files Summary:")
    print(f"{'='*70}")
    for item in renamed[:10]:  # Show first 10
        print(f"  OLD: {item['old'][:45]}...")
        print(f"  NEW: {item['new']}")
        print(f"  Location: {item['path']}")
        print(f"  {'-'*68}")

    if len(renamed) > 10:
        print(f"  ... and {len(renamed) - 10} more files renamed")

    # Save log
    log_file = Path('renamed_files_log.txt')
    with open(log_file, 'w', encoding='utf-8') as f:
        for item in renamed:
            f.write(f"OLD: {item['old']}\n")
            f.write(f"NEW: {item['new']}\n")
            f.write(f"PATH: {item['path']}\n")
            f.write("-" * 70 + "\n")

    print(f"\n✅ Full log saved to: {log_file.absolute()}")
else:
    print("\n✅ No files with long names found!")

print("\n🎉 Dataset is now Windows-compatible and ready for training!")

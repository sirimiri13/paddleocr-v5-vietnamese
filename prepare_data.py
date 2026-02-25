#!/usr/bin/env python3
"""
Data Preparation Script for Kaggle
Xử lý dataset với cấu trúc images/ + rec_gt.txt
"""

import argparse
import os
import shutil
import random
import codecs
from pathlib import Path
from collections import Counter
from tqdm import tqdm


def prepare_kaggle_dataset(input_dir, images_dir='images', label_file='rec_gt.txt', 
                           train_ratio=0.9, max_samples=None, seed=42):
    """
    Chuẩn bị dataset từ Kaggle input
    
    Args:
        input_dir: Thư mục input từ Kaggle (/kaggle/input/your-dataset)
        images_dir: Tên thư mục chứa ảnh
        label_file: Tên file labels
        train_ratio: Tỷ lệ train/val
        max_samples: Giới hạn số samples (None = all)
        seed: Random seed
    """
    
    print("="*70)
    print("🚀 PaddleOCR v5 Vietnamese - Data Preparation for Kaggle")
    print("="*70)
    
    random.seed(seed)
    
    # Paths
    images_path = os.path.join(input_dir, images_dir)
    label_path = os.path.join(input_dir, label_file)
    
    # Check input
    if not os.path.exists(images_path):
        print(f"❌ Images directory not found: {images_path}")
        return False
    
    if not os.path.exists(label_path):
        print(f"❌ Label file not found: {label_path}")
        return False
    
    print(f"\n📂 Input:")
    print(f"   Images: {images_path}")
    print(f"   Labels: {label_path}")
    
    # Read labels
    print(f"\n📖 Reading labels...")
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"   Total lines: {len(lines):,}")
    
    # Parse and validate
    valid_data = []
    missing_count = 0
    
    for line in tqdm(lines, desc="Validating"):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        img_name = parts[0]
        text = parts[1]
        
        # Check image exists
        img_path = os.path.join(images_path, img_name)
        if os.path.exists(img_path):
            valid_data.append((img_name, text, img_path))
        else:
            missing_count += 1
    
    print(f"\n✓ Valid samples: {len(valid_data):,}")
    if missing_count > 0:
        print(f"⚠️  Missing images: {missing_count:,}")
    
    # Limit samples if needed
    if max_samples and max_samples < len(valid_data):
        random.shuffle(valid_data)
        valid_data = valid_data[:max_samples]
        print(f"📊 Limited to: {len(valid_data):,} samples")
    
    # Shuffle and split
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * train_ratio)
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_data):,} ({train_ratio*100:.1f}%)")
    print(f"   Val:   {len(val_data):,} ({(1-train_ratio)*100:.1f}%)")
    
    # Create output directories
    os.makedirs('data/train_data', exist_ok=True)
    os.makedirs('data/val_data', exist_ok=True)
    os.makedirs('dict', exist_ok=True)
    
    # Copy train data
    print(f"\n📁 Copying training images...")
    train_labels = []
    for img_name, text, img_path in tqdm(train_data, desc="Train"):
        dst = os.path.join('data/train_data', img_name)
        shutil.copy2(img_path, dst)
        train_labels.append(f"train_data/{img_name}\t{text}\n")
    
    # Copy val data
    print(f"📁 Copying validation images...")
    val_labels = []
    for img_name, text, img_path in tqdm(val_data, desc="Val"):
        dst = os.path.join('data/val_data', img_name)
        shutil.copy2(img_path, dst)
        val_labels.append(f"val_data/{img_name}\t{text}\n")
    
    # Save label files
    with open('data/train_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    
    with open('data/val_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
    
    print(f"\n✓ Label files created:")
    print(f"   data/train_list.txt")
    print(f"   data/val_list.txt")
    
    # Create Vietnamese dictionary
    print(f"\n📝 Creating Vietnamese dictionary...")
    chars = set()
    char_freq = Counter()
    
    for _, text, _ in valid_data:
        chars.update(text)
        char_freq.update(text)
    
    # Add standard Vietnamese characters
    vietnamese = 'aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz'
    chars.update(vietnamese)
    chars.update(vietnamese.upper())
    chars.update('0123456789')
    chars.update(' .,;:!?-_()[]{}/@#$%^&*+=~`\'"<>|\\°§€£¥₫')
    
    # Sort by frequency
    sorted_chars = [char for char, _ in char_freq.most_common()]
    for char in sorted(chars):
        if char not in sorted_chars:
            sorted_chars.append(char)
    
    # Save dictionary
    dict_path = 'dict/vi_dict.txt'
    with codecs.open(dict_path, 'w', encoding='utf-8') as f:
        for char in sorted_chars:
            f.write(char + '\n')
    
    print(f"✓ Dictionary created: {dict_path}")
    print(f"   Total characters: {len(sorted_chars)}")
    
    # Statistics
    print(f"\n📊 Statistics:")
    print(f"   Total valid samples: {len(valid_data):,}")
    print(f"   Train samples: {len(train_data):,}")
    print(f"   Val samples: {len(val_data):,}")
    print(f"   Unique characters: {len(sorted_chars)}")
    
    if char_freq:
        print(f"\n   Top 10 characters:")
        for char, count in char_freq.most_common(10):
            display = char if char != ' ' else '<space>'
            print(f"      '{display}': {count:,}")
    
    # Sample check
    print(f"\n📋 Sample data (first 3):")
    for i, (img_name, text, _) in enumerate(train_data[:3], 1):
        print(f"   {i}. {img_name}: {text[:50]}...")
    
    print(f"\n" + "="*70)
    print(f"✅ Data preparation completed successfully!")
    print(f"="*70)
    
    print(f"\n📝 Next steps:")
    print(f"   1. Check data:")
    print(f"      ls -lh data/train_data/ | head")
    print(f"      head data/train_list.txt")
    print(f"\n   2. Start training:")
    print(f"      bash train_kaggle.sh")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Prepare dataset for PaddleOCR v5 training on Kaggle'
    )
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory from Kaggle (e.g., /kaggle/input/vietnamese-ocr-250k)'
    )
    parser.add_argument(
        '--images_dir',
        type=str,
        default='images',
        help='Images directory name (default: images)'
    )
    parser.add_argument(
        '--label_file',
        type=str,
        default='rec_gt.txt',
        help='Label file name (default: rec_gt.txt)'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.9,
        help='Train/val split ratio (default: 0.9)'
    )
    parser.add_argument(
        '--max_samples',
        type=int,
        default=None,
        help='Maximum samples to use (default: all)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )
    
    args = parser.parse_args()
    
    success = prepare_kaggle_dataset(
        input_dir=args.input_dir,
        images_dir=args.images_dir,
        label_file=args.label_file,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples,
        seed=args.seed
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

#!/usr/bin/env python3
"""
Smart Data Preparation - Auto-detect image paths
Tự động tìm ảnh và fix path issues
"""

import argparse, os, shutil, random, codecs
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def find_images_directory(input_dir):
    """Tự động tìm thư mục chứa ảnh"""
    print(f"\n🔍 Searching for images...")
    
    for dirname in ['images', 'imgs', 'img', 'Pictures', 'photos', 'data', '.']:
        img_dir = os.path.join(input_dir, dirname) if dirname != '.' else input_dir
        if os.path.exists(img_dir):
            img_count = len(list(Path(img_dir).glob('*.jpg'))) + len(list(Path(img_dir).glob('*.png')))
            if img_count > 0:
                print(f"✓ Found {img_count} images in: {dirname if dirname != '.' else 'root'}/")
                return dirname if dirname != '.' else None
    return None

def prepare_kaggle_dataset(input_dir, images_dir=None, label_file='rec_gt.txt', 
                           train_ratio=0.9, max_samples=None, seed=42):
    print("="*70)
    print("🚀 PaddleOCR v5 - Smart Data Preparation")
    print("="*70)
    
    random.seed(seed)
    
    # Auto-detect images directory
    if images_dir is None:
        images_dir = find_images_directory(input_dir)
    
    images_path = os.path.join(input_dir, images_dir) if images_dir else input_dir
    label_path = os.path.join(input_dir, label_file)
    
    print(f"\n📂 Paths:")
    print(f"   Images: {images_path}")
    print(f"   Labels: {label_path}")
    
    if not os.path.exists(label_path):
        print(f"❌ Label file not found!")
        # Try to find it
        for name in ['rec_gt.txt', 'labels.txt', 'gt.txt', 'train.txt']:
            alt_path = os.path.join(input_dir, name)
            if os.path.exists(alt_path):
                label_path = alt_path
                print(f"✓ Found: {name}")
                break
        else:
            return False
    
    # Read labels
    with open(label_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"\n📖 Total lines: {len(lines):,}")
    print(f"\n📋 Sample (first 3):")
    for i, line in enumerate(lines[:3]):
        print(f"   {i+1}. {line.strip()[:80]}...")
    
    # Parse with smart path matching
    valid_data = []
    missing = 0
    
    for line in tqdm(lines, desc="Processing"):
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue
        
        img_name = parts[0].strip()
        text = '\t'.join(parts[1:]).strip()
        
        # Try multiple path combinations
        possible_paths = [
            os.path.join(images_path, img_name),
            os.path.join(images_path, os.path.basename(img_name)),
            os.path.join(input_dir, img_name),
            os.path.join(input_dir, os.path.basename(img_name)),
        ]
        
        found_path = None
        for p in possible_paths:
            if os.path.exists(p):
                found_path = p
                break
        
        if found_path:
            valid_data.append((os.path.basename(found_path), text, found_path))
        else:
            missing += 1
    
    print(f"\n📊 Results:")
    print(f"   ✓ Valid: {len(valid_data):,}")
    print(f"   ✗ Missing: {missing:,}")
    
    if len(valid_data) == 0:
        print(f"\n❌ No images found!")
        print(f"\n🔧 Debug:")
        sample_imgs = list(Path(images_path).glob('*'))[:5]
        print(f"   Files in {images_path}:")
        for f in sample_imgs:
            print(f"      {f.name}")
        print(f"\n   First line format: {lines[0].strip()[:100]}")
        return False
    
    # Limit & split
    if max_samples and max_samples < len(valid_data):
        random.shuffle(valid_data)
        valid_data = valid_data[:max_samples]
    
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * train_ratio)
    train_data, val_data = valid_data[:split_idx], valid_data[split_idx:]
    
    print(f"\n📊 Split: Train {len(train_data):,} | Val {len(val_data):,}")
    
    # Create dirs
    os.makedirs('data/train_data', exist_ok=True)
    os.makedirs('data/val_data', exist_ok=True)
    os.makedirs('dict', exist_ok=True)
    
    # Copy images & create labels
    train_labels = []
    for img_name, text, img_path in tqdm(train_data, desc="Train"):
        shutil.copy2(img_path, f'data/train_data/{img_name}')
        train_labels.append(f"train_data/{img_name}\t{text}\n")
    
    val_labels = []
    for img_name, text, img_path in tqdm(val_data, desc="Val"):
        shutil.copy2(img_path, f'data/val_data/{img_name}')
        val_labels.append(f"val_data/{img_name}\t{text}\n")
    
    with open('data/train_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    with open('data/val_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
    
    # Create dictionary
    chars = set()
    char_freq = Counter()
    for _, text, _ in valid_data:
        chars.update(text)
        char_freq.update(text)
    
    vietnamese = 'aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz'
    chars.update(vietnamese + vietnamese.upper() + '0123456789' + ' .,;:!?-_()[]{}/@#$%^&*+=~`\'"<>|\\°§€£¥₫')
    
    sorted_chars = [c for c, _ in char_freq.most_common()] + [c for c in sorted(chars) if c not in char_freq]
    
    with codecs.open('dict/vi_dict.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_chars))
    
    print(f"\n✅ Done! {len(sorted_chars)} chars in dict")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--images_dir', default=None)
    parser.add_argument('--label_file', default='rec_gt.txt')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--max_samples', type=int, default=None)
    args = parser.parse_args()
    prepare_kaggle_dataset(**vars(args))

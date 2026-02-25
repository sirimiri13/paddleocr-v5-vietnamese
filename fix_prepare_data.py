#!/usr/bin/env python3
"""
Quick Fix for rec_gt.txt with 'images/' prefix
Xử lý trường hợp path trong label có prefix 'images/'
"""

import os
import shutil
import random
import codecs
from pathlib import Path
from collections import Counter
from tqdm import tqdm

def fix_and_prepare(input_dir, train_ratio=0.9, max_samples=None):
    """
    Fix path và prepare data
    """
    print("="*70)
    print("🔧 Quick Fix & Prepare Data")
    print("="*70)
    
    # Paths
    images_dir = os.path.join(input_dir, 'images')
    label_file = os.path.join(input_dir, 'rec_gt.txt')
    
    print(f"\n📂 Input:")
    print(f"   Directory: {input_dir}")
    print(f"   Images: {images_dir}")
    print(f"   Labels: {label_file}")
    
    # Check
    if not os.path.exists(label_file):
        print(f"❌ Label file not found!")
        return False
    
    if not os.path.exists(images_dir):
        print(f"❌ Images directory not found!")
        return False
    
    # Read labels
    print(f"\n📖 Reading labels...")
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"   Total lines: {len(lines):,}")
    
    # Show sample
    print(f"\n📋 Sample (first 3):")
    for i in range(min(3, len(lines))):
        print(f"   {i+1}. {lines[i].strip()[:80]}")
    
    # Process
    print(f"\n⚙️  Processing...")
    valid_data = []
    missing = 0
    
    for line in tqdm(lines, desc="Parsing"):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split('\t')
        if len(parts) < 2:
            continue
        
        img_path_in_label = parts[0].strip()
        text = '\t'.join(parts[1:]).strip()
        
        # FIX: Remove 'images/' prefix if exists
        if img_path_in_label.startswith('images/'):
            img_filename = img_path_in_label[7:]  # Remove 'images/'
        elif img_path_in_label.startswith('images\\'):
            img_filename = img_path_in_label[7:]  # Remove 'images\'
        else:
            img_filename = os.path.basename(img_path_in_label)
        
        # Full path to actual image
        actual_img_path = os.path.join(images_dir, img_filename)
        
        if os.path.exists(actual_img_path):
            valid_data.append((img_filename, text, actual_img_path))
        else:
            missing += 1
            if missing <= 3:  # Show first 3 missing
                print(f"   Missing: {img_path_in_label} -> {actual_img_path}")
    
    print(f"\n📊 Results:")
    print(f"   ✓ Valid: {len(valid_data):,}")
    print(f"   ✗ Missing: {missing:,}")
    
    if len(valid_data) == 0:
        print(f"\n❌ No valid data!")
        return False
    
    # Limit if needed
    if max_samples and max_samples < len(valid_data):
        random.shuffle(valid_data)
        valid_data = valid_data[:max_samples]
        print(f"   Limited to: {len(valid_data):,}")
    
    # Split
    random.seed(42)
    random.shuffle(valid_data)
    split_idx = int(len(valid_data) * train_ratio)
    train_data = valid_data[:split_idx]
    val_data = valid_data[split_idx:]
    
    print(f"\n📊 Split:")
    print(f"   Train: {len(train_data):,}")
    print(f"   Val:   {len(val_data):,}")
    
    # Create directories
    os.makedirs('data/train_data', exist_ok=True)
    os.makedirs('data/val_data', exist_ok=True)
    os.makedirs('dict', exist_ok=True)
    
    # Copy train data
    print(f"\n📁 Copying images...")
    train_labels = []
    for img_name, text, img_path in tqdm(train_data, desc="Train"):
        dst = os.path.join('data/train_data', img_name)
        shutil.copy2(img_path, dst)
        train_labels.append(f"train_data/{img_name}\t{text}\n")
    
    # Copy val data
    val_labels = []
    for img_name, text, img_path in tqdm(val_data, desc="Val"):
        dst = os.path.join('data/val_data', img_name)
        shutil.copy2(img_path, dst)
        val_labels.append(f"val_data/{img_name}\t{text}\n")
    
    # Save labels
    with open('data/train_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(train_labels)
    with open('data/val_list.txt', 'w', encoding='utf-8') as f:
        f.writelines(val_labels)
    
    print(f"✓ Labels saved")
    
    # Create dictionary
    print(f"\n📝 Creating dictionary...")
    chars = set()
    char_freq = Counter()
    
    for _, text, _ in valid_data:
        chars.update(text)
        char_freq.update(text)
    
    # Add Vietnamese chars
    vietnamese = 'aàáảãạăằắẳẵặâầấẩẫậbcdđeèéẻẽẹêềếểễệfghiìíỉĩịjklmnoòóỏõọôồốổỗộơờớởỡợpqrstuùúủũụưừứửữựvwxyỳýỷỹỵz'
    chars.update(vietnamese + vietnamese.upper() + '0123456789 .,;:!?-_()[]{}/@#$%^&*+=~`\'"<>|\\°§€£¥₫')
    
    sorted_chars = [c for c, _ in char_freq.most_common()]
    for c in sorted(chars):
        if c not in sorted_chars:
            sorted_chars.append(c)
    
    with codecs.open('dict/vi_dict.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted_chars))
    
    print(f"✓ Dictionary: {len(sorted_chars)} characters")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"✅ SUCCESS!")
    print(f"{'='*70}")
    print(f"\n📊 Summary:")
    print(f"   Valid samples: {len(valid_data):,}")
    print(f"   Train: {len(train_data):,}")
    print(f"   Val: {len(val_data):,}")
    print(f"   Characters: {len(sorted_chars)}")
    
    print(f"\n📝 Next step:")
    print(f"   bash train_test_10k.sh  # Test với 10k samples (~30 phút)")
    print(f"   bash train_full.sh      # Full training (~16-20 giờ)")
    
    return True


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, 
                       help='Path to FinalData directory')
    parser.add_argument('--train_ratio', type=float, default=0.9)
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Limit samples (for testing)')
    
    args = parser.parse_args()
    
    success = fix_and_prepare(
        input_dir=args.input_dir,
        train_ratio=args.train_ratio,
        max_samples=args.max_samples
    )
    
    exit(0 if success else 1)

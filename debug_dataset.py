#!/usr/bin/env python3
"""
Debug Script - Check Dataset Structure
Kiểm tra cấu trúc dataset để tìm lỗi
"""

import os
import sys
from pathlib import Path

def debug_dataset(input_dir):
    """Check dataset structure and show detailed info"""
    
    print("="*70)
    print("🔍 DATASET DEBUGGER")
    print("="*70)
    
    print(f"\n📂 Input directory: {input_dir}")
    print(f"   Exists: {os.path.exists(input_dir)}")
    
    if not os.path.exists(input_dir):
        print(f"\n❌ Directory not found!")
        return
    
    # List all files and directories
    print(f"\n📁 Contents of {input_dir}:")
    items = list(Path(input_dir).glob('*'))
    for item in sorted(items)[:20]:
        item_type = "DIR " if item.is_dir() else "FILE"
        print(f"   [{item_type}] {item.name}")
    
    if len(items) > 20:
        print(f"   ... and {len(items)-20} more items")
    
    # Find label files
    print(f"\n📄 Looking for label files:")
    label_files = []
    for name in ['rec_gt.txt', 'labels.txt', 'gt.txt', 'train.txt', 'annotations.txt']:
        path = os.path.join(input_dir, name)
        if os.path.exists(path):
            size = os.path.getsize(path)
            label_files.append(name)
            print(f"   ✓ {name} ({size:,} bytes)")
    
    if not label_files:
        print(f"   ❌ No label file found!")
    
    # Find image directories
    print(f"\n🖼️  Looking for image directories:")
    img_dirs = []
    for name in ['images', 'imgs', 'img', 'Pictures', 'photos', 'data']:
        dir_path = os.path.join(input_dir, name)
        if os.path.isdir(dir_path):
            jpg_count = len(list(Path(dir_path).glob('*.jpg')))
            png_count = len(list(Path(dir_path).glob('*.png')))
            total = jpg_count + png_count
            if total > 0:
                img_dirs.append(name)
                print(f"   ✓ {name}/ ({total:,} images: {jpg_count} jpg, {png_count} png)")
    
    # Check root for images
    root_jpg = len(list(Path(input_dir).glob('*.jpg')))
    root_png = len(list(Path(input_dir).glob('*.png')))
    if root_jpg + root_png > 0:
        img_dirs.append('ROOT')
        print(f"   ✓ Root directory ({root_jpg + root_png:,} images)")
    
    if not img_dirs:
        print(f"   ❌ No images found!")
    
    # Analyze label file if found
    if label_files:
        label_file = os.path.join(input_dir, label_files[0])
        print(f"\n📝 Analyzing {label_files[0]}:")
        
        with open(label_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        print(f"   Total lines: {len(lines):,}")
        
        print(f"\n   First 5 lines:")
        for i, line in enumerate(lines[:5]):
            print(f"   {i+1}. {line.strip()[:100]}")
        
        # Analyze format
        if lines:
            first_line = lines[0].strip()
            has_tab = '\t' in first_line
            parts = first_line.split('\t') if has_tab else first_line.split()
            
            print(f"\n   Format analysis:")
            print(f"      Separator: {'TAB' if has_tab else 'SPACE'}")
            print(f"      Parts: {len(parts)}")
            if parts:
                print(f"      Image path: {parts[0]}")
                print(f"      Has slash: {'/' in parts[0] or '\\\\' in parts[0]}")
        
        # Check path patterns
        print(f"\n   Path patterns (first 10):")
        path_patterns = {}
        for line in lines[:100]:
            parts = line.strip().split('\t')
            if parts:
                img_path = parts[0]
                # Extract directory structure
                if '/' in img_path:
                    prefix = img_path.split('/')[0]
                elif '\\\\' in img_path:
                    prefix = img_path.split('\\\\')[0]
                else:
                    prefix = 'FILENAME_ONLY'
                
                path_patterns[prefix] = path_patterns.get(prefix, 0) + 1
        
        for pattern, count in sorted(path_patterns.items(), key=lambda x: -x[1])[:10]:
            print(f"      {pattern}: {count} files")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"📊 SUMMARY:")
    print(f"{'='*70}")
    print(f"   Label files found: {len(label_files)}")
    print(f"   Image directories: {len(img_dirs)}")
    
    if label_files and img_dirs:
        print(f"\n✅ Dataset structure looks OK")
        print(f"\n💡 Recommended command:")
        print(f"   python prepare_data.py \\")
        print(f"       --input_dir {input_dir} \\")
        if img_dirs[0] != 'ROOT':
            print(f"       --images_dir {img_dirs[0]} \\")
        print(f"       --label_file {label_files[0]}")
    else:
        print(f"\n❌ Dataset structure has issues")
        if not label_files:
            print(f"   - No label file found (expected: rec_gt.txt)")
        if not img_dirs:
            print(f"   - No images found")
    
    print()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        debug_dataset(sys.argv[1])
    else:
        print("Usage: python debug_dataset.py /path/to/dataset")
        print("\nFor Kaggle:")
        print("  python debug_dataset.py /kaggle/input/vietnamese-ocr-250k")

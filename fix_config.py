#!/usr/bin/env python3
"""
Auto-fix config paths
Tự động sửa config với absolute paths
"""

import os
import sys

def fix_config(config_file='config_kaggle.yml'):
    """Fix paths in config to use absolute paths"""
    
    # Get project root (current directory)
    project_root = os.getcwd()
    
    print(f"🔧 Fixing config: {config_file}")
    print(f"   Project root: {project_root}")
    
    if not os.path.exists(config_file):
        print(f"❌ Config not found: {config_file}")
        return False
    
    # Read config
    with open(config_file, 'r') as f:
        content = f.read()
    
    # Replace relative paths with absolute
    replacements = {
        './data/': f'{project_root}/data/',
        './dict/': f'{project_root}/dict/',
        './output/': f'{project_root}/output/',
        './pretrain_models/': f'{project_root}/pretrain_models/',
        './inference/': f'{project_root}/inference/',
    }
    
    original_content = content
    for old, new in replacements.items():
        content = content.replace(old, new)
    
    # Write back
    if content != original_content:
        with open(config_file, 'w') as f:
            f.write(content)
        print(f"✓ Config fixed with absolute paths")
        return True
    else:
        print(f"✓ Config already has correct paths")
        return True

if __name__ == '__main__':
    config = sys.argv[1] if len(sys.argv) > 1 else 'config_kaggle.yml'
    fix_config(config)

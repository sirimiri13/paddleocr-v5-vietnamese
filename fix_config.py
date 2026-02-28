#!/usr/bin/env python3
"""
Fix PaddleOCR v5 config for multi-head (CTC+NRTR) recognition.

Ensures:
- MultiLabelEncode is used instead of CTCLabelEncode
- KeepKeys includes label_ctc, label_gtc, length, valid_ratio
- Both Train and Eval datasets are fixed
"""

import sys
import yaml
import copy


def fix_transforms(transforms):
    """Fix transform list to use MultiLabelEncode and correct KeepKeys."""
    if not transforms:
        return transforms

    new_transforms = []
    has_multi_label = False
    has_ctc_label = False

    for t in transforms:
        if isinstance(t, dict):
            # Replace CTCLabelEncode with MultiLabelEncode
            if 'CTCLabelEncode' in t:
                print(f"  → Replacing CTCLabelEncode with MultiLabelEncode")
                params = t['CTCLabelEncode'] or {}
                if params is None:
                    params = {}
                params['gtc_encode'] = 'NRTRLabelEncode'
                new_transforms.append({'MultiLabelEncode': params})
                has_multi_label = True
                continue

            # If MultiLabelEncode already exists, ensure gtc_encode is set
            if 'MultiLabelEncode' in t:
                has_multi_label = True
                params = t['MultiLabelEncode'] or {}
                if params is None:
                    params = {}
                if 'gtc_encode' not in params or params['gtc_encode'] is None:
                    print(f"  → Setting gtc_encode to NRTRLabelEncode")
                    params['gtc_encode'] = 'NRTRLabelEncode'
                    t['MultiLabelEncode'] = params
                new_transforms.append(t)
                continue

            # Fix KeepKeys to include all required keys
            if 'KeepKeys' in t:
                keep_params = t['KeepKeys'] or {}
                if keep_params is None:
                    keep_params = {}
                keys = keep_params.get('keep_keys', [])

                required_keys = ['image', 'label_ctc', 'label_gtc', 'length', 'valid_ratio']
                # Check if it's using old-style keys
                if 'label' in keys and 'label_ctc' not in keys:
                    print(f"  → Fixing KeepKeys: {keys} -> {required_keys}")
                    keep_params['keep_keys'] = required_keys
                elif 'label_gtc' not in keys:
                    # Fix: label_sar -> label_gtc for NRTRHead
                    if 'label_sar' in keys:
                        print(f"  → Fixing KeepKeys (SAR->NRTR): {keys} -> {required_keys}")
                    else:
                        print(f"  → Fixing KeepKeys: {keys} -> {required_keys}")
                    keep_params['keep_keys'] = required_keys
                else:
                    print(f"  → KeepKeys already correct: {keys}")

                new_transforms.append({'KeepKeys': keep_params})
                continue

        new_transforms.append(t)

    return new_transforms


def fix_config(config_path):
    """Fix config file for multi-head recognition."""
    print(f"\n{'='*60}")
    print(f"Fixing config: {config_path}")
    print(f"{'='*60}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Fix Train transforms
    try:
        train_transforms = config['Train']['dataset']['transforms']
        print("\n[Train transforms]")
        config['Train']['dataset']['transforms'] = fix_transforms(train_transforms)
    except (KeyError, TypeError) as e:
        print(f"  Warning: Could not fix Train transforms: {e}")

    # Fix Eval transforms
    try:
        eval_transforms = config['Eval']['dataset']['transforms']
        print("\n[Eval transforms]")
        config['Eval']['dataset']['transforms'] = fix_transforms(eval_transforms)
    except (KeyError, TypeError) as e:
        print(f"  Warning: Could not fix Eval transforms: {e}")

    # Write fixed config
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print(f"\n✅ Config fixed and saved: {config_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python fix_config.py <config.yml>")
        sys.exit(1)

    config_path = sys.argv[1]
    fix_config(config_path)

#!/usr/bin/env python3
"""
Test Inference Script
Test trained model với ảnh mẫu
"""

import argparse
import os
import sys

try:
    from paddleocr import PaddleOCR
    import cv2
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Error importing: {e}")
    print("Installing required packages...")
    os.system("pip install -q paddleocr opencv-python matplotlib")
    from paddleocr import PaddleOCR
    import cv2
    import matplotlib.pyplot as plt


def test_model(image_path, rec_model_dir='inference', dict_path='dict/vi_dict.txt'):
    """Test OCR model"""
    
    print("="*70)
    print("🧪 Testing PaddleOCR v5 Vietnamese Model")
    print("="*70)
    
    # Check files
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False
    
    if not os.path.exists(rec_model_dir):
        print(f"❌ Model directory not found: {rec_model_dir}")
        print("Please export model first: bash export_model.sh")
        return False
    
    if not os.path.exists(dict_path):
        print(f"❌ Dictionary not found: {dict_path}")
        return False
    
    print(f"\n📂 Configuration:")
    print(f"   Image: {image_path}")
    print(f"   Model: {rec_model_dir}")
    print(f"   Dictionary: {dict_path}")
    
    # Load image info
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot read image")
        return False
    
    h, w = img.shape[:2]
    print(f"\n📐 Image Info:")
    print(f"   Size: {w}x{h}")
    print(f"   Format: {image_path.split('.')[-1].upper()}")
    
    # Initialize OCR
    print(f"\n🔧 Initializing PaddleOCR...")
    
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='vi',
        rec_model_dir=rec_model_dir,
        rec_char_dict_path=dict_path,
        use_gpu=True,
        show_log=False
    )
    
    print("✓ Model loaded")
    
    # Run OCR
    print(f"\n🚀 Running OCR...")
    result = ocr.ocr(image_path, cls=True)
    
    if not result or not result[0]:
        print("⚠️  No text detected")
        return True
    
    # Display results
    print(f"\n📊 Results ({len(result[0])} text regions):")
    print("-"*70)
    
    for idx, line in enumerate(result[0], 1):
        bbox, (text, confidence) = line
        
        print(f"\n{idx}. Text: {text}")
        print(f"   Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
        print(f"   BBox: {bbox[0]} -> {bbox[2]}")
    
    # Calculate average confidence
    avg_conf = sum([line[1][1] for line in result[0]]) / len(result[0])
    print(f"\n📈 Average Confidence: {avg_conf:.4f} ({avg_conf*100:.2f}%)")
    
    # Visualize (optional)
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.title(f'OCR Test Result - {len(result[0])} regions detected')
        plt.axis('off')
        
        output_path = 'test_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n💾 Visualization saved: {output_path}")
        
    except Exception as e:
        print(f"\n⚠️  Could not create visualization: {e}")
    
    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Test PaddleOCR v5 Vietnamese model')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--rec_model_dir', type=str, default='inference', 
                       help='Recognition model directory')
    parser.add_argument('--dict_path', type=str, default='dict/vi_dict.txt',
                       help='Character dictionary path')
    
    args = parser.parse_args()
    
    success = test_model(args.image, args.rec_model_dir, args.dict_path)
    
    return 0 if success else 1


if __name__ == '__main__':
    exit(main())

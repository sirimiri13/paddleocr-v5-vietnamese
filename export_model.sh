#!/bin/bash

# Export Model Script for Kaggle
# Convert trained model to inference format

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          PaddleOCR v5 Vietnamese - Export Model             ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Configuration
CONFIG="config_kaggle.yml"
MODEL_PATH="output/vi_ppocr_v5/best_accuracy"
OUTPUT_DIR="inference"

echo -e "\n${BLUE}Configuration:${NC}"
echo "   Config: $CONFIG"
echo "   Model: $MODEL_PATH"
echo "   Output: $OUTPUT_DIR"

# Check model exists
if [ ! -f "$MODEL_PATH.pdparams" ]; then
    echo "❌ Trained model not found at $MODEL_PATH"
    echo "Please complete training first"
    exit 1
fi

echo -e "\n${GREEN}Exporting model...${NC}"

cd PaddleOCR

python tools/export_model.py \
    -c ../$CONFIG \
    -o Global.pretrained_model=../$MODEL_PATH \
       Global.save_inference_dir=../$OUTPUT_DIR

cd ..

# Check export success
if [ -f "$OUTPUT_DIR/inference.pdmodel" ] && [ -f "$OUTPUT_DIR/inference.pdiparams" ]; then
    echo -e "\n╔══════════════════════════════════════════════════════════════╗"
    echo -e "║              ${GREEN}✅ Export Completed!${NC}                          ║"
    echo -e "╚══════════════════════════════════════════════════════════════╝"
    
    echo -e "\n${GREEN}📦 Exported files:${NC}"
    ls -lh $OUTPUT_DIR/
    
    echo -e "\n${GREEN}📝 Create downloadable package:${NC}"
    echo "   tar -czf vi_ppocr_v5_model.tar.gz inference/ dict/ config_kaggle.yml"
    echo ""
    echo "   Then download from Kaggle Output tab!"
    
    echo -e "\n${GREEN}🧪 Test model:${NC}"
    echo "   python test_model.py --image data/val_data/sample.jpg"
else
    echo "❌ Export failed!"
    exit 1
fi

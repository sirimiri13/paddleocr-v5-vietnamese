#!/bin/bash

# Training Script for Kaggle
# Optimized for T4 x2 GPU

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        PaddleOCR v5 Vietnamese - Training                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check if resume mode
RESUME=false
if [ "$1" == "--resume" ]; then
    RESUME=true
    echo -e "${YELLOW}📌 Resume mode enabled${NC}"
fi

# Configuration
CONFIG="config_kaggle.yml"
GPUS="0,1"

echo -e "\n${BLUE}Configuration:${NC}"
echo "   Config: $CONFIG"
echo "   GPUs: $GPUS"
echo "   Resume: $RESUME"

# Check prerequisites
echo -e "\n${BLUE}Checking prerequisites...${NC}"

if [ ! -f "$CONFIG" ]; then
    echo -e "${YELLOW}⚠️  Config not found, will be created by PaddleOCR${NC}"
fi

if [ ! -d "data" ]; then
    echo -e "❌ Data directory not found!"
    echo "Please run: python prepare_data.py --input_dir /kaggle/input/your-dataset"
    exit 1
fi

if [ ! -f "data/train_list.txt" ]; then
    echo -e "❌ train_list.txt not found!"
    echo "Please run prepare_data.py first"
    exit 1
fi

if [ ! -f "dict/vi_dict.txt" ]; then
    echo -e "❌ Dictionary not found!"
    echo "Please run prepare_data.py first"
    exit 1
fi

if [ ! -d "pretrain_models/ch_PP-OCRv5_rec_train" ]; then
    echo -e "❌ Pretrained model not found!"
    echo "Please run setup_kaggle.sh first"
    exit 1
fi

echo "✓ All prerequisites met"

# Log file
LOG_FILE="logs/train_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Build training command
cd PaddleOCR

if [ "$RESUME" = true ]; then
    # Resume from checkpoint
    CMD="python -m paddle.distributed.launch \
        --gpus '$GPUS' \
        tools/train.py \
        -c ../$CONFIG \
        -o Global.checkpoints=../output/vi_ppocr_v5/latest"
else
    # Fresh training
    CMD="python -m paddle.distributed.launch \
        --gpus '$GPUS' \
        tools/train.py \
        -c ../$CONFIG"
fi

# Start training
echo -e "\n${GREEN}🚀 Starting training...${NC}"
echo "   Log: $LOG_FILE"
echo "   Time: $(date)"
echo ""
echo -e "${YELLOW}Training will take 15-20 hours for 250k samples${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop (checkpoints will be saved)${NC}"
echo ""

# Run training with logging
eval $CMD 2>&1 | tee ../$LOG_FILE

cd ..

# Training completed
echo -e "\n╔══════════════════════════════════════════════════════════════╗"
echo -e "║              ${GREEN}✅ Training Completed!${NC}                         ║"
echo -e "╚══════════════════════════════════════════════════════════════╝"

echo -e "\n${GREEN}📊 Check results:${NC}"
echo "   Best model: output/vi_ppocr_v5/best_accuracy.pdparams"
echo "   Latest checkpoint: output/vi_ppocr_v5/latest.pdparams"
echo "   Training log: $LOG_FILE"

echo -e "\n${GREEN}📝 Next steps:${NC}"
echo "   1. Export model:"
echo "      bash export_model.sh"
echo ""
echo "   2. Test inference:"
echo "      python test_model.py --image data/val_data/sample.jpg"
echo ""
echo "   3. Download model:"
echo "      tar -czf model.tar.gz inference/ dict/"
echo "      # Download from Kaggle Output tab"

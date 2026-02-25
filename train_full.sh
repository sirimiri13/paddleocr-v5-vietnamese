#!/bin/bash

# Full Training Script
# Train với toàn bộ dataset sau khi test 10k xong

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       PaddleOCR v5 - Full Training (All Data)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Check if test was done
if [ -f "output/test_10k/best_accuracy.pdparams" ]; then
    echo -e "\n${GREEN}✓ Test model found${NC}"
    
    # Show test results
    if [ -f "logs/test_10k_"*.log ]; then
        TEST_LOG=$(ls -t logs/test_10k_*.log | head -1)
        echo -e "\n${BLUE}Test Results:${NC}"
        tail -30 $TEST_LOG | grep -i "acc:" | tail -3 || echo "  (check log for details)"
    fi
    
    echo ""
    read -p "Continue with full training? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
else
    echo -e "\n${YELLOW}⚠️  No test model found${NC}"
    echo "Recommend running test first: bash train_test_10k.sh"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Cancelled."
        exit 0
    fi
fi

# Restore full dataset
echo -e "\n${GREEN}[1/3] Restoring full dataset...${NC}"

if [ -f "data/train_full.txt" ]; then
    cp data/train_full.txt data/train_list.txt
    cp data/val_full.txt data/val_list.txt
    
    echo "✓ Full dataset restored:"
    echo "   Train: $(wc -l < data/train_list.txt) samples"
    echo "   Val: $(wc -l < data/val_list.txt) samples"
else
    echo "✓ Using current dataset:"
    echo "   Train: $(wc -l < data/train_list.txt) samples"
    echo "   Val: $(wc -l < data/val_list.txt) samples"
fi

# Check prerequisites
echo -e "\n${GREEN}[2/3] Checking prerequisites...${NC}"

if [ ! -f "config_kaggle.yml" ]; then
    echo "❌ config_kaggle.yml not found"
    exit 1
fi

if [ ! -d "pretrain_models/ch_PP-OCRv5_rec_train" ]; then
    echo "❌ Pretrained model not found"
    echo "Run: bash setup_kaggle.sh"
    exit 1
fi

if [ ! -f "dict/vi_dict.txt" ]; then
    echo "❌ Dictionary not found"
    echo "Run: python prepare_data.py ..."
    exit 1
fi

echo "✓ All prerequisites met"

# Estimate time
TRAIN_SAMPLES=$(wc -l < data/train_list.txt)
if [ $TRAIN_SAMPLES -lt 50000 ]; then
    TIME_EST="3-4 hours"
elif [ $TRAIN_SAMPLES -lt 100000 ]; then
    TIME_EST="7-9 hours"
elif [ $TRAIN_SAMPLES -lt 200000 ]; then
    TIME_EST="12-15 hours"
else
    TIME_EST="16-20 hours"
fi

echo -e "\n${BLUE}Training Info:${NC}"
echo "   Samples: $TRAIN_SAMPLES"
echo "   Estimated time: $TIME_EST"
echo "   Epochs: 100"
echo "   Batch size: 96"
echo "   GPUs: 2"

# Confirm
echo -e "\n${YELLOW}⚠️  This will take $TIME_EST${NC}"
echo -e "${YELLOW}Checkpoints saved every 5 epochs${NC}"
echo ""
read -p "Start full training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Start training
echo -e "\n${GREEN}[3/3] Starting full training...${NC}"
echo -e "${YELLOW}Started at: $(date)${NC}"
echo ""

LOG_FILE="logs/train_full_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

cd PaddleOCR

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c ../config_kaggle.yml \
    2>&1 | tee ../$LOG_FILE

cd ..

# Training completed
echo -e "\n╔══════════════════════════════════════════════════════════════╗"
echo -e "║           ${GREEN}✅ Full Training Completed!${NC}                     ║"
echo -e "╚══════════════════════════════════════════════════════════════╝"

echo -e "\n${GREEN}📊 Results:${NC}"
echo "   Best model: output/vi_ppocr_v5/best_accuracy.pdparams"
echo "   Latest checkpoint: output/vi_ppocr_v5/latest.pdparams"
echo "   Log: $LOG_FILE"

echo -e "\n${BLUE}📈 Final accuracy:${NC}"
tail -50 $LOG_FILE | grep -i "acc:" | tail -5

echo -e "\n${GREEN}📝 Next steps:${NC}"
echo "   1. Export model:"
echo "      bash export_model.sh"
echo ""
echo "   2. Test model:"
echo "      python test_model.py --image data/val_data/sample.jpg"
echo ""
echo "   3. Package for download:"
echo "      tar -czf vi_ppocr_v5_model.tar.gz inference/ dict/ config_kaggle.yml"
echo ""
echo -e "${GREEN}🎉 Training completed at: $(date)${NC}"

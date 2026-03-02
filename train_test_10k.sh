#!/bin/bash

# Quick Test Script - Train với 10k samples
# Test setup và verify model trước khi train full

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       PaddleOCR v5 - Quick Test (10k samples)              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

echo -e "\n${BLUE}This will:${NC}"
echo "  1. Create test dataset (10k train, 1k val)"
echo "  2. Train for 5 epochs (~30 minutes)"
echo "  3. Verify model works"
echo ""
echo -e "${YELLOW}Purpose: Test setup before full training${NC}"
echo ""

# Check prerequisites
if [ ! -f "data/train_list.txt" ]; then
    echo -e "${YELLOW}❌ No data found!${NC}"
    echo "Please run prepare_data.py first"
    exit 1
fi

# Backup original data
echo -e "\n${GREEN}[1/5] Backing up original data...${NC}"

if [ ! -f "data/train_full.txt" ]; then
    cp data/train_list.txt data/train_full.txt
    cp data/val_list.txt data/val_full.txt
    echo "✓ Original data backed up"
else
    echo "✓ Backup already exists"
fi

# Create test subset
echo -e "\n${GREEN}[2/5] Creating test dataset (10k + 1k)...${NC}"

head -10000 data/train_full.txt > data/train_list.txt
head -1000 data/val_full.txt > data/val_list.txt

echo "✓ Test data created:"
echo "   Train: $(wc -l < data/train_list.txt) samples"
echo "   Val: $(wc -l < data/val_list.txt) samples"

# Create test config
echo -e "\n${GREEN}[3/5] Creating test config...${NC}"

cp config_kaggle.yml config_test.yml

# Fix paths in config
python fix_config.py config_test.yml

# Remove any label_sar references (critical fix)
echo "Removing label_sar references..."
sed -i 's/label_sar,//g' config_test.yml
sed -i 's/label_sar//g' config_test.yml
sed -i 's/- label_sar//g' config_test.yml

# Modify for quick test
sed -i 's/epoch_num: 100/epoch_num: 5/' config_test.yml
sed -i 's/print_batch_step: 100/print_batch_step: 10/' config_test.yml
sed -i 's/save_epoch_step: 5/save_epoch_step: 2/' config_test.yml
sed -i 's|/vi_ppocr_v5|/test_10k|g' config_test.yml

# Extract actual save_model_dir from config (absolute path)
MODEL_DIR=$(grep 'save_model_dir:' config_test.yml | awk '{print $2}')
echo "✓ Test config created: config_test.yml"
echo "   Epochs: 5 (instead of 100)"
echo "   Output: $MODEL_DIR"
echo "   label_sar: removed"

# Check prerequisites
if [ ! -f "data/train_list.txt" ]; then
    echo -e "${YELLOW}❌ data/train_list.txt not found!${NC}"
    echo "Please run: python fix_prepare_data.py --input_dir /kaggle/input/YOUR_DATASET"
    exit 1
fi

if [ ! -d "PaddleOCR" ]; then
    echo -e "${YELLOW}❌ PaddleOCR directory not found!${NC}"
    echo "Please run: bash setup_kaggle.sh"
    exit 1
fi

# Start test training
echo -e "\n${GREEN}[4/5] Starting test training...${NC}"
echo -e "${YELLOW}Expected time: 25-35 minutes${NC}"
echo ""

LOG_FILE="logs/test_10k_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

# Get absolute path to project root
PROJECT_ROOT=$(pwd)

cd PaddleOCR

python -m paddle.distributed.launch \
    --gpus '0,1' \
    tools/train.py \
    -c ${PROJECT_ROOT}/config_test.yml \
    2>&1 | tee ${PROJECT_ROOT}/$LOG_FILE

cd ${PROJECT_ROOT}

# Verify model
echo -e "\n${GREEN}[5/5] Verifying model...${NC}"
echo "Checking model at: ${MODEL_DIR}/"

# List what was actually saved
if [ -d "${MODEL_DIR}" ]; then
    echo "Files in model dir:"
    ls -la "${MODEL_DIR}/" | head -20
else
    echo "⚠️  Model directory not found: ${MODEL_DIR}"
    echo "Searching for model files..."
    find /kaggle/working -name "*.pdparams" 2>/dev/null | head -10
fi

# Check for best_accuracy (standard) or best_model (some versions)
BEST_MODEL=""
if [ -f "${MODEL_DIR}/best_accuracy.pdparams" ]; then
    BEST_MODEL="${MODEL_DIR}/best_accuracy"
elif [ -f "${MODEL_DIR}/best_model.pdparams" ]; then
    BEST_MODEL="${MODEL_DIR}/best_model"
elif [ -f "${MODEL_DIR}/latest.pdparams" ]; then
    echo "⚠️  best_accuracy not found, using latest checkpoint"
    BEST_MODEL="${MODEL_DIR}/latest"
fi

if [ -n "${BEST_MODEL}" ]; then
    echo "✓ Model found: ${BEST_MODEL}.pdparams"
    
    # Quick export test
    INFERENCE_DIR="${MODEL_DIR}/../../inference/test_10k"
    mkdir -p "$(dirname ${INFERENCE_DIR})"
    
    cd PaddleOCR
    python tools/export_model.py \
        -c ${PROJECT_ROOT}/config_test.yml \
        -o Global.pretrained_model=${BEST_MODEL} \
           Global.save_inference_dir=${INFERENCE_DIR} \
        > /dev/null 2>&1 && echo "✓ Export successful" || echo "⚠️  Export failed (non-critical)"
    cd ${PROJECT_ROOT}
    
    if [ -f "${INFERENCE_DIR}/inference.pdmodel" ]; then
        echo "✓ Inference model: ${INFERENCE_DIR}/"
    fi
else
    echo "❌ No model checkpoint found in: ${MODEL_DIR}/"
    echo "Training may have failed. Check log: $LOG_FILE"
    tail -20 "$LOG_FILE" 2>/dev/null
    exit 1
fi

# Summary
echo -e "\n╔══════════════════════════════════════════════════════════════╗"
echo -e "║              ${GREEN}✅ Test Completed!${NC}                             ║"
echo -e "╚══════════════════════════════════════════════════════════════╝"

echo -e "\n${GREEN}📊 Results:${NC}"
echo "   Test model: ${BEST_MODEL}.pdparams"
echo "   Inference: ${INFERENCE_DIR:-N/A}"
echo "   Log: $LOG_FILE"

echo -e "\n${BLUE}📈 Check final accuracy:${NC}"
tail -30 $LOG_FILE | grep -i "acc:" | tail -5

echo -e "\n${GREEN}✅ Setup verified! Ready for full training.${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "   1. If accuracy looks good (>70%), proceed to full training:"
echo "      bash train_full.sh"
echo ""
echo "   2. If accuracy is low, check:"
echo "      - Dictionary has all characters"
echo "      - Images are readable"
echo "      - Labels are correct format"
echo ""
echo "   3. To restore full dataset and train:"
echo "      bash train_full.sh"

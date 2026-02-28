#!/bin/bash

# Kaggle Setup Script for PaddleOCR v5 Vietnamese
# Cài đặt môi trường hoàn chỉnh trên Kaggle

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      PaddleOCR v5 Vietnamese - Kaggle Setup                ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Step 1: Check GPU
echo -e "\n${BLUE}[1/7]${NC} ${GREEN}Checking GPU...${NC}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
    echo -e "${YELLOW}⚠️  No GPU detected!${NC}"
}

# Step 2: Install PaddlePaddle
echo -e "\n${BLUE}[2/7]${NC} ${GREEN}Installing PaddlePaddle 3.0 GPU...${NC}"
pip install -q paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/ || {
    echo -e "${YELLOW}GPU version failed, installing CPU version...${NC}"
    pip install -q paddlepaddle==3.0.0b1
}

# Step 3: Clone PaddleOCR
echo -e "\n${BLUE}[3/7]${NC} ${GREEN}Cloning PaddleOCR...${NC}"
if [ ! -d "PaddleOCR" ]; then
    git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git
    echo "✓ PaddleOCR cloned"
else
    echo "✓ PaddleOCR already exists"
fi

# Step 4: Install dependencies
echo -e "\n${BLUE}[4/7]${NC} ${GREEN}Installing dependencies...${NC}"
cd PaddleOCR
pip install -q -r requirements.txt
pip install -q visualdl shapely scikit-image imgaug lmdb tqdm
cd ..

# Step 5: Download pretrained model
echo -e "\n${BLUE}[5/7]${NC} ${GREEN}Downloading pretrained model...${NC}"
mkdir -p pretrain_models

if [ ! -f "pretrain_models/en_PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
    echo "Downloading Latin PP-OCRv5 mobile model..."
    wget -q --show-progress https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/latin_PP-OCRv5_mobile_rec_pretrained.pdparams \
        -O pretrain_models/en_PP-OCRv5_mobile_rec_pretrained.pdparams
    echo "✓ Pretrained model downloaded"
else
    echo "✓ Pretrained model already exists"
fi

# Step 6: Create directories
echo -e "\n${BLUE}[6/7]${NC} ${GREEN}Creating directories...${NC}"
mkdir -p data/train_data
mkdir -p data/val_data
mkdir -p output/vi_ppocr_v5
mkdir -p inference
mkdir -p dict
mkdir -p logs

echo "✓ Directories created"

# Step 7: Verify installation
echo -e "\n${BLUE}[7/7]${NC} ${GREEN}Verifying installation...${NC}"

python3 << 'PYTHON'
import paddle
print(f"✓ PaddlePaddle version: {paddle.__version__}")
print(f"✓ CUDA available: {paddle.device.is_compiled_with_cuda()}")
print(f"✓ GPU count: {paddle.device.cuda.device_count()}")

import sys
if paddle.device.is_compiled_with_cuda():
    print(f"✓ GPU ready for training")
else:
    print("⚠️  CPU only mode")
PYTHON

# Summary
echo -e "\n╔══════════════════════════════════════════════════════════════╗"
echo -e "║                  ${GREEN}✅ Setup Completed!${NC}                        ║"
echo -e "╚══════════════════════════════════════════════════════════════╝"

echo -e "\n${GREEN}📝 Next steps:${NC}"
echo "   1. Prepare your data:"
echo "      python prepare_data.py \\"
echo "        --input_dir /kaggle/input/your-dataset \\"
echo "        --images_dir images \\"
echo "        --label_file rec_gt.txt"
echo ""
echo "   2. Start training:"
echo "      bash train_kaggle.sh"
echo ""
echo "   3. Monitor progress:"
echo "      tail -f logs/train.log"

echo -e "\n${GREEN}🎯 Ready to train!${NC}\n"

# PaddleOCR v5 Vietnamese - Training trên Kaggle

Repository hoàn chỉnh để training PaddleOCR v5 cho tiếng Việt trên Kaggle với GPU miễn phí.

## 📋 Tổng quan

- ✅ Training hoàn toàn trên Kaggle (GPU T4 x2 miễn phí)
- ✅ Hỗ trợ dataset với cấu trúc `images/` + `rec_gt.txt`
- ✅ Tự động split train/val
- ✅ Tạo dictionary tiếng Việt
- ✅ 1 notebook - chạy hết
- ✅ Timeline: ~16-20 giờ cho 250k samples

## 🚀 Quick Start - 4 Bước

### Bước 1: Upload Dataset lên Kaggle

Dataset của bạn có cấu trúc:
```
your-dataset/
├── images/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── rec_gt.txt
```

File `rec_gt.txt` format:
```
img_001.jpg	Xin chào Việt Nam
img_002.jpg	PaddleOCR v5 rất mạnh
```

**Upload lên Kaggle:**
1. Vào https://www.kaggle.com/datasets
2. Click "New Dataset"
3. Upload folder dataset
4. Title: `vietnamese-ocr-250k` (hoặc tên bạn thích)
5. Click "Create"

### Bước 2: Fork GitHub Repo

1. Fork repo này: https://github.com/YOUR_USERNAME/paddleocr-v5-vietnamese
2. Hoặc clone về local:
```bash
git clone https://github.com/YOUR_USERNAME/paddleocr-v5-vietnamese.git
```

### Bước 3: Tạo Kaggle Notebook

1. Vào https://www.kaggle.com/code
2. Click "New Notebook"
3. **Settings** (góc phải):
   - Accelerator: **GPU T4 x2**
   - Internet: **ON**
   - Persistence: **ON**
4. Click "Add data" → Tìm dataset của bạn → Add

### Bước 4: Copy Notebook Code

Copy code từ file `kaggle_training.ipynb` trong repo này vào notebook, hoặc tải notebook trực tiếp.

**Chạy theo thứ tự từng cell** → Xong!

## 🎯 Training Workflow (2 giai đoạn)

### Giai đoạn 1: Test với 10k samples (~30 phút) ⚡
**Mục đích**: Verify setup, config, data trước khi train full

```bash
bash train_test_10k.sh
```

Script này sẽ:
- Backup dataset gốc
- Tạo subset 10k train + 1k val
- Train 5 epochs (~30 phút)
- Export model test
- Verify model hoạt động

**Kết quả mong đợi**: Accuracy >70% sau 5 epochs

### Giai đoạn 2: Training Full (~16-20 giờ) 🚀
**Sau khi test OK**, train với toàn bộ data:

```bash
bash train_full.sh
```

Script này sẽ:
- Restore full dataset
- Confirm với bạn trước khi start
- Train 100 epochs với full data
- Save checkpoint mỗi 5 epochs

---

## 📓 Nội dung Notebook

### Cell 1: Clone Project (1 phút)
```python
!git clone https://github.com/YOUR_USERNAME/paddleocr-v5-vietnamese.git
%cd paddleocr-v5-vietnamese
```

### Cell 2: Setup Environment (5 phút)
```python
!bash setup_kaggle.sh
```

### Cell 3: Prepare Data (2-5 phút)
```python
# Split train/val và tạo dictionary
!python prepare_data.py \
    --input_dir /kaggle/input/vietnamese-ocr-250k \
    --images_dir images \
    --label_file rec_gt.txt \
    --train_ratio 0.9
```

### Cell 4a: Test với 10k samples (30 phút) - KHUYẾN NGHỊ
```python
# Test setup trước khi train full
!bash train_test_10k.sh
```

### Cell 4b: Training Full (15-20 giờ)
```python
# Sau khi test OK, train full data
!bash train_full.sh
```

### Cell 5: Export Model (2 phút)
```python
!bash export_model.sh
!tar -czf model.tar.gz inference/ dict/
```

Download từ Output tab! ✅

## 📊 Timeline

| Bước | Thời gian | Ghi chú |
|------|-----------|---------|
| Upload dataset | 10-30 phút | 1 lần duy nhất |
| Setup Kaggle | 5-7 phút | Mỗi session |
| Prepare data | 2-5 phút | Tự động split & dictionary |
| Training 250k | 16-20 giờ | Fit trong 30h/week free |
| Export | 2 phút | |

**Tổng**: ~17-21 giờ

## 🎯 Kết quả Mong đợi

| Dataset Size | Training Time | Accuracy |
|--------------|---------------|----------|
| 50k samples  | 3-4 giờ      | ~90%     |
| 100k samples | 7-9 giờ      | ~93%     |
| 250k samples | 16-20 giờ    | >95%     |

## 📁 Cấu trúc Project

```
paddleocr-v5-vietnamese/
├── README.md                    # File này
├── kaggle_training.ipynb        # Notebook hoàn chỉnh
├── setup_kaggle.sh              # Setup script
├── prepare_data.py              # Xử lý data
├── train_kaggle.sh              # Training script
├── export_model.sh              # Export script
├── test_model.py                # Test inference
├── config_kaggle.yml            # Config cho Kaggle
└── requirements.txt             # Dependencies
```

## 💡 Tips & Tricks

### 1. Test với dataset nhỏ trước
```python
# Test với 10k samples
!python prepare_data.py \
    --input_dir /kaggle/input/vietnamese-ocr-250k \
    --images_dir images \
    --label_file rec_gt.txt \
    --train_ratio 0.9 \
    --max_samples 10000  # Limit samples
```

### 2. Monitor Training
```python
# Check log realtime
!tail -f logs/train.log

# Check GPU
!nvidia-smi
```

### 3. Resume Training (nếu timeout)
```python
# Resume từ checkpoint
!bash train_kaggle.sh --resume
```

### 4. Save Checkpoint
```python
# Save checkpoint để backup
!tar -czf checkpoint_epoch_30.tar.gz output/
# Upload lên Kaggle Dataset
```

## 🔧 Troubleshooting

### GPU Out of Memory?
```yaml
# Sửa trong config_kaggle.yml
batch_size: 48  # Giảm từ 96
```

### Dataset không tìm thấy?
```python
# Check dataset path
!ls /kaggle/input/
# Sửa đúng tên trong prepare_data.py
```

### Training chậm?
```python
# Giảm num_workers
# Trong config_kaggle.yml: num_workers: 2
```

## 📚 Chi tiết từng file

### `prepare_data.py`
- Đọc `rec_gt.txt`
- Copy images từ `images/` sang `train_data/` và `val_data/`
- Tạo `train_list.txt` và `val_list.txt`
- Tạo dictionary tiếng Việt tự động
- Split theo tỷ lệ 90/10 (configurable)

### `setup_kaggle.sh`
- Install PaddlePaddle 3.0 GPU
- Clone PaddleOCR
- Install dependencies
- Tạo directories

### `train_kaggle.sh`
- Training với config tối ưu cho Kaggle
- Multi-GPU (T4 x2)
- Save checkpoint tự động
- Log đầy đủ

### `export_model.sh`
- Export sang inference format
- Dọn dẹp files không cần
- Tạo package để download

## 🆘 Support

Gặp vấn đề? 

1. Check [Troubleshooting](#troubleshooting)
2. Xem log: `!tail -100 logs/train.log`
3. Check GPU: `!nvidia-smi`
4. Tạo issue trên GitHub

## 📝 Notes

- Kaggle free tier: **30 GPU hours/week**
- Session timeout: **12 giờ** (enable Persistence để save)
- Dataset max size: **20GB** (free tier)
- Internet bắt buộc ON để download pretrained model

## ✅ Checklist

Trước khi training:
- [ ] Dataset uploaded lên Kaggle
- [ ] Notebook created với GPU T4 x2
- [ ] Internet ON
- [ ] Dataset added vào notebook
- [ ] Fork/clone GitHub repo

Sau khi training:
- [ ] Model exported
- [ ] Downloaded về local
- [ ] Test inference thành công
- [ ] Backup checkpoint (nếu cần)

## 🎓 License

MIT License

## 🤝 Contributing

Pull requests are welcome!

---

**Ready to train?** Mở `kaggle_training.ipynb` và bắt đầu! 🚀

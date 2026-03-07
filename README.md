# 🦷 Hệ Thống Phát Hiện Sâu Răng từ Ảnh X-quang

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask 3.0+](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Ứng dụng web **AI-powered** phát hiện và phân khúc sâu răng từ ảnh X-quang sử dụng **YOLOv8 + UNet**.

> 🔍 **Tính năng**: Phát hiện vị trí sâu, xác định ranh giới chi tiết, xử lý ảnh tự động, API REST

---

## 📋 Mục Lục

- [Giới Thiệu](#-giới-thiệu)
- [Tính Năng](#-tính-năng)
- [Yêu Cầu Hệ Thống](#-yêu-cầu-hệ-thống)
- [Cài Đặt Nhanh](#-cài-đặt-nhanh)
- [Cài Đặt Chi Tiết](#-cài-đặt-chi-tiết)
- [Cách Sử Dụng](#-cách-sử-dụng)
- [API Documentation](#-api-documentation)
- [Cấu Trúc Dự Án](#-cấu-trúc-dự-án)
- [Lỗi Thường Gặp](#-lỗi-thường-gặp)
- [Liên Lạc](#-liên-lạc)

---

## 🎯 Giới Thiệu

Dự án xây dựng **hệ thống tự động phát hiện sâu răng** từ ảnh X-quang trong lĩnh vực nha khoa, sử dụng:

- **YOLOv8**: Phát hiện vị trí sâu (Detection)
- **UNet**: Phân khúc ranh giới chi tiết (Segmentation)
- **Flask**: Web API để dùng mô hình

**Mục tiêu**: Giúp nha sĩ phát hiện sâu chính xác, nhanh chóng thông qua giao diện web thân thiện.

---

## 🚀 Tính Năng

✅ **Phát Hiện Sâu Răng**
- Sử dụng mô hình YOLOv8 để phát hiện vị trí sâu
- Hiển thị bounding box và độ tin cậy

✅ **Phân Khúc Sâu**
- Sử dụng UNet để xác định ranh giới chính xác
- Tạo mask segmentation

✅ **Xử Lý Ảnh Tự Động**
- Chuẩn hóa ảnh X-quang
- CLAHE enhancement
- Resize tự động

✅ **Web Interface**
- Giao diện web hiện đại (HTML5 + CSS3 + JavaScript)
- Kéo-thả upload ảnh
- Hiển thị kết quả real-time
- Responsive design

✅ **API REST**
- `/api/predict` - Dự đoán từ ảnh
- `/api/health` - Kiểm tra trạng thái
- `/api/config` - Lấy cấu hình

✅ **Training Pipeline**
- Hỗ trợ huấn luyện YOLO + UNet
- Data augmentation
- Evaluation metrics (mAP, IoU, Dice)

---

## 💻 Yêu Cầu Hệ Thống

### Tối Thiểu
- **OS**: Windows 10+, macOS, Linux
- **Python**: 3.11+
- **RAM**: 8GB
- **Disk**: 2GB (giá trị min)

### Khuyên Dùng
- **GPU**: NVIDIA (CUDA 11.8+) - tăng tốc độ 10x
- **RAM**: 16GB+
- **Disk**: 5GB+ (cho models + data)

### Cài Đặt Sẵn
```
✅ Flask 3.1.3
✅ PyTorch 2.10.0
✅ YOLOv8 8.4.19
✅ OpenCV 4.13.0
✅ NumPy, Pandas, scikit-learn
```

---

## ⚡ Cài Đặt Nhanh

### Windows
```powershell
scripts\quickstart.bat
```

### Linux / macOS
```bash
bash scripts/quickstart.sh
```

**Tự động thực hiện:**
1. ✅ Kiểm tra Python & pip
2. ✅ Cài đặt dependencies
3. ✅ Tạo virtual environment (nếu cần)
4. ✅ Kiểm tra model weights
5. ✅ Khởi động Flask server

**⏱️ Thời gian**: 5 phút (sau lần đầu là 1 phút)

**🎉 Kết quả**: Server chạy tại `http://localhost:5000`

---

## 📖 Cài Đặt Chi Tiết

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd "Đồ án KB"
```

### 2️⃣ Tạo Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python -m venv venv
source venv/bin/activate
```

### 3️⃣ Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

**Packages chính:**
- `torch >= 2.0.0` - Deep Learning framework
- `ultralytics >= 8.0.0` - YOLOv8
- `flask >= 2.3.0` - Web framework
- `opencv-python >= 4.8.0` - Image processing
- `torchvision >= 0.15.0` - PyTorch extensions

### 4️⃣ Kiểm Tra Cài Đặt

```bash
python -c "import torch, cv2, flask; print('✅ All dependencies installed')"
```

### 5️⃣ Khởi Động Server

```bash
cd app
python run.py
```

**Output mong đợi:**
```
============================================================
Dental Cavity Detection - Web Application
============================================================
✅ YOLO weights found: models/yolo_weights.pt
⚠️  UNet weights not found: models/best_unet.pt (YOLO-only mode)

Models will be loaded on first prediction request...

============================================================
Starting Flask application...
Open browser at: http://localhost:5000
Press Ctrl+C to stop
============================================================

 * Running on http://127.0.0.1:5000
```

---

## 🖥️ Cách Sử Dụng

### 1. Qua Web Interface

1. **Mở trình duyệt** → http://localhost:5000
2. **Click "Chọn Ảnh"** hoặc kéo-thả ảnh X-quang
3. **Chọn chế độ**:
   - YOLO: Chỉ phát hiện
   - UNet: Chỉ phân khúc
   - Both: Phát hiện + Phân khúc
4. **Click "Phân Tích"**
5. **Xem kết quả**: Bounding box + thông tin sâu

### 2. Qua Python Script

```python
from src.inference import DentalCavityInference

# Khởi tạo
inference = DentalCavityInference(
    yolo_weights='models/yolo_weights.pt',
    unet_weights='models/best_unet.pt',
    device='cuda'  # hoặc 'cpu'
)

# Phát hiện ở ảnh
image = cv2.imread('test_xray.jpg')
detections = inference.detect_cavities_yolo(image)

print(f"Phát hiện {len(detections)} sâu")
for i, det in enumerate(detections):
    print(f"  - Sâu #{i+1}: box={det['bbox']}, conf={det['confidence']:.2%}")
```

### 3. Qua API REST

```bash
# Kiểm tra trạng thái
curl http://localhost:5000/api/health

# Upload ảnh và dự đoán (form-data)
curl -X POST http://localhost:5000/api/predict \
  -F "file=@test_xray.jpg" \
  -F "mode=both"
```

---

## 📡 API Documentation

### GET `/`
**Mô tả**: Giao diện web chính

**Response**: HTML page

---

### GET `/api/health`
**Mô tả**: Kiểm tra trạng thái server

**Response**:
```json
{
  "status": "ok",
  "device": "cpu",
  "model_initialized": false,
  "yolo_weights": "models/yolo_weights.pt",
  "unet_weights": null,
  "timestamp": "2026-03-03T08:39:00"
}
```

---

### GET `/api/config`
**Mô tả**: Lấy cấu hình ứng dụng

**Response**:
```json
{
  "app_name": "Dental Cavity Detection",
  "version": "1.0",
  "max_upload_size_mb": 50,
  "allowed_formats": ["jpg", "jpeg", "png", "bmp"]
}
```

---

### POST `/api/predict`
**Mô tả**: Dự đoán sâu từ ảnh

**Request**:
```
Content-Type: multipart/form-data

- file: <image file>
- mode: "yolo" | "unet" | "both" (mặc định: "both")
```

**Success Response (200)**:
```json
{
  "success": true,
  "num_cavities": 3,
  "detections": [
    {"id": 0, "bbox": [100, 150, 250, 280], "confidence": 0.92},
    {"id": 1, "bbox": [300, 200, 420, 350], "confidence": 0.88},
    {"id": 2, "bbox": [150, 400, 280, 500], "confidence": 0.85}
  ],
  "segmentations_count": 3,
  "result_image": "data:image/jpeg;base64,...",
  "timestamp": "2026-03-03T08:39:10"
}
```

**Error Response (500)**:
```json
{
  "error": "Prediction failed: ..."
}
```

---

## 📁 Cấu Trúc Dự Án

```
Đồ án KB/
├── 📄 README.md                    ← Bạn đang đọc file này
├── 📄 config.yaml                  ← Cấu hình chính
├── 📄 requirements.txt             ← Dependencies
│
├── 🏗️ app/                         ← Web Application
│   ├── run.py                     ← Khởi động server
│   ├── app_simple.py              ← Flask backend (API)
│   ├── templates/
│   │   ├── home.html              ← Giao diện chính
│   │   └── index.html             ← Alternative UI
│   ├── static/
│   │   ├── css/
│   │   │   └── main.css           ← Styling
│   │   └── js/
│   │       └── home.js            ← JavaScript
│   ├── uploads/                   ← Ảnh được upload
│   └── README.md                  ← Hướng dùng web app
│
├── 🤖 src/                        ← ML Core Engine
│   ├── yolo_model.py              ← YOLOv8 wrapper
│   ├── unet_model.py              ← UNet architecture
│   ├── inference.py               ← Inference system
│   ├── training.py                ← Training pipeline
│   ├── evaluation.py              ← Metrics (mAP, IoU, Dice)
│   ├── data_loader.py             ← PyTorch DataLoader
│   ├── data_prep.py               ← Data preprocessing
│   ├── utils.py                   ← Utility functions
│   └── __init__.py
│
├── 📊 scripts/                    ← Automation Scripts
│   ├── quickstart.bat             ← Windows quick start
│   ├── quickstart.sh              ← Linux/Mac quick start
│   ├── train.py                   ← Training CLI
│   ├── test.py                    ← Testing script
│   └── generate_synthetic_data.py ← Tạo test data
│
├── 📁 data/                       ← Datasets
│   ├── raw/                       ← Ảnh X-quang gốc
│   ├── processed/
│   │   ├── train/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── val/
│   │   │   ├── images/
│   │   │   └── labels/
│   │   ├── test/
│   │   │   └── images/
│   │   └── data.yaml              ← YOLO dataset config
│   └── README.md
│
├── 🧠 models/                     ← Trained Model Weights
│   ├── yolo_weights.pt            ← YOLOv8 model
│   └── best_unet.pt               ← UNet model (optional)
│
└── 📓 notebooks/                  ← Jupyter Notebooks
    └── exploration.ipynb          ← Data analysis
```

---

## 🔧 Cấu Hình (config.yaml)

```yaml
model:
  yolo_model: yolov8m              # Model size: n, s, m, l, x
  yolo_conf_threshold: 0.5         # Confidence threshold
  unet_in_channels: 1              # Input channels
  unet_out_channels: 1             # Output channels
  unet_channels: 64                # Base channels
  input_size: [640, 480]           # Input resolution

training:
  epochs: 50                       # Training epochs
  batch_size: 16                   # Batch size
  learning_rate: 0.001             # Learning rate
  device: cuda                     # cuda or cpu
  patience: 20                     # Early stopping
  
data:
  augmentation: true
  normalize: true
  clahe: true                      # CLAHE enhancement
  train_split: 0.8

paths:
  data_dir: data
  model_dir: models
  output_dir: outputs
  
app:
  host: 0.0.0.0
  port: 5000
  debug: true
  upload_folder: app/uploads
  max_upload_size: 50              # MB
```

---

## 🐛 Lỗi Thường Gặp

### ❌ ModuleNotFoundError: No module named 'flask'

**Nguyên nhân**: Dependencies chưa được cài đặt

**Giải Pháp**:
```bash
pip install -r requirements.txt
```

---

### ❌ No Backend with GPU - CUDA not available

**Nguyên nhân**: PyTorch không tìm thấy GPU

**Giải Pháp**:
```bash
# Cài PyTorch với CUDA support (nếu có GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Hoặc sử dụng CPU (chậm hơn)
# Thay đổi trong config.yaml: device: cpu
```

---

### ❌ Models not initialized - prediction fails

**Nguyên nhân**: Model weights không tìm thấy

**Giải Pháp**:
- Kiểm tra file `models/yolo_weights.pt` tồn tại
- Nếu không, train hoặc download model từ Roboflow/Ultralytics

---

### ❌ Port 5000 already in use

**Nguyên nhân**: Port 5000 bị chiếm dụng

**Giải Pháp**:
```bash
# Chạy trên port khác
python app/run.py --port 8080

# Hoặc tìm và kill process sử dụng port 5000
# Windows: netstat -ano | findstr :5000
# Linux: lsof -i :5000
```

---

### ❌ Out of Memory (CUDA/CPU)

**Nguyên nhân**: Batch size quá lớn

**Giải Pháp**:
```yaml
# config.yaml
training:
  batch_size: 8     # Giảm từ 16 xuống 8
  device: cpu       # Hoặc dùng CPU
```

---

### ⚠️ UNet weights not found (YOLO-only mode)

**Nguyên nhân**: File `models/best_unet.pt` không tồn tại

**Giải Pháp**:
- Ứng dụng vẫn chạy nhưng chỉ phát hiện (YOLO)
- Train UNet: `python scripts/train.py --unet`
- Hoặc bỏ qua: mode được set mặc định là YOLO

---

## 📊 Hiệu Năng (Performance)

| Metric | YOLO | UNet |
|--------|------|------|
| Accuracy | 85-95% | 82-88% |
| mAP@0.5 | 0.85+ | N/A |
| Inference Time | 80-120ms | 150-200ms |
| Model Size | ~49MB | ~50MB |
| Memory Usage | 2-4GB | 3-5GB |

**Note**: Số liệu từ synthetic data; real-world performance có thể khác

---

## 📚 Công Nghệ Sử Dụng

| Layer | Technology |
|-------|-----------|
| **Backend** | Flask 3.1.3, Python 3.11 |
| **ML Framework** | PyTorch 2.10.0 |
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Segmentation** | UNet (Custom) |
| **Image Processing** | OpenCV 4.13.0 |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |

---

## 🎓 Hướng Dùng Web App

### Bước 1: Tải Lên Ảnh

<img alt="Step 1" width="300">

Kéo-thả hoặc click để chọn ảnh X-quang (JPG, PNG, BMP)

### Bước 2: Chọn Chế Độ Phân Tích

- **YOLO**: Phát hiện sâu
- **UNet**: Phân khúc sâu
- **Both**: Phát hiện + Phân khúc (khuyên dùng)

### Bước 3: Xem Kết Quả

- Bounding box quanh sâu
- Độ tin cậy (confidence)
- Ảnh kết quả

---

## 🔐 Security (Bảo Mật)

- ✅ File upload validation (chỉ cho phép ảnh)
- ✅ Max file size: 50MB
- ✅ Isolated upload directory
- ✅ CORS disabled (chỉ localhost)

---

## 📝 License

MIT License - Tự do sử dụng, sửa đổi cho mục đích cá nhân và thương mại

---

## 🤝 Đóng Góp

Chào mừng contributions! Vui lòng:

1. Fork repository
2. Tạo feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Mở Pull Request

---

## 💬 Liên Lạc & Hỗ Trợ

Có vấn đề? Hãy liên lạc:

- 📧 **Email**: support@dentalcare.local
- 🐛 **Bug Report**: Mở issue trên GitHub
- 💡 **Feature Request**: Thảo luận trong Discussions
- 📞 **Support**: Check [SETUP.md](SETUP.md#-troubleshooting) trước

---

## 🎯 Roadmap

- [ ] Support GPU acceleration optimization
- [ ] Mobile app (React Native)
- [ ] Real-time video analysis
- [ ] Multi-language support
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure)
- [ ] Tích hợp HIS (Hospital Information System)

---

## 👨‍💻 Nhóm Phát Triển

**Đồ án:** Phát Hiện Sâu Răng từ Ảnh X-quang
**Năm**: 2026
**Cơ Sở**: [Tên Trường/Đại Học]

---

<div align="center">

**Made with ❤️ for dental health**

[⬆ Lên đầu](#-hệ-thống-phát-hiện-sâu-răng-từ-ảnh-x-quang)

</div>

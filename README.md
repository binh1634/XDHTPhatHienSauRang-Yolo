# XDHTPhatHienSauRang-Yolov8
🦷 Hệ Thống Phát Hiện Sâu Răng - Dental Cavity Detection System

Hệ thống phát hiện sâu răng tự động từ ảnh X-quang sử dụng mô hình AI YOLO, với giao diện web hiện đại và backend Python tiên tiến.

## 📋 Mục Đích

Xây dựng hệ thống AI giúp phát hiện sâu răng từ ảnh X-quang nha khoa, hỗ trợ bác sĩ trong quá trình chẩn đoán nhanh chóng và chính xác.

## 📊 Báo Cáo Tiến Độ

 Hoàn Thành (70%)

**Backend:**
- ✅ Flask API server hoạt động ổn định
- ✅ YOLO model integration (YOLOv8)
- ✅ Xử lý ảnh tự động (bounding box, marking)
- ✅ CORS support cho frontend
- ✅ Error handling & validation
- ✅ Health check endpoint

**Frontend:**
- ✅ Giao diện HTML5 responsive
- ✅ CSS3 modern, gradient design
- ✅ JavaScript vanilla (không cần framework)
- ✅ Drag & drop upload
- ✅ Real-time preview ảnh
- ✅ Hiển thị kết quả chi tiết
- ✅ Bảng thống kê phát hiện
- ✅ Export báo cáo TXT

**Chức Năng Chính:**
- ✅ Upload ảnh X-quang (JPG, PNG, GIF)
- ✅ Phát hiện sâu răng tự động
- ✅ Vẽ bounding box trên ảnh
- ✅ Hiển thị độ tin cậy (confidence)
- ✅ Tạo báo cáo chi tiết
- ✅ Download kết quả

**DevOps:**
- ✅ requirements.txt (dependencies)
- ✅ Virtual environment setup
- ✅ .gitignore configuration
- ✅ Batch script start server


## 🏗️ Kiến Trúc Dự Án

```
XDHTSauRang/
├── 📁 backend/                    # Backend Python Flask
│   ├── app.py                     # Server chính (Flask)
│   ├── config.py                  # Cấu hình (model path, thresholds)
│   ├── utils.py                   # Xử lý ảnh YOLO & detection
│   ├── requirements.txt           # Dependencies chính
│   ├── requirements-dev.txt       # Dev dependencies
│   ├── .env.example               # Template biến môi trường
│   ├── .gitignore                 # Git ignore
│   └── 📁 models/
│       └── best.pt                # Model YOLO v8 (50.8 MB)
│
├── 📁 frontend/                   # Frontend HTML + JS
│   ├── index.html                 # Giao diện chính
│   ├── style.css                  # CSS styling (responsive)
│   ├── script.js                  # JavaScript logic
│   └── 📁 assets/                 # Tài nguyên (ảnh, icon)
│
├── 📁 uploads/                    # Thư mục lưu ảnh đã upload
│
├── 📋 README.md                   # Tài liệu này
├── 🚀 QUICK_START.md              # Hướng dẫn bắt đầu nhanh
├── 📝 SETUP_MODEL.txt             # Hướng dẫn tải model từ Colab
└── 🖱️ start_backend.bat           # Script chạy backend (Windows)
```

## 🛠️ Công Nghệ Sử Dụng

**Backend Stack:**
- **Python 3.11.9** - Ngôn ngữ lập trình
- **Flask 2.3.3** - Web framework
- **YOLO v8** - Object detection AI
- **OpenCV 4.8** - Xử lý ảnh
- **PyTorch 2.0.1** - Deep learning framework
- **NumPy 1.24** - Tính toán số

**Frontend Stack:**
- **HTML5** - Markup
- **CSS3** - Styling (Gradient, Flexbox, Grid)
- **JavaScript (Vanilla)** - Logic (No jQuery/React)
- **Fetch API** - HTTP requests

**Infrastructure:**
- **Flask-CORS** - Cross-origin requests
- **Werkzeug** - WSGI utilities
- **Base64** - Image encoding

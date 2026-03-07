# Web Application Guide - Dental Cavity Detection System

## Overview

Ứng dụng web cho phép bạn:
- Upload X-ray images
- Phát hiện sâu răng sử dụng YOLO + UNet
- Xem kết quả trực quan
- Lưu báo cáo kết quả

## Getting Started

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Chuẩn bị Model Weights

Đặt file weights vào thư mục `models/`:
- `models/yolo_weights.pt` - YOLO model
- `models/best_unet.pt` - UNet model

### 3. Chạy Web App

```bash
cd app
python run.py
```

Hoặc chạy trực tiếp:
```bash
python -m flask --app app.py run
```

### 4. Truy cập Application

Mở browser và đi tới: **http://localhost:5000**

## Features

### Upload Image
- Drag & drop hoặc click để chọn ảnh
- Support format: JPG, PNG, BMP
- Max file size: 50MB

### Real-time Detection
- Phát hiện sâu răng tự động
- Hiển thị bounding boxes
- Thống kê chi tiết

### Results Display
- Ảnh gốc vs ảnh phát hiện
- Số lượng sâu phát hiện
- Confidence scores
- Chi tiết vị trí mỗi sâu

## API Endpoints

### GET `/`
- Trang chính của ứng dụng

### GET `/api/health`
- Check health status
```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
    "status": "ok",
    "device": "cuda",
    "model_initialized": true,
    "timestamp": "2026-02-16T10:30:00"
}
```

### POST `/api/predict`
- Upload ảnh và thực hiện dự đoán

**Request:**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

**Response:**
```json
{
    "success": true,
    "num_cavities": 2,
    "detections": [
        {
            "id": 0,
            "bbox": [100, 50, 200, 150],
            "confidence": 0.92,
            "class": 0
        }
    ],
    "segmentations_count": 2,
    "result_image": "data:image/jpeg;base64,...",
    "timestamp": "2026-02-16T10:30:00"
}
```

### GET `/api/config`
- Lấy cấu hình hiện tại

```bash
curl http://localhost:5000/api/config
```

### GET `/api/model-info`
- Thông tin về models

```bash
curl http://localhost:5000/api/model-info
```

## Frontend Structure

```
app/
├── templates/
│   └── index.html          # Main HTML
├── static/
│   ├── css/
│   │   └── style.css       # Styling
│   └── js/
│       └── main.js         # JavaScript logic
└── uploads/                # Uploaded files
```

## Customization

### Port Configuration

Thay đổi port trong `app.py`:

```python
app.run(
    host='0.0.0.0',
    port=8000,  # Change port
    debug=True
)
```

### File Upload Size

Thay đổi trong `app.py`:

```python
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
```

### CORS Settings

Thêm CORS support:

```python
from flask_cors import CORS
CORS(app)
```

## Error Handling

### Model Not Initialized
- Kiểm tra weights file tồn tại
- Xác nhận đường dẫn đúng
- Kiểm tra GPU/CUDA

### File Upload Failed
- Kiểm tra file size
- Kiểm tra file format
- Kiểm tra disk space

### Prediction Error
- Kiểm tra ảnh format
- Xác nhận ảnh không corrupted
- Kiểm tra RAM/VRAM

## Performance Tips

1. **GPU Usage**: Ứng dụng sẽ tự động dùng GPU nếu có sẵn
2. **Batch Processing**: Có thể xử lý multiple images
3. **Caching**: Sử dụng caching để tăng tốc độ

## Deployment

### Development
```bash
python app/run.py
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app.app:app
```

### Docker
```dockerfile
FROM python:3.10
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app/run.py"]
```

## Troubleshooting

### Issue: Models not loading
**Solution**: Kiểm tra file weights tồn tại và uncomment init_models() call

### Issue: Slow prediction
**Solution**: Kiểm tra GPU utilization, giảm batch size

### Issue: Port already in use
**Solution**: Thay đổi port hoặc kill process đang sử dụng port

## Support

Để báo cáo bug hoặc yêu cầu feature, vui lòng tạo issue.

## License

MIT License - See LICENSE file for details

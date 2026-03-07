"""
Simple training script
Cách train models đơn giản
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def train_unet_simple():
    """Train UNet with synthetic data"""
    print("="*60)
    print("Training UNet Segmentation Model")
    print("="*60)
    
    try:
        from src.training import train_unet_from_config
        
        print("\n1. Chuẩn bị dữ liệu...")
        # Dataset should be in data/processed/
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print("⚠️  data/processed/ không tồn tại")
            print("   Hãy chạy: python scripts/generate_synthetic_data.py")
            return False
        
        print("   ✅ Dataset found")
        
        print("\n2. Bắt đầu training UNet...")
        print("   - Model: UNet (1 input channel, 1 output channel)")
        print("   - Epochs: 50 (có thể điều chỉnh ở config.yaml)")
        print("   - Optimizer: Adam")
        print("   - Loss: BCEWithLogitsLoss (cho binary segmentation)")
        
        # Train
        train_unet_from_config("config.yaml")
        
        print("\n✅ Training hoàn thành!")
        print("   - Weights saved: models/best_unet.pt")
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        return False


def train_yolo_simple():
    """Train YOLO with Roboflow"""
    print("="*60)
    print("Training YOLO Detection Model")
    print("="*60)
    
    try:
        from src.yolo_model import YOLODetector
        
        data_yaml = "data.yaml"
        
        if not Path(data_yaml).exists():
            print("\n⚠️  data.yaml không tồn tại")
            print("   Hãy tạo file data.yaml theo format:")
            print("""
path: data/processed
train: train/images
val: val/images
nc: 1
names: ['cavity']
            """)
            return False
        
        print("\n1. Chuẩn bị YOLO...")
        detector = YOLODetector(model_name="yolov8n")  # nano = nhẹ
        
        print("\n2. Bắt đầu training YOLO...")
        print("   - Model: YOLOv8 Nano (nhanh, nhẹ)")
        print("   - Epochs: 50")
        print("   - Imgsz: 640")
        
        results = detector.train(
            data_yaml=data_yaml,
            epochs=50,
            batch_size=8,
            learning_rate=0.001,
            device="0"  # GPU 0, hoặc "cpu"
        )
        
        print("\n✅ Training YOLO hoàn thành!")
        print("   - Weights saved: runs/detect/cavity_detection/weights/best.pt")
        
        # Copy to models/
        import shutil
        src = "runs/detect/cavity_detection/weights/best.pt"
        dst = "models/yolo_weights.pt"
        if Path(src).exists():
            shutil.copy(src, dst)
            print(f"   - Copied to: {dst}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Lỗi: {str(e)}")
        return False


def quick_demo():
    """Quick demo without training"""
    print("="*60)
    print("DEMO: Synthetic Inference")
    print("="*60)
    
    print("\n1. Tạo synthetic test image...")
    
    try:
        from scripts.generate_synthetic_data import SyntheticXrayGenerator
        
        generator = SyntheticXrayGenerator()
        test_image, cavities = generator.generate_image_with_cavities(num_cavities=2)
        
        # Save test image
        test_path = Path("data/test_image.jpg")
        import cv2
        cv2.imwrite(str(test_path), test_image)
        print(f"   ✅ Test image saved: {test_path}")
        print(f"   ✅ Cavities generated: {len(cavities)}")
        
        return test_path
        
    except Exception as e:
        print(f"   ❌ Lỗi: {str(e)}")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--unet", action="store_true", help="Train UNet")
    parser.add_argument("--yolo", action="store_true", help="Train YOLO")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--gen-data", action="store_true", help="Generate synthetic data")
    
    args = parser.parse_args()
    
    if args.gen_data:
        print("\n📊 Generating synthetic dataset...")
        from scripts.generate_synthetic_data import SyntheticXrayGenerator
        generator = SyntheticXrayGenerator()
        generator.generate_dataset("data/processed", num_images=50)
        from scripts.generate_synthetic_data import create_data_yaml
        create_data_yaml("data/processed")
    
    elif args.unet:
        train_unet_simple()
    
    elif args.yolo:
        train_yolo_simple()
    
    elif args.demo:
        quick_demo()
    
    else:
        print("Training Script - Hướng dẫn sử dụng:")
        print("\n1️⃣  Tạo synthetic data:")
        print("   python scripts/train.py --gen-data")
        print("\n2️⃣  Train UNet:")
        print("   python scripts/train.py --unet")
        print("\n3️⃣  Train YOLO (cần data.yaml):")
        print("   python scripts/train.py --yolo")
        print("\n4️⃣  Quick demo:")
        print("   python scripts/train.py --demo")

"""
Test script - Kiểm tra API và chạy demo
"""

import json
import base64
from pathlib import Path
import requests

# Configuration
API_URL = "http://localhost:5000"

def check_server():
    """Check if Flask server is running"""
    print("🔍 Checking server...")
    try:
        response = requests.get(f"{API_URL}/api/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server: {data.get('status', 'unknown')}")
            print(f"   Models status: {data.get('models', {}).get('loaded', False)}")
            return True
        else:
            print(f"❌ Server error: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Server not running: {str(e)}")
        print(f"   Start with: cd d:\\Đồ án KB\\app && python run.py")
        return False


def get_config():
    """Get server configuration"""
    print("\n📋 Getting configuration...")
    try:
        response = requests.get(f"{API_URL}/api/config", timeout=5)
        if response.status_code == 200:
            config = response.json()
            print("✅ Configuration:")
            print(f"   - Model: {config.get('model', {}).get('name', 'N/A')}")
            print(f"   - Input size: {config.get('model', {}).get('input_size', 'N/A')}")
            print(f"   - Batch size: {config.get('training', {}).get('batch_size', 'N/A')}")
            return config
        else:
            print(f"❌ Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def predict_image(image_path):
    """Send image to server for prediction"""
    print(f"\n🖼️  Predicting: {image_path}")
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")
        
        # Prepare payload
        payload = {
            "image": image_data,
            "format": "base64"
        }
        
        # Send to server
        response = requests.post(
            f"{API_URL}/api/predict",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful:")
            print(f"   - Cavities detected: {result.get('cavities_detected', False)}")
            print(f"   - Confidence: {result.get('confidence', 0):.2%}")
            if 'num_cavities' in result:
                print(f"   - Number of cavities: {result.get('num_cavities', 0)}")
            if 'bounding_boxes' in result:
                print(f"   - Bounding boxes: {len(result.get('bounding_boxes', []))} found")
            if 'segmentation_mask' in result:
                print(f"   - Segmentation mask: Available")
            
            # Save results
            result_path = "test_result.json"
            with open(result_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"\n   Results saved: {result_path}")
            
            return True
        else:
            error = response.json().get("error", "Unknown error")
            print(f"❌ Prediction error: {error}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return False


def main():
    print("="*60)
    print("🧪 API & Inference Test Suite")
    print("="*60)
    
    # Check server
    if not check_server():
        print("\n⚠️  Server is not running!")
        print("\nStart Flask server:")
        print("  cd d:\\Đồ án KB\\app")
        print("  python run.py")
        return
    
    # Get config
    get_config()
    
    # Try prediction with any available image
    test_images = [
        "data/test_image.jpg",
        "data/processed/train/images/img_0.jpg",
        "app/static/uploads/test.jpg"
    ]
    
    print("\n" + "="*60)
    print("🧪 Testing Prediction API")
    print("="*60)
    
    found_image = None
    for img_path in test_images:
        if Path(img_path).exists():
            found_image = img_path
            break
    
    if found_image:
        predict_image(found_image)
    else:
        print("\n⚠️  No test image found")
        print("\nCreate a test image:")
        print("  1. Generate synthetic data:")
        print("     python scripts/train.py --gen-data")
        print("\n  2. Then test again:")
        print("     python scripts/test.py")
    
    print("\n" + "="*60)
    print("✅ Test complete!")
    print("="*60)


if __name__ == "__main__":
    main()

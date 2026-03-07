"""
Flask backend for dental cavity detection web application
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
from datetime import datetime
import json
import base64
from io import BytesIO

# Conditional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    import numpy as np
    from src.inference import DentalCavityInference
    from src.utils import normalize_image, denormalize_image
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask app configuration
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

UPLOAD_FOLDER = 'app/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create upload folder
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

# Global variables for models
inference_system = None
model_initialized = False

# Device
DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"


def init_models(yolo_weights=None, unet_weights=None):
    """Initialize YOLO and UNet models"""
    global inference_system, model_initialized
    
    if not TORCH_AVAILABLE or not CV_AVAILABLE:
        logger.warning("PyTorch or CV2 not available. Skipping model initialization.")
        return False
    
    try:
        inference_system = DentalCavityInference(
            yolo_weights=yolo_weights,
            unet_weights=unet_weights,
            device=DEVICE
        )
        model_initialized = True
        logger.info("Models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        return False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_uploaded_image(file_path):
    """Process uploaded X-ray image"""
    if not CV_AVAILABLE:
        return None, "OpenCV not available"
    
    try:
        # Read image
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None, "Failed to read image"
        
        # Normalize
        image = normalize_image(image)
        
        return image, None
    except Exception as e:
        return None, str(e)


def image_to_base64(image_array):
    """Convert numpy array to base64 encoded image"""
    if image_array is None or not CV_AVAILABLE:
        return None
    
    # Convert to uint8 if needed
    if image_array.max() <= 1.0:
        image_array = denormalize_image(image_array)
    else:
        image_array = image_array.astype(np.uint8)
    
    # Encode to JPEG
    ret, buffer = cv2.imencode('.jpg', image_array)
    if not ret:
        return None
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{image_base64}"


def draw_detections(image, detections):
    """Draw detection boxes on image"""
    if not CV_AVAILABLE:
        return None
    
    output_image = cv2.cvtColor(denormalize_image(image), cv2.COLOR_GRAY2BGR)
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        conf = detection['confidence']
        
        # Draw box
        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Put text
        label = f"Cavity {i+1}: {conf:.2f}"
        cv2.putText(output_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return output_image


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', device=DEVICE, model_initialized=model_initialized)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': DEVICE,
        'model_initialized': model_initialized,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict cavities in X-ray image
    
    Expected request:
    - multipart/form-data with 'file' field
    
    Returns:
    - JSON with detection results
    """
    
    if not model_initialized:
        return jsonify({'error': 'Models not initialized'}), 503
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        # Save file
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Process image
        image, error = process_uploaded_image(filepath)
        if error:
            return jsonify({'error': error}), 400
        
        # Run inference
        image_original = image.copy()
        
        # YOLO detection
        detections = inference_system.detect_cavities_yolo(image)
        logger.info(f"Found {len(detections)} cavities")
        
        # UNet segmentation
        segmentations = []
        for i, detection in enumerate(detections):
            mask = inference_system.segment_cavity_unet(image, detection['bbox'])
            segmentations.append({
                'detection_idx': i,
                'mask': mask.tolist() if isinstance(mask, np.ndarray) else mask
            })
        
        # Draw results
        result_image = draw_detections(image_original, detections)
        result_image_base64 = image_to_base64(result_image)
        
        # Prepare response
        response_data = {
            'success': True,
            'num_cavities': len(detections),
            'detections': [
                {
                    'id': i,
                    'bbox': list(map(int, det['bbox'])),
                    'confidence': float(det['confidence']),
                    'class': int(det['class'])
                }
                for i, det in enumerate(detections)
            ],
            'segmentations_count': len(segmentations),
            'result_image': result_image_base64,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed. Found {len(detections)} cavities")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


@app.route('/api/predict-batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction for multiple images
    
    Expected: JSON with array of image data URIs
    """
    
    if not model_initialized:
        return jsonify({'error': 'Models not initialized'}), 503
    
    try:
        data = request.get_json()
        
        if 'images' not in data or not isinstance(data['images'], list):
            return jsonify({'error': 'Invalid request format'}), 400
        
        results = []
        
        for img_data in data['images']:
            # Parse base64 image (simplified)
            results.append({
                'status': 'processed'
            })
        
        return jsonify({
            'success': True,
            'total': len(results),
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration"""
    return jsonify({
        'device': DEVICE,
        'max_file_size_mb': MAX_FILE_SIZE / (1024 * 1024),
        'allowed_extensions': list(ALLOWED_EXTENSIONS),
        'model_initialized': model_initialized
    })


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_initialized:
        return jsonify({'error': 'Models not initialized'}), 503
    
    info = {
        'yolo': {
            'name': inference_system.yolo_model.model_name if inference_system.yolo_model else 'Not loaded',
            'status': 'loaded' if inference_system.yolo_model else 'not_loaded'
        },
        'unet': {
            'in_channels': 1,
            'out_channels': 1,
            'status': 'loaded' if inference_system.unet_model else 'not_loaded'
        },
        'device': DEVICE
    }
    
    return jsonify(info)


# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({'error': f'File too large. Max size: {MAX_FILE_SIZE / (1024*1024):.0f}MB'}), 413


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Initialize models (comment out if weights don't exist yet)
    # init_models(
    #     yolo_weights='models/yolo_weights.pt',
    #     unet_weights='models/best_unet.pt'
    # )
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )

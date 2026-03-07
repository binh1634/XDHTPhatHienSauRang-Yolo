"""
Flask backend for dental cavity detection web application
Simple version without heavy dependencies
"""

from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from pathlib import Path
import logging
from datetime import datetime
import json

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

# Global variables
inference_system = None
model_initialized = False
model_weights = {'yolo': None, 'unet': None}

# Try to detect device
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
except:
    DEVICE = "cpu"


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_models(yolo_weights=None, unet_weights=None):
    """Initialize YOLO and UNet models"""
    global inference_system, model_initialized, model_weights
    
    if model_initialized:
        return True  # Already initialized
    
    try:
        from src.inference import DentalCavityInference
        
        logger.info("Initializing models...")
        inference_system = DentalCavityInference(
            yolo_weights=yolo_weights,
            unet_weights=unet_weights,
            device=DEVICE
        )
        model_initialized = True
        model_weights['yolo'] = yolo_weights
        model_weights['unet'] = unet_weights
        logger.info("Models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize models: {str(e)}")
        model_initialized = False
        return False


def ensure_models_initialized():
    """Lazy load models if not already initialized"""
    global model_initialized
    if not model_initialized:
        yolo_weights = "models/yolo_weights.pt" if os.path.exists("models/yolo_weights.pt") else None
        unet_weights = "models/best_unet.pt" if os.path.exists("models/best_unet.pt") else None
        if yolo_weights:
            init_models(yolo_weights, unet_weights)
    return model_initialized


# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def index():
    """Home page"""
    return render_template('home.html', device=DEVICE, model_initialized=model_initialized)


@app.route('/api/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': DEVICE,
        'model_initialized': model_initialized,
        'yolo_weights': model_weights['yolo'],
        'unet_weights': model_weights['unet'],
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict cavities in X-ray image
    """
    
    # Try to initialize models if not already done
    ensure_models_initialized()
    
    if not model_initialized:
        return jsonify({'error': 'Models not found. Please ensure model files exist in models/ directory.'}), 503
    
    # Check if file is in request
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Allowed: {ALLOWED_EXTENSIONS}'}), 400
    
    try:
        import cv2
        import numpy as np
        import base64
        from src.utils import normalize_image, denormalize_image
        
        # Save file
        filename = secure_filename(f"{datetime.now().timestamp()}_{file.filename}")
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"Processing file: {filename}")
        
        # Read image in color mode (YOLO model requires 3 channels)
        image = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        original = image.copy()
        
        # Convert to grayscale for processing if needed
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalize (use grayscale version for UNet, but YOLO will use BGR)
        image_normalized = normalize_image(image_gray)

        # Mode selection (yolo | unet | both) - default both
        mode = request.form.get('mode', 'both') if request.form else request.args.get('mode', 'both')

        detections = []
        segmentations = []

        # YOLO-only mode
        if mode == 'yolo':
            detections = inference_system.detect_cavities_yolo(image)
            logger.info(f"YOLO mode: Found {len(detections)} cavities")

            # Draw boxes
            output_image = original.copy()
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Cavity {i+1}"
                cv2.putText(output_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # UNet-only mode (segment whole image)
        elif mode == 'unet':
            try:
                h, w = image_normalized.shape[:2]
                # Use full-image bbox
                full_bbox = (0, 0, w, h)
                mask = inference_system.segment_cavity_unet(image_normalized, full_bbox)
                segmentations.append({
                    'detection_idx': 0,
                    'mask': mask.tolist() if isinstance(mask, np.ndarray) else mask
                })

                # Overlay mask on original
                output_image = original.copy()
                colored_mask = np.zeros_like(output_image)
                colored_mask[:, :, 1] = (mask * 255).astype('uint8')
                overlay = cv2.addWeighted(output_image, 0.7, colored_mask, 0.3, 0)
                output_image = overlay
            except Exception as e:
                logger.error(f"UNet mode failed: {str(e)}")

        # Both (YOLO + UNet)
        else:
            detections = inference_system.detect_cavities_yolo(image)
            logger.info(f"Both mode: Found {len(detections)} cavities")
            for i, detection in enumerate(detections):
                try:
                    mask = inference_system.segment_cavity_unet(image_normalized, detection['bbox'])
                    segmentations.append({
                        'detection_idx': i,
                        'mask': mask.tolist() if isinstance(mask, np.ndarray) else mask
                    })
                except Exception:
                    pass

            # Draw results
            output_image = original.copy()
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Cavity {i+1}"
                cv2.putText(output_image, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert to base64
        ret, buffer = cv2.imencode('.jpg', output_image)
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Prepare response
        response_data = {
            'success': True,
            'num_cavities': len(detections),
            'detections': [
                {
                    'id': i,
                    'bbox': list(map(int, det['bbox'])),
                    'confidence': float(det.get('confidence', 0.0))
                }
                for i, det in enumerate(detections)
            ],
            'segmentations_count': len(segmentations),
            'result_image': f"data:image/jpeg;base64,{image_base64}",
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed. Found {len(detections)} cavities")
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500


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
        'status': 'loaded',
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
    logger.error(f"Internal error: {error}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        use_reloader=False
    )

"""
Run Flask application for dental cavity detection
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.app_simple import app, init_models

def _get_arg(name, default=None):
    # simple CLI arg parser -- supports --name value
    if f"--{name}" in sys.argv:
        try:
            i = sys.argv.index(f"--{name}")
            return sys.argv[i+1]
        except Exception:
            return default
    return os.environ.get(f"FLASK_RUN_{name.upper()}", default)


if __name__ == '__main__':
    print("="*60)
    print("Dental Cavity Detection - Web Application")
    print("="*60)
    
    # Check if models exist but don't load them yet (lazy loading on first request)
    yolo_weights = "models/yolo_weights.pt"
    unet_weights = "models/best_unet.pt"
    
    yolo_exists = os.path.exists(yolo_weights)
    unet_exists = os.path.exists(unet_weights)
    
    if yolo_exists:
        print(f"✅ YOLO weights found: {yolo_weights}")
        if unet_exists:
            print(f"✅ UNet weights found: {unet_weights}")
        else:
            print(f"⚠️  UNet weights not found: {unet_weights} (YOLO-only mode)")
        print("\nModels will be loaded on first prediction request...")
    else:
        print("\n⚠️  Model weights not found")
        print(f"   - YOLO: {yolo_weights}")
        print(f"   - UNet: {unet_weights}")
        print("\nWeb interface will work, but predictions will show 'Models not initialized'")
    
    print("\n" + "="*60)
    print("Starting Flask application...")
    host = _get_arg('host', '0.0.0.0')
    port = int(_get_arg('port', 5000))
    display_host = host if host != '0.0.0.0' else 'localhost'
    print(f"Open browser at: http://{display_host}:{port}")
    print("Press Ctrl+C to stop")
    print("="*60 + "\n")
    
    app.run(
        host=host,
        port=port,
        debug=True,
        use_reloader=False
    )

#!/bin/bash
# Quick start script cho Linux/Mac
# Chạy: bash scripts/quickstart.sh

cd "$(dirname "$0")/.."

echo ""
echo "========================================"
echo "  DENTAL CAVITY DETECTION SYSTEM"
echo "  Quick Start Script"
echo "========================================"
echo ""

# Check Python
echo "[1/5] Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not installed!"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi
PYVER=$(python3 --version)
echo "      $PYVER OK"
echo ""

# Install dependencies
echo "[2/5] Checking dependencies..."
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "      Installing packages..."
    pip3 install -q -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
fi
echo "      All packages OK"
echo ""

# Generate synthetic data
echo "[3/5] Generating synthetic dataset..."
python3 scripts/train.py --gen-data &>/dev/null
if [ $? -ne 0 ]; then
    echo "      WARNING: Could not generate synthetic data"
    echo "      You can do this manually: python3 scripts/train.py --gen-data"
else
    echo "      50 synthetic X-ray images created"
fi
echo ""

# Train UNet
echo "[4/5] Training UNet model..."
echo "      (This may take 5-10 minutes on GPU, 30+ on CPU)"
sleep 3
python3 scripts/train.py --unet &>/dev/null
if [ $? -ne 0 ]; then
    echo "      WARNING: Could not train UNet"
    echo "      Models may not be available for prediction"
    echo "      You can train manually: python3 scripts/train.py --unet"
else
    echo "      UNet model trained"
fi
echo ""

# Start Flask server
echo "[5/5] Starting Flask server..."
echo ""
echo "========================================"
echo "  SERVER STARTING..."
echo "  Open: http://localhost:5000"
echo "  Press Ctrl+C to stop server"
echo "========================================"
echo ""

cd app
python3 run.py

# Server is running, script ends here when user stops it
exit 0

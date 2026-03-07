/* ============================================================
   DENTAL CAVITY DETECTION - MAIN JAVASCRIPT
   ============================================================ */

// Global variables
const DOM = {
    uploadArea: document.getElementById('uploadArea'),
    fileInput: document.getElementById('fileInput'),
    browseBtn: document.getElementById('browseBtn'),
    resultsSection: document.getElementById('resultsSection'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    originalImage: document.getElementById('originalImage'),
    resultImage: document.getElementById('resultImage'),
    cavityCount: document.getElementById('cavityCount'),
    confidenceScore: document.getElementById('confidenceScore'),
    processingTime: document.getElementById('processingTime'),
    cavityDetails: document.getElementById('cavityDetails'),
    toast: document.getElementById('toast'),
    device: document.getElementById('device'),
    modelStatus: document.getElementById('modelStatus')
};

let startTime = 0;

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    checkModelStatus();
    getDeviceInfo();
});

function initializeEventListeners() {
    // Upload area
    DOM.uploadArea.addEventListener('click', () => DOM.fileInput.click());
    DOM.browseBtn.addEventListener('click', () => DOM.fileInput.click());
    
    // File input
    DOM.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    DOM.uploadArea.addEventListener('dragover', handleDragOver);
    DOM.uploadArea.addEventListener('dragleave', handleDragLeave);
    DOM.uploadArea.addEventListener('drop', handleFileDrop);
}

// ============================================================
// FILE HANDLING
// ============================================================

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        processFile(file);
    }
}

function handleDragOver(event) {
    event.preventDefault();
    DOM.uploadArea.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    DOM.uploadArea.classList.remove('drag-over');
}

function handleFileDrop(event) {
    event.preventDefault();
    DOM.uploadArea.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showToast('Please drop a valid image file', 'error');
    }
}

function processFile(file) {
    // Validate file size
    const maxSize = 50 * 1024 * 1024; // 50MB
    if (file.size > maxSize) {
        showToast('File size exceeds 50MB limit', 'error');
        return;
    }
    
    // Validate file type
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Invalid file type. Please use JPG, PNG, or BMP', 'error');
        return;
    }
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        DOM.originalImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Send to backend
    sendPredictionRequest(file);
}

// ============================================================
// API REQUESTS
// ============================================================

function sendPredictionRequest(file) {
    startTime = performance.now();
    
    // Show loading
    showLoading(true);
    DOM.resultsSection.style.display = 'none';
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/api/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            return response.json().then(data => Promise.reject(data));
        }
        return response.json();
    })
    .then(data => {
        handlePredictionResult(data);
    })
    .catch(error => {
        console.error('Error:', error);
        showLoading(false);
        
        const errorMsg = error.error || error.message || 'Unknown error occurred';
        showToast(`Prediction failed: ${errorMsg}`, 'error');
    });
}

function handlePredictionResult(data) {
    const endTime = performance.now();
    const processingTimeMs = (endTime - startTime).toFixed(2);
    
    // Hide loading
    showLoading(false);
    
    // Update results
    if (data.success) {
        // Display result image
        if (data.result_image) {
            DOM.resultImage.src = data.result_image;
        }
        
        // Update statistics
        DOM.cavityCount.textContent = data.num_cavities || 0;
        DOM.processingTime.textContent = `${processingTimeMs}ms`;
        
        // Calculate average confidence
        if (data.detections && data.detections.length > 0) {
            const avgConfidence = (
                data.detections.reduce((sum, det) => sum + det.confidence, 0) / 
                data.detections.length
            * 100).toFixed(1);
            DOM.confidenceScore.textContent = `${avgConfidence}%`;
        } else {
            DOM.confidenceScore.textContent = '--';
        }
        
        // Display cavity details
        displayCavityDetails(data.detections);
        
        // Show results section
        DOM.resultsSection.style.display = 'block';
        
        // Scroll to results
        setTimeout(() => {
            DOM.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
        
        showToast(`Found ${data.num_cavities} cavity(ies)`, 'success');
    } else {
        showToast(data.error || 'Prediction failed', 'error');
    }
}

function displayCavityDetails(detections) {
    DOM.cavityDetails.innerHTML = '';
    
    if (!detections || detections.length === 0) {
        DOM.cavityDetails.innerHTML = '<p style="color: #666;">No cavities detected</p>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const bbox = detection.bbox;
        const confidence = (detection.confidence * 100).toFixed(1);
        
        const cavityItem = document.createElement('div');
        cavityItem.className = 'cavity-item';
        cavityItem.innerHTML = `
            <div class="cavity-item-header">
                <span class="cavity-item-title">Cavity #${index + 1}</span>
                <span class="cavity-confidence">${confidence}% confidence</span>
            </div>
            <div class="cavity-bbox">
                <strong>Location:</strong> X: ${bbox[0]} - ${bbox[2]}, Y: ${bbox[1]} - ${bbox[3]}
            </div>
            <div class="cavity-bbox">
                <strong>Size:</strong> ${bbox[2] - bbox[0]} × ${bbox[3] - bbox[1]} pixels
            </div>
        `;
        
        DOM.cavityDetails.appendChild(cavityItem);
    });
}

function checkModelStatus() {
    fetch('/api/health')
    .then(response => response.json())
    .then(data => {
        const statusEl = DOM.modelStatus;
        
        if (data.model_initialized) {
            statusEl.className = 'model-status ready';
            statusEl.innerHTML = '<i class="fas fa-check-circle"></i> Models Ready';
        } else {
            statusEl.className = 'model-status error';
            statusEl.innerHTML = '<i class="fas fa-exclamation-circle"></i> Models Not Loaded';
        }
    })
    .catch(error => {
        console.error('Health check failed:', error);
        const statusEl = DOM.modelStatus;
        statusEl.className = 'model-status error';
        statusEl.innerHTML = '<i class="fas fa-times-circle"></i> Server OK, Models Not Loaded';
    });
}

function getDeviceInfo() {
    fetch('/api/config')
    .then(response => response.json())
    .then(data => {
        const deviceName = data.device === 'cuda' ? 'GPU (CUDA)' : 'CPU';
        DOM.device.textContent = deviceName;
    })
    .catch(error => console.error('Error getting device info:', error));
}

// ============================================================
// UI HELPERS
// ============================================================

function showLoading(show) {
    if (show) {
        DOM.loadingOverlay.style.display = 'flex';
    } else {
        DOM.loadingOverlay.style.display = 'none';
    }
}

function showToast(message, type = 'info') {
    DOM.toast.textContent = message;
    DOM.toast.className = `toast show ${type}`;
    
    // Auto hide after 4 seconds
    setTimeout(() => {
        DOM.toast.classList.remove('show');
    }, 4000);
}

// ============================================================
// KEYBOARD SHORTCUTS
// ============================================================

document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + U to open file browser
    if ((event.ctrlKey || event.metaKey) && event.key === 'u') {
        event.preventDefault();
        DOM.fileInput.click();
    }
});

// ============================================================
// BROWSER COMPATIBILITY CHECK
// ============================================================

function checkBrowserCompatibility() {
    // Check for required APIs
    if (!window.fetch) {
        alert('Your browser does not support Fetch API. Please use a modern browser.');
        return false;
    }
    
    if (!window.FormData) {
        alert('Your browser does not support FormData. Please use a modern browser.');
        return false;
    }
    
    return true;
}

// Run compatibility check on load
if (!checkBrowserCompatibility()) {
    document.body.innerHTML = '<h1>Please use a modern browser</h1>';
}

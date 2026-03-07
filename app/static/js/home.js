/* ============================================================
   DENTALCARE PRO - HOME PAGE JAVASCRIPT
   ============================================================ */

const DOM = {
    uploadAreaMain: document.getElementById('uploadAreaMain'),
    fileInputMain: document.getElementById('fileInputMain'),
    browseBtnMain: document.getElementById('browseBtnMain'),
    resultsModal: document.getElementById('resultsModal'),
    closeModal: document.getElementById('closeModal'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    modalOriginalImage: document.getElementById('modalOriginalImage'),
    modalResultImage: document.getElementById('modalResultImage'),
    modalCavityCount: document.getElementById('modalCavityCount'),
    modalAvgConfidence: document.getElementById('modalAvgConfidence'),
    modalProcessingTime: document.getElementById('modalProcessingTime'),
    cavityDetailsList: document.getElementById('cavityDetailsList'),
    patientName: document.getElementById('patientName'),
    examDate: document.getElementById('examDate'),
    printReportBtn: document.getElementById('printReportBtn'),
    downloadReportBtn: document.getElementById('downloadReportBtn'),
    resultsTableBody: document.getElementById('resultsTableBody'),
    toast: document.getElementById('toast')
};

let currentResultData = null;
let analysisHistory = [];

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    setExamDate();
});

function initializeEventListeners() {
    // Upload
    DOM.uploadAreaMain.addEventListener('click', () => DOM.fileInputMain.click());
    DOM.browseBtnMain.addEventListener('click', () => DOM.fileInputMain.click());
    DOM.fileInputMain.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    DOM.uploadAreaMain.addEventListener('dragover', handleDragOver);
    DOM.uploadAreaMain.addEventListener('dragleave', handleDragLeave);
    DOM.uploadAreaMain.addEventListener('drop', handleFileDrop);
    
    // Modal
    DOM.closeModal.addEventListener('click', closeModal);
    window.addEventListener('click', function(event) {
        if (event.target === DOM.resultsModal) {
            closeModal();
        }
    });
    
    // Reports
    DOM.printReportBtn.addEventListener('click', printReport);
    DOM.downloadReportBtn.addEventListener('click', downloadReport);
    
    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function(e) {
            document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function setExamDate() {
    const today = new Date().toISOString().split('T')[0];
    DOM.examDate.value = today;
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
    DOM.uploadAreaMain.classList.add('drag-over');
}

function handleDragLeave(event) {
    event.preventDefault();
    DOM.uploadAreaMain.classList.remove('drag-over');
}

function handleFileDrop(event) {
    event.preventDefault();
    DOM.uploadAreaMain.classList.remove('drag-over');
    
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        processFile(file);
    } else {
        showToast('Vui lòng kéo thả một ảnh', 'error');
    }
}

function processFile(file) {
    // Validate
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
        showToast('Ảnh quá lớn (max 50MB)', 'error');
        return;
    }
    
    const allowedTypes = ['image/jpeg', 'image/png', 'image/bmp'];
    if (!allowedTypes.includes(file.type)) {
        showToast('Định dạng không hỗ trợ. Vui lòng dùng JPG, PNG, hoặc BMP', 'error');
        return;
    }
    
    // Display preview
    const reader = new FileReader();
    reader.onload = (e) => {
        DOM.modalOriginalImage.src = e.target.result;
    };
    reader.readAsDataURL(file);
    
    // Send prediction
    sendPredictionRequest(file);
}

// ============================================================
// API REQUESTS
// ============================================================

function sendPredictionRequest(file) {
    showLoading(true);
    
    const formData = new FormData();
    formData.append('file', file);
    // Read selected model mode from radio buttons (default 'both')
    try {
        const modeEl = document.querySelector('input[name="modelMode"]:checked');
        const mode = modeEl ? modeEl.value : 'both';
        formData.append('mode', mode);
    } catch (e) {
        formData.append('mode', 'both');
    }
    
    const startTime = performance.now();
    
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
        const endTime = performance.now();
        data.processingTime = (endTime - startTime).toFixed(0);
        handlePredictionResult(data);
    })
    .catch(error => {
        showLoading(false);
        const errorMsg = error.error || error.message || 'Lỗi không xác định';
        showToast(`Phân tích thất bại: ${errorMsg}`, 'error');
    });
}

function handlePredictionResult(data) {
    showLoading(false);
    
    if (data.success) {
        currentResultData = data;
        
        // Update modal
        DOM.modalResultImage.src = data.result_image;
        DOM.modalCavityCount.textContent = data.num_cavities || 0;
        
        // Calculate average confidence
        if (data.detections && data.detections.length > 0) {
            const avgConfidence = (
                data.detections.reduce((sum, det) => sum + det.confidence, 0) / 
                data.detections.length * 100
            ).toFixed(1);
            DOM.modalAvgConfidence.textContent = `${avgConfidence}%`;
        } else {
            DOM.modalAvgConfidence.textContent = '--';
        }
        
        DOM.modalProcessingTime.textContent = `${data.processingTime}ms`;
        
        // Display cavity details
        displayCavityDetails(data.detections);
        
        // Add to history
        addToHistory(data);
        
        // Show modal
        DOM.resultsModal.classList.add('show');
        
        showToast(`Phát hiện ${data.num_cavities} sâu`, 'success');
    } else {
        showToast(data.error || 'Phân tích thất bại', 'error');
    }
}

function displayCavityDetails(detections) {
    DOM.cavityDetailsList.innerHTML = '';
    
    if (!detections || detections.length === 0) {
        DOM.cavityDetailsList.innerHTML = '<p style="color: #666; text-align: center;">Không phát hiện sâu</p>';
        return;
    }
    
    detections.forEach((detection, index) => {
        const item = document.createElement('div');
        item.className = 'cavity-item';
        
        const confidence = (detection.confidence * 100).toFixed(1);
        const bbox = detection.bbox;
        
        item.innerHTML = `
            <div class="cavity-item-info">
                <strong>Sâu #${index + 1}</strong>
                <span>Vị trí: X [${bbox[0]} - ${bbox[2]}], Y [${bbox[1]} - ${bbox[3]}]</span>
            </div>
            <div class="cavity-confidence">${confidence}%</div>
        `;
        
        DOM.cavityDetailsList.appendChild(item);
    });
}

function addToHistory(data) {
    const record = {
        date: new Date().toLocaleString('vi-VN'),
        patient: DOM.patientName.value || 'N/A',
        cavities: data.num_cavities,
        confidence: data.detections.length > 0 
            ? (data.detections.reduce((s, d) => s + d.confidence, 0) / data.detections.length * 100).toFixed(1)
            : 'N/A'
    };
    
    analysisHistory.unshift(record);
    
    // Update table
    if (analysisHistory.length > 0) {
        DOM.resultsTableBody.innerHTML = analysisHistory.slice(0, 5).map((r, i) => `
            <tr>
                <td>${r.date}</td>
                <td>${r.patient}</td>
                <td>${r.cavities}</td>
                <td>${r.confidence}%</td>
                <td>
                    <button class="btn btn-sm" onclick="alert('Chi tiết kết quả')">Xem</button>
                </td>
            </tr>
        `).join('');
    }
}

// ============================================================
// MODAL FUNCTIONS
// ============================================================

function closeModal() {
    DOM.resultsModal.classList.remove('show');
}

function printReport() {
    if (!currentResultData) return;
    
    const printContent = `
        <!DOCTYPE html>
        <html lang="vi">
        <head>
            <meta charset="UTF-8">
            <title>Báo Cáo Phân Tích X-quang</title>
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; }
                h1 { text-align: center; color: #0066cc; }
                .section { margin: 20px 0; page-break-inside: avoid; }
                .info { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
                .info-item { border: 1px solid #ddd; padding: 10px; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                table, th, td { border: 1px solid #ddd; padding: 10px; text-align: left; }
                th { background-color: #f0f0f0; }
                img { max-width: 100%; height: auto; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>BÁNG CÁO PHÂN TÍCH X-QUANG</h1>
            
            <div class="section">
                <h2>Thông Tin Bệnh Nhân</h2>
                <div class="info">
                    <div class="info-item"><strong>Tên:</strong> ${DOM.patientName.value || 'N/A'}</div>
                    <div class="info-item"><strong>Ngày Khám:</strong> ${DOM.examDate.value}</div>
                </div>
            </div>
            
            <div class="section">
                <h2>Kết Quả Phân Tích</h2>
                <table>
                    <tr>
                        <th>Tiêu Chí</th>
                        <th>Giá Trị</th>
                    </tr>
                    <tr>
                        <td>Số Sâu Phát Hiện</td>
                        <td>${currentResultData.num_cavities}</td>
                    </tr>
                    <tr>
                        <td>Độ Tin Cậy Trung Bình</td>
                        <td>${DOM.modalAvgConfidence.textContent}</td>
                    </tr>
                    <tr>
                        <td>Thời Gian Xử Lý</td>
                        <td>${DOM.modalProcessingTime.textContent}</td>
                    </tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Hình Ảnh</h2>
                <p><strong>Ảnh Gốc:</strong></p>
                <img src="${DOM.modalOriginalImage.src}" alt="Original">
                <p><strong>Ảnh Kết Quả:</strong></p>
                <img src="${DOM.modalResultImage.src}" alt="Result">
            </div>
            
            <div class="section">
                <h2>Khuyến Cáo</h2>
                <ul>
                    <li>Thực hiện kiểm tra và điều trị sâu răng sớm</li>
                    <li>Vệ sinh răng miệng sạch sẽ hàng ngày</li>
                    <li>Hạn chế thực phẩm chứa đường</li>
                    <li>Tái khám định kỳ 3-6 tháng</li>
                </ul>
            </div>
            
            <p style="margin-top: 40px; text-align: center; color: #999; font-size: 0.9rem;">
                Báng cáo được tạo bởi DentalCare Pro - ${new Date().toLocaleString('vi-VN')}
            </p>
        </body>
        </html>
    `;
    
    const printWindow = window.open('', '', 'width=800,height=600');
    printWindow.document.write(printContent);
    printWindow.document.close();
    printWindow.print();
}

function downloadReport() {
    if (!currentResultData) return;
    
    const reportData = {
        patient: DOM.patientName.value || 'N/A',
        examDate: DOM.examDate.value,
        cavitiesDetected: currentResultData.num_cavities,
        averageConfidence: DOM.modalAvgConfidence.textContent,
        processingTime: DOM.modalProcessingTime.textContent,
        detections: currentResultData.detections,
        timestamp: new Date().toISOString()
    };
    
    const dataStr = JSON.stringify(reportData, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `dental-report-${Date.now()}.json`;
    link.click();
    
    showToast('Báng cáo đã tải xuống', 'success');
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
    
    setTimeout(() => {
        DOM.toast.classList.remove('show');
    }, 4000);
}

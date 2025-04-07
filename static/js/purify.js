/**
 * JavaScript for the Purification page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Dropzone
    const myDropzone = setupDropzone('uploadForm');
    
    // Purify button click handler
    const purifyBtn = document.getElementById('purify-btn');
    if (purifyBtn) {
        purifyBtn.addEventListener('click', function() {
            handlePurification();
        });
    }
    
    // New purify button click handler (after results)
    const newPurifyBtn = document.getElementById('new-purify-btn');
    if (newPurifyBtn) {
        newPurifyBtn.addEventListener('click', function() {
            resetUI();
        });
    }
    
    // Download purified image click handler
    const downloadPurifiedBtn = document.getElementById('download-purified');
    if (downloadPurifiedBtn) {
        downloadPurifiedBtn.addEventListener('click', function(e) {
            if (!currentSession.resultId) {
                e.preventDefault();
                showNotification('error', 'Error', 'No purified image available for download.');
            }
        });
    }
    
    // Check for transferred image from other pages
    const lastSession = getLastSessionInfo();
    if (lastSession && lastSession.sessionId) {
        currentSession.sessionId = lastSession.sessionId;
        currentSession.filename = lastSession.filename;
        
        // Enable purify button
        if (purifyBtn) {
            purifyBtn.disabled = false;
        }
        
        // Show the image as uploaded
        if (myDropzone) {
            const mockFile = { name: lastSession.filename, size: 12345 };
            myDropzone.emit("addedfile", mockFile);
            myDropzone.emit("thumbnail", mockFile, `/get_image/${lastSession.sessionId}/${lastSession.filename}`);
            myDropzone.emit("complete", mockFile);
            myDropzone.files.push(mockFile);
        }
        
        // Clear the stored session to prevent infinite loop
        localStorage.removeItem('lastSession');
    }
});

/**
 * Handle the purification process
 */
function handlePurification() {
    // Check if we have a session
    if (!currentSession.sessionId) {
        showNotification('error', 'Error', 'No image uploaded. Please upload an image first.');
        return;
    }
    
    // Show processing section
    switchSection('processing-section');
    
    // Send purification request
    fetch('/purify_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: currentSession.sessionId
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store result ID
            currentSession.resultId = data.result_id;
            
            // Display results
            displayPurificationResults(data);
            
            // Switch to results section
            switchSection('results-section');
        } else {
            showNotification('error', 'Purification Failed', data.message);
            switchSection('upload-section');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('error', 'Error', 'An error occurred during purification.');
        switchSection('upload-section');
    });
}

/**
 * Display purification results in the UI
 * @param {Object} data - The purification results data
 */
function displayPurificationResults(data) {
    // Display comparison image
    const comparisonImage = document.getElementById('comparison-image');
    if (comparisonImage) {
        comparisonImage.src = data.comparison_url;
        comparisonImage.alt = 'Comparison of original and purified images';
    }
    
    // Update metrics
    const psnrValue = document.getElementById('psnr-value');
    if (psnrValue) {
        psnrValue.textContent = `${formatNumber(data.psnr, 2)} dB`;
    }
    
    const ssimValue = document.getElementById('ssim-value');
    if (ssimValue) {
        ssimValue.textContent = formatNumber(data.ssim, 4);
    }
    
    // Update download links
    const downloadPurified = document.getElementById('download-purified');
    if (downloadPurified) {
        downloadPurified.href = data.download_url;
    }
    
    const generateReportBtn = document.getElementById('generate-report-btn');
    if (generateReportBtn) {
        generateReportBtn.href = data.report_url;
    }
    
    // Indicate quality with color
    if (data.psnr > 30) {
        psnrValue.className = 'fw-bold text-success';
    } else if (data.psnr > 20) {
        psnrValue.className = 'fw-bold text-warning';
    } else {
        psnrValue.className = 'fw-bold text-danger';
    }
    
    if (data.ssim > 0.9) {
        ssimValue.className = 'fw-bold text-success';
    } else if (data.ssim > 0.7) {
        ssimValue.className = 'fw-bold text-warning';
    } else {
        ssimValue.className = 'fw-bold text-danger';
    }
}
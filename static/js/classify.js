/**
 * JavaScript for the Classification page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Dropzone
    const myDropzone = setupDropzone('uploadForm');
    
    // Classification button click handler
    const classifyBtn = document.getElementById('classify-btn');
    if (classifyBtn) {
        classifyBtn.addEventListener('click', function() {
            handleClassification();
        });
    }
    
    // Check for transferred image from other pages
    const lastSession = getLastSessionInfo();
    if (lastSession && lastSession.sessionId) {
        currentSession.sessionId = lastSession.sessionId;
        currentSession.filename = lastSession.filename;
        
        // Enable classify button
        if (classifyBtn) {
            classifyBtn.disabled = false;
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
    
    // Purify button click handler (appears after classification)
    const purifyBtn = document.getElementById('purify-btn');
    if (purifyBtn) {
        purifyBtn.addEventListener('click', function() {
            transferImage('/purify', currentSession.sessionId, currentSession.resultId);
        });
    }
});

/**
 * Handle the classification process
 */
function handleClassification() {
    // Check if we have a session
    if (!currentSession.sessionId) {
        showNotification('error', 'Error', 'No image uploaded. Please upload an image first.');
        return;
    }
    
    // Show processing section
    switchSection('processing-section');
    
    // Send classification request
    fetch('/classify_image', {
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
            displayClassificationResults(data);
            
            // Switch to results section
            switchSection('results-section');
        } else {
            showNotification('error', 'Classification Failed', data.message);
            switchSection('upload-section');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('error', 'Error', 'An error occurred during classification.');
        switchSection('upload-section');
    });
}

/**
 * Display classification results in the UI
 * @param {Object} data - The classification results data
 */
function displayClassificationResults(data) {
    // Display uploaded image
    const uploadedImage = document.getElementById('uploaded-image');
    if (uploadedImage) {
        uploadedImage.src = `/get_image/${currentSession.sessionId}/${currentSession.filename}`;
        uploadedImage.alt = `Uploaded image: ${currentSession.filename}`;
    }
    
    // Update prediction text
    const predictionText = document.getElementById('prediction-text');
    if (predictionText) {
        predictionText.textContent = data.predicted_class;
    }
    
    // Update prediction description
    const predictionDescription = document.getElementById('prediction-description');
    if (predictionDescription) {
        const descriptions = {
            'Clean': 'This image appears to be a clean MRI scan without signs of adversarial perturbation.',
            'FGSM': 'This image shows signs of a Fast Gradient Sign Method (FGSM) attack, which adds perturbations in the direction of the gradient.',
            'BIM': 'This image shows signs of a Basic Iterative Method (BIM) attack, which is an iterative version of FGSM with smaller steps.',
            'PGD': 'This image shows signs of a Projected Gradient Descent (PGD) attack, which is a strong iterative attack with random initialization.'
        };
        
        predictionDescription.textContent = descriptions[data.predicted_class] || 'Unknown classification result.';
    }
    
    // Style the result alert based on prediction
    const resultAlert = document.getElementById('result-alert');
    if (resultAlert) {
        // Remove any existing classes
        resultAlert.className = 'alert mb-3';
        
        // Add appropriate class based on prediction
        switch(data.predicted_class.toLowerCase()) {
            case 'clean':
                resultAlert.classList.add('clean-result');
                break;
            case 'fgsm':
                resultAlert.classList.add('fgsm-result');
                break;
            case 'bim':
                resultAlert.classList.add('bim-result');
                break;
            case 'pgd':
                resultAlert.classList.add('pgd-result');
                break;
        }
    }
    
    // Update probability bars
    updateProbabilityBars(data.probabilities, data.class_names);
    
    // Update report link
    const reportBtn = document.getElementById('generate-report-btn');
    if (reportBtn) {
        reportBtn.href = data.report_url;
    }
    
    // Show purify button if not clean image
    const purifyBtn = document.getElementById('purify-btn');
    if (purifyBtn && data.predicted_class.toLowerCase() !== 'clean') {
        purifyBtn.classList.remove('d-none');
    } else if (purifyBtn) {
        purifyBtn.classList.add('d-none');
    }
}

/**
 * Update the probability bars with classification results
 * @param {Array} probabilities - Array of probability values
 * @param {Array} classNames - Array of class names
 */
function updateProbabilityBars(probabilities, classNames) {
    // Map classes to their respective elements
    const classElements = {
        'Clean': { prob: 'clean-prob', bar: 'clean-bar' },
        'FGSM': { prob: 'fgsm-prob', bar: 'fgsm-bar' },
        'BIM': { prob: 'bim-prob', bar: 'bim-bar' },
        'PGD': { prob: 'pgd-prob', bar: 'pgd-bar' }
    };
    
    // Update each probability bar
    for (let i = 0; i < probabilities.length; i++) {
        const className = classNames[i];
        const probability = probabilities[i];
        const elements = classElements[className];
        
        if (elements) {
            // Update text
            const probText = document.getElementById(elements.prob);
            if (probText) {
                probText.textContent = formatProbability(probability);
            }
            
            // Update bar width
            const probBar = document.getElementById(elements.bar);
            if (probBar) {
                probBar.style.width = formatProbability(probability);
            }
        }
    }
}
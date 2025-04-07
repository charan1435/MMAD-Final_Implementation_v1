/**
 * JavaScript for the Attack Generator page
 */

document.addEventListener('DOMContentLoaded', function() {
    // Initialize Dropzone
    const myDropzone = setupDropzone('uploadForm');
    
    // Attack button click handler
    const generateAttackBtn = document.getElementById('generate-attack-btn');
    if (generateAttackBtn) {
        generateAttackBtn.addEventListener('click', function() {
            handleAttackGeneration();
        });
    }
    
    // New attack button click handler (after results)
    const newAttackBtn = document.getElementById('new-attack-btn');
    if (newAttackBtn) {
        newAttackBtn.addEventListener('click', function() {
            resetUI();
        });
    }
    
    // Download adversarial image click handler
    const downloadAdversarialBtn = document.getElementById('download-adversarial');
    if (downloadAdversarialBtn) {
        downloadAdversarialBtn.addEventListener('click', function(e) {
            if (!currentSession.resultId) {
                e.preventDefault();
                showNotification('error', 'Error', 'No adversarial image available for download.');
            }
        });
    }
    
    // Purify this image button click handler
    const purifyThisBtn = document.getElementById('purify-this-btn');
    if (purifyThisBtn) {
        purifyThisBtn.addEventListener('click', function() {
            transferImage('/purify', currentSession.sessionId, currentSession.resultId);
        });
    }
    
    // Epsilon slider value display
    const epsilonRange = document.getElementById('epsilonRange');
    const epsilonValue = document.getElementById('epsilonValue');
    
    if (epsilonRange && epsilonValue) {
        epsilonRange.addEventListener('input', function() {
            epsilonValue.textContent = epsilonRange.value;
        });
    }
    
    // Check for transferred image from other pages
    const lastSession = getLastSessionInfo();
    if (lastSession && lastSession.sessionId) {
        currentSession.sessionId = lastSession.sessionId;
        currentSession.filename = lastSession.filename;
        
        // Enable attack button
        if (generateAttackBtn) {
            generateAttackBtn.disabled = false;
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
 * Handle the attack generation process
 */
function handleAttackGeneration() {
    // Check if we have a session
    if (!currentSession.sessionId) {
        showNotification('error', 'Error', 'No image uploaded. Please upload an image first.');
        return;
    }
    
    // Get attack parameters
    const attackType = document.querySelector('input[name="attackType"]:checked').value;
    const epsilon = document.getElementById('epsilonRange').value;
    
    // Show processing section
    switchSection('processing-section');
    
    // Send attack generation request
    fetch('/attack_image', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            session_id: currentSession.sessionId,
            attack_type: attackType,
            epsilon: epsilon
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Store result ID
            currentSession.resultId = data.result_id;
            
            // Display results
            displayAttackResults(data);
            
            // Switch to results section
            switchSection('results-section');
        } else {
            showNotification('error', 'Attack Generation Failed', data.message);
            switchSection('upload-section');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showNotification('error', 'Error', 'An error occurred during attack generation.');
        switchSection('upload-section');
    });
}

/**
 * Display attack generation results in the UI
 * @param {Object} data - The attack generation results data
 */
function displayAttackResults(data) {
    // Display comparison image
    const comparisonImage = document.getElementById('comparison-image');
    if (comparisonImage) {
        comparisonImage.src = data.comparison_url;
        comparisonImage.alt = 'Comparison of original and adversarial images';
    }
    
    // Update attack details
    const attackTypeValue = document.getElementById('attack-type-value');
    if (attackTypeValue) {
        attackTypeValue.textContent = data.attack_type.toUpperCase();
    }
    
    const epsilonValue = document.getElementById('epsilon-value');
    if (epsilonValue) {
        epsilonValue.textContent = data.epsilon;
    }
    
    const l2Distance = document.getElementById('l2-distance');
    if (l2Distance) {
        l2Distance.textContent = formatNumber(data.l2_distance, 4);
    }
    
    const linfDistance = document.getElementById('linf-distance');
    if (linfDistance) {
        linfDistance.textContent = formatNumber(data.linf_distance, 4);
    }
    
    // Update classification results
    const predictionText = document.getElementById('prediction-text');
    if (predictionText) {
        predictionText.textContent = data.predicted_class;
    }
    
    // Style the result alert based on prediction
    const resultAlert = document.getElementById('result-alert');
    if (resultAlert) {
        // Remove any existing classes
        resultAlert.className = 'alert mb-2';
        
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
    
    // Update confidence bar
    const attackConfidenceBar = document.getElementById('attack-confidence-bar');
    if (attackConfidenceBar) {
        const classIndex = data.class_names.indexOf(data.predicted_class);
        if (classIndex !== -1) {
            const confidence = data.probabilities[classIndex];
            attackConfidenceBar.style.width = formatProbability(confidence);
        }
    }
    
    // Update download links
    const downloadAdversarial = document.getElementById('download-adversarial');
    if (downloadAdversarial) {
        downloadAdversarial.href = data.download_url;
    }
    
    const generateReportBtn = document.getElementById('generate-report-btn');
    if (generateReportBtn) {
        generateReportBtn.href = data.report_url;
    }
}
/**
 * Main JavaScript for Adversarial MRI Defense System
 */

// Store session information
let currentSession = {
    sessionId: null,
    filename: null,
    resultId: null
};

// Configure Dropzone
Dropzone.autoDiscover = false;

document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl)
    });

    // Setup flash messages auto-hide
    setTimeout(function() {
        const flashMessages = document.querySelectorAll('.alert-dismissible');
        flashMessages.forEach(function(message) {
            const alert = new bootstrap.Alert(message);
            alert.close();
        });
    }, 5000);
});

/**
 * Configure and initialize Dropzone for file uploads
 * @param {string} elementId - The ID of the upload form element
 * @param {function} successCallback - Callback function to run on successful upload
 */
function setupDropzone(elementId, successCallback) {
    const uploadForm = document.getElementById(elementId);
    
    if (!uploadForm) return;
    
    const myDropzone = new Dropzone(uploadForm, {
        url: "/upload",
        paramName: "file",
        maxFilesize: 5, // MB
        acceptedFiles: "image/jpeg,image/png,image/jpg,image/tif,image/bmp",
        addRemoveLinks: true,
        dictDefaultMessage: "Drop files here or click to upload",
        dictRemoveFile: "Remove",
        autoProcessQueue: true,
        maxFiles: 1,
        createImageThumbnails: true,
        init: function() {
            this.on("success", function(file, response) {
                if (response.success) {
                    console.log("File uploaded successfully:", response);
                    
                    // Store session information
                    currentSession.sessionId = response.session_id;
                    currentSession.filename = response.filename;
                    
                    // Enable the action button
                    const actionButton = document.getElementById(elementId.replace("uploadForm", "classify-btn")) || 
                                        document.getElementById(elementId.replace("uploadForm", "purify-btn")) ||
                                        document.getElementById(elementId.replace("uploadForm", "generate-attack-btn"));
                    
                    if (actionButton) {
                        actionButton.disabled = false;
                    }
                    
                    // Run callback if provided
                    if (successCallback && typeof successCallback === 'function') {
                        successCallback(response);
                    }
                } else {
                    console.error("Upload failed:", response.message);
                    showNotification('error', 'Upload Failed', response.message);
                }
            });
            
            this.on("error", function(file, errorMessage) {
                console.error("Upload error:", errorMessage);
                showNotification('error', 'Upload Error', errorMessage);
            });
            
            this.on("maxfilesexceeded", function(file) {
                this.removeAllFiles();
                this.addFile(file);
            });
        }
    });
    
    return myDropzone;
}

/**
 * Show a notification to the user
 * @param {string} type - The type of notification (success, error, info, warning)
 * @param {string} title - The notification title
 * @param {string} message - The notification message
 */
function showNotification(type, title, message) {
    // Create notification element if doesn't exist
    let notificationContainer = document.getElementById('notification-container');
    
    if (!notificationContainer) {
        notificationContainer = document.createElement('div');
        notificationContainer.id = 'notification-container';
        notificationContainer.style.position = 'fixed';
        notificationContainer.style.top = '20px';
        notificationContainer.style.right = '20px';
        notificationContainer.style.zIndex = '9999';
        document.body.appendChild(notificationContainer);
    }
    
    // Create notification
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show`;
    notification.innerHTML = `
        <strong>${title}</strong>: ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
    `;
    
    // Add to container
    notificationContainer.appendChild(notification);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.remove();
        }, 300);
    }, 5000);
}

/**
 * Switch between different UI sections (upload, processing, results)
 * @param {string} showSection - The ID of the section to show
 */
function switchSection(showSection) {
    const sections = ['upload-section', 'processing-section', 'results-section'];
    
    sections.forEach(section => {
        const element = document.getElementById(section);
        if (element) {
            if (section === showSection) {
                element.classList.remove('d-none');
            } else {
                element.classList.add('d-none');
            }
        }
    });
}

/**
 * Reset the UI to the initial state (upload section)
 */
function resetUI() {
    // Reset current session
    currentSession = {
        sessionId: null,
        filename: null,
        resultId: null
    };
    
    // Reset Dropzone (remove all files)
    const uploadForm = document.getElementById('uploadForm');
    if (uploadForm && uploadForm.dropzone) {
        uploadForm.dropzone.removeAllFiles();
    }
    
    // Disable action buttons
    const actionButtons = document.querySelectorAll('#classify-btn, #purify-btn, #generate-attack-btn');
    actionButtons.forEach(button => {
        if (button) {
            button.disabled = true;
        }
    });
    
    // Switch to upload section
    switchSection('upload-section');
}

/**
 * Format a probability as a percentage
 * @param {number} prob - The probability value (0-1)
 * @returns {string} Formatted percentage
 */
function formatProbability(prob) {
    return (prob * 100).toFixed(2) + '%';
}

/**
 * Format a number with a specified precision
 * @param {number} value - The number to format
 * @param {number} precision - The decimal places to include
 * @returns {string} Formatted number
 */
function formatNumber(value, precision = 2) {
    return Number(value).toFixed(precision);
}

/**
 * Store image session information in browser storage
 */
function storeSessionInfo() {
    if (currentSession.sessionId && currentSession.resultId) {
        try {
            localStorage.setItem('lastSession', JSON.stringify(currentSession));
        } catch (e) {
            console.warn('Could not store session in localStorage:', e);
        }
    }
}

/**
 * Retrieve last session information from browser storage
 * @returns {Object|null} The last session information or null
 */
function getLastSessionInfo() {
    try {
        const lastSession = localStorage.getItem('lastSession');
        return lastSession ? JSON.parse(lastSession) : null;
    } catch (e) {
        console.warn('Could not retrieve session from localStorage:', e);
        return null;
    }
}

/**
 * Transfer an image from one page to another
 * @param {string} destination - The destination URL
 * @param {string} sessionId - The session ID
 * @param {string} resultId - The result ID
 */
function transferImage(destination, sessionId, resultId) {
    storeSessionInfo();
    window.location.href = destination;
}
// Function to display sound detection results
function displaySoundDetection(data) {
    const resultsContainer = document.getElementById('results-container');
    
    // Create a new result card
    const resultCard = document.createElement('div');
    resultCard.className = 'result-card';
    
    // Format timestamp
    const timestamp = new Date().toLocaleTimeString();
    
    // Check if it's a speech detection with transcription
    let detailsHTML = '';
    if (data.label === 'Speech' && data.transcription) {
        // Get the speech recognition engine used
        const engineUsed = data.transcription_engine || 'unknown';
        const engineBadge = `<span class="engine-badge ${engineUsed}">${engineUsed}</span>`;
        
        // Add emoji if available
        const emoji = data.emoji ? data.emoji : '🗣️';
        
        // Get sentiment info if available
        let sentimentInfo = '';
        if (data.sentiment && data.sentiment.category) {
            sentimentInfo = `<div class="sentiment-info">
                <span class="sentiment-category">${data.sentiment.category}</span>
                <span class="sentiment-emoji">${emoji}</span>
            </div>`;
        }
        
        // Check if there was a Google API error but Vosk fallback worked
        if (data.google_api_error && engineUsed === 'vosk') {
            showToast(`Google Speech API error: ${data.google_api_error}. Using Vosk fallback.`, 'warning');
        }
        
        detailsHTML = `
            <div class="speech-content">
                <div class="speech-header">
                    <span class="transcription-label">Transcription ${engineBadge}</span>
                    ${sentimentInfo}
                </div>
                <p class="transcription-text">"${data.transcription}"</p>
            </div>
        `;
    }
    
    // Format the basic result information
    resultCard.innerHTML = `
        <div class="result-header">
            <span class="result-label">${data.label}</span>
            <span class="result-time">${timestamp}</span>
        </div>
        <div class="result-details">
            <div class="result-confidence">Confidence: ${(parseFloat(data.accuracy) * 100).toFixed(1)}%</div>
            <div class="result-db">Volume: ${parseFloat(data.db).toFixed(1)} dB</div>
        </div>
        ${detailsHTML}
    `;
    
    // Add to results container
    resultsContainer.prepend(resultCard);
    
    // Limit the number of displayed results to avoid performance issues
    const MAX_RESULTS = 50;
    while (resultsContainer.children.length > MAX_RESULTS) {
        resultsContainer.removeChild(resultsContainer.lastChild);
    }
}

// Toast notification system
function showToast(message, type = 'info') {
    // Create toast container if it doesn't exist
    let toastContainer = document.getElementById('toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.id = 'toast-container';
        document.body.appendChild(toastContainer);
    }
    
    // Create toast element
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-content">
            <span class="toast-icon">${type === 'error' ? '❌' : type === 'warning' ? '⚠️' : 'ℹ️'}</span>
            <span class="toast-message">${message}</span>
        </div>
        <button class="toast-close">×</button>
    `;
    
    // Add to container
    toastContainer.appendChild(toast);
    
    // Add close button functionality
    const closeButton = toast.querySelector('.toast-close');
    closeButton.addEventListener('click', () => {
        toast.classList.add('toast-hiding');
        setTimeout(() => {
            toast.remove();
        }, 300);
    });
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.classList.add('toast-hiding');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.remove();
                }
            }, 300);
        }
    }, 5000);
}

// Socket.io error event handler
document.addEventListener('DOMContentLoaded', () => {
    // Assuming socket is defined elsewhere in the application
    if (typeof socket !== 'undefined') {
        // Listen for error events from the server
        socket.on('error', (data) => {
            showToast(data.message || 'An error occurred with speech recognition', 'error');
        });
        
        // Listen for connection errors
        socket.on('connect_error', (error) => {
            showToast('Connection error: ' + error.message, 'error');
        });
    }
});

// Add CSS for the engine badges and toast notifications
document.head.insertAdjacentHTML('beforeend', `
<style>
.engine-badge {
    display: inline-block;
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.8em;
    font-weight: bold;
    margin-left: 5px;
}
.engine-badge.google {
    background-color: #4285F4;
    color: white;
}
.engine-badge.vosk {
    background-color: #F57C00;
    color: white;
}
.engine-badge.unknown {
    background-color: #9E9E9E;
    color: white;
}
.speech-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.transcription-label {
    font-weight: bold;
}
.sentiment-info {
    display: flex;
    align-items: center;
}
.sentiment-category {
    margin-right: 5px;
    font-style: italic;
}
.sentiment-emoji {
    font-size: 1.2em;
}

/* Toast Notification Styles */
#toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 9999;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.toast {
    min-width: 250px;
    max-width: 350px;
    background-color: #333;
    color: white;
    padding: 12px;
    border-radius: 4px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    display: flex;
    justify-content: space-between;
    align-items: center;
    animation: toast-in 0.3s ease-in-out;
}

.toast.toast-hiding {
    animation: toast-out 0.3s ease-in-out forwards;
}

.toast.info {
    background-color: #2196F3;
}

.toast.warning {
    background-color: #FF9800;
}

.toast.error {
    background-color: #F44336;
}

.toast-content {
    display: flex;
    align-items: center;
    gap: 10px;
}

.toast-icon {
    font-size: 1.2em;
}

.toast-message {
    flex: 1;
}

.toast-close {
    background: none;
    border: none;
    color: white;
    font-size: 1.2em;
    cursor: pointer;
    padding: 0;
    margin-left: 10px;
}

@keyframes toast-in {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes toast-out {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}
</style>
`); 
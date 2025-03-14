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

// Add CSS for the engine badges
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
</style>
`); 
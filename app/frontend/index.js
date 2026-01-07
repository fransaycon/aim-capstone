const API_URL = 'https://aim-capstone.onrender.com';

document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Get form data
    const formData = new FormData(e.target);
    const data = {
        age: parseInt(formData.get('age')),
        study_hours: parseFloat(formData.get('study_hours')),
        class_attendance: parseFloat(formData.get('class_attendance')),
        sleep_hours: parseFloat(formData.get('sleep_hours')),
        gender: formData.get('gender'),
        course: formData.get('course'),
        internet_access: formData.get('internet_access'),
        sleep_quality: formData.get('sleep_quality'),
        study_method: formData.get('study_method'),
        facility_rating: formData.get('facility_rating'),
        exam_difficulty: formData.get('exam_difficulty'),
    };

    // Hide empty state, show loading
    document.getElementById('emptyState').style.display = 'none';
    document.getElementById('loading').style.display = 'flex';
    document.getElementById('result').style.display = 'none';
    document.getElementById('submitBtn').disabled = true;

    try {
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Hide loading
        document.getElementById('loading').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;

        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';

        // Check if response was successful (status 200-299)
        if (!response.ok) {
            // Handle error response
            resultDiv.className = 'result error';
            let errorHtml = `
                <h3>❌ Prediction Failed</h3>
                <p>${result.error || 'An error occurred'}</p>
            `;

            if (result.details) {
                errorHtml += `<div class="error-details"><pre>${JSON.stringify(result.details, null, 2)}</pre></div>`;
            }

            resultDiv.innerHTML = errorHtml;
            return;
        }

        // Success - format and display recommendations
        const formattedRecommendations = result.recommendations
            .split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .join('<br>');

        resultDiv.className = 'result success';
        resultDiv.innerHTML = `
            <h3 style="text-align: center; color: #155724;">Predicted Exam Score</h3>
            <div class="score">${result.predicted_score.toFixed(1)}%</div>
            <p class="score-label">Based on the provided student information</p>
            
            <div class="recommendations">
                <h3>📋 Personalized Recommendations</h3>
                <div class="recommendations-content">
                    ${formattedRecommendations}
                </div>
            </div>
        `;

    } catch (error) {
        // Hide loading
        document.getElementById('loading').style.display = 'none';
        document.getElementById('submitBtn').disabled = false;

        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.className = 'result error';
        resultDiv.innerHTML = `
            <h3>❌ Connection Error</h3>
            <p>Could not connect to the prediction server.</p>
            <p style="font-size: 12px; margin-top: 10px;">Error: ${error.message}</p>
            <p style="font-size: 12px;">Make sure the Flask server is running on ${API_URL}</p>
        `;
    }
});
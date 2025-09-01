document.addEventListener('DOMContentLoaded', () => {
    const username = localStorage.getItem('username');
    if (!username) {
        alert('You must be logged in.');
        window.location.href = 'login.html';
        return;
    }

    const sentenceInput = document.getElementById('sentenceInput');
    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('result');

    predictBtn.addEventListener('click', async () => {
        const sentence = sentenceInput.value.trim();
        if (!sentence) {
            alert('Please enter a sentence.');
            return;
        }

        try {
            const response = await fetch('http://127.0.0.1:5000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ sentence, username })
            });

            const data = await response.json();

            if (data.error) {
                resultDiv.innerHTML = `<p class="error">${data.error}</p>`;
            } else {
                resultDiv.innerHTML = `
                    <p><strong>Sentiment:</strong> ${data.sentiment}</p>
                    <p><strong>Score:</strong> ${data.score.toFixed(2)}</p>
                `;
            }
        } catch (err) {
            console.error(err);
            alert('Error connecting to server.');
        }
    });
});

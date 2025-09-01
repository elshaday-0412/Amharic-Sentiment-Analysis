document.addEventListener('DOMContentLoaded', async () => {
    const username = localStorage.getItem('username'); // or from session
    if(!username){
        alert('You must be logged in.');
        window.location.href = 'login.html';
        return;
    }

    try {
        const response = await fetch(`/history/${username}`);
        const data = await response.json();

        const historyList = document.getElementById('historyList');
        if(data.success && data.history.length > 0){
            data.history.forEach(item => {
                const div = document.createElement('div');
                div.classList.add('history-item');
                div.innerHTML = `
                    <p><strong>Sentence:</strong> ${item.sentence}</p>
                    <p><strong>Sentiment:</strong> ${item.sentiment}</p>
                    <p><strong>Score:</strong> ${item.score.toFixed(2)}</p>
                    <hr>
                `;
                historyList.appendChild(div);
            });
        } else {
            historyList.innerHTML = '<p>No history yet.</p>';
        }
    } catch(err){
        console.error(err);
        alert('Error loading history.');
    }
});

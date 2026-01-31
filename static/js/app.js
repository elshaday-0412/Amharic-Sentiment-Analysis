const API_URL = "http://127.0.0.1:5000";

// Show errors
function showError(msg) { 
  alert(msg); 
}

// Predict
function predictSentiment() {
  const text = document.getElementById("sentenceInput").value;
  if (!text) return showError("Enter some text to analyze");

  fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sentence: text })
  })
  .then(res => {
    if (!res.ok) throw new Error("Server error: " + res.status);
    return res.json();
  })
  .then(data => {
    if (data.error) showError(data.error);
    else {
      const resultDiv = document.getElementById("result");
      if (resultDiv) {
        resultDiv.innerText = `Prediction: ${data.sentiment} (${(data.score*100).toFixed(2)}%)`;
      }
    }
  })
  .catch(err => showError("Fetch failed: " + err.message));
}

// DOM ready
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("predictBtn")?.addEventListener("click", predictSentiment);
});

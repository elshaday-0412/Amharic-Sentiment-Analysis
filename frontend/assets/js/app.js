/*
  app.js
  Handles Register, Login, Logout, Predict, History
  + Updates UI dynamically based on session state
*/

const API_URL = "http://127.0.0.1:5000";

// --- Utility: show/hide sections ---
function showSection(id) {
  document.querySelectorAll(".section").forEach(sec => sec.style.display = "none");
  const target = document.getElementById(id);
  if (target) target.style.display = "block";
}

// --- Register ---
function registerUser() {
  const username = document.getElementById("regUsername").value;
  const password = document.getElementById("regPassword").value;

  fetch(`${API_URL}/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ username, password })
  })
  .then(res => res.json())
  .then(data => {
    alert(data.message || "Registered");
    if (data.message && data.message.includes("success")) {
      showSection("loginSection");
    }
  })
  .catch(err => console.error("Register error:", err));
}

// --- Login ---
function loginUser() {
  const username = document.getElementById("loginUsername").value;
  const password = document.getElementById("loginPassword").value;

  fetch(`${API_URL}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ username, password })
  })
  .then(res => res.json())
  .then(data => {
    alert(data.message || "Logged in");
    if (data.message && data.message.includes("success")) {
      showSection("predictSection");
    }
  })
  .catch(err => console.error("Login error:", err));
}

// --- Logout ---
function logoutUser() {
  fetch(`${API_URL}/logout`, {
    method: "POST",
    credentials: "include"
  })
  .then(res => res.json())
  .then(data => {
    alert(data.message || "Logged out");
    showSection("loginSection");
  })
  .catch(err => console.error("Logout error:", err));
}

// --- Predict sentiment ---
function predictSentiment() {
  const userText = document.getElementById("sentenceInput").value;

  fetch(`${API_URL}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ text: userText })
  })
  .then(res => res.json())
  .then(data => {
    document.getElementById("result").innerText =
      "Prediction: " + data.sentiment;
    loadHistory(); // refresh history automatically
  })
  .catch(err => console.error("Prediction error:", err));
}

// --- Get history ---
function loadHistory() {
  fetch(`${API_URL}/history`, {
    method: "GET",
    credentials: "include"
  })
  .then(res => res.json())
  .then(data => {
    const historyDiv = document.getElementById("history");
    historyDiv.innerHTML = "";

    if (data.history && data.history.length > 0) {
      data.history.forEach(item => {
        const p = document.createElement("p");
        p.innerText = `${item.text} → ${item.sentiment}`;
        historyDiv.appendChild(p);
      });
    } else {
      historyDiv.innerText = "No history yet.";
    }
  })
  .catch(err => console.error("History error:", err));
}

// --- Attach button listeners ---
document.addEventListener("DOMContentLoaded", () => {
  // Default to login view
  showSection("loginSection");

  const regBtn = document.getElementById("registerBtn");
  if (regBtn) regBtn.addEventListener("click", registerUser);

  const loginBtn = document.getElementById("loginBtn");
  if (loginBtn) loginBtn.addEventListener("click", loginUser);

  const logoutBtn = document.getElementById("logoutBtn");
  if (logoutBtn) logoutBtn.addEventListener("click", logoutUser);

  const predictBtn = document.getElementById("predictBtn");
  if (predictBtn) predictBtn.addEventListener("click", predictSentiment);

  const historyBtn = document.getElementById("historyBtn");
  if (historyBtn) historyBtn.addEventListener("click", loadHistory);
});

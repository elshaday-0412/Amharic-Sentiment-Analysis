document.addEventListener('DOMContentLoaded', () => {

    // -----------------------------
    // Sign-Up
    // -----------------------------
    const createBtn = document.getElementById('createAccountBtn');
    if (createBtn) {
        createBtn.onclick = async (e) => {
            e.preventDefault();

            const username = document.getElementById('signupUsername').value.trim();
            const password = document.getElementById('signupPassword').value;

            if(username.length < 3){
                alert('Username must be at least 3 characters long.');
                return;
            }
            if(password.length < 8){
                alert('Password must be at least 8 characters long.');
                return;
            }

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });

                const data = await response.json();

                if(data.success){
                    alert(data.message);
                    window.location.href = 'login.html';
                } else {
                    alert(data.message);
                }
            } catch(err) {
                console.error(err);
                alert('Error connecting to server.');
            }
        }
    }

    // -----------------------------
    // Login
    // -----------------------------
    const loginBtn = document.getElementById('loginBtn');
    if (loginBtn) {
        loginBtn.onclick = async (e) => {
            e.preventDefault();

            const username = document.getElementById('loginUsername').value.trim();
            const password = document.getElementById('loginPassword').value;

            if(username.length < 3){
                alert('Username must be at least 3 characters long.');
                return;
            }
            if(password.length < 8){
                alert('Password must be at least 8 characters long.');
                return;
            }

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ username, password })
                });

                const data = await response.json();

                if(data.success){
                    alert(data.message);
                    window.location.href = 'frontend/Predictor.html';
                } else {
                    alert(data.message);
                }
            } catch(err) {
                console.error(err);
                alert('Error connecting to server.');
            }
        }
    }

    // -----------------------------
    // Show / Hide password toggle
    // -----------------------------
    document.querySelectorAll('.toggle-password').forEach(toggle => {
        toggle.addEventListener('click', () => {
            const input = document.querySelector(toggle.dataset.target);
            if(input.type === 'password'){
                input.type = 'text';
                toggle.textContent = 'Hide';
            } else {
                input.type = 'password';
                toggle.textContent = 'Show';
            }
        });
    });

});


    document.getElementById('form1').addEventListener('submit', async (e) => {
e.preventDefault();

const username = document.getElementById('text1').value;
const password = document.getElementById('pass1').value;

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
});


from flask import Flask, request, jsonify, session
from flask_cors import CORS
import torch
import os, json, pickle, secrets
from model.sentiment_model import SentimentModel, load_model
from utils.preprocess import preprocess_text, tokenize

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", secrets.token_hex(32))

# Enable CORS for frontend running on a different port (like localhost:5500)
CORS(app, supports_credentials=True)

# -----------------------------
# Load vocab
# -----------------------------
VOCAB_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'vocab.pkl')
with open(VOCAB_PATH, 'rb') as f:
    vocab = pickle.load(f)
vocab_size = len(vocab)

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'sentiment_model_best.pt')
model = SentimentModel(vocab_size, 256, 128, 3, 2, True, 0.5)
load_model(model, MODEL_PATH)
model.eval()

# -----------------------------
# Users storage
# -----------------------------
USERS_FILE = os.path.join(os.path.dirname(__file__), 'users.json')
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        json.dump({}, f)

HISTORY_DIR = os.path.join(os.path.dirname(__file__), 'user_history')
os.makedirs(HISTORY_DIR, exist_ok=True)

def load_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f, indent=2)

def save_history(username, sentence, sentiment, score):
    filepath = os.path.join(HISTORY_DIR, f'{username}.json')
    history = []
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            history = json.load(f)
    history.append({'sentence': sentence, 'sentiment': sentiment, 'score': score})
    with open(filepath, 'w') as f:
        json.dump(history, f, indent=2)

# -----------------------------
# Routes
# -----------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'success': False, 'message': 'Username and password required.'})

    if len(username) < 3 or len(password) < 8:
        return jsonify({'success': False, 'message': 'Username must be ≥3, password ≥8 characters.'})

    users = load_users()
    if username in users:
        return jsonify({'success': False, 'message': 'Username already taken.'})

    users[username] = {'password': password}  # TODO: hash passwords
    save_users(users)
    return jsonify({'success': True, 'message': 'Account created successfully.'})

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    users = load_users()
    if username not in users or users[username]['password'] != password:
        return jsonify({'success': False, 'message': 'Invalid username or password.'})

    session.permanent = True
    session['username'] = username
    return jsonify({'success': True, 'message': 'Logged in successfully.'})

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('username', None)
    return jsonify({'success': True, 'message': 'Logged out successfully.'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sentence = data.get('sentence')
    username = data.get('username', session.get('username'))

    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400

    processed = preprocess_text(sentence)
    tokens = tokenize(processed)
    input_tensor = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]

    if len(input_tensor) == 0:
        return jsonify({'error': 'Empty input after preprocessing'}), 400

    input_tensor = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1).squeeze(0)
        labels = ['negative', 'neutral', 'positive']
        sentiment_idx = torch.argmax(probs).item()
        sentiment = labels[sentiment_idx]
        score = probs[sentiment_idx].item()

    if username:
        save_history(username, sentence, sentiment, score)

    return jsonify({'sentiment': sentiment, 'score': score})

@app.route('/history/<username>', methods=['GET'])
def get_history(username):
    filepath = os.path.join(HISTORY_DIR, f'{username}.json')
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            history = json.load(f)
        return jsonify({'success': True, 'history': history})
    return jsonify({'success': True, 'history': []})

# -----------------------------
# Run app
# -----------------------------
if __name__ == '__main__':
    from datetime import timedelta
    app.permanent_session_lifetime = timedelta(days=7)
    app.run(host='127.0.0.1', port=5000, debug=True)

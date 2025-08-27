from flask import Flask, request, jsonify
from model.sentiment_model import SentimentModel
from utils.preprocess import preprocess_text

app = Flask(__name__)
model = SentimentModel()

@app.route('/api/sentiment', methods=['POST'])
def analyze_sentiment():
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    processed_text = preprocess_text(text)
    sentiment = model.predict(processed_text)
    
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
from flask_cors import CORS
import torch
import os
import pandas as pd

from model.sentiment_model import SentimentModel, load_model
from utils.preprocess import preprocess_text, tokenize, build_vocab

app = Flask(__name__)
CORS(app)  # allow all origins

# -----------------------------
# Load dataset and build vocab
# -----------------------------
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'final.csv')
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
sentences = df['sentence'].astype(str).tolist()
vocab = build_vocab(sentences)
vocab_size = len(vocab)
print(f"Vocab size: {vocab_size}")

# -----------------------------
# Model parameters
# -----------------------------
embedding_dim = 64
hidden_dim = 128
output_dim = 1
n_layers = 2
bidirectional = True
dropout = 0.5

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'saved_models', 'sentiment_model.pt')
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

model = SentimentModel(vocab_size, embedding_dim, hidden_dim, output_dim,
                       n_layers, bidirectional, dropout)
load_model(model, MODEL_PATH)
model.eval()
print("Model loaded successfully")

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'sentence' not in data:
            return jsonify({'error': 'No sentence provided'}), 400

        sentence = data['sentence']
        processed = preprocess_text(sentence)
        tokens = tokenize(processed)

        input_tensor = [vocab.get(token, vocab.get('<UNK>', 0)) for token in tokens]
        if len(input_tensor) == 0:
            return jsonify({'error': 'Empty input after preprocessing'}), 400

        input_tensor = torch.tensor(input_tensor, dtype=torch.long).unsqueeze(0)  # [1, seq_len]

        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            # Optional: classify neutral if close to 0.5
            if 0.45 <= prob <= 0.55:
                sentiment = 'neutral'
            else:
                sentiment = 'positive' if prob > 0.5 else 'negative'

        return jsonify({'sentiment': sentiment, 'score': prob})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -----------------------------
# Run Flask app
# -----------------------------
if __name__ == '__main__':
    # localhost for development, port 5000
    app.run(host='127.0.0.1', port=5000, debug=True)


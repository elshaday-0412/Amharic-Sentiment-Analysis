from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import torch, os, pickle, logging
import torch.nn as nn
from collections import Counter

# -----------------------------
# Flask app setup
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, 'frontend'),
    static_folder=os.path.join(BASE_DIR, 'static')
)

CORS(app)  # allow frontend fetch

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# -----------------------------
# Model & vocab paths
# -----------------------------
VOCAB_PATH = os.path.join(BASE_DIR, 'saved_models', 'vocab.pkl')
MODEL_PATH = os.path.join(BASE_DIR, 'saved_models', 'sentiment_model_best.pt')

# -----------------------------
# Define SentimentModel (same as training)
# -----------------------------
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=64, output_dim=3,
                 n_layers=2, bidirectional=True, dropout=0.5, padding_idx=0):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, num_layers=n_layers,
            bidirectional=bidirectional, dropout=dropout, batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        
        # Additional layer for more capacity (matches your training)
        self.fc1 = nn.Linear(hidden_dim*4 if bidirectional else hidden_dim*2, hidden_dim//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        
        # Use multiple hidden states for more information (matches training)
        if self.lstm.bidirectional:
            forward_hidden = h_n[-2]  # Last forward layer
            backward_hidden = h_n[-1]  # Last backward layer
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
            
            # Also use the last output for more context
            last_output = lstm_out[:, -1, :]
            combined = torch.cat((last_hidden, last_output), dim=1)
        else:
            last_hidden = h_n[-1]
            last_output = lstm_out[:, -1, :]
            combined = torch.cat((last_hidden, last_output), dim=1)
        
        out = self.dropout(combined)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        return self.fc2(out)

# -----------------------------
# Load vocab
# -----------------------------
try:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)
    print(f"✓ Vocab loaded ({vocab_size} words)")
except Exception as e:
    logger.error(f"Failed to load vocab: {str(e)}")
    vocab = {}
    vocab_size = 0

# -----------------------------
# Load model
# -----------------------------
try:
    model = SentimentModel(
        vocab_size,
        embedding_dim=128,
        hidden_dim=64,
        output_dim=3,
        n_layers=2,
        bidirectional=True,
        dropout=0.5,
        padding_idx=vocab.get("<PAD>", 0)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

# -----------------------------
# Preprocess functions (same as training)
# -----------------------------
def preprocess_text(text):
    """Preprocess text - lowercase and basic cleaning"""
    return text.lower().strip()

def tokenize(text):
    """Tokenize text into words"""
    return text.split()

def build_vocab(sentences, min_freq=1, max_size=50000):
    """Build vocabulary from sentences (for reference)"""
    word_counts = Counter()
    for sentence in sentences:
        tokens = tokenize(preprocess_text(str(sentence)))
        word_counts.update(tokens)
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    idx = 2
    for word, count in word_counts.most_common():
        if count >= min_freq and idx < max_size:
            vocab[word] = idx
            idx += 1
    
    return vocab

def sentence_to_tensor(sentence, max_length=200):
    """Convert sentence to tensor (same as training)"""
    tokens = tokenize(preprocess_text(str(sentence)))
    # Use vocab.get with default value of UNK index
    ids = [vocab.get(t, vocab.get("<UNK>", 1)) for t in tokens[:max_length]]
    if len(ids) == 0:
        ids = [vocab.get("<UNK>", 1)]
    # Ensure indices are within bounds
    ids = [min(idx, len(vocab) - 1) for idx in ids]
    return torch.tensor(ids, dtype=torch.long)

# -----------------------------
# Frontend routes
# -----------------------------
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/predictor')
def predictor_page():
    """predictor page"""
    return render_template('predictor.html')

# -----------------------------
# Prediction route
# -----------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    if not data: 
        return jsonify({'error': 'No data provided'}), 400

    sentence = data.get('sentence')
    if not sentence: 
        return jsonify({'error': 'No sentence provided'}), 400

    try:
        # Preprocess and convert to tensor
        input_tensor = sentence_to_tensor(sentence)
        
        if len(input_tensor) == 0:
            return jsonify({'error': 'Empty input after preprocessing'}), 400

        # Add batch dimension and predict
        input_tensor = input_tensor.unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).squeeze(0)
            labels = ['negative', 'neutral', 'positive']
            idx = torch.argmax(probs).item()
            sentiment = labels[idx]
            score = probs[idx].item()

        return jsonify({
            'sentiment': sentiment, 
            'score': round(score, 4),
            'probabilities': {
                'negative': round(probs[0].item(), 4),
                'neutral': round(probs[1].item(), 4),
                'positive': round(probs[2].item(), 4)
            }
        })

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Prediction failed'}), 500

# -----------------------------
# Health check route
# -----------------------------
@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'vocab_size': len(vocab)
    })

# -----------------------------
# Run Flask
# -----------------------------
if __name__ == '__main__':
    print("="*50)
    print("Starting Flask server...")
    print("="*50)
    app.run(host='127.0.0.1', port=5000, debug=True)
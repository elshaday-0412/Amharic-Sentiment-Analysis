import os
import torch
import pandas as pd
import pickle
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.preprocess import preprocess_text, tokenize, text_to_tensor
from train import SentimentModel

# -----------------------------
# Load validation data
# -----------------------------
VAL_PATH = "data/validation_set.csv"
val_df = pd.read_csv(VAL_PATH)

# Normalize labels (handles Pos/NEG/neu/extra spaces)
val_df['label'] = val_df['label'].astype(str).str.lower().str.strip()

# Keep only valid labels
valid_labels = {"negative": 0, "neutral": 1, "positive": 2}
val_df = val_df[val_df['label'].isin(valid_labels.keys())]

if len(val_df) == 0:
    raise ValueError("Validation set is empty after filtering labels!")

y_true = val_df['label'].map(valid_labels).tolist()
sentences = val_df['sentence'].astype(str).tolist()

# -----------------------------
# Load vocab (from training)
# -----------------------------
VOCAB_PATH = os.path.join("saved_models", "vocab.pkl")
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# -----------------------------
# Prepare validation tensors
# -----------------------------
X = []
for sentence in sentences:
    tokens = tokenize(preprocess_text(sentence))
    tensor = text_to_tensor(tokens, vocab)
    tensor = torch.as_tensor(tensor, dtype=torch.long).view(-1)
    X.append(tensor)

if len(X) == 0:
    raise ValueError("No sentences were converted! Check preprocessing or vocab.")

X_padded = pad_sequence(X, batch_first=True, padding_value=0)
y_tensor = torch.tensor(y_true, dtype=torch.long)

# -----------------------------
# Load best model
# -----------------------------
embedding_dim = 256
hidden_dim = 128
output_dim = 3
n_layers = 2
bidirectional = True
dropout = 0.5

model = SentimentModel(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    bidirectional=bidirectional,
    dropout=dropout
)

BEST_MODEL_PATH = os.path.join("saved_models", "sentiment_model_best.pt")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# -----------------------------
# Run validation
# -----------------------------
with torch.no_grad():
    outputs = model(X_padded)
    preds = torch.argmax(outputs, dim=1)

y_pred = preds.cpu().numpy()

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

print("\n📊 Validation Metrics (Best Model):")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

import torch
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.preprocess import preprocess_text, tokenize, build_vocab, text_to_tensor
from train import SentimentModel  # import your model class

# -----------------------------
# Load validation data
# -----------------------------
VAL_PATH = "data/validation_set.csv"  # replace with your validation CSV path
val_df = pd.read_csv(VAL_PATH)

# Keep only valid labels
val_df = val_df[val_df['label'].isin(['positive', 'neutral', 'negative'])]

label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
y_true = val_df['label'].map(label2id).tolist()
sentences = val_df['sentence'].astype(str).tolist()

# -----------------------------
# Load vocab
# -----------------------------
# Make sure you use the same vocab as training
TRAIN_CSV = "data/final.csv"
train_df = pd.read_csv(TRAIN_CSV)
vocab = build_vocab(train_df['sentence'].astype(str).tolist())
vocab_size = len(vocab)

# -----------------------------
# Prepare validation tensors
# -----------------------------
X = []
for sentence in sentences:
    tokens = tokenize(preprocess_text(sentence))
    tensor = text_to_tensor(tokens, vocab)
    tensor = torch.as_tensor(tensor, dtype=torch.long).view(-1)
    X.append(tensor)

X_padded = pad_sequence(X, batch_first=True, padding_value=0)
y_tensor = torch.tensor(y_true, dtype=torch.long)

# -----------------------------
# Load model
# -----------------------------
embedding_dim = 256
hidden_dim = 128
output_dim = 3
n_layers = 2
bidirectional = True
dropout = 0.5

model = SentimentModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    n_layers=n_layers,
    bidirectional=bidirectional,
    dropout=dropout
)

# Load the weights
model.load_state_dict(torch.load("saved_models/sentiment_model_final.pt"))
model.eval()

# -----------------------------
# Run validation
# -----------------------------
with torch.no_grad():
    outputs = model(X_padded)
    preds = torch.argmax(outputs, dim=1)

y_pred = preds.cpu().numpy()

acc = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

print("Validation Metrics:")
print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from utils.preprocess import preprocess_text, tokenize, build_vocab, text_to_tensor
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# -----------------------------
# Define the LSTM model
# -----------------------------
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.5):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        if self.lstm.bidirectional:
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)
        else:
            last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        out = self.fc(out)
        return out

# -----------------------------
# Load dataset
# -----------------------------
DATA_PATH = os.path.join('data', 'final.csv')
df = pd.read_csv(DATA_PATH)

# Keep only valid labels
df = df[df['label'].isin(['positive', 'neutral', 'negative'])]
label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
labels = df['label'].map(label_map).tolist()
sentences = df['sentence'].astype(str).tolist()

# -----------------------------
# Build vocabulary
# -----------------------------
vocab = build_vocab(sentences)
vocab_size = len(vocab)

# -----------------------------
# Prepare data tensors
# -----------------------------
X = []
y = []

for sentence, label in zip(sentences, labels):
    tokens = tokenize(preprocess_text(sentence))
    tensor = text_to_tensor(tokens, vocab)
    tensor = torch.as_tensor(tensor, dtype=torch.long).view(-1)
    X.append(tensor)
    y.append(torch.tensor(label, dtype=torch.long))

X_padded = pad_sequence(X, batch_first=True, padding_value=0)
y_tensor = torch.stack(y)

# -----------------------------
# Create model
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

# -----------------------------
# Weighted loss for class imbalance
# -----------------------------
class_counts = df['label'].value_counts().sort_index()  # negative, neutral, positive
class_weights = 1.0 / class_counts.values
class_weights = torch.tensor(class_weights, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

# -----------------------------
# Training loop with metrics
# -----------------------------
epochs = 100
os.makedirs('saved_models', exist_ok=True)
best_acc = 0.0  # track best validation accuracy

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_padded)  # [batch_size, 3]
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    scheduler.step()

    # Compute metrics
    with torch.no_grad():
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        y_true = y_tensor.cpu().numpy()
        y_pred = preds.cpu().numpy()

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
          f"Acc: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    # -------------------------
    # Save checkpoints every 20 epochs
    # -------------------------
    if (epoch + 1) % 20 == 0:
        checkpoint_path = f'saved_models/sentiment_epoch{epoch+1}.pt'
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    # -------------------------
    # Save best model
    # -------------------------
    if acc > best_acc:
        best_acc = acc
        best_path = 'saved_models/sentiment_model_best.pt'
        torch.save(model.state_dict(), best_path)
        print(f"New best model saved: {best_path}")

# -----------------------------
# Save final model
# -----------------------------
final_path = os.path.join('saved_models', 'sentiment_model_final.pt')
torch.save(model.state_dict(), final_path)
print(f"Final model saved: {final_path}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.nn.utils.rnn import pad_sequence
from utils.preprocess import preprocess_text, tokenize
from collections import Counter

# Define LSTM model
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=3, bidirectional=True, dropout=0.4, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(hidden_dim*4 if bidirectional else hidden_dim*2, hidden_dim//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        emb = self.embedding(x)
        lstm_out, (h_n, c_n) = self.lstm(emb)
        
        if self.lstm.bidirectional:
            forward_hidden = h_n[-2]
            backward_hidden = h_n[-1]
            last_hidden = torch.cat((forward_hidden, backward_hidden), dim=1)
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

def build_vocab(sentences, min_freq=1, max_size=50000):
    from collections import Counter
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

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    DATA_PATH = os.path.join('data', 'final.csv')
    df = pd.read_csv(DATA_PATH)
    df['label'] = df['label'].astype(str).str.lower().str.strip()
    keep = ['positive', 'neutral', 'negative']
    df = df[df['label'].isin(keep)]
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['label_id'] = df['label'].map(label_map)

    # Reduce dataset to 50,000 samples
    df = df.head(50000)
    print(f"Reduced dataset to {len(df)} samples")

    # PROPER 70-20-10 SPLIT: Train (70%) - Validation (20%) - Test (10%)
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label_id'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.333, stratify=temp_df['label_id'], random_state=42)
    
    print(f"\nDataset split:")
    print(f"Training samples: {len(train_df)} (70%)")
    print(f"Validation samples: {len(val_df)} (20%)") 
    print(f"Test samples: {len(test_df)} (10%)")

    # Check class distribution
    print("\nClass distribution:")
    print(f"Training: {dict(train_df['label_id'].value_counts().sort_index())}")
    print(f"Validation: {dict(val_df['label_id'].value_counts().sort_index())}")
    print(f"Test: {dict(test_df['label_id'].value_counts().sort_index())}")

    # Build vocab only on training data
    vocab = build_vocab(train_df['sentence'].astype(str).tolist(), min_freq=2, max_size=50000)
    print(f"\nVocabulary size: {len(vocab)}")

    # Prepare tensors function
    def sentence_to_tensor(sentence, max_length=200):
        tokens = tokenize(preprocess_text(str(sentence)))
        ids = [vocab.get(t, vocab["<UNK>"]) for t in tokens[:max_length]]
        if len(ids) == 0:
            ids = [vocab["<UNK>"]]
        return torch.tensor(ids, dtype=torch.long)

    def prepare_tensors(df_split, max_length=200):
        X, y = [], []
        for s, label in zip(df_split['sentence'], df_split['label_id']):
            X.append(sentence_to_tensor(s, max_length))
            y.append(torch.tensor(label, dtype=torch.long))
        X_padded = pad_sequence(X, batch_first=True, padding_value=vocab["<PAD>"])
        y_tensor = torch.stack(y)
        return X_padded, y_tensor

    # Prepare all three datasets (force recreation)
    print("Preparing tensors for all splits...")
    X_train, y_train = prepare_tensors(train_df, max_length=200)
    X_val, y_val = prepare_tensors(val_df, max_length=200)
    X_test, y_test = prepare_tensors(test_df, max_length=200)

    print(f"Shapes - Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

    # Create DataLoaders for all three sets
    batch_size = 32
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    embedding_dim = 128
    hidden_dim = 64
    output_dim = 3
    n_layers = 2
    bidirectional = True
    dropout = 0.5
    
    model = SentimentModel(len(vocab), embedding_dim, hidden_dim, output_dim, 
                          n_layers, bidirectional, dropout, padding_idx=vocab["<PAD>"]).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scaler = GradScaler()

    # Training loop - USE VALIDATION SET FOR EVALUATION
    max_epochs = 10
    best_val_acc = 0.0
    best_epoch = 0

    print(f"\nStarting training - WILL STOP AT EPOCH {max_epochs}...")
    for epoch in range(max_epochs):
        # TRAINING PHASE
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            with autocast(device_type='cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_acc = 100 * train_correct / train_total
        avg_loss = total_loss / len(train_loader)

        # VALIDATION PHASE (NOT TEST!)
        model.eval()
        val_pred, val_true = [], []
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:  # Use VALIDATION set
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1)
                val_pred.extend(preds.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())

        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average='weighted', zero_division=0)
        
        # DEBUG: Check distributions
        pred_counts = Counter(val_pred)
        true_counts = Counter(val_true)
        print(f"Epoch {epoch+1}/{max_epochs}, Loss: {avg_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        print(f"  Val True: {dict(true_counts)}, Val Pred: {dict(pred_counts)}")

        # Update best model based on VALIDATION accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "saved_models/sentiment_model_best.pt")
            print(f"  ↗ NEW BEST VALIDATION MODEL!")

    # Load the best model (based on validation performance)
    model.load_state_dict(torch.load("saved_models/sentiment_model_best.pt"))
    torch.save(model.state_dict(), "saved_models/sentiment_model_final.pt")
    
    # FINAL TEST PHASE (ONLY USED ONCE!)
    print("\n=== FINAL TEST EVALUATION ===")
    model.eval()
    test_pred, test_true = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:  # Use TEST set (first and only time)
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            with autocast(device_type='cuda'):
                outputs = model(batch_X)
            preds = torch.argmax(outputs, dim=1)
            test_pred.extend(preds.cpu().numpy())
            test_true.extend(batch_y.cpu().numpy())

    test_acc = accuracy_score(test_true, test_pred)
    test_f1 = f1_score(test_true, test_pred, average='weighted', zero_division=0)
    test_precision = precision_score(test_true, test_pred, average='weighted', zero_division=0)
    test_recall = recall_score(test_true, test_pred, average='weighted', zero_division=0)
    
    print(f"Final Test Results:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  F1 Score: {test_f1:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    # Save vocab
    with open("saved_models/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)

if __name__ == '__main__':
    main()
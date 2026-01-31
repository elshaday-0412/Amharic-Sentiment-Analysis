import torch
import torch.nn as nn

class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,
                 n_layers=2, bidirectional=True, dropout=0.5, padding_idx=0):
        super(SentimentModel, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)

        # ⚡ same as train.py: 2 fully connected layers
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc1 = nn.Linear(fc_input_dim * 2, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, output_dim)

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

# -----------------------------
# Save / Load utilities
# -----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

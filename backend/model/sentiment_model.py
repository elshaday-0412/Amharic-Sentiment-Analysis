import torch
import torch.nn as nn

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
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        """
        text: [batch_size, seq_len]
        Returns: [batch_size, 1] (logits for BCEWithLogitsLoss)
        """

        # Embedding + dropout
        embedded = self.dropout(self.embedding(text))  # [batch_size, seq_len, embedding_dim]

        # Compute actual lengths (assuming padding index = 0)
        lengths = (text != 0).sum(dim=1)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)

        # Concatenate last forward and backward hidden states if bidirectional
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch_size, hidden_dim*2]
        else:
            hidden = hidden[-1]  # [batch_size, hidden_dim]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)  # [batch_size, 1]

        return output

# -----------------------------
# Save / Load utilities
# -----------------------------
def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path, device='cpu'):
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
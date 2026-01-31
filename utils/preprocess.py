import torch
def preprocess_text(text):
    text = text.lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text
def tokenize(text):

    return text.split()

def build_vocab(sentences, min_freq=1):
    freq = {}
    for sentence in sentences:
        for token in tokenize(preprocess_text(sentence)):
            freq[token] = freq.get(token, 0) + 1
    vocab = {token: idx for idx, (token, count) in enumerate(freq.items()) if count >= min_freq}
    vocab['<UNK>'] = len(vocab)
    return vocab

def text_to_tensor(tokens, vocab):
    indices = [vocab.get(token, vocab['<UNK>']) for token in tokens]
    return torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # shape: (1, seq_len)


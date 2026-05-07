import torch
import torch.nn as nn


class ToneClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout_p=0.2):
        super(ToneClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        _, hidden = self.gru(embedded)
        hidden = hidden.squeeze(0)
        output = self.fc(hidden)
        return output
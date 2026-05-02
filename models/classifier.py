import torch
import torch.nn as nn

class ToneClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_classes, dropout_p=0.2):
        super(ToneClassifier, self).__init__()
        
        # 1. Embedding Layer: Turns word IDs into dense vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_p)
        
        # 2. GRU: Processes the sequence of word embeddings
        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        
        # 3. Fully Connected Layer: Maps the final hidden state to our classes
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.dropout(self.embedding(x))
        
        # Pass through GRU. We only care about 'hidden' (the final state)
        _, hidden = self.gru(embedded)
        
        # hidden shape: (1, batch_size, hidden_size). Squeeze to remove the 1st dimension.
        hidden = hidden.squeeze(0)
        
        # Output shape: (batch_size, num_classes)
        output = self.fc(hidden)
        return output
import torch
import torch.nn as nn

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # 1. Embedding Layer: Turns integer word IDs into dense vectors
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # 2. GRU Layer: Processes the sequence
        # Using batch_first=True matches our DataLoader output
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input_tensor):
        # input_tensor shape: (batch_size, sequence_length)
        
        embedded = self.dropout(self.embedding(input_tensor))
        # embedded shape: (batch_size, sequence_length, hidden_size)
        
        # Pass the embedded sequence through the GRU
        # output contains the features from the last layer of the GRU for each timestep
        # hidden contains the final state $h_t$ for the entire sequence
        output, hidden = self.gru(embedded)
        
        return output, hidden
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=50, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        
        # Attention Layers
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden, encoder_outputs):
        # input_tensor is a single target word ID: (batch_size, 1)
        embedded = self.embedding(input_tensor)
        embedded = self.dropout(embedded)
        
        # Calculate Attention Weights
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(1), hidden.squeeze(0)), 1)), dim=1)
        
        # --- NEW FIX: Pad or truncate encoder_outputs to exactly match max_length ---
        seq_len = encoder_outputs.shape[1]
        if seq_len < self.max_length:
            # Create padding of zeros and concatenate it to the end of the sequence
            padding = torch.zeros(
                encoder_outputs.shape[0], 
                self.max_length - seq_len, 
                self.hidden_size, 
                device=encoder_outputs.device
            )
            encoder_outputs = torch.cat((encoder_outputs, padding), dim=1)
        elif seq_len > self.max_length:
            # Truncate if the sequence is somehow longer than max_length
            encoder_outputs = encoder_outputs[:, :self.max_length, :]
        # ----------------------------------------------------------------------------
        
        # Apply weights to encoder outputs to get the "Context Vector"
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        # Combine embedded input with the context vector
        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)
        
        # Pass through GRU
        output, hidden = self.gru(output, hidden)
        
        # Predict the next word ID in the target vocabulary
        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)
        
        return output, hidden, attn_weights
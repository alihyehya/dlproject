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
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_tensor, hidden, encoder_outputs):
        embedded = self.dropout(self.embedding(input_tensor))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded.squeeze(1), hidden.squeeze(0)), 1)), dim=1)

        seq_len = encoder_outputs.shape[1]
        if seq_len < self.max_length:
            padding = torch.zeros(
                encoder_outputs.shape[0],
                self.max_length - seq_len,
                self.hidden_size,
                device=encoder_outputs.device
            )
            encoder_outputs = torch.cat((encoder_outputs, padding), dim=1)
        elif seq_len > self.max_length:
            encoder_outputs = encoder_outputs[:, :self.max_length, :]

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)

        output = torch.cat((embedded.squeeze(1), attn_applied.squeeze(1)), 1)
        output = self.attn_combine(output).unsqueeze(1)
        output = F.relu(output)

        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output.squeeze(1)), dim=1)

        return output, hidden, attn_weights
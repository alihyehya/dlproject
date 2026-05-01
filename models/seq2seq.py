import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, sos_token_id=1):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_token_id = sos_token_id

    def forward(self, source, target, teacher_forcing_ratio=0.5):
        batch_size = source.shape[0]
        target_len = target.shape[1]
        target_vocab_size = self.decoder.output_size
        
        # Tensor to store decoder outputs
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(self.device)
        
        # 1. Pass the entire source sequence through the Encoder
        encoder_outputs, hidden = self.encoder(source)
        
        # 2. Prepare the first input to the Decoder (the <SOS> token)
        decoder_input = torch.tensor([[self.sos_token_id]] * batch_size, device=self.device)
        
        # 3. Decode word-by-word
        for t in range(1, target_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            # Decide whether to use teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            
            # Get the highest predicted token index
            top1 = output.argmax(1) 
            
            # If teacher forcing, use the actual ground-truth word as the next input
            # Otherwise, use the model's own prediction as the next input
            decoder_input = target[:, t].unsqueeze(1) if teacher_force else top1.unsqueeze(1)
            
        return outputs
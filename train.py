import torch
import torch.nn as nn
from torch import optim
from utils.data_loader import ToneDataset, collate_batch, TOY_DATA
from torch.utils.data import DataLoader
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.seq2seq import Seq2Seq

# 1. Configuration & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256
BATCH_SIZE = 2
LEARNING_RATE = 0.001
EPOCHS = 100

# 2. Load the Data
dataset = ToneDataset(TOY_DATA)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

# 3. Initialize the Models
# Add 4 to vocab sizes to account for PAD, SOS, EOS, UNK
input_vocab_size = dataset.source_vocab.n_words
output_vocab_size = dataset.target_vocab.n_words

encoder = EncoderRNN(input_size=input_vocab_size, hidden_size=HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_vocab_size).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# 4. Optimizer and Loss Function
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# CRITICAL: We tell the loss function to completely ignore <PAD> tokens (index 0)
# Otherwise, the model wastes effort learning how to predict empty space.
criterion = nn.CrossEntropyLoss(ignore_index=0)

def train():
    model.train() # Set the model to training mode (enables dropout)
    
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (source, target) in enumerate(dataloader):
            source, target = source.to(device), target.to(device)
            
            # Step A: Clear the old gradients from the last step
            optimizer.zero_grad()
            
            # Step B: Forward pass (execute the network)
            # outputs shape: (batch_size, sequence_length, vocab_size)
            output = model(source, target)
            
            # Step C: Calculate the loss
            # PyTorch's CrossEntropyLoss expects inputs of shape (N, C) and targets of shape (N)
            # So we flatten our 3D tensors into 2D matrices
            output_dim = output.shape[-1]
            
            # We slice output[:, 1:] and target[:, 1:] to ignore the <SOS> token at index 0
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            
            # Step D: Backward pass (calculate the gradients)
            loss.backward()
            
            # Step E: Update the hardware weights (the registers/parameters)
            optimizer.step()
            
            total_loss += loss.item()
            
        # Print progress
        avg_loss = total_loss / len(dataloader)
        if epoch % 10 == 0:
            print(f"Epoch: {epoch}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    print("Starting training...")
    train()
    print("Training complete!")
    
    # Save the model weights
    torch.save(model.state_dict(), "tone_translator.pth")
    print("Model saved to tone_translator.pth")
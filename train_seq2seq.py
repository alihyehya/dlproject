import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import random
import re

from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.seq2seq import Seq2Seq

# ==========================================
# SEQ2SEQ TRAINING CONFIGURATION
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 15 
MAX_LENGTH = 50

# ==========================================
# 1. TRANSLATION VOCABULARY
# ==========================================
class Seq2SeqVocab:
    def __init__(self):
        # We include the specific routing token for the decoder
        self.word2index = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2, "<SOS_POLITE>": 3}
        self.index2word = {0: "<PAD>", 1: "<EOS>", 2: "<UNK>", 3: "<SOS_POLITE>"}
        self.n_words = 4

    def normalize_string(self, s):
        s = str(s).lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def add_sentence(self, sentence):
        for word in self.normalize_string(sentence).split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def sentence_to_tensor(self, sentence, start_token=None):
        sentence = self.normalize_string(sentence)
        indexes = []
        if start_token:
            indexes.append(self.word2index[start_token])
            
        for word in sentence.split(' ')[:MAX_LENGTH - 2]: # Leave room for SOS/EOS
            indexes.append(self.word2index.get(word, self.word2index["<UNK>"]))
            
        indexes.append(self.word2index["<EOS>"])
        return torch.tensor(indexes, dtype=torch.long)

# ==========================================
# 2. DATASET PREPARATION
# ==========================================
class ParadetoxDataset(Dataset):
    def __init__(self):
        print("Downloading Paradetox dataset...")
        # Load the dataset
        ds = load_dataset("s-nlp/paradetox", split="train")
        
        self.pairs = []
        self.vocab = Seq2SeqVocab()
        
        print("Building Seq2Seq vocabulary from pairs...")
        for row in ds:
            toxic = row['en_toxic_comment']
            polite = row['en_neutral_comment']
            
            self.vocab.add_sentence(toxic)
            self.vocab.add_sentence(polite)
            self.pairs.append((toxic, polite))
            
    def __len__(self): return len(self.pairs)
    
    def __getitem__(self, idx):
        toxic, polite = self.pairs[idx]
        # Source gets standard processing
        source_tensor = self.vocab.sentence_to_tensor(toxic)
        # Target gets the <SOS_POLITE> token at the very beginning
        target_tensor = self.vocab.sentence_to_tensor(polite, start_token="<SOS_POLITE>")
        return source_tensor, target_tensor

def collate_seq2seq(batch):
    sources, targets = zip(*batch)
    sources_padded = pad_sequence(sources, padding_value=0, batch_first=True)
    targets_padded = pad_sequence(targets, padding_value=0, batch_first=True)
    return sources_padded, targets_padded

# ==========================================
# 3. TRAINING LOOP
# ==========================================
def train():
    dataset = ParadetoxDataset()
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_seq2seq)
    
    print(f"Vocab Size: {dataset.vocab.n_words}")
    print(f"Total Parallel Sentences: {len(dataset)}")
    
    encoder = EncoderRNN(input_size=dataset.vocab.n_words, hidden_size=HIDDEN_SIZE).to(DEVICE)
    decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=dataset.vocab.n_words, max_length=MAX_LENGTH).to(DEVICE)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Ignore <PAD> so the model doesn't waste time learning empty space
    criterion = nn.CrossEntropyLoss(ignore_index=0) 
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for batch_idx, (source, target) in enumerate(dataloader):
            source, target = source.to(DEVICE), target.to(DEVICE)
            
            optimizer.zero_grad()
            
            # The style token is the first element of the target sequence
            start_tokens = target[:, 0].unsqueeze(1) 
            
            # Forward pass
            output = model(source, target, start_tokens)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            target = target[:, 1:].reshape(-1)
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch: {epoch+1}/{EPOCHS} | Average Loss: {avg_loss:.4f}")

    print("Training Complete. Saving Seq2Seq models...")
    torch.save(model.state_dict(), "tone_translator.pth")
    torch.save(dataset.vocab, "seq2seq_vocab.pth")

if __name__ == "__main__":
    train()
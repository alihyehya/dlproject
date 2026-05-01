import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# A small toy dataset: (Casual, Professional)
TOY_DATA = [
    ("hey whats up", "Hello, how are you?"),
    ("this code is trash", "This code requires significant refactoring."),
    ("my bad i forgot", "I apologize for the oversight."),
    ("fix this bug asap", "Please address this issue at your earliest convenience."),
    ("im gonna be late", "I will be arriving later than expected.")
]

import re

class Vocabulary:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.index2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.n_words = 4  # Count SOS, EOS, PAD, UNK

    def normalize_string(self, s):
        """Lowercases, trims, and removes non-letter characters"""
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def add_sentence(self, sentence):
        sentence = self.normalize_string(sentence)
        for word in sentence.split(' '):
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def sentence_to_tensor(self, sentence):
        """Converts a sentence to a tensor of token IDs"""
        sentence = self.normalize_string(sentence)
        indexes = [self.word2index.get(word, self.word2index["<UNK>"]) for word in sentence.split(' ')]
        indexes.append(self.word2index["<EOS>"]) # Always append EOS
        return torch.tensor(indexes, dtype=torch.long)
    
class ToneDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.source_vocab = Vocabulary()
        self.target_vocab = Vocabulary()
        
        # Build the vocabularies
        for casual, formal in self.pairs:
            self.source_vocab.add_sentence(casual)
            self.target_vocab.add_sentence(formal)
            
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        casual, formal = self.pairs[idx]
        casual_tensor = self.source_vocab.sentence_to_tensor(casual)
        formal_tensor = self.target_vocab.sentence_to_tensor(formal)
        return casual_tensor, formal_tensor

def collate_batch(batch):
    """Pads batches of sentences to the length of the longest sentence."""
    source_list, target_list = [], []
    for src, tgt in batch:
        source_list.append(src)
        target_list.append(tgt)
        
    # Pad sequences with 0 (<PAD>)
    source_padded = pad_sequence(source_list, padding_value=0, batch_first=True)
    target_padded = pad_sequence(target_list, padding_value=0, batch_first=True)
    
    return source_padded, target_padded

# --- Testing Phase 1 ---
if __name__ == "__main__":
    # Initialize Dataset
    dataset = ToneDataset(TOY_DATA)
    print(f"Source Vocab Size: {dataset.source_vocab.n_words}")
    print(f"Target Vocab Size: {dataset.target_vocab.n_words}")
    
    # Initialize DataLoader
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_batch)
    
    # Grab one batch
    source_batch, target_batch = next(iter(dataloader))
    print("\nSource Batch Tensor Shape:", source_batch.shape) # (batch_size, max_seq_length)
    print("Source Batch Tensor:\n", source_batch)
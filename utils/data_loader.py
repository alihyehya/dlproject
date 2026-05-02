import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import re

# Mocking a parallel dataset you would build using an LLM to generate the targets
# Format: (Source Text, Target Text, Source Class ID, Target Style Token)
# Classes: 0: Toxic, 1: Negative, 2: Informal
MULTI_DATA = [
    ("you are incredibly stupid", "I respectfully disagree with your approach.", 0, "<SOS_POLITE>"),
    ("this food is actual garbage", "The meal did not meet my expectations.", 1, "<SOS_POSITIVE>"),
    ("sup bro gotta go", "Hello, I must depart now.", 2, "<SOS_FORMAL>")
]

class Vocabulary:
    def __init__(self):
        # Added special style tokens
        self.word2index = {
            "<PAD>": 0, "<EOS>": 1, "<UNK>": 2, 
            "<SOS_POLITE>": 3, "<SOS_POSITIVE>": 4, "<SOS_FORMAL>": 5
        }
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 6

    def normalize_string(self, s):
        s = s.lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def add_sentence(self, sentence):
        for word in self.normalize_string(sentence).split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def sentence_to_tensor(self, sentence, start_token=None, max_len=50):
        sentence = self.normalize_string(sentence)
        indexes = []
        if start_token:
            indexes.append(self.word2index[start_token])
            
        # Add words, but leave room for EOS token
        for word in sentence.split(' ')[:max_len-2]:
            indexes.append(self.word2index.get(word, self.word2index["<UNK>"]))
            
        indexes.append(self.word2index["<EOS>"])
        return torch.tensor(indexes, dtype=torch.long)

class MultiToneDataset(Dataset):
    def __init__(self, data_tuples):
        self.data = data_tuples
        self.vocab = Vocabulary() # Using shared vocab for simplicity in multi-task
        
        for src, tgt, _, _ in self.data:
            self.vocab.add_sentence(src)
            self.vocab.add_sentence(tgt)
            
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        src, tgt, class_id, style_token = self.data[idx]
        src_tensor = self.vocab.sentence_to_tensor(src)
        tgt_tensor = self.vocab.sentence_to_tensor(tgt, start_token=style_token)
        style_idx = self.vocab.word2index[style_token]
        return src_tensor, tgt_tensor, torch.tensor(class_id, dtype=torch.long), torch.tensor([style_idx], dtype=torch.long)

def collate_batch_multi(batch):
    src_list, tgt_list, class_list, style_list = [], [], [], []
    for src, tgt, cls, style in batch:
        src_list.append(src)
        tgt_list.append(tgt)
        class_list.append(cls)
        style_list.append(style)
        
    src_padded = pad_sequence(src_list, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_list, padding_value=0, batch_first=True)
    classes = torch.stack(class_list)
    styles = torch.stack(style_list)
    
    return src_padded, tgt_padded, classes, styles
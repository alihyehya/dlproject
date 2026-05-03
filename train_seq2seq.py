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
HIDDEN_SIZE = 512        # Increased from 256 for better capacity
BATCH_SIZE = 64          # Increased from 32 for faster training (fewer steps/epoch)
LEARNING_RATE = 0.003    # Higher LR to converge faster
EPOCHS = 10         # Fewer epochs, but scheduled TF decay compensates
MAX_LENGTH = 50

# ==========================================
# 1. TRANSLATION VOCABULARY
# ==========================================
class Seq2SeqVocab:
    def __init__(self):
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
            if word and word not in self.word2index:
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

    def sentence_to_tensor(self, sentence, start_token=None):
        sentence = self.normalize_string(sentence)
        indexes = []
        if start_token:
            indexes.append(self.word2index[start_token])
        for word in sentence.split(' ')[:MAX_LENGTH - 2]:
            if word:
                indexes.append(self.word2index.get(word, self.word2index["<UNK>"]))
        indexes.append(self.word2index["<EOS>"])
        return torch.tensor(indexes, dtype=torch.long)


# ==========================================
# 2. DATASET PREPARATION
# ==========================================
class ParadetoxDataset(Dataset):
    def __init__(self):
        print("Downloading Paradetox dataset...")
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        toxic, polite = self.pairs[idx]
        source_tensor = self.vocab.sentence_to_tensor(toxic)
        # Target starts with <SOS_POLITE> routing token
        target_tensor = self.vocab.sentence_to_tensor(polite, start_token="<SOS_POLITE>")
        return source_tensor, target_tensor


def collate_seq2seq(batch):
    sources, targets = zip(*batch)
    sources_padded = pad_sequence(sources, padding_value=0, batch_first=True)
    targets_padded = pad_sequence(targets, padding_value=0, batch_first=True)
    return sources_padded, targets_padded


# ==========================================
# 3. SCHEDULED TEACHER FORCING FORWARD PASS
#    (Replaces the old Seq2Seq.forward if you
#     want to keep it self-contained here)
# ==========================================
def forward_with_teacher_forcing(model, source, target, start_tokens,
                                  teacher_forcing_ratio, device):
    """
    Custom forward that does scheduled teacher forcing token-by-token.
    Returns output tensor of shape (batch, target_len, vocab_size).
    """
    batch_size  = source.shape[0]
    target_len  = target.shape[1]
    output_size = model.decoder.output_size   # decoder must expose this attribute

    outputs = torch.zeros(batch_size, target_len, output_size).to(device)

    encoder_outputs, hidden = model.encoder(source)

    decoder_input = start_tokens   # shape: (batch, 1)

    for t in range(1, target_len):
        output, hidden, _ = model.decoder(decoder_input, hidden, encoder_outputs)
        outputs[:, t] = output   # output shape: (batch, vocab_size)

        # Decide: teacher force (use ground-truth) or free-run (use own prediction)
        use_teacher  = random.random() < teacher_forcing_ratio
        top1         = output.argmax(1).unsqueeze(1)          # (batch, 1)
        decoder_input = target[:, t].unsqueeze(1) if use_teacher else top1

    return outputs


# ==========================================
# 4. TRAINING LOOP
# ==========================================
def train():
    dataset    = ParadetoxDataset()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_seq2seq,
        num_workers=2,          # parallel data loading
        pin_memory=True         # faster GPU transfer
    )

    print(f"Device      : {DEVICE}")
    print(f"Vocab Size  : {dataset.vocab.n_words}")
    print(f"Total Pairs : {len(dataset)}")
    print(f"Hidden Size : {HIDDEN_SIZE}  |  Batch: {BATCH_SIZE}  |  LR: {LEARNING_RATE}")

    encoder = EncoderRNN(
        input_size=dataset.vocab.n_words,
        hidden_size=HIDDEN_SIZE
    ).to(DEVICE)

    decoder = AttnDecoderRNN(
        hidden_size=HIDDEN_SIZE,
        output_size=dataset.vocab.n_words,
        max_length=MAX_LENGTH
    ).to(DEVICE)

    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)   # ignore <PAD>

    # Optional: cosine LR decay for smoother convergence
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0

        # Scheduled teacher forcing: 1.0 at epoch 0, decays to 0.0 at last epoch
        tf_ratio = max(0.0, 1.0 - (epoch / EPOCHS))
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]  Teacher-Forcing Ratio: {tf_ratio:.2f}")

        for batch_idx, (source, target) in enumerate(dataloader):
            source, target = source.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            # The style/routing token is the first column of the target
            start_tokens = target[:, 0].unsqueeze(1)   # (batch, 1)

            # ---- Scheduled teacher-forcing forward pass ----
            output = forward_with_teacher_forcing(
                model, source, target, start_tokens,
                teacher_forcing_ratio=tf_ratio,
                device=DEVICE
            )

            # Align output & target (skip the SOS token at position 0)
            output_dim = output.shape[-1]
            output_flat = output[:, 1:].reshape(-1, output_dim)
            target_flat = target[:, 1:].reshape(-1)

            loss = criterion(output_flat, target_flat)
            loss.backward()

            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 50 == 0:
                print(f"  Step {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f">>> Epoch {epoch+1}/{EPOCHS} Complete | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")

    print("\nTraining Complete. Saving Seq2Seq models...")
    torch.save(model.state_dict(), "tone_translator.pth")
    torch.save(dataset.vocab, "seq2seq_vocab.pth")
    print("Saved: tone_translator.pth  |  seq2seq_vocab.pth")


if __name__ == "__main__":
    train()
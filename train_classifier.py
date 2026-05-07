import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import re

from models.classifier import ToneClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256
EMBEDDING_DIM = 128
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
SAMPLES_PER_CLASS = 10000


class ClassifierVocab:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.n_words = 2

    def normalize_string(self, s):
        s = str(s).lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s

    def add_sentence(self, sentence):
        for word in self.normalize_string(sentence).split(' '):
            if word not in self.word2index:
                self.word2index[word] = self.n_words
                self.n_words += 1

    def sentence_to_tensor(self, sentence, max_len=50):
        sentence = self.normalize_string(sentence)
        indexes = [self.word2index.get(word, self.word2index["<UNK>"]) for word in sentence.split(' ')[:max_len]]
        return torch.tensor(indexes, dtype=torch.long)


def prepare_data():
    print(f"Downloading datasets and sampling {SAMPLES_PER_CLASS} per class...")

    jigsaw = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train")
    toxic_texts = jigsaw.filter(lambda x: x['toxic'] == 1)['comment_text'][:SAMPLES_PER_CLASS]
    polite_texts = jigsaw.filter(lambda x: x['toxic'] == 0)['comment_text'][:SAMPLES_PER_CLASS]

    yelp = load_dataset("fancyzhx/yelp_polarity", split="train")
    negative_texts = yelp.filter(lambda x: x['label'] == 0)['text'][:SAMPLES_PER_CLASS]
    positive_texts = yelp.filter(lambda x: x['label'] == 1)['text'][:SAMPLES_PER_CLASS]

    dataset_tuples = []
    dataset_tuples.extend([(text, 0) for text in toxic_texts])
    dataset_tuples.extend([(text, 1) for text in polite_texts])
    dataset_tuples.extend([(text, 2) for text in negative_texts])
    dataset_tuples.extend([(text, 3) for text in positive_texts])

    return dataset_tuples


class TextClassificationDataset(Dataset):
    def __init__(self, data_tuples):
        self.data = data_tuples
        self.vocab = ClassifierVocab()

        print("Building vocabulary...")
        for text, _ in self.data:
            self.vocab.add_sentence(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        text_tensor = self.vocab.sentence_to_tensor(text)
        return text_tensor, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    texts, labels = zip(*batch)
    texts_padded = pad_sequence(texts, padding_value=0, batch_first=True)
    labels = torch.stack(labels)
    return texts_padded, labels


def train():
    data_tuples = prepare_data()
    dataset = TextClassificationDataset(data_tuples)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    print(f"Vocab Size: {dataset.vocab.n_words}")
    print(f"Total Samples: {len(dataset)}")

    model = ToneClassifier(
        vocab_size=dataset.vocab.n_words,
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_classes=4
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch_texts, batch_labels in dataloader:
            batch_texts, batch_labels = batch_texts.to(DEVICE), batch_labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += batch_labels.size(0)
            correct_predictions += (predicted == batch_labels).sum().item()

        avg_loss = total_loss / len(dataloader)
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"Epoch: {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.2f}%")

    print("Training Complete. Saving model...")
    torch.save(model.state_dict(), "tone_classifier.pth")
    torch.save(dataset.vocab, "classifier_vocab.pth")


if __name__ == "__main__":
    train()
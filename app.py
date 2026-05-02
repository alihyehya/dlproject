import torch
import gradio as gr
import re
import sys

from models.classifier import ToneClassifier
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256

# ==========================================
# 1. LOAD CLASSIFIER AND VOCAB
# ==========================================
class ClassifierVocab:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<UNK>": 1}
        self.n_words = 2
    def normalize_string(self, s):
        s = str(s).lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    def sentence_to_tensor(self, sentence, max_len=50):
        sentence = self.normalize_string(sentence)
        indexes = [self.word2index.get(word, self.word2index["<UNK>"]) for word in sentence.split(' ')[:max_len]]
        return torch.tensor(indexes, dtype=torch.long)

sys.modules['__main__'].ClassifierVocab = ClassifierVocab
classifier_vocab = torch.load("classifier_vocab.pth", map_location=device, weights_only=False)

classifier_model = ToneClassifier(vocab_size=classifier_vocab.n_words, embedding_dim=128, hidden_size=256, num_classes=4).to(device)
classifier_model.load_state_dict(torch.load("tone_classifier.pth", map_location=device))
classifier_model.eval() 

CLASS_LABELS = {0: "Toxic 🤬", 1: "Polite 🤝", 2: "Negative 📉", 3: "Positive 🌟"}

# ==========================================
# 2. LOAD SEQ2SEQ AND VOCAB
# ==========================================
class Seq2SeqVocab:
    def __init__(self):
        self.word2index = {"<PAD>": 0, "<EOS>": 1, "<UNK>": 2, "<SOS_POLITE>": 3}
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.n_words = 4
    def normalize_string(self, s):
        s = str(s).lower().strip()
        s = re.sub(r"([.!?])", r" \1", s)
        s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
        return s
    def sentence_to_tensor(self, sentence, start_token=None):
        sentence = self.normalize_string(sentence)
        indexes = []
        if start_token: indexes.append(self.word2index[start_token])
        for word in sentence.split(' ')[:50]: indexes.append(self.word2index.get(word, self.word2index["<UNK>"]))
        indexes.append(self.word2index["<EOS>"])
        return torch.tensor(indexes, dtype=torch.long)

sys.modules['__main__'].Seq2SeqVocab = Seq2SeqVocab
try:
    seq2seq_vocab = torch.load("seq2seq_vocab.pth", map_location=device, weights_only=False)
    encoder = EncoderRNN(input_size=seq2seq_vocab.n_words, hidden_size=HIDDEN_SIZE).to(device)
    decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=seq2seq_vocab.n_words).to(device)
    translator_model = Seq2Seq(encoder, decoder, device).to(device)
    translator_model.load_state_dict(torch.load("tone_translator.pth", map_location=device))
    translator_model.eval()
    print("✅ Full Pipeline Loaded Successfully.")
except FileNotFoundError:
    print("⚠️ Warning: Seq2Seq models not found. Run train_seq2seq.py first.")

# ==========================================
# 3. PIPELINE LOGIC
# ==========================================
def process_text(user_text):
    if not user_text.strip(): return "Please enter text.", "", "N/A"
        
    with torch.no_grad():
        # --- STAGE 1: CLASSIFY ---
        class_tensor = classifier_vocab.sentence_to_tensor(user_text).unsqueeze(0).to(device)
        class_logits = classifier_model(class_tensor)
        predicted_id = class_logits.argmax(1).item()
        detected_tone = CLASS_LABELS.get(predicted_id, "Unknown")
        
        # --- STAGE 2: TRANSLATE (ONLY IF TOXIC) ---
        if predicted_id == 0:  # 0 is Toxic
            trans_tensor = seq2seq_vocab.sentence_to_tensor(user_text).unsqueeze(0).to(device)
            encoder_outputs, hidden = translator_model.encoder(trans_tensor)
            
            # Start decoder with <SOS_POLITE>
            sos_token_id = seq2seq_vocab.word2index["<SOS_POLITE>"]
            decoder_input = torch.tensor([[sos_token_id]], device=device)
            decoded_words = []
            
            for t in range(50):
                output, hidden, _ = translator_model.decoder(decoder_input, hidden, encoder_outputs)
                top1_id = output.argmax(1).item()
                
                if top1_id == seq2seq_vocab.word2index["<EOS>"]: break
                decoded_words.append(seq2seq_vocab.index2word[top1_id])
                decoder_input = torch.tensor([[top1_id]], device=device)
                
            final_translation = " ".join(decoded_words)
            action_taken = "Rewritten for politeness."
        else:
            final_translation = user_text # Leave it alone if it's fine!
            action_taken = "No change needed."
            
        return detected_tone, action_taken, final_translation

# ==========================================
# 4. UI BUILDER
# ==========================================
interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(lines=3, placeholder="Type a message...", label="Input Text"),
    outputs=[
        gr.Textbox(label="1. Detected Tone"), 
        gr.Textbox(label="2. Pipeline Action"),
        gr.Textbox(label="3. Final Output (Detoxified if necessary)")
    ],
    title="Intelligent Tone Pipeline",
    description="Classifies the tone into 4 categories. If the text is classified as 'Toxic 🤬', the deep learning translator automatically rewrites it to be polite.",
    theme="default"
)

if __name__ == "__main__":
    interface.launch(share=False)
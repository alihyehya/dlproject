import torch
import gradio as gr
import re
import sys

from models.classifier import ToneClassifier
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.seq2seq import Seq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 512   # Must match the value used during training


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
        indexes = [
            self.word2index.get(word, self.word2index["<UNK>"])
            for word in sentence.split(' ')[:max_len]
            if word
        ]
        return torch.tensor(indexes, dtype=torch.long)


sys.modules['__main__'].ClassifierVocab = ClassifierVocab
classifier_vocab = torch.load("classifier_vocab.pth", map_location=device, weights_only=False)

classifier_model = ToneClassifier(
    vocab_size=classifier_vocab.n_words,
    embedding_dim=128,
    hidden_size=256,
    num_classes=4
).to(device)
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
        if start_token:
            indexes.append(self.word2index[start_token])
        for word in sentence.split(' ')[:50]:
            if word:
                indexes.append(self.word2index.get(word, self.word2index["<UNK>"]))
        indexes.append(self.word2index["<EOS>"])
        return torch.tensor(indexes, dtype=torch.long)


sys.modules['__main__'].Seq2SeqVocab = Seq2SeqVocab

seq2seq_loaded = False
try:
    seq2seq_vocab = torch.load("seq2seq_vocab.pth", map_location=device, weights_only=False)

    encoder = EncoderRNN(
        input_size=seq2seq_vocab.n_words,
        hidden_size=HIDDEN_SIZE
    ).to(device)

    decoder = AttnDecoderRNN(
        hidden_size=HIDDEN_SIZE,
        output_size=seq2seq_vocab.n_words
    ).to(device)

    translator_model = Seq2Seq(encoder, decoder, device).to(device)
    translator_model.load_state_dict(torch.load("tone_translator.pth", map_location=device))
    translator_model.eval()

    seq2seq_loaded = True
    print("✅ Full Pipeline Loaded Successfully.")
except FileNotFoundError:
    print("⚠️  Warning: Seq2Seq models not found. Run train_seq2seq.py first.")


# ==========================================
# 3. BEAM SEARCH DECODER
#    Replaces greedy argmax — dramatically
#    improves output quality.
# ==========================================
def beam_search_decode(vocab, model, enc_input_tensor,
                        beam_width=5, max_len=50):
    """
    Beam search decoding for the Seq2Seq translator.

    Args:
        vocab          : Seq2SeqVocab instance
        model          : trained Seq2Seq model (eval mode)
        enc_input_tensor: (1, seq_len) LongTensor on device
        beam_width     : number of beams to maintain
        max_len        : maximum output tokens

    Returns:
        Decoded string (best beam)
    """
    eos_id = vocab.word2index["<EOS>"]
    sos_id = vocab.word2index["<SOS_POLITE>"]

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(enc_input_tensor)

    # Each beam is a tuple: (cumulative_log_prob, token_id_list, hidden_state)
    beams     = [(0.0, [sos_id], hidden)]
    completed = []

    for _ in range(max_len):
        if not beams:
            break

        new_beams = []
        for score, tokens, h in beams:
            # If this beam already ended, move to completed
            if tokens[-1] == eos_id:
                completed.append((score, tokens))
                continue

            dec_input = torch.tensor([[tokens[-1]]], device=device)

            with torch.no_grad():
                output, new_h, _ = model.decoder(dec_input, h, encoder_outputs)

            # output shape: (1, vocab_size)
            log_probs = torch.log_softmax(output, dim=1)
            top_log_probs, top_ids = log_probs.topk(beam_width, dim=1)

            for i in range(beam_width):
                tok_id    = top_ids[0][i].item()
                new_score = score + top_log_probs[0][i].item()
                new_beams.append((new_score, tokens + [tok_id], new_h))

        # Keep only the top beam_width candidates
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

        # Early stop if we have enough completed beams
        if len(completed) >= beam_width:
            break

    # Collect any unfinished beams as well
    # completed items are 2-tuples (score, tokens)
    # active beams are 3-tuples (score, tokens, hidden) — normalise to 2-tuple
    all_candidates = completed + [(s, t) for s, t, *_ in beams]

    # Pick the highest-scoring sequence
    best_score, best_tokens = sorted(
        all_candidates, key=lambda x: x[0], reverse=True
    )[0]

    # Convert token IDs → words, stripping SOS and EOS
    words = [
        vocab.index2word[tok]
        for tok in best_tokens[1:]          # skip <SOS_POLITE>
        if tok != eos_id and tok in vocab.index2word
    ]

    return " ".join(words) if words else "[empty translation]"


# ==========================================
# 4. PIPELINE LOGIC
# ==========================================
def process_text(user_text):
    if not user_text.strip():
        return "Please enter some text.", "", "N/A"

    with torch.no_grad():
        # --- STAGE 1: CLASSIFY ---
        class_tensor = classifier_vocab.sentence_to_tensor(user_text).unsqueeze(0).to(device)
        class_logits  = classifier_model(class_tensor)
        predicted_id  = class_logits.argmax(1).item()
        detected_tone = CLASS_LABELS.get(predicted_id, "Unknown")

        # --- STAGE 2: TRANSLATE (ONLY IF TOXIC) ---
        if predicted_id == 0:  # Toxic
            if not seq2seq_loaded:
                return detected_tone, "Seq2Seq model not loaded.", user_text

            trans_tensor = seq2seq_vocab.sentence_to_tensor(user_text).unsqueeze(0).to(device)

            # Beam search — much better than greedy argmax
            final_translation = beam_search_decode(
                vocab=seq2seq_vocab,
                model=translator_model,
                enc_input_tensor=trans_tensor,
                beam_width=5,
                max_len=50
            )
            action_taken = "Rewritten for politeness (beam search, width=5)."

        else:
            final_translation = user_text
            action_taken      = "No change needed."

    return detected_tone, action_taken, final_translation


# ==========================================
# 5. GRADIO UI
# ==========================================
interface = gr.Interface(
    fn=process_text,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Type a message here...",
        label="Input Text"
    ),
    outputs=[
        gr.Textbox(label="1. Detected Tone"),
        gr.Textbox(label="2. Pipeline Action"),
        gr.Textbox(label="3. Final Output (Detoxified if necessary)")
    ],
    title="Intelligent Tone Pipeline",
    description=(
        "Classifies the tone into 4 categories. "
        "If the text is classified as 'Toxic 🤬', the deep learning translator "
        "automatically rewrites it to be polite using beam search decoding."
    ),
    theme="default",
    examples=[
        ["Maybe if you had half a brain, you'd understand this incredibly simple concept."],
        ["I appreciate your effort on this project, great work!"],
        ["This is absolutely terrible and I hate it."],
        ["Thank you for taking the time to explain this."],
    ]
)

if __name__ == "__main__":
    interface.launch(share=False)
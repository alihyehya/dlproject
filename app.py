import torch
import gradio as gr
from utils.data_loader import ToneDataset, TOY_DATA, collate_batch
from models.encoder import EncoderRNN
from models.decoder import AttnDecoderRNN
from models.seq2seq import Seq2Seq

# 1. Setup and Load the Saved Model (Mocking the data load for the Toy Data)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_SIZE = 256

# Rebuild vocabs so we know the token IDs
dataset = ToneDataset(TOY_DATA)
input_vocab = dataset.source_vocab
output_vocab = dataset.target_vocab

# Initialize architecture and load weights
encoder = EncoderRNN(input_size=input_vocab.n_words, hidden_size=HIDDEN_SIZE).to(device)
decoder = AttnDecoderRNN(hidden_size=HIDDEN_SIZE, output_size=output_vocab.n_words).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

# Load the weights we saved at the end of train.py
# (Ensure train.py has actually been run first to create this file!)
try:
    model.load_state_dict(torch.load("tone_translator.pth"))
    model.eval() # CRITICAL: Disables dropout for deterministic inference
except FileNotFoundError:
    print("Warning: tone_translator.pth not found. The model will output random garbage.")

# 2. The Core Translation Logic
def translate_sentence(casual_text):
    with torch.no_grad(): # Disables gradient tracking to save memory/compute
        # A. Preprocess the input string
        input_tensor = input_vocab.sentence_to_tensor(casual_text).unsqueeze(0).to(device)
        
        # B. Pass through the Encoder
        encoder_outputs, hidden = model.encoder(input_tensor)
        
        # C. Initialize the Decoder's first input with <SOS> (Start of Sequence)
        decoder_input = torch.tensor([[model.sos_token_id]], device=device)
        
        decoded_words = []
        
        # D. Autonomously generate words loop
        for t in range(50): # Hard limit of 50 words to prevent infinite loops
            output, hidden, _ = model.decoder(decoder_input, hidden, encoder_outputs)
            
            # Get the ID of the word with the highest probability
            top1_id = output.argmax(1).item()
            
            # If the model predicts <EOS>, the sentence is finished
            if top1_id == output_vocab.word2index["<EOS>"]:
                break
                
            # Convert ID back to a string and append to our list
            decoded_word = output_vocab.index2word[top1_id]
            decoded_words.append(decoded_word)
            
            # The current prediction becomes the input for the next timestep
            decoder_input = torch.tensor([[top1_id]], device=device)
            
        # E. Join the list of words into a clean sentence
        return " ".join(decoded_words)
    
    # 3. Build the UI
interface = gr.Interface(
    fn=translate_sentence, # The function we just wrote
    inputs=gr.Textbox(
        lines=2, 
        placeholder="Type a casual message here...", 
        label="Input: Casual Text"
    ),
    outputs=gr.Textbox(
        label="Output: Professional Tone"
    ),
    title="Tone Translation Copilot",
    description="A Seq2Seq Deep Learning model with Attention. Translates casual chat into professional language.",
    theme="default"
)

if __name__ == "__main__":
    # This launches the web server
    interface.launch(share=False)
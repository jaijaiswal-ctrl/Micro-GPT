import streamlit as st
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# 1. CONFIGURATION (Must match your gpt.py exactly)
# ==============================================================================
class Config:
    batch_size = 1
    block_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 4
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ==============================================================================
# 2. THE ARCHITECTURE (The "Shape" of the Brain)
# ==============================================================================
class TextProcessor:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, l): return ''.join([self.itos[i] for i in l])

class AttentionHead(nn.Module):
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
    def forward(self, x):
        B, T, C = x.shape
        k, q = self.key(x), self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return F.softmax(wei, dim=-1) @ self.value(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(Config.n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)
    def forward(self, x):
        return self.proj(torch.cat([h(x) for h in self.heads], dim=-1))

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd))
    def forward(self, x): return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = MultiHeadAttention(n_head, n_embd // n_head)
        self.ffwd = FeedForward(n_embd)
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(Config.n_embd, Config.n_head) for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=Config.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        return logits, None
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ==============================================================================
# 3. STREAMLIT WEB INTERFACE
# ==============================================================================
st.set_page_config(page_title="Micro-GPT AI", page_icon="🚀")

# Style the page a bit
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 My First Mini-GPT")
st.markdown("### A character-level Transformer trained on my MacBook Pro!")

# We use cache_resource so the model only loads once and doesn't slow down the website
@st.cache_resource
def load_ai():
    # 1. Rebuild tokenizer from input.txt
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        tokenizer = TextProcessor(text)
        
        # 2. Load model and weights
        model = MiniGPT(tokenizer.vocab_size).to(Config.device)
        model.load_state_dict(torch.load('gpt_brain.pth', map_location=Config.device))
        model.eval()
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading AI: {e}")
        return None, None

# Initialize the AI
with st.spinner("Loading AI Brain..."):
    model, tokenizer = load_ai()

if model and tokenizer:
    st.divider()
    
    # User Input Area
    prompt = st.text_input("Enter a prompt:", placeholder="e.g. BRUTUS: I believe that")

    # Layout for settings
    col1, col2 = st.columns(2)
    with col1:
        temp = st.slider("Creativity (Temperature)", 0.1, 1.5, 0.8, 0.1)
    with col2:
        length = st.slider("Output Length", 10, 500, 100, 10)

    if st.button("Generate Response"):
        if prompt:
            with st.spinner("AI is thinking..."):
                # Process prompt
                context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=Config.device)
                # Generate text
                generated = model.generate(context, max_new_tokens=length, temperature=temp)[0].tolist()
                # Decode result
                result = tokenizer.decode(generated)
                # Show result in a nice box
                st.subheader("AI Response:")
                st.write(result)
        else:
            st.warning("Please enter a prompt first!")
else:
    st.error("Could not find gpt_brain.pth or input.txt. Please run gpt.py first!")

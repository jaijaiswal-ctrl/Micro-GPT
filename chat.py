import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# CONFIGURATION (Must be exactly the same as gpt.py)
# ==============================================================================
class Config:
    batch_size = 1
    block_size = 64
    n_embd = 128
    n_head = 4
    n_layer = 4
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ==============================================================================
# ARCHITECTURE (Must be exactly the same as gpt.py)
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
# CHAT LOGIC
# ==============================================================================
if __name__ == "__main__":
    # 1. Load the text to rebuild the tokenizer
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = TextProcessor(text)
    
    # 2. Initialize model and LOAD the saved brain
    model = MiniGPT(tokenizer.vocab_size).to(Config.device)
    model.load_state_dict(torch.load('gpt_brain.pth', map_location=Config.device))
    model.eval() # Set to evaluation mode
    
    print("\n🚀 AI Brain Loaded! You can now chat.")
    print("Type 'quit' to stop.\n")
    
    while True:
        user_input = input("Prompt: ")
        if user_input.lower() == 'quit': break
        
        # Encode user input
        context = torch.tensor([tokenizer.encode(user_input)], dtype=torch.long, device=Config.device)
        
        # Generate response
        generated = model.generate(context, max_new_tokens=100, temperature=0.8)[0].tolist()
        
        print("\nAI:", tokenizer.decode(generated))
        print("-" * 30)

import torch
import torch.nn as nn
from torch.nn import functional as F

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
class Config:
    """The most stable configuration for Apple Silicon to prevent OOM errors."""
    batch_size = 16       # Small batch = zero memory crashes
    block_size = 64       
    max_iters = 10000     
    learning_rate = 1e-3  
    n_embd = 128          
    n_head = 4            
    n_layer = 4           
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# ==============================================================================
# DATA PROCESSING
# ==============================================================================
class TextProcessor:
    """Handles character-level tokenization and dataset management."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        
    def encode(self, s):
        return [self.stoi[c] for c in s]
    
    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

# ==============================================================================
# TRANSFORMER ARCHITECTURE
# ==============================================================================
class AttentionHead(nn.Module):
    """Single head of self-attention."""
    def __init__(self, n_embd, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores (Scaled Dot-Product Attention)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        return wei @ self.value(x)

class MultiHeadAttention(nn.Module):
    """Parallel attention heads to capture diverse contextual relationships."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([AttentionHead(Config.n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(Config.n_embd, Config.n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """Position-wise feed-forward network for feature transformation."""
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """One Transformer block combining communication and computation."""
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Applying residual connections to maintain gradient flow
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT(nn.Module):
    """Decoder-only Transformer architecture for next-token prediction."""
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, Config.n_embd)
        self.position_embedding = nn.Embedding(Config.block_size, Config.n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(Config.n_embd, Config.n_head) for _ in range(Config.n_layer)])
        self.ln_f = nn.LayerNorm(Config.n_embd)
        self.lm_head = nn.Linear(Config.n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # The indentation below this line is critical!
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=Config.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # We use .reshape() to avoid the non-contiguous memory error
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -Config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================================================
# TRAINING ENGINE
# ==============================================================================
def train_model(text_data):
    processor = TextProcessor(text_data)
    
    # IMPORTANT: Keep data on CPU to save GPU memory
    data = torch.tensor(processor.encode(text_data), dtype=torch.long).to('cpu')
    
    # Keep the model on the GPU for speed
    model = MiniGPT(processor.vocab_size).to(Config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
    
    print(f"🚀 Stable-Mode Training started. Data: CPU | Model: {Config.device}")
    
    data_len = len(data) - Config.block_size
    
    for i in range(Config.max_iters):
        # 1. Sample indices on CPU
        ix = torch.randint(0, data_len, (Config.batch_size,))
        
        # 2. Slice and stack on CPU
        x_cpu = torch.stack([data[j:j+Config.block_size] for j in ix])
        y_cpu = torch.stack([data[j+1:j+Config.block_size+1] for j in ix])
        
        # 3. Move ONLY this small batch to GPU
        x = x_cpu.to(Config.device)
        y = y_cpu.to(Config.device)
        
        logits, loss = model(x, y)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        if i % 500 == 0:
            print(f"Iteration {i}/{Config.max_iters} | Loss: {loss.item():.4f}")
            
    return model, processor

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        with open('input.txt', 'r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f"Loaded {len(raw_text)} chars from input.txt")
    except FileNotFoundError:
        print("Error: input.txt not found.")
        exit()

    model, tokenizer = train_model(raw_text)
        # This saves the "weights" (the knowledge) to a file called gpt_brain.pth
    torch.save(model.state_dict(), 'gpt_brain.pth')
    print("\n✅ Brain saved to gpt_brain.pth!")

    
    # Test generation with a prompt
    prompt = "BRUTUS: " 
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=Config.device)
    generated = model.generate(context, max_new_tokens=100, temperature=0.8)[0].tolist()
    
    print("\n--- Generated Result ---")
    print(tokenizer.decode(generated))

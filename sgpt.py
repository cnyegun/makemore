import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Download Shakespeare first if needed
try:
    with open('shakespeare.txt', 'r') as f:
        text = f.read()
    print(f"Loaded {len(text)} characters")
except:
    print("Download with: wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    exit(1)

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}

# UPGRADED CONFIG FOR MORE INTELLIGENCE
block_size = 128      # Was 64 (2x longer memory - remembers whole sentences!)
batch_size = 16       # Reduced from 32 (trading batch for capacity)
n_embd = 192          # Was 64 (3x wider - bigger brain)
n_head = 6            # Was 4 (more attention heads)
n_layer = 5           # Was 3 (deeper network - 5 transformer blocks)
dropout = 0.2
learning_rate = 3e-4  # Lower LR for stability with bigger model
max_iters = 15000     # Was 5000 (actually train to convergence)
eval_interval = 1000

print(f"Config: {n_layer} layers, {n_embd} dims, {n_head} heads, {block_size} context")
print(f"Parameters: ~{sum([n_embd*vocab_size*2 + n_layer*(4*n_embd*n_embd + 2*n_embd*4*n_embd)])/1e6:.2f}M")

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),  # GELU > ReLU for transformers
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Pre-norm (more stable than post-norm)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmartGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        
        # Better initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
        return logits, loss
        
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Prepare data with train/val split
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

model = SmartGPT().to(device)
print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.1)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_interval)
        for k in range(eval_interval):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("Training smarter model...")
start_time = time.time()

for iter in range(max_iters):
    # Learning rate decay with cosine schedule
    if iter < 1000:
        lr = learning_rate * iter / 1000  # Warmup
    else:
        progress = (iter - 1000) / (max_iters - 1000)
        lr = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Prevent explosions
    optimizer.step()
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:5d} | train loss: {losses['train']:.4f} | val loss: {losses['val']:.4f} | lr: {lr:.6f}")

elapsed = time.time() - start_time
print(f"\nTraining completed in {elapsed:.1f} seconds ({max_iters/elapsed:.1f} iter/sec)")

# Generate with different temperatures
print("\n" + "="*60)
print("GENERATED TEXT (temperature=0.7):")
print("="*60)
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = model.generate(context, max_new_tokens=1000, temperature=0.7)[0].tolist()
print(''.join([itos[i] for i in generated]))

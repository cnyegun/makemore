import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} (AMD 780M)")

words = open("names.txt", "r").read().splitlines()
stoi = {chr(i): (i + 1 - ord("a")) for i in range(ord("a"), ord("z") + 1)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 5
emb_dim = 24
hidden_dim = 512
batch_size = 256
steps = 100000  # 20k steps!

X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        X.append(context.copy())
        Y.append(stoi[ch])
        context = context[1:] + [stoi[ch]]

X = torch.tensor(X, dtype=torch.long, device=device)
Y = torch.tensor(Y, dtype=torch.long, device=device)
n_samples = len(X)

C = torch.randn(27, emb_dim, device=device)
W1 = torch.randn(block_size * emb_dim, hidden_dim, device=device)
b1 = torch.randn(hidden_dim, device=device)
W2 = torch.randn(hidden_dim, 27, device=device)
b2 = torch.randn(27, device=device)

params = [C, W1, b1, W2, b2]
for p in params:
    p.requires_grad = True

optimizer = torch.optim.Adam(params, lr=0.01)

best_loss = float('inf')
best_step = 0

print(f"Training {steps} steps...")

for step in range(steps):
    ix = torch.randint(0, n_samples, (batch_size,), device=device)
    
    emb = C[X[ix]].view(batch_size, -1)
    h = torch.tanh(emb @ W1 + b1)
    logits = h @ W2 + b2
    
    # Label smoothing prevents overfitting/confident gibberish
    loss = F.cross_entropy(logits, Y[ix], label_smoothing=0.1)
    
    optimizer.zero_grad()
    loss.backward()
    
    # Cosine decay over 20k steps
    lr = 0.01 * 0.5 * (1 + np.cos(np.pi * step / steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    optimizer.step()
    
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_step = step
        # Save best model
        torch.save({
            'C': C, 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2,
            'step': step, 'loss': loss.item()
        }, 'best_model.pt')
    
    if step % 2000 == 0:
        print(f"step {step:5d}: loss = {loss.item():.4f} | lr = {lr:.6f} | best = {best_loss:.4f} @ {best_step}")

print(f"\nTraining done! Best loss: {best_loss:.4f} at step {best_step}")
print("Loading best model for generation...")

# Load best checkpoint
checkpoint = torch.load('best_model.pt')
C, W1, b1, W2, b2 = checkpoint['C'], checkpoint['W1'], checkpoint['b1'], checkpoint['W2'], checkpoint['b2']
print(f"Loaded model from step {checkpoint['step']} with loss {checkpoint['loss']:.4f}")

# Generation with multiple temperatures
print("\n=== Cool names (temp=0.5) ===")
for _ in range(10):
    out = ""
    context = [0] * block_size
    while True:
        x = torch.tensor([context], device=device)
        emb = C[x].view(1, -1)
        h = torch.tanh(emb @ W1 + b1)
        logits = h @ W2 + b2
        probs = (logits / 0.5).softmax(dim=-1)
        ix = torch.multinomial(probs, 1).item()
        if ix == 0 or len(out) > 20:
            break
        out += itos[ix]
        context = context[1:] + [ix]
    print(out)

print("\n=== Normal names (temp=0.8) ===")
for _ in range(10):
    out = ""
    context = [0] * block_size
    while True:
        x = torch.tensor([context], device=device)
        emb = C[x].view(1, -1)
        h = torch.tanh(emb @ W1 + b1)
        logits = h @ W2 + b2
        probs = (logits / 0.8).softmax(dim=-1)
        ix = torch.multinomial(probs, 1).item()
        if ix == 0 or len(out) > 20:
            break
        out += itos[ix]
        context = context[1:] + [ix]
    print(out)

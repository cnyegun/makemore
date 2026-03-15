import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import Adam
from tinygrad import Device
import numpy as np

# AMD GPU (ROCm/AMDGPU backend)
Device.DEFAULT = 'AMD'
print(f"Using device: {Device.DEFAULT}")

words = open("names.txt", "r").read().splitlines()
stoi = {chr(i): (i + 1 - ord("a")) for i in range(ord("a"), ord("z") + 1)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

block_size = 5
X, Y = [], []
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context[:])
        Y.append(ix)
        context = context[1:] + [ix]

# Move to AMD GPU
X = Tensor(X).to('AMD')
Y = Tensor(Y).to('AMD')

Tensor.manual_seed(6969)

emb_dim = 12
hidden_dim = 200

C = Tensor.randn(27, emb_dim).to('AMD')
W1 = Tensor.randn(block_size * emb_dim, hidden_dim).to('AMD')
b1 = Tensor.randn(hidden_dim).to('AMD')
W2 = Tensor.randn(hidden_dim, 27).to('AMD')
b2 = Tensor.randn(27).to('AMD')

params = [C, W1, b1, W2, b2]
optimizer = Adam(params, lr=0.01)

Tensor.training = True

batch_size = 128
n_samples = X.shape[0]

for n in range(5000):
    ix = np.random.randint(0, n_samples, size=(batch_size,))
    X_batch = X[Tensor(ix).to('AMD')]
    Y_batch = Y[Tensor(ix).to('AMD')]
    
    optimizer.zero_grad()
    emb = C[X_batch]
    h = emb.reshape(batch_size, -1).dot(W1) + b1
    h = h.tanh()
    logits = h.dot(W2) + b2
    loss = -logits.log_softmax()[Tensor.arange(batch_size).to('AMD'), Y_batch].mean()

    if n % 500 == 0:
        print(f"step {n}: loss = {loss.numpy().item():.4f}")

    loss.backward()
    optimizer.step()

print("Training done!")

for _ in range(20):
    out = ""
    context = [0] * block_size
    while True:
        emb = C[Tensor(context).to('AMD')]
        h = emb.reshape(-1).dot(W1) + b1
        h = h.tanh()
        logits = h.dot(W2) + b2
        probs = logits.softmax().cpu().numpy()
        
        ix = np.random.choice(27, p=probs)
        if ix == 0:
            break
        out += itos[ix]
        context = context[1:] + [ix]
    print(out)

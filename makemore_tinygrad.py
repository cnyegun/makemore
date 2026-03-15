import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn.optim import SGD
import numpy as np

words = open("names.txt", "r").read().splitlines()
stoi = {chr(i): (i + 1 - ord("a")) for i in range(ord("a"), ord("z") + 1)}
stoi["."] = 0
itos = {i: s for s, i in stoi.items()}

X, Y = [], []
block_size = 3
for w in words:
    context = [0] * block_size
    for ch in w + ".":
        ix = stoi[ch]
        X.append(context[:])  # Important: copy the list
        Y.append(ix)
        context = context[1:] + [ix]

X = Tensor(X)
Y = Tensor(Y)

print(f"Dataset size: {X.shape[0]}")  # If this is >5000, full-batch will OOM

Tensor.manual_seed(6969)
C = Tensor.randn(27, 2)
W1 = Tensor.randn(6, 100)
b1 = Tensor.randn(100)
W2 = Tensor.randn(100, 27)
b2 = Tensor.randn(27)

params = [C, W1, b1, W2, b2]
optimizer = SGD(params, lr=0.1)
Tensor.training = True

batch_size = 64  # Key: Process only 64 samples at a time
n_samples = X.shape[0]

for n in range(500):  # Can safely go back to 500 iterations
    # Random minibatch indices
    ix = np.random.randint(0, n_samples, size=(batch_size,))
    X_batch = X[ix]
    Y_batch = Y[ix]
    
    optimizer.zero_grad()
    
    # Forward pass on small batch: [64, 3, 2]
    emb = C[X_batch]
    h = emb.reshape(batch_size, -1).dot(W1) + b1
    h = h.tanh()
    logits = h.dot(W2) + b2
    
    # Efficient loss computation
    loss = logits.log_softmax().get_loss(Y_batch)
    
    if n % 20 == 0:
        print(f"step {n}: loss = {loss.numpy().item():.4f}")
    
    loss.backward()
    optimizer.step()

# Generation (unchanged)
for _ in range(10):
    out = ""
    context = [0] * block_size
    while True:
        emb = C[Tensor(context)]
        h = emb.reshape(-1).dot(W1) + b1
        h = h.tanh()
        logits = h.dot(W2) + b2
        probs = logits.softmax().numpy()
        ix = np.random.choice(27, p=probs)
        if ix == 0:
            break
        out += itos[ix]
        context = context[1:] + [ix]
    print(out)

import torch
from pathlib import Path
import torch.nn.functional as F

"""
Load words
"""
def load_words():
    words = Path(__file__).parent.joinpath("names.txt").read_text().splitlines()
    #
    print(f"num words:{len(words)}")
    print(f"first 10: {words[:10]}")
    print(f" min len: {min(len(w) for w in words)}, max len: {max(len(w) for w in words)}")
    return words


"""
Bigram count table:

    .set gives every unique character across all names - 26 lowercase letters 
    stoi maps each letter to an index, a->1, b>2, etc.
    We reserve index 0 for the special . token (start/end marker)
    itos is the reverse map, it is used to decode back to characters later
"""
def build_bigram_counts(words):
    chars = sorted(list(set("".join(words))))
    stoi = {s: i + 1 for i, s in enumerate(chars)}
    stoi["."] = 0
    itos = {i: s for s, i in stoi.items()}

    N = torch.zeros((27, 27), dtype=torch.int32)

    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            N[i1, i2] += 1

    print(N.shape)
    print(f"'.' -> 'e' count: {N[stoi['.'], stoi['e']].item()}")
    print(f"'e' -> 'm' count: {N[stoi['e'], stoi['m']].item()}")
    print(f"'a' -> '.' count: {N[stoi['a'], stoi['.']].item()}")
    return N, stoi, itos


def compute_probabilities(N):
    P = (N + 1).float()
    P /= P.sum(dim=1, keepdim=True)
    return P

"""
Caluclates loss, by iterating over every true bigram looks up predicted probability from P and sums probabilites, then flips to get negative log-likelihhood (negative). Lower average NLL means the model assigns higher probability to the real data.
"""
def report_loss(words, P, stoi):
    log_likelihood = 0.0
    n = 0
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            prob = P[i1, i2]
            log_likelihood += torch.log(prob)
            n += 1

    nll = -log_likelihood
    print(f"log likelihood: {log_likelihood.item():.4f}")
    print(f"negative log likelihood: {nll.item():.4f}")
    print(f"average nll (loss): {(nll / n).item():.4f}")

"""
Turns raw bigram couints into supervised data, each previous-char index maps to its next char label 
Returns two 1D, this is what NN will train on
"""
def build_bigram_dataset(words, stoi):
    xs, ys = [], []
    for w in words:
        chs = ["."] + list(w) + ["."]
        for ch1, ch2 in zip(chs, chs[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)
    return X, Y

"""
Initailize a learnable weight matrix, same shape as the count table, and optimize with manual gradient descent
Cross entropy gives negative log-likelihood loss directly; we just backprop and update 
"""
def train_bigram_linear(X, Y, num_steps=1000, learning_rate=50.0):
    g = torch.Generator().manual_seed(2147483647)
    W = torch.randn((27, 27), generator=g, requires_grad=True)
    for step in range(num_steps):
        logits = W[X]
        loss = F.cross_entropy(logits, Y)
        W.grad = None 
        loss.backward()
        W.data -= learning_rate * W.grad 
        if step % 100 == 0 or step == num_steps - 1:
            print(f"step {step}: loss {loss.item():.4f}")
    return W

"""
Use learned weights to compute average cross-entropy loss w/o tracking gradients.
"""
def evaluate_bigram_linear(W, X, Y):
    with torch.no_grad():
        logits = W[X]
        loss = F.cross_entropy(logits, Y)
    return loss.item()

def sample_bigram_linear(W, itos, num_samples=10, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    for _ in range(num_samples):
        out = []
        ix = 0 
        while True:
            logits = W[ix]
            probs = F.softmax(logits, dim=0)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break 
            out.append(itos[ix])
        print("".join(out))

def split_words(words, proportions=(0.8, 0.1, 0.1), seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    """
    Random permutation
    e.g ()! / ()()!
    """
    order = torch.randperm(len(words), generator=g).tolist()
    n_train = int(proportions[0] * len(words))
    n_dev = int(proportions[1] * len(words))
    train_idx = order[:n_train]
    dev_idx = order[n_train:n_train + n_dev]
    test_idx = order[n_train + n_dev:]
    train_words = [words[i] for i in train_idx]
    dev_words = [words[i] for i in dev_idx]
    test_words = [words[i] for i in test_idx]
    return train_words, dev_words, test_words

def build_data(words, stoi, block_size):
    """
    Builds input/output pairs for a given context length (block_size).
    The context starts filled with zeros (out . token) to represent "start of word 
    Each step shifts the window and appends the current character index; the label is the next character
    """
    X, Y = [], []
    for w in words:
        context = [0] * block_size 
        for ch in list(w) + ["."]:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]
    X = torch.tensor(X, dtype=torch.long)
    Y = torch.tensor(Y, dtype=torch.long)
    return X, Y

def init_mlp_parameters(vocab_size, block_size, n_emb=10, n_hidden=200, seed=2147483647):
    """
    seeds the RNG for reproducbiiltiy and creates the MLP parameters: embedding matrix C, hidden layer, weights/bias, outputs weights/bias
    uses small random inits; hidden layer applies He init scaled for tanh (5/3 factor).
    marks every tensor as requiring gradients so we can optimize them manually. vocab_size is 26, and block_size will be the context length.
    """
    g = torch.Generator().manual_seed(seed)
    C = torch.randn((vocab_size, n_emb), generator=g) * 0.01 
    W1 = torch.randn((block_size * n_emb, n_hidden), generator=g) * (5 / 3) / (block_size * n_emb) ** 0.5 
    b1 = torch.zeros(n_hidden)
    W2 = torch.randn((n_hidden, vocab_size), generator=g) * 0.01 
    b2 = torch.zeros(vocab_size)
    params = [C, W1, b1, W2, b2]
    for p in params: 
        p.requires_grad = True 
    return params

def mlp_forward(params, X):
    """
    Performs forward pass for MLP: embedding lookup, flatten, hidden tanh, and final logits 
    Takes the parameter list plus a batch of indices shaped(batch_size, block_size) and returns the raw logits
    """
    C, W1, b1, W2, b2 = params 
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2 
    return logits 

def train_mlp(params, X, Y, batch_size=32, max_steps=20000, learning_rate=0.1, seed=2147483647):
    C, W1, b1, W2, b2 = params 
    g = torch.Generator().manual_seed(seed)
    for step in range(max_steps):
        idx = torch.randint(0, X.shape[0], (batch_size,), generator=g)
        Xb = X[idx]
        Yb = Y[idx]
        logits = mlp_forward(params, Xb)
        loss = F.cross_entropy(logits, Yb)

        for p in params:
            p.grad = None 
        loss.backward()
        for p in params:
            p.data -= learning_rate * p.grad

        if step % 1000 == 0 or step == max_steps - 1:
            print(f"step {step}: loss {loss.item():.4f}")
    return params

def evaluate_mlp(params, X, Y):
    with torch.no_grad():
        logits = mlp_forward(params, X)
        loss = F.cross_entropy(logits, Y)
    return loss.item()

def sample_mlp(params, stoi, itos, block_size, num_samples=10, seed=2147483647):
    g = torch.Generator().manual_seed(seed)
    C, W1, b1, W2, b2 = params 
    for _ in range(num_samples):
        context = [0] * block_size 
        out = []
        while True:
            x = torch.tensor([context], dtype=torch.long)
            logits = mlp_forward(params, x)
            probs = F.softmax(logits, dim=1)
            ix = torch.multinomial(probs, num_samples=1, generator=g).item()
            if ix == 0:
                break 
            out.append(itos[ix])
            context = context[1:] + [ix]
        print("".join(out))

def main():
    words = load_words()
    N, stoi, itos = build_bigram_counts(words)
    P = compute_probabilities(N)
    report_loss(words, P, stoi)

    X, Y = build_bigram_dataset(words, stoi)
    W = train_bigram_linear(X, Y)
    loss = evaluate_bigram_linear(W, X, Y)
    print(f"trained bigram loss: {loss:.4f}")
    sample_bigram_linear(W, itos)


if __name__ == "__main__":
    main()

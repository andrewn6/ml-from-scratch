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

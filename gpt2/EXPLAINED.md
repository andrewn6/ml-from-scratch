# GPT-2, explained

A line-by-line walkthrough of `main.py`, in the style of 3blue1brown's neural network
and transformer videos. Geometry first, code second.

Part I covers the architecture (what the model is).
Part II covers the training run (what happens over 19,073 steps).

Line numbers refer to `gpt2/main.py`.

---

# Part I: the architecture

## 0. The whole thing in one sentence

A stack of 12 identical blocks reads a 1024-slot conveyor belt of 768-dimensional
vectors, and each block does exactly two things to it: **move information sideways
between positions** (attention), then **think about each position on its own** (MLP).
At the end you dot every vector against every word in the vocabulary and softmax.

Everything else in the file is plumbing: getting bytes off disk, keeping numbers in a
healthy range, and spreading the work across GPUs.

## 1. The four numbers to hold in your head

```
B = 64      batch: how many independent sequences at once
T = 1024    time: context window, how many tokens per sequence
C = 768     channels: the width of the residual stream (n_embd)
V = 50304   vocab
nh = 12     heads,  hs = C // nh = 64  head size
```

Every tensor in the forward pass is some rearrangement of `(B, T, C)`. If you ever lose
the plot, ask "what are B, T, C right now" and you are back.

## 2. Config (lines 16 to 47)

```python
block_size = 256      # line 16
n_embd = 384          # line 20
dropout = 0.2         # line 23
```

**These lines are dead.** Leftovers from the makemore/nanoGPT lesson. Nothing reads
them. The real config is the dataclass:

```python
@dataclass
class GPTConfig:                  # line 41
    block_size: int = 1024        # max sequence length
    vocab_size: int = 50257       # 50,000 BPE merges + 256 raw bytes + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

That vocab number is worth unpacking. GPT-2's tokenizer starts with 256 tokens (every
possible byte, so nothing is ever unrepresentable), runs 50,000 byte-pair merges on top,
and adds one special token `<|endoftext|>` = 50256. 256 + 50000 + 1 = 50257.

The real hyperparameters:

```python
max_lr = 6e-4                 # line 25, from the GPT-3 paper's 124M row
min_lr = max_lr * 0.1         # decay to 10%, not to 0
warmup_steps = 715            # 375M warmup tokens / 524288 tokens per step
max_steps = 19073             # 10B tokens / 524288 tokens per step
```

Not magic numbers. The GPT-3 paper's schedule converted into steps for this batch size.

## 3. CausalSelfAttention (lines 49 to 76)

This is the only place in the entire model where information moves between token
positions. Everything else is per-position. Internalize that and the architecture stops
being mysterious.

### The setup

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # line 54: 768 -> 2304
self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # line 56: 768 -> 768
self.c_proj.NANOGPT_SCALE_INIT = 1                          # line 57: a flag, read later at init
```

Line 54 is one matrix doing the job of three. Conceptually you have `W_Q`, `W_K`, `W_V`,
each 768 to 768. Stacking them into one 768 by 2304 matrix means one big matmul instead
of three small ones, which the GPU vastly prefers. Purely a performance move,
mathematically identical.

`NANOGPT_SCALE_INIT` is not a torch thing. It is a sticky note attached to the two layers
that write into the residual stream, so `_init_weights` at line 133 can find them later.

### The forward, in geometric terms

```python
B, T, C = x.size()              # line 63: (64, 1024, 768)
qkv = self.c_attn(x)            # line 67: (B, T, 3C) = (64, 1024, 2304)
q, k, v = qkv.split(self.n_embd, dim=2)   # line 68: three tensors of (B, T, 768)
```

For every one of the 64 x 1024 token positions you now have three 768-dim vectors:

- **query**: "here is what I am looking for"
- **key**: "here is what I am"
- **value**: "here is what I will hand over if you pick me"

The classic example: the token `creature` emits a query meaning roughly _are there any
adjectives to my left describing me?_ The token `fluffy` emits a key meaning _I am an
adjective describing a noun_. Those two vectors point in similar directions, so their dot
product is large, so `creature` attends to `fluffy`, and `fluffy`'s value vector (carrying
"make this noun fluffier") gets added into `creature`'s position.

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 69
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 70
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 71
```

The multi-head split, and it is pure bookkeeping, zero arithmetic.
`(B, T, 768)` -> `(B, T, 12, 64)` -> `(B, 12, T, 64)`.

The `view` chops the 768 numbers into 12 contiguous slabs of 64. The `transpose(1, 2)`
pulls the head axis up next to batch so torch treats `(B, nh)` as 64 x 12 = 768 completely
independent 1024-by-64 attention problems, batched.

Simply: a token's query is a list of 768 numbers. Cut that list into 12 chunks of 64.
Chunk 0 (numbers 0 to 63) is head 0's query, chunk 1 (numbers 64 to 127) is head 1's
query, and so on. No number is copied, computed, or thrown away. You just drew 11 dividing
lines through a list you already had, and then told torch "treat each chunk as its own
separate problem."

That is the whole trick of multi-head attention. You are not running 12 attention layers.
You are running one attention layer whose query/key/value spaces have been partitioned
into 12 non-overlapping 64-dim subspaces, so head 3 can track subject-verb agreement while
head 7 tracks quotation marks, and neither can see the other's coordinates.

### The actual attention

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # line 72
```

One line, and it hides the most important equation in the file:

```
attn = softmax( (Q @ K^T) / sqrt(hs)  +  causal_mask )   # (B, nh, T, T)
y    = attn @ V                                          # (B, nh, T, hs)
```

**`Q @ K^T` is a T by T table.** Row _i_, column _j_ is the dot product of token _i_'s
query with token _j_'s key: "how much does position _i_ care about position _j_".
1024 by 1024, per head, per batch element.

**Why divide by `sqrt(hs)`.** If the entries of q and k are roughly unit variance and
independent, their dot product over 64 dimensions has variance 64, so standard deviation 8. Feed numbers of size +/-8 into a softmax and it saturates into a one-hot spike: one
token gets weight 0.999, everything else gets ~0, and the gradient through the softmax
dies. Dividing by `sqrt(64) = 8` restores unit variance and keeps the distribution soft
and trainable. This is not a heuristic, it is the variance algebra.

Simply: a dot product is a sum of 64 terms. Sums of many random terms get big, and they
get big at the rate `sqrt(how many terms)`, so 64 terms means roughly 8x bigger than one
term. Softmax cares about the _gaps_ between scores, not their absolute size, so inflated
scores mean inflated gaps, and softmax turns a big gap into "winner takes literally
everything." Dividing by 8 undoes the inflation the summing caused. Nothing about the
model wanted the scores that large, it was an artifact of adding up 64 numbers.

**`is_causal=True` is the arrow of time.** It adds `-inf` to every entry above the
diagonal before the softmax, so `exp(-inf) = 0` and token _i_ is structurally incapable of
seeing token _i+1_. This is what makes the whole thing trainable in parallel: a single
forward pass on a 1024-token sequence gives you 1024 simultaneous next-token predictions,
each one honestly blind to its own answer. Without the mask you would need 1024 separate
forward passes to get the same training signal.

**Why the fused kernel matters.** That T by T attention matrix is 1024 x 1024 = 1M floats.
Times 12 heads, times batch 64, that is 805M floats, over 3GB in fp32, per layer, and it
would have to round-trip to HBM. FlashAttention computes the softmax in tiles inside SRAM
and never materializes the full matrix in memory. Same math, roughly 4x faster,
dramatically less memory. The single biggest speedup in the file, and it costs one keyword
argument.

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)   # line 73
y = self.c_proj(y)                                 # line 75
```

Undo the head split: `(B, nh, T, hs)` back to `(B, T, C)`, gluing the 12 heads' 64-dim
outputs side by side into one 768-vector. The `.contiguous()` is mandatory, not
decorative: `transpose` only rewrites strides, it does not move bytes, and `view` refuses
to operate on a non-contiguous tensor.

Simply: memory is one long flat line of numbers. A tensor's shape is just a rule for
walking that line, and `transpose` swaps the walking rule without touching a single byte,
so the numbers are still physically laid out in the old order. `view` needs the numbers to
be sitting in memory in exactly the order it is about to read them, so it refuses.
`.contiguous()` does the actual copy that puts them in the new physical order, and then
`view` is happy.

Then `c_proj` mixes across the head boundary. Until this line, head 5's output lives
strictly in channels 320 to 383. `c_proj` is what lets the heads' findings combine, and
what decides how loudly this whole attention layer writes back into the residual stream.

## 4. MLP (lines 85 to 97)

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)   # line 88:  768 -> 3072
self.gelu   = nn.GELU(approximate="tanh")                   # line 89
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # line 90:  3072 -> 768
```

Up 4x, squash, back down. Applied to every position independently, with no communication
between positions at all. Position 7 has no idea position 8 exists inside this module.

The reading that actually sticks:

- Each of the 3072 **rows** of `c_fc` is a question asked of the residual vector. Row 1041
  might be a direction that lights up on "this is a basketball player".
- **GELU** turns the answer into a soft yes/no. Strongly negative goes to ~0 (question not
  triggered), positive passes through roughly unchanged.
- Each of the 3072 **columns** of `c_proj` is a fact to add back. Column 1041 might be the
  768-dim direction meaning "sport: basketball".

So the MLP is a soft key-value lookup table with 3072 entries. This is where the model
stores facts, and it is why the MLP holds about two thirds of the parameters (56.6M of the
85M in the blocks) despite attention getting all the attention.

**Why `approximate="tanh"`.** True GELU is `x * phi(x)` where phi is the Gaussian CDF,
which needs `erf`. In 2018 `erf` was slow in TensorFlow, so the paper used a tanh
polynomial approximation. No longer necessary, kept to match the original exactly.
Historical fidelity, not math.

**Why GELU and not ReLU.** ReLU is exactly flat at zero for all negative inputs, so a
neuron that drifts negative receives exactly zero gradient forever and is dead. GELU has a
small negative dip and is smooth everywhere, so there is always some gradient to climb
back out on.

## 5. Block, and the residual stream (lines 99 to 110)

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))    # line 108
    x = x + self.mlp(self.ln_2(x))     # line 109
    return x
```

Two lines. The most important design decision in the file.

Picture `x` as a **conveyor belt** running from the embeddings straight through all 12
layers to the final softmax. Each sublayer does not transform the belt, it **reads a copy,
computes a correction, and adds it back**. Nothing is ever overwritten, only accumulated.

Two consequences:

**Gradients.** The derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`. That `1`
is a gradient highway straight from the loss to the embeddings, no attenuation. Without
it, the gradient has to survive being multiplied through 12 layers of Jacobians, and it
will not.

**Semantics.** Layer 9 can read something layer 2 wrote, because layer 2's contribution is
still sitting in the stream. The residual stream is shared memory that every layer can
read from and write to.

Note **where the LayerNorms are**: `ln_1` is applied to the _input of the sublayer_, not
to the sum. This is the "pre-norm" formulation. The original 2017 Transformer put the norm
after the addition (`x = ln(x + attn(x))`), which puts a normalization directly on the
gradient highway and makes deep stacks much harder to train. GPT-2 moved it inside. The
residual path from `x` to output is now completely clean: additions only, no
normalization, no nonlinearity.

LayerNorm itself: for each token's 768 numbers independently, subtract the mean, divide by
the standard deviation, then apply a learned per-channel scale and shift. The volume knob
that stops the accumulating stream from drifting to enormous magnitudes.

Simply: it is grading on a curve. Take one token's 768 numbers, recenter them so they
average 0, and rescale them so their spread is 1. The _pattern_ of which channels are high
and which are low survives completely, only the overall loudness is standardized. So a
layer downstream always receives numbers in a predictable range no matter what the 11
layers before it decided to add. The learned scale and shift then let the model undo the
standardization per channel if it turns out it wanted the loudness after all.

## 6. GPT, the outer shell (lines 112 to 156)

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # line 119: token embeddings
    wpe = nn.Embedding(config.block_size, config.n_embd),   # line 120: position embeddings
    h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # line 121
    ln_f = nn.LayerNorm(config.n_embd)                      # line 122
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)   # line 124
```

The `ModuleDict` with these exact names (`wte`, `wpe`, `h`, `ln_f`) is not stylistic. It
makes `state_dict()` keys line up character for character with HuggingFace's, which is the
only reason `from_pretrained` can work by name matching.

`wte` is a lookup table of 50304 rows, one 768-dim vector per token. `wpe` is 1024 rows,
one per slot in the context.

### The forward pass

```python
pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # line 145: [0, 1, ..., T-1]
pos_emb = self.transformer.wpe(pos)   # line 146: (T, C)
tok_emb = self.transformer.wte(idx)   # line 147: (B, T, C)
x = tok_emb + pos_emb                 # line 148: (B, T, C) via broadcast
```

Line 148 is the one people trip on. **You add the position vector to the content vector.**
Not concatenate. Add. `(T, 768)` broadcasts across the batch dimension against
`(B, T, 768)`.

That feels lossy, and it is, but 768 dimensions is a lot of room. There is easily space
for a "position" subspace and a "content" subspace to coexist without stomping each other,
and the network learns to keep them separable. It has to happen somehow, because attention
as defined above is completely permutation-invariant: shuffle the tokens and the attention
math gives you the same set of outputs. Position information exists in this model _only_
because of line 148.

GPT-2's `wpe` is **learned**, not the sinusoidal scheme from the 2017 paper. Just 1024 free
vectors trained by gradient descent, and also the hard reason the context is capped at
1024: slot 1025 has no row in the table.

```python
for block in self.transformer.h:   # line 149
    x = block(x)                   # 12 times, shape never changes
x = self.transformer.ln_f(x)       # line 151: final norm
logits = self.lm_head(x)           # line 152: (B, T, C) -> (B, T, V)
```

The 12-layer loop is where 85M of the 124M parameters live, and the shape is
`(64, 1024, 768)` at every single step. Nothing changes shape. It just gets refined.

`lm_head` is the unembedding: dot every 768-dim output vector against all 50304 token
vectors. High dot product means "this direction points at that token".

```python
if targets is not None:                                                       # line 154
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # line 155
```

Flatten `(B, T, V)` into `(B*T, V)` and `(B, T)` into `(B*T,)`. Cross entropy does not care
that the rows came from a batch versus a time axis, they are just 65536 independent
classification problems.

Cross entropy here is `-log(probability the model assigned to the correct next token)`,
averaged. Which gives you the free sanity check: at init the model is uniform over 50304
tokens, so `-log(1/50304) = 10.82`. **If your very first loss is not about 10.8, something
is broken before you even start training.**

## 7. Initialization and weight tying (lines 126 to 140)

```python
self.transformer.wte.weight = self.lm_head.weight   # line 126
```

One line, 38.6M parameters saved, about 31% of the model.

Both matrices are `vocab x n_embd`. `wte` maps token id to vector, `lm_head` maps vector to
token score. Line 126 makes them **the same tensor object**, not a copy. Gradients from
both uses accumulate into one buffer.

The justification is semantic, not just economic: the direction in space that _means_ the
token "dog" ought to be the same direction that _predicts_ the token "dog". Empirically it
also improves loss, not just memory.

```python
def _init_weights(self, module):                              # line 130
    if isinstance(module, nn.Linear):
        std = 0.02                                            # line 132
        if hasattr(module, 'NANOGPT_SCALE_INIT'):             # line 133
            std *= (2 * self.config.n_layer) ** -0.5          # line 134
        torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
```

`0.02` comes straight from the GPT-2 source. Roughly `1/sqrt(768) = 0.036` in spirit, the
standard "keep activation variance at 1" scaling.

Line 134 is the subtle one, and where the sticky note from line 57 gets read. Each block
adds two contributions to the residual stream, so across 12 layers the stream is the sum
of 2 x 12 = 24 independent contributions. Summing 24 independent unit-variance things gives
variance 24, standard deviation ~4.9. Do that naively and activations grow steadily with
depth.

Scaling the _writing_ layers (`attn.c_proj` and `mlp.c_proj`, the only two that write back
into the stream) by `1/sqrt(24) = 0.204` keeps the stream at unit variance no matter how
deep you stack. This is **not** applied to `c_attn` or `c_fc`, which read from the stream
rather than write to it. That distinction is the whole point of the flag.

Simply: 24 people each pour a cup of water into the same bucket, and the bucket overflows.
The fix is to tell each person to pour less. How much less? If they each pour `1/sqrt(24)`
of a cup, the bucket ends up exactly as full as one cup, because randomly-signed
contributions partially cancel and so add up at the `sqrt` rate rather than linearly. Same
`sqrt` bookkeeping as the attention scaling above, applied to depth instead of head size.
And you only tell the _pourers_ to pour less, not the people reading the water level,
which is exactly why the flag is on `c_proj` and not on `c_attn` or `c_fc`.

```python
self.apply(self._init_weights)   # line 128
```

`nn.Module.apply` walks the entire tree recursively and calls the function on every
submodule. LayerNorm is deliberately skipped, because torch already defaults it to
scale=1, shift=0, which is exactly what you want.

### The param count, for calibration

```
wte      50304 x 768                        =  38,633,472
wpe       1024 x 768                        =     786,432
12 blocks x 7,087,872                       =  85,054,464
ln_f                                        =       1,536
lm_head  tied to wte, costs nothing         =           0
                                              -----------
                                              124,475,904   the "124M" in GPT-2 124M
```

## 8. from_pretrained (lines 168 to 208)

Surgery to load OpenAI's actual weights into your class.

```python
config_args = {
    'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    ...
}[model_type]
```

Worth staring at that table: scaling GPT-2 is almost entirely "make it deeper and wider",
not "change the design". XL is the same 6 lines of `Block.forward`, 48 times, at width 1600.

```python
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]           # line 186
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # line 193
```

Those are the causal mask buffers, constants rather than learned parameters. HF stores
them in the state dict. Your model does not even have them because `is_causal=True`
generates the mask on the fly, so line 186 is a harmless no-op on your side and line 193
is doing the real work.

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']              # line 196
...
sd[k].copy_(sd_hf[k].t())                                          # line 202
```

The original GPT-2 was written in TensorFlow using `Conv1D`, which stores weights as
`(in, out)`. PyTorch's `nn.Linear` stores `(out, in)`. Four weight matrices per block are
stored the other way around, so they get transposed on the way in. A 2019 TensorFlow
artifact leaking into the present, nothing deeper.

The `assert len(sd_keys_hf) == len(sd_keys)` on line 197 is the load-bearing safety net: if
your architecture drifted from OpenAI's by even one tensor, you find out here instead of
getting garbage output.

## 9. configure_optimizers (lines 216 to 236)

```python
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]   # line 220
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]    # line 221
```

The rule is simple: **2D and up gets weight decay, 1D does not.**

2D means matmul weights and embeddings, tensors that participate in mixing many inputs.
Pulling those toward zero is real regularization, it discourages any single weight from
dominating.

1D means biases and LayerNorm gains. A bias is one number offsetting one channel; decaying
it just biases the model toward zero output for no benefit. A LayerNorm gain is initialized
at exactly 1.0 by design, and decaying it toward 0 actively fights the initialization.

For this config you get **50 decayed tensors holding 124,354,560 params** and **98
non-decayed tensors holding 121,344 params**. The 50 is `wte + wpe + 4 matrices x 12
blocks`. `lm_head.weight` does not appear separately because `named_parameters()`
deduplicates tied tensors, which is a nice free confirmation that line 126 actually worked.

```python
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                              betas=(0.9, 0.95), eps=1e-8, fused=used_fused)   # line 235
```

`betas=(0.9, 0.95)`: the second value is torch's default 0.999 lowered to 0.95, per the
GPT-3 paper. Beta2 controls the decay of the running average of squared gradients. 0.999
has an effective memory of ~1000 steps, sluggish when the loss landscape is changing fast
early in training. 0.95 gives ~20 steps and adapts quicker.

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters   # line 231
used_fused = fused_available and device_type == "cuda"                         # line 232
```

Feature detection by introspecting the function signature, because the `fused` kwarg did
not exist in older torch. The fused kernel does the entire Adam update for all 148 tensors
in a single CUDA kernel launch instead of ~150 tiny ones. CUDA only, hence the guard.

## 10. get_lr, the learning rate schedule (lines 30 to 38)

```python
if it < warmup_steps:
    return max_lr * (it + 1) / warmup_steps        # line 32: linear ramp up
if it > max_steps:
    return min_lr                                  # line 34: floor
decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))   # line 37: 1 -> 0 smoothly
return min_lr + coeff * (max_lr - min_lr)               # line 38
```

**Warmup (steps 0 to 715).** At init the model is random and the gradients are large and
mostly wrong. Full-size steps immediately can blow up the loss in the first dozen
iterations. The ramp lets Adam's moment estimates stabilize before you commit. The
`it + 1` avoids a literal lr of 0 on step 0.

**Cosine decay (715 to 19073).** `0.5 * (1 + cos(pi * ratio))` slides smoothly from 1 down
to 0. Big confident steps early to find the right basin, tiny careful steps late to settle
into it. Cosine specifically because it spends more time near the max at the start and
eases off gently, versus a step schedule which jolts.

**Floor at 10%.** Never let the lr hit exactly zero, which would freeze learning entirely
for the tail of the run.

## 11. DataLoaderLite (lines 240 to 291)

```python
def next_batch(self):
    buf = self.tokens[self.current_position : self.current_position + B*T + 1]   # line 281
    x = (buf[:-1]).view(B, T)   # line 282: inputs
    y = (buf[1:]).view(B, T)    # line 283: targets
```

The `+ 1` and the two off-by-one slices are the entire supervised learning setup for
language modeling, and there is no labeling step anywhere. Grab `B*T+1` tokens, hand the
model the first `B*T`, and ask it to predict the same window shifted right by one.

Every one of the 65536 positions gets a training signal, and every one is causally honest
thanks to the mask. The labels are the data.

```python
self.current_position += B * T * self.num_processes   # line 285
```

and in `reset`:

```python
self.current_position = self.B * self.T * self.process_rank   # line 277
```

The multi-GPU interleave. With 8 GPUs, rank 0 starts at 0, rank 1 at 65536, rank 7 at
458752, and everyone advances by 8 x 65536 each call. The ranks tile the shard perfectly
with no overlap and no coordination.

```python
if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):   # line 287
    self.current_shard = (self.current_shard + 1) % len(self.shards)
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = B * T * self.process_rank
```

Look ahead: if the _next_ batch would run off the end, roll to the next shard now rather
than returning a short batch. The `%` wraps around so training can run forever.

`reset()` exists so validation always evaluates on shard 0 from position 0, giving you a
val loss curve that is actually comparable across steps instead of drifting with the data.

## 12. get_most_likely_row, the HellaSwag eval (lines 301 to 313)

HellaSwag gives you a context and 4 candidate endings, one correct. There is no
classification head here, so you score all 4 by asking which one the language model finds
least surprising.

```python
shift_logits = (logits[..., :-1, :]).contiguous()   # line 302
shift_tokens = (tokens[..., 1:]).contiguous()       # line 303
```

Same off-by-one alignment as the data loader: logits at position _t_ predict token _t+1_,
so drop the last logit and the first token.

```python
shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')   # line 306
```

`reduction='none'` is the key. Normally cross entropy averages everything into one scalar.
Here you need the per-token loss preserved so you can mask, so you get a `(4, T-1)` grid.

```python
shift_mask = (mask[..., 1:]).contiguous()          # line 308
masked_shift_losses = shift_losses * shift_mask    # line 309
avg_loss = sum_loss / shift_mask.sum(dim=1)        # line 311
pred_norm = avg_loss.argmin().item()               # line 312
```

The mask is 1 only on completion tokens. All 4 rows share an identical context prefix, so
scoring the context would add the same constant to every row and add pure noise. Only the
endings differentiate.

Dividing by `shift_mask.sum(dim=1)` rather than a fixed length is the "norm" in
`pred_norm`: it converts total loss into **average loss per token**, so a 3-token ending
competes fairly against a 12-token one. Without it you would systematically pick the
shortest option.

Simply: loss is a cost, and every extra token adds more cost, so long endings always look
worse than short ones just for being long. Comparing totals would be comparing the price
of a weekly shop to the price of a sandwich. Dividing by the token count turns it into
cost-per-token, which is the price-per-item comparison you actually wanted.

`argmin` because lower loss means higher probability means "the model thinks this ending is
most natural".

## 13. Device and DDP setup (lines 315 to 343)

```python
ddp = int(os.environ.get('RANK', -1)) != -1   # line 316
```

Clean detection: `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE`. If `RANK` is
absent you were launched as a plain `python main.py`, so single process.

- `ddp_rank`: global process id across all machines (0 to 7 on one 8-GPU box)
- `ddp_local_rank`: id within _this_ machine, which picks the physical GPU
- `master_process`: rank 0 only, the one allowed to print, log, and checkpoint, so 8
  processes do not write 8 copies of everything

`init_process_group(backend='nccl')` sets up NVIDIA's collective communication library, the
thing that makes `all_reduce` fast over NVLink.

## 14. The batch size math (lines 347 to 358)

```python
total_batch_size = 524288   # line 347: 2^19, half a million tokens, from the GPT-3 paper
B = 64                      # line 348: what actually fits in GPU memory
T = 1024                    # line 349
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)   # line 352
```

The resolution of a genuine conflict. The paper says the optimizer step should see 0.5M
tokens. Your GPU can hold 64 x 1024 = 65536 tokens. You cannot have both at once.

So you fake it: run 8 forward/backward passes, let the gradients accumulate in `.grad`
(which PyTorch does automatically, since `backward()` accumulates rather than overwrites),
and only then call `optimizer.step()`. Mathematically identical to one giant batch, just
serialized in time.

On 8 GPUs, `grad_accum_steps` drops to 1 and the same total batch is achieved in parallel
instead. The batch size is held fixed and the code adapts, which is the correct dependency
direction.

## 15. The training step (lines 486 to 516)

```python
model.train()                       # line 487
optimizer.zero_grad()               # line 488: clear last step's gradients
loss_accum = 0.0
for micro_step in range(grad_accum_steps):        # line 490
    x, y = train_loader.next_batch()
    x, y = x.to(device), y.to(device)
    if ddp:
        model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)   # line 494
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):           # line 495
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps  # line 497
    loss_accum += loss.detach()     # line 498
    loss.backward()                 # line 499
```

**Line 497 is the one everybody gets wrong.** `F.cross_entropy` already averages over the
65536 tokens in the micro batch. Summing 8 of those gives you a _sum of 8 means_, which is
8x too large. You want the mean over all 524288 tokens. Divide each by 8 first, and the
accumulated gradient comes out exactly equal to the true full-batch gradient.

Simply: you want the class average across 8 classrooms. Each classroom hands you its own
average, say 80, 90, and so on. Adding those 8 numbers gives 680, which is not an average
of anything. You wanted 85. Since the classrooms are all the same size, dividing each one
by 8 before adding gets you there. `backward()` only knows how to add into `.grad`, it has
no way to divide at the end, so you do the dividing up front on line 497.

**Line 494** is performance surgery. By default DDP fires an `all_reduce` across all GPUs
after every `backward()`. During accumulation that is 8 full syncs of 124M gradients when
you only need one. Setting the flag False on micro steps 0 to 6 and True on step 7
collapses that to a single sync.

**Line 498**, `.detach()` matters. Without it, `loss_accum` would hold onto the autograd
graph of all 8 micro batches, and you would OOM. You only want the number for printing.

**Line 495**, bfloat16 autocast. bf16 has the _same 8 exponent bits as fp32_, only fewer
mantissa bits. That is why there is no `GradScaler` anywhere in this file: fp16 has 5
exponent bits and gradients routinely underflow to zero, requiring loss scaling. bf16
trades precision for range and sidesteps the whole problem.

Simply: a float is stored like scientific notation, `1.2345 x 10^-7`. The exponent bits
are the `10^-7` part and decide the smallest and largest number you can represent at all.
The mantissa bits are the `1.2345` part and decide how many significant digits you keep.
bf16 keeps fp32's full range and gives up digits, so a tiny gradient still exists, just
sloppily. fp16 gave up range instead, so a tiny gradient becomes exactly 0.0 and is gone
forever. Sloppy is recoverable, gone is not.

Autocast is selective: matmuls
run in bf16, but softmax, layernorm, and the loss stay in fp32 where precision actually
matters. Master weights stay fp32 throughout.

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # line 502
```

Compute the global L2 norm across all 148 gradient tensors as if they were one long
vector. If it exceeds 1.0, scale everything down uniformly so it equals 1.0. Direction
preserved, only magnitude capped.

Occasionally a batch is pathological (a weird document, a run of rare tokens) and produces
a gradient 50x normal, which can wreck the model in a single step. Clipping bounds the
damage any one batch can do.

The returned `norm` is one of the best free diagnostics you have. It should start around 1
and settle to roughly 0.1 to 0.3. **A sudden spike means a bad batch or the start of
instability**, and it usually shows up in the norm before it shows up in the loss.

```python
if device_type == "cuda":
    torch.cuda.synchronize()   # line 508
```

Necessary for honest timing. CUDA calls are asynchronous: `optimizer.step()` returns the
instant the work is _queued_, not when it is _done_. Without this line you would be timing
how fast Python can submit work, which is a beautiful and completely fictitious number.

## 16. Generation (lines 450 to 484)

```python
while xgen.size(1) < max_length:            # line 461
    logits, loss = model(xgen)              # line 465: (B, T, vocab)
    logits = logits[:, -1, :]               # line 467: only the last position
    probs = F.softmax(logits, dim=-1)       # line 469
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)   # line 472
    ix = torch.multinomial(topk_probs, 1, generator=sample_rng)  # line 475
    xcol = torch.gather(topk_indices, -1, ix)                    # line 477
    xgen = torch.cat((xgen, xcol), dim=1)                        # line 479
```

Line 467 is the strange asymmetry of this whole architecture. During _training_ all 1024
positions produce useful predictions simultaneously. During _generation_ you compute all
1024 and throw away 1023 of them, keeping only the last. That is exactly the inefficiency
KV-caching exists to fix, and this code does not implement it (fine, it is generating 32
tokens for a vibe check, not serving traffic).

**Top-k = 50** is the quality guard. The tail of a 50304-way softmax holds tens of
thousands of tokens with individually tiny but collectively non-trivial probability. Sample
naively and eventually you draw something absurd, and because the model then conditions on
its own mistake, it derails and never recovers. Truncating to the top 50 and renormalizing
keeps the sample inside the plausible region.

Line 475 samples an _index into the top-50 list_, so line 477's `gather` is required to
translate that back into a real vocabulary id. Miss that step and you decode garbage.

Sampling instead of `argmax` because greedy decoding gets trapped in repetition loops
almost immediately. The `sample_rng` is a separate `Generator` seeded at `42 + ddp_rank`,
deliberately kept off the global RNG so sampling does not perturb the training data order,
and so each rank produces different samples.

## 17. Things to know about this specific file

**Lines 16 to 23 are dead.** `block_size = 256`, `n_embd = 384`, `n_head = 6`,
`n_layer = 6`, `dropout = 0.2`, `learning_rate = 3e-4`, `eval_interval`, `eval_iters`.
Nothing reads any of them. Notably there is **no dropout anywhere in this model**, correct
for a single-epoch 10B-token pretraining run where you cannot overfit, but the variable
sitting there suggests otherwise.

**`vocab_size=50304` on line 362** while `GPTConfig` defaults to 50257. The "ugly number to
nice number" trick: 50304 is 128 x 393. CUDA matmul kernels tile in powers of two, and a
dimension of 50257 forces an inefficient ragged remainder tile. You add 47 tokens that can
never appear in the data, the model quickly learns to give them near-zero probability, and
you get a real speedup for free. Adding parameters to go faster is unintuitive and correct.

**MPS and autocast.** Line 339 sets `device_type = "cuda" if device.startswith("cuda") else
"cpu"`, so on an M-series Mac the device is `mps` but `device_type` is `cpu`.
`torch.autocast(device_type="cpu", ...)` does not intercept mps ops, so the bf16 autocast
is a **silent no-op** there. No crash, just fp32. Same for
`torch.set_float32_matmul_precision('high')` on line 360, which only affects CUDA TF32.

**B=64, T=1024 needs roughly 80GB.** To smoke test locally, drop to `B=2` or `B=4` and cut
`max_steps` to something like 20.

**Timing on MPS is fictional.** Line 507 only synchronizes on CUDA, so `dt` and `tok/sec`
on Apple silicon measure queue submission, not real work.

**`master_process` on line 269** is read inside `DataLoaderLite.__init__` but defined at
module scope on lines 325/331. It works because instantiation on line 357 happens after,
but it is an implicit global dependency that would break if you imported this class from
another file.

**There is no resume.** The checkpoint dict at line 407 saves model, config, step, and val
loss, but not optimizer state or RNG, and nothing reads it back. If the run dies at step
12000 you restart from zero.

## Part I compression

1. **Attention is the only place information moves between positions.** Everything else is
   per-token.
2. **The residual stream is a conveyor belt** that every layer reads from and adds to,
   never overwrites. That is what makes 12 layers trainable and 96 layers possible.
3. **The causal mask is what makes training parallel.** 1024 honest predictions from one
   forward pass.
4. **`sqrt(head_size)` and `(2*n_layer)^-0.5` are the same idea twice:** keep the variance
   at 1, or the softmax saturates and the activations explode.
5. **Half this file is not the model.** It is the machinery for training it fast: grad
   accumulation, bf16, flash attention, fused Adam, DDP, vocab padding. The model is about
   60 lines. The engineering is 400.

---

# Part II: what happens across 19,073 steps

## 1. The landscape

The model has **124,475,904 parameters**. Every one is a knob you can turn.

So picture a space with 124,475,904 axes. A single point in that space is one complete
setting of every knob, which is to say, one entire model. Now hang a number over every
point: the average cross-entropy loss that model gets on FineWeb. That number is
**height**.

You now have a landscape. Not a 3D landscape, but the intuition survives the dimension
count better than you would expect. Training is one thing: **you are standing somewhere on
this surface and walking downhill.**

Line 128, `self.apply(self._init_weights)`, is where you get dropped in. Random Gaussians,
std 0.02. That starting point sits at height **10.82**, which is not an arbitrary altitude.
It is exactly `log(50304)`, the loss of a model that assigns equal probability to every
token in the vocabulary. You start at the altitude of "knows literally nothing."

19,073 steps later you want to be at about **3.29**. That is the whole job.

## 2. One step, in slow motion

**Feel the ground.** You need to know which way is downhill. In one dimension you would
just wiggle the knob and see if the loss went up or down. In 124 million dimensions,
wiggling each knob one at a time would mean 124 million forward passes per step. That is
not a slow approach, it is a fundamentally impossible one.

**So instead you compute the gradient.** The gradient is a vector with 124,475,904
components, one per knob, and component _i_ answers: _if I nudge knob i up by a hair, how
much does the loss change?_ Point that vector's direction and you are pointing at steepest
**uphill**. Negate it and you have your direction of travel.

Two properties of that vector matter, and people usually only internalize the first:

- Its **direction** is the way to go.
- Its **magnitude per component** tells you which knobs matter. A weight with a large
  partial derivative is a weight the loss cares a lot about. A weight with a near-zero
  partial derivative is currently irrelevant and will barely move. The gradient is
  simultaneously a direction and a ranking of importance.

**One backward pass gets you all of it.** This is the miracle, and the thing worth actually
understanding. `loss.backward()` on line 499 computes all 124 million partial derivatives
in roughly **twice the cost of one forward pass.** Not 124 million times. Twice.

The reason is that the chain rule runs backwards. Start at the loss with `dL/dL = 1`. Push
that backwards through `lm_head`: now you know how the loss responds to every one of the
final 768-dim vectors. Push through `ln_f`, then block 12, block 11, and so on down to the
embeddings. At every layer you already hold "how does the loss respond to my output," and
the layer's local derivative converts that into "how does the loss respond to my inputs and
my weights." Every quantity gets computed exactly once and reused by everything downstream.

That is the entire reason deep learning exists as a practical field. Not the architecture.
This.

**The gradient you compute is a guess.** The gradient you actually want is the one for the
_true distribution of all English text_, which is unavailable. What you get is the gradient
on 524,288 tokens sampled from FineWeb. That is a **noisy estimate** of the real thing:
right on average, wrong on any given step.

That is what line 347's `total_batch_size = 524288` is buying. Bigger batch, less noise,
straighter walk. Too small and you stagger drunkenly and can outright fall over. The noise
is not purely harmful (a bit of it helps rattle you out of bad regions) but at this scale
you want it controlled.

Which is why grad accumulation on lines 490 to 499 is not a hack. Eight micro-batches of
65,536 tokens each, gradients piling up in `.grad`, is arithmetically identical to one
honest batch of 524,288. You are assembling one confident measurement out of eight cheap
ones.

## 3. Adam, or why you do not just walk downhill

Plain gradient descent says: `param -= lr * grad`. Take the direction, take a fixed-size
step, repeat.

Nobody trains transformers that way, and the reason is geometric. The landscape is not a
nice round bowl. It is a **long narrow ravine**. Some directions are steep cliffs, others
are nearly flat valley floors stretching for miles. A single global step size cannot serve
both: big enough to make progress along the flat valley means you ricochet violently off
the cliff walls, and small enough to be stable on the cliffs means you crawl along the
valley floor forever.

AdamW (line 235) fixes this with two ideas layered on top of each other.

**Momentum (beta1 = 0.9).** Instead of stepping along the current gradient, step along a
running average of recent gradients. Physically: you are a ball with mass rolling downhill,
not a point teleporting. Consistent downhill directions accumulate speed. The noisy
back-and-forth components from batch sampling cancel themselves out. `0.9` means roughly a
10-step memory.

**Per-parameter step size (beta2 = 0.95).** Track a running average of each gradient
component _squared_, then divide that component's step by its square root. The effect: a
weight that has been receiving huge gradients gets a small step, a weight receiving tiny
gradients gets a proportionally larger one. **Every one of the 124 million dimensions gets
its own ruler, rescaled continuously.** The ravine gets stretched into something much
closer to a round bowl before you take the step.

That is why `0.999` was lowered to `0.95` here. Beta2 is the memory of that per-parameter
rescaling. `0.999` remembers ~1000 steps, far too sluggish when the landscape is genuinely
reshaping itself in the first thousand steps of training. `0.95` remembers ~20 and keeps up.

**The W in AdamW.** Weight decay pulls every 2D weight slightly toward zero on every step,
independent of the gradient. A constant gentle downward pressure on magnitude. Only the
matmul weights get it (line 220), because shrinking a bias or a LayerNorm gain is just
vandalism.

**Gradient clipping (line 502)** is the seatbelt. Take all 124 million gradient components
as one enormous vector, measure its length, and if it exceeds 1.0, scale the whole thing
down until it equals exactly 1.0. Direction preserved perfectly, only distance capped.
Occasionally you draw a genuinely pathological batch that produces a gradient 50 times
normal, and without the clip that single step launches you off the map.

## 4. The step size changes on purpose

`get_lr` on lines 30 to 38, geometrically.

**Warmup, steps 0 to 715.** At initialization you are at a random point on a landscape you
have measured exactly zero times. The gradients are enormous and mostly wrong, and Adam's
two running averages are still cold, initialized at zero and not yet meaningful. Taking
full-size steps here reliably blows the loss up in the first dozen iterations. So the
learning rate ramps linearly from ~0 to 6e-4. You are letting your eyes adjust before you
start running.

**Cosine decay, steps 715 to 19,073.** Early, you are far from anywhere good, and precision
is worthless. You do not care whether you land at the bottom of _this_ valley, you care
about finding the right continent. Big confident strides. Later, you are already in a good
basin, and the only thing left is to settle into its lowest point, which requires small
careful steps. The cosine shape holds you near the max for a while, then eases off
smoothly, then flattens gently at the end rather than slamming to a halt.

**Floor at 10%.** Never exactly zero, because a learning rate of zero means learning has
stopped and the last portion of your run is pure heat.

The mental image: hunting for the lowest point in a mountain range. Day one you drive.
Final hour you walk in circles slowly, watching your altimeter.

## 5. What the loss number actually means

Cross-entropy is `-log(probability the model assigned to the correct next token)`, averaged
over every position. It is measured in nats. The intuitive form is to exponentiate it,
which gives **perplexity**:

| loss  | perplexity | plain English                                      |
| ----- | ---------- | -------------------------------------------------- |
| 10.82 | 50,304     | uniform over the entire vocabulary. knows nothing. |
| 6.9   | ~1,000     | has learned which tokens are common.               |
| 4.6   | ~100       | local grammar works.                               |
| 3.9   | ~50        | fluent-ish, mostly coherent sentences.             |
| 3.29  | ~27        | GPT-2 124M territory.                              |

Perplexity is "how many options is the model effectively torn between." At step 0 it is
genuinely torn between 50,304. At the end it is torn between about 27. That is the whole
run: going from 50,304-way confusion to 27-way confusion.

And notice this is a **logarithmic** scale, which is why the loss curve is so cruelly
shaped. Dropping 10.8 to 6.9 cuts confusion by 50x and happens in the first minute or two.
Dropping 3.4 to 3.29 is a few percent and takes hours. **Each additional nat costs
exponentially more compute than the one before it.** This single fact is why frontier labs
spend nine figures on training runs.

## 6. The curriculum nobody wrote

Nothing in the file orders the lessons. But the model learns in a strikingly consistent
order anyway, because gradient descent is greedy and always grabs the cheapest available
loss reduction first. Step counts below are approximate.

**Steps 0 to ~50: unigram statistics.** The cheapest win available is to stop pretending
all tokens are equally likely. ` the` is roughly 5% of English text and `<|endoftext|>`
shows up at every document boundary. The model can capture this while ignoring the input
entirely. Loss falls off a cliff, 10.8 down to roughly 7, in under a minute. Almost all of
the visually dramatic drop in your loss curve is this, and it is the least interesting
thing that happens.

**Steps ~50 to ~500: local context.** Attention heads start firing on the previous token or
two. The model learns ` United` is often followed by ` States`, that `(` eventually wants
`)`, that a space usually follows a period. Roughly bigram and trigram behavior, delivered
by the `wpe` position embeddings and the earliest attention heads. Loss into the 5s.

**Steps ~500 to ~3,000: syntax and induction.** The important qualitative shift.
**Induction heads** form: pairs of attention heads that, having seen the pattern `[A][B]`
earlier in the context, will predict `[B]` the next time `[A]` appears. This is the first
genuinely non-local, in-context capability, and it is what "copy the name that was
mentioned 400 tokens ago" is made of. In many training runs this shows up as a visible bump
or brief plateau in the loss curve, because it is a discrete circuit forming rather than a
smooth parameter drift. Loss into the low 4s.

**Steps ~3,000 to 19,073: everything else, slowly.** Facts get packed into the MLPs.
Long-range coherence improves. Style, register, subject-verb agreement across clauses. The
loss curve looks almost flat here, and this is where the majority of your compute goes and
where nearly all the actual quality lives.

Which is exactly why the code generates samples every 250 steps (lines 450 to 484) and runs
HellaSwag. Once the loss curve goes visually flat it stops telling you much. The samples
still change dramatically. HellaSwag crawling off 0.25 (random guessing among 4) toward
~0.30 is measuring something the fourth decimal place of the loss will not show you.

## 7. Why the curve is smooth when the landscape is not

Reasonable objection: with 124 million knobs and a wildly non-convex loss, why does this
ever work? Should you not get stuck in a bad local minimum immediately?

The resolution is genuinely counterintuitive: **high dimension makes this easier, not
harder.**

For a point to be a local minimum, the surface must curve _upward_ in every single one of
the 124 million directions. Every one. If even one direction curves downward, you are not
stuck, you are on a **saddle point**, and you can keep going.

The odds of 124 million independent-ish directions all happening to curve up, at a point
that is not close to globally good, are effectively nil. So bad local minima essentially do
not occur at this scale. What you get instead is saddle points and long flat plateaus,
which are annoying (they slow you down) but not fatal, and momentum is specifically good at
coasting through them.

There is a second smoothing effect: your reported loss is averaged over 524,288 tokens.
Individual tokens are wildly variable, some trivially predictable and some impossible.
Averaging half a million of them per step gives you a number with tiny variance. The curve
you see is smooth partly because the landscape is more forgiving than intuition suggests,
and partly because you are looking at it through a very large averaging window.

## 8. Reading the scrolling numbers

```
step  1247 | loss: 4.238715 | lr 5.9821e-04 | norm: 0.2841 | dt: 1043.21ms | tok/sec: 502654.19
```

- **loss** should fall fast then flatten. Small jitter step to step is batch noise and is
  expected. Sustained rise means trouble.
- **lr** should trace the schedule exactly: linear up for 715 steps, then cosine down. It
  is a printout of an equation, so if it is wrong you have a bug, not a training problem.
- **norm** is your best early-warning instrument. Should start near 1 and settle into 0.1
  to 0.3. **A spike here precedes a loss spike.** If it pins at 1.0 constantly you are
  clipping every step, which means your learning rate is too high for your batch.
- **dt** should be nearly constant. Creeping upward suggests thermal throttling or a memory
  issue.
- **tok/sec** is your efficiency number, and multiplied by 19,073 x 524,288 it tells you
  when you will be done.

## 9. The size of the thing

The standard estimate for transformer training cost is `6 x params x tokens`:

```
6 x 124,475,904 x 10,000,000,000  =  approximately 7.5 x 10^18 floating point operations
```

Where the 6 comes from: every parameter is used in one multiply and one add per token, so
2 operations for the forward pass. The backward pass costs twice the forward, because it
computes both "how does the loss respond to my inputs" and "how does the loss respond to
my weights," which is 4 more. 2 + 4 = 6, per parameter, per token.

7.5 **exaFLOPs** to move one number from 10.82 to 3.29.

For scale, if you could do one multiplication per second by hand, without sleeping, it
would take about 240 billion years. This run does it in a couple of hours on 8 GPUs.

And the whole apparatus, the flash attention, the bf16 autocast, the fused Adam, the vocab
padded to 50304, the fused QKV projection, exists for exactly one reason: to move that
finish line from "next month" to "before dinner."

## Part II compression

1. **Training is walking downhill on a 124-million-dimensional surface.** That is the
   entire mental model.
2. **Backprop gets you all 124 million partial derivatives for the price of two forward
   passes.** Everything else is engineering; this is the actual idea.
3. **Your gradient is a noisy estimate.** Batch size buys you accuracy. That is what the
   524,288 is for.
4. **Adam gives every dimension its own ruler,** which turns a ravine into something
   walkable.
5. **Loss is log-scale, so progress is exponentially expensive.** The dramatic early drop
   is the model learning that ` the` is common. The flat boring part is where the
   intelligence goes in.

Watch `norm` more closely than `loss`. The loss tells you where you have been; the norm
tells you whether you are about to have a problem.

---

# Appendix: running it

```bash
git clone https://github.com/andrewn6/ml-from-scratch.git
cd ml-from-scratch
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch numpy tiktoken transformers datasets tqdm
```

Do not use `uv sync` or `uv run` in this repo on Linux. `pyproject.toml` pulls in `manim`,
which pulls `pycairo`, which has no Linux wheels and needs `libcairo2-dev` to build.

Data (writes to `gpt2/edu_fineweb10B/`, which is gitignored and must be regenerated):

```bash
cd gpt2
python fineweb.py --shards 2      # pipeline check, ~400MB. shard 0 = val, shard 1 = train
python fineweb.py                 # real run, 100 shards, 10B tokens, ~20GB
```

`--shards 2` gives one train shard of 100M tokens. `max_steps = 19073` consumes 10B tokens,
so the loader wraps and you re-read that shard about 100 times. Fine for a smoke test,
useless as a real run.

Smoke test (revert afterwards):

```bash
sed -i 's/^max_steps = 19073.*/max_steps = 20/; s/^warmup_steps = 715/warmup_steps = 5/; s/^B = 64 /B = 4 /' main.py
python main.py
git checkout main.py
```

What to check, in order: it finds the shards; it prints
`num decayed parameter tensors: 50, with 124,354,560 parameters` and `98 / 121,344`; the
first validation loss is about 10.82; train loss drops off 10.8 within a few steps; `norm`
is finite and order 1.

Train:

```bash
python main.py                                      # 1 gpu
torchrun --standalone --nproc_per_node=8 main.py    # 8 gpus
```

`main.py` reads `edu_fineweb10B` relative to your current directory, so it must be launched
from inside `gpt2/`.

Set `B` to fit the card. Leave `total_batch_size = 524288` alone, grad accum absorbs the
change. Keep `B` a power of two or the assert on line 350 trips.

| GPU                   | B   | grad_accum (1 GPU) |
| --------------------- | --- | ------------------ |
| H100 / A100 80GB      | 64  | 8                  |
| A100 40GB / L40S 48GB | 32  | 16                 |
| 4090 / A6000          | 16  | 32                 |
| smaller               | 8   | 64                 |

Logs land in `gpt2/log/log.txt` as three line types:

```
0 val 10.9834
0 hella 0.2503
0 train 10.982314
```

Healthy end state: train loss under 3.3, HellaSwag climbing off 0.25 toward ~0.30, `norm`
in the 0.1 to 0.3 band.

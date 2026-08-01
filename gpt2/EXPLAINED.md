# GPT-2, explained

A line-by-line walkthrough of `main.py`, in the style of 3blue1brown's neural network
and transformer videos. Pictures first, code second.

Part I is the architecture: what the model is.
Part II is the training run: what happens over 19,073 steps.

Line numbers refer to `gpt2/main.py`.

---

# Part I: the architecture

## 0. The whole thing in one sentence

Start with what the model is, before what it contains. It is one very large mathematical
function. Text goes in, and what comes out is a probability for every token that could come
next. That is the entire job. Everything below is how that function is built.

The name says the same thing. GPT is **G**enerative, it writes its own text rather than
choosing from options. **P**re-trained, it learns from an enormous pile of general text first
and can be specialized later. **T**ransformer, the architecture underneath, which is what these
1200 lines are about.

To generate, you run that function, sample a token from the distribution it gives you, stick
that token on the end of your input, and run it again. Predict, sample, repeat.

Now the shape of it. Imagine a conveyor belt with 1024 slots on it. Each slot holds a list of
768 numbers.

Twelve identical machines sit along the belt. Each machine does exactly two things. First it
lets the slots pass information sideways to each other. That is attention. Then it lets each
slot think privately, with no idea the other slots exist. That is the MLP.

At the end of the belt you compare every slot's list of numbers against every word in the
vocabulary, and turn the comparison into probabilities.

That is the model. Everything else in the file is plumbing: getting bytes off disk, keeping
numbers in a healthy range, and spreading the work across GPUs.

## 1. The four numbers to hold in your head

```
B = 64      batch: how many independent sequences at once
T = 1024    time: context window, how many tokens per sequence
C = 768     channels: the width of the residual stream (n_embd)
V = 50304   vocab
nh = 12     heads,  hs = C // nh = 64  head size
```

Every tensor in the forward pass is some rearrangement of `(B, T, C)`. If you ever lose the
plot, stop and ask: what are B, T, and C right now? That question alone gets you back.

## 2. Config (lines 16 to 47)

```python
block_size = 256      # line 16
n_embd = 384          # line 20
dropout = 0.2         # line 23
```

**These lines are dead.** They are leftovers from the makemore and nanoGPT lessons. Nothing in
the file reads them. The real config is the dataclass:

```python
@dataclass
class GPTConfig:                  # line 41
    block_size: int = 1024        # max sequence length
    vocab_size: int = 50257       # 50,000 BPE merges + 256 raw bytes + 1 <|endoftext|>
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
```

That vocab number is worth unpacking, because it looks arbitrary and is not.

Start with 256 tokens, one for every possible byte. Now nothing you could ever type is
unrepresentable, since in the worst case it comes through one byte at a time. On top of that,
run 50,000 byte-pair merges, each one gluing a common pair together into a new single token.
Then add one special token, `<|endoftext|>`, numbered 50256. Add them up: 256 + 50000 + 1 = 50257.

The real hyperparameters:

```python
max_lr = 6e-4                 # line 25, from the GPT-3 paper's 124M row
min_lr = max_lr * 0.1         # decay to 10%, not to 0
warmup_steps = 715            # 375M warmup tokens / 524288 tokens per step
max_steps = 19073             # 10B tokens / 524288 tokens per step
```

None of these are magic. They are the GPT-3 paper's schedule, which is written in tokens,
divided by how many tokens this setup does per step. That is all a step count is here.

## 3. CausalSelfAttention (lines 49 to 76)

This is the only place in the entire model where information moves between token positions.
Every other line treats each position alone. Hold onto that and the architecture stops being
mysterious.

### What attention is for

Before any of the code, the problem it solves.

Read these three phrases: the American shrew mole, one mole of carbon dioxide, the doctor took
a biopsy of the mole. The word `mole` means something completely different in each one.

Now look at what `wte` does on line 147. It is a lookup table. Token id goes in, vector comes
out. So `mole` gets the exact same 768 numbers in all three phrases, because a lookup table has
no idea what surrounds it. At that moment the model is holding one generic, blurry, averaged
sense of `mole`, and nothing else.

That is the problem. **Attention is what turns the generic vector into the specific one.**

The embedding gives you the word's meaning out of context. Attention lets the surrounding
tokens push it toward the meaning it has in _this_ sentence, moving it to a different part of
the space, one that encodes "small burrowing animal" rather than "unit of chemical quantity".

And attention is how information moves at all, including across long distances. In "the animal
didn't cross the street because it was tired", when the model reaches `it`, something has to
connect that pronoun back to `animal`, which is eight tokens back. Attention is the only
mechanism in the model capable of doing that. Afterwards `it` is no longer a vague pronoun. It
carries `animal` with it.

So each token asks one question: **which other tokens are relevant to me right now?** The rest
of this section is how that question gets asked in matrix form.

### The setup

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # line 54: 768 -> 2304
self.c_proj = nn.Linear(config.n_embd, config.n_embd)       # line 56: 768 -> 768
self.c_proj.NANOGPT_SCALE_INIT = 1                          # line 57: a flag, read later at init
```

Line 54 is one matrix doing the job of three. What you conceptually have is `W_Q`, `W_K`, and
`W_V`, each mapping 768 numbers to 768 numbers. Stack them side by side and you get one matrix
that is 768 wide and 2304 tall. Now it is one big matmul instead of three small ones, which is
what a GPU wants. The math is identical. This is purely about speed.

`NANOGPT_SCALE_INIT` is not a torch feature. It is a sticky note. Two layers in each block
write their output back onto the conveyor belt, and those two need special treatment when the
weights get initialized. Rather than hunt for them later by name, you tag them now, and
`_init_weights` at line 133 checks for the tag.

### The forward, as a picture

```python
B, T, C = x.size()              # line 63: (64, 1024, 768)
qkv = self.c_attn(x)            # line 67: (B, T, 3C) = (64, 1024, 2304)
q, k, v = qkv.split(self.n_embd, dim=2)   # line 68: three tensors of (B, T, 768)
```

There are 64 x 1024 token positions in play. Each one now holds three vectors of 768 numbers,
and each vector has a job:

- **query**: here is what I am looking for.
- **key**: here is what I am.
- **value**: here is what I will hand over if you pick me.

The one-line version, which is worth memorizing: **queries and keys decide which tokens matter.
Values decide what information actually gets copied.** Those are two separate jobs, and keeping
them separate is the whole design.

The classic example. The token `creature` puts out a query that means roughly _are there any
adjectives to my left describing me?_ The token `fluffy` puts out a key that means _I am an
adjective describing a noun_. Those two vectors point in similar directions. Similar directions
means a large dot product. So `creature` attends to `fluffy`, and `fluffy`'s value vector,
which carries something like "make this noun fluffier", gets added into `creature`'s position
on the belt. The result sits in a different spot in the space than plain `creature` did: it now
encodes a fluffy creature.

A dot product, since everything below leans on it: multiply two vectors component by component
and add up the results. It is one number, and it measures how aligned the two vectors are.
Higher means stronger match, lower means weaker. That is the entire scoring mechanism.

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 69
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 70
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # line 71
```

This is the multi-head split. Here is the thing to know about it: no arithmetic happens. None.
It is bookkeeping.

A token's query is a list of 768 numbers. Cut that list into 12 chunks of 64. Numbers 0 to 63
are head 0's query. Numbers 64 to 127 are head 1's query. And so on. Nothing was copied,
nothing was computed, nothing was thrown away. You drew 11 dividing lines through a list you
already had.

That is what `view` does, turning `(B, T, 768)` into `(B, T, 12, 64)`. Then `transpose(1, 2)`
moves the head axis up next to the batch axis, giving `(B, 12, T, 64)`. Now torch sees
64 x 12 = 768 completely separate attention problems, each one 1024 by 64, and runs them all at
once.

So multi-head attention is not 12 attention layers. It is one attention layer whose query, key,
and value spaces have been cut into 12 non-overlapping 64-dimensional slices. Head 3 can track
subject-verb agreement while head 7 tracks quotation marks, and neither one can see the other's
coordinates.

### The actual attention

```python
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # line 72
```

One line, and it hides the most important equation in the file:

```
attn = softmax( (Q @ K^T) / sqrt(hs)  +  causal_mask )   # (B, nh, T, T)
y    = attn @ V                                          # (B, nh, T, hs)
```

**`Q @ K^T` is a table, T by T.** Row _i_, column _j_ holds the dot product of token _i_'s
query with token _j_'s key. Read it as: how much does position _i_ care about position _j_? The
table is 1024 by 1024, and there is one of them per head, per batch element.

**What softmax does**, since this equation is where it first bites. Those raw dot products are
useless as weights. They are arbitrary sized, some negative, and they do not add up to
anything. To blend other tokens' values together you need weights that behave like proportions:
all positive, all summing to 1.

Softmax gets you there in three moves. Raise `e` to the power of each score, which makes
everything positive. Add all those results up to get a total. Divide each one by that total.
Now every number is between 0 and 1 and the whole list sums to 1, with the biggest original
score getting the biggest share.

The row of weights that comes out is called the **attention pattern**, and it is readable. It
says exactly how much this token is drawing from each other token. In "the animal didn't cross
the street because it was tired", the row for `it` should put a lot of its weight on `animal`.
Notice that attention is never picking one token. It is mixing many, with some contributing
more than others.

**Why divide by `sqrt(hs)`?**

A dot product is a sum of 64 terms. Sums of many random terms get big, and there is a rate to
how big: they grow like the square root of how many terms you added. 64 terms, so about 8x
bigger than a single term.

Now, softmax does not care about the absolute size of the scores. It cares about the _gaps_
between them. So inflating the scores by 8x inflates every gap by 8x, and softmax turns a big
gap into "the winner takes literally everything." One token gets probability 0.999, everything
else gets essentially zero, and the gradient through the softmax dies.

Dividing by `sqrt(64) = 8` undoes exactly the inflation that the summing caused. Nothing about
the model ever wanted scores that large. It was an artifact of adding up 64 numbers, so you
subtract the artifact back out. This is not a tuned heuristic, it is variance algebra.

**`is_causal=True` is the arrow of time.** Before the softmax, it adds `-inf` to every entry
above the diagonal. Since `exp(-inf) = 0`, those entries become exactly zero probability, and
token _i_ becomes structurally incapable of seeing token _i+1_.

Worth asking why it is `-inf` and not just zero. If you zeroed the scores after the softmax,
the surviving weights would no longer add up to 1, so they would stop being proportions and you
would have to renormalize by hand. Doing it _before_ the softmax solves it in one move: `-inf`
exponentiates to 0, that token contributes nothing to the total, and the remaining weights
still sum to 1 automatically. You get the blocking and the normalization from the same step.

This is what makes the whole thing trainable in parallel, and it is worth being precise about
why. One forward pass on a 1024-token sequence gives you 1024 next-token predictions at once.
Each of those predictions is honestly blind to its own answer, because the mask forbids looking
right. Without the mask you would have to do 1024 separate forward passes to get the same
training signal.

**Why the fused kernel matters.** Count the size of that attention table. 1024 x 1024 is about
1M floats. Times 12 heads, times 64 batch elements, is 805M floats. That is over 3GB in fp32,
for one layer, and it would have to be written out to memory and read back.

FlashAttention does not build the table. It computes the softmax in small tiles that fit in
fast on-chip memory, and never materializes the full thing. Same math, roughly 4x faster, and
dramatically less memory. It is the single biggest speedup in the file and it costs one keyword
argument.

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)   # line 73
y = self.c_proj(y)                                 # line 75
```

Line 73 undoes the head split, `(B, nh, T, hs)` back to `(B, T, C)`, gluing the 12 heads'
64-number outputs side by side into one 768-number list.

The `.contiguous()` is mandatory, and the reason is worth a paragraph.

Memory is one long flat line of numbers. A tensor's shape is just a rule for walking that line.
`transpose` swaps the walking rule and does not touch a single byte, so after a transpose the
numbers are still physically sitting in their old order. `view` cannot work with that. It needs
the numbers laid out in memory in exactly the order it is about to read them, so it refuses.
`.contiguous()` does the actual copy that puts them in the new physical order. Then `view` is
happy.

Then `c_proj` mixes across the head boundary. Up until this line, head 5's output lives strictly
in channels 320 to 383 and nowhere else. `c_proj` is what lets the heads' findings combine with
each other. It also decides how loudly this whole attention layer writes back onto the belt.

Which gives you the cleanest way to hold multi-head attention in your head. Adjectives updating
nouns is only one of the many ways context can change a word's meaning, so you run 12 of these
at once, each with its own idea of what to look for. **Every head proposes a change to the
token's vector. The changes get summed, and the sum is what lands on the belt** at line 108.
Twelve opinions, one edit.

### The whole flow, in order

Worth having the sequence memorized, because every line above is one step of it:

```
token embedding (no context yet)
  -> build a query, key, and value for each token
  -> compare each query against every key with a dot product
  -> raw attention scores
  -> scale by sqrt(head size), mask out the future
  -> softmax into the attention pattern (weights summing to 1)
  -> blend the value vectors using those weights
  -> add the result back onto the token's vector, now with context
```

**A note on what is not here.** This is _self_-attention: queries, keys, and values all come
from the same sequence. There is a variant called cross-attention where the queries come from
one dataset and the keys and values come from another, say English text querying French text
during translation, or an ongoing transcript querying audio. The machinery is identical. And
because there is no notion of a future token to protect there, cross-attention has no causal
mask. It does not appear in this file, but knowing it exists tells you which parts of the design
above are essential and which are choices.

## 4. MLP (lines 85 to 97)

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)   # line 88:  768 -> 3072
self.gelu   = nn.GELU(approximate="tanh")                   # line 89
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)   # line 90:  3072 -> 768
```

Up 4x, squash, back down. This runs on every position independently. There is no communication
between positions here at all. Inside this module, position 7 has no idea position 8 exists.

The reading that actually sticks is to think of it as a lookup table.

- Each of the 3072 **rows** of `c_fc` is a question asked of the incoming vector. Row 1041
  might be a direction that lights up on "this is a basketball player".
- **GELU** turns the answer into a soft yes or no. Strongly negative goes to about 0, meaning
  the question did not trigger. Positive passes through roughly unchanged.
- Each of the 3072 **columns** of `c_proj` is a fact to add back. Column 1041 might be the
  768-dimensional direction meaning "sport: basketball".

So the MLP is a soft key-value lookup with 3072 entries. Ask 3072 questions, and for each one
that fires, add its answer back onto the belt.

This is where the model stores facts, and it is why the MLP holds about two thirds of the
parameters in the blocks, 56.6M out of 85M, despite attention getting all the attention.

**Why `approximate="tanh"`?** True GELU is `x * phi(x)`, where phi is the Gaussian CDF, and
computing it needs the `erf` function. Back in 2018 `erf` was slow in TensorFlow, so the paper
used a tanh polynomial that comes very close. That is no longer necessary. It is kept here to
match the original exactly. Historical fidelity, not math.

**Why GELU and not ReLU?** ReLU is perfectly flat at zero for every negative input. Flat means
zero slope, and zero slope means zero gradient. So a neuron that drifts negative gets no
gradient at all, forever, and is dead. GELU dips slightly below zero and is smooth everywhere,
so there is always some slope to climb back out on.

## 5. Block, and the conveyor belt (lines 99 to 110)

```python
def forward(self, x):
    x = x + self.attn(self.ln_1(x))    # line 108
    x = x + self.mlp(self.ln_2(x))     # line 109
    return x
```

Two lines, and the most important design decision in the file.

Notice that neither line replaces `x`. Both of them add to it. Each sublayer reads a copy of
the belt, works out a correction, and adds the correction back on. Nothing is ever overwritten.
Everything accumulates. That accumulating thing is called the **residual stream**, and it runs
unbroken from the embeddings all the way to the final softmax.

Two things follow from that, and they are both large.

**Gradients get a highway.** The derivative of `x + f(x)` with respect to `x` is `1 + f'(x)`.
That `1` is a clean path from the loss all the way back to the embeddings with nothing
attenuating it. Without it, the gradient has to survive being multiplied through 12 layers'
worth of Jacobians on the way back, and it will not.

**Layers can talk across distance.** Layer 9 can read something layer 2 wrote, because layer
2's contribution is still sitting there in the stream. The residual stream is shared memory
that every layer can read from and write to.

Now look at **where the LayerNorms are**. `ln_1` is applied to the input of the sublayer. It is
not applied to the sum. This is the "pre-norm" arrangement.

The original 2017 Transformer did it the other way, `x = ln(x + attn(x))`, which puts a
normalization directly on the gradient highway and makes deep stacks much harder to train.
GPT-2 moved the norm inside. The path from `x` to the output is now completely clean: additions
only, no normalization, no nonlinearity.

As for LayerNorm itself, it is grading on a curve. Take one token's 768 numbers. Shift them so
they average 0. Rescale them so their spread is 1. The _pattern_ of which channels are high and
which are low survives completely. Only the overall loudness is standardized.

That means a layer downstream always receives numbers in a predictable range, no matter what
the 11 layers before it decided to add. And afterwards a learned scale and shift is applied per
channel, so if the model actually wanted some of that loudness back, it can put it back.

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

The `ModuleDict` and those names inside it, `wte`, `wpe`, `h`, `ln_f`, are not a style choice.
PyTorch builds the keys of
`state_dict()` out of attribute names, so naming them this way makes the keys line up character
for character with HuggingFace's. That is the only reason `from_pretrained` can work by
matching names.

`wte` is a lookup table with 50304 rows, one 768-number vector per token. `wpe` is a lookup
table with 1024 rows, one per slot in the context.

### The forward pass

```python
pos = torch.arange(0, T, dtype=torch.long, device=idx.device)   # line 145: [0, 1, ..., T-1]
pos_emb = self.transformer.wpe(pos)   # line 146: (T, C)
tok_emb = self.transformer.wte(idx)   # line 147: (B, T, C)
x = tok_emb + pos_emb                 # line 148: (B, T, C) via broadcast
```

Line 148 is the one people trip on. **You add the position vector to the content vector.** Not
concatenate. Add. The shapes work out because `(T, 768)` broadcasts across the batch dimension
against `(B, T, 768)`.

Adding two things together feels like it should destroy information, and in principle it can.
But 768 dimensions is an enormous amount of room. There is easily space for a "position"
subspace and a "content" subspace to sit alongside each other without stomping on each other,
and the network learns to keep them separable.

It has to happen somehow, because attention as described above is completely blind to order.
Shuffle the tokens and the attention math hands back the same set of outputs. Position
information exists in this model _only_ because of line 148.

One difference from the 2017 paper: GPT-2's `wpe` is **learned**. Not the sinusoidal formula,
just 1024 free vectors trained by gradient descent like everything else. That is also the hard
reason the context is capped at 1024. Slot 1025 has no row in the table.

```python
for block in self.transformer.h:   # line 149
    x = block(x)                   # 12 times, shape never changes
x = self.transformer.ln_f(x)       # line 151: final norm
logits = self.lm_head(x)           # line 152: (B, T, C) -> (B, T, V)
```

That 12-layer loop is where 85M of the 124M parameters live. The shape is `(64, 1024, 768)` at
every single step of it. Nothing ever changes shape. The contents just get refined.

`lm_head` is the unembedding, and it is the exact mirror of what `wte` did at the very start.
`wte` was embedding: take a discrete token id and move it into the model's vector space.
`lm_head` moves back the other way: take a vector and turn it into a score for every token in
the vocabulary. Same shape, opposite jobs.

Mechanically it takes each 768-number output vector and dots it against all 50304 token
vectors. A high dot product means "this direction is pointing at that token".

The 50304 raw scores that come out are called **logits**. One per token in the vocabulary, and
higher means the model currently considers that token more likely. They are not probabilities
yet, they are arbitrary sized and can be negative. Softmax is what converts them, exactly as it
did inside attention.

And notice what this is not. The model is not looking up an answer in a table. It builds one
final vector out of the entire context, and then asks: which token vectors does this thing point
toward most strongly?

```python
if targets is not None:                                                       # line 154
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) # line 155
```

Flatten `(B, T, V)` into `(B*T, V)` and `(B, T)` into `(B*T,)`. Cross entropy does not care
that some rows came from the batch axis and some from the time axis. To it they are just 65536
independent classification problems.

Cross entropy here is one thing: `-log(the probability the model gave to the correct next
token)`, averaged.

Which gives you a free sanity check. At init the model is random, so it spreads probability
evenly over all 50304 tokens. Each correct token therefore gets probability 1/50304, and
`-log(1/50304) = 10.82`. **If your very first loss is not about 10.8, something is broken
before training has even started.**

## 7. Initialization and weight tying (lines 126 to 140)

```python
self.transformer.wte.weight = self.lm_head.weight   # line 126
```

One line. It saves 38.6M parameters, about 31% of the model.

Both matrices have shape `vocab x n_embd`. `wte` turns a token id into a vector. `lm_head`
turns a vector into a score per token. They are the same shape because they are doing the same
job in opposite directions.

Line 126 makes them **the same tensor object**, not a copy. Both uses feed gradients into one
buffer.

The justification is not only economic, it is semantic. The direction in space that _means_ the
token "dog" ought to be the same direction that _predicts_ the token "dog". Empirically it
improves the loss too, not just the memory bill.

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

The `0.02` comes straight from the GPT-2 source. It is in the spirit of `1/sqrt(768) = 0.036`,
the standard "keep the activations at variance 1" scaling.

Line 134 is the subtle one, and it is where the sticky note from line 57 gets read.

Here is the problem it solves. Each block adds two things to the belt. Across 12 blocks, that
is 2 x 12 = 24 separate contributions all landing on the same running total. Twenty-four people
each pour a cup of water into one bucket, and the bucket overflows. In the actual numbers: sum
24 independent unit-variance things and you get variance 24, which is a standard deviation of
about 4.9. Do that naively and the activations grow steadily with depth.

The fix is to tell each person to pour less. How much less? If they each pour `1/sqrt(24)` of a
cup, which is 0.204 of a cup, the bucket ends up about as full as one cup. The reason it is a
square root and not 1/24 is that the contributions have random signs, so they partially cancel
rather than stacking up neatly. Randomly signed contributions add at the square-root rate, so
you correct at the square-root rate.

Notice this is the same bookkeeping as the `sqrt(head_size)` division in attention. There it was
applied to the 64 terms in a dot product. Here it is applied to the 24 writes down the depth of
the network. Same idea, twice.

And you only tell the _pourers_ to pour less. Not the people reading the water level. That is
exactly why the flag sits on `attn.c_proj` and `mlp.c_proj`, the only two layers that write back
into the stream, and not on `c_attn` or `c_fc`, which read from it.

```python
self.apply(self._init_weights)   # line 128
```

`nn.Module.apply` walks the entire module tree and calls your function on every submodule.
LayerNorm is deliberately left alone, because torch already starts it at scale 1 and shift 0,
which is exactly what you want.

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

This function is surgery. It loads OpenAI's actual weights into your class.

```python
config_args = {
    'gpt2':        dict(n_layer=12, n_head=12, n_embd=768),   # 124M
    'gpt2-medium': dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
    ...
}[model_type]
```

That table is worth staring at. Scaling GPT-2 up is almost entirely "make it deeper and wider".
It is not "change the design". XL is the same 6 lines of `Block.forward`, run 48 times, at
width 1600.

```python
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]           # line 186
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # line 193
```

Those two entries are causal mask buffers. They are constants, not learned parameters, and
HuggingFace stores them in the state dict anyway. Your model does not even have them, because
`is_causal=True` generates the mask on the fly. So line 186 is a harmless no-op on your side,
and line 193 is doing the real work.

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']              # line 196
...
sd[k].copy_(sd_hf[k].t())                                          # line 202
```

The original GPT-2 was written in TensorFlow using `Conv1D`, which stores its weights as
`(in, out)`. PyTorch's `nn.Linear` stores them as `(out, in)`. So four weight matrices per block
arrive stored the other way around and have to be transposed on the way in. A 2019 TensorFlow
artifact leaking into the present. Nothing deeper than that.

The `assert len(sd_keys_hf) == len(sd_keys)` on line 197 is the load-bearing safety net. If your
architecture has drifted from OpenAI's by even one tensor, you find out right here, instead of
finding out later by getting garbage output.

## 9. configure_optimizers (lines 216 to 236)

```python
decay_params   = [p for n, p in param_dict.items() if p.dim() >= 2]   # line 220
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]    # line 221
```

The rule is simple: **2D and up gets weight decay, 1D does not.**

2D means matmul weights and embeddings. Those are tensors that mix many inputs together, and
pulling them gently toward zero is real regularization. It discourages any single weight from
dominating.

1D means biases and LayerNorm gains, and for both of them decay is actively wrong. A bias is one
number offsetting one channel, so shrinking it just biases the model toward outputting zero for
no benefit. A LayerNorm gain starts at exactly 1.0 on purpose, so shrinking it toward 0 fights
the initialization you deliberately chose.

For this config you get **50 decayed tensors holding 124,354,560 params** and **98 non-decayed
tensors holding 121,344 params**. Where does 50 come from? `wte` and `wpe` make 2, plus 4
matrices per block times 12 blocks makes 48.

`lm_head.weight` does not show up separately, because `named_parameters()` deduplicates tied
tensors. Which is a nice free confirmation that line 126 actually worked.

```python
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate,
                              betas=(0.9, 0.95), eps=1e-8, fused=used_fused)   # line 235
```

`betas=(0.9, 0.95)`. That second number is torch's default of 0.999, lowered to 0.95, on the
GPT-3 paper's recommendation. Beta2 sets how long the optimizer remembers past squared
gradients. 0.999 works out to a memory of about 1000 steps, which is sluggish when the loss
landscape is changing fast early in training. 0.95 gives about 20 steps, and keeps up.

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters   # line 231
used_fused = fused_available and device_type == "cuda"                         # line 232
```

This is feature detection by looking at the function's own signature, because the `fused`
keyword did not exist in older torch versions. What it buys you: the fused kernel does the
entire Adam update for all 148 tensors in one CUDA kernel launch, instead of about 150 tiny
ones. It is CUDA only, hence the guard.

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

**Warmup, steps 0 to 715.** At the start the model is random, so the gradients are large and
mostly wrong. Taking full-size steps right away can blow the loss up within a dozen iterations.
The ramp gives Adam's running averages time to become meaningful before you commit to them. The
`it + 1` is there so step 0 does not get a learning rate of literally zero.

**Cosine decay, steps 715 to 19073.** `0.5 * (1 + cos(pi * ratio))` slides smoothly from 1 down
to 0. Big confident steps early to find the right region, tiny careful steps late to settle into
it. Cosine specifically, rather than a step schedule, because it holds near the maximum for a
while and then eases off gently instead of jolting.

**Floor at 10%.** Never let the learning rate hit exactly zero. Zero means learning has stopped,
and the tail of the run would be doing nothing at all.

## 11. DataLoaderLite (lines 240 to 291)

```python
def next_batch(self):
    buf = self.tokens[self.current_position : self.current_position + B*T + 1]   # line 281
    x = (buf[:-1]).view(B, T)   # line 282: inputs
    y = (buf[1:]).view(B, T)    # line 283: targets
```

That `+ 1` and those two off-by-one slices are the entire supervised learning setup for language
modeling. There is no labeling step anywhere, and there never was one.

Grab `B*T+1` tokens. Hand the model the first `B*T` of them. Ask it to predict the same window
shifted right by one. The answer to every question is just the next token, which was sitting
right there in the text the whole time. The labels are the data.

Every one of the 65536 positions gets a training signal out of this, and every one of them is
causally honest, thanks to the mask.

```python
self.current_position += B * T * self.num_processes   # line 285
```

and in `reset`:

```python
self.current_position = self.B * self.T * self.process_rank   # line 277
```

That is the multi-GPU interleave. With 8 GPUs, rank 0 starts at position 0, rank 1 starts at
65536, rank 7 starts at 458752, and every rank advances by 8 x 65536 on each call. They tile the
shard perfectly, with no overlap, and without ever talking to each other about it.

```python
if self.current_position + (B*T*self.num_processes + 1) > len(self.tokens):   # line 287
    self.current_shard = (self.current_shard + 1) % len(self.shards)
    self.tokens = load_tokens(self.shards[self.current_shard])
    self.current_position = B * T * self.process_rank
```

This looks ahead rather than reacting. If the _next_ batch would run off the end of the shard,
roll to the next shard now, instead of handing back a short batch. The `%` wraps around at the
end of the list, so training can run forever.

`reset()` exists so that validation always starts from shard 0, position 0. That way your
validation loss curve is comparable from step to step, instead of quietly drifting as the data
underneath it changes.

## 12. get_most_likely_row, the HellaSwag eval (lines 301 to 313)

HellaSwag hands you a context and 4 candidate endings, one of which is correct. There is no
classification head anywhere in this model, so you cannot just ask it which one. Instead you
score all 4 and pick the one the language model finds least surprising.

```python
shift_logits = (logits[..., :-1, :]).contiguous()   # line 302
shift_tokens = (tokens[..., 1:]).contiguous()       # line 303
```

The same off-by-one alignment as the data loader. The logits at position _t_ are a prediction
about token _t+1_, so you drop the last logit, which predicts a token you do not have, and the
first token, which nothing predicted.

```python
shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')   # line 306
```

`reduction='none'` is the key. Normally cross entropy averages everything down into one number.
Here you need the per-token losses kept separate so you can mask some of them out, so you ask
for the full `(4, T-1)` grid instead.

```python
shift_mask = (mask[..., 1:]).contiguous()          # line 308
masked_shift_losses = shift_losses * shift_mask    # line 309
avg_loss = sum_loss / shift_mask.sum(dim=1)        # line 311
pred_norm = avg_loss.argmin().item()               # line 312
```

The mask is 1 only on the completion tokens. All 4 rows share an identical context prefix, so
scoring the context would add the exact same number to every row. That helps you distinguish
nothing and only adds noise. Only the endings differentiate.

Then line 311. Loss is a cost, and every extra token adds more cost, so a long ending always
looks worse than a short one purely for being long. Comparing totals would be comparing the
price of a weekly shop against the price of a sandwich. Dividing by the number of tokens turns
it into cost per token, which is the price-per-item comparison you actually wanted. That
division is the "norm" in `pred_norm`. Without it you would systematically pick the shortest
option.

`argmin` because lower loss means higher probability, which means the model thinks that ending
is the most natural continuation.

## 13. Device and DDP setup (lines 315 to 343)

```python
ddp = int(os.environ.get('RANK', -1)) != -1   # line 316
```

A clean bit of detection. `torchrun` sets `RANK`, `LOCAL_RANK`, and `WORLD_SIZE` in the
environment. If `RANK` is missing, you were launched as a plain `python main.py`, so there is
one process and nothing to coordinate.

- `ddp_rank`: this process's global id across all machines, 0 to 7 on one 8-GPU box.
- `ddp_local_rank`: its id within _this_ machine, which is what picks the physical GPU.
- `master_process`: rank 0 only. The one process allowed to print, log, and checkpoint, so that
  8 processes do not write 8 copies of everything.

`init_process_group(backend='nccl')` starts up NVIDIA's collective communication library, which
is the thing that makes `all_reduce` fast over NVLink.

## 14. The batch size math (lines 347 to 358)

```python
total_batch_size = 524288   # line 347: 2^19, half a million tokens, from the GPT-3 paper
B = 64                      # line 348: what actually fits in GPU memory
T = 1024                    # line 349
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)   # line 352
```

This is the resolution of a genuine conflict. The paper says each optimizer step should see
about 0.5M tokens. Your GPU can hold 64 x 1024 = 65536 tokens at a time. You cannot have both at
once.

So you fake it. Run 8 forward and backward passes in a row. Let the gradients pile up in
`.grad`, which PyTorch does automatically, since `backward()` adds to `.grad` rather than
overwriting it. Only then call `optimizer.step()`.

The result is mathematically identical to one giant batch of 524288 tokens. It is just spread
out in time instead of held in memory all at once.

On 8 GPUs, `grad_accum_steps` drops to 1 and the same total batch gets assembled in parallel
instead of in sequence. Notice the dependency direction: the batch size is held fixed and the
code adapts around it. That is the correct way round.

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

**Line 497 is the one everybody gets wrong.**

You want the average loss across all 524288 tokens. But `F.cross_entropy` has already averaged
over the 65536 tokens inside each micro batch. So what you have is 8 separate averages, and
adding 8 averages together does not give you an average of anything.

Think of it as the class average across 8 classrooms. Each classroom hands you its own average:
80, 90, and so on. Add those 8 numbers and you get 680. You wanted 85. Since the classrooms are
all the same size, dividing each average by 8 before adding gets you exactly there.

That is line 497. And it has to happen up front, on each micro batch, because `backward()` only
knows how to add into `.grad`. There is no hook at the end where you could divide.

**Line 494** is performance surgery. By default DDP fires an `all_reduce` across all GPUs after
every single `backward()`. During accumulation, that would be 8 full syncs of 124M gradients
when you only ever needed one, at the end. Setting the flag to False on micro steps 0 through 6
and True on step 7 collapses it to a single sync.

**Line 498**, and specifically the `.detach()`. Without it, `loss_accum` would keep a reference
to the autograd graph of all 8 micro batches, and you would run out of memory. You only want the
number, for printing.

**Line 495**, the bfloat16 autocast.

A float is stored like scientific notation: `1.2345 x 10^-7`. The exponent bits are the `10^-7`
part, and they decide the smallest and largest numbers you can represent at all. The mantissa
bits are the `1.2345` part, and they decide how many significant digits you keep.

bf16 has the _same 8 exponent bits as fp32_ and simply fewer mantissa bits. It keeps fp32's full
range and gives up digits. So a tiny gradient still exists, just sloppily.

fp16 made the opposite trade. It has only 5 exponent bits, so it gave up range, and a tiny
gradient becomes exactly 0.0 and is gone forever. Sloppy is recoverable, gone is not. That is
why fp16 needs a `GradScaler` and why there is no `GradScaler` anywhere in this file.

Autocast is also selective about where it applies. Matmuls run in bf16, but softmax, layernorm,
and the loss stay in fp32, where precision actually matters. The master copy of the weights
stays fp32 throughout.

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # line 502
```

Take all 148 gradient tensors, pretend they are one long vector, and measure its length. If that
length is more than 1.0, scale everything down uniformly until it equals 1.0. The direction is
preserved exactly. Only the distance is capped.

The reason you want this: occasionally a batch is pathological. A weird document, a run of rare
tokens, and it produces a gradient 50x the normal size, which can wreck the model in a single
step. Clipping bounds the damage any one batch is allowed to do.

The returned `norm` is one of the best free diagnostics you have. It should start around 1 and
settle to roughly 0.1 to 0.3. **A sudden spike means a bad batch or the beginning of
instability**, and it usually shows up in the norm before it shows up in the loss.

```python
if device_type == "cuda":
    torch.cuda.synchronize()   # line 508
```

This is needed for honest timing. CUDA calls are asynchronous, so `optimizer.step()` returns the
instant the work has been _queued_, not when it has been _done_. Without this line you would be
timing how fast Python can hand out instructions, which is a beautiful and completely fictitious
number.

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

Line 467 is the strange asymmetry at the heart of this architecture. During _training_, all 1024
positions produce useful predictions at once, and you use every one of them. During
_generation_, you compute all 1024 and throw away 1023 of them, keeping only the last.

That waste is exactly what KV-caching exists to fix, and this code does not implement it. Which
is fine. It is generating 32 tokens for a vibe check, not serving traffic.

**Top-k = 50** is the quality guard. The tail of a 50304-way softmax holds tens of thousands of
tokens, each with a tiny probability, but collectively they add up to something non-trivial.
Sample naively for long enough and you will eventually draw something absurd. And because the
model then conditions on its own mistake, it derails and never recovers. Truncating to the top
50 and renormalizing keeps every sample inside the plausible region.

Line 475 samples an _index into the top-50 list_, not a vocabulary id. So line 477's `gather` is
required to translate it back into a real token. Miss that step and you decode garbage.

**The knob that is missing here is temperature.** The usual trick is to divide the logits by a
number T before the softmax. A large T flattens the distribution, giving the unlikely tokens
more of a share and producing weirder text. A small T sharpens it toward the top choice, giving
safer and more repetitive text. T = 1 is the distribution exactly as the model reports it, and
that is effectively what this code does, since line 469 softmaxes the logits untouched. So this
file controls sampling entirely through top-k, not through temperature.

Sampling rather than `argmax` because greedy decoding falls into repetition loops almost
immediately. The `sample_rng` is a separate `Generator`, seeded at `42 + ddp_rank`, kept
deliberately off the global RNG so that sampling does not perturb the order of the training
data, and so each rank produces different samples.

## 17. Things to know about this specific file

**Lines 16 to 23 are dead.** `block_size = 256`, `n_embd = 384`, `n_head = 6`, `n_layer = 6`,
`dropout = 0.2`, `learning_rate = 3e-4`, `eval_interval`, `eval_iters`. Nothing reads any of
them. Worth flagging: there is **no dropout anywhere in this model**. That is correct for a
single-epoch 10B-token pretraining run, where you cannot really overfit, but the variable
sitting there suggests otherwise.

**`vocab_size=50304` on line 362**, while `GPTConfig` defaults to 50257. This is the "ugly number
to nice number" trick. 50304 is 128 x 393. CUDA matmul kernels work in tiles sized by powers of
two, and a dimension of 50257 forces an inefficient ragged remainder tile at the edge. So you
add 47 tokens that can never appear in the data, the model quickly learns to give them near-zero
probability, and you get a real speedup for free. Adding parameters in order to go faster is
unintuitive and correct.

**MPS and autocast.** Line 339 sets `device_type = "cuda" if device.startswith("cuda") else
"cpu"`. So on an M-series Mac the device is `mps` while `device_type` is `cpu`. And
`torch.autocast(device_type="cpu", ...)` does not intercept mps ops, which means the bf16
autocast is a **silent no-op** there. No crash, just fp32. The same goes for
`torch.set_float32_matmul_precision('high')` on line 360, which only affects CUDA TF32.

**B=64, T=1024 needs roughly 80GB.** To smoke test locally, drop to `B=2` or `B=4` and cut
`max_steps` to something like 20.

**Timing on MPS is fictional.** Line 507 only synchronizes on CUDA, so `dt` and `tok/sec` on
Apple silicon are measuring queue submission, not real work.

**`master_process` on line 269** is read inside `DataLoaderLite.__init__`, but it is defined at
module scope on lines 325 and 331. It works because instantiation on line 357 happens after
that, but it is an implicit global dependency that would break the moment you imported this
class from another file.

**There is no resume.** The checkpoint dict at line 407 saves the model, the config, the step,
and the val loss. It does not save optimizer state or RNG state, and nothing in the file ever
reads it back. If the run dies at step 12000, you restart from zero.

## Part I compression

1. **Attention is the only place information moves between positions.** Everything else is
   per-token.
2. **The residual stream is a conveyor belt** that every layer reads from and adds to, never
   overwrites. That is what makes 12 layers trainable and 96 layers possible.
3. **The causal mask is what makes training parallel.** 1024 honest predictions from one forward
   pass.
4. **`sqrt(head_size)` and `(2*n_layer)^-0.5` are the same idea twice:** keep the variance at 1,
   or the softmax saturates and the activations explode.
5. **Half this file is not the model.** It is machinery for training the model fast: grad
   accumulation, bf16, flash attention, fused Adam, DDP, vocab padding. The model is about 60
   lines. The engineering is 400.

---

# Part II: what happens across 19,073 steps

## 1. The landscape

The model has **124,475,904 parameters**. Every one of them is a knob you can turn.

So picture a space with 124,475,904 axes. One point in that space fixes every knob at once,
which is to say one point _is_ one entire model. Now hang a number over every point: the average
cross-entropy loss that model gets on FineWeb. Read that number as **height**.

You now have a landscape. It is not a 3D landscape, but the intuition survives the dimension
count much better than you would expect. And training is one thing: **you are standing somewhere
on this surface and walking downhill.**

Line 128, `self.apply(self._init_weights)`, is where you get dropped in. Random Gaussians, std
0.02. That starting point sits at height **10.82**, which is not an arbitrary altitude. It is
exactly `log(50304)`, the loss of a model that gives equal probability to every token in the
vocabulary. You start at the altitude of "knows literally nothing".

19,073 steps later you want to be at about **3.29**. That is the whole job.

## 2. One step, in slow motion

**First, feel the ground.** You need to know which way is downhill.

With one knob, you would just wiggle it and see whether the loss went up or down. With 124
million knobs, wiggling each one separately means 124 million forward passes for a single step.
That is not slow. That is impossible.

**So instead you compute the gradient.** The gradient is a vector with 124,475,904 components,
one per knob. Component _i_ answers exactly one question: _if I nudge knob i up by a hair, how
much does the loss change?_ Point that vector's direction and you are pointing straight uphill.
Negate it and you have your direction of travel.

Two properties of that vector matter, and people usually only internalize the first.

- Its **direction** is the way to go.
- Its **magnitude, component by component,** tells you which knobs matter. A weight with a large
  partial derivative is one the loss cares a lot about. A weight with a near-zero partial
  derivative is currently irrelevant and will barely move. The gradient is simultaneously a
  direction and a ranking of importance.

**And one backward pass gets you all of it.** This is the miracle, and the part actually worth
understanding. `loss.backward()` on line 499 computes all 124 million partial derivatives for
roughly **twice the cost of one forward pass.** Not 124 million times. Twice.

The reason is that the chain rule runs backwards, and each step reuses the last one.

Start at the loss, where `dL/dL = 1`. Push that back through `lm_head`, and now you know how the
loss responds to every one of the final 768-number vectors. Push it back through `ln_f`, then
block 12, then block 11, and so on down to the embeddings. At every layer you are already
holding "how does the loss respond to my output", and that layer's own local derivative turns it
into "how does the loss respond to my inputs, and to my weights". Every quantity gets computed
exactly once and reused by everything downstream of it.

That is the entire reason deep learning exists as a practical field. Not the architecture. This.

**One catch: the gradient you compute is a guess.** The gradient you actually want is the one for
the true distribution of all English text, and that is not available to you. What you get
instead is the gradient on 524,288 tokens sampled from FineWeb. That is a **noisy estimate**:
right on average, wrong on any particular step.

Which is what line 347's `total_batch_size = 524288` is buying. Bigger batch, less noise,
straighter walk. Too small a batch and you stagger drunkenly, and can outright fall over. The
noise is not purely harmful, a little of it helps rattle you out of bad regions, but at this
scale you want it controlled.

And that is why grad accumulation on lines 490 to 499 is not a hack. Eight micro-batches of
65,536 tokens, gradients piling up in `.grad`, is arithmetically identical to one honest batch
of 524,288. You are assembling one confident measurement out of eight cheap ones.

## 3. Adam, or why you do not just walk downhill

Plain gradient descent says: `param -= lr * grad`. Take the direction, take a fixed-size step,
repeat.

Nobody trains transformers that way, and the reason is a shape.

The landscape is not a nice round bowl. It is a **long narrow ravine**. Some directions are steep
cliffs. Others are nearly flat valley floors stretching for miles. One global step size cannot
serve both. Big enough to make progress along the flat valley means you ricochet violently off
the cliff walls. Small enough to be stable on the cliffs means you crawl along the valley floor
forever.

AdamW, line 235, fixes this with two ideas stacked on top of each other.

**Momentum, beta1 = 0.9.** Instead of stepping along the current gradient, step along a running
average of recent gradients. Physically: you are a ball with mass rolling downhill, not a point
teleporting from place to place. Directions that are consistently downhill build up speed. The
noisy back-and-forth components, the ones that came from which batch you happened to draw,
cancel themselves out. `0.9` works out to roughly a 10-step memory.

**Per-parameter step size, beta2 = 0.95.** Track a running average of each gradient component
_squared_, then divide that component's step by its square root. The effect: a weight that has
been getting huge gradients takes a small step, and a weight getting tiny gradients takes a
proportionally larger one. **Every one of the 124 million dimensions gets its own ruler, and the
rulers get rescaled continuously.** Which stretches the ravine into something much closer to a
round bowl before you step.

That is why 0.999 was lowered to 0.95 here. Beta2 is the memory of that rescaling. 0.999
remembers about 1000 steps, which is far too sluggish when the landscape is genuinely reshaping
itself over the first thousand steps of training. 0.95 remembers about 20, and keeps up.

**The W in AdamW.** Weight decay pulls every 2D weight slightly toward zero on every step,
regardless of what the gradient says. Constant, gentle downward pressure on magnitude. Only the
matmul weights get it, line 220, because shrinking a bias or a LayerNorm gain is just vandalism.

**Gradient clipping, line 502,** is the seatbelt. Take all 124 million gradient components as one
enormous vector, measure its length, and if it is longer than 1.0, scale the whole thing down
until it is exactly 1.0. Direction preserved perfectly, only distance capped. Every so often you
draw a genuinely pathological batch that produces a gradient 50 times the normal size, and
without the clip that one step launches you off the map.

## 4. The step size changes on purpose

`get_lr` on lines 30 to 38, but as geometry.

**Warmup, steps 0 to 715.** At initialization you are standing at a random point on a landscape
you have measured exactly zero times. The gradients are enormous and mostly wrong. Adam's two
running averages start at zero and are not yet meaningful. Take full-size steps here and you
reliably blow the loss up within a dozen iterations. So the learning rate ramps linearly from
near 0 up to 6e-4. You are letting your eyes adjust before you start running.

**Cosine decay, steps 715 to 19,073.** Early on you are far from anywhere good, and precision is
worthless. You do not care whether you land at the exact bottom of _this_ valley, you care about
finding the right continent. Big confident strides. Later you are already in a good basin, and
all that is left is settling into its lowest point, which needs small careful steps. The cosine
shape holds you near the maximum for a while, then eases off smoothly, then flattens gently at
the end rather than slamming to a halt.

**Floor at 10%.** Never exactly zero, because a learning rate of zero means learning has stopped,
and the last stretch of your run would be pure heat.

The mental image: you are hunting for the lowest point in a mountain range. On day one you
drive. In the final hour you walk in slow circles, watching your altimeter.

## 5. What the loss number actually means

Cross entropy is `-log(the probability the model gave the correct next token)`, averaged over
every position. It comes out in nats, which are not intuitive. Exponentiate it and you get
**perplexity**, which is:

| loss  | perplexity | plain English                                      |
| ----- | ---------- | -------------------------------------------------- |
| 10.82 | 50,304     | uniform over the entire vocabulary. knows nothing. |
| 6.9   | ~1,000     | has learned which tokens are common.               |
| 4.6   | ~100       | local grammar works.                               |
| 3.9   | ~50        | fluent-ish, mostly coherent sentences.             |
| 3.29  | ~27        | GPT-2 124M territory.                              |

Perplexity is "how many options is the model effectively torn between". At step 0 it is
genuinely torn between 50,304 of them. At the end it is torn between about 27. That is the whole
run: going from 50,304-way confusion down to 27-way confusion.

Now notice that this is a **logarithmic** scale, and that is what makes the loss curve so cruelly
shaped.

Going from 10.8 to 6.9 cuts the confusion by 50x, and it happens in the first minute or two.
Going from 3.4 to 3.29 is a few percent, and it takes hours. **Each additional nat costs
exponentially more compute than the one before it.** That single fact is why frontier labs spend
nine figures on training runs.

## 6. The curriculum nobody wrote

Nothing in the file orders the lessons. There is no syllabus. And yet the model learns things in
a strikingly consistent order, because gradient descent is greedy: at every step it grabs
whatever loss reduction is cheapest right now. Step counts below are approximate.

**Steps 0 to ~50: unigram statistics.** The cheapest win available is to stop pretending all
tokens are equally likely. ` the` is roughly 5% of English text. `<|endoftext|>` shows up at
every document boundary. The model can pick all of this up while ignoring its input entirely.
The loss falls off a cliff here, 10.8 down to roughly 7, in under a minute. Almost all of the
visually dramatic drop in your loss curve is this, and it is the least interesting thing that
happens in the entire run.

**Steps ~50 to ~500: local context.** Attention heads start firing on the previous token or two.
The model learns that ` United` is often followed by ` States`, that a `(` eventually wants a
`)`, that a space usually follows a period. Roughly bigram and trigram behavior, delivered by
the `wpe` position embeddings and the earliest attention heads. Loss drops into the 5s.

**Steps ~500 to ~3,000: syntax and induction.** This is the important qualitative shift.
**Induction heads** form: pairs of attention heads that, having seen the pattern `[A][B]` earlier
in the context, will predict `[B]` the next time `[A]` shows up. That is the first genuinely
non-local, in-context capability the model has, and it is what "copy the name that was mentioned
400 tokens ago" is made out of. In many training runs this appears as a visible bump or a brief
plateau in the loss curve, because it is a discrete circuit snapping into place rather than a
smooth parameter drift. Loss into the low 4s.

**Steps ~3,000 to 19,073: everything else, slowly.** Facts get packed into the MLPs. Long-range
coherence improves. Style, register, subject-verb agreement across clauses. The loss curve looks
almost flat through here, and this is where the majority of your compute goes and where nearly
all of the actual quality lives.

Which is exactly why the code generates samples every 250 steps, lines 450 to 484, and runs
HellaSwag. Once the loss curve goes visually flat it stops telling you much. The samples still
change dramatically. HellaSwag crawling up off 0.25, which is random guessing among 4 options,
toward ~0.30 is measuring something that the fourth decimal place of the loss will never show
you.

## 7. Why the curve is smooth when the landscape is not

Here is a reasonable objection. With 124 million knobs and a wildly bumpy loss surface, why does
any of this work? Should you not get stuck in some bad local minimum immediately?

The resolution is genuinely counterintuitive: **high dimension makes this easier, not harder.**

For a point to be a local minimum, the surface has to curve _upward_ in every single one of the
124 million directions. Every one. If even one direction curves downward, you are not stuck at
all. You are on a **saddle point**, and you can keep going.

The odds of 124 million roughly independent directions all happening to curve up, at a point that
is not already globally good, are effectively nil. So bad local minima essentially do not occur
at this scale. What you get instead is saddle points and long flat plateaus, which are annoying
because they slow you down, but are not fatal. And momentum is specifically good at coasting
through them.

There is a second smoothing effect, and it is about the measurement rather than the landscape.
Your reported loss is averaged over 524,288 tokens. Individual tokens vary wildly, some trivially
predictable and some genuinely impossible. Average half a million of them per step and you get a
number with tiny variance. So the curve you see is smooth partly because the landscape is more
forgiving than intuition suggests, and partly because you are looking at it through a very large
averaging window.

## 8. Reading the scrolling numbers

```
step  1247 | loss: 4.238715 | lr 5.9821e-04 | norm: 0.2841 | dt: 1043.21ms | tok/sec: 502654.19
```

- **loss** should fall fast, then flatten. Small jitter from step to step is batch noise and is
  expected. A sustained rise means trouble.
- **lr** should trace the schedule exactly: linear up for 715 steps, then cosine down. It is a
  printout of an equation, so if it looks wrong you have a bug, not a training problem.
- **norm** is your best early-warning instrument. It should start near 1 and settle into 0.1 to
  0.3. **A spike here comes before a loss spike.** If it pins at exactly 1.0 constantly, you are
  clipping on every single step, which means your learning rate is too high for your batch size.
- **dt** should be nearly constant. Creeping upward suggests thermal throttling or a memory
  issue.
- **tok/sec** is your efficiency number. Multiply it out against 19,073 x 524,288 and it tells
  you when you will be done.

## 9. The size of the thing

The standard estimate for transformer training cost is `6 x params x tokens`:

```
6 x 124,475,904 x 10,000,000,000  =  approximately 7.5 x 10^18 floating point operations
```

Where does the 6 come from? Every parameter is used in one multiply and one add per token, so
that is 2 operations for the forward pass. The backward pass costs twice the forward, because it
has to compute both "how does the loss respond to my inputs" and "how does the loss respond to my
weights". That is 4 more. 2 + 4 = 6, per parameter, per token.

So 7.5 **exaFLOPs** to move one number from 10.82 to 3.29.

For scale: if you could do one multiplication per second by hand, without ever sleeping, it would
take you about 240 billion years. This run does it in a couple of hours on 8 GPUs.

And the whole apparatus, the flash attention, the bf16 autocast, the fused Adam, the vocab padded
to 50304, the fused QKV projection, exists for exactly one reason. To move that finish line from
"next month" to "before dinner".

## Part II compression

1. **Training is walking downhill on a 124-million-dimensional surface.** That is the entire
   mental model.
2. **Backprop gets you all 124 million partial derivatives for the price of two forward passes.**
   Everything else is engineering. This is the actual idea.
3. **Your gradient is a noisy estimate.** Batch size buys you accuracy. That is what the 524,288
   is for.
4. **Adam gives every dimension its own ruler,** which turns a ravine into something walkable.
5. **Loss is log-scale, so progress is exponentially expensive.** The dramatic early drop is the
   model learning that ` the` is common. The flat boring part is where the intelligence goes in.

Watch `norm` more closely than `loss`. The loss tells you where you have been. The norm tells you
whether you are about to have a problem.

---

# Appendix: running it

```bash
git clone https://github.com/andrewn6/ml-from-scratch.git
cd ml-from-scratch
uv venv --python 3.14
source .venv/bin/activate
uv pip install torch numpy tiktoken transformers datasets tqdm
```

Do not use `uv sync` or `uv run` in this repo on Linux. `pyproject.toml` pulls in `manim`, which
pulls `pycairo`, which has no Linux wheels and needs `libcairo2-dev` to build.

Data, which writes to `gpt2/edu_fineweb10B/`. That directory is gitignored and must be
regenerated:

```bash
cd gpt2
python fineweb.py --shards 2      # pipeline check, ~400MB. shard 0 = val, shard 1 = train
python fineweb.py                 # real run, 100 shards, 10B tokens, ~20GB
```

`--shards 2` gives you one train shard of 100M tokens. `max_steps = 19073` consumes 10B tokens,
so the loader wraps around and re-reads that one shard about 100 times. Fine for a smoke test,
useless as a real run.

Smoke test, and revert afterwards:

```bash
sed -i 's/^max_steps = 19073.*/max_steps = 20/; s/^warmup_steps = 715/warmup_steps = 5/; s/^B = 64 /B = 4 /' main.py
python main.py
git checkout main.py
```

What to check, in order. It finds the shards. It prints `num decayed parameter tensors: 50, with
124,354,560 parameters` and `98 / 121,344`. The first validation loss is about 10.82. The train
loss drops off 10.8 within a few steps. `norm` is finite and around 1.

Train:

```bash
python main.py                                      # 1 gpu
torchrun --standalone --nproc_per_node=8 main.py    # 8 gpus
```

`main.py` reads `edu_fineweb10B` relative to your current directory, so it has to be launched
from inside `gpt2/`.

Set `B` to fit your card. Leave `total_batch_size = 524288` alone, grad accum absorbs the change.
Keep `B` a power of two or the assert on line 350 trips.

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

Healthy end state: train loss under 3.3, HellaSwag climbing off 0.25 toward ~0.30, and `norm`
sitting in the 0.1 to 0.3 band.

"""
=============================================================================
DAY 3A: EXPERIMENTS — Break the Model to Understand It
=============================================================================
Author: Jathin | Healthcare LLM Project

We run 5 experiments, each removing or changing ONE thing from the
Day 2 Mini GPT. By seeing what breaks, you deeply understand why
each component exists.

How to run:
    python 03a_experiments.py

This takes ~15-20 minutes total (5 experiments × ~3 min each)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# =============================================================================
# CONFIGURATION
# =============================================================================

batch_size = 32
block_size = 64
max_iters = 2000       # Shorter training — enough to see differences
eval_interval = 500
learning_rate = 3e-4
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2

torch.manual_seed(42)


# =============================================================================
# DATA LOADING (same as Day 2)
# =============================================================================

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]


def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - block_size, (batch_size,))
    x = torch.stack([d[i:i+block_size] for i in ix]).to(device)
    y = torch.stack([d[i+1:i+block_size+1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# =============================================================================
# BUILDING BLOCKS (modular so we can swap pieces)
# =============================================================================

class SingleHead(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        return weights @ v


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SingleHead(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# EXPERIMENT MODELS
# =============================================================================

class FullModel(nn.Module):
    """Complete Mini GPT from Day 2 (baseline)."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _block(self):
        class Block(nn.Module):
            def __init__(self_b):
                super().__init__()
                self_b.sa = MultiHead(n_head, n_embd // n_head)
                self_b.ffwd = FeedForward()
                self_b.ln1 = nn.LayerNorm(n_embd)
                self_b.ln2 = nn.LayerNorm(n_embd)
            def forward(self_b, x):
                x = x + self_b.sa(self_b.ln1(x))
                x = x + self_b.ffwd(self_b.ln2(x))
                return x
        return Block()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class NoPositionModel(nn.Module):
    """EXPERIMENT 1: Remove position embeddings."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # NO position embedding!
        self.blocks = nn.Sequential(*[self._block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _block(self):
        class Block(nn.Module):
            def __init__(self_b):
                super().__init__()
                self_b.sa = MultiHead(n_head, n_embd // n_head)
                self_b.ffwd = FeedForward()
                self_b.ln1 = nn.LayerNorm(n_embd)
                self_b.ln2 = nn.LayerNorm(n_embd)
            def forward(self_b, x):
                x = x + self_b.sa(self_b.ln1(x))
                x = x + self_b.ffwd(self_b.ln2(x))
                return x
        return Block()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.tok_emb(idx)  # NO position embedding added!
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class SingleHeadModel(nn.Module):
    """EXPERIMENT 2: Use 1 attention head instead of 4."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _block(self):
        class Block(nn.Module):
            def __init__(self_b):
                super().__init__()
                # 1 head with full n_embd size instead of 4 heads with n_embd//4
                self_b.sa = MultiHead(1, n_embd)
                self_b.ffwd = FeedForward()
                self_b.ln1 = nn.LayerNorm(n_embd)
                self_b.ln2 = nn.LayerNorm(n_embd)
            def forward(self_b, x):
                x = x + self_b.sa(self_b.ln1(x))
                x = x + self_b.ffwd(self_b.ln2(x))
                return x
        return Block()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class NoFFNModel(nn.Module):
    """EXPERIMENT 3: Remove feed-forward networks."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _block(self):
        class Block(nn.Module):
            def __init__(self_b):
                super().__init__()
                self_b.sa = MultiHead(n_head, n_embd // n_head)
                # NO feed-forward network!
                self_b.ln1 = nn.LayerNorm(n_embd)
            def forward(self_b, x):
                x = x + self_b.sa(self_b.ln1(x))
                # No FFN step!
                return x
        return Block()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


class NoResidualModel(nn.Module):
    """EXPERIMENT 4: Remove residual connections."""
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[self._block() for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def _block(self):
        class Block(nn.Module):
            def __init__(self_b):
                super().__init__()
                self_b.sa = MultiHead(n_head, n_embd // n_head)
                self_b.ffwd = FeedForward()
                self_b.ln1 = nn.LayerNorm(n_embd)
                self_b.ln2 = nn.LayerNorm(n_embd)
            def forward(self_b, x):
                # NO residual connections! (no "+ x")
                x = self_b.sa(self_b.ln1(x))       # was: x + self_b.sa(...)
                x = self_b.ffwd(self_b.ln2(x))     # was: x + self_b.ffwd(...)
                return x
        return Block()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok = self.tok_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=device))
        x = tok + pos
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =============================================================================
# RUN ALL EXPERIMENTS
# =============================================================================

def train_model(model, name, iters=2000):
    """Train a model and return final val loss + sample text."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    params = sum(p.numel() for p in model.parameters())

    start = time.time()
    for i in range(iters + 1):
        if i % eval_interval == 0:
            losses = estimate_loss(model)
            elapsed = time.time() - start
            print(f"  Step {i:4d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f} | {elapsed:.0f}s")

        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Final loss
    final = estimate_loss(model)

    # Generate sample
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = decode(model.generate(ctx, 200)[0].tolist())

    return final['val'].item(), params, sample


def main():
    print("=" * 70)
    print("DAY 3A: EXPERIMENTS — What Happens When We Break Things?")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training each model for {max_iters} iterations\n")

    experiments = [
        ("BASELINE (Full Model)",       FullModel),
        ("NO POSITION EMBEDDINGS",      NoPositionModel),
        ("SINGLE ATTENTION HEAD",       SingleHeadModel),
        ("NO FEED-FORWARD NETWORK",     NoFFNModel),
        ("NO RESIDUAL CONNECTIONS",     NoResidualModel),
    ]

    results = []

    for name, ModelClass in experiments:
        print(f"\n{'='*70}")
        print(f"EXPERIMENT: {name}")
        print(f"{'='*70}")

        torch.manual_seed(42)  # Same initialization for fair comparison
        model = ModelClass()
        val_loss, params, sample = train_model(model, name)
        results.append((name, val_loss, params, sample))

        print(f"\n  Sample output:")
        print(f"  {sample[:150]}...")

    # =============================================================================
    # RESULTS SUMMARY
    # =============================================================================

    print("\n" + "=" * 70)
    print("EXPERIMENT RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Experiment':<30s} | {'Val Loss':>9s} | {'Params':>10s} | {'vs Baseline':>12s}")
    print("-" * 70)

    baseline_loss = results[0][1]

    for name, val_loss, params, _ in results:
        diff = val_loss - baseline_loss
        diff_str = f"+{diff:.3f} WORSE" if diff > 0.01 else f"{diff:+.3f}"
        if name.startswith("BASELINE"):
            diff_str = "—"
        print(f"{name:<30s} | {val_loss:9.4f} | {params:>10,d} | {diff_str:>12s}")

    print(f"""
{'='*70}
WHAT YOU LEARNED:
{'='*70}

1. NO POSITION EMBEDDINGS → Loss increased
   The model can't tell "cat sat" from "sat cat".
   Position info is ESSENTIAL for understanding word order.

2. SINGLE ATTENTION HEAD → Loss increased
   One head = one way of looking at context.
   Multiple heads = multiple perspectives (spelling, grammar, meaning).
   Diversity of attention patterns matters!

3. NO FEED-FORWARD → Loss increased significantly
   Attention GATHERS info but can't PROCESS it.
   FFN does the "thinking" — it's where most parameters live.
   Like reading a medical chart but not being able to reason about it.

4. NO RESIDUAL CONNECTIONS → Loss increased the most
   Without skip connections, gradients vanish in deep networks.
   The model essentially can't learn.
   This is why ResNets (2015) were such a breakthrough.

KEY INSIGHT: Every component exists for a reason. Remove any one
and the model gets measurably worse. This is why the Transformer
architecture has dominated AI for 8+ years — it's well-designed.
""")


if __name__ == '__main__':
    main()
"""
=============================================================================
DAY 3C: SCALED GPT — Bigger Model + Pro Training Techniques
=============================================================================
Author: Jathin | Healthcare LLM Project

What's new vs Day 2:
  - 4x bigger model (~400K+ params vs 211K)
  - Learning rate warmup + cosine decay (used by GPT-3, LLaMA)
  - Gradient accumulation (train with effectively larger batches)
  - AdamW optimizer with weight decay
  - GELU activation (what GPT uses, instead of ReLU)
  - Longer context window (128 vs 64)

How to run:
    python 03c_scaled_gpt.py
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    batch_size = 32
    block_size = 128
    grad_accum_steps = 2
    max_iters = 3000
    eval_interval = 100
    eval_iters = 30
    max_lr = 6e-4
    min_lr = 6e-5
    warmup_iters = 100
    n_embd = 96
    n_head = 6
    n_layer = 6
    dropout = 0.15
    weight_decay = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(it, config):
    """Linear warmup + Cosine decay. Same schedule as GPT-3 and LLaMA."""
    if it < config.warmup_iters:
        return config.max_lr * (it + 1) / config.warmup_iters

    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# =============================================================================
# MODEL COMPONENTS
# =============================================================================

class GELU(nn.Module):
    """Gaussian Error Linear Unit — activation function used by GPT."""
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


class CausalSelfAttention(nn.Module):
    """Efficient multi-head attention in a SINGLE module."""

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

        self.register_buffer("bias", torch.tril(
            torch.ones(config.block_size, config.block_size)
        ).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (head_size ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class FeedForward(nn.Module):
    """Feed-forward with GELU activation (GPT-style)."""
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with Pre-LN (GPT-2 style)."""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = FeedForward(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ScaledGPT(nn.Module):
    """
    Scaled GPT model with professional training features.

    New vs Day 2:
    - Efficient multi-head attention (single matrix multiply)
    - GELU activation instead of ReLU
    - Weight initialization (GPT-2 style)
    - Weight tying between embedding and output
    """

    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config

        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        param_count = sum(p.numel() for p in self.parameters())
        print(f"   Parameters: {param_count:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)

        for block in self.h:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate with optional top-k sampling."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# DATA
# =============================================================================

class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])


def get_batch(train_data, val_data, split, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
    return x, y


# =============================================================================
# TRAINING
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(train_data, val_data, split, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def run_training(model, train_data, val_data, tokenizer, config):
    """Training loop with gradient accumulation and LR scheduling."""

    decay_params = []
    nodecay_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=config.max_lr, betas=(0.9, 0.95))

    print("=" * 65)
    print("TRAINING SCALED GPT")
    print("=" * 65)
    print(f"  Device:           {config.device}")
    print(f"  Parameters:       {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Context size:     {config.block_size}")
    print(f"  Batch size:       {config.batch_size} x {config.grad_accum_steps} accum = {config.batch_size * config.grad_accum_steps} effective")
    print(f"  Embedding dim:    {config.n_embd}")
    print(f"  Heads:            {config.n_head}")
    print(f"  Layers:           {config.n_layer}")
    print(f"  LR schedule:      warmup {config.warmup_iters} then cosine decay")
    print(f"  Peak LR:          {config.max_lr}")
    print(f"  Weight decay:     {config.weight_decay}")
    print(f"  Activation:       GELU (same as GPT)")
    print(f"  Total steps:      {config.max_iters}")
    print()
    print(f"  {'Step':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10} | {'Time':>8}")
    print("  " + "-" * 60)

    steps_list = []
    train_losses = []
    val_losses = []
    lr_history = []
    start_time = time.time()

    for iter_num in range(config.max_iters + 1):

        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            lr = get_lr(iter_num, config)
            elapsed = time.time() - start_time
            pct = iter_num / config.max_iters * 100
            print(f"  {iter_num:6d} | {losses['train']:11.4f} | {losses['val']:11.4f} | {lr:10.6f} | {elapsed:7.1f}s | {pct:.0f}%")

            steps_list.append(iter_num)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())
            lr_history.append(lr)

        lr = get_lr(iter_num, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        optimizer.zero_grad(set_to_none=True)

        for micro_step in range(config.grad_accum_steps):
            xb, yb = get_batch(train_data, val_data, 'train', config)
            _, loss = model(xb, yb)
            loss = loss / config.grad_accum_steps
            loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total_time = time.time() - start_time
    print()
    print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss:   {val_losses[-1]:.4f}")

    return steps_list, train_losses, val_losses, lr_history


def plot_results(steps, train_losses, val_losses, lr_history, config):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(steps, train_losses, 'b-', label='Train', linewidth=2)
    ax1.plot(steps, val_losses, 'r-', label='Val', linewidth=2)
    ax1.axhline(y=2.58, color='gray', linestyle=':', label='Bigram (Day 1)', linewidth=1)
    ax1.axhline(y=1.97, color='orange', linestyle='--', label='Mini GPT (Day 2)', linewidth=1)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Scaled GPT Training - Day 3')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    all_lrs = [get_lr(i, config) for i in range(config.max_iters)]
    ax2.plot(range(config.max_iters), all_lrs, 'g-', linewidth=2)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('LR Schedule: Warmup + Cosine Decay')
    ax2.grid(True, alpha=0.3)
    ax2.axvline(x=config.warmup_iters, color='red', linestyle='--', label='Warmup ends')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('training_loss_day3.png', dpi=150)
    print(f"\n  Plot saved to: training_loss_day3.png")


# =============================================================================
# GENERATION SHOWCASE
# =============================================================================

def showcase(model, tokenizer, config):
    model.eval()
    print("\n" + "=" * 65)
    print("GENERATION SHOWCASE: Day 1 vs Day 2 vs Day 3")
    print("=" * 65)

    ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    print("\n--- Standard Generation (500 chars, temp=0.8) ---")
    with torch.no_grad():
        out = model.generate(ctx, 500, temperature=0.8)
    print(tokenizer.decode(out[0].tolist()))

    print("\n--- Top-k=10 Sampling (200 chars) --- [NEW!]")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    with torch.no_grad():
        out = model.generate(ctx, 200, temperature=0.8, top_k=10)
    print(tokenizer.decode(out[0].tolist()))

    print("\n--- Top-k=5 Sampling (200 chars) --- [Very focused]")
    ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    with torch.no_grad():
        out = model.generate(ctx, 200, temperature=0.7, top_k=5)
    print(tokenizer.decode(out[0].tolist()))

    print("""
=================================================================
PROGRESSION SUMMARY
=================================================================

  Day 1 (Bigram):     4,225 params  | 1 char context  | Loss: 2.58
  Day 2 (Mini GPT):   211,777 params | 64 char context | Loss: 1.97
  Day 3 (Scaled GPT): ~400K params  | 128 char context | Loss: ???

  New techniques today:
  - GELU activation (what GPT uses)
  - Learning rate warmup + cosine decay
  - Gradient accumulation
  - Top-k sampling
  - Weight tying
  - Gradient clipping
  - AdamW with weight decay
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='input.txt')
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = Config()
    if args.iters:
        config.max_iters = args.iters
    if args.device:
        config.device = args.device

    torch.manual_seed(config.seed)

    print("=" * 65)
    print("  DAY 3C: SCALED GPT")
    print("  Bigger Model + Pro Training Techniques")
    print("  Estimated time: ~5-8 minutes on CPU")
    print("=" * 65)

    if not os.path.exists(args.data):
        print(f"\n  Data file '{args.data}' not found!")
        print(f"  Make sure input.txt is in the same folder.")
        return

    print(f"\n  Loading data...")
    with open(args.data, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"  Vocab: {tokenizer.vocab_size} | Train: {len(train_data):,} | Val: {len(val_data):,}")

    print(f"\n  Creating Scaled GPT...")
    model = ScaledGPT(tokenizer.vocab_size, config).to(config.device)
    print()

    steps, train_losses, val_losses, lr_history = run_training(
        model, train_data, val_data, tokenizer, config
    )

    plot_results(steps, train_losses, val_losses, lr_history, config)

    showcase(model, tokenizer, config)

    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_chars': tokenizer.chars,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        },
    }, 'scaled_gpt_model.pt')
    print(f"  Model saved to: scaled_gpt_model.pt")

    print("""
=================================================================
  DAY 3 COMPLETE!
=================================================================

  Today you learned and implemented:
    - Learning rate warmup + cosine decay
    - Gradient accumulation
    - GELU activation (what GPT uses)
    - Weight tying (reduces params + improves quality)
    - Top-k sampling
    - Gradient clipping
    - AdamW with weight decay

  Push to GitHub:
    git add .
    git commit -m "Day 3: Experiments + BPE tokenizer + Scaled GPT"
    git push

  You are now using the SAME training techniques as GPT-3 and LLaMA.
""")


if __name__ == '__main__':
    main()
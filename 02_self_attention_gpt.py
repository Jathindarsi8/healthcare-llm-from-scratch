"""
=============================================================================
DAY 2: SELF-ATTENTION + MINI GPT FROM SCRATCH
=============================================================================
Author: Jathin | Healthcare LLM Project
Description: Building on the bigram model from Day 1, we add:
             1. Self-attention (single head)
             2. Multi-head attention
             3. Transformer blocks (attention + feed-forward)
             4. Position embeddings
             → A complete mini GPT!

How to run:
    python 02_self_attention_gpt.py

    Optional flags:
    python 02_self_attention_gpt.py --iters 5000      (default: 5000)
    python 02_self_attention_gpt.py --device cuda      (if you have a GPU)

Reference: Andrej Karpathy's "Let's build GPT" lecture
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """
    Hyperparameters for our mini GPT.

    Compared to yesterday's bigram model:
    - Added: n_embd, n_head, n_layer, dropout
    - Increased: block_size (8 → 64), max_iters (5000 → 5000)

    Compared to real GPT-2 Small:
    - GPT-2: n_embd=768, n_head=12, n_layer=12, block_size=1024
    - Ours:  n_embd=64,  n_head=4,  n_layer=4,  block_size=64
    """
    # Training
    batch_size = 32
    block_size = 64           # Context length (was 8 in bigram!)
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4      # Lower than bigram (deeper model needs smaller steps)
    eval_iters = 200

    # Model architecture
    n_embd = 64               # Embedding dimension
    n_head = 4                # Number of attention heads
    n_layer = 4               # Number of transformer blocks
    dropout = 0.2             # Dropout rate (prevents overfitting)

    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42


# =============================================================================
# DATA (Same as Day 1)
# =============================================================================

class CharTokenizer:
    """Character-level tokenizer (same as Day 1)."""
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        return ''.join([self.itos[i] for i in indices])


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()
    tokenizer = CharTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return tokenizer, data[:n], data[n:]


def get_batch(split, train_data, val_data, config):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix])
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix])
    return x.to(config.device), y.to(config.device)


# =============================================================================
# SELF-ATTENTION: The Core Innovation
# =============================================================================

class SingleHeadAttention(nn.Module):
    """
    ONE head of self-attention.

    This is the fundamental building block. Understanding this = understanding
    how every LLM in the world works.

    Step by step:
    1. Each token creates a Query ("what am I looking for?")
    2. Each token creates a Key ("what do I contain?")
    3. Each token creates a Value ("here's my actual information")
    4. Compute attention scores: Q @ K^T (who is relevant to whom?)
    5. Mask future tokens (can't cheat by looking ahead!)
    6. Softmax to get weights (probabilities that sum to 1)
    7. Weighted sum of Values (combine relevant information)
    """

    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()

        # Three linear projections — the ONLY learnable parameters in attention!
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # What I have
        self.query = nn.Linear(n_embd, head_size, bias=False)  # What I want
        self.value = nn.Linear(n_embd, head_size, bias=False)  # My content

        # Causal mask — lower triangular matrix
        # This prevents tokens from attending to future positions
        # register_buffer = not a parameter, but saved with model
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, T, C] — batch of token embeddings

        Returns: [B, T, head_size] — attention output
        """
        B, T, C = x.shape

        # Step 1-3: Create Q, K, V for each token
        k = self.key(x)     # [B, T, head_size]
        q = self.query(x)   # [B, T, head_size]
        v = self.value(x)   # [B, T, head_size]

        # Step 4: Compute attention scores
        # q @ k^T = [B, T, head_size] @ [B, head_size, T] = [B, T, T]
        # Each element [i, j] = "how relevant is token j to token i?"
        head_size = k.shape[-1]
        scores = q @ k.transpose(-2, -1) * head_size ** -0.5  # Scale by √d_k

        # Step 5: Mask future positions
        # Set future positions to -infinity so softmax gives them 0 weight
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        # Step 6: Softmax → attention weights (each row sums to 1)
        weights = F.softmax(scores, dim=-1)  # [B, T, T]
        weights = self.dropout(weights)

        # Step 7: Weighted sum of values
        out = weights @ v  # [B, T, T] @ [B, T, head_size] = [B, T, head_size]

        return out


# =============================================================================
# MULTI-HEAD ATTENTION
# =============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in PARALLEL.

    Why multiple heads?
    - Head 1 might learn: "look at nearby characters for spelling patterns"
    - Head 2 might learn: "look at the start of the current word"
    - Head 3 might learn: "look at punctuation for sentence boundaries"
    - Head 4 might learn: "look at character patterns for common words"

    Each head has a smaller dimension (n_embd / n_head),
    and their outputs are concatenated back to n_embd.

    4 heads × 16 dims each = 64 total (same as n_embd)
    """

    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([
            SingleHeadAttention(head_size, n_embd, block_size, dropout)
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)    # Output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Run all heads in parallel, concatenate results
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # [B, T, n_embd]
        out = self.dropout(self.proj(out))
        return out


# =============================================================================
# FEED-FORWARD NETWORK
# =============================================================================

class FeedForward(nn.Module):
    """
    Simple feed-forward network applied to each position independently.

    After attention gathers information from context,
    the feed-forward network PROCESSES that information.

    Think of it as:
    - Attention = "gather relevant information from other tokens"
    - Feed-Forward = "think about what that information means"

    Architecture: Linear → ReLU → Linear → Dropout
    The inner dimension is 4x the embedding dimension (standard in GPT).
    """

    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),      # Expand: 64 → 256
            nn.ReLU(),                            # Activation
            nn.Linear(4 * n_embd, n_embd),       # Contract: 256 → 64
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# TRANSFORMER BLOCK
# =============================================================================

class TransformerBlock(nn.Module):
    """
    One Transformer block = Attention + FeedForward + LayerNorm + Residuals

    The full flow:
        x → LayerNorm → MultiHeadAttention → + x (residual)
          → LayerNorm → FeedForward        → + x (residual)

    Residual connections (the "+ x" parts):
    - Allow gradients to flow directly back through the network
    - Without them, deep networks can't train (gradients vanish)
    - Think of it as: "start with what you have, then ADD new information"

    Layer Normalization:
    - Normalizes values to mean=0, std=1 at each layer
    - Prevents values from exploding or vanishing as they pass through layers
    - Applied BEFORE attention and feed-forward (Pre-LN, used by GPT-2+)

    Stack N of these blocks = a Transformer!
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head  # 64 / 4 = 16 per head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = FeedForward(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)   # Normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)   # Normalize before feed-forward

    def forward(self, x):
        # Attention with residual connection
        x = x + self.sa(self.ln1(x))      # "Add new attention info to what I already have"

        # Feed-forward with residual connection
        x = x + self.ffwd(self.ln2(x))    # "Add processed info to what I already have"

        return x


# =============================================================================
# THE COMPLETE MINI GPT MODEL
# =============================================================================

class MiniGPT(nn.Module):
    """
    A complete GPT-style language model!

    Architecture:
        Token Embedding + Position Embedding
        → N × Transformer Blocks
        → LayerNorm
        → Linear → Logits

    This is the SAME architecture as GPT-2, GPT-3, LLaMA, etc.
    The only differences are the sizes (n_embd, n_head, n_layer, etc.)

    Comparison:
    ┌──────────────┬──────────┬───────────┬──────────┐
    │              │ Our Model│ GPT-2 Sm  │ GPT-3    │
    ├──────────────┼──────────┼───────────┼──────────┤
    │ n_embd       │ 64       │ 768       │ 12288    │
    │ n_head       │ 4        │ 12        │ 96       │
    │ n_layer      │ 4        │ 12        │ 96       │
    │ block_size   │ 64       │ 1024      │ 2048     │
    │ Parameters   │ ~210K    │ 124M      │ 175B     │
    └──────────────┴──────────┴───────────┴──────────┘
    """

    def __init__(self, vocab_size, config):
        super().__init__()

        # Token embedding: each token → a vector of size n_embd
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)

        # Position embedding: each POSITION → a vector of size n_embd
        # This tells the model WHERE each token is in the sequence
        # Without this, attention can't distinguish "ABC" from "CBA"!
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # Stack of Transformer blocks
        self.blocks = nn.Sequential(*[
            TransformerBlock(config.n_embd, config.n_head, config.block_size, config.dropout)
            for _ in range(config.n_layer)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection: embedding → vocab logits
        self.lm_head = nn.Linear(config.n_embd, vocab_size)

        # Store config
        self.block_size = config.block_size
        self.device = config.device

    def forward(self, idx, targets=None):
        """
        idx: [B, T] token indices
        targets: [B, T] target indices (optional)
        """
        B, T = idx.shape

        # Token embeddings + position embeddings
        tok_emb = self.token_embedding(idx)                          # [B, T, n_embd]
        pos_emb = self.position_embedding(torch.arange(T, device=self.device))  # [T, n_embd]
        x = tok_emb + pos_emb                                        # [B, T, n_embd]

        # Pass through Transformer blocks
        x = self.blocks(x)                                            # [B, T, n_embd]

        # Final layer norm
        x = self.ln_f(x)                                              # [B, T, n_embd]

        # Project to vocabulary
        logits = self.lm_head(x)                                      # [B, T, vocab_size]

        # Compute loss if targets provided
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = targets.view(B*T)
            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate text autoregressively (same concept as Day 1!)

        Key difference from bigram: we CROP the context to block_size
        because position embeddings only go up to block_size.
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size (position embedding limit)
            idx_cond = idx[:, -self.block_size:]

            # Get predictions
            logits, _ = self(idx_cond)

            # Take last position, apply temperature
            logits = logits[:, -1, :] / temperature

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# =============================================================================
# TRAINING (Same loop as Day 1!)
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model, train_data, val_data, tokenizer, config):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    train_losses, val_losses, steps = [], [], []

    print("=" * 60)
    print("TRAINING MINI GPT")
    print("=" * 60)
    print(f"Device:       {config.device}")
    print(f"Parameters:   {sum(p.numel() for p in model.parameters()):,}")
    print(f"Iterations:   {config.max_iters}")
    print(f"Context size: {config.block_size} characters")
    print(f"Embedding:    {config.n_embd} dimensions")
    print(f"Heads:        {config.n_head}")
    print(f"Layers:       {config.n_layer}")
    print()
    print(f"{'Step':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Time':>8}")
    print("-" * 50)

    start_time = time.time()

    for iter in range(config.max_iters + 1):
        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - start_time
            print(f"{iter:6d} | {losses['train']:11.4f} | {losses['val']:11.4f} | {elapsed:7.1f}s")
            steps.append(iter)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())

        xb, yb = get_batch('train', train_data, val_data, config)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"   Final train loss: {train_losses[-1]:.4f}")
    print(f"   Final val loss:   {val_losses[-1]:.4f}")

    return steps, train_losses, val_losses


def plot_losses(steps, train_losses, val_losses, bigram_val_loss=None):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(steps, val_losses, 'r-', label='Val Loss', linewidth=2)

    if bigram_val_loss:
        plt.axhline(y=bigram_val_loss, color='gray', linestyle='--',
                     label=f'Bigram baseline ({bigram_val_loss:.2f})', linewidth=1)

    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Mini GPT Training — Day 2 (Self-Attention)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.annotate(f'Final: {val_losses[-1]:.3f}',
                 xy=(steps[-1], val_losses[-1]),
                 xytext=(-80, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig('training_loss_day2.png', dpi=150)
    print(f"📊 Loss plot saved to: training_loss_day2.png")


# =============================================================================
# COMPARISON: Bigram vs Mini GPT
# =============================================================================

def compare_models(model, tokenizer, config):
    print("\n" + "=" * 60)
    print("COMPARISON: Bigram (Day 1) vs Mini GPT (Day 2)")
    print("=" * 60)

    print("""
┌─────────────────┬──────────────┬──────────────┐
│                 │ Bigram (D1)  │ Mini GPT (D2)│
├─────────────────┼──────────────┼──────────────┤
│ Parameters      │ 4,225        │ ~210,000     │
│ Context window  │ 1 character  │ 64 characters│
│ Attention       │ None         │ Multi-head   │
│ Layers          │ 0            │ 4            │
│ Position info   │ No           │ Yes          │
└─────────────────┴──────────────┴──────────────┘
""")

    # Generate samples
    print("--- Mini GPT Generation (500 chars, temp=0.8) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=500, temperature=0.8)
    print(tokenizer.decode(generated[0].tolist()))

    print("\n--- Mini GPT Generation (200 chars, temp=0.5 — more focused) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=200, temperature=0.5)
    print(tokenizer.decode(generated[0].tolist()))

    print("\n--- Mini GPT Generation (200 chars, temp=1.2 — more creative) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=200, temperature=1.2)
    print(tokenizer.decode(generated[0].tolist()))


# =============================================================================
# ATTENTION VISUALIZATION
# =============================================================================

def visualize_attention(model, tokenizer, config):
    """Show what the attention heads are looking at."""
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)

    # Get a sample input
    test_text = "First Citizen"
    tokens = tokenizer.encode(test_text)
    x = torch.tensor([tokens], dtype=torch.long, device=config.device)

    # Extract attention weights from first block, first head
    model.eval()
    with torch.no_grad():
        # Get embeddings
        tok_emb = model.token_embedding(x)
        pos_emb = model.position_embedding(torch.arange(x.shape[1], device=config.device))
        emb = tok_emb + pos_emb

        # Get first block's attention
        block = model.blocks[0]
        ln_out = block.ln1(emb)

        # Get Q, K from first head
        head = block.sa.heads[0]
        q = head.query(ln_out)
        k = head.key(ln_out)

        T = x.shape[1]
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        tril = torch.tril(torch.ones(T, T, device=config.device))
        scores = scores.masked_fill(tril == 0, float('-inf'))
        weights = F.softmax(scores, dim=-1)

    # Display attention matrix
    print(f"\nInput: '{test_text}'")
    print(f"Tokens: {[tokenizer.itos[t] for t in tokens]}")
    print(f"\nAttention weights (Head 1, Block 1):")
    print("Each row shows: how much this character attends to each previous character\n")

    chars = [tokenizer.itos[t] for t in tokens]
    header = "      " + "".join(f"  {c:>4s}" for c in chars)
    print(header)
    print("      " + "-" * (6 * len(chars)))

    for i in range(len(chars)):
        row = f"  {chars[i]:>2s} | "
        for j in range(len(chars)):
            w = weights[0, i, j].item()
            if j <= i:
                if w > 0.3:
                    row += f" {w:.2f}*"   # Star = high attention
                else:
                    row += f" {w:.2f} "
            else:
                row += "   -  "    # Masked (future)
        print(row)

    print("\n* = high attention (>0.3)")
    print("- = masked (can't see future tokens)")

    model.train()


# =============================================================================
# PARAMETER BREAKDOWN
# =============================================================================

def parameter_breakdown(model, config):
    print("\n" + "=" * 60)
    print("PARAMETER BREAKDOWN — Where Are the 210K Parameters?")
    print("=" * 60)

    total = 0
    print(f"\n{'Component':<40s} {'Shape':<25s} {'Params':>10s}")
    print("-" * 75)

    for name, param in model.named_parameters():
        num = param.numel()
        total += num

        # Make names readable
        clean_name = name.replace('.', ' → ')
        print(f"{clean_name:<40s} {str(list(param.shape)):<25s} {num:>10,d}")

    print("-" * 75)
    print(f"{'TOTAL':<40s} {'':25s} {total:>10,d}")

    print(f"""
Key observations:
- Token embedding: {config.n_embd} × vocab_size = lots of parameters
- Position embedding: {config.block_size} × {config.n_embd} positions
- Each attention head has Q, K, V projections
- Feed-forward inner dimension is 4 × {config.n_embd} = {4 * config.n_embd}
- Most parameters are in the feed-forward networks!
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Day 2: Mini GPT with Self-Attention')
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

    print("=" * 60)
    print("DAY 2: MINI GPT WITH SELF-ATTENTION")
    print("Healthcare LLM Project")
    print("=" * 60)
    print()

    # Check data file
    if not os.path.exists(args.data):
        print(f"❌ Data file '{args.data}' not found!")
        print(f"Make sure 'input.txt' is in this folder.")
        print(f"It should be there from Day 1.")
        return

    # Load data
    print("📂 Loading data...")
    tokenizer, train_data, val_data = load_data(args.data)
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Train tokens: {len(train_data):,}")
    print(f"   Val tokens: {len(val_data):,}")
    print()

    # Create model
    model = MiniGPT(tokenizer.vocab_size, config).to(config.device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: MiniGPT")
    print(f"   Parameters: {num_params:,}")
    print(f"   That's {num_params / 4225:.0f}x bigger than yesterday's bigram model!")
    print()

    # Show untrained output
    print("--- BEFORE TRAINING (random weights) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=100)
    print(tokenizer.decode(generated[0].tolist()))
    print()

    # Train!
    steps, train_losses, val_losses = train(
        model, train_data, val_data, tokenizer, config
    )

    # Plot with bigram baseline
    plot_losses(steps, train_losses, val_losses, bigram_val_loss=2.58)

    # Compare bigram vs mini GPT
    compare_models(model, tokenizer, config)

    # Show attention patterns
    visualize_attention(model, tokenizer, config)

    # Parameter breakdown
    parameter_breakdown(model, config)

    # Save model
    save_path = 'mini_gpt_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_chars': tokenizer.chars,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
            'dropout': config.dropout,
        }
    }, save_path)
    print(f"\n💾 Model saved to: {save_path}")

    print("\n" + "=" * 60)
    print("🎉 DAY 2 COMPLETE!")
    print("=" * 60)
    print(f"""
What you built today:
  ✅ Self-attention mechanism (Query, Key, Value)
  ✅ Multi-head attention (4 heads, each finding different patterns)
  ✅ Transformer blocks (attention + feed-forward + residuals)
  ✅ Position embeddings (so the model knows token order)
  ✅ A complete Mini GPT with ~{num_params:,} parameters!

Compare with Day 1:
  📊 Bigram val loss:    2.58
  📊 Mini GPT val loss:  {val_losses[-1]:.2f}
  📊 Improvement:        {((2.58 - val_losses[-1]) / 2.58 * 100):.1f}%

The generated text should be MUCH more readable now because
the model can see {config.block_size} characters of context instead of just 1!

Day 3 preview: We'll explore training on different datasets,
scaling up the model, and advanced techniques like learning
rate scheduling and gradient accumulation.

Push to GitHub:
  git add .
  git commit -m "Day 2: Mini GPT with self-attention"
  git push
""")


if __name__ == '__main__':
    main()
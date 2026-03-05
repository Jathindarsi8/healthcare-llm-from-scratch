"""
=============================================================================
BIGRAM LANGUAGE MODEL FROM SCRATCH
=============================================================================
Author: Jathin | Healthcare LLM Project - Phase 1
Description: Complete character-level bigram language model in PyTorch.
             This is the foundation for building GPT from scratch.

How to run:
    1. Download dataset:
       curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

    2. Run this script:
       python 01_bigram_model.py

    3. (Optional) Try with medical text:
       python 01_bigram_model.py --data medical_abstracts.txt

Reference: Based on Andrej Karpathy's "Let's build GPT" lecture
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    """All hyperparameters in one place."""
    batch_size = 32          # Number of sequences processed in parallel
    block_size = 8           # Maximum context length (chars to look back)
    max_iters = 5000         # Total training iterations
    eval_interval = 500      # How often to print loss
    learning_rate = 1e-3     # Adam optimizer learning rate
    eval_iters = 200         # Batches to average for loss estimation
    seed = 42                # For reproducibility

    # Automatically select device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =============================================================================
# DATA LOADING & TOKENIZATION
# =============================================================================

class CharTokenizer:
    """
    Character-level tokenizer.

    This is the simplest tokenizer possible:
    - Each unique character gets an integer ID
    - Vocabulary size = number of unique characters

    In a real LLM, you'd use BPE (Byte Pair Encoding) with 32K-100K tokens.
    But character-level teaches the same concepts.

    Healthcare Note:
    ----------------
    For medical text, character-level tokenization actually handles things like
    "ICD-10: E11.65" naturally since every character is a token. BPE tokenizers
    would split this differently and might lose the structure of medical codes.
    """

    def __init__(self, text):
        # Get all unique characters and sort them
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Create bidirectional mappings
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # char → int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int → char

    def encode(self, text):
        """Convert string to list of integers."""
        return [self.stoi[c] for c in text]

    def decode(self, indices):
        """Convert list of integers back to string."""
        return ''.join([self.itos[i] for i in indices])

    def __repr__(self):
        return f"CharTokenizer(vocab_size={self.vocab_size}, chars={''.join(self.chars[:20])}...)"


def load_data(filepath, config):
    """
    Load text file and prepare train/val splits.

    Returns:
        tokenizer: CharTokenizer instance
        train_data: torch.Tensor of training token IDs
        val_data: torch.Tensor of validation token IDs
    """
    print(f"📂 Loading data from: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"   Dataset size: {len(text):,} characters")
    print(f"   First 100 chars: {text[:100]!r}")

    # Build tokenizer
    tokenizer = CharTokenizer(text)
    print(f"   {tokenizer}")

    # Encode entire text into a tensor
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    # 90/10 train/val split
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    print(f"   Train tokens: {len(train_data):,}")
    print(f"   Val tokens:   {len(val_data):,}")
    print()

    return tokenizer, train_data, val_data


def get_batch(split, train_data, val_data, config):
    """
    Generate a random batch of input-target pairs.

    For language modeling:
        Input:  [H, e, l, l, o,  , W, o]
        Target: [e, l, l, o,  , W, o, r]   ← shifted right by 1

    The model learns: given these characters, predict the next one.

    Returns:
        x: [batch_size, block_size] input tensor
        y: [batch_size, block_size] target tensor
    """
    data = train_data if split == 'train' else val_data

    # Pick random starting positions
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # Create input and target tensors
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + config.block_size + 1] for i in ix])

    # Move to device (CPU or GPU)
    x, y = x.to(config.device), y.to(config.device)

    return x, y


# =============================================================================
# THE BIGRAM LANGUAGE MODEL
# =============================================================================

class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    Architecture:
        Input token → Embedding lookup → Logits (next token scores)

    How it works:
        - An embedding table of shape [vocab_size, vocab_size]
        - Row i contains the log-probabilities: "given char i, predict next char"
        - This is equivalent to counting bigram frequencies in the training data!

    Parameters:
        - Only the embedding table: vocab_size × vocab_size
        - For vocab_size=65: just 4,225 parameters

    Comparison to GPT:
        - GPT adds Transformer blocks (attention + FFN) between embedding and output
        - GPT-2 Small: 124 million parameters
        - GPT-4: ~1.8 trillion parameters (estimated)
        - Same training loop, same loss function, same generation method!
    """

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx: [B, T] tensor of token indices (B=batch, T=time/sequence)
            targets: [B, T] tensor of target indices (optional)

        Returns:
            logits: [B, T, C] prediction scores (C=vocab_size)
            loss: scalar cross-entropy loss (None if no targets)
        """
        # Each token looks up its row in the embedding table
        # This row contains scores for what the next token should be
        logits = self.token_embedding_table(idx)  # [B, T, C]

        if targets is None:
            loss = None
        else:
            # Reshape for PyTorch's cross_entropy function
            # cross_entropy expects: input=[N, C], target=[N]
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)     # [B*T, C]
            targets_reshaped = targets.view(B * T)       # [B*T]

            # Cross-entropy loss
            # This measures how well the predicted distribution matches the true next token
            # Lower = better predictions
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressive text generation.

        Process:
            1. Feed current sequence through model
            2. Take logits for the LAST position only
            3. Convert to probabilities with softmax
            4. Sample a token from the distribution
            5. Append to sequence
            6. Repeat

        This is EXACTLY how ChatGPT generates text — one token at a time!

        Args:
            idx: [B, T] starting context (can be just one token)
            max_new_tokens: how many tokens to generate

        Returns:
            idx: [B, T + max_new_tokens] the extended sequence
        """
        for _ in range(max_new_tokens):
            # Get predictions for current sequence
            logits, _ = self(idx)                            # [B, T, C]

            # We only care about the last position's prediction
            logits = logits[:, -1, :]                        # [B, C]

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)                # [B, C]

            # Sample from the probability distribution
            # (This is what gives generation its randomness/creativity)
            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            # Append the new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)          # [B, T+1]

        return idx


# =============================================================================
# TRAINING
# =============================================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    """
    Estimate average loss over multiple batches.

    Why not just use the current batch's loss?
    - Individual batch loss is noisy (depends on which random samples were chosen)
    - Averaging over many batches gives a more stable estimate
    - Like evaluating a model on a test set vs. a single example

    The @torch.no_grad() decorator tells PyTorch not to track gradients,
    saving memory and computation during evaluation.
    """
    out = {}
    model.eval()  # Switch to evaluation mode (disables dropout, etc.)

    for split in ['train', 'val']:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(split, train_data, val_data, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()  # Switch back to training mode
    return out


def train(model, train_data, val_data, tokenizer, config):
    """
    Full training loop with logging and visualization.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Track losses for plotting
    train_losses = []
    val_losses = []
    steps = []

    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Iterations: {config.max_iters}")
    print(f"Batch size: {config.batch_size}")
    print(f"Block size: {config.block_size}")
    print(f"Learning rate: {config.learning_rate}")
    print()
    print(f"{'Step':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Time':>8}")
    print("-" * 50)

    start_time = time.time()

    for iter in range(config.max_iters + 1):

        # Evaluate periodically
        if iter % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - start_time
            print(f"{iter:6d} | {losses['train']:11.4f} | {losses['val']:11.4f} | {elapsed:7.1f}s")

            # Save for plotting
            steps.append(iter)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())

        # Get a training batch
        xb, yb = get_batch('train', train_data, val_data, config)

        # Forward pass
        logits, loss = model(xb, yb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    total_time = time.time() - start_time
    print(f"\n✅ Training complete in {total_time:.1f}s")
    print(f"   Final train loss: {train_losses[-1]:.4f}")
    print(f"   Final val loss:   {val_losses[-1]:.4f}")

    # Theoretical minimum loss for reference
    # Random guessing: -log(1/vocab_size)
    random_loss = -torch.log(torch.tensor(1.0 / tokenizer.vocab_size)).item()
    print(f"   Random baseline:  {random_loss:.4f} (loss if guessing randomly)")
    print(f"   Improvement:      {((random_loss - val_losses[-1]) / random_loss * 100):.1f}% better than random")

    return steps, train_losses, val_losses


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_losses(steps, train_losses, val_losses, save_path='training_loss.png'):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(steps, val_losses, 'r-', label='Val Loss', linewidth=2)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel('Cross-Entropy Loss', fontsize=12)
    plt.title('Bigram Language Model — Training Progress', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)

    # Add annotations
    plt.annotate(f'Final: {val_losses[-1]:.3f}',
                 xy=(steps[-1], val_losses[-1]),
                 xytext=(-80, 20), textcoords='offset points',
                 arrowprops=dict(arrowstyle='->', color='red'),
                 fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"📊 Loss plot saved to: {save_path}")


# =============================================================================
# TEXT GENERATION DEMO
# =============================================================================

def generation_demo(model, tokenizer, config):
    """
    Demonstrate text generation with the trained model.
    Shows multiple samples with different starting characters.
    """
    print("\n" + "=" * 60)
    print("TEXT GENERATION DEMO")
    print("=" * 60)

    # Generate from empty context (newline character)
    print("\n--- Sample 1: Open-ended generation (500 chars) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    generated = model.generate(context, max_new_tokens=500)
    print(tokenizer.decode(generated[0].tolist()))

    # Generate from specific starting characters
    starters = ['T', 'H', 'W', 'I']
    for char in starters:
        if char in tokenizer.stoi:
            print(f"\n--- Sample: Starting with '{char}' (100 chars) ---")
            context = torch.tensor([[tokenizer.stoi[char]]],
                                   dtype=torch.long, device=config.device)
            generated = model.generate(context, max_new_tokens=100)
            print(tokenizer.decode(generated[0].tolist()))

    # Temperature-based generation
    print("\n" + "=" * 60)
    print("TEMPERATURE EXPERIMENT")
    print("=" * 60)
    print("""
Temperature controls randomness in generation:
  Low (0.5)  → More predictable, repetitive
  Normal (1) → Balanced
  High (2.0) → More random, creative, chaotic
    """)

    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)

    for temp in [0.5, 1.0, 2.0]:
        print(f"\n--- Temperature = {temp} ---")
        generated = generate_with_temperature(model, context.clone(), 200, temp)
        print(tokenizer.decode(generated[0].tolist()))


def generate_with_temperature(model, idx, max_new_tokens, temperature=1.0):
    """
    Generate text with temperature control.

    Temperature scales the logits before softmax:
    - T < 1: Makes the distribution sharper (more confident, less random)
    - T = 1: Standard sampling
    - T > 1: Makes the distribution flatter (more random, more creative)

    This is the SAME temperature parameter used in ChatGPT's API!
    """
    for _ in range(max_new_tokens):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature  # Scale by temperature
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# =============================================================================
# BONUS: Analyze what the model learned
# =============================================================================

def analyze_model(model, tokenizer):
    """
    Peek inside the model to understand what it learned.
    Extract the most likely next character for each character.
    """
    print("\n" + "=" * 60)
    print("MODEL ANALYSIS: What the Bigram Model Learned")
    print("=" * 60)

    # Get the embedding table (our only parameter!)
    weights = model.token_embedding_table.weight.data  # [vocab_size, vocab_size]

    # For each character, find the top-3 most likely next characters
    print(f"\n{'Given':>8} | {'Top 1':>8} | {'Top 2':>8} | {'Top 3':>8}")
    print("-" * 50)

    # Show interesting characters
    interesting = list("aeioustndhlr .\n")

    for char in interesting:
        if char not in tokenizer.stoi:
            continue
        idx = tokenizer.stoi[char]
        probs = F.softmax(weights[idx], dim=0)
        top3 = torch.topk(probs, 3)

        display_char = repr(char) if char in '\n ' else f"'{char}'"
        top_chars = [f"'{tokenizer.itos[i.item()]}' ({v.item():.2f})"
                 for i, v in zip(top3.indices, top3.values)]

        print(f"{display_char:>8} | {top_chars[0]:>12} | {top_chars[1]:>12} | {top_chars[2]:>12}")

    print("""
Observations:
- After 'q', the model likely predicts 'u' (English pattern!)
- After ' ' (space), it predicts common word-starting letters
- After '.', it might predict space or newline
- After vowels, it predicts common consonants

These are bigram frequency statistics, learned by gradient descent!
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Bigram Language Model from Scratch')
    parser.add_argument('--data', type=str, default='input.txt',
                        help='Path to training text file')
    parser.add_argument('--iters', type=int, default=None,
                        help='Override max training iterations')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override learning rate')
    args = parser.parse_args()

    # Setup
    config = Config()
    if args.iters:
        config.max_iters = args.iters
    if args.lr:
        config.learning_rate = args.lr

    torch.manual_seed(config.seed)

    print("=" * 60)
    print("BIGRAM LANGUAGE MODEL FROM SCRATCH")
    print("Phase 1: Healthcare LLM Project")
    print("=" * 60)
    print()

    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Data file '{args.data}' not found!")
        print()
        print("Download Shakespeare dataset:")
        print("  curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
        print()
        print("Or provide your own text file:")
        print("  python 01_bigram_model.py --data your_file.txt")
        return

    # Load data
    tokenizer, train_data, val_data = load_data(args.data, config)

    # Create model
    model = BigramLanguageModel(tokenizer.vocab_size).to(config.device)
    print(f"🧠 Model created: BigramLanguageModel")
    print(f"   Vocab size: {tokenizer.vocab_size}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Show untrained generation
    print("--- BEFORE TRAINING (random weights) ---")
    context = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    untrained_output = model.generate(context, max_new_tokens=100)
    print(tokenizer.decode(untrained_output[0].tolist()))
    print()

    # Train
    steps, train_losses, val_losses = train(model, train_data, val_data, tokenizer, config)

    # Plot losses
    plot_losses(steps, train_losses, val_losses)

    # Generate text
    generation_demo(model, tokenizer, config)

    # Analyze
    analyze_model(model, tokenizer)

    # Save model
    save_path = 'bigram_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_chars': tokenizer.chars,
        'config': {
            'vocab_size': tokenizer.vocab_size,
            'batch_size': config.batch_size,
            'block_size': config.block_size,
        }
    }, save_path)
    print(f"\n💾 Model saved to: {save_path}")

    print("\n" + "=" * 60)
    print("🎉 PHASE 1 COMPLETE!")
    print("=" * 60)
    print("""
What you've built:
  ✅ Character-level tokenizer
  ✅ Bigram language model (simplest LLM possible)
  ✅ Full training loop with evaluation
  ✅ Text generation with temperature control
  ✅ Model analysis and visualization

What to try next:
  1. Train on medical text (PubMed abstracts)
  2. Compare Shakespeare vs Medical generated text
  3. Move to Phase 2: Add self-attention → build a real Transformer!

Key insight:
  Your bigram model and GPT-4 use the SAME training loop and loss function.
  The only difference is the model architecture between input and output.
  Phase 2 fills that gap with Transformer blocks (attention + feed-forward).
""")


if __name__ == '__main__':
    main()
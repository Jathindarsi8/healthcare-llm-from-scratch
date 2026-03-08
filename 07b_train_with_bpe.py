"""
=============================================================================
DAY 7B: TRAIN GPT WITH BPE TOKENIZATION
=============================================================================
Author: Jathin | Healthcare LLM Project

Trains the GPT model using BPE tokens instead of characters.
This is how REAL LLMs work — the final piece of the puzzle.

How to run:
    python 07a_train_medical_tokenizer.py   (first!)
    python 07b_train_with_bpe.py

This is a MAJOR upgrade:
  Character model: 128 token window ≈ 20 words
  BPE model:       128 token window ≈ 80-100 words
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import os
import json
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Try HuggingFace tokenizers
try:
    from tokenizers import Tokenizer as HFTokenizer
    HAS_HF = True
except ImportError:
    HAS_HF = False


# =============================================================================
# LOAD TOKENIZER
# =============================================================================

class BPEWrapper:
    """Unified wrapper for both HF and custom tokenizers."""

    def __init__(self, tokenizer_dir='medical_tokenizer'):
        config_path = os.path.join(tokenizer_dir, 'config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.is_hf = self.config.get('is_hf', False)
        self.vocab_size = self.config['vocab_size']

        if self.is_hf and HAS_HF:
            tok_path = os.path.join(tokenizer_dir, 'tokenizer.json')
            self.tokenizer = HFTokenizer.from_file(tok_path)
            # Get actual vocab size from the tokenizer
            self.vocab_size = self.tokenizer.get_vocab_size()
        else:
            # Load custom tokenizer
            tok_path = os.path.join(tokenizer_dir, 'tokenizer_custom.json')
            with open(tok_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.vocab = {int(k): v for k, v in data['vocab'].items()}
            self.inverse_vocab = data['inverse_vocab']
            self.merges = {}
            for k, v in data['merges'].items():
                parts = k.split('|||')
                self.merges[(parts[0], parts[1])] = v
            self.vocab_size = data['vocab_size']
            self.is_hf = False

    def encode(self, text):
        if self.is_hf:
            encoded = self.tokenizer.encode(text)
            return encoded.ids
        else:
            words = text.split()
            all_ids = []
            for word in words:
                tokens = list(word)
                for pair, new_token in self.merges.items():
                    i = 0
                    while i < len(tokens) - 1:
                        if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                            tokens = tokens[:i] + [new_token] + tokens[i + 2:]
                        else:
                            i += 1
                for token in tokens:
                    if token in self.inverse_vocab:
                        all_ids.append(self.inverse_vocab[token])
            return all_ids

    def decode(self, ids):
        if self.is_hf:
            return self.tokenizer.decode(ids)
        else:
            tokens = [self.vocab.get(id, '') for id in ids]
            return ' '.join(tokens)


# =============================================================================
# MODEL (Same architecture, different vocab size)
# =============================================================================

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(
            torch.ones(block_size, block_size)
        ).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        hs = C // self.n_head
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (hs ** -0.5)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = self.attn_dropout(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class BPEGPT(nn.Module):
    """
    GPT model with BPE tokenization.

    KEY DIFFERENCE from character-level:
    - vocab_size is now 4096 (BPE tokens) instead of 65 (characters)
    - Each token represents a WORD or SUBWORD, not a single character
    - Same 128-token window now sees ~80-100 words instead of ~20
    """

    def __init__(self, vocab_size, n_embd=128, n_head=8, n_layer=6, block_size=128, dropout=0.15):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
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
# CONFIGURATION
# =============================================================================

class Config:
    batch_size = 16
    block_size = 128
    grad_accum_steps = 4
    max_iters = 3000
    eval_interval = 100
    eval_iters = 20
    max_lr = 3e-4
    min_lr = 3e-5
    warmup_iters = 150
    n_embd = 128
    n_head = 8
    n_layer = 6
    dropout = 0.15
    weight_decay = 0.1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 42


def get_lr(it, config):
    if it < config.warmup_iters:
        return config.max_lr * (it + 1) / config.warmup_iters
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# =============================================================================
# TRAINING
# =============================================================================

def get_batch(data, config):
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i:i+config.block_size] for i in ix]).to(config.device)
    y = torch.stack([data[i+1:i+config.block_size+1] for i in ix]).to(config.device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, config):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(config.eval_iters)
        for k in range(config.eval_iters):
            X, Y = get_batch(data, config)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(model, train_data, val_data, config):
    decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': config.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0},
    ], lr=config.max_lr, betas=(0.9, 0.95))

    steps_list, train_losses, val_losses = [], [], []
    start_time = time.time()

    print(f"  {'Step':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'LR':>10} | {'Time':>8}")
    print("  " + "-" * 60)

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

        lr = get_lr(iter_num, config)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.grad_accum_steps):
            xb, yb = get_batch(train_data, config)
            _, loss = model(xb, yb)
            (loss / config.grad_accum_steps).backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    total = time.time() - start_time
    print(f"\n  Done! {total:.1f}s ({total/60:.1f} min)")
    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")
    return steps_list, train_losses, val_losses


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
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
    print("  DAY 7B: TRAIN GPT WITH BPE TOKENIZATION")
    print("  This is how REAL LLMs work!")
    print("  Estimated time: ~8-10 minutes on CPU")
    print("=" * 65)

    # Load tokenizer
    tokenizer_dir = 'medical_tokenizer'
    if not os.path.exists(tokenizer_dir):
        print(f"\n  Tokenizer not found! Run first: python 07a_train_medical_tokenizer.py")
        return

    print(f"\n  Loading BPE tokenizer...")
    bpe = BPEWrapper(tokenizer_dir)
    print(f"  Vocab size: {bpe.vocab_size}")

    # Load and tokenize data
    data_file = bpe.config.get('data_file', 'prepared_medical_data.txt')
    if not os.path.exists(data_file):
        for f in ['prepared_medical_data.txt', 'pubmed_medical_data.txt', 'medical_text.txt']:
            if os.path.exists(f):
                data_file = f
                break

    print(f"  Loading data from: {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  Encoding with BPE tokenizer...")
    token_ids = bpe.encode(text)
    print(f"  Text characters: {len(text):,}")
    print(f"  BPE tokens: {len(token_ids):,}")
    print(f"  Compression: {len(text) / len(token_ids):.1f}x")

    # Validate token IDs
    max_id = max(token_ids) if token_ids else 0
    actual_vocab = max_id + 1
    if actual_vocab > bpe.vocab_size:
        print(f"  Adjusting vocab size: {bpe.vocab_size} -> {actual_vocab}")
        bpe.vocab_size = actual_vocab

    data = torch.tensor(token_ids, dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    print(f"  Train tokens: {len(train_data):,}")
    print(f"  Val tokens: {len(val_data):,}")

    # Create model
    print(f"\n  Creating BPE GPT model...")
    model = BPEGPT(
        vocab_size=bpe.vocab_size,
        n_embd=config.n_embd,
        n_head=config.n_head,
        n_layer=config.n_layer,
        block_size=config.block_size,
        dropout=config.dropout,
    ).to(config.device)

    params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {params:,}")
    print(f"  Vocab: {bpe.vocab_size} BPE tokens (vs 65 characters before)")

    # Train
    print(f"\n" + "=" * 65)
    print(f"  TRAINING BPE GPT")
    print(f"=" * 65)

    steps, train_losses, val_losses = train(model, train_data, val_data, config)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(steps, train_losses, 'b-', label='Train', linewidth=2)
    plt.plot(steps, val_losses, 'r-', label='Val', linewidth=2)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('BPE Medical GPT Training - Day 7')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_day7.png', dpi=150)
    print(f"\n  Plot saved: training_loss_day7.png")

    # Generate
    print(f"\n" + "=" * 65)
    print(f"  GENERATION WITH BPE MODEL")
    print(f"=" * 65)

    model.eval()
    with torch.no_grad():
        print(f"\n  --- Open generation (100 tokens) ---")
        ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        out = model.generate(ctx, 100, temperature=0.8, top_k=40)
        text_out = bpe.decode(out[0].tolist())
        print(f"  {text_out}")

        # Generate from "Patient"
        print(f"\n  --- Starting with 'Patient' (100 tokens) ---")
        patient_ids = bpe.encode("Patient")
        if patient_ids:
            ctx = torch.tensor([patient_ids], dtype=torch.long, device=config.device)
            out = model.generate(ctx, 100, temperature=0.8, top_k=40)
            text_out = bpe.decode(out[0].tolist())
            print(f"  {text_out}")

        print(f"\n  --- Starting with 'Diagnosis' (100 tokens) ---")
        diag_ids = bpe.encode("Diagnosis")
        if diag_ids:
            ctx = torch.tensor([diag_ids], dtype=torch.long, device=config.device)
            out = model.generate(ctx, 100, temperature=0.8, top_k=40)
            text_out = bpe.decode(out[0].tolist())
            print(f"  {text_out}")

    # Save
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer_dir': tokenizer_dir,
        'config': {
            'vocab_size': bpe.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        },
    }, 'bpe_medical_gpt.pt')
    print(f"\n  Model saved: bpe_medical_gpt.pt")

    print(f"""
{'='*65}
  DAY 7 COMPLETE!
{'='*65}

  What you built today:
    - Trained a medical BPE tokenizer ({bpe.vocab_size} tokens)
    - Trained GPT with BPE tokens instead of characters
    - Model parameters: {params:,}
    - Compression ratio: {len(text) / len(token_ids):.1f}x more efficient

  KEY ACHIEVEMENT:
    Your model now reads text the same way GPT-4 does.
    No more character-by-character processing.
    Each token is a meaningful word or subword.

  The complete progression:
    Day 1: Characters, no context    → garbage
    Day 2: Characters, 64 context    → broken English
    Day 3: Characters, 128 context   → readable text
    Day 4: Characters, medical data  → medical patterns
    Day 7: BPE tokens, medical data  → real medical language!

  Push to GitHub:
    git add .
    git commit -m "Day 7: Medical BPE tokenizer + BPE GPT training"
    git push
""")


if __name__ == '__main__':
    main()
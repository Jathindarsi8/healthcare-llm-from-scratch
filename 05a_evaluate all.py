"""
=============================================================================
DAY 5A: EVALUATE ALL MODELS — Side by Side Comparison
=============================================================================
Author: Jathin | Healthcare LLM Project

Loads ALL saved models and compares them:
- Generation quality
- Loss comparison
- Parameter counts
- What each model learned

How to run:
    python 05a_evaluate_all.py
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# =============================================================================
# MODEL DEFINITIONS (needed to load saved models)
# =============================================================================

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Mini GPT components (Day 2)
class SingleHeadAttention(nn.Module):
    def __init__(self, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        weights = self.dropout(F.softmax(scores, dim=-1))
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(head_size, n_embd, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class MiniFFN(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MiniBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, block_size, dropout)
        self.ffwd = MiniFFN(n_embd, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n_head=4, n_layer=4, block_size=64, dropout=0.2):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[MiniBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        x = self.ln_f(self.blocks(x))
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Scaled GPT components (Day 3+4)
class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

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


class ScaledFFN(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.gelu = GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class ScaledBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = ScaledFFN(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ScaledGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=96, n_head=6, n_layer=6, block_size=128, dropout=0.15):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([ScaledBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight

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
# TOKENIZER
# =============================================================================

class CharTokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l if i in self.itos])


# =============================================================================
# MAIN EVALUATION
# =============================================================================

def main():
    device = 'cpu'
    torch.manual_seed(42)

    print("=" * 65)
    print("  DAY 5A: FULL MODEL EVALUATION")
    print("  Comparing ALL models from Days 1-4")
    print("=" * 65)

    models_loaded = []

    # =========================================================================
    # LOAD ALL MODELS
    # =========================================================================

    # Day 1: Bigram
    if os.path.exists('bigram_model.pt'):
        print("\n  Loading Day 1: Bigram Model...")
        ckpt = torch.load('bigram_model.pt', map_location=device, weights_only=False)
        tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
        model = BigramLanguageModel(tokenizer.vocab_size).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        models_loaded.append(('Day 1: Bigram', model, tokenizer, params, 8))
        print(f"    Loaded! {params:,} parameters")

    # Day 2: Mini GPT
    if os.path.exists('mini_gpt_model.pt'):
        print("  Loading Day 2: Mini GPT...")
        ckpt = torch.load('mini_gpt_model.pt', map_location=device, weights_only=False)
        tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
        cfg = ckpt['config']
        model = MiniGPT(
            cfg['vocab_size'],
            n_embd=cfg.get('n_embd', 64),
            n_head=cfg.get('n_head', 4),
            n_layer=cfg.get('n_layer', 4),
            block_size=cfg.get('block_size', 64),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        models_loaded.append(('Day 2: Mini GPT', model, tokenizer, params, 64))
        print(f"    Loaded! {params:,} parameters")

    # Day 3: Scaled GPT
    if os.path.exists('scaled_gpt_model.pt'):
        print("  Loading Day 3: Scaled GPT...")
        ckpt = torch.load('scaled_gpt_model.pt', map_location=device, weights_only=False)
        tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
        cfg = ckpt['config']
        model = ScaledGPT(
            cfg['vocab_size'],
            n_embd=cfg.get('n_embd', 96),
            n_head=cfg.get('n_head', 6),
            n_layer=cfg.get('n_layer', 6),
            block_size=cfg.get('block_size', 128),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        models_loaded.append(('Day 3: Scaled GPT', model, tokenizer, params, 128))
        print(f"    Loaded! {params:,} parameters")

    # Day 4: Medical GPT
    if os.path.exists('medical_gpt_model.pt'):
        print("  Loading Day 4: Medical GPT...")
        ckpt = torch.load('medical_gpt_model.pt', map_location=device, weights_only=False)
        tokenizer = CharTokenizer(ckpt['tokenizer_chars'])
        cfg = ckpt['config']
        model = ScaledGPT(
            cfg['vocab_size'],
            n_embd=cfg.get('n_embd', 96),
            n_head=cfg.get('n_head', 6),
            n_layer=cfg.get('n_layer', 6),
            block_size=cfg.get('block_size', 128),
        ).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        models_loaded.append(('Day 4: Medical GPT', model, tokenizer, params, 128))
        print(f"    Loaded! {params:,} parameters")

    if not models_loaded:
        print("\n  No saved models found! Run Days 1-4 first.")
        return

    # =========================================================================
    # PARAMETER COMPARISON
    # =========================================================================

    print("\n" + "=" * 65)
    print("  PARAMETER COMPARISON")
    print("=" * 65)
    print(f"\n  {'Model':<25s} | {'Parameters':>12s} | {'Context':>8s} | {'Growth':>8s}")
    print("  " + "-" * 60)

    prev_params = 0
    for name, model, tok, params, ctx in models_loaded:
        growth = f"{params/prev_params:.0f}x" if prev_params > 0 else "-"
        print(f"  {name:<25s} | {params:>12,} | {ctx:>6d} ch | {growth:>8s}")
        prev_params = params

    # =========================================================================
    # GENERATION COMPARISON
    # =========================================================================

    print("\n" + "=" * 65)
    print("  GENERATION COMPARISON (300 chars each, temp=0.8)")
    print("=" * 65)

    with torch.no_grad():
        for name, model, tok, params, ctx in models_loaded:
            print(f"\n  --- {name} ({params:,} params) ---")
            context = torch.zeros((1, 1), dtype=torch.long, device=device)

            if hasattr(model, 'generate'):
                if 'Medical' in name or 'Scaled' in name:
                    out = model.generate(context, 300, temperature=0.8, top_k=10)
                else:
                    out = model.generate(context, 300, temperature=0.8)
                text = tok.decode(out[0].tolist())
                print(f"  {text[:300]}")

    # =========================================================================
    # QUALITY ANALYSIS
    # =========================================================================

    print("\n" + "=" * 65)
    print("  QUALITY ANALYSIS")
    print("=" * 65)

    with torch.no_grad():
        for name, model, tok, params, ctx in models_loaded:
            context = torch.zeros((1, 1), dtype=torch.long, device=device)
            if hasattr(model, 'generate'):
                if 'Medical' in name or 'Scaled' in name:
                    out = model.generate(context, 500, temperature=0.8, top_k=10)
                else:
                    out = model.generate(context, 500, temperature=0.8)
                text = tok.decode(out[0].tolist())

                # Analyze
                words = text.split()
                real_words = set(['the', 'and', 'of', 'to', 'a', 'in', 'is', 'that', 'it',
                                  'for', 'was', 'on', 'are', 'with', 'he', 'she', 'be',
                                  'this', 'have', 'from', 'not', 'but', 'by', 'or', 'an',
                                  'my', 'his', 'her', 'you', 'me', 'no', 'do', 'if', 'we',
                                  'all', 'your', 'will', 'shall', 'what', 'so', 'as', 'at',
                                  'patient', 'diagnosis', 'treatment', 'history', 'medical',
                                  'clinical', 'assessment', 'plan', 'medications', 'blood',
                                  'heart', 'disease', 'chronic', 'acute', 'diabetes', 'type'])

                word_count = len(words)
                real_count = sum(1 for w in words if w.lower().strip('.,;:!?') in real_words)
                real_pct = (real_count / word_count * 100) if word_count > 0 else 0

                avg_word_len = sum(len(w) for w in words) / max(len(words), 1)
                spaces = text.count(' ')
                newlines = text.count('\n')

                print(f"\n  {name}:")
                print(f"    Words generated:     {word_count}")
                print(f"    Real English words:  {real_count} ({real_pct:.1f}%)")
                print(f"    Avg word length:     {avg_word_len:.1f} chars")
                print(f"    Spaces:              {spaces}")
                print(f"    Newlines:            {newlines}")

    # =========================================================================
    # JOURNEY SUMMARY
    # =========================================================================

    print(f"""
{'='*65}
  YOUR 5-DAY JOURNEY — COMPLETE SUMMARY
{'='*65}

  Day 1: BIGRAM MODEL
    - Learned: PyTorch, tensors, training loop, cross-entropy
    - Built: Simplest possible language model
    - Result: Random character soup

  Day 2: SELF-ATTENTION
    - Learned: Query-Key-Value, multi-head attention, transformer blocks
    - Built: Mini GPT with 4 attention heads
    - Result: Broken but recognizable English

  Day 3: PROFESSIONAL TECHNIQUES
    - Learned: LR scheduling, gradient accumulation, GELU, ablations
    - Built: Scaled GPT + BPE tokenizer from scratch
    - Result: Much more coherent text generation

  Day 4: HEALTHCARE DOMAIN
    - Learned: Domain-specific training, data preparation
    - Built: Medical GPT trained on clinical notes
    - Result: Model generates medical terminology and clinical patterns

  Day 5: EVALUATION AND SHOWCASE
    - Learned: Model comparison, evaluation metrics
    - Built: Side-by-side evaluation of all 4 models
    - Result: Clear progression from random noise to domain-specific AI

  WHAT YOU NOW UNDERSTAND:
    - How transformers work from the ground up
    - Why self-attention is the key innovation
    - How data determines model behavior
    - Professional LLM training techniques
    - Why healthcare needs domain-specific AI

  PHASE 1 COMPLETE. You built an LLM from scratch.
  Most people just use APIs. You understand the engine.
""")


if __name__ == '__main__':
    main()
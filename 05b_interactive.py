"""
=============================================================================
DAY 5B: INTERACTIVE TEXT GENERATOR
=============================================================================
Author: Jathin | Healthcare LLM Project

Type a starting text and watch your trained models complete it!
Switch between Shakespeare and Medical models.

How to run:
    python 05b_interactive.py
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os


# =============================================================================
# MODEL DEFINITION (Scaled GPT - same as Day 3/4)
# =============================================================================

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


class ScaledGPT(nn.Module):
    def __init__(self, vocab_size, n_embd=96, n_head=6, n_layer=6, block_size=128, dropout=0.15):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.h = nn.ModuleList([TransformerBlock(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
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
# INTERACTIVE SESSION
# =============================================================================

def load_model(path, device='cpu'):
    if not os.path.exists(path):
        return None, None
    ckpt = torch.load(path, map_location=device, weights_only=False)
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
    return model, tokenizer


def interactive_session():
    device = 'cpu'

    print("=" * 65)
    print("  INTERACTIVE TEXT GENERATOR")
    print("  Healthcare LLM Project — Day 5")
    print("=" * 65)

    # Load models
    available = {}

    print("\n  Loading models...")

    shk_model, shk_tok = load_model('scaled_gpt_model.pt', device)
    if shk_model:
        available['1'] = ('Shakespeare GPT', shk_model, shk_tok)
        print("    [1] Shakespeare GPT - loaded")

    med_model, med_tok = load_model('medical_gpt_model.pt', device)
    if med_model:
        available['2'] = ('Medical GPT', med_model, med_tok)
        print("    [2] Medical GPT - loaded")

    if not available:
        print("\n  No models found! Run Days 3-4 first.")
        return

    # Settings
    current_model = list(available.keys())[0]
    temperature = 0.8
    top_k = 10
    num_chars = 300

    print(f"""
  =================================================================
  COMMANDS:
    Type any text     → Model completes it
    /switch           → Switch between Shakespeare and Medical model
    /temp 0.5         → Change temperature (0.1 to 2.0)
    /topk 10          → Change top-k sampling (1 to 50)
    /length 500       → Change output length
    /compare          → Generate from BOTH models side by side
    /samples          → Show pre-made example prompts
    /quit             → Exit
  =================================================================
  """)

    name, model, tok = available[current_model]
    print(f"  Active model: {name}")
    print(f"  Temperature: {temperature} | Top-k: {top_k} | Length: {num_chars}")
    print()

    while True:
        try:
            user_input = input("  You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        # Commands
        if user_input == '/quit':
            print("\n  Goodbye! Phase 1 complete.")
            break

        elif user_input == '/switch':
            keys = list(available.keys())
            idx = keys.index(current_model)
            current_model = keys[(idx + 1) % len(keys)]
            name, model, tok = available[current_model]
            print(f"\n  Switched to: {name}\n")
            continue

        elif user_input.startswith('/temp'):
            try:
                temperature = float(user_input.split()[1])
                temperature = max(0.1, min(2.0, temperature))
                print(f"\n  Temperature set to: {temperature}\n")
            except (IndexError, ValueError):
                print("\n  Usage: /temp 0.8\n")
            continue

        elif user_input.startswith('/topk'):
            try:
                top_k = int(user_input.split()[1])
                top_k = max(1, min(50, top_k))
                print(f"\n  Top-k set to: {top_k}\n")
            except (IndexError, ValueError):
                print("\n  Usage: /topk 10\n")
            continue

        elif user_input.startswith('/length'):
            try:
                num_chars = int(user_input.split()[1])
                num_chars = max(50, min(1000, num_chars))
                print(f"\n  Output length set to: {num_chars}\n")
            except (IndexError, ValueError):
                print("\n  Usage: /length 300\n")
            continue

        elif user_input == '/compare':
            print(f"\n  Generating from ALL models...\n")
            prompt_text = input("  Enter starting text: ").strip()
            if not prompt_text:
                prompt_text = "The"

            with torch.no_grad():
                for key in available:
                    aname, amodel, atok = available[key]
                    encoded = atok.encode(prompt_text)
                    if not encoded:
                        print(f"\n  [{aname}] Cannot encode that text\n")
                        continue
                    ctx = torch.tensor([encoded], dtype=torch.long, device=device)
                    out = amodel.generate(ctx, num_chars, temperature=temperature, top_k=top_k)
                    text = atok.decode(out[0].tolist())
                    print(f"  --- {aname} ---")
                    print(f"  {text}\n")
            continue

        elif user_input == '/samples':
            print("""
  Example prompts to try:

  For Shakespeare model:
    The king
    My lord
    What is

  For Medical model:
    Patient
    Diagnosis
    Assessment
    HISTORY
    The treatment
""")
            continue

        # Generate text
        encoded = tok.encode(user_input)
        if not encoded:
            print(f"\n  Cannot encode that text. Try different characters.\n")
            continue

        ctx = torch.tensor([encoded], dtype=torch.long, device=device)

        with torch.no_grad():
            out = model.generate(ctx, num_chars, temperature=temperature, top_k=top_k)
            text = tok.decode(out[0].tolist())

        print(f"\n  [{name}]")
        print(f"  {text}\n")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    interactive_session()
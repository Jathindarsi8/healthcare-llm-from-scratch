"""
=============================================================================
DAY 4B: MEDICAL GPT — Train on Healthcare Text
=============================================================================
Author: Jathin | Healthcare LLM Project

Train the SAME architecture from Day 3 on MEDICAL text instead of Shakespeare.
Then compare both models side by side.

Same model + different data = completely different behavior.
THIS is the power of domain-specific training.

How to run:
    python 04a_medical_data.py     (create dataset first!)
    python 04b_medical_gpt.py      (then train)
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


def get_lr(it, config):
    if it < config.warmup_iters:
        return config.max_lr * (it + 1) / config.warmup_iters
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.max_lr - config.min_lr)


# =============================================================================
# MODEL (Same architecture as Day 3 — only data changes!)
# =============================================================================

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
        ))


class CausalSelfAttention(nn.Module):
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


class GPTModel(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.h = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

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


def train_model(model, train_data, val_data, config, model_name="Model"):
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

    print(f"\n  Training {model_name}...")
    print(f"  {'Step':>6} | {'Train Loss':>11} | {'Val Loss':>11} | {'Time':>8}")
    print("  " + "-" * 50)

    steps_list = []
    train_losses = []
    val_losses = []
    start_time = time.time()

    for iter_num in range(config.max_iters + 1):
        if iter_num % config.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, config)
            elapsed = time.time() - start_time
            pct = iter_num / config.max_iters * 100
            print(f"  {iter_num:6d} | {losses['train']:11.4f} | {losses['val']:11.4f} | {elapsed:7.1f}s | {pct:.0f}%")
            steps_list.append(iter_num)
            train_losses.append(losses['train'].item())
            val_losses.append(losses['val'].item())

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
    print(f"\n  Done! {total_time:.1f}s | Final val loss: {val_losses[-1]:.4f}")

    return steps_list, train_losses, val_losses


# =============================================================================
# COMPARISON
# =============================================================================

def compare_models(med_model, med_tokenizer, shk_model, shk_tokenizer, config):
    print("\n" + "=" * 65)
    print("  HEAD-TO-HEAD: Shakespeare GPT vs Medical GPT")
    print("=" * 65)

    med_model.eval()
    shk_model.eval()

    print("\n  SAME architecture. SAME training. DIFFERENT data.")
    print("  Watch how the data changes everything:\n")

    # Generate from both models
    with torch.no_grad():
        # Shakespeare model
        ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        shk_out = shk_model.generate(ctx, 400, temperature=0.8, top_k=10)
        shk_text = shk_tokenizer.decode(shk_out[0].tolist())

        # Medical model
        ctx = torch.zeros((1, 1), dtype=torch.long, device=config.device)
        med_out = med_model.generate(ctx, 400, temperature=0.8, top_k=10)
        med_text = med_tokenizer.decode(med_out[0].tolist())

    print("  " + "-" * 60)
    print("  SHAKESPEARE MODEL (trained on plays):")
    print("  " + "-" * 60)
    print(f"  {shk_text[:400]}")

    print("\n  " + "-" * 60)
    print("  MEDICAL MODEL (trained on clinical notes):")
    print("  " + "-" * 60)
    print(f"  {med_text[:400]}")

    # Generate with different prompts from medical model
    print("\n  " + "-" * 60)
    print("  MEDICAL MODEL — More samples:")
    print("  " + "-" * 60)

    prompts = ["P", "D", "A", "H", "M"]
    prompt_names = ["P (Patient...)", "D (Diagnosis...)", "A (Assessment...)",
                    "H (History...)", "M (Medications...)"]

    with torch.no_grad():
        for char, name in zip(prompts, prompt_names):
            if char in med_tokenizer.stoi:
                ctx = torch.tensor([[med_tokenizer.stoi[char]]],
                                   dtype=torch.long, device=config.device)
                out = med_model.generate(ctx, 150, temperature=0.7, top_k=10)
                text = med_tokenizer.decode(out[0].tolist())
                print(f"\n  Starting with {name}:")
                print(f"  {text[:150]}")

    print("""
  =================================================================
  KEY INSIGHT:
  =================================================================

  Both models have the EXACT same architecture:
  - Same number of parameters (~400K)
  - Same attention mechanism (6 heads, 6 layers)
  - Same training techniques (warmup, cosine decay, etc.)

  The ONLY difference is the training data:
  - Shakespeare model learned: thou, hath, thy, wherefore
  - Medical model learned: patient, diagnosis, treatment, ICD-10

  THIS is why domain-specific training matters for healthcare AI.
  A general-purpose LLM trained on internet text will never understand
  medical terminology as deeply as one trained on clinical notes.

  Your 5 years at BCBS taught you this intuitively.
  Now you've proven it with code.
""")


def plot_comparison(shk_steps, shk_losses, med_steps, med_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(shk_steps, shk_losses, 'b-', label='Shakespeare GPT', linewidth=2)
    plt.plot(med_steps, med_losses, 'r-', label='Medical GPT', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Validation Loss')
    plt.title('Shakespeare vs Medical GPT Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_loss_day4.png', dpi=150)
    print(f"  Plot saved to: training_loss_day4.png")


# =============================================================================
# VOCABULARY ANALYSIS
# =============================================================================

def analyze_vocabularies(shk_tokenizer, med_tokenizer):
    print("\n" + "=" * 65)
    print("  VOCABULARY ANALYSIS")
    print("=" * 65)

    shk_chars = set(shk_tokenizer.chars)
    med_chars = set(med_tokenizer.chars)

    shared = shk_chars & med_chars
    shk_only = shk_chars - med_chars
    med_only = med_chars - shk_chars

    print(f"\n  Shakespeare vocab size: {len(shk_chars)}")
    print(f"  Medical vocab size:     {len(med_chars)}")
    print(f"  Shared characters:      {len(shared)}")

    if shk_only:
        print(f"  Shakespeare only:       {sorted(shk_only)}")
    if med_only:
        print(f"  Medical only:           {sorted(med_only)}")

    print("""
  Both use character-level tokenization, so vocab differences are small.
  The real difference is in CHARACTER FREQUENCY and PATTERNS:

  Shakespeare: More uppercase (character names), archaic punctuation
  Medical: More numbers (lab values, codes), colons, periods in codes
""")


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
    print("  DAY 4B: MEDICAL GPT")
    print("  Training on Healthcare Text")
    print("  Estimated time: ~8-10 minutes on CPU")
    print("=" * 65)

    # Check files
    if not os.path.exists('medical_text.txt'):
        print("\n  medical_text.txt not found!")
        print("  Run this first: python 04a_medical_data.py")
        return

    if not os.path.exists('input.txt'):
        print("\n  input.txt (Shakespeare) not found!")
        return

    # =========================================================================
    # STEP 1: Load Medical Data
    # =========================================================================
    print("\n  Loading medical data...")
    with open('medical_text.txt', 'r', encoding='utf-8') as f:
        med_text = f.read()
    med_tokenizer = CharTokenizer(med_text)
    med_data = torch.tensor(med_tokenizer.encode(med_text), dtype=torch.long)
    n = int(0.9 * len(med_data))
    med_train = med_data[:n]
    med_val = med_data[n:]
    print(f"  Medical: {len(med_data):,} tokens | Vocab: {med_tokenizer.vocab_size}")

    # =========================================================================
    # STEP 2: Load Shakespeare Data
    # =========================================================================
    print("\n  Loading Shakespeare data...")
    with open('input.txt', 'r', encoding='utf-8') as f:
        shk_text = f.read()
    shk_tokenizer = CharTokenizer(shk_text)
    shk_data = torch.tensor(shk_tokenizer.encode(shk_text), dtype=torch.long)
    n2 = int(0.9 * len(shk_data))
    shk_train = shk_data[:n2]
    shk_val = shk_data[n2:]
    print(f"  Shakespeare: {len(shk_data):,} tokens | Vocab: {shk_tokenizer.vocab_size}")

    # =========================================================================
    # STEP 3: Train Medical Model
    # =========================================================================
    print("\n" + "=" * 65)
    print("  TRAINING MEDICAL GPT")
    print("=" * 65)
    med_model = GPTModel(med_tokenizer.vocab_size, config).to(config.device)
    med_params = sum(p.numel() for p in med_model.parameters())
    print(f"  Parameters: {med_params:,}")

    med_steps, med_train_losses, med_val_losses = train_model(
        med_model, med_train, med_val, config, "Medical GPT"
    )

    # =========================================================================
    # STEP 4: Train Shakespeare Model (for comparison)
    # =========================================================================
    print("\n" + "=" * 65)
    print("  TRAINING SHAKESPEARE GPT (for comparison)")
    print("=" * 65)

    torch.manual_seed(config.seed)
    shk_model = GPTModel(shk_tokenizer.vocab_size, config).to(config.device)
    print(f"  Parameters: {sum(p.numel() for p in shk_model.parameters()):,}")

    shk_steps, shk_train_losses, shk_val_losses = train_model(
        shk_model, shk_train, shk_val, config, "Shakespeare GPT"
    )

    # =========================================================================
    # STEP 5: Compare!
    # =========================================================================
    analyze_vocabularies(shk_tokenizer, med_tokenizer)
    compare_models(med_model, med_tokenizer, shk_model, shk_tokenizer, config)
    plot_comparison(shk_steps, shk_val_losses, med_steps, med_val_losses)

    # =========================================================================
    # STEP 6: Save
    # =========================================================================
    torch.save({
        'model_state_dict': med_model.state_dict(),
        'tokenizer_chars': med_tokenizer.chars,
        'config': {
            'vocab_size': med_tokenizer.vocab_size,
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'n_layer': config.n_layer,
            'block_size': config.block_size,
        },
    }, 'medical_gpt_model.pt')
    print(f"\n  Medical model saved to: medical_gpt_model.pt")

    print(f"""
=================================================================
  DAY 4 COMPLETE!
=================================================================

  What you did today:
    - Created a medical text dataset (clinical notes + medical knowledge)
    - Trained a GPT model on medical text
    - Trained a GPT model on Shakespeare (same architecture)
    - Compared both models side by side
    - Proved that DATA determines what a model knows

  Your medical model now generates:
    - Clinical note patterns
    - Medical terminology
    - ICD-10 code references
    - Assessment and plan structures

  Your Shakespeare model generates:
    - Character dialogue
    - Elizabethan English
    - Play formatting

  Same brain, different education. Just like doctors vs poets.

  Push to GitHub:
    git add .
    git commit -m "Day 4: Medical GPT - trained on healthcare text"
    git push
""")


if __name__ == '__main__':
    main()
"""
=============================================================================
DAY 7A: TRAIN A MEDICAL BPE TOKENIZER
=============================================================================
Author: Jathin | Healthcare LLM Project

Trains a BPE tokenizer on your medical data from Day 6.
Uses HuggingFace tokenizers library (production-grade, blazing fast).
Falls back to custom implementation if not available.

How to run:
    pip install tokenizers
    python 07a_train_medical_tokenizer.py

Output: medical_tokenizer/ (saved tokenizer files)
=============================================================================
"""

import os
import json
import time
import re
from collections import Counter

# Try to import HuggingFace tokenizers (fast, production-grade)
try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False


# =============================================================================
# OPTION 1: HuggingFace BPE Tokenizer (Production-Grade)
# =============================================================================

def train_hf_tokenizer(data_file, vocab_size=4096, save_dir='medical_tokenizer'):
    """
    Train a BPE tokenizer using HuggingFace tokenizers library.

    This is the SAME library used to train tokenizers for:
    - GPT-2, GPT-3, GPT-4
    - LLaMA, Mistral
    - Most production LLMs

    It's written in Rust, so it's extremely fast.
    """
    print("  Using HuggingFace tokenizers library (production-grade)")
    print(f"  Target vocab size: {vocab_size}")

    # Create a BPE tokenizer
    tokenizer = Tokenizer(models.BPE())

    # Pre-tokenizer: split on whitespace and punctuation
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Decoder
    tokenizer.decoder = decoders.ByteLevel()

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|pad|>", "<|eos|>", "<|unk|>"],
        min_frequency=2,
        show_progress=True,
    )

    # Train!
    print(f"  Training tokenizer on {data_file}...")
    start = time.time()
    tokenizer.train([data_file], trainer)
    elapsed = time.time() - start
    print(f"  Training complete in {elapsed:.1f}s")

    # Save
    os.makedirs(save_dir, exist_ok=True)
    tokenizer.save(os.path.join(save_dir, 'tokenizer.json'))
    print(f"  Saved to: {save_dir}/tokenizer.json")

    return tokenizer


# =============================================================================
# OPTION 2: Custom BPE Tokenizer (From Scratch)
# =============================================================================

class MedicalBPETokenizer:
    """
    BPE Tokenizer built from scratch — same algorithm as Day 3,
    but improved and trained on real medical data.
    """

    def __init__(self):
        self.merges = {}
        self.vocab = {}
        self.inverse_vocab = {}
        self.vocab_size = 0

    def _get_pairs(self, token_lists, freqs):
        pairs = Counter()
        for tokens, freq in zip(token_lists, freqs):
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += freq
        return pairs

    def _merge_pair(self, token_lists, pair, new_token):
        new_lists = []
        for tokens in token_lists:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_lists.append(new_tokens)
        return new_lists

    def train(self, text, vocab_size=4096):
        print(f"  Using custom BPE implementation")
        print(f"  Target vocab size: {vocab_size}")

        start = time.time()

        # Split into words
        words = text.split()
        word_freqs = Counter(words)

        # Initial tokenization: characters
        token_lists = []
        freq_list = []
        for word, freq in word_freqs.most_common():
            token_lists.append(list(word))
            freq_list.append(freq)

        # Build initial vocab
        all_chars = set()
        for tokens in token_lists:
            all_chars.update(tokens)

        self.vocab = {i: ch for i, ch in enumerate(sorted(all_chars))}
        self.inverse_vocab = {ch: i for i, ch in self.vocab.items()}
        next_id = len(self.vocab)

        # Add special tokens
        for special in ['<|pad|>', '<|eos|>', '<|unk|>']:
            self.vocab[next_id] = special
            self.inverse_vocab[special] = next_id
            next_id += 1

        num_merges = vocab_size - len(self.vocab)
        print(f"  Initial vocab: {len(self.vocab)} chars")
        print(f"  Merges to perform: {num_merges}")

        for step in range(num_merges):
            pairs = self._get_pairs(token_lists, freq_list)

            if not pairs:
                break

            best_pair = pairs.most_common(1)[0][0]
            new_token = best_pair[0] + best_pair[1]

            self.merges[best_pair] = new_token
            self.vocab[next_id] = new_token
            self.inverse_vocab[new_token] = next_id
            next_id += 1

            token_lists = self._merge_pair(token_lists, best_pair, new_token)

            if step < 10 or step % 500 == 0:
                print(f"    Merge {step}: '{best_pair[0]}' + '{best_pair[1]}' → '{new_token}'")

        elapsed = time.time() - start
        self.vocab_size = len(self.vocab)
        print(f"  Training complete in {elapsed:.1f}s")
        print(f"  Final vocab size: {self.vocab_size}")

    def encode(self, text):
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
                else:
                    unk_id = self.inverse_vocab.get('<|unk|>', 0)
                    all_ids.append(unk_id)
        return all_ids

    def decode(self, ids):
        tokens = []
        for id in ids:
            if id in self.vocab:
                tokens.append(self.vocab[id])
        return ' '.join(tokens)

    def tokenize(self, text):
        words = text.split()
        all_tokens = []
        for word in words:
            tokens = list(word)
            for pair, new_token in self.merges.items():
                i = 0
                while i < len(tokens) - 1:
                    if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                        tokens = tokens[:i] + [new_token] + tokens[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(tokens)
        return all_tokens

    def save(self, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        data = {
            'vocab': {str(k): v for k, v in self.vocab.items()},
            'inverse_vocab': self.inverse_vocab,
            'merges': {f"{k[0]}|||{k[1]}": v for k, v in self.merges.items()},
            'vocab_size': self.vocab_size,
        }
        path = os.path.join(save_dir, 'tokenizer_custom.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved to: {path}")

    @classmethod
    def load(cls, save_dir):
        path = os.path.join(save_dir, 'tokenizer_custom.json')
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {int(k): v for k, v in data['vocab'].items()}
        tok.inverse_vocab = data['inverse_vocab']
        tok.merges = {}
        for k, v in data['merges'].items():
            parts = k.split('|||')
            tok.merges[(parts[0], parts[1])] = v
        tok.vocab_size = data['vocab_size']
        return tok


# =============================================================================
# ANALYSIS AND COMPARISON
# =============================================================================

def analyze_tokenizer(tokenizer, is_hf=False):
    """Analyze how the tokenizer handles medical text."""

    test_sentences = [
        "Patient presents with type 2 diabetes mellitus",
        "ICD-10 code E11.65 for diabetes with hyperglycemia",
        "Metformin 1000mg PO BID prescribed for glucose control",
        "Blood pressure 142/88 mmHg heart rate 78 bpm",
        "Echocardiogram shows LVEF 35 percent with wall motion abnormality",
        "Assessment: Acute decompensated heart failure",
        "HbA1c 8.2 percent above target of 7 percent",
        "Continue lisinopril 20mg daily and atorvastatin 40mg QHS",
        "Patient denies chest pain shortness of breath or palpitations",
        "Follow up in 4 weeks with repeat labs and medication review",
    ]

    print(f"\n  Medical Text Tokenization:")
    print(f"  {'Sentence':<55s} | {'Chars':>5s} | {'Tokens':>6s} | {'Ratio':>5s}")
    print(f"  {'-'*80}")

    total_chars = 0
    total_tokens = 0

    for sent in test_sentences:
        if is_hf:
            encoded = tokenizer.encode(sent)
            num_tokens = len(encoded.ids)
            token_strs = encoded.tokens
        else:
            token_strs = tokenizer.tokenize(sent)
            num_tokens = len(token_strs)

        num_chars = len(sent)
        ratio = num_chars / max(num_tokens, 1)
        total_chars += num_chars
        total_tokens += num_tokens

        display = sent[:52] + "..." if len(sent) > 55 else sent
        print(f"  {display:<55s} | {num_chars:>5d} | {num_tokens:>6d} | {ratio:>5.1f}x")

    overall_ratio = total_chars / max(total_tokens, 1)
    print(f"  {'-'*80}")
    print(f"  {'OVERALL':<55s} | {total_chars:>5d} | {total_tokens:>6d} | {overall_ratio:>5.1f}x")

    # Show detailed tokenization for key medical phrases
    print(f"\n  Detailed tokenization of medical terms:")

    key_phrases = [
        "E11.65",
        "metformin",
        "hypertension",
        "echocardiogram",
        "HbA1c",
        "LVEF",
        "ICD-10",
    ]

    for phrase in key_phrases:
        if is_hf:
            encoded = tokenizer.encode(phrase)
            tokens = encoded.tokens
        else:
            tokens = tokenizer.tokenize(phrase)
        print(f"    '{phrase}' → {tokens} ({len(tokens)} tokens)")

    return overall_ratio


def compare_with_character(text_sample):
    """Compare BPE vs character-level tokenization."""
    char_tokens = len(text_sample)
    words = len(text_sample.split())

    print(f"\n  Character-level vs BPE on same text ({words} words):")
    print(f"    Character tokens: {char_tokens}")
    print(f"    This means with block_size=128:")
    print(f"    Character model sees: ~{128 // max(char_tokens // words, 1)} words of context")
    print(f"    BPE model sees:       ~{128 * words // max(char_tokens // 4, 1)} words of context")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  DAY 7A: TRAIN MEDICAL BPE TOKENIZER")
    print("  Healthcare LLM Project — Phase 2")
    print("=" * 65)

    # Find data file
    data_file = None
    for f in ['prepared_medical_data.txt', 'pubmed_medical_data.txt', 'medical_text.txt']:
        if os.path.exists(f):
            data_file = f
            break

    if not data_file:
        print("\n  No medical data found! Run Day 6 first:")
        print("    python 06a_download_pubmed.py")
        print("    python 06b_prepare_data.py")
        return

    print(f"\n  Data file: {data_file}")

    with open(data_file, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"  Data size: {len(text):,} characters")

    vocab_size = 4096
    save_dir = 'medical_tokenizer'

    # Train tokenizer
    print(f"\n  Training BPE tokenizer (vocab_size={vocab_size})...\n")

    if HAS_HF_TOKENIZERS:
        tokenizer = train_hf_tokenizer(data_file, vocab_size, save_dir)
        is_hf = True

        # Analyze
        ratio = analyze_tokenizer(tokenizer, is_hf=True)

        # Show vocab samples
        vocab = tokenizer.get_vocab()
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

        print(f"\n  Vocabulary highlights:")
        print(f"  Total tokens: {len(vocab)}")

        # Find medical tokens
        medical_tokens = [(tok, id) for tok, id in sorted_vocab
                          if len(tok) > 3 and not tok.startswith('Ġ') and tok.isalpha()]
        medical_tokens.sort(key=lambda x: len(x[0]), reverse=True)

        print(f"\n  Longest learned tokens (most merged):")
        for tok, id in medical_tokens[:20]:
            print(f"    '{tok}' (id={id})")

    else:
        print("  HuggingFace tokenizers not installed.")
        print("  Using custom BPE implementation.\n")
        print("  Tip: For faster training, install: pip install tokenizers\n")

        tokenizer = MedicalBPETokenizer()
        tokenizer.train(text, vocab_size)
        tokenizer.save(save_dir)
        is_hf = False

        ratio = analyze_tokenizer(tokenizer, is_hf=False)

    # Compare with character-level
    sample = text[:500]
    compare_with_character(sample)

    # Save vocab size for Day 7b
    config = {
        'vocab_size': vocab_size,
        'data_file': data_file,
        'is_hf': is_hf,
        'compression_ratio': ratio,
    }
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"""
{'='*65}
  MEDICAL BPE TOKENIZER COMPLETE
{'='*65}

  Tokenizer saved to: {save_dir}/
  Vocab size: {vocab_size}
  Compression ratio: {ratio:.1f}x vs character-level

  What this means:
    Your model now reads medical text {ratio:.0f}x more efficiently.
    Same context window = {ratio:.0f}x more medical information per prediction.

  The tokenizer learned to keep intact:
    - Common medical words (patient, diagnosis, treatment)
    - Drug names (metformin, lisinopril, atorvastatin)
    - Medical abbreviations
    - Clinical documentation patterns

  Next: python 07b_train_with_bpe.py
  This trains your GPT model using BPE tokens instead of characters!
""")


if __name__ == '__main__':
    main()
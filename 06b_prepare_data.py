"""
=============================================================================
DAY 6B: PREPARE MEDICAL DATA FOR TRAINING
=============================================================================
Author: Jathin | Healthcare LLM Project

Cleans, analyzes, and prepares the PubMed data for LLM training.
Shows you exactly what your model will learn from.

How to run:
    python 06b_prepare_data.py

Input: pubmed_medical_data.txt (from 06a)
Output: prepared_medical_data.txt (cleaned, ready for training)
=============================================================================
"""

import os
import re
from collections import Counter


def clean_text(text):
    """Clean medical text for training."""

    # Remove excessive whitespace
    text = re.sub(r'\n{4,}', '\n\n\n', text)
    text = re.sub(r' {3,}', '  ', text)

    # Remove common artifacts
    text = text.replace('\r\n', '\n')
    text = text.replace('\r', '\n')

    # Remove very short lines (likely noise)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if len(stripped) == 0:
            cleaned_lines.append('')
        elif len(stripped) >= 3:  # Keep lines with 3+ characters
            cleaned_lines.append(stripped)

    text = '\n'.join(cleaned_lines)

    return text


def analyze_vocabulary(text):
    """Analyze the medical vocabulary in the dataset."""

    words = re.findall(r'[a-zA-Z]+', text.lower())
    word_freq = Counter(words)

    # Medical terms to look for
    medical_terms = {
        'conditions': ['diabetes', 'hypertension', 'heart', 'failure', 'kidney',
                       'disease', 'cancer', 'stroke', 'pneumonia', 'asthma',
                       'copd', 'depression', 'anxiety', 'obesity', 'infection',
                       'fibrillation', 'infarction', 'neuropathy', 'retinopathy'],
        'clinical': ['patient', 'diagnosis', 'treatment', 'symptoms', 'assessment',
                     'clinical', 'therapy', 'prognosis', 'etiology', 'pathology',
                     'acute', 'chronic', 'presenting', 'examination', 'history'],
        'medications': ['metformin', 'insulin', 'lisinopril', 'atorvastatin',
                        'aspirin', 'warfarin', 'metoprolol', 'amlodipine',
                        'furosemide', 'omeprazole', 'gabapentin', 'sertraline'],
        'lab_values': ['hemoglobin', 'creatinine', 'glucose', 'cholesterol',
                       'troponin', 'potassium', 'sodium', 'albumin',
                       'bilirubin', 'platelets', 'hba1c', 'egfr'],
        'procedures': ['echocardiogram', 'catheterization', 'biopsy', 'endoscopy',
                       'intubation', 'dialysis', 'surgery', 'transplant',
                       'spirometry', 'colonoscopy', 'radiography', 'ultrasound'],
        'abbreviations': ['icd', 'cpt', 'bid', 'tid', 'prn', 'po', 'iv',
                          'mg', 'ml', 'bp', 'hr', 'ecg', 'ct', 'mri',
                          'bmi', 'bun', 'cbc', 'bnp', 'gfr', 'lvef'],
    }

    return word_freq, medical_terms


def analyze_patterns(text):
    """Find clinical documentation patterns."""

    patterns = {
        'SOAP Notes': len(re.findall(r'(?i)(subjective|objective|assessment|plan)\s*:', text)),
        'ICD Codes': len(re.findall(r'(?i)ICD[\-\s]*10?\s*:?\s*[A-Z]\d', text)),
        'Vital Signs': len(re.findall(r'(?i)(blood pressure|bp|heart rate|hr|temperature|spo2)', text)),
        'Medications': len(re.findall(r'(?i)\d+\s*mg', text)),
        'Lab Values': len(re.findall(r'(?i)(mg/dl|mmol|ng/ml|pg/ml|percent|%)', text)),
        'Clinical Headers': len(re.findall(r'(?i)(chief complaint|history|physical exam|assessment|plan|diagnosis)', text)),
        'Medical Sentences': len(re.findall(r'(?i)patient (presents|reports|denies|complains)', text)),
    }

    return patterns


def create_training_splits(text, train_ratio=0.9):
    """Split data into training and validation sets."""

    n = len(text)
    train_end = int(n * train_ratio)

    # Split at a paragraph boundary near the target point
    while train_end < n and text[train_end] != '\n':
        train_end += 1

    train_text = text[:train_end]
    val_text = text[train_end:]

    return train_text, val_text


def main():
    print("=" * 65)
    print("  DAY 6B: PREPARE MEDICAL DATA FOR TRAINING")
    print("=" * 65)

    input_file = 'pubmed_medical_data.txt'

    if not os.path.exists(input_file):
        print(f"\n  {input_file} not found!")
        print(f"  Run first: python 06a_download_pubmed.py")
        return

    # Load data
    print(f"\n  Loading {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    print(f"  Raw size: {len(raw_text):,} characters")

    # =========================================================================
    # STEP 1: CLEAN
    # =========================================================================
    print(f"\n  Step 1: Cleaning text...")
    clean = clean_text(raw_text)
    print(f"  Cleaned size: {len(clean):,} characters")
    print(f"  Removed: {len(raw_text) - len(clean):,} characters of noise")

    # =========================================================================
    # STEP 2: ANALYZE VOCABULARY
    # =========================================================================
    print(f"\n  Step 2: Analyzing medical vocabulary...")
    word_freq, medical_terms = analyze_vocabulary(clean)

    total_words = sum(word_freq.values())
    unique_words = len(word_freq)

    print(f"  Total words:  {total_words:,}")
    print(f"  Unique words: {unique_words:,}")

    print(f"\n  Top 20 most common words:")
    for word, count in word_freq.most_common(20):
        bar = "#" * min(int(count / total_words * 500), 40)
        print(f"    {word:<15s} {count:>8,} ({count/total_words*100:>5.2f}%) {bar}")

    print(f"\n  Medical term frequency:")
    print(f"  {'Category':<20s} | {'Found Terms':>12s} | {'Total Occurrences':>18s}")
    print(f"  {'-'*58}")

    for category, terms in medical_terms.items():
        found = 0
        total = 0
        for term in terms:
            count = word_freq.get(term, 0)
            if count > 0:
                found += 1
                total += count
        print(f"  {category:<20s} | {found:>8d}/{len(terms):<3d} | {total:>18,}")

    # Show specific medical terms
    print(f"\n  Key medical terms in dataset:")
    key_terms = ['patient', 'diabetes', 'hypertension', 'heart', 'treatment',
                 'diagnosis', 'clinical', 'chronic', 'acute', 'medication',
                 'assessment', 'kidney', 'blood', 'failure', 'disease']

    for term in key_terms:
        count = word_freq.get(term, 0)
        if count > 0:
            print(f"    '{term}': {count:,} occurrences")

    # =========================================================================
    # STEP 3: ANALYZE CLINICAL PATTERNS
    # =========================================================================
    print(f"\n  Step 3: Clinical documentation patterns found:")
    patterns = analyze_patterns(clean)

    for pattern, count in patterns.items():
        print(f"    {pattern:<25s}: {count:>6,}")

    # =========================================================================
    # STEP 4: CHARACTER ANALYSIS
    # =========================================================================
    print(f"\n  Step 4: Character analysis...")
    chars = sorted(list(set(clean)))
    vocab_size = len(chars)
    print(f"  Unique characters: {vocab_size}")
    print(f"  Characters: {''.join(chars[:50])}...")

    # Character frequency
    char_freq = Counter(clean)
    print(f"\n  Top 15 characters by frequency:")
    for ch, count in char_freq.most_common(15):
        display = repr(ch) if ch in ' \n\t' else f"'{ch}'"
        pct = count / len(clean) * 100
        print(f"    {display:<8s}: {count:>10,} ({pct:.2f}%)")

    # =========================================================================
    # STEP 5: SAVE PREPARED DATA
    # =========================================================================
    output_file = 'prepared_medical_data.txt'

    print(f"\n  Step 5: Saving prepared data...")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(clean)

    # Also save train/val splits
    train_text, val_text = create_training_splits(clean)

    with open('medical_train.txt', 'w', encoding='utf-8') as f:
        f.write(train_text)

    with open('medical_val.txt', 'w', encoding='utf-8') as f:
        f.write(val_text)

    print(f"  Saved: {output_file} ({len(clean):,} chars)")
    print(f"  Saved: medical_train.txt ({len(train_text):,} chars)")
    print(f"  Saved: medical_val.txt ({len(val_text):,} chars)")

    # =========================================================================
    # STEP 6: SHOW SAMPLES
    # =========================================================================
    print(f"\n  Step 6: Sample text from prepared dataset:")
    print(f"  " + "-" * 60)
    print(f"  {clean[:500]}")
    print(f"  " + "-" * 60)

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"""
{'='*65}
  DATA PREPARATION COMPLETE
{'='*65}

  Dataset ready for training:
    File:       {output_file}
    Characters: {len(clean):,}
    Words:      {total_words:,}
    Vocab size: {vocab_size} characters
    Train set:  {len(train_text):,} characters
    Val set:    {len(val_text):,} characters

  Comparison with previous datasets:
    Shakespeare (Day 1):     1,115,394 characters
    Medical sample (Day 4):  ~200,000 characters
    PubMed data (Day 6):     {len(clean):,} characters

  What your model will learn from this data:
    - Real medical terminology and abbreviations
    - Clinical documentation patterns
    - ICD-10 code references
    - Medication names and dosages
    - Lab value interpretation language
    - Assessment and treatment planning vocabulary

  Push to GitHub:
    git add .
    git commit -m "Day 6: PubMed medical data download and preparation"
    git push

  Tomorrow (Day 7): Train a medical BPE tokenizer on this data!
""")


if __name__ == '__main__':
    main()
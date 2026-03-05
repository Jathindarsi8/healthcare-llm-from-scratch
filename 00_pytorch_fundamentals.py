"""
=============================================================================
PYTORCH FUNDAMENTALS - Hands-on Exercises
=============================================================================
Phase 1, Week 1 | Healthcare LLM Project

Run this file section by section to learn PyTorch from scratch.
Each section builds on the previous one.

How to run:
    python 00_pytorch_fundamentals.py
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


def section_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


# =============================================================================
# SECTION 1: TENSORS - The Foundation
# =============================================================================

def tensors_basics():
    section_header("SECTION 1: TENSORS")

    # --- Creating Tensors ---
    print("--- Creating Tensors ---")

    # From Python lists
    x = torch.tensor([1.0, 2.0, 3.0])
    print(f"From list:     {x}")
    print(f"  Shape:       {x.shape}")
    print(f"  Dtype:       {x.dtype}")
    print(f"  Device:      {x.device}")

    # Common creation functions
    zeros = torch.zeros(2, 3)          # 2x3 matrix of zeros
    ones = torch.ones(2, 3)            # 2x3 matrix of ones
    rand = torch.rand(2, 3)            # 2x3 uniform random [0,1)
    randn = torch.randn(2, 3)          # 2x3 normal distribution
    arange = torch.arange(0, 10, 2)    # [0, 2, 4, 6, 8]

    print(f"\nZeros:\n{zeros}")
    print(f"Random Normal:\n{randn}")
    print(f"Arange: {arange}")

    # --- Tensor Shapes (CRITICAL for deep learning!) ---
    print("\n--- Tensor Shapes ---")

    # Think of shapes as nested containers:
    # [3]       = 1D: a list of 3 numbers
    # [2, 3]    = 2D: 2 rows, 3 columns (matrix)
    # [4, 2, 3] = 3D: 4 matrices, each 2x3 (batch of matrices)

    # Healthcare example: A batch of patient data
    # Shape: [batch_size, num_features]
    # 4 patients, each with 5 features (age, bmi, systolic_bp, diagnosis_count, claims_total)
    patient_data = torch.randn(4, 5)
    print(f"Patient batch shape: {patient_data.shape}")
    print(f"  Patients: {patient_data.shape[0]}")
    print(f"  Features: {patient_data.shape[1]}")

    # Language model shapes:
    # [B, T]    = batch of token sequences
    # [B, T, C] = batch of token sequences with vocab logits
    B, T, C = 2, 8, 65  # batch=2, time=8, channels/vocab=65
    logits = torch.randn(B, T, C)
    print(f"\nLM logits shape: {logits.shape}")
    print(f"  Batch size: {B}")
    print(f"  Sequence length: {T}")
    print(f"  Vocab size: {C}")

    # --- Reshaping (you'll do this A LOT) ---
    print("\n--- Reshaping ---")

    x = torch.arange(12)
    print(f"Original: {x} (shape: {x.shape})")

    # view: reshape (must be contiguous in memory)
    print(f"view(3,4):\n{x.view(3, 4)}")
    print(f"view(2,2,3):\n{x.view(2, 2, 3)}")

    # -1 means "infer this dimension"
    print(f"view(4,-1):\n{x.view(4, -1)}")  # 12/4 = 3 -> [4, 3]
    
     # This exact reshape is used in the bigram model:
    B, T, C = 2, 3, 4
    logits = torch.randn(B, T, C)
    print(f"\nLogits shape: {logits.shape}")
    print(f"Reshaped for cross_entropy: {logits.view(B*T, C).shape}")
    # cross_entropy expects [N, C] not [B, T, C]

    # --- Indexing ---
    print("\n--- Indexing ---")
    x = torch.tensor([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])

    print(f"Full tensor:\n{x}")
    print(f"Row 0:      {x[0]}")       # [1, 2, 3]
    print(f"Col 1:      {x[:, 1]}")    # [2, 5, 8]
    print(f"Last row:   {x[-1]}")      # [7, 8, 9]
    print(f"Last col:   {x[:, -1]}")   # [3, 6, 9]

    # This is used in generation: get last position's logits
    # logits[:, -1, :] → last timestep for all batches

    # --- GPU ---
    print("\n--- Device (CPU/GPU) ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Available device: {device}")

    x = torch.randn(3, 3)
    x_device = x.to(device)
    print(f"Tensor device: {x_device.device}")

    return True


# =============================================================================
# SECTION 2: AUTOGRAD — Automatic Differentiation
# =============================================================================

def autograd_basics():
    section_header("SECTION 2: AUTOGRAD (Automatic Differentiation)")

    print("Autograd is how PyTorch computes gradients automatically.")
    print("This is the ENTIRE basis of how neural networks learn.\n")

    # --- Simple Example ---
    print("--- Simple Gradient ---")
    x = torch.tensor(3.0, requires_grad=True)  # Track this variable
    y = x ** 2 + 2 * x + 1                     # y = x² + 2x + 1

    print(f"x = {x.item()}")
    print(f"y = x² + 2x + 1 = {y.item()}")

    y.backward()  # Compute dy/dx
    
    if x.grad is not None: print(f"dy/dx = 2x + 2 = 2({x.item()}) + 2 = {x.grad.item()}")
    # Should be 8.0 ✓

    # --- Why This Matters ---
    print("\n--- Gradient Descent in Action ---")
    print("Goal: Find x that minimizes y = (x - 5)²")
    print("The minimum is at x = 5\n")

    x = torch.tensor(0.0, requires_grad=True)  # Start at x=0
    learning_rate = 0.1

    for step in range(20):
        y = (x - 5) ** 2           # Loss function
        y.backward()               # Compute gradient

        # Manual gradient descent (normally optimizer does this)

        with torch.no_grad(): 
            grad = x.grad
            assert grad is not None
            x -= learning_rate * grad
            grad.zero_()

            print(f"Step {step:2d}: x = {x.item():.4f}, loss = {y.item():.4f}")

    print(f"\nFinal x = {x.item():.4f} (target was 5.0)")

    # --- Vector Gradients ---
    print("\n--- Vector Gradients (like weight updates) ---")

    # Simulating a single neuron: y = w·x + b
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)  # weights
    b = torch.tensor(0.5, requires_grad=True)               # bias
    x = torch.tensor([2.0, 1.0, 0.5])                       # input (no grad needed)
    target = torch.tensor(5.0)                                # true answer

    # Forward pass
    prediction = torch.dot(w, x) + b   # 1*2 + 2*1 + 3*0.5 + 0.5 = 6.0
    loss = (prediction - target) ** 2   # (6.0 - 5.0)² = 1.0

    # Backward pass
    loss.backward()

    print(f"Prediction: {prediction.item():.2f}")
    print(f"Target:     {target.item():.2f}")
    print(f"Loss:       {loss.item():.2f}")
    print(f"dL/dw:      {w.grad}")
    if b.grad is not None:
        print(f"dL/db: {b.grad.item():.2f}")
    else:
        print("dL/db: No gradient computed")

    return True


# =============================================================================
# SECTION 3: BUILDING NEURAL NETWORKS
# =============================================================================

def neural_network_basics():
    section_header("SECTION 3: NEURAL NETWORKS")

    # --- nn.Module: The Building Block ---
    print("--- Building a Simple Classifier ---")
    print("Healthcare example: Predict claim approval (approve/deny/review)\n")

    class ClaimClassifier(nn.Module):
        """
        A simple feedforward network for claim classification.

        Input: 10 features (diagnosis codes, procedure codes, patient info, etc.)
        Hidden: 64 neurons with ReLU activation
        Output: 3 classes (approve, deny, flag for review)
        """
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(10, 64)    # 10 inputs → 64 hidden
            self.layer2 = nn.Linear(64, 32)    # 64 → 32
            self.layer3 = nn.Linear(32, 3)     # 32 → 3 outputs

        def forward(self, x):
            x = F.relu(self.layer1(x))   # Activation after first layer
            x = F.relu(self.layer2(x))   # Activation after second layer
            x = self.layer3(x)           # No activation on output (logits)
            return x

    model = ClaimClassifier()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: ClaimClassifier")
    print(f"Total parameters: {total_params:,}")
    print(f"  Layer 1: {10*64 + 64} (weights + bias)")
    print(f"  Layer 2: {64*32 + 32}")
    print(f"  Layer 3: {32*3 + 3}")

    # Forward pass with dummy data
    dummy_claims = torch.randn(4, 10)  # 4 claims, 10 features each
    logits = model(dummy_claims)

    print(f"\nInput shape:  {dummy_claims.shape}")
    print(f"Output shape: {logits.shape}")    # [4, 3]
    print(f"Raw logits:\n{logits}")

    # Convert to probabilities
    probs = F.softmax(logits, dim=-1)
    print(f"\nProbabilities:\n{probs}")
    print(f"Predictions: {probs.argmax(dim=-1)}")  # 0=approve, 1=deny, 2=review

    # --- nn.Embedding: Critical for Language Models ---
    print("\n--- nn.Embedding (used in EVERY language model) ---")

    # Embedding = a lookup table
    # Given a token ID → return a vector (the token's "meaning")

    vocab_size = 100   # 100 unique tokens
    embed_dim = 32     # Each token represented as a 32-dimensional vector

    embedding = nn.Embedding(vocab_size, embed_dim)

    # Look up embeddings for token IDs [5, 23, 67]
    token_ids = torch.tensor([5, 23, 67])
    vectors = embedding(token_ids)

    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embed_dim}")
    print(f"Token IDs: {token_ids}")
    print(f"Embedding shape: {vectors.shape}")  # [3, 32]
    print(f"Parameters: {vocab_size * embed_dim:,}")

    # In the bigram model:
    # embedding = nn.Embedding(65, 65)  ← each char maps to a 65-dim vector
    # Those 65 values ARE the logits for predicting the next character!

    return True


# =============================================================================
# SECTION 4: THE TRAINING LOOP — Practice
# =============================================================================

def training_loop_practice():
    section_header("SECTION 4: FULL TRAINING LOOP")

    print("Training a neural network to learn XOR (classic problem)\n")
    print("XOR truth table:")
    print("  0 XOR 0 = 0")
    print("  0 XOR 1 = 1")
    print("  1 XOR 0 = 1")
    print("  1 XOR 1 = 0")
    print()

    # Data
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    # Model (needs hidden layer — XOR is not linearly separable!)
    model = nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    # THE TRAINING LOOP — same pattern for any model!
    print(f"{'Epoch':>6} | {'Loss':>8} | {'Accuracy':>10}")
    print("-" * 30)

    for epoch in range(1001):
        # 1. Forward pass
        logits = model(X)

        # 2. Compute loss
        loss = loss_fn(logits, y)

        # 3. Backward pass
        optimizer.zero_grad()
        loss.backward()

        # 4. Update weights
        optimizer.step()

        # Evaluate
        if epoch % 200 == 0:
            with torch.no_grad():
                preds = model(X).argmax(dim=-1)
                accuracy = (preds == y).float().mean().item()
                print(f"{epoch:6d} | {loss.item():8.4f} | {accuracy:10.1%}")

    # Final predictions
    with torch.no_grad():
        final_logits = model(X)
        final_probs = F.softmax(final_logits, dim=-1)
        final_preds = final_logits.argmax(dim=-1)

    print(f"\nFinal predictions:")
    for i in range(4):
        x_val = X[i].tolist()
        print(f"  {int(x_val[0])} XOR {int(x_val[1])} = {final_preds[i].item()} "
              f"(confidence: {final_probs[i][final_preds[i]].item():.2%})")

    return True


# =============================================================================
# SECTION 5: CROSS-ENTROPY LOSS — Deep Dive
# =============================================================================

def cross_entropy_deep_dive():
    section_header("SECTION 5: CROSS-ENTROPY LOSS (Used in all LLMs)")

    print("Cross-entropy measures how well predicted probabilities")
    print("match the true distribution. It's THE loss function for LLMs.\n")

    # --- Manual computation ---
    print("--- Manual Cross-Entropy ---")

    # Model predicts logits for 3 classes
    logits = torch.tensor([2.0, 1.0, 0.1])  # Raw scores
    target = 0  # True class is 0

    # Step 1: Softmax → probabilities
    probs = F.softmax(logits, dim=0)
    print(f"Logits: {logits}")
    print(f"Softmax probabilities: {probs}")
    print(f"  P(class 0) = {probs[0]:.4f}")
    print(f"  P(class 1) = {probs[1]:.4f}")
    print(f"  P(class 2) = {probs[2]:.4f}")
    print(f"  Sum = {probs.sum():.4f}")  # Always sums to 1

    # Step 2: Negative log of correct class probability
    manual_loss = -torch.log(probs[target])
    print(f"\nManual loss: -log(P(class {target})) = -log({probs[target]:.4f}) = {manual_loss:.4f}")

    # Step 3: Compare with PyTorch
    pytorch_loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([target]))
    print(f"PyTorch loss: {pytorch_loss.item():.4f}")
    print(f"Match: {torch.isclose(manual_loss, pytorch_loss).item()}")

    # --- For Language Models ---
    print("\n--- Cross-Entropy for Language Models ---")
    print("Given character 'h', predict next character from vocab of 5:")
    print("  Vocab: ['a', 'e', 'h', 'l', 'o']")
    print("  True next: 'e' (index 1)")

    logits_lm = torch.tensor([0.5, 2.5, 0.1, 1.0, 0.3])  # Model's scores
    target_lm = 1  # 'e'

    probs_lm = F.softmax(logits_lm, dim=0)
    loss_lm = F.cross_entropy(logits_lm.unsqueeze(0), torch.tensor([target_lm]))

    print(f"\n  Probabilities: {dict(zip(['a','e','h','l','o'], [f'{p:.3f}' for p in probs_lm.tolist()]))}")
    print(f"  P('e') = {probs_lm[1]:.3f} -> Loss = {loss_lm.item():.3f}")
    print(f"\n  If model was perfect: P('e')=1.0 -> Loss = -log(1) = 0")
    print(f"  If model was random:  P('e')=0.2 -> Loss = -log(0.2) = {-torch.log(torch.tensor(0.2)).item():.3f}")

    return True


# =============================================================================
# SECTION 6: KEY CONCEPTS QUIZ
# =============================================================================

def concepts_quiz():
    section_header("SELF-CHECK QUIZ")

    questions = [
        {
            "q": "What does requires_grad=True do?",
            "a": "Tells PyTorch to track all operations on this tensor so it can\n"
                 "   compute gradients via backpropagation. Without it, .backward() won't work."
        },
        {
            "q": "Why do we call optimizer.zero_grad() before backward()?",
            "a": "Because PyTorch ACCUMULATES gradients by default. Without zeroing,\n"
                 "   gradients from previous batches add up, giving incorrect updates."
        },
        {
            "q": "What's the difference between logits and probabilities?",
            "a": "Logits are raw, unbounded scores from the model.\n"
                 "   Probabilities = softmax(logits), bounded to [0,1] and sum to 1."
        },
        {
            "q": "Why does cross_entropy expect [N, C] and not [B, T, C]?",
            "a": "PyTorch's cross_entropy treats the first dim as batch.\n"
                 "   For language models with shape [B, T, C], we reshape to [B*T, C]\n"
                 "   so each position is treated as an independent prediction."
        },
        {
            "q": "What does nn.Embedding actually do?",
            "a": "It's a lookup table. Given integer index i, return row i of a matrix.\n"
                 "   The matrix values are learnable parameters, trained via backprop.\n"
                 "   In LLMs, this converts token IDs -> dense vectors."
        },
        {
            "q": "Why can't we just use nn.Linear layers without activation functions?",
            "a": "Linear(Linear(x)) = Linear(x). Multiple linear layers collapse into one.\n"
                 "   Activations (ReLU, GELU) introduce non-linearity, letting the network\n"
                 "   learn complex patterns. Without them, a 100-layer network = 1-layer network."
        },
        {
            "q": "What's the difference between model.eval() and model.train()?",
            "a": "model.train() enables dropout and batch norm in training mode.\n"
                 "   model.eval() disables them for consistent evaluation.\n"
                 "   Always use model.eval() with torch.no_grad() for evaluation."
        },
    ]

    for i, qa in enumerate(questions, 1):
        print(f"Q{i}: {qa['q']}")
        print(f"A:  {qa['a']}\n")


# =============================================================================
# RUN ALL SECTIONS
# =============================================================================

if __name__ == '__main__':
    print("*** PYTORCH FUNDAMENTALS - Healthcare LLM Project ***")
    print("=" * 60)

    sections = [
        ("Tensors", tensors_basics),
        ("Autograd", autograd_basics),
        ("Neural Networks", neural_network_basics),
        ("Training Loop", training_loop_practice),
        ("Cross-Entropy Loss", cross_entropy_deep_dive),
        ("Concepts Quiz", concepts_quiz),
    ]

    for name, func in sections:
        func()

    print("\n" + "=" * 60)
    print("[OK] ALL SECTIONS COMPLETE!")
    print("=" * 60)
    print("""
Next steps:
  1. Run 01_bigram_model.py to build your first language model
  2. Experiment with changing hyperparameters here
  3. Try the exercises below

Exercises to try:
  - Modify the ClaimClassifier to have 4 hidden layers instead of 2
  - Change the XOR training to use SGD instead of Adam (what happens?)
  - Create an embedding for 1000 medical codes with 128 dimensions
  - Compute cross-entropy loss manually for a 10-class problem
""")
 
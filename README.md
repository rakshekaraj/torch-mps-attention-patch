# 🔧 PyTorch-MPS-AttentionFix

> Optimizing Scaled Dot-Product Attention on Apple’s MPS Backend for Large Sequence Stability

---

### ⚠️ Disclaimer

This repository contains custom modifications to the internal `Attention.mm` implementation of PyTorch's MPS backend.  
It is **not an official contribution** and is intended **solely for research, experimentation, and educational purposes**.  
This project is **not affiliated with or endorsed by Meta or the PyTorch team**.

Use at your own risk. Modifying core PyTorch source files can lead to **unexpected behavior**, instability, or compatibility issues in future versions.

---

### 🧠 Overview

PyTorch's `scaled_dot_product_attention` function on the MPS backend struggles with long sequences (typically >12,000 tokens) due to Metal's memory allocation constraints.  
This project proposes a **custom chunking-based implementation** that:
- Dynamically adjusts chunk sizes to prevent Metal buffer overflows.
- Preserves FP32 numerical stability (unlike FP16-based fallbacks).
- Adds safety checks to avoid shape mismatches and MPS assertion crashes.
- Enables processing sequences up to 16,384 tokens without precision loss.

---

### 🔍 What's Inside

- [`Attention.mm`](./Attention.mm) — Custom Objective-C++ implementation with dynamic chunking and memory safeguards.
- [`test2.py`](./test2.py) — A simple test script to validate behavior across different sequence lengths.

---

### ✅ How It Works

1. **Adaptive Chunking**:  
   Based on input sequence length, the system selects a safe chunk size (e.g., 2048) to split the Q, K, V matrices.

2. **Graph-Safe Execution**:  
   Uses Apple’s `MPSGraph` API to execute matrix multiplications in a loop, chunk-by-chunk.

3. **Shape Consistency**:  
   All chunk shapes are padded or trimmed to prevent placeholder mismatches across Metal operations.

4. **No FP16 Hacks**:  
   All calculations are done in FP32 for better stability and accuracy.

---

### 🧪 Run the Test

```bash
python test2.py

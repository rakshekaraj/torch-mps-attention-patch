import torch

device = torch.device("mps")

def test_attention(seq_len_q, seq_len_kv):
    print(f"\n[TEST] Running with seq_len_q={seq_len_q}, seq_len_kv={seq_len_kv}")

    q = torch.randn(1, 12, seq_len_q, 128, device=device)
    k = torch.randn(1, 12, seq_len_kv, 128, device=device)
    v = torch.randn(1, 12, seq_len_kv, 128, device=device)

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print(f" Success: Output Shape {out.shape}")
    except Exception as e:
        print(f" Failure: {e}")

# Test cases
test_attention(2048, 2048)       # Base test
test_attention(8000, 8000)       # Moderate load
test_attention(10000, 10000)     # High load with chunking
test_attention(12000, 12000)     # Known MPS challenge range
test_attention(16384, 16384)     # Push to limits

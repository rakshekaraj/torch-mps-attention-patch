# import torch

# device = torch.device("mps")

# def test_size1(seq_len):
#     q = torch.randn(1, 12, seq_len, 128).to(device)
#     k = torch.randn(1, 12, seq_len, 128).to(device)
#     v = torch.randn(1, 12, seq_len, 128).to(device)

#     # DEBUG: Print the chunk size we expect
#     chunk_size = 2048 if seq_len > 13000 else seq_len
#     print(f"[DEBUG] Using chunk size: {chunk_size} for seq_len={seq_len}")

#     # Apply chunking manually in test script for debugging
#     for i in range(0, seq_len, chunk_size):
#         q_chunk = q[:, :, i:i + chunk_size, :]
#         k_chunk = k[:, :, i:i + chunk_size, :]
#         v_chunk = v[:, :, i:i + chunk_size, :]

#         print(f"[DEBUG] Processing chunk {i} to {i + chunk_size}")

#         out = torch.nn.functional.scaled_dot_product_attention(q_chunk, k_chunk, v_chunk)
#         print(out.shape)

# test_size1(18720)


#---------------------------------------

import torch

device = torch.device("mps")

def test_attention(seq_len_q, seq_len_kv):
    print(f"\n[TEST] Running with seq_len_q={seq_len_q}, seq_len_kv={seq_len_kv}")

    q = torch.randn(1, 12, seq_len_q, 128).to(device)
    k = torch.randn(1, 12, min(seq_len_q, seq_len_kv), 128).to(device)  # Limit k to q's chunk size
    v = torch.randn(1, 12, min(seq_len_q, seq_len_kv), 128).to(device)  # Limit v similarly

    try:
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        print(f"✅ Success: Output Shape {out.shape}")
    except Exception as e:
        print(f"❌ Failure: {e}")

# ✅ Small sequence length (should work fine)
test_attention(2048, 2048)

test_attention(8000, 8000)

# ✅ Large sequence length (should work fine with chunking)
test_attention(10000, 10000)

# 🔥 Problematic range (13,500 - 26,770) - Metal backend has known issues here
test_attention(12000, 12000)

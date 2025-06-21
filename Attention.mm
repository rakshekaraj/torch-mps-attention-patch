#include <string>
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <iostream>
#include <optional>
#include <ATen/ops/cat.h>  // Explicitly include cat.h for at::cat

#include <ATen/core/Tensor.h>
#include <ATen/native/mps/OperationUtils.h>
#include <ATen/native/transformers/attention.h>
#include <ATen/native/transformers/sdp_utils_cpp.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_scaled_dot_product_attention_math_for_mps_native.h>
#include <ATen/ops/empty_native.h>
#endif

namespace at {
namespace native {

// Dynamically determine optimal chunk size based on available memory
static int64_t get_chunk_size(int64_t seq_len) {
    if (seq_len > 20000) return 1000;  // Very large sequences → smallest chunks
    if (seq_len > 14000) return 1536;  // Adjust to avoid OOM for 14k+
    if (seq_len > 10000) return 2048;  // Large sequences → smaller chunks
    if (seq_len > 8000) return 4088;   // Medium sequences → bigger chunks
    return seq_len;                    // Small seq_len → full sequence
}

std::tuple<Tensor, Tensor> _scaled_dot_product_attention_math_mps(const Tensor& query,
                                                                  const Tensor& key,
                                                                  const Tensor& value,
                                                                  const std::optional<Tensor>& attn_mask,
                                                                  double dropout_p,
                                                                  bool is_causal,
                                                                  const std::optional<Tensor>& dropout_mask,
                                                                  std::optional<double> scale) {
    const auto macOS15_0_plus = is_macos_13_or_newer(MacOSVersion::MACOS_VER_15_0_PLUS);

    // Ensure dropout is disabled
    TORCH_CHECK(dropout_p == 0.0, "_scaled_dot_product_attention_math_mps: dropout_p != 0.0 is not supported");

    // Ensure tensors are contiguous
    TORCH_CHECK(macOS15_0_plus || (query.is_contiguous() && key.is_contiguous() && value.is_contiguous()),
                "_scaled_dot_product_attention_math_mps: query, key, and value must be contiguous");

    // Extract dimensions
    int64_t batchSize = query.size(0);
    int64_t num_heads = query.size(1);
    int64_t seq_len = query.size(2);
    int64_t head_dim = query.size(3);
    int64_t maxSeqLength = key.size(2);

    // Determine optimal chunk size
    int64_t chunk_size = get_chunk_size(seq_len);

    // int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;
    int64_t num_chunks = (seq_len + chunk_size - 1) / chunk_size;  // Ensures rounding up

    // int64_t num_chunks = seq_len/chunk_size;
    // int64_t adjusted_chunk_size = chunk_size/num_chunks;  // Ensure consistent chunking
    int64_t adjusted_chunk_size = (seq_len + num_chunks - 1) / num_chunks;

    std::cout << "[DEBUG] Using chunk size: " << adjusted_chunk_size << " for seq_len=" << seq_len << std::endl;

    // Output storage per chunk
    std::vector<Tensor> out_chunks;
    std::vector<Tensor> attn_chunks;

    // Loop through chunks
    for (int64_t i = 0; i < seq_len; i += adjusted_chunk_size) {
        int64_t end = std::min(i + adjusted_chunk_size, seq_len);
        int64_t current_chunk_size = end - i;


        // // Adjust last chunk: Instead of keeping it small, increase previous chunks
        // if (i + adjusted_chunk_size >= seq_len && current_chunk_size < adjusted_chunk_size) {
        //     std::cout << "[DEBUG] Expanding previous chunks to avoid small last chunk!" << std::endl;
        //     adjusted_chunk_size = (seq_len + num_chunks - 1) / num_chunks;  // Recalculate chunk size
        //     i = 0;  // Restart loop to reprocess
        //     out_chunks.clear();
        //     attn_chunks.clear();
        //     continue;
        // }

        // Chunking Q, K, and V
        auto q_chunk = query.narrow(2, i, current_chunk_size);
        auto k_chunk = key.narrow(2, i, current_chunk_size);
        auto v_chunk = value.narrow(2, i, current_chunk_size);

        // auto k_chunk = key.narrow(2, 0, end);
        // auto v_chunk = value.narrow(2, 0, end);

        //**Pad the last chunk if needed**
        if (current_chunk_size < adjusted_chunk_size) {
            int64_t pad_size = adjusted_chunk_size - current_chunk_size;
            q_chunk = at::cat({q_chunk, at::zeros({batchSize, num_heads, pad_size, head_dim}, query.options())}, 2);
            k_chunk = at::cat({k_chunk, at::zeros({batchSize, num_heads, pad_size, head_dim}, query.options())}, 2);
            v_chunk = at::cat({v_chunk, at::zeros({batchSize, num_heads, pad_size, head_dim}, query.options())}, 2);
        }

        //         if (current_chunk_size != adjusted_chunk_size) {
        //     throw std::runtime_error("Chunk size mismatch! Expected: " + std::to_string(adjusted_chunk_size) +
        //                              " but got: " + std::to_string(current_chunk_size));
        // }

        std::cout << "[DEBUG] Processing chunk: " << i << " to " << end << std::endl;
        std::cout << "[DEBUG] Q_chunk shape: " << q_chunk.sizes() << std::endl;
        std::cout << "[DEBUG] K_chunk shape: " << k_chunk.sizes() << std::endl;
        std::cout << "[DEBUG] V_chunk shape: " << v_chunk.sizes() << std::endl;


        if (q_chunk.size(2) != adjusted_chunk_size || 
            k_chunk.size(2) != adjusted_chunk_size || 
            v_chunk.size(2) != adjusted_chunk_size) {
            throw std::runtime_error("Chunk size mismatch! Expected: " + 
                                    std::to_string(adjusted_chunk_size) + " but got: " +
                                    std::to_string(q_chunk.size(2)));
        }

        // Allocate memory only for this chunk
        auto out_chunk = at::empty({batchSize, num_heads, adjusted_chunk_size, head_dim}, query.options());
        auto attn_chunk = at::empty({batchSize, num_heads, adjusted_chunk_size, maxSeqLength}, query.options());

        // Debug logging
        std::cout << "[DEBUG] Processing chunk: " << i << " to " << end << std::endl;
        std::cout << "[DEBUG] Q_chunk shape: " << q_chunk.sizes() << std::endl;
        std::cout << "[DEBUG] K_chunk shape: " << k_chunk.sizes() << std::endl;
        std::cout << "[DEBUG] V_chunk shape: " << v_chunk.sizes() << std::endl;

        using namespace mps;
        struct CachedGraph : public MPSCachedGraph {
            CachedGraph(MPSGraph* graph) : MPSCachedGraph(graph) {}
            MPSGraphTensor* qTensor = nil;
            MPSGraphTensor* kTensor = nil;
            MPSGraphTensor* vTensor = nil;
            MPSGraphTensor* outputTensor = nil;
            MPSGraphTensor* attnTensor = nil;
        };

        @autoreleasepool {
            auto mkey = __func__ + std::to_string(chunk_size) + ":" + std::to_string(is_causal);
            auto cachedGraph = LookUpOrCreateCachedGraph<CachedGraph>(mkey, [&](auto mpsGraph, auto graph) {
                auto qTensor = mpsGraphRankedPlaceHolder(mpsGraph, q_chunk);
                auto kTensor = mpsGraphRankedPlaceHolder(mpsGraph, k_chunk);
                auto vTensor = mpsGraphRankedPlaceHolder(mpsGraph, v_chunk);

                auto kT = [mpsGraph transposeTensor:kTensor dimension:2 withDimension:3 name:nil];
                auto scale_factor = sdp::calculate_scale(query, scale).expect_float();
                auto scaleTensor = [mpsGraph constantWithScalar:scale_factor shape:getMPSShape({1}) dataType:MPSDataTypeFloat32];

                auto qk_chunk = [mpsGraph matrixMultiplicationWithPrimaryTensor:qTensor secondaryTensor:kT name:nil];
                qk_chunk = [mpsGraph multiplicationWithPrimaryTensor:qk_chunk secondaryTensor:scaleTensor name:nil];

                auto sm_chunk = [mpsGraph softMaxWithTensor:qk_chunk axis:3 name:nil];
                auto output_chunk = [mpsGraph matrixMultiplicationWithPrimaryTensor:sm_chunk secondaryTensor:vTensor name:nil];

                graph->qTensor = qTensor;
                graph->kTensor = kTensor;
                graph->vTensor = vTensor;
                graph->outputTensor = output_chunk;
                graph->attnTensor = sm_chunk;
            });

            auto qPlaceholder = Placeholder(cachedGraph->qTensor, q_chunk);
            auto kPlaceholder = Placeholder(cachedGraph->kTensor, k_chunk);
            auto vPlaceholder = Placeholder(cachedGraph->vTensor, v_chunk);
            auto outputPlaceholder = Placeholder(cachedGraph->outputTensor, out_chunk);
            auto attnPlaceholder = Placeholder(cachedGraph->attnTensor, attn_chunk);

            NSDictionary* feeds = dictionaryFromPlaceholders(qPlaceholder, kPlaceholder, vPlaceholder);
            NSDictionary* outs = dictionaryFromPlaceholders(outputPlaceholder, attnPlaceholder);
            runMPSGraph(getCurrentMPSStream(), cachedGraph->graph(), feeds, outs);
        }

        // Collect results
        out_chunks.push_back(out_chunk);
        attn_chunks.push_back(attn_chunk);
    }

    // Concatenate all chunked outputs
    auto out = at::cat(out_chunks, 2);
    auto attn = at::cat(attn_chunks, 2);

    return {std::move(out), std::move(attn)};
}

} // namespace native
} // namespace at

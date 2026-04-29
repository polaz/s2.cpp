#pragma once
// s2_model.h — Slow-AR + Fast-AR model (Dual-AR Qwen3 transformer)
//
// Loads the Dual-AR portion from a unified GGUF file and provides
// prefill / step / fast decode operations with KV cache.
// Direct port from ggml/examples/fish-speech-slow-ar/main.cpp

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#ifdef GGML_USE_VULKAN
#include "ggml-vulkan.h"
#endif
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include <cstdint>
#include <string>
#include <vector>

namespace s2 {

struct ModelLayer {
    ggml_tensor * attention_norm = nullptr;
    ggml_tensor * ffn_norm       = nullptr;
    ggml_tensor * q_norm         = nullptr;
    ggml_tensor * k_norm         = nullptr;
    ggml_tensor * wqkv           = nullptr;
    ggml_tensor * wo             = nullptr;
    ggml_tensor * w1             = nullptr;
    ggml_tensor * w2             = nullptr;
    ggml_tensor * w3             = nullptr;
};

struct ModelHParams {
    int32_t context_length     = 0;
    int32_t vocab_size         = 0;
    int32_t embedding_length   = 0;
    int32_t feed_forward_length = 0;
    int32_t block_count        = 0;
    int32_t head_count         = 0;
    int32_t head_count_kv      = 0;
    int32_t codebook_size      = 0;
    int32_t num_codebooks      = 0;
    int32_t semantic_begin_id  = 0;
    int32_t semantic_end_id    = 0;
    float   rope_freq_base     = 10000.0f;
    float   rms_norm_eps       = 1e-5f;
    bool    tie_word_embeddings = true;
    bool    attention_qk_norm  = false;
    bool    scale_codebook_embeddings = false;

    // Fast decoder
    int32_t fast_context_length     = 0;
    int32_t fast_embedding_length   = 0;
    int32_t fast_feed_forward_length = 0;
    int32_t fast_block_count        = 0;
    int32_t fast_head_count         = 0;
    int32_t fast_head_count_kv      = 0;
    int32_t fast_head_dim           = 0;
    float   fast_rope_freq_base     = 10000.0f;
    float   fast_rms_norm_eps       = 1e-5f;
    bool    fast_attention_qk_norm  = false;
    bool    fast_has_project_in     = false;
    bool    has_fast_decoder        = false;
};

struct ModelWeights {
    ggml_context * ctx_w = nullptr;
    ggml_backend_buffer_t model_buf = nullptr;

    ggml_tensor * embeddings           = nullptr;
    ggml_tensor * codebook_embeddings  = nullptr;
    ggml_tensor * norm                 = nullptr;
    ggml_tensor * fast_project_in      = nullptr;
    ggml_tensor * fast_embeddings      = nullptr;
    ggml_tensor * fast_norm            = nullptr;
    ggml_tensor * fast_output          = nullptr;

    std::vector<ModelLayer> layers;
    std::vector<ModelLayer> fast_layers;
};

struct StepResult {
    std::vector<float> hidden;   // (embedding_length,) last-token hidden state
    std::vector<float> logits;   // (vocab_size,) logits
};

class SlowARModel {
public:
    SlowARModel();
    ~SlowARModel();

    // Load model from GGUF. gpu_device=-1 means CPU only.
    bool load(const std::string & gguf_path, int32_t gpu_device = -1, int32_t backend_type = -1);

    // Initialize KV cache for generation; also builds persistent decode graphs.
    bool init_kv_cache(int32_t max_seq_len);

    // Reset KV cache position (for new generation without reallocating buffers)
    void reset();

    void clear_kv_cache();

    // Prefill: process prompt tokens. flat_tokens: (num_codebooks+1)*n_tokens
    bool prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                 int32_t n_threads, StepResult & result);

    // Step: process a single timestep using the persistent decode graph.
    bool step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
              StepResult & result);

    // Fast decoder: generate next codebook logits given hidden state and prefix codes
    bool fast_decode(const std::vector<float> & hidden,
                     const std::vector<int32_t> & prefix_codes,
                     int32_t n_threads,
                     std::vector<float> & logits_out);

    const ModelHParams & hparams() const { return hparams_; }

    // Restrict logits readback to [begin, end) — positions outside are returned as -inf.
    // Call before generation to reduce PCIe transfer from vocab_size to semantic range only.
    void set_logits_range(int32_t begin, int32_t end) {
        logits_range_begin_ = begin;
        logits_range_end_   = end;
    }

private:
    ModelHParams   hparams_;
    ModelWeights   weights_;
    ggml_backend_t backend_       = nullptr;
    ggml_backend_t backend_gpu_   = nullptr;
    ggml_backend_t backend_cpu_   = nullptr;
    ggml_gallocr_t allocr_        = nullptr;
    ggml_context * ctx_kv_       = nullptr;
    ggml_backend_buffer_t kv_buf_ = nullptr;
    ggml_tensor *  memory_k_   = nullptr;
    ggml_tensor *  memory_v_   = nullptr;
    int32_t        max_seq_len_ = 0;
    int32_t        n_past_     = 0;
    int32_t        n_gpu_layers_ = 0;
    int32_t        head_dim_   = 0;
    int32_t        logits_range_begin_ = 0;
    int32_t        logits_range_end_   = 0;

    // Per-layer staging buffers for current-step K/V (live in kv_buf_).
    // Graph writes here (fixed destination); post-graph async D2D copy → correct KV slot.
    std::vector<ggml_tensor *> k_cur_stage_;
    std::vector<ggml_tensor *> v_cur_stage_;

    // Pre-allocated KV copy destination views (avoid ggml_init/free per step).
    // data ptr updated each step to point to the correct n_past_ KV slot.
    ggml_context *             ctx_copy_kv_  = nullptr;
    std::vector<ggml_tensor *> k_copy_dst_;   // view into memory_k_, data updated per step
    std::vector<ggml_tensor *> v_copy_dst_;   // view into memory_v_, data updated per step

    // Persistent kq_mask buffer — avoids per-step malloc and enables incremental updates.
    // kq_mask_buf_[0..n_past_-1]=0, [n_past_..max-1]=-inf, [max_seq_len_]=0.
    std::vector<float> kq_mask_buf_;

    // F16 copies of embedding tensors for CUDA get_rows compatibility
    struct {
        ggml_context *       ctx = nullptr;
        ggml_backend_buffer_t buf = nullptr;
        ggml_tensor *        embeddings          = nullptr;
        ggml_tensor *        codebook_embeddings = nullptr;
        ggml_tensor *        fast_embeddings     = nullptr;
    } emb_f16_;

    // ---------------------------------------------------------------------------
    // Persistent decode graph for single-token step (rebuilt each init_kv_cache)
    // kq_mask shape: (max_seq_len_+1, 1) — fixed for stable gallocr allocation.
    // KV past reads the full layer slice; current K/V is appended via concat.
    // k_write_views[il]->data is updated per step to point to the current n_past_
    // slot in memory_k_/memory_v_.
    // ---------------------------------------------------------------------------
    struct DecodeState {
        ggml_context *             ctx              = nullptr;
        ggml_cgraph  *             graph            = nullptr;
        // Input tensors (content updated per call via ggml_backend_tensor_set)
        ggml_tensor *              semantic_ids     = nullptr;
        ggml_tensor *              positions        = nullptr;
        ggml_tensor *              semantic_mask    = nullptr;
        ggml_tensor *              token_scale      = nullptr; // null if !scale_codebook_embeddings
        std::vector<ggml_tensor *> cb_id_tensors;
        ggml_tensor *              kq_mask          = nullptr; // shape: (max_seq_len_+1, 1)
        // Per-layer KV write view tensors; ->data pointer updated per step
        std::vector<ggml_tensor *> k_write_views;
        std::vector<ggml_tensor *> v_write_views;
        // Output tensors
        ggml_tensor *              hidden_last      = nullptr;
        ggml_tensor *              logits           = nullptr;
    } decode_;

    // ---------------------------------------------------------------------------
    // Persistent fast decode graphs, one per n_tokens value (built lazily).
    // fast_decode_states_[n_tokens-1] is built on first call for that n_tokens.
    // These survive clear_kv_cache / init_kv_cache cycles (no KV cache dependency).
    // ---------------------------------------------------------------------------
    struct FastDecodeState {
        ggml_context *  ctx        = nullptr;
        ggml_cgraph  *  graph      = nullptr;
        ggml_gallocr_t  allocr     = nullptr;
        ggml_tensor  *  hidden0    = nullptr;
        ggml_tensor  *  prefix_ids = nullptr; // null if n_tokens == 1
        ggml_tensor  *  positions  = nullptr;
        ggml_tensor  *  kq_mask    = nullptr;
        ggml_tensor  *  logits     = nullptr;
    };
    std::vector<FastDecodeState> fast_decode_states_; // indexed [n_tokens-1]

    static bool backend_requires_single_token_semantic_prefill(ggml_backend_t gpu);

    // Prefill / multi-token eval path (stateless: builds+frees graph each call)
    bool eval_cached(const std::vector<int32_t> & flat_tokens,
                     int32_t n_tokens, int32_t n_threads,
                     StepResult & result);

    // Build the persistent single-token decode graph. Called from init_kv_cache.
    bool build_decode_graph();

    // Run 2 dummy step() calls to trigger CUDA graph capture for decode_.graph.
    // Resets n_past_=0 and restores kq_mask after warmup.
    void warmup_decode_graph();

    // Build a persistent fast decode graph for a given n_tokens. Called lazily.
    bool build_fast_decode_graph(int32_t n_tokens, FastDecodeState & out);
};

} // namespace s2

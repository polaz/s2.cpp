#include "../include/s2_config.h"
#include "../include/s2_model.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <algorithm>
#include <stdexcept>
#ifdef __linux__
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace s2 {

// ---------------------------------------------------------------------------
// Helpers (graph‑level, no side effects)
// ---------------------------------------------------------------------------

static ggml_tensor * repeat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                    const char * label = "repeat") {
    if (!ggml_can_repeat(a, b)) {
        std::fprintf(stderr, "%s a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_repeat(ctx, a, b);
}

static ggml_tensor * mul_mat_checked(ggml_context * ctx, ggml_tensor * a, ggml_tensor * b,
                                     const char * label = "mul_mat") {
    const bool can_mul =
        a->ne[0] == b->ne[0] &&
        (b->ne[2] % a->ne[2] == 0) &&
        (b->ne[3] % a->ne[3] == 0);
    if (!can_mul || ggml_is_transposed(a)) {
        std::fprintf(stderr,
            "%s transposed=%d a=(%lld,%lld,%lld,%lld) b=(%lld,%lld,%lld,%lld)\n",
            label, ggml_is_transposed(a) ? 1 : 0,
            (long long)a->ne[0], (long long)a->ne[1], (long long)a->ne[2], (long long)a->ne[3],
            (long long)b->ne[0], (long long)b->ne[1], (long long)b->ne[2], (long long)b->ne[3]);
        std::fflush(stderr);
    }
    return ggml_mul_mat(ctx, a, b);
}

static ggml_tensor * rms_norm_weighted(ggml_context * ctx, ggml_tensor * x,
                                       ggml_tensor * weight, float eps) {
    ggml_tensor * cur = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * w = weight;
    if (w->type != cur->type) {
        w = ggml_cast(ctx, w, cur->type);
    }
    w = repeat_checked(ctx, w, cur, "repeat:rms_norm");
    return ggml_mul(ctx, cur, w);
}

static ggml_tensor * repeat_interleave_heads(ggml_context * ctx, ggml_tensor * x,
                                              int32_t repeat_factor) {
    if (repeat_factor == 1) return x;
    ggml_tensor * xf = (x->type != GGML_TYPE_F32) ? ggml_cast(ctx, x, GGML_TYPE_F32) : x;
    ggml_tensor * x4 = ggml_reshape_4d(ctx, ggml_cont(ctx, xf),
                                        xf->ne[0], 1, xf->ne[1], xf->ne[2]);
    ggml_tensor * target = ggml_new_tensor_4d(ctx, GGML_TYPE_F32,
                                               xf->ne[0], repeat_factor, xf->ne[1], xf->ne[2]);
    ggml_tensor * repeated = ggml_repeat(ctx, x4, target);
    return ggml_reshape_3d(ctx, ggml_cont(ctx, repeated),
                           xf->ne[0], xf->ne[1] * repeat_factor, xf->ne[2]);
}

static ggml_tensor * last_token_view(ggml_context * ctx, ggml_tensor * x, int32_t n_tokens) {
    return ggml_view_2d(ctx, x, x->ne[0], 1, x->nb[1], (n_tokens - 1) * x->nb[1]);
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

SlowARModel::SlowARModel() {}

SlowARModel::~SlowARModel() {
    // Free persistent decode graph
    if (decode_.ctx) ggml_free(decode_.ctx);

    // Free persistent fast decode graphs
    for (auto & fds : fast_decode_states_) {
        if (fds.ctx)   ggml_free(fds.ctx);
        if (fds.allocr) ggml_gallocr_free(fds.allocr);
    }

    if (ctx_kv_)          ggml_free(ctx_kv_);
    if (kv_buf_)          ggml_backend_buffer_free(kv_buf_);
    if (weights_.ctx_w)   ggml_free(weights_.ctx_w);
    if (weights_.model_buf) ggml_backend_buffer_free(weights_.model_buf);
    if (allocr_)          ggml_gallocr_free(allocr_);
    if (backend_gpu_ && backend_gpu_ != backend_) ggml_backend_free(backend_gpu_);
    if (backend_cpu_ && backend_cpu_ != backend_) ggml_backend_free(backend_cpu_);
    if (backend_)         ggml_backend_free(backend_);
    if (emb_f16_.buf)     ggml_backend_buffer_free(emb_f16_.buf);
    if (emb_f16_.ctx)     ggml_free(emb_f16_.ctx);
}

bool SlowARModel::backend_requires_single_token_semantic_prefill(ggml_backend_t gpu) {
    if (!gpu) return false;
    // CUDA does not support ggml_get_rows for quantized types, so we
    // process semantic prompt tokens one-by-one when running on CUDA.
#ifdef GGML_USE_CUDA
    return ggml_backend_is_cuda(gpu);
#else
    (void)gpu;
    return false;
#endif
}

// ---------------------------------------------------------------------------
// load()
// ---------------------------------------------------------------------------

bool SlowARModel::load(const std::string & gguf_path, int32_t gpu_device, int32_t backend_type) {
    // Always init CPU backend first
    backend_cpu_ = ggml_backend_cpu_init();
    if (!backend_cpu_) {
        std::cerr << "[Model] Failed to init CPU backend." << std::endl;
        return false;
    }
    // Keep CPU pool at 1 thread — GPU is primary; spinning workers waste cores.
    ggml_backend_cpu_set_n_threads(backend_cpu_, 1);

    if (gpu_device >= 0) {
#ifdef GGML_USE_VULKAN
        if (!backend_gpu_ && backend_type == 0) {
            backend_gpu_ = ggml_backend_vk_init(static_cast<size_t>(gpu_device));
            if (!backend_gpu_) {
                if(!SuppressNonEssentialVerbosity) { std::cerr << "[Model] Vulkan init failed, falling back to CPU." << std::endl; }
            }
        }
#endif
#ifdef GGML_USE_CUDA
        if (!backend_gpu_ && backend_type == 1) {
            backend_gpu_ = ggml_backend_cuda_init(static_cast<size_t>(gpu_device));
            if (!backend_gpu_) {
                if(!SuppressNonEssentialVerbosity) { std::cerr << "[Model] Cuda init failed, falling back to CPU." << std::endl; }
            }
        }
#endif

#ifdef GGML_USE_METAL
        if (!backend_gpu_ && backend_type == 2) {
            backend_gpu_ = ggml_backend_metal_init();
            if (!backend_gpu_) {
                if (!SuppressNonEssentialVerbosity) {
                    std::cerr << "[Model] Metal init failed, falling back to CPU." << std::endl;
                }
            }
        }
#endif
        if (!backend_gpu_)
        {
            if (!SuppressNonEssentialVerbosity) { std::cerr << "[Model] NPU not compiled, falling back to CPU." << std::endl; }
        }
    }

    // Tentatively use GPU; we'll refine after inspecting tensor types below.
    backend_ = backend_gpu_ ? backend_gpu_ : backend_cpu_;

    struct gguf_init_params params = { /*no_alloc=*/true, /*ctx=*/&weights_.ctx_w };
    gguf_context * ctx_gguf = gguf_init_from_file(gguf_path.c_str(), params);
    if (!ctx_gguf) {
        std::cerr << "[Model] Failed to load GGUF from " << gguf_path << std::endl;
        return false;
    }

    if(!SuppressNonEssentialVerbosity) { std::cout << "[Model] Reading metadata from " << gguf_path << std::endl; }

    // Helpers to read GGUF metadata
    auto get_u32 = [&](const char * key, uint32_t def) -> uint32_t {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { if(!SuppressNonEssentialVerbosity) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; } return def; }
        uint32_t v = gguf_get_val_u32(ctx_gguf, id);
        if(!SuppressNonEssentialVerbosity) { std::cout << "[GGUF] " << key << " = " << v << "\n"; }
        return v;
    };
    auto get_f32 = [&](const char * key, float def) -> float {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { if(!SuppressNonEssentialVerbosity) { std::cerr << "[GGUF] missing key: " << key << " (using default " << def << ")\n"; } return def; }
        float v = gguf_get_val_f32(ctx_gguf, id);
        if(!SuppressNonEssentialVerbosity) { std::cout << "[GGUF] " << key << " = " << v << "\n"; }
        return v;
    };
    auto get_bool = [&](const char * key, bool def) -> bool {
        int id = gguf_find_key(ctx_gguf, key);
        if (id < 0) { if(!SuppressNonEssentialVerbosity) { std::cerr << "[GGUF] missing key: " << key << " (using default " << (def?"true":"false") << ")\n"; } return def; }
        bool v = gguf_get_val_bool(ctx_gguf, id);
        if(!SuppressNonEssentialVerbosity) { std::cout << "[GGUF] " << key << " = " << (v?"true":"false") << "\n"; }
        return v;
    };

    hparams_ = ModelHParams();

    // Determine architecture prefix from the file
    std::string arch_prefix = "fish-speech.";
    {
        int arch_id = gguf_find_key(ctx_gguf, "general.architecture");
        if (arch_id >= 0) {
            std::string arch = gguf_get_val_str(ctx_gguf, arch_id);
            arch_prefix = arch + ".";
            hparams_.has_fast_decoder = (arch == "fish-speech");
            if(!SuppressNonEssentialVerbosity) { std::cout << "[Model] Architecture: " << arch << std::endl; }
        }
    }

    // Main model hparams (from arch-prefixed keys)
    hparams_.context_length      = (int32_t)get_u32((arch_prefix + "context_length").c_str(), 32768);
    hparams_.vocab_size          = (int32_t)get_u32((arch_prefix + "vocab_size").c_str(), 155776);
    hparams_.embedding_length    = (int32_t)get_u32((arch_prefix + "embedding_length").c_str(), 2560);
    hparams_.feed_forward_length = (int32_t)get_u32((arch_prefix + "feed_forward_length").c_str(), 9728);
    hparams_.block_count         = (int32_t)get_u32((arch_prefix + "block_count").c_str(), 36);
    hparams_.head_count          = (int32_t)get_u32((arch_prefix + "attention.head_count").c_str(), 32);
    hparams_.head_count_kv       = (int32_t)get_u32((arch_prefix + "attention.head_count_kv").c_str(), 8);
    hparams_.rope_freq_base      = get_f32((arch_prefix + "rope.freq_base").c_str(), 1e6f);
    hparams_.rms_norm_eps        = get_f32((arch_prefix + "attention.layer_norm_rms_epsilon").c_str(), 1e-6f);

    // Fish-speech specific keys
    hparams_.codebook_size            = (int32_t)get_u32("fish_speech.codebook_size", 4096);
    hparams_.num_codebooks            = (int32_t)get_u32("fish_speech.num_codebooks", 10);
    hparams_.semantic_begin_id        = (int32_t)get_u32("fish_speech.semantic_begin_id", 151678);
    hparams_.semantic_end_id          = (int32_t)get_u32("fish_speech.semantic_end_id", 155773);
    hparams_.tie_word_embeddings      = get_bool("fish_speech.tie_word_embeddings", true);
    hparams_.attention_qk_norm        = get_bool("fish_speech.attention_qk_norm", false);
    hparams_.scale_codebook_embeddings = get_bool("fish_speech.scale_codebook_embeddings", false);

    // Fast decoder hparams
    if (hparams_.has_fast_decoder) {
        hparams_.fast_context_length   = (int32_t)get_u32("fish_speech.fast_context_length", 11);
        hparams_.fast_embedding_length = (int32_t)get_u32("fish_speech.fast_embedding_length", 2560);
        hparams_.fast_feed_forward_length = (int32_t)get_u32("fish_speech.fast_feed_forward_length", 9728);
        hparams_.fast_block_count      = (int32_t)get_u32("fish_speech.fast_block_count", 4);
        hparams_.fast_head_count       = (int32_t)get_u32("fish_speech.fast_head_count", 32);
        hparams_.fast_head_count_kv    = (int32_t)get_u32("fish_speech.fast_head_count_kv", 8);
        hparams_.fast_head_dim         = (int32_t)get_u32("fish_speech.fast_head_dim", 128);
        hparams_.fast_rope_freq_base   = get_f32("fish_speech.fast_rope_freq_base", 1e6f);
        hparams_.fast_rms_norm_eps     = get_f32("fish_speech.fast_layer_norm_rms_eps", 1e-6f);
        hparams_.fast_attention_qk_norm = get_bool("fish_speech.fast_attention_qk_norm", false);
        hparams_.fast_has_project_in   = get_bool("fish_speech.fast_project_in", false);
    }

    if(!SuppressNonEssentialVerbosity) {
    std::cout << "[Model] Layers: " << hparams_.block_count
              << ", Dim: " << hparams_.embedding_length
              << ", Vocab: " << hparams_.vocab_size
              << ", head_count: " << hparams_.head_count
              << ", has_fast_decoder: " << hparams_.has_fast_decoder << std::endl;
    }

    // ---------------------------------------------------------------------------
    // Load tensor pointers (metadata only — data loaded below)
    // ---------------------------------------------------------------------------
    auto req_t = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, name.c_str());
        if (!t) {
            throw std::runtime_error("missing tensor: " + name);
        }
        return t;
    };
    auto opt_t = [&](const std::string & name) -> ggml_tensor * {
        return ggml_get_tensor(weights_.ctx_w, name.c_str());
    };

    try {
        weights_.embeddings          = req_t("embeddings.weight");
        weights_.codebook_embeddings = req_t("codebook_embeddings.weight");
        weights_.norm                = req_t("norm.weight");

        weights_.layers.resize(hparams_.block_count);
        for (int32_t i = 0; i < hparams_.block_count; ++i) {
            auto & layer = weights_.layers[i];
            std::string stem = "layers." + std::to_string(i) + ".";

            layer.attention_norm = req_t(stem + "attention_norm.weight");
            layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
            layer.wqkv           = req_t(stem + "attention.wqkv.weight");
            layer.wo             = req_t(stem + "attention.wo.weight");
            layer.w1             = req_t(stem + "feed_forward.w1.weight");
            layer.w2             = req_t(stem + "feed_forward.w2.weight");
            layer.w3             = req_t(stem + "feed_forward.w3.weight");

            if (hparams_.attention_qk_norm) {
                layer.q_norm = req_t(stem + "attention.q_norm.weight");
                layer.k_norm = req_t(stem + "attention.k_norm.weight");
            }
        }

        if (hparams_.has_fast_decoder) {
            if (hparams_.fast_has_project_in) {
                weights_.fast_project_in = req_t("fast_project_in.weight");
            }
            weights_.fast_embeddings = req_t("fast_embeddings.weight");
            weights_.fast_norm       = req_t("fast_norm.weight");
            weights_.fast_output     = req_t("fast_output.weight");

            weights_.fast_layers.resize(hparams_.fast_block_count);
            for (int32_t i = 0; i < hparams_.fast_block_count; ++i) {
                auto & layer = weights_.fast_layers[i];
                std::string stem = "fast_layers." + std::to_string(i) + ".";

                layer.attention_norm = req_t(stem + "attention_norm.weight");
                layer.ffn_norm       = req_t(stem + "ffn_norm.weight");
                layer.wqkv           = req_t(stem + "attention.wqkv.weight");
                layer.wo             = req_t(stem + "attention.wo.weight");
                layer.w1             = req_t(stem + "feed_forward.w1.weight");
                layer.w2             = req_t(stem + "feed_forward.w2.weight");
                layer.w3             = req_t(stem + "feed_forward.w3.weight");

                if (hparams_.fast_attention_qk_norm) {
                    layer.q_norm = req_t(stem + "attention.q_norm.weight");
                    layer.k_norm = req_t(stem + "attention.k_norm.weight");
                }
            }
        }
    } catch (const std::exception & e) {
        std::cerr << "[Model] " << e.what() << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }

    // -----------------------------------------------------------------------
    // Weight allocation — decide GPU vs CPU based on actual tensor types
    // -----------------------------------------------------------------------
    bool use_gpu_for_weights = !!backend_gpu_;
    bool need_f16_emb = false;
    if (backend_gpu_) {
#ifdef GGML_USE_CUDA
        if (ggml_backend_is_cuda(backend_gpu_)) {
            auto cuda_supports_get_rows = [](ggml_type t) -> bool {
                switch (t) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        return true;
                    default:
                        return false;
                }
            };
            ggml_type emb_type = weights_.embeddings->type;
            if (!cuda_supports_get_rows(emb_type)) {
                need_f16_emb = true;
                if (!SuppressNonEssentialVerbosity) {
                    std::cerr << "[Model] CUDA get_rows unsupported for type "
                              << ggml_type_name(emb_type)
                              << " — dequantizing embeddings to F16 for GPU offload." << std::endl;
                }
            }
        }
#endif
    }

    backend_ = (use_gpu_for_weights) ? backend_gpu_ : backend_cpu_;
    n_gpu_layers_ = use_gpu_for_weights ? hparams_.block_count : 0;

    if (!SuppressNonEssentialVerbosity) {
        std::cerr << "[Model] Using " << (use_gpu_for_weights ? "GPU" : "CPU") << " for weights." << std::endl;
    }

    weights_.model_buf = ggml_backend_alloc_ctx_tensors(weights_.ctx_w, backend_);
    if (!weights_.model_buf) {
        std::cerr << "[Model] Failed to allocate backend buffer for weights." << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }

    allocr_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));

    // Load tensor data from GGUF file
    const size_t data_offset = gguf_get_data_offset(ctx_gguf);
    const int64_t n_tensors  = gguf_get_n_tensors(ctx_gguf);
    std::FILE * f = std::fopen(gguf_path.c_str(), "rb");
    if (!f) {
        std::cerr << "[Model] Cannot reopen " << gguf_path << " for data loading." << std::endl;
        gguf_free(ctx_gguf);
        return false;
    }
    std::vector<uint8_t> tmp;
    for (int64_t ti = 0; ti < n_tensors; ++ti) {
        const char * tname = gguf_get_tensor_name(ctx_gguf, ti);
        ggml_tensor * t = ggml_get_tensor(weights_.ctx_w, tname);
        if (!t) continue;
        const size_t toff  = data_offset + gguf_get_tensor_offset(ctx_gguf, ti);
        const size_t tsize = ggml_nbytes(t);
        if (tmp.size() < tsize) tmp.resize(tsize);
#ifdef _WIN32
        _fseeki64(f, (int64_t)toff, SEEK_SET);
#else
        fseeko(f, (off_t)toff, SEEK_SET);
#endif
        if (std::fread(tmp.data(), 1, tsize, f) != tsize) {
            std::cerr << "[Model] Failed to read tensor: " << tname << std::endl;
            std::fclose(f);
            gguf_free(ctx_gguf);
            return false;
        }
        ggml_backend_tensor_set(t, tmp.data(), 0, tsize);
    }
    tmp.clear();
    tmp.shrink_to_fit();
    std::fclose(f);

    // -------------------------------------------------------------------
    // For CUDA + K-quant: dequantize embedding tensors to F16 on GPU
    // -------------------------------------------------------------------
    if (need_f16_emb) {
        size_t ctx_size = ggml_tensor_overhead() * 4 + 4096;
        ggml_init_params ep = { ctx_size, nullptr, true };
        emb_f16_.ctx = ggml_init(ep);

        emb_f16_.embeddings = ggml_new_tensor_2d(emb_f16_.ctx, GGML_TYPE_F16,
            weights_.embeddings->ne[0], weights_.embeddings->ne[1]);
        emb_f16_.codebook_embeddings = ggml_new_tensor_2d(emb_f16_.ctx, GGML_TYPE_F16,
            weights_.codebook_embeddings->ne[0], weights_.codebook_embeddings->ne[1]);
        if (weights_.fast_embeddings) {
            emb_f16_.fast_embeddings = ggml_new_tensor_2d(emb_f16_.ctx, GGML_TYPE_F16,
                weights_.fast_embeddings->ne[0], weights_.fast_embeddings->ne[1]);
        }

        emb_f16_.buf = ggml_backend_alloc_ctx_tensors(emb_f16_.ctx, backend_gpu_);
        if (!emb_f16_.buf) {
            std::cerr << "[Model] Warning: failed to alloc F16 embeddings on GPU." << std::endl;
        } else {
            auto dequant_set = [&](ggml_tensor * src, ggml_tensor * dst) {
                if (!src || !dst) return;
                const int64_t ncols = src->ne[0];
                const int64_t nrows = ggml_nrows(src);
                const int blck  = ggml_blck_size(src->type);
                const int tsz   = ggml_type_size(src->type);
                const auto * tr = ggml_get_type_traits(src->type);

                std::vector<uint8_t> src_data(ggml_nbytes(src));
                ggml_backend_tensor_get(src, src_data.data(), 0, src_data.size());

                std::vector<float>       row32(ncols);
                std::vector<ggml_fp16_t> row16(ncols);

                for (int64_t r = 0; r < nrows; ++r) {
                    const uint8_t * rp = src_data.data() + r * src->nb[1];
                    for (int64_t c = 0; c < ncols; c += blck) {
                        tr->to_float(rp + (c / blck) * tsz, row32.data() + c, blck);
                    }
                    for (int64_t i = 0; i < ncols; ++i)
                        row16[i] = ggml_fp32_to_fp16(row32[i]);
                    ggml_backend_tensor_set(dst, row16.data(),
                        (size_t)r * dst->nb[1], (size_t)ncols * sizeof(ggml_fp16_t));
                }
            };

            dequant_set(weights_.embeddings, emb_f16_.embeddings);
            dequant_set(weights_.codebook_embeddings, emb_f16_.codebook_embeddings);
            dequant_set(weights_.fast_embeddings, emb_f16_.fast_embeddings);

            if (!SuppressNonEssentialVerbosity) {
                std::cerr << "[Model] F16 embeddings created on GPU for K-quant model." << std::endl;
            }
        }
    }

#ifdef __linux__
    {
        int fd = ::open(gguf_path.c_str(), O_RDONLY);
        if (fd >= 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
            ::close(fd);
        }
    }
#endif

    if(!SuppressNonEssentialVerbosity) { std::cout << "[Model] Weights loaded. Total tensors: " << n_tensors << std::endl; }

    gguf_free(ctx_gguf);
    return true;
}

// ---------------------------------------------------------------------------
// init_kv_cache()
// ---------------------------------------------------------------------------

bool SlowARModel::init_kv_cache(int32_t max_seq_len) {
    max_seq_len_ = max_seq_len;
    n_past_      = 0;

    const int32_t dim = hparams_.embedding_length;
    if (dim == 0) return true;

    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim_ = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim_ = hparams_.embedding_length / hparams_.head_count;
    }

    const int32_t n_head_kv = hparams_.head_count_kv;
    const int32_t n_layer   = hparams_.block_count;

    // ctx_kv_ holds: memory_k_, memory_v_, + 2*n_layer staging tensors.
    const size_t ctx_kv_size = (2ull + 2ull * (size_t)n_layer) * ggml_tensor_overhead() + (1ull << 20);
    ggml_init_params p = {
        /*.mem_size =*/ ctx_kv_size,
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc =*/ true,
    };
    ctx_kv_ = ggml_init(p);
    if (!ctx_kv_) {
        std::cerr << "[Model] Failed to init KV context." << std::endl;
        return false;
    }

    memory_k_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim_, n_head_kv, max_seq_len, n_layer);
    memory_v_ = ggml_new_tensor_4d(ctx_kv_, GGML_TYPE_F16, head_dim_, n_head_kv, max_seq_len, n_layer);

    // One-slot staging buffers per layer for the persistent decode graph.
    k_cur_stage_.resize(n_layer);
    v_cur_stage_.resize(n_layer);
    for (int32_t il = 0; il < n_layer; ++il) {
        k_cur_stage_[il] = ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F16, head_dim_, n_head_kv, 1);
        v_cur_stage_[il] = ggml_new_tensor_3d(ctx_kv_, GGML_TYPE_F16, head_dim_, n_head_kv, 1);
    }

    kv_buf_ = ggml_backend_alloc_ctx_tensors(ctx_kv_, backend_);
    if (!kv_buf_) {
        std::cerr << "[Model] Failed to allocate KV cache buffer." << std::endl;
        return false;
    }

    ggml_backend_tensor_memset(memory_k_, 0, 0, ggml_nbytes(memory_k_));
    ggml_backend_tensor_memset(memory_v_, 0, 0, ggml_nbytes(memory_v_));

    // Build the persistent single-token decode graph now that KV cache is ready.
    if (!build_decode_graph()) {
        std::cerr << "[Model] Failed to build persistent decode graph." << std::endl;
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// reset() / clear_kv_cache()
// ---------------------------------------------------------------------------

void SlowARModel::reset() {
    n_past_ = 0;
}

void SlowARModel::clear_kv_cache() {
    // Free persistent decode graph — it holds views into memory_k_/memory_v_
    // which are about to be freed.
    if (decode_.ctx) {
        ggml_free(decode_.ctx);
        decode_.ctx = nullptr;
        decode_.graph = nullptr;
        decode_.semantic_ids = nullptr;
        decode_.positions    = nullptr;
        decode_.semantic_mask = nullptr;
        decode_.token_scale  = nullptr;
        decode_.cb_id_tensors.clear();
        decode_.kq_mask      = nullptr;
        decode_.k_write_views.clear();
        decode_.v_write_views.clear();
        decode_.hidden_last  = nullptr;
        decode_.logits       = nullptr;
    }

    if (kv_buf_) {
        ggml_backend_buffer_free(kv_buf_);
        kv_buf_ = nullptr;
    }
    if (ctx_kv_) {
        ggml_free(ctx_kv_);
        ctx_kv_ = nullptr;
    }

    memory_k_ = nullptr;
    memory_v_ = nullptr;
    k_cur_stage_.clear();
    v_cur_stage_.clear();

    max_seq_len_ = 0;
    n_past_ = 0;
}

// ---------------------------------------------------------------------------
// build_decode_graph() — called once per init_kv_cache
//
// Builds a persistent single-token decode graph with FIXED tensor shapes so
// that GGML can capture and replay a CUDA graph.  Key design choices:
//
//  • kq_mask is (max_seq_len_+1, 1) — never changes shape; content updated
//    per step to mask out unwritten KV slots.
//  • k/v past views span ALL max_seq_len_ slots; invalid future slots are
//    masked to -inf in kq_mask.
//  • Current K/V written to fixed staging buffers (k_cur_stage_[il]) via
//    ggml_cpy inside the graph; post-graph async D2D copies to KV slot.
//  • No gallocr re-plan per step — graph allocated once, reused forever.
// ---------------------------------------------------------------------------

bool SlowARModel::build_decode_graph() {
    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;
    const int32_t q_size    = n_head * head_dim_;
    const int32_t kv_size   = n_head_kv * head_dim_;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim_));
    const float sem_scale   = 1.0f / std::sqrt(static_cast<float>(hparams_.num_codebooks + 1));
    (void)kv_size;
    (void)sem_scale;

    const size_t ctx_sz = 12ull * 1024ull * 1024ull;
    ggml_init_params p = { ctx_sz, nullptr, /*no_alloc=*/true };
    decode_.ctx = ggml_init(p);
    if (!decode_.ctx) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(decode_.ctx, 32768, false);

    // Fixed-shape input tensors (content set per step via tensor_set)
    decode_.semantic_ids  = ggml_new_tensor_1d(decode_.ctx, GGML_TYPE_I32, 1);
    decode_.positions     = ggml_new_tensor_1d(decode_.ctx, GGML_TYPE_I32, 1);
    decode_.semantic_mask = ggml_new_tensor_2d(decode_.ctx, GGML_TYPE_F32, 1, 1);
    if (hparams_.scale_codebook_embeddings) {
        decode_.token_scale = ggml_new_tensor_2d(decode_.ctx, GGML_TYPE_F32, 1, 1);
    }

    decode_.cb_id_tensors.resize(hparams_.num_codebooks);
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        decode_.cb_id_tensors[cb] = ggml_new_tensor_1d(decode_.ctx, GGML_TYPE_I32, 1);
    }

    // kq_mask: (max_seq_len_+1, 1) — FIXED shape, content updated per step.
    decode_.kq_mask = ggml_new_tensor_2d(decode_.ctx, GGML_TYPE_F32, (int64_t)max_seq_len_ + 1, 1);

    // Embedding lookup
    ggml_tensor * emb_get = emb_f16_.embeddings ? emb_f16_.embeddings : weights_.embeddings;
    ggml_tensor * x = ggml_get_rows(decode_.ctx, emb_get, decode_.semantic_ids);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(decode_.ctx, x, GGML_TYPE_F32);

    ggml_tensor * cb_emb = emb_f16_.codebook_embeddings
                         ? emb_f16_.codebook_embeddings
                         : weights_.codebook_embeddings;
    ggml_tensor * codebook_sum = nullptr;
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_tensor * emb = ggml_get_rows(decode_.ctx, cb_emb, decode_.cb_id_tensors[cb]);
        if (emb->type != GGML_TYPE_F32) emb = ggml_cast(decode_.ctx, emb, GGML_TYPE_F32);
        codebook_sum = (codebook_sum == nullptr) ? emb : ggml_add(decode_.ctx, codebook_sum, emb);
    }
    if (codebook_sum != nullptr) {
        codebook_sum = ggml_mul(decode_.ctx, codebook_sum,
                                ggml_repeat(decode_.ctx, decode_.semantic_mask, codebook_sum));
        x = ggml_add(decode_.ctx, x, codebook_sum);
    }
    if (decode_.token_scale != nullptr) {
        x = ggml_mul(decode_.ctx, x, ggml_repeat(decode_.ctx, decode_.token_scale, x));
    }

    decode_.k_write_views.resize(hparams_.block_count);
    decode_.v_write_views.resize(hparams_.block_count);

    for (int32_t il = 0; il < hparams_.block_count; ++il) {
        const auto & layer = weights_.layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(decode_.ctx, x, layer.attention_norm, hparams_.rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(decode_.ctx, layer.wqkv, attn_in, "mul_mat:wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(decode_.ctx, qkv, q_size,  1, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(decode_.ctx, qkv, n_head_kv * head_dim_, 1, qkv->nb[1], (size_t)q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(decode_.ctx, qkv, n_head_kv * head_dim_, 1, qkv->nb[1], (size_t)(q_size + n_head_kv * head_dim_) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(decode_.ctx, ggml_cont(decode_.ctx, q2d), head_dim_, n_head,    1);
        ggml_tensor * k = ggml_reshape_3d(decode_.ctx, ggml_cont(decode_.ctx, k2d), head_dim_, n_head_kv, 1);
        ggml_tensor * v = ggml_reshape_3d(decode_.ctx, ggml_cont(decode_.ctx, v2d), head_dim_, n_head_kv, 1);

        if (hparams_.attention_qk_norm) {
            q = rms_norm_weighted(decode_.ctx, q, layer.q_norm, hparams_.rms_norm_eps);
            k = rms_norm_weighted(decode_.ctx, k, layer.k_norm, hparams_.rms_norm_eps);
        }

        q = ggml_rope_ext(decode_.ctx, q, decode_.positions, nullptr, head_dim_, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(decode_.ctx, k, decode_.positions, nullptr, head_dim_, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        // Write current K/V to fixed staging buffers (FIXED destination pointer).
        // ggml_cpy returns a view of k_cur_stage_[il] — used as concat input to
        // ensure the copy executes before the concat reads from staging.
        ggml_tensor * k_cpy = ggml_cpy(decode_.ctx, k, k_cur_stage_[il]);
        ggml_tensor * v_cpy = ggml_cpy(decode_.ctx, v, v_cur_stage_[il]);
        decode_.k_write_views[il] = k_cpy;
        decode_.v_write_views[il] = v_cpy;

        // Past K/V: read ALL max_seq_len_ slots (FIXED shape).
        // Unwritten slots are masked to -inf via kq_mask.
        const size_t layer_off_k = static_cast<size_t>(il) * memory_k_->nb[3];
        const size_t layer_off_v = static_cast<size_t>(il) * memory_v_->nb[3];
        ggml_tensor * k_past = ggml_view_3d(decode_.ctx, memory_k_,
            head_dim_, n_head_kv, max_seq_len_,
            memory_k_->nb[1], memory_k_->nb[2], layer_off_k);
        ggml_tensor * v_past = ggml_view_3d(decode_.ctx, memory_v_,
            head_dim_, n_head_kv, max_seq_len_,
            memory_v_->nb[1], memory_v_->nb[2], layer_off_v);

        // Concat past (max_seq_len_) with current (1) → max_seq_len_+1 total.
        // k_cpy is a view of k_cur_stage_, guaranteeing cpy executes before concat.
        ggml_tensor * k_all = ggml_concat(decode_.ctx, k_past, k_cpy, 2);
        ggml_tensor * v_all = ggml_concat(decode_.ctx, v_past, v_cpy, 2);

        ggml_tensor * k_rep = repeat_interleave_heads(decode_.ctx, k_all, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(decode_.ctx, v_all, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(decode_.ctx, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(decode_.ctx, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(decode_.ctx, K, Q, "mul_mat:kq");
        ggml_tensor * KQf = ggml_soft_max_ext(decode_.ctx, KQ, decode_.kq_mask, attn_scale, 0.0f);

        ggml_tensor * V       = ggml_cont(decode_.ctx, ggml_permute(decode_.ctx, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(decode_.ctx, V, KQf, "mul_mat:kqv");
        ggml_tensor * KQVm    = ggml_permute(decode_.ctx, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(decode_.ctx, KQVm,
                                          ggml_new_tensor_2d(decode_.ctx, GGML_TYPE_F32, q_size, 1));
        ggml_tensor * attn_out = mul_mat_checked(decode_.ctx, layer.wo, attn_cur, "mul_mat:wo");

        ggml_tensor * h     = ggml_add(decode_.ctx, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(decode_.ctx, h, layer.ffn_norm, hparams_.rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(decode_.ctx, layer.w1, ff_in, "mul_mat:w1");
        ggml_tensor * up    = mul_mat_checked(decode_.ctx, layer.w3, ff_in, "mul_mat:w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(decode_.ctx, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(decode_.ctx, layer.w2, ff_h, "mul_mat:w2");

        x = ggml_add(decode_.ctx, h, ff_out);
    }

    ggml_tensor * slow_out  = rms_norm_weighted(decode_.ctx, x, weights_.norm, hparams_.rms_norm_eps);
    ggml_tensor * slow_cont = ggml_cont(decode_.ctx, slow_out);
    decode_.hidden_last = ggml_cpy(decode_.ctx,
        last_token_view(decode_.ctx, slow_cont, 1),
        ggml_new_tensor_2d(decode_.ctx, GGML_TYPE_F32, dim, 1));
    decode_.logits = mul_mat_checked(decode_.ctx, weights_.embeddings, decode_.hidden_last, "mul_mat:logits");
    ggml_build_forward_expand(gf, decode_.logits);
    decode_.graph = gf;

    if (!ggml_gallocr_alloc_graph(allocr_, gf)) {
        std::fprintf(stderr, "[build_decode_graph] gallocr alloc failed\n");
        ggml_free(decode_.ctx);
        decode_.ctx  = nullptr;
        decode_.graph = nullptr;
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// prefill() / step()
// ---------------------------------------------------------------------------

bool SlowARModel::prefill(const std::vector<int32_t> & flat_tokens, int32_t n_tokens,
                          int32_t n_threads, StepResult & result) {
    // Use a temporary gallocr for prefill so the large compute buffer
    // (sized for n_tokens) is freed immediately after, leaving allocr_
    // fresh for decode steps that start small (n_kv=1) and grow gradually.
    ggml_gallocr_t prefill_allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!prefill_allocr) return false;
    std::swap(allocr_, prefill_allocr);
    bool ok = eval_cached(flat_tokens, n_tokens, n_threads, result);
    std::swap(allocr_, prefill_allocr);
    ggml_gallocr_free(prefill_allocr);
    return ok;
}

bool SlowARModel::step(const std::vector<int32_t> & flat_tokens, int32_t n_threads,
                       StepResult & result) {
    // Fall back to eval_cached if persistent graph wasn't built.
    if (!decode_.ctx || !decode_.graph) {
        return eval_cached(flat_tokens, 1, n_threads, result);
    }

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    if (static_cast<int32_t>(flat_tokens.size()) != codebook_dim) {
        std::fprintf(stderr, "[step] expected %d ints, got %zu\n", codebook_dim, flat_tokens.size());
        return false;
    }
    if (n_past_ >= max_seq_len_) {
        std::fprintf(stderr, "[step] KV cache full (%d >= %d)\n", n_past_, max_seq_len_);
        return false;
    }

    const int32_t semantic = flat_tokens[0];
    const bool is_semantic = (semantic >= hparams_.semantic_begin_id &&
                               semantic <= hparams_.semantic_end_id);
    const float sem_scale  = 1.0f / std::sqrt(static_cast<float>(codebook_dim));

    // Fill kq_mask: [0..n_past_-1]=0 (valid past), [n_past_..max_seq_len_-1]=-inf,
    // [max_seq_len_]=0 (current token appended last via concat).
    const int32_t mask_len = max_seq_len_ + 1;
    std::vector<float> kq_data(mask_len, -std::numeric_limits<float>::infinity());
    for (int32_t i = 0; i < n_past_; ++i) kq_data[i] = 0.0f;
    kq_data[max_seq_len_] = 0.0f;
    ggml_backend_tensor_set(decode_.kq_mask, kq_data.data(), 0, (size_t)mask_len * sizeof(float));

    // Input ids and position
    int32_t pos = n_past_;
    ggml_backend_tensor_set(decode_.semantic_ids,  &semantic, 0, sizeof(int32_t));
    ggml_backend_tensor_set(decode_.positions,     &pos,      0, sizeof(int32_t));
    float sem_f = is_semantic ? 1.0f : 0.0f;
    ggml_backend_tensor_set(decode_.semantic_mask, &sem_f,    0, sizeof(float));
    if (decode_.token_scale) {
        float ts = is_semantic ? sem_scale : 1.0f;
        ggml_backend_tensor_set(decode_.token_scale, &ts, 0, sizeof(float));
    }
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        int32_t cb_id = 0;
        if (is_semantic) {
            cb_id = flat_tokens[cb + 1] + cb * hparams_.codebook_size;
        }
        ggml_backend_tensor_set(decode_.cb_id_tensors[cb], &cb_id, 0, sizeof(int32_t));
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, decode_.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[step] compute failed\n");
        return false;
    }

    // Post-graph: async D2D copy staging → correct KV slot for this step.
    // Copies are issued on the compute stream (no per-copy sync); the next
    // graph_compute on the same stream will see the updated KV cache.
    if (backend_gpu_) {
        const size_t kv_copy_ctx_size = 2ull * (size_t)hparams_.block_count * ggml_tensor_overhead() + 4096;
        ggml_init_params pc = { kv_copy_ctx_size, nullptr, /*no_alloc=*/true };
        ggml_context * ctx_copy = ggml_init(pc);
        if (ctx_copy) {
            for (int32_t il = 0; il < hparams_.block_count; ++il) {
                const size_t layer_off_k = (size_t)il * memory_k_->nb[3];
                const size_t slot_off_k  = (size_t)n_past_ * memory_k_->nb[2];
                ggml_tensor * k_dst = ggml_view_3d(ctx_copy, memory_k_,
                    head_dim_, hparams_.head_count_kv, 1,
                    memory_k_->nb[1], memory_k_->nb[2], layer_off_k + slot_off_k);
                k_dst->buffer = memory_k_->buffer;
                ggml_backend_tensor_copy_async(backend_gpu_, backend_gpu_, k_cur_stage_[il], k_dst);

                const size_t layer_off_v = (size_t)il * memory_v_->nb[3];
                const size_t slot_off_v  = (size_t)n_past_ * memory_v_->nb[2];
                ggml_tensor * v_dst = ggml_view_3d(ctx_copy, memory_v_,
                    head_dim_, hparams_.head_count_kv, 1,
                    memory_v_->nb[1], memory_v_->nb[2], layer_off_v + slot_off_v);
                v_dst->buffer = memory_v_->buffer;
                ggml_backend_tensor_copy_async(backend_gpu_, backend_gpu_, v_cur_stage_[il], v_dst);
            }
            ggml_free(ctx_copy);
        }
    } else {
        // CPU: blocking copy (tiny tensors, negligible cost)
        const size_t kv_copy_ctx_size = 2ull * (size_t)hparams_.block_count * ggml_tensor_overhead() + 4096;
        ggml_init_params pc = { kv_copy_ctx_size, nullptr, /*no_alloc=*/true };
        ggml_context * ctx_copy = ggml_init(pc);
        if (ctx_copy) {
            for (int32_t il = 0; il < hparams_.block_count; ++il) {
                const size_t layer_off_k = (size_t)il * memory_k_->nb[3];
                const size_t slot_off_k  = (size_t)n_past_ * memory_k_->nb[2];
                ggml_tensor * k_dst = ggml_view_3d(ctx_copy, memory_k_,
                    head_dim_, hparams_.head_count_kv, 1,
                    memory_k_->nb[1], memory_k_->nb[2], layer_off_k + slot_off_k);
                ggml_backend_tensor_copy(k_cur_stage_[il], k_dst);

                const size_t layer_off_v = (size_t)il * memory_v_->nb[3];
                const size_t slot_off_v  = (size_t)n_past_ * memory_v_->nb[2];
                ggml_tensor * v_dst = ggml_view_3d(ctx_copy, memory_v_,
                    head_dim_, hparams_.head_count_kv, 1,
                    memory_v_->nb[1], memory_v_->nb[2], layer_off_v + slot_off_v);
                ggml_backend_tensor_copy(v_cur_stage_[il], v_dst);
            }
            ggml_free(ctx_copy);
        }
    }

    // Read hidden state and (partial) logits from GPU.
    result.hidden.resize(hparams_.embedding_length);
    ggml_backend_tensor_get(decode_.hidden_last, result.hidden.data(), 0,
                            (size_t)hparams_.embedding_length * sizeof(float));

    result.logits.assign(hparams_.vocab_size, -std::numeric_limits<float>::infinity());
    if (logits_range_end_ > logits_range_begin_) {
        const int32_t n_read = logits_range_end_ - logits_range_begin_;
        ggml_backend_tensor_get(decode_.logits,
                                result.logits.data() + logits_range_begin_,
                                (size_t)logits_range_begin_ * sizeof(float),
                                (size_t)n_read * sizeof(float));
    } else {
        ggml_backend_tensor_get(decode_.logits, result.logits.data(), 0,
                                (size_t)hparams_.vocab_size * sizeof(float));
    }

    n_past_++;
    return true;
}

// ---------------------------------------------------------------------------
// eval_cached() — prefill / multi-token path (stateless per call)
// ---------------------------------------------------------------------------

bool SlowARModel::eval_cached(const std::vector<int32_t> & flat_tokens,
                               int32_t n_tokens, int32_t n_threads,
                               StepResult & result) {
    if (n_tokens <= 0) return false;

    const int32_t codebook_dim = hparams_.num_codebooks + 1;
    if (static_cast<int32_t>(flat_tokens.size()) != n_tokens * codebook_dim) {
        std::fprintf(stderr, "[eval_cached] expected %d ints for %d tokens, got %zu\n",
            n_tokens * codebook_dim, n_tokens, flat_tokens.size());
        return false;
    }
    if (n_past_ + n_tokens > max_seq_len_) {
        std::fprintf(stderr, "[eval_cached] KV cache overflow (%d + %d > %d)\n",
            n_past_, n_tokens, max_seq_len_);
        return false;
    }

    const int32_t dim       = hparams_.embedding_length;
    const int32_t n_head    = hparams_.head_count;
    const int32_t n_head_kv = hparams_.head_count_kv;

    int32_t head_dim = 0;
    if (hparams_.attention_qk_norm && !weights_.layers.empty() && weights_.layers[0].q_norm) {
        head_dim = static_cast<int32_t>(weights_.layers[0].q_norm->ne[0]);
    } else {
        head_dim = static_cast<int32_t>(weights_.layers[0].wo->ne[0] / n_head);
    }

    const int32_t q_size   = n_head * head_dim;
    const int32_t kv_size  = n_head_kv * head_dim;
    const float attn_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const float sem_scale  = 1.0f / std::sqrt(static_cast<float>(codebook_dim));

    std::vector<int32_t> semantic_vals(n_tokens);
    std::vector<int32_t> pos_vals(n_tokens);
    std::vector<float>   semantic_mask_vals(n_tokens, 0.0f);
    std::vector<float>   token_scale_vals;
    std::vector<std::vector<int32_t>> cb_vals(hparams_.num_codebooks, std::vector<int32_t>(n_tokens, 0));

    if (hparams_.scale_codebook_embeddings) {
        token_scale_vals.resize(n_tokens);
    }

    for (int32_t t = 0; t < n_tokens; ++t) {
        const int32_t semantic = flat_tokens[t * codebook_dim];
        const bool is_semantic = (semantic >= hparams_.semantic_begin_id &&
                                  semantic <= hparams_.semantic_end_id);

        semantic_vals[t]      = semantic;
        pos_vals[t]           = n_past_ + t;
        semantic_mask_vals[t] = is_semantic ? 1.0f : 0.0f;

        if (!token_scale_vals.empty()) {
            token_scale_vals[t] = is_semantic ? sem_scale : 1.0f;
        }

        for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
            if (!is_semantic) continue;
            const int32_t v = flat_tokens[t * codebook_dim + cb + 1];
            cb_vals[cb][t] = v + cb * hparams_.codebook_size;
        }
    }

    static size_t ctx_size = 0;
    static std::vector<uint8_t> ctx_buf;
    if (ctx_size == 0) {
        ctx_size = 10u * 1024u * 1024u;
        ctx_buf.resize(ctx_size);
    }
    ggml_init_params p = { ctx_size, ctx_buf.data(), true };
    ggml_context * ctx0 = ggml_init(p);
    if (!ctx0) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(ctx0, 32768, false);

    ggml_tensor * semantic_ids   = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * positions      = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_tensor * semantic_mask  = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    ggml_tensor * token_scale    = nullptr;
    if (hparams_.scale_codebook_embeddings) {
        token_scale = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, 1, n_tokens);
    }

    ggml_tensor * emb_get = emb_f16_.embeddings ? emb_f16_.embeddings : weights_.embeddings;
    ggml_tensor * x = ggml_get_rows(ctx0, emb_get, semantic_ids);
    if (x->type != GGML_TYPE_F32) x = ggml_cast(ctx0, x, GGML_TYPE_F32);

    std::vector<ggml_tensor *> cb_id_tensors(hparams_.num_codebooks);
    ggml_tensor * codebook_sum = nullptr;
    ggml_tensor * cb_emb = emb_f16_.codebook_embeddings ? emb_f16_.codebook_embeddings : weights_.codebook_embeddings;
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_tensor * ids = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
        cb_id_tensors[cb] = ids;
        ggml_tensor * emb = ggml_get_rows(ctx0, cb_emb, ids);
        if (emb->type != GGML_TYPE_F32) emb = ggml_cast(ctx0, emb, GGML_TYPE_F32);
        codebook_sum = (codebook_sum == nullptr) ? emb : ggml_add(ctx0, codebook_sum, emb);
    }

    if (codebook_sum != nullptr) {
        codebook_sum = ggml_mul(ctx0, codebook_sum,
                                ggml_repeat(ctx0, semantic_mask, codebook_sum));
        x = ggml_add(ctx0, x, codebook_sum);
    }
    if (token_scale != nullptr) {
        x = ggml_mul(ctx0, x, ggml_repeat(ctx0, token_scale, x));
    }

    const int32_t n_kv = n_past_ + n_tokens;
    ggml_tensor * kq_mask = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_kv, n_tokens);

    for (int32_t il = 0; il < hparams_.block_count; ++il) {
        const auto & layer = weights_.layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(ctx0, x, layer.attention_norm, hparams_.rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(ctx0, layer.wqkv, attn_in, "mul_mat:wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(ctx0, qkv, q_size, n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], q_size * elem_size);
        ggml_tensor * v2d = ggml_view_2d(ctx0, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(ctx0, ggml_cont(ctx0, q2d), head_dim, n_head, n_tokens);
        ggml_tensor * k = ggml_reshape_3d(ctx0, ggml_cont(ctx0, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(ctx0, ggml_cont(ctx0, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.attention_qk_norm) {
            q = rms_norm_weighted(ctx0, q, layer.q_norm, hparams_.rms_norm_eps);
            k = rms_norm_weighted(ctx0, k, layer.k_norm, hparams_.rms_norm_eps);
        }

        q = ggml_rope_ext(ctx0, q, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(ctx0, k, positions, nullptr, head_dim, 0,
                          hparams_.context_length, hparams_.rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        const size_t layer_off_k = static_cast<size_t>(il) * memory_k_->nb[3];
        const size_t layer_off_v = static_cast<size_t>(il) * memory_v_->nb[3];
        const size_t token_off_k = static_cast<size_t>(n_past_) * memory_k_->nb[2];
        const size_t token_off_v = static_cast<size_t>(n_past_) * memory_v_->nb[2];

        ggml_tensor * k_slot = ggml_view_3d(ctx0, memory_k_,
            head_dim, n_head_kv, n_tokens,
            memory_k_->nb[1], memory_k_->nb[2],
            layer_off_k + token_off_k);
        ggml_tensor * v_slot = ggml_view_3d(ctx0, memory_v_,
            head_dim, n_head_kv, n_tokens,
            memory_v_->nb[1], memory_v_->nb[2],
            layer_off_v + token_off_v);
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, k, k_slot));
        ggml_build_forward_expand(gf, ggml_cpy(ctx0, v, v_slot));

        ggml_tensor * k_mem = k;
        ggml_tensor * v_mem = v;
        if (n_past_ > 0) {
            ggml_tensor * k_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_k_, static_cast<int64_t>(n_past_) * kv_size, layer_off_k),
                head_dim, n_head_kv, n_past_);
            ggml_tensor * v_past = ggml_reshape_3d(ctx0,
                ggml_view_1d(ctx0, memory_v_, static_cast<int64_t>(n_past_) * kv_size, layer_off_v),
                head_dim, n_head_kv, n_past_);
            if (k_past->type != k->type) k_past = ggml_cast(ctx0, k_past, k->type);
            if (v_past->type != v->type) v_past = ggml_cast(ctx0, v_past, v->type);
            k_mem = ggml_concat(ctx0, k_past, k, 2);
            v_mem = ggml_concat(ctx0, v_past, v, 2);
        }

        if (n_head != n_head_kv && q->type != GGML_TYPE_F32) {
            q = ggml_cast(ctx0, q, GGML_TYPE_F32);
        }
        ggml_tensor * k_rep = repeat_interleave_heads(ctx0, k_mem, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(ctx0, v_mem, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(ctx0, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(ctx0, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(ctx0, K, Q, "mul_mat:kq");
        ggml_tensor * KQf = ggml_soft_max_ext(ctx0, KQ, kq_mask, attn_scale, 0.0f);

        ggml_tensor * V       = ggml_cont(ctx0, ggml_permute(ctx0, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(ctx0, V, KQf, "mul_mat:kqv");
        ggml_tensor * KQVm    = ggml_permute(ctx0, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(ctx0, KQVm,
                                          ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(ctx0, layer.wo, attn_cur, "mul_mat:wo");

        ggml_tensor * h     = ggml_add(ctx0, x, attn_out);
        ggml_tensor * ff_in = rms_norm_weighted(ctx0, h, layer.ffn_norm, hparams_.rms_norm_eps);
        ggml_tensor * gate  = mul_mat_checked(ctx0, layer.w1, ff_in, "mul_mat:w1");
        ggml_tensor * up    = mul_mat_checked(ctx0, layer.w3, ff_in, "mul_mat:w3");
        ggml_tensor * ff_h  = ggml_swiglu_split(ctx0, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(ctx0, layer.w2, ff_h, "mul_mat:w2");

        x = ggml_add(ctx0, h, ff_out);
    }

    ggml_tensor * slow_out  = rms_norm_weighted(ctx0, x, weights_.norm, hparams_.rms_norm_eps);
    ggml_tensor * slow_cont = ggml_cont(ctx0, slow_out);
    ggml_tensor * hidden_last = ggml_cpy(ctx0,
        last_token_view(ctx0, slow_cont, n_tokens),
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, dim, 1));

    ggml_tensor * logits = mul_mat_checked(ctx0, weights_.embeddings, hidden_last, "mul_mat:logits");
    ggml_build_forward_expand(gf, logits);

    if (!ggml_gallocr_alloc_graph(allocr_, gf)) {
        std::fprintf(stderr, "[eval_cached] gallocr alloc failed\n");
        ggml_free(ctx0);
        return false;
    }

    std::vector<float> kq_mask_data((size_t)n_kv * n_tokens, -INFINITY);
    for (int32_t i_q = 0; i_q < n_tokens; ++i_q) {
        for (int32_t i_kv = 0; i_kv <= n_past_ + i_q; ++i_kv) {
            kq_mask_data[(size_t)i_q * n_kv + i_kv] = 0.0f;
        }
    }
    ggml_backend_tensor_set(kq_mask, kq_mask_data.data(), 0, kq_mask_data.size() * sizeof(float));

    ggml_backend_tensor_set(semantic_ids,  semantic_vals.data(), 0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(positions,     pos_vals.data(),       0, n_tokens * sizeof(int32_t));
    ggml_backend_tensor_set(semantic_mask, semantic_mask_vals.data(), 0, n_tokens * sizeof(float));
    if (token_scale) {
        ggml_backend_tensor_set(token_scale, token_scale_vals.data(), 0, n_tokens * sizeof(float));
    }
    for (int32_t cb = 0; cb < hparams_.num_codebooks; ++cb) {
        ggml_backend_tensor_set(cb_id_tensors[cb], cb_vals[cb].data(), 0, n_tokens * sizeof(int32_t));
    }

    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, gf) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[eval_cached] compute failed\n");
        ggml_free(ctx0);
        return false;
    }

    result.hidden.resize(dim);
    ggml_backend_tensor_get(hidden_last, result.hidden.data(), 0, (size_t)dim * sizeof(float));

    result.logits.assign(hparams_.vocab_size, -std::numeric_limits<float>::infinity());
    if (logits_range_end_ > logits_range_begin_) {
        const int32_t n_read = logits_range_end_ - logits_range_begin_;
        ggml_backend_tensor_get(logits,
                                result.logits.data() + logits_range_begin_,
                                (size_t)logits_range_begin_ * sizeof(float),
                                (size_t)n_read * sizeof(float));
    } else {
        ggml_backend_tensor_get(logits, result.logits.data(), 0,
                                (size_t)hparams_.vocab_size * sizeof(float));
    }

    ggml_free(ctx0);
    n_past_ += n_tokens;
    return true;
}

// ---------------------------------------------------------------------------
// build_fast_decode_graph() — called lazily for each distinct n_tokens value
// ---------------------------------------------------------------------------

bool SlowARModel::build_fast_decode_graph(int32_t n_tokens, FastDecodeState & fds) {
    const int32_t fast_dim  = hparams_.fast_embedding_length;
    const int32_t n_head    = hparams_.fast_head_count;
    const int32_t n_head_kv = hparams_.fast_head_count_kv;
    const int32_t head_dim  = (hparams_.fast_head_dim > 0)
                                ? hparams_.fast_head_dim
                                : fast_dim / n_head;
    const int32_t q_size    = n_head * head_dim;
    const int32_t kv_size   = n_head_kv * head_dim;
    const float attn_scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const size_t ctx_sz = 4u * 1024u * 1024u;
    ggml_init_params p = { ctx_sz, nullptr, /*no_alloc=*/true };
    fds.ctx = ggml_init(p);
    if (!fds.ctx) return false;

    ggml_cgraph * gf = ggml_new_graph_custom(fds.ctx, 16384, false);

    fds.hidden0   = ggml_new_tensor_2d(fds.ctx, GGML_TYPE_F32, hparams_.embedding_length, 1);
    fds.positions = ggml_new_tensor_1d(fds.ctx, GGML_TYPE_I32, n_tokens);
    fds.kq_mask   = ggml_new_tensor_2d(fds.ctx, GGML_TYPE_F32, n_tokens, n_tokens);

    ggml_tensor * projected = (weights_.fast_project_in != nullptr)
        ? mul_mat_checked(fds.ctx, weights_.fast_project_in, fds.hidden0, "mul_mat:fast_project_in")
        : fds.hidden0;
    if (projected->type != GGML_TYPE_F32) {
        projected = ggml_cast(fds.ctx, projected, GGML_TYPE_F32);
    }

    ggml_tensor * x = projected;
    if (n_tokens > 1) {
        fds.prefix_ids = ggml_new_tensor_1d(fds.ctx, GGML_TYPE_I32, (int64_t)(n_tokens - 1));
        ggml_tensor * fast_emb = emb_f16_.fast_embeddings
                               ? emb_f16_.fast_embeddings
                               : weights_.fast_embeddings;
        ggml_tensor * prefix_emb = ggml_get_rows(fds.ctx, fast_emb, fds.prefix_ids);
        if (prefix_emb->type != GGML_TYPE_F32) {
            prefix_emb = ggml_cast(fds.ctx, prefix_emb, GGML_TYPE_F32);
        }
        x = ggml_concat(fds.ctx, x, prefix_emb, 1);
    }

    for (int32_t il = 0; il < hparams_.fast_block_count; ++il) {
        const auto & layer = weights_.fast_layers[il];

        ggml_tensor * attn_in = rms_norm_weighted(fds.ctx, x, layer.attention_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * qkv     = mul_mat_checked(fds.ctx, layer.wqkv, attn_in, "mul_mat:fast_wqkv");
        const size_t elem_size = ggml_element_size(qkv);

        ggml_tensor * q2d = ggml_view_2d(fds.ctx, qkv, q_size,  n_tokens, qkv->nb[1], 0);
        ggml_tensor * k2d = ggml_view_2d(fds.ctx, qkv, kv_size, n_tokens, qkv->nb[1], q_size  * elem_size);
        ggml_tensor * v2d = ggml_view_2d(fds.ctx, qkv, kv_size, n_tokens, qkv->nb[1], (q_size + kv_size) * elem_size);

        ggml_tensor * q = ggml_reshape_3d(fds.ctx, ggml_cont(fds.ctx, q2d), head_dim, n_head,    n_tokens);
        ggml_tensor * k = ggml_reshape_3d(fds.ctx, ggml_cont(fds.ctx, k2d), head_dim, n_head_kv, n_tokens);
        ggml_tensor * v = ggml_reshape_3d(fds.ctx, ggml_cont(fds.ctx, v2d), head_dim, n_head_kv, n_tokens);

        if (hparams_.fast_attention_qk_norm) {
            q = rms_norm_weighted(fds.ctx, q, layer.q_norm, hparams_.fast_rms_norm_eps);
            k = rms_norm_weighted(fds.ctx, k, layer.k_norm, hparams_.fast_rms_norm_eps);
        }

        q = ggml_rope_ext(fds.ctx, q, fds.positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);
        k = ggml_rope_ext(fds.ctx, k, fds.positions, nullptr, head_dim, 0,
                          hparams_.fast_context_length, hparams_.fast_rope_freq_base,
                          1.0f, 0.0f, 1.0f, 1.0f, 1.0f);

        ggml_tensor * k_rep = repeat_interleave_heads(fds.ctx, k, n_head / n_head_kv);
        ggml_tensor * v_rep = repeat_interleave_heads(fds.ctx, v, n_head / n_head_kv);

        ggml_tensor * Q   = ggml_permute(fds.ctx, q,     0, 2, 1, 3);
        ggml_tensor * K   = ggml_permute(fds.ctx, k_rep, 0, 2, 1, 3);
        ggml_tensor * KQ  = mul_mat_checked(fds.ctx, K, Q, "mul_mat:fast_kq");
        ggml_tensor * KQf = ggml_soft_max_ext(fds.ctx, KQ, fds.kq_mask, attn_scale, 0.0f);

        ggml_tensor * V       = ggml_cont(fds.ctx, ggml_permute(fds.ctx, v_rep, 1, 2, 0, 3));
        ggml_tensor * KQV     = mul_mat_checked(fds.ctx, V, KQf, "mul_mat:fast_kqv");
        ggml_tensor * KQVm    = ggml_permute(fds.ctx, KQV, 0, 2, 1, 3);
        ggml_tensor * attn_cur = ggml_cpy(fds.ctx, KQVm,
                                          ggml_new_tensor_2d(fds.ctx, GGML_TYPE_F32, q_size, n_tokens));
        ggml_tensor * attn_out = mul_mat_checked(fds.ctx, layer.wo, attn_cur, "mul_mat:fast_wo");

        ggml_tensor * h      = ggml_add(fds.ctx, x, attn_out);
        ggml_tensor * ff_in  = rms_norm_weighted(fds.ctx, h, layer.ffn_norm, hparams_.fast_rms_norm_eps);
        ggml_tensor * gate   = mul_mat_checked(fds.ctx, layer.w1, ff_in, "mul_mat:fast_w1");
        ggml_tensor * up     = mul_mat_checked(fds.ctx, layer.w3, ff_in, "mul_mat:fast_w3");
        ggml_tensor * ff_h   = ggml_swiglu_split(fds.ctx, gate, up);
        ggml_tensor * ff_out = mul_mat_checked(fds.ctx, layer.w2, ff_h, "mul_mat:fast_w2");

        x = ggml_add(fds.ctx, h, ff_out);
    }

    ggml_tensor * fast_out  = rms_norm_weighted(fds.ctx, x, weights_.fast_norm, hparams_.fast_rms_norm_eps);
    ggml_tensor * fast_cont = ggml_cont(fds.ctx, fast_out);
    ggml_tensor * fast_last = ggml_cpy(fds.ctx,
        last_token_view(fds.ctx, fast_cont, n_tokens),
        ggml_new_tensor_2d(fds.ctx, GGML_TYPE_F32, fast_dim, 1));
    fds.logits = mul_mat_checked(fds.ctx, weights_.fast_output, fast_last, "mul_mat:fast_logits");
    ggml_build_forward_expand(gf, fds.logits);
    fds.graph = gf;

    fds.allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
    if (!fds.allocr || !ggml_gallocr_alloc_graph(fds.allocr, gf)) {
        std::fprintf(stderr, "[build_fast_decode_graph] gallocr alloc failed (n_tokens=%d)\n", n_tokens);
        ggml_free(fds.ctx);
        fds.ctx = nullptr;
        if (fds.allocr) { ggml_gallocr_free(fds.allocr); fds.allocr = nullptr; }
        return false;
    }

    return true;
}

// ---------------------------------------------------------------------------
// fast_decode() — uses persistent graph, built lazily per n_tokens
// ---------------------------------------------------------------------------

bool SlowARModel::fast_decode(const std::vector<float> & hidden_in,
                               const std::vector<int32_t> & prefix_tokens,
                               int32_t n_threads,
                               std::vector<float> & logits_out) {
    if (!hparams_.has_fast_decoder) {
        std::fprintf(stderr, "[fast_decode] model has no fast decoder\n");
        return false;
    }
    if (static_cast<int32_t>(hidden_in.size()) != hparams_.embedding_length) {
        std::fprintf(stderr, "[fast_decode] expected hidden size %d, got %zu\n",
            hparams_.embedding_length, hidden_in.size());
        return false;
    }
    if (static_cast<int32_t>(prefix_tokens.size()) >= hparams_.num_codebooks) {
        std::fprintf(stderr, "[fast_decode] prefix too long (%zu >= %d)\n",
            prefix_tokens.size(), hparams_.num_codebooks);
        return false;
    }

    const int32_t n_tokens = static_cast<int32_t>(prefix_tokens.size()) + 1;

    // Grow the state vector on demand (indexed by n_tokens-1).
    if (static_cast<int32_t>(fast_decode_states_.size()) < n_tokens) {
        fast_decode_states_.resize(n_tokens);
    }
    FastDecodeState & fds = fast_decode_states_[n_tokens - 1];

    // Build and allocate the graph on first use for this n_tokens value.
    if (!fds.ctx) {
        if (!build_fast_decode_graph(n_tokens, fds)) return false;
    }

    // --- Fill kq_mask (causal, full square for n_tokens) ---
    std::vector<float> kq_mask_data((size_t)n_tokens * n_tokens, -INFINITY);
    for (int32_t i_q = 0; i_q < n_tokens; ++i_q) {
        for (int32_t i_kv = 0; i_kv <= i_q; ++i_kv) {
            kq_mask_data[(size_t)i_q * n_tokens + i_kv] = 0.0f;
        }
    }
    ggml_backend_tensor_set(fds.kq_mask, kq_mask_data.data(), 0, kq_mask_data.size() * sizeof(float));

    // --- Positions (always 0..n_tokens-1 for fast decoder) ---
    std::vector<int32_t> pos_vals(n_tokens);
    for (int32_t i = 0; i < n_tokens; ++i) pos_vals[i] = i;
    ggml_backend_tensor_set(fds.positions, pos_vals.data(), 0, n_tokens * sizeof(int32_t));

    // --- Hidden state and prefix IDs ---
    ggml_backend_tensor_set(fds.hidden0, hidden_in.data(), 0, hidden_in.size() * sizeof(float));
    if (fds.prefix_ids) {
        ggml_backend_tensor_set(fds.prefix_ids, prefix_tokens.data(), 0,
                                prefix_tokens.size() * sizeof(int32_t));
    }

    // --- Compute ---
    if (ggml_backend_is_cpu(backend_)) {
        ggml_backend_cpu_set_n_threads(backend_, n_threads);
    }
    if (ggml_backend_graph_compute(backend_, fds.graph) != GGML_STATUS_SUCCESS) {
        std::fprintf(stderr, "[fast_decode] compute failed\n");
        return false;
    }

    logits_out.resize(hparams_.codebook_size);
    ggml_backend_tensor_get(fds.logits, logits_out.data(), 0,
                            hparams_.codebook_size * sizeof(float));
    return true;
}

} // namespace s2

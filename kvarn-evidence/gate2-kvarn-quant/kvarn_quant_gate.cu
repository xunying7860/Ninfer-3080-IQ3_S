// kvarn_quant_gate.cu — GATE 2: KVarN G128 CUDA 量化/存储 kernel 与标量 oracle 的
//                       record 逐字节对拍（sm_86 / RTX 3080）
// ---------------------------------------------------------------------------
// 本文件的两块内容：
//   (A) 设备端：从 beellama.cpp 逐行抄来的 KVarN G128 量化 kernel
//       ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu（commit 58a1629274d8d74ea5a6acf34f30fb5946c2422b）
//         kvarn.cu:12-19   KVAR_N_* 常量
//         kvarn.cu:456-472 kvarn_wht_128
//         kvarn.cu:579-591 kvarn_warp_min / kvarn_warp_max
//         kvarn.cu:593-622 kvarn_reduce_std_ranges
//         kvarn.cu:624-658 kvarn_update_best_from_std
//         kvarn.cu:545-577 kvarn_std_col / kvarn_std_row   (std 分母 = KVAR_N_DIM-1 = 127)
//         kvarn.cu:660-757 kvarn_quantize_tile             (tile 常驻不原地改 + 打包先 memset)
//         kvarn.cu:510-543 kvarn_wht_cross_slices + 1113-1131 的 store 旋转段
//         kvarn.cu:1059-1143 kvarn_store_kernel_headwide   (128 线程 / opt-in smem 的存储 kernel)
//       唯一改动 = kvarn_quantize_tile 末尾追加 3 行 [GATE2] 诊断（默认 nullptr，对 record 无影响）。
//   (B) 主机端：读 dumprec 落盘的输入 tile + oracle record，跑 kernel，逐字节比。
// ---------------------------------------------------------------------------
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(3); } } while (0)

// ===========================================================================
// (A) 设备端 —— 逐行抄自 kvarn.cu
// ===========================================================================
static constexpr int KVAR_N_DIM = 128;                                                    // kvarn.cu:12
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;                        // kvarn.cu:14
static constexpr int KVAR_N_REDUCE_FLOATS = 4 * 4;                                        // kvarn.cu:15
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;  // kvarn.cu:16
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);          // kvarn.cu:17

// ---- kvarn.cu:456-472 ----
static __device__ void kvarn_wht_128(float * values) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (threadIdx.x < 64) {
            const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (threadIdx.x < KVAR_N_DIM) {
        values[threadIdx.x] *= 0.08838834764831845f;
    }
    __syncthreads();
}

// ---- kvarn.cu:579-591 ----
static __device__ __forceinline__ float kvarn_warp_min(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}
static __device__ __forceinline__ float kvarn_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}

// ---- kvarn.cu:593-622 ----
static __device__ void kvarn_reduce_std_ranges(
        const float * col_std,
        const float * row_std,
        float * reduce) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    float col_min = kvarn_warp_min(col_std[threadIdx.x]);
    float col_max = kvarn_warp_max(col_std[threadIdx.x]);
    float row_min = kvarn_warp_min(row_std[threadIdx.x]);
    float row_max = kvarn_warp_max(row_std[threadIdx.x]);

    if (lane == 0) {
        reduce[warp * 4 + 0] = col_min;
        reduce[warp * 4 + 1] = col_max;
        reduce[warp * 4 + 2] = row_min;
        reduce[warp * 4 + 3] = row_max;
    }
    __syncthreads();

    if (threadIdx.x < 4) {
        const int metric = threadIdx.x;
        float value = reduce[metric];
        for (int w = 1; w < 4; ++w) {
            const float next = reduce[w * 4 + metric];
            value = (metric == 0 || metric == 2) ? fminf(value, next) : fmaxf(value, next);
        }
        reduce[metric] = value;
    }
    __syncthreads();
}

// ---- kvarn.cu:624-658 ----
static __device__ void kvarn_update_best_from_std(
        const float * candidate_col,
        const float * candidate_row,
        bool candidate_is_log,
        float * best_col,
        float * best_row,
        float * col_std,
        float * row_std,
        float * best_imbalance,
        float * better,
        float * reduce) {
    const int i = threadIdx.x;
    kvarn_reduce_std_ranges(col_std, row_std, reduce);

    if (i == 0) {
        const float col_min = reduce[0];
        const float col_max = reduce[1];
        const float row_min = reduce[2];
        const float row_max = reduce[3];
        const float imbalance =
            col_max / fmaxf(col_min, 1e-8f) +
            row_max / fmaxf(row_min, 1e-8f);
        *better = imbalance <= *best_imbalance ? 1.0f : 0.0f;
        if (*better != 0.0f) {
            *best_imbalance = imbalance;
        }
    }
    __syncthreads();

    if (*better != 0.0f) {
        best_col[i] = candidate_is_log ? expf(candidate_col[i]) : candidate_col[i];
        best_row[i] = candidate_is_log ? expf(candidate_row[i]) : candidate_row[i];
    }
    __syncthreads();
}

// ---- kvarn.cu:545-577（std 分母 = KVAR_N_DIM - 1 = 127）----
static __device__ float kvarn_std_col(
        const float * tile,
        const float * s_col,
        const float * s_row,
        int col) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sc = s_col[col];
    for (int row = 0; row < KVAR_N_DIM; ++row) {
        const float value = tile[row * KVAR_N_DIM + col] / (sc * s_row[row]);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}
static __device__ float kvarn_std_row(
        const float * tile,
        const float * s_col,
        const float * s_row,
        int row) {
    float sum = 0.0f;
    float sum_sq = 0.0f;
    const float sr = s_row[row];
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float value = tile[row * KVAR_N_DIM + col] / (s_col[col] * sr);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / KVAR_N_DIM;
    return sqrtf(fmaxf((sum_sq - KVAR_N_DIM * mean * mean) / (KVAR_N_DIM - 1), 0.0f));
}

// ---- kvarn.cu:660-757（唯一改动：末尾 3 行 [GATE2] 诊断；默认 nullptr 时行为与上游完全一致）----
static __device__ void kvarn_quantize_tile(
        uint8_t * record,
        int bits,
        int iterations,
        float * shared,
        float * gate_best_col = nullptr,     // [GATE2]
        float * gate_best_row = nullptr) {   // [GATE2]
    float * tile = shared;
    float * log_s_col = tile + KVAR_N_TILE_VALUES;
    float * log_s_row = log_s_col + KVAR_N_DIM;
    float * s_col = log_s_row + KVAR_N_DIM;
    float * s_row = s_col + KVAR_N_DIM;
    float * best_col = s_row + KVAR_N_DIM;
    float * best_row = best_col + KVAR_N_DIM;
    float * col_std = best_row + KVAR_N_DIM;
    float * row_std = col_std + KVAR_N_DIM;
    float * best_imbalance = row_std + KVAR_N_DIM;
    float * better = best_imbalance + 1;
    float * reduce = better + 1;

    log_s_col[threadIdx.x] = 0.0f;
    log_s_row[threadIdx.x] = 0.0f;
    s_col[threadIdx.x] = 1.0f;
    s_row[threadIdx.x] = 1.0f;
    best_col[threadIdx.x] = 1.0f;
    best_row[threadIdx.x] = 1.0f;
    __syncthreads();

    col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
    row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) {
        *best_imbalance = 3.402823466e+38F;
    }
    __syncthreads();
    kvarn_update_best_from_std(
            s_col, s_row, false,
            best_col, best_row, col_std, row_std,
            best_imbalance, better, reduce);

    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(col_std[threadIdx.x], 1e-3f), 1e3f);
        log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
        s_col[threadIdx.x] = expf(log_s_col[threadIdx.x]);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        const float row = fminf(fmaxf(row_std[threadIdx.x], 1e-3f), 1e3f);
        const float log_s_row_new = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
        log_s_row[threadIdx.x] = log_s_row_new;
        s_row[threadIdx.x] = expf(log_s_row_new);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        kvarn_update_best_from_std(
                s_col, s_row, false,
                best_col, best_row, col_std, row_std,
                best_imbalance, better, reduce);
    }

    const int row = threadIdx.x;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }

    const int qmax = (1 << bits) - 1;
    const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = KVAR_N_DIM * bits / 8;
    uint8_t * row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) {
        row_payload[i] = 0;                                  // 打包先 memset（kvarn.cu:736-738）
    }
    for (int col = 0; col < KVAR_N_DIM; ++col) {
        const float x = tile[row * KVAR_N_DIM + col] / (best_col[col] * best_row[row]);
        const uint8_t q = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= ((q >> bit) & 1u) << (dst_bit % 8);
        }
    }

    const int payload_bytes = KVAR_N_TILE_VALUES * bits / 8;
    half * scale_axis = (half *) (record + payload_bytes);
    half * zp_axis = scale_axis + KVAR_N_DIM;
    half * other_axis = zp_axis + KVAR_N_DIM;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row] = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
    __syncthreads();
    // [GATE2] 追加的诊断导出（上游无此三行；对 record 无任何影响）
    if (gate_best_col != nullptr) { gate_best_col[row] = best_col[row]; }
    if (gate_best_row != nullptr) { gate_best_row[row] = best_row[row]; }
    __syncthreads();
}

// ---- kvarn.cu:510-543 ----
static __device__ void kvarn_wht_cross_slices(float values[4], int head_slices) {
    if (head_slices == 1) {
        return;
    }
    for (int stride = 1; stride < head_slices; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < stride; ++i) {
                const int a_idx = base + i;
                const int b_idx = base + stride + i;
                if (b_idx >= head_slices) {
                    continue;
                }
                const float a = values[a_idx];
                const float b = values[b_idx];
                values[a_idx] = a + b;
                values[b_idx] = a - b;
            }
        }
    }
    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        if (slice < head_slices) {
            values[slice] *= scale;
        }
    }
}

// ---- 量化 kernel 的启动壳：128 线程 / 动态 smem = KVAR_N_SHARED_BYTES（opt-in）----
__global__ void kvarn_quant_tile_kernel(
        const float * __restrict__ tile_in,
        uint8_t * __restrict__ record,
        int bits,
        int iterations,
        float * __restrict__ gate_best_col,
        float * __restrict__ gate_best_row) {
    extern __shared__ float shared[];
    float * tile = shared;
    for (int i = threadIdx.x; i < KVAR_N_TILE_VALUES; i += blockDim.x) {
        tile[i] = tile_in[i];                     // tile 常驻 smem，之后全程只读（不原地改）
    }
    __syncthreads();
    kvarn_quantize_tile(record, bits, iterations, shared, gate_best_col, gate_best_row);
}

// ===========================================================================
// (B) 主机端 —— 读 dumprec 的 bin，跑 kernel，逐字节比
// ===========================================================================
struct Rec {
    std::string name; int rows, cols, bits, iters, best_iter; float best_imb;
    std::vector<float> s_col, s_row, tile; std::vector<uint8_t> rec;
};

static bool rd(void * dst, size_t n, FILE * f) { return fread(dst, 1, n, f) == n; }
static bool r_u32(uint32_t & v, FILE * f) { return rd(&v, 4, f); }
static bool r_i32(int32_t & v, FILE * f) { return rd(&v, 4, f); }

static bool load_dump(const char * path, std::vector<Rec> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开 %s\n", path); return false; }
    uint32_t magic = 0, ver = 0, n = 0;
    if (!r_u32(magic, f) || !r_u32(ver, f) || !r_u32(n, f)) { fclose(f); return false; }
    if (magic != 0x324B5647u || ver != 2u) { fprintf(stderr, "magic/ver 不对 (%08x,%u)\n", magic, ver); fclose(f); return false; }
    for (uint32_t k = 0; k < n; ++k) {
        Rec D;
        uint32_t nl = 0; if (!r_u32(nl, f)) { fclose(f); return false; }
        D.name.resize(nl); if (nl && !rd(&D.name[0], nl, f)) { fclose(f); return false; }
        int32_t a = 0;
        if (!r_i32(a, f)) { fclose(f); return false; } D.rows = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.cols = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.bits = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.iters = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.best_iter = a;
        if (!rd(&D.best_imb, 4, f)) { fclose(f); return false; }
        D.s_col.resize(D.cols); D.s_row.resize(D.rows);
        if (!rd(D.s_col.data(), 4 * size_t(D.cols), f)) { fclose(f); return false; }
        if (!rd(D.s_row.data(), 4 * size_t(D.rows), f)) { fclose(f); return false; }
        D.tile.resize(size_t(D.rows) * D.cols);
        if (!rd(D.tile.data(), 4 * D.tile.size(), f)) { fclose(f); return false; }
        uint32_t rl = 0; if (!r_u32(rl, f)) { fclose(f); return false; }
        D.rec.resize(rl);
        if (rl && !rd(D.rec.data(), rl, f)) { fclose(f); return false; }
        out.push_back(std::move(D));
    }
    fclose(f);
    return true;
}

static float h_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = sign;
        else { int32_t ee = 127 - 15 + 1; uint32_t mm = m;
               while (!(mm & 0x400u)) { mm <<= 1; --ee; } mm &= 0x3FFu;
               u = sign | ((uint32_t)ee << 23) | (mm << 13); }
    } else if (e == 0x1Fu) u = sign | 0x7F800000u | (m << 13);
    else u = sign | ((e + 127 - 15) << 23) | (m << 13);
    float f; memcpy(&f, &u, 4); return f;
}
static uint16_t rd_h16(const uint8_t * p) { uint16_t h; memcpy(&h, p, 2); return h; }

// 反量化：x[r][c] = (q*scale_axis[r] + zp_axis[r]) * other_axis[c]
static void dequant(const uint8_t * rec, int bits, float * out) {
    const int rows = KVAR_N_DIM, cols = KVAR_N_DIM;
    const size_t payload = size_t(rows) * cols * bits / 8;
    const int row_bytes = cols * bits / 8;
    for (int r = 0; r < rows; ++r) {
        const float scale = h_f16_to_f32(rd_h16(rec + payload + size_t(r) * 2));
        const float zp    = h_f16_to_f32(rd_h16(rec + payload + size_t(rows) * 2 + size_t(r) * 2));
        for (int c = 0; c < cols; ++c) {
            float other = h_f16_to_f32(rd_h16(rec + payload + size_t(2 * rows) * 2 + size_t(c) * 2));
            uint8_t q = 0;
            const size_t bit_offset = size_t(c) * bits;
            for (int b = 0; b < bits; ++b) {
                const size_t sb = bit_offset + b;
                q |= uint8_t(((rec[r * row_bytes + sb / 8] >> (sb % 8)) & 1u) << b);
            }
            out[size_t(r) * cols + c] = (float(q) * scale + zp) * other;
        }
    }
}

int main(int argc, char ** argv) {
    bool verbose = false, sens = false;
    int first_arg = 1;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-v")) { verbose = true; first_arg = i + 1; }
        if (!strcmp(argv[i], "-s")) { sens = true;    first_arg = i + 1; }
    }
    if (first_arg >= argc) {
        fprintf(stderr, "用法: ./gate [-v] [-s] <dump.bin> [dump2.bin ...]\n"
                        "  -v 逐 record 打印；-s 追加 1-ulp 输入扰动敏感性探针\n");
        return 2;
    }

    // ---- 设备 / smem 报告（实测）----
    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int max_optin = 0, max_block = 0, max_smem_sm = 0, reserved = 0, smem_sm_native = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&max_optin,      cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_block,      cudaDevAttrMaxSharedMemoryPerBlock,      dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&max_smem_sm,    cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&reserved,       cudaDevAttrReservedSharedMemoryPerBlock, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&smem_sm_native, cudaDevAttrMaxSharedMemoryPerMultiprocessor, dev));

    printf("================ GATE2: KVarN G128 CUDA 量化/存储 kernel ================\n");
    printf("[device] %s  sm_%d%d  (compute capability %d.%d)\n",
           prop.name, prop.major, prop.minor, prop.major, prop.minor);
    printf("[device] smem/block 默认上限      cudaDevAttrMaxSharedMemoryPerBlock      = %d B\n", max_block);
    printf("[device] smem/block opt-in 上限   ...PerBlockOptin                      = %d B  <== 设备上限\n", max_optin);
    printf("[device] smem/SM                  ...PerMultiprocessor                  = %d B\n", max_smem_sm);
    printf("[device] 每 block 预留 smem        ...ReservedSharedMemoryPerBlock       = %d B\n", reserved);
    printf("[need  ] KVAR_N_SHARED_FLOATS = 16384 + 8*128 + 2 + 16 = %d floats\n", KVAR_N_SHARED_FLOATS);
    printf("[need  ] KVAR_N_SHARED_BYTES                          = %d B (%.2f KB)  <== 实测所需动态 smem\n",
           KVAR_N_SHARED_BYTES, KVAR_N_SHARED_BYTES / 1024.0);
    printf("[need  ] 每 block 线程数 = 128（KVAR_N_DIM；kvarn.cu:678 起 threadIdx.x 即 row）\n");

    size_t pmax = 0; (void)pmax;
    cudaError_t pe = cudaFuncSetAttribute(kvarn_quant_tile_kernel,
                                          cudaFuncAttributeMaxDynamicSharedMemorySize, KVAR_N_SHARED_BYTES);
    CUDA_CHECK(pe);
    CUDA_CHECK(cudaFuncSetAttribute(kvarn_quant_tile_kernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));
    cudaFuncAttributes fa{}; CUDA_CHECK(cudaFuncGetAttributes(&fa, kvarn_quant_tile_kernel));
    printf("[attr  ] kernel 静态 smem=%zu B, 寄存器=%d/线程, maxDynamicSharedSizeBytes=%d B\n",
           fa.sharedSizeBytes, fa.numRegs, fa.maxDynamicSharedSizeBytes);
    int occ = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kvarn_quant_tile_kernel, 128, KVAR_N_SHARED_BYTES));
    printf("[attr  ] 占用率：%d blocks/SM（128 线程 + %d B 动态 smem/block）=> %d 线程/SM (上限 %d)\n",
           occ, KVAR_N_SHARED_BYTES, occ * 128, prop.maxThreadsPerMultiProcessor);
    (void)pmax; (void)smem_sm_native;

    float * d_in = nullptr; uint8_t * d_rec = nullptr;
    float * d_bc = nullptr; float * d_br = nullptr;
    const size_t REC_MAX = 2048 * 8 + 768;
    CUDA_CHECK(cudaMalloc(&d_in, sizeof(float) * KVAR_N_TILE_VALUES));
    CUDA_CHECK(cudaMalloc(&d_rec, REC_MAX));
    CUDA_CHECK(cudaMalloc(&d_bc, sizeof(float) * KVAR_N_DIM));
    CUDA_CHECK(cudaMalloc(&d_br, sizeof(float) * KVAR_N_DIM));

    int total_recs = 0, total_byte_diff = 0, verbatim_fail = 0;
    int files_ok = 0;
    int s_recs = 0, s_diffrecs = 0, s_bytes = 0; long long s_nulp = 0; double s_dq = 0;

    for (int fi = first_arg; fi < argc; ++fi) {
        std::vector<Rec> recs;
        if (!load_dump(argv[fi], recs)) { fprintf(stderr, "load 失败: %s\n", argv[fi]); return 3; }
        int f_recs = 0, f_diffrecs = 0, f_diffbytes = 0;
        double f_dc = 0, f_dr = 0, f_dq = 0, f_dqr = 0;
        if (verbose) {
            printf("\n---- %s : %zu records ----\n", argv[fi], recs.size());
            printf("%-22s %2s | rec B | 字节不同 | maxd|s_col| maxd|s_row| maxd|deq| maxrel|deq| | 首差异@ | 段\n",
                   "name", "b");
        }
        files_ok++;
        std::vector<float> a_deq(KVAR_N_TILE_VALUES), b_deq(KVAR_N_TILE_VALUES);
        for (auto & D : recs) {
            if (D.rows != KVAR_N_DIM || D.cols != KVAR_N_DIM) {
                if (verbose) printf("%-22s SKIP (rows=%d cols=%d)\n", D.name.c_str(), D.rows, D.cols);
                continue;
            }
            CUDA_CHECK(cudaMemcpy(d_in, D.tile.data(), sizeof(float) * KVAR_N_TILE_VALUES, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_rec, 0xAA, REC_MAX));
            kvarn_quant_tile_kernel<<<1, KVAR_N_DIM, KVAR_N_SHARED_BYTES>>>(d_in, d_rec, D.bits, D.iters, d_bc, d_br);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<uint8_t> got(D.rec.size());
            CUDA_CHECK(cudaMemcpy(got.data(), d_rec, D.rec.size(), cudaMemcpyDeviceToHost));
            std::vector<float> bc(KVAR_N_DIM), br(KVAR_N_DIM);
            CUDA_CHECK(cudaMemcpy(bc.data(), d_bc, 4 * KVAR_N_DIM, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(br.data(), d_br, 4 * KVAR_N_DIM, cudaMemcpyDeviceToHost));

            int nd = 0, first = -1;
            for (size_t i = 0; i < D.rec.size(); ++i) {
                if (got[i] != D.rec[i]) { if (first < 0) first = (int)i; ++nd; }
            }
            double dc = 0, dr = 0;
            for (int i = 0; i < KVAR_N_DIM; ++i) {
                dc = fmax(dc, fabs((double)bc[i] - D.s_col[i]));
                dr = fmax(dr, fabs((double)br[i] - D.s_row[i]));
            }
            // 反量化域的差（codec 级：同一份输入、两条路径各自的 record 解出来的值）
            dequant(got.data(), D.bits, a_deq.data());
            dequant(D.rec.data(), D.bits, b_deq.data());
            double dq = 0, bscale = 0;
            for (size_t i = 0; i < b_deq.size(); ++i) bscale = fmax(bscale, fabs((double)b_deq[i]));
            for (size_t i = 0; i < a_deq.size(); ++i) dq = fmax(dq, fabs((double)a_deq[i] - b_deq[i]));
            const double dqr = bscale > 0 ? dq / bscale : 0.0;
            const int payload = 2048 * D.bits;
            const char * seg = "-";
            if (first >= 0) {
                if (first < payload) seg = "payload";
                else if (first < payload + 256) seg = "scale_axis";
                else if (first < payload + 512) seg = "zp_axis";
                else seg = "other_axis";
            }
            if (verbose || nd != 0) {
                printf("%-22s %2d | %5zu | %8d | %.3e  %.3e | %.3e %.3e | %8d | %s\n",
                       D.name.c_str(), D.bits, D.rec.size(), nd, dc, dr, dq, dqr, first, seg);
            }
            // ---- 1-ulp 输入扰动敏感性探针：算法自身对 ulp 级扰动放大多少？ ----
            if (sens) {
                std::vector<float> pert(D.tile);
                int nulp = 0;
                for (auto & v : pert) { const float w = std::nextafterf(v, INFINITY); if (w != v) { v = w; ++nulp; } }
                CUDA_CHECK(cudaMemcpy(d_in, pert.data(), sizeof(float) * KVAR_N_TILE_VALUES, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemset(d_rec, 0xAA, REC_MAX));
                kvarn_quant_tile_kernel<<<1, KVAR_N_DIM, KVAR_N_SHARED_BYTES>>>(d_in, d_rec, D.bits, D.iters, d_bc, d_br);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
                std::vector<uint8_t> got2(D.rec.size());
                CUDA_CHECK(cudaMemcpy(got2.data(), d_rec, D.rec.size(), cudaMemcpyDeviceToHost));
                std::vector<float> c_deq(KVAR_N_TILE_VALUES);
                dequant(got2.data(), D.bits, c_deq.data());
                int nd2 = 0; for (size_t i = 0; i < got.size(); ++i) if (got2[i] != got[i]) ++nd2;
                double dq2 = 0; for (size_t i = 0; i < c_deq.size(); ++i) dq2 = fmax(dq2, fabs((double)c_deq[i] - a_deq[i]));
                s_recs++; if (nd2) s_diffrecs++; s_bytes += nd2; s_dq = fmax(s_dq, dq2); s_nulp += nulp;
            }
            total_recs++; total_byte_diff += nd; f_recs++; f_diffbytes += nd; f_dc = fmax(f_dc, dc); f_dr = fmax(f_dr, dr);
            f_dq = fmax(f_dq, dq); f_dqr = fmax(f_dqr, dqr);
            if (nd != 0) { verbatim_fail++; f_diffrecs++; }
        }
        printf("[file] %-28s recs=%3d 有差异record=%3d 差异字节=%5d | maxd|s_col|=%.3e maxd|s_row|=%.3e maxd|deq|=%.3e maxrel|deq|=%.3e\n",
               argv[fi], f_recs, f_diffrecs, f_diffbytes, f_dc, f_dr, f_dq, f_dqr);
    }

    printf("\n================ 汇总 ================\n");
    printf("records=%d  files=%d  字节不完全一致的 record 数=%d  差异字节总数=%d\n",
           total_recs, files_ok, verbatim_fail, total_byte_diff);
    if (sens) {
        printf("[sens ] 1-ulp 输入扰动探针：records=%d 有差异record=%d 差异字节=%d  扰动元素数=%lld  最大反量化差=%.3e\n",
               s_recs, s_diffrecs, s_bytes, s_nulp, s_dq);
    }
    printf("required dynamic smem = %d B (%.2f KB)  |  设备 opt-in 上限 = %d B (%.2f KB)  |  余量 = %d B\n",
           KVAR_N_SHARED_BYTES, KVAR_N_SHARED_BYTES / 1024.0,
           max_optin, max_optin / 1024.0, max_optin - KVAR_N_SHARED_BYTES);
    printf("判定：%s\n", (total_recs > 0 && verbatim_fail == 0) ? "PASS（全部 record 逐字节一致）"
                                                             : "FAIL（见上表逐条差异）");
    cudaFree(d_in); cudaFree(d_rec); cudaFree(d_bc); cudaFree(d_br);
    return verbatim_fail == 0 ? 0 : 1;
}

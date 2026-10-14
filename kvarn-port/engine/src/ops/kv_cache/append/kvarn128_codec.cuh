#pragma once

// ===========================================================================
// kvarn128_codec.cuh —— G128 KVarN 的**设备端**编解码（encode / decode）
//
// 量化数学 = gate 2 已验通的 kvarn_quantize_tile（逐行抄自上游 kvarn.cu:660-757，
// 与标量 oracle 在 iters=0 下 576/576 record、0 差异字节）。本文件唯一改动是
// **写入端**：不再吐上游那套「record = payload + 3 个轴」的一整块内存，而是按 ninfer 的
// 平面约定（[页][token][slot]，见 kvarn_layout.h）落进 4 个 plane。
//
// encode 与 decode 共用 kvarn_layout.h 的槽位映射与寻址 —— 这是"能跑但数值错"这类
// 静默故障的唯一有效防线：两边不可能各写一套还没有人来对。
// ===========================================================================

#include "ops/kv_cache/hadamard_d256.cuh"
#include "ops/kv_cache/kvarn_layout.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ninfer::ops {

// bits ≤ 4 时一条 record 的最大字节（8960 = 2048*4 + 768）；K4V4 是本轮唯一注册档。
inline constexpr int kKvarnMaxRecordBytes = 2048 * 4 + 768;

// ===========================================================================
// 一、上游数值路径的 device 助手（gate 2 逐行口径）
// ===========================================================================

// ---- 旋转：一个 warp 拥有一个 D256 token 行（lane l 载 dims l+32r）----
// gate 1 已证 ninfer 的 normalized_hadamard_d256_inplace 与 beellama 的
// kvarn_wht_128 + kvarn_wht_cross_slices 结构逐位相同（3,145,728 元素 max|Δ| = 0），
// 且 ninfer 侧归一化更准（末级单点精确 2^-4）⇒ 照搬 ninfer 现成的，不补 cross-slice 蝶形。
__device__ __forceinline__ void kvarn128_rotate_token(float (&values)[8], int lane) {
    normalized_hadamard_d256_inplace(values, lane);
}

__device__ __forceinline__ float kvarn_warp_min_(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fminf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}
__device__ __forceinline__ float kvarn_warp_max_(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}
__device__ __forceinline__ void kvarn_reduce_std_ranges_(const float* col_std, const float* row_std,
                                                         float* reduce) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const float col_min = kvarn_warp_min_(col_std[threadIdx.x]);
    const float col_max = kvarn_warp_max_(col_std[threadIdx.x]);
    const float row_min = kvarn_warp_min_(row_std[threadIdx.x]);
    const float row_max = kvarn_warp_max_(row_std[threadIdx.x]);
    if (lane == 0) {
        reduce[warp * 4 + 0] = col_min;
        reduce[warp * 4 + 1] = col_max;
        reduce[warp * 4 + 2] = row_min;
        reduce[warp * 4 + 3] = row_max;
    }
    __syncthreads();
    if (threadIdx.x < 4) {
        const int metric = threadIdx.x;
        float value      = reduce[metric];
        for (int w = 1; w < 4; ++w) {
            const float next = reduce[w * 4 + metric];
            value = (metric == 0 || metric == 2) ? fminf(value, next) : fmaxf(value, next);
        }
        reduce[metric] = value;
    }
    __syncthreads();
}
__device__ __forceinline__ void kvarn_update_best_from_std_(const float* candidate_col,
                                                            const float* candidate_row,
                                                            bool candidate_is_log, float* best_col,
                                                            float* best_row, float* col_std,
                                                            float* row_std, float* best_imbalance,
                                                            float* better, float* reduce) {
    const int i = threadIdx.x;
    kvarn_reduce_std_ranges_(col_std, row_std, reduce);
    if (i == 0) {
        const float col_min = reduce[0];
        const float col_max = reduce[1];
        const float row_min = reduce[2];
        const float row_max = reduce[3];
        const float imbalance = col_max / fmaxf(col_min, 1e-8f) + row_max / fmaxf(row_min, 1e-8f);
        *better = imbalance <= *best_imbalance ? 1.0f : 0.0f;
        if (*better != 0.0f) { *best_imbalance = imbalance; }
    }
    __syncthreads();
    if (*better != 0.0f) {
        best_col[i] = candidate_is_log ? expf(candidate_col[i]) : candidate_col[i];
        best_row[i] = candidate_is_log ? expf(candidate_row[i]) : candidate_row[i];
    }
    __syncthreads();
}
// std 分母固定 = kKvarnSliceDim - 1 = 127（样本标准差）。写成 128 会让 scale 系统性偏
// sqrt(128/127) ≈ 1.0039（doc 03 §3.4 点名的坑）。
__device__ __forceinline__ float kvarn_std_col(const float* tile, const float* s_col,
                                               const float* s_row, int col) {
    float sum = 0.0f, sum_sq = 0.0f;
    const float sc = s_col[col];
    for (int row = 0; row < kKvarnSliceDim; ++row) {
        const float value = tile[row * kKvarnSliceDim + col] / (sc * s_row[row]);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / kKvarnSliceDim;
    return sqrtf(fmaxf((sum_sq - kKvarnSliceDim * mean * mean) / (kKvarnSliceDim - 1), 0.0f));
}
__device__ __forceinline__ float kvarn_std_row(const float* tile, const float* s_col,
                                               const float* s_row, int row) {
    float sum = 0.0f, sum_sq = 0.0f;
    const float sr = s_row[row];
    for (int col = 0; col < kKvarnSliceDim; ++col) {
        const float value = tile[row * kKvarnSliceDim + col] / (s_col[col] * sr);
        sum += value;
        sum_sq += value * value;
    }
    const float mean = sum / kKvarnSliceDim;
    return sqrtf(fmaxf((sum_sq - kKvarnSliceDim * mean * mean) / (kKvarnSliceDim - 1), 0.0f));
}

// ---- 量化：与 gate 2 的 kvarn_quantize_tile 逐行相同（唯一差别是导出 best_col/best_row）----
// record 布局保持上游口径：payload[row*row_bytes + col*bits]，三个轴紧随其后。
// row/col 的语义由调用方决定（K: row=dim, col=token；V: row=token, col=dim），
// 与上游 kvarn_quantize_stage 的 token/dim 装载一致。128 个线程必须全部参与（函数内有 __syncthreads）。
__device__ __forceinline__ void kvarn128_quantize_tile(std::uint8_t* record, int bits,
                                                       int iterations, float* shared,
                                                       float* best_col_out, float* best_row_out) {
    float* tile           = shared;
    float* log_s_col      = tile + kKvarnTileTokens * kKvarnSliceDim;
    float* log_s_row      = log_s_col + kKvarnSliceDim;
    float* s_col          = log_s_row + kKvarnSliceDim;
    float* s_row          = s_col + kKvarnSliceDim;
    float* best_col       = s_row + kKvarnSliceDim;
    float* best_row       = best_col + kKvarnSliceDim;
    float* col_std        = best_row + kKvarnSliceDim;
    float* row_std        = col_std + kKvarnSliceDim;
    float* best_imbalance = row_std + kKvarnSliceDim;
    float* better         = best_imbalance + 1;
    float* reduce         = better + 1;

    log_s_col[threadIdx.x] = 0.0f;
    log_s_row[threadIdx.x] = 0.0f;
    s_col[threadIdx.x]     = 1.0f;
    s_row[threadIdx.x]     = 1.0f;
    best_col[threadIdx.x]  = 1.0f;
    best_row[threadIdx.x]  = 1.0f;
    __syncthreads();

    col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
    row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
    __syncthreads();
    if (threadIdx.x == 0) { *best_imbalance = 3.402823466e+38F; }
    __syncthreads();
    kvarn_update_best_from_std_(s_col, s_row, false, best_col, best_row, col_std, row_std,
                                best_imbalance, better, reduce);

    for (int iter = 0; iter < iterations; ++iter) {
        const float col = fminf(fmaxf(col_std[threadIdx.x], 1e-3f), 1e3f);
        log_s_col[threadIdx.x] = fminf(fmaxf(log_s_col[threadIdx.x] + logf(col), -0.3f), 10.0f);
        s_col[threadIdx.x]     = expf(log_s_col[threadIdx.x]);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        const float row = fminf(fmaxf(row_std[threadIdx.x], 1e-3f), 1e3f);
        const float log_s_row_new = fminf(fmaxf(log_s_row[threadIdx.x] + logf(row), -0.3f), 10.0f);
        log_s_row[threadIdx.x] = log_s_row_new;
        s_row[threadIdx.x]     = expf(log_s_row_new);
        __syncthreads();

        row_std[threadIdx.x] = kvarn_std_row(tile, s_col, s_row, threadIdx.x);
        col_std[threadIdx.x] = kvarn_std_col(tile, s_col, s_row, threadIdx.x);
        __syncthreads();

        kvarn_update_best_from_std_(s_col, s_row, false, best_col, best_row, col_std, row_std,
                                    best_imbalance, better, reduce);
    }

    const int row = threadIdx.x;
    float lo = 3.402823466e+38F;
    float hi = -3.402823466e+38F;
    for (int col = 0; col < kKvarnSliceDim; ++col) {
        const float x = tile[row * kKvarnSliceDim + col] / (best_col[col] * best_row[row]);
        lo = fminf(lo, x);
        hi = fmaxf(hi, x);
    }
    const int qmax      = (1 << bits) - 1;
    const float scale   = fmaxf((hi - lo) / qmax, 1e-10f);
    const int row_bytes = kKvarnSliceDim * bits / 8;
    std::uint8_t* row_payload = record + row * row_bytes;
    for (int i = 0; i < row_bytes; ++i) { row_payload[i] = 0; } // 打包先 memset（上游 kvarn.cu:736-738）
    for (int col = 0; col < kKvarnSliceDim; ++col) {
        const float x = tile[row * kKvarnSliceDim + col] / (best_col[col] * best_row[row]);
        const std::uint8_t q =
            (std::uint8_t)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
        const int bit_offset = col * bits;
        for (int bit = 0; bit < bits; ++bit) {
            const int dst_bit = bit_offset + bit;
            row_payload[dst_bit / 8] |= (std::uint8_t)(((q >> bit) & 1u) << (dst_bit % 8));
        }
    }
    const int payload_bytes = kKvarnTileTokens * kKvarnSliceDim * bits / 8;
    __half* scale_axis = (__half*)(record + payload_bytes);
    __half* zp_axis    = scale_axis + kKvarnSliceDim;
    __half* other_axis = zp_axis + kKvarnSliceDim;
    scale_axis[row] = __float2half_rn(best_row[row] * scale);
    zp_axis[row]    = __float2half_rn(best_row[row] * lo);
    other_axis[row] = __float2half_rn(best_col[row]);
    __syncthreads();
    if (best_col_out != nullptr) { best_col_out[row] = best_col[row]; }
    if (best_row_out != nullptr) { best_row_out[row] = best_row[row]; }
}

// ===========================================================================
// 二、落到 ninfer 平面（写入端）
// ===========================================================================

// 码流：平面每 (页, head) 是 [token 0..63][byte 0..CodeLeading)，字节 b 覆盖全局维
// [b*DimsPerByte, b*DimsPerByte + DimsPerByte)；低 bits 位是较低维。
// **payload 里的 row/col 是 slice 内的局部下标**（record 是 per-slice 的 128×128），所以
// 这里必须把全局维 d 折成 d - dim_offset 再去取码 —— 用全局维去索引 payload 会读到隔壁 slice。
// K: row=dim, col=token；V: row=token, col=dim。
template <int Bits, bool Value, int KVHeads, int CodeLeading>
__device__ __forceinline__ void kvarn128_pack_codes_to_plane(std::uint8_t* plane, int head,
                                                             std::int32_t page0,
                                                             std::int32_t page1,
                                                             const std::uint8_t* payload,
                                                             int dim_offset) {
    constexpr int RowBytes           = kKvarnSliceDim * Bits / 8;
    constexpr int DimsPerByte        = 8 / Bits;
    constexpr int SliceBytesPerToken = kKvarnSliceDim * Bits / 8;
    const int first_byte             = dim_offset * Bits / 8;
    for (int i = threadIdx.x; i < kKvarnTileTokens * SliceBytesPerToken; i += blockDim.x) {
        const int t        = i / SliceBytesPerToken;
        const int b        = first_byte + i % SliceBytesPerToken;
        const int dim_local0 = b * DimsPerByte - dim_offset; // slice 内维（0..127）
        std::uint8_t byte  = 0;
#pragma unroll
        for (int k = 0; k < DimsPerByte; ++k) {
            const int dl  = dim_local0 + k;
            const int row = Value ? t : dl;
            const int col = Value ? dl : t;
            int q = 0;
            const std::size_t bo = static_cast<std::size_t>(col) * Bits;
            for (int bit = 0; bit < Bits; ++bit) {
                q |= ((payload[static_cast<std::size_t>(row) * RowBytes + (bo + bit) / 8] >>
                       ((bo + bit) % 8)) &
                      1u)
                     << bit;
            }
            byte |= static_cast<std::uint8_t>(q) << (k * Bits);
        }
        const int page = t >= kKvarnPageTokens ? page1 : page0;
        const int off  = t & (kKvarnPageTokens - 1);
        plane[kvarn_plane_element_offset(CodeLeading, KVHeads, page, head, off, b)] = byte;
    }
}

// scale 平面：P1 页自足（维轴在两页各存一份）。
// sa = best_row*scale、zp = best_row*lo、oa = best_col（槽位映射见 kvarn_layout.h）。
template <bool Value, int KVHeads, int ScaleLeading>
__device__ __forceinline__ void kvarn128_write_scale_plane(
    __half* plane, int head, std::int32_t page0, std::int32_t page1,
    const float (*sa)[kKvarnSliceDim], const float (*zp)[kKvarnSliceDim],
    const float (*oa)[kKvarnSliceDim]) {
    constexpr int PerPage = kKvarnPageTokens * ScaleLeading;
    for (int i = threadIdx.x; i < kKvarnPagesPerTile * PerPage; i += blockDim.x) {
        const int pi   = i / PerPage;
        const int r    = i % PerPage;
        const int p    = r / ScaleLeading;          // 页内 token 下标
        const int j    = r % ScaleLeading;          // 槽
        const int page = pi == 0 ? page0 : page1;
        const int tt   = pi * kKvarnPageTokens + p; // 瓦片内 token 下标
        __half h;
        if constexpr (!Value) {
            if (j < kKvarnKScaleSlotZp) {
                const int d = 4 * p + j; // 全局维（P1 §3：row = d/4、col = d%4）
                h = __float2half_rn(sa[d / kKvarnSliceDim][d % kKvarnSliceDim]);
            } else if (j < kKvarnKScaleSlotOther) {
                const int d = 4 * p + (j - kKvarnKScaleSlotZp);
                h = __float2half_rn(zp[d / kKvarnSliceDim][d % kKvarnSliceDim]);
            } else {
                h = __float2half_rn(oa[j - kKvarnKScaleSlotOther][tt]);
            }
        } else {
            if (j < kKvarnVScaleSlotScale) {
                const int d = 4 * p + j;
                h = __float2half_rn(oa[d / kKvarnSliceDim][d % kKvarnSliceDim]);
            } else if (j < kKvarnVScaleSlotZp) {
                h = __float2half_rn(sa[j - kKvarnVScaleSlotScale][tt]);
            } else {
                h = __float2half_rn(zp[j - kKvarnVScaleSlotZp][tt]);
            }
        }
        plane[kvarn_plane_element_offset(ScaleLeading, KVHeads, page, head, p, j)] = h;
    }
}

// ===========================================================================
// 三、从 ninfer 平面读回（decode）
// ===========================================================================
// **与 encode 共用 kvarn_layout.h 的槽位映射**，所以"写进去读不出来"这类不可能是布局问题。
// 这是路线 B（materialize）与"record 解码 == 主机反量化"对拍的唯一解码实现。
template <int Bits, bool Value, int KVHeads, int CodeLeading, int ScaleLeading>
__device__ __forceinline__ void kvarn128_decode_plane(const std::uint8_t* plane,
                                                      const __half* scale, int head,
                                                      std::int32_t page0, std::int32_t page1,
                                                      __half* out, int dim_offset) {
    constexpr int DimsPerByte = 8 / Bits;
    const int slice           = dim_offset / kKvarnSliceDim;
    for (int i = threadIdx.x; i < kKvarnTileTokens * kKvarnSliceDim; i += blockDim.x) {
        const int t  = i / kKvarnSliceDim;
        const int dl = i % kKvarnSliceDim; // slice 内维
        const int d  = dim_offset + dl;    // 全 head 维（0..255）
        const int page = t >= kKvarnPageTokens ? page1 : page0;
        const int off  = t & (kKvarnPageTokens - 1);
        const int b    = d / DimsPerByte;
        const int by =
            (int)plane[kvarn_plane_element_offset(CodeLeading, KVHeads, page, head, off, b)];
        const int q      = (by >> ((d % DimsPerByte) * Bits)) & ((1 << Bits) - 1);
        float val;
        if constexpr (!Value) {
            const float s = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, kvarn_axis_row(d), kvarn_k_scale_col_scale(d))]);
            const float z = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, kvarn_axis_row(d), kvarn_k_scale_col_zp(d))]);
            const float o = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, off, kvarn_k_scale_col_other(slice))]);
            val = (q * s + z) * o;
        } else {
            const float o = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, kvarn_axis_row(d), kvarn_v_scale_col_other(d))]);
            const float s = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, off, kvarn_v_scale_col_scale(slice))]);
            const float z = __half2float(scale[kvarn_plane_element_offset(
                ScaleLeading, KVHeads, page, head, off, kvarn_v_scale_col_zp(slice))]);
            val = (q * s + z) * o;
        }
        out[t * kKvarnHeadDim + d] = __float2half_rn(val);
    }
}

// ===========================================================================
// 四、整瓦片编解码的 device 主体 —— 引擎的 __global__ 与独立对拍 harness **共用同一份**，
//     杜绝"测的是一份、跑的是一份"（G64 旧路线就是在这里分叉的）。
// ===========================================================================
template <int Bits, int KVHeads>
__device__ __forceinline__ void kvarn128_encode_tile(
    const __half* __restrict__ stage /* [2][KVHeads][128][256] 旋转域 */, int slot, int head,
    std::uint8_t* __restrict__ k_code, __half* __restrict__ k_scale,
    std::uint8_t* __restrict__ v_code, __half* __restrict__ v_scale, std::int32_t page0,
    std::int32_t page1, float* __restrict__ smem_tile /* 16384 floats, 与 quantize 共用 */,
    std::uint8_t* __restrict__ smem_rec, float (*sa)[kKvarnSliceDim], float (*zp)[kKvarnSliceDim],
    float (*oa)[kKvarnSliceDim], float* __restrict__ bc, float* __restrict__ br, int iters) {
    const std::int64_t stage_plane = static_cast<std::int64_t>(kKvarnTileTokens) * kKvarnHeadDim *
                                     KVHeads * kKvarnPagesPerTile;
    for (int which = 0; which < 2; ++which) { // 0=K, 1=V
        for (int slice = 0; slice < kKvarnSlices; ++slice) {
            for (int i = threadIdx.x; i < kKvarnSliceDim * kKvarnSliceDim; i += blockDim.x) {
                const int r     = i / kKvarnSliceDim;
                const int c     = i % kKvarnSliceDim;
                const int token = which ? r : c; // K: row=dim, col=token；V: row=token, col=dim
                const int dim   = which ? c : r;
                smem_tile[i] = __half2float(
                    stage[which * stage_plane +
                          (static_cast<std::int64_t>(slot) * KVHeads + head) * kKvarnTileTokens *
                              kKvarnHeadDim +
                          static_cast<std::int64_t>(token) * kKvarnHeadDim +
                          slice * kKvarnSliceDim + dim]);
            }
            __syncthreads();
            kvarn128_quantize_tile(smem_rec, Bits, iters, smem_tile, bc, br);
            // 三个轴从 record（fp16）取回 —— 平面值与 record 逐位一致，别自己再算一遍
            const int payload_bytes = kKvarnTileTokens * kKvarnSliceDim * Bits / 8;
            const __half* ax        = reinterpret_cast<const __half*>(smem_rec + payload_bytes);
            for (int i = threadIdx.x; i < kKvarnSliceDim; i += blockDim.x) {
                sa[slice][i] = __half2float(ax[i]);
                zp[slice][i] = __half2float(ax[kKvarnSliceDim + i]);
                oa[slice][i] = __half2float(ax[2 * kKvarnSliceDim + i]);
            }
            __syncthreads();
            if (which == 0) {
                kvarn128_pack_codes_to_plane<Bits, false, KVHeads, kvarn_code_leading(Bits)>(
                    k_code, head, page0, page1, smem_rec, slice * kKvarnSliceDim);
            } else {
                kvarn128_pack_codes_to_plane<Bits, true, KVHeads, kvarn_code_leading(Bits)>(
                    v_code, head, page0, page1, smem_rec, slice * kKvarnSliceDim);
            }
            __syncthreads(); // smem_rec 下一轮要被覆写
        }
        if (which == 0) {
            kvarn128_write_scale_plane<false, KVHeads, kKvarnKScaleLeading>(k_scale, head, page0,
                                                                           page1, sa, zp, oa);
        } else {
            kvarn128_write_scale_plane<true, KVHeads, kKvarnVScaleLeading>(v_scale, head, page0,
                                                                          page1, sa, zp, oa);
        }
        __syncthreads();
    }
}

template <int Bits, int KVHeads>
__device__ __forceinline__ void kvarn128_decode_tile_device(
    const std::uint8_t* __restrict__ k_code, const __half* __restrict__ k_scale,
    const std::uint8_t* __restrict__ v_code, const __half* __restrict__ v_scale, int head,
    std::int32_t page0, std::int32_t page1, __half* __restrict__ out_k, __half* __restrict__ out_v) {
    for (int slice = 0; slice < kKvarnSlices; ++slice) {
        kvarn128_decode_plane<Bits, false, KVHeads, kvarn_code_leading(Bits), kKvarnKScaleLeading>(
            k_code, k_scale, head, page0, page1, out_k, slice * kKvarnSliceDim);
        kvarn128_decode_plane<Bits, true, KVHeads, kvarn_code_leading(Bits), kKvarnVScaleLeading>(
            v_code, v_scale, head, page0, page1, out_v, slice * kKvarnSliceDim);
    }
}

} // namespace ninfer::ops

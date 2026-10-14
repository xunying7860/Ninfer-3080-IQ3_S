// =============================================================================
// kvarn128_decode.cuh
//   KVarN G128 · P1 页自足解码器件（页 → 未旋转 bf16 tile）
//   产出：decode-g128/（PC 侧 sm_86 / RTX 3080 实跑验通）
//
// 口径来源（全部只读引用，未改）：
//   * ../P1几何与偏移表-20261013.md §2/§3（leading 定案值 + 页内槽位表）
//   * ../offsets-g128/out/offsets_p1_g128.csv, out/p1_slot_map_g128.csv（页内三段偏移）
//   * ../两道gate结论-20261013.md（gate1：逆旋转照搬 ninfer，精确 2^-4；gate2：量化器口径）
//   * ../参考-beellama核对.md（record 布局 R1..R6：K 转置 / V 不转置、LSB-first、每行整字节对齐）
//
// 定案值（不要再推）：
//   k_code_leading  = 32*kb 字节/token 行      （256 维 × kb bit / 8）
//   v_code_leading  = 32*vb 字节/token 行
//   k_scale_leading = 10 half  （scale_axis 4 + zp_axis 4 + other 2）/ 行
//   v_scale_leading = 8  half  （other 4 + scale_axis 2 + zp_axis 2）/ 行
//   page = 64 token（引擎 kPagedKVPageSize），tile = 128 token = 2 页
//   head_dim = 256 = 2 slice × 128（record_dim）
//
// 折叠口径（消费端铁律）：写端已把行因子 s_row 折进 scale_axis/zp_axis，
//   解码式只能是   deq = (q * scale_axis + zp_axis) * other_axis
//   其中 other_axis = 列因子 s_col。**消费端不得再乘 s_row**。
//
// 旋转：plane 里是旋转域值（相对 beellama 的 D256 store 口径）。喂未旋转 bf16 要
//   做一次 normalized_hadamard_d256_inplace（ninfer 侧，精确 2^-4，自逆）。
//   gate1 已证：结构层与上游逐位相同，配对层是恒等 map[j]=j。
//
// 依赖：CUDA（cuda_runtime.h / cuda_fp16.h / cuda_bf16.h）。除逆旋转外无引擎依赖。
//   逆旋转默认内嵌 ninfer 原文副本（下方 detail 段）；若已 include 引擎真头，
//   定义 KVARN128_EMBED_HADAMARD=0 即可复用引擎的 ninfer::ops::normalized_hadamard_d256_inplace。
// =============================================================================
#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#ifndef KVARN128_EMBED_HADAMARD
#define KVARN128_EMBED_HADAMARD 1
#endif

#if !KVARN128_EMBED_HADAMARD
#include "ops/kv_cache/hadamard_d256.cuh"  // 引擎真头（只读 include）
#endif

namespace kvarn128 {

// -----------------------------------------------------------------------------
// 1. 槽位常量（P1 定案值）
// -----------------------------------------------------------------------------
inline constexpr int kPageTokens    = 64;   // 引擎 kPagedKVPageSize
inline constexpr int kPageShift     = 6;    // 引擎 kPagedKVPageShift
inline constexpr int kPageMask      = kPageTokens - 1;
inline constexpr int kHeadDim       = 256;  // D256
inline constexpr int kSlices        = 2;    // head_slices
inline constexpr int kSliceDim      = 128;  // record_dim
inline constexpr int kKScaleLeading = 10;   // half/token 行
inline constexpr int kVScaleLeading = 8;    // half/token 行
inline constexpr int kMaxBits       = 8;

// 码 plane 每 token 行的字节数（= 256 维 × bits / 8）
__host__ __device__ __forceinline__ constexpr int code_leading_of(int bits) { return 32 * bits; }

// 行内列槽（页内列索引；轴条目语义见 P1几何与偏移表 §3）
// K scale plane（10 列）：0..3 scale_axis[维 d=4*row+col] / 4..7 zp_axis 同式 / 8 slice0 other / 9 slice1 other
inline constexpr int kKScaleAxisCol = 0;
inline constexpr int kKZpAxisCol    = 4;
inline constexpr int kKOtherCol     = 8;   // + slice
inline constexpr int kKZAscRowCount  = 64; // 256 维 / 4 列 = 64 行

// V scale plane（8 列）：0..3 other_axis[维 d=4*row+col] / 4 slice0 scale / 5 slice1 scale / 6 slice0 zp / 7 slice1 zp
inline constexpr int kVOtherCol     = 0;
inline constexpr int kVScaleAxisCol = 4;   // + slice
inline constexpr int kVZpAxisCol    = 6;   // + slice

// 槽位公式（页内行/列）。d = slice*128 + r 是全局维；index 的语义见注释。
__host__ __device__ __forceinline__ int axis_page_row(int d) { return d >> 2; }        // 维轴：行 = d/4
__host__ __device__ __forceinline__ int axis_page_col(int d) { return d & 3; }         // 维轴：列 = d%4
__host__ __device__ __forceinline__ int token_page_row(int token_in_tile) { return token_in_tile & kPageMask; }

// -----------------------------------------------------------------------------
// 2. 页内寻址式 —— 与引擎 src/ops/kernel/paged_kv_address.cuh:47-61 逐字同式
//    offset(元素) = Leading*64*(head + Heads*physical_page) + Leading*page_offset + leading
//    physical_page = block_table[position >> 6]  （消费核只依赖 position 自己那一页，
//    引擎真头 GPU 探针 4352 格 0 失配、128 位置 0 失配 ⇒ 不需要 sibling）
//    plane 形状 PageMajor = {Leading, 64, Heads, physical_pages}
// -----------------------------------------------------------------------------
template <int LeadingExtent, int HeadExtent>
__host__ __device__ __forceinline__ int64_t page_head_offset(int32_t physical_page, int32_t head) {
    return static_cast<int64_t>(LeadingExtent) * kPageTokens *
           (static_cast<int64_t>(head) + static_cast<int64_t>(HeadExtent) * physical_page);
}

template <int LeadingExtent, int HeadExtent>
__host__ __device__ __forceinline__ int64_t element_offset(int32_t physical_page, int32_t head,
                                                          int32_t page_offset, int32_t leading) {
    return page_head_offset<LeadingExtent, HeadExtent>(physical_page, head) +
           static_cast<int64_t>(LeadingExtent) * page_offset + leading;
}

// 运行期 leading/heads 版（本器件按运行期参数工作；引擎可用上面的模板常量版把乘加折掉）
__host__ __device__ __forceinline__ int64_t page_head_offset_rt(int leading, int heads,
                                                               int32_t physical_page, int32_t head) {
    return static_cast<int64_t>(leading) * kPageTokens *
           (static_cast<int64_t>(head) + static_cast<int64_t>(heads) * physical_page);
}
__host__ __device__ __forceinline__ int64_t element_offset_rt(int leading, int heads,
                                                             int32_t physical_page, int32_t head,
                                                             int32_t page_offset, int32_t leading_col) {
    return page_head_offset_rt(leading, heads, physical_page, head) +
           static_cast<int64_t>(leading) * page_offset + leading_col;
}
__host__ __device__ __forceinline__ int64_t element_offset_rt(const int32_t* block_table, int leading,
                                                             int heads, int32_t head, int32_t position,
                                                             int32_t leading_col) {
    const int32_t phys = block_table[position >> kPageShift];
    return element_offset_rt(leading, heads, phys, head, position & kPageMask, leading_col);
}

// -----------------------------------------------------------------------------
// 3. 位串（连续位流、位内 LSB-first、每行整字节对齐）
// -----------------------------------------------------------------------------
__host__ __device__ __forceinline__ uint32_t load_bits_lsb(const uint8_t* __restrict__ p, int bit_off, int n) {
    const int byte = bit_off >> 3;
    const int sh   = bit_off & 7;
    uint32_t w = static_cast<uint32_t>(p[byte]);
    if (sh + n > 8) w |= static_cast<uint32_t>(p[byte + 1]) << 8;   // 只在真的跨字节时才读第二字节
    return (w >> sh) & ((1u << n) - 1u);
}

__host__ __device__ __forceinline__ float h2f(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// -----------------------------------------------------------------------------
// 4. 解码原语 —— 页内单元素（旋转域 f32）
//    code_page  : 指向「本页」码 plane 的第 0 行（即 plane 基址 + Leading*64*(head+Heads*phys)）
//    scale_page : 指向「本页」scale plane 的第 0 行（FP16）
//    token_local: 页内行 0..63
//    d          : 全局维 0..255，d = slice*128 + r
//    IsValue    : false = K（scale/zp 是维轴、other 是 token 轴）
//                 true  = V（other 是维轴、scale/zp 是 token 轴）
// -----------------------------------------------------------------------------
template <bool IsValue>
__device__ __forceinline__ float decode_element(const uint8_t* __restrict__ code_page,
                                                         int code_leading, int bits,
                                                         const uint16_t* __restrict__ scale_page,
                                                         int scale_leading, int token_local, int d) {
    const uint32_t q = load_bits_lsb(code_page + static_cast<int64_t>(code_leading) * token_local,
                                     d * bits, bits);
    const int slice = d >> 7;   // d / kSliceDim

    if (!IsValue) {  // ---- K ----
        const int arow = axis_page_row(d);
        const int acol = axis_page_col(d);
        // scale_axis / zp_axis：每维 d 一条（行 = d/4，列 = d%4 与 4+d%4）
        const float sa = h2f(scale_page[static_cast<int64_t>(scale_leading) * arow + kKScaleAxisCol + acol]);
        const float zp = h2f(scale_page[static_cast<int64_t>(scale_leading) * arow + kKZpAxisCol + acol]);
        // other_axis：每 token 一条，两 slice 各一份（列 = 8 + slice）
        const float so = h2f(scale_page[static_cast<int64_t>(scale_leading) * token_local + kKOtherCol + slice]);
        return __fmul_rn(__fadd_rn(__fmul_rn(static_cast<float>(q), sa), zp), so);
    } else {         // ---- V ----
        const int arow = axis_page_row(d);
        const int acol = axis_page_col(d);
        // other_axis：每维 d 一条（行 = d/4，列 = d%4）
        const float so = h2f(scale_page[static_cast<int64_t>(scale_leading) * arow + kVOtherCol + acol]);
        // scale_axis / zp_axis：每 token 一条，两 slice 各一份（列 = 4+slice / 6+slice）
        const float sa = h2f(scale_page[static_cast<int64_t>(scale_leading) * token_local + kVScaleAxisCol + slice]);
        const float zp = h2f(scale_page[static_cast<int64_t>(scale_leading) * token_local + kVZpAxisCol + slice]);
        return __fmul_rn(__fadd_rn(__fmul_rn(static_cast<float>(q), sa), zp), so);
    }
}

// -----------------------------------------------------------------------------
// 5. 逆旋转 —— normalized Hadamard-256（ninfer 侧，精确 2^-4，自逆）
// -----------------------------------------------------------------------------
#if KVARN128_EMBED_HADAMARD
namespace detail {
// ---- 逐字复制自 ninfer engine/ninfer-all-kvarn/src/ops/kv_cache/hadamard_d256.cuh
//      sha256 3d00813a0a3c112484ef5a12c0011ec7ee8d0848aa17fadccaf9fbdeb0b14fa7
//      契约：一个 warp 拥有一整行 D256；lane l 持维度 l + 32*r，r = 0..7。
//      只取本器件需要的两个函数，其余（h64/h4 片段）引擎按需自取。
template <int Columns>
__device__ __forceinline__ void hadamard_d32_columns_inplace(float (&values)[Columns], int lane) {
    constexpr unsigned FullMask = 0xffffffffu;
#pragma unroll
    for (int stride = 1; stride <= 16; stride <<= 1) {
#pragma unroll
        for (int column = 0; column < Columns; ++column) {
            const float value = values[column];
            const float peer  = __shfl_xor_sync(FullMask, value, stride);
            values[column] = (lane & stride) == 0 ? __fadd_rn(value, peer) : __fsub_rn(peer, value);
        }
    }
}
__device__ __forceinline__ void normalized_hadamard_d256_inplace(float (&values)[8], int lane) {
    hadamard_d32_columns_inplace(values, lane);
#pragma unroll
    for (int span = 1; span < 8; span <<= 1) {
#pragma unroll
        for (int base = 0; base < 8; base += 2 * span) {
#pragma unroll
            for (int offset = 0; offset < span; ++offset) {
                const float low              = values[base + offset];
                const float high             = values[base + offset + span];
                values[base + offset]        = __fadd_rn(low, high);
                values[base + offset + span] = __fsub_rn(low, high);
            }
        }
    }
#pragma unroll
    for (float& value : values) { value = __fmul_rn(value, 0x1p-4f); }
}
}  // namespace detail
#endif

// 逆旋转（自逆）：作用于 lane 持有的 8 个寄存器（维度 = lane + 32*r）
__device__ __forceinline__ void inverse_rotate_d256_row(float (&vals)[8], int lane) {
#if KVARN128_EMBED_HADAMARD
    detail::normalized_hadamard_d256_inplace(vals, lane);
#else
    ninfer::ops::normalized_hadamard_d256_inplace(vals, lane);
#endif
}

// -----------------------------------------------------------------------------
// 6. warp 级行解码：lane l 负责维度 l + 32*r（r = 0..7）
// -----------------------------------------------------------------------------
template <bool IsValue>
__device__ __forceinline__ void decode_row_rotated(const uint8_t* __restrict__ code_page, int code_leading,
                                                   int bits, const uint16_t* __restrict__ scale_page,
                                                   int scale_leading, int token_local, int lane,
                                                   float (&vals)[8]) {
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        vals[r] = decode_element<IsValue>(code_page, code_leading, bits, scale_page, scale_leading,
                                          token_local, lane + 32 * r);
    }
}

__device__ __forceinline__ void pack_bf16_row(const float (&vals)[8], int lane,
                                              __nv_bfloat16* __restrict__ out_row) {
#pragma unroll
    for (int r = 0; r < 8; ++r) out_row[lane + 32 * r] = __float2bfloat16_rn(vals[r]);
}

// 单 token 全流程：解码 →（可选）逆旋转 → bf16 落 out_row[256]
template <bool IsValue>
__device__ __forceinline__ void decode_token_to_bf16(const uint8_t* __restrict__ code_page, int code_leading,
                                                     int bits, const uint16_t* __restrict__ scale_page,
                                                     int scale_leading, int token_local, int lane,
                                                     bool inverse_rotate,
                                                     __nv_bfloat16* __restrict__ out_row) {
    float vals[8];
    decode_row_rotated<IsValue>(code_page, code_leading, bits, scale_page, scale_leading, token_local, lane,
                                vals);
    if (inverse_rotate) inverse_rotate_d256_row(vals, lane);
    pack_bf16_row(vals, lane, out_row);
}

// -----------------------------------------------------------------------------
// 7. 整页驱动（示例/可直接改造成引擎的 staging 核）
//    blockDim.x = 32*Warps 个线程；每个 warp 负责 token = warp_id, warp_id+Warps, ...
//    输出 out_tile[token_local * 256 + d]（token-major，d = slice*128 + r）。
//    valid 之外的 token 由调用方掩码（本函数不写零，避免掩盖错误）。
//    out_rot 非空时另存旋转域 f32（仅供对拍/诊断，生产可传 nullptr）。
// -----------------------------------------------------------------------------
template <bool IsValue, int Warps>
__device__ __forceinline__ void decode_page_bf16(const uint8_t* __restrict__ code_page, int code_leading,
                                                 int bits, const uint16_t* __restrict__ scale_page,
                                                 int scale_leading, bool inverse_rotate,
                                                 __nv_bfloat16* __restrict__ out_tile,
                                                 float* __restrict__ out_rot) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int t = warp; t < kPageTokens; t += Warps) {
        float vals[8];
        decode_row_rotated<IsValue>(code_page, code_leading, bits, scale_page, scale_leading, t, lane, vals);
        if (out_rot != nullptr) {
#pragma unroll
            for (int r = 0; r < 8; ++r) out_rot[static_cast<int64_t>(t) * kHeadDim + lane + 32 * r] = vals[r];
        }
        if (inverse_rotate) inverse_rotate_d256_row(vals, lane);
        __nv_bfloat16* row = out_tile + static_cast<int64_t>(t) * kHeadDim;
        pack_bf16_row(vals, lane, row);
    }
}

}  // namespace kvarn128

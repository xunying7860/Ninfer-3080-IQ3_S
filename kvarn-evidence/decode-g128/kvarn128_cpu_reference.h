// =============================================================================
// kvarn128_cpu_reference.h
//   P1 页解码的「主机参照实现」（纯 C++17，无 CUDA 头、无引擎头）。
//
// 本文件是 kvarn128_decode.cuh 的**独立第二实现**：槽位公式、寻址式、位串读法、
// 折叠口径、逆旋转都按文档规则从零重写（不共享常量、不调 device 函数），
// 用于与 device 实现逐元素对拍（判据：0 差或 ≤1e-6）。
//
// 规则出处：../P1几何与偏移表-20261013.md §3、../参考-beellama核对.md R1/R5、
//           ../两道gate结论-20261013.md §gate1（逆旋转 = ninfer normalized_hadamard_d256）。
// =============================================================================
#pragma once

#include <cstdint>
#include <cstring>
#include <cmath>

namespace kvarn128_cpu {

// ---- 定案常量（独立写死，与 .cuh 不共享）----
inline constexpr int kPageTokens    = 64;
inline constexpr int kPageShift     = 6;
inline constexpr int kHeadDim       = 256;
inline constexpr int kSlices        = 2;
inline constexpr int kSliceDim      = 128;
inline constexpr int kKScaleLeading = 10;
inline constexpr int kVScaleLeading = 8;

inline constexpr int k_code_leading(int bits) { return 32 * bits; }

// ---- fp16 → f32（标量，逐位；与 oracle 的 f16_to_f32 同语义）----
inline float f16_to_f32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = sign;
        else {
            int32_t ee = 127 - 15 + 1;
            uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; --ee; }
            mm &= 0x3FFu;
            u = sign | (static_cast<uint32_t>(ee) << 23) | (mm << 13);
        }
    } else if (e == 0x1Fu) {
        u = sign | 0x7F800000u | (m << 13);
    } else {
        u = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// ---- 位串：连续位流、位内 LSB-first ----
inline uint32_t get_bits_lsb(const uint8_t* p, long long bit_off, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; ++i) {
        const long long b = bit_off + i;
        v |= static_cast<uint32_t>((p[b >> 3] >> (b & 7)) & 1u) << i;
    }
    return v;
}

// ---- 页内寻址式（照抄引擎 paged_kv_address.cuh:47-61）----
inline long long element_offset(int leading, int heads, int physical_page, int head, int page_offset,
                               int leading_col) {
    return static_cast<long long>(leading) * kPageTokens *
               (static_cast<long long>(head) + static_cast<long long>(heads) * physical_page) +
           static_cast<long long>(leading) * page_offset + leading_col;
}

// ---- K 槽位（P1几何与偏移表 §3.2）----
//   维轴：scale_axis 列 0..3 / zp_axis 列 4..7，行 = d/4，列 = d%4（d = slice*128 + r）
//   token 轴：other_axis 列 8 + slice，行 = token%64
inline float k_scale_axis(const uint16_t* page, int d)  { return f16_to_f32(page[10 * (d >> 2) + 0 + (d & 3)]); }
inline float k_zp_axis(const uint16_t* page, int d)     { return f16_to_f32(page[10 * (d >> 2) + 4 + (d & 3)]); }
inline float k_other_axis(const uint16_t* page, int tok, int slice) { return f16_to_f32(page[10 * (tok & 63) + 8 + slice]); }

// ---- V 槽位（§3.3）----
//   维轴：other_axis 列 0..3，行 = d/4，列 = d%4
//   token 轴：scale_axis 列 4+slice / zp_axis 列 6+slice，行 = token%64
inline float v_other_axis(const uint16_t* page, int d) { return f16_to_f32(page[8 * (d >> 2) + 0 + (d & 3)]); }
inline float v_scale_axis(const uint16_t* page, int tok, int slice) { return f16_to_f32(page[8 * (tok & 63) + 4 + slice]); }
inline float v_zp_axis(const uint16_t* page, int tok, int slice)    { return f16_to_f32(page[8 * (tok & 63) + 6 + slice]); }

// ---- 单元素解码（旋转域）----
//   折叠口径：deq = (q*scale_axis + zp_axis) * other_axis，不再乘行因子。
inline float decode_element(bool is_value, const uint8_t* code_page, int bits, const uint16_t* scale_page,
                            int token_local, int d) {
    const int leading = 32 * bits;
    const uint32_t q = get_bits_lsb(code_page + static_cast<long long>(leading) * token_local, d * bits, bits);
    const int slice = d / kSliceDim;
    float sa, zp, so;
    if (!is_value) {                       // K
        sa = k_scale_axis(scale_page, d);
        zp = k_zp_axis(scale_page, d);
        so = k_other_axis(scale_page, token_local, slice);
    } else {                               // V
        so = v_other_axis(scale_page, d);
        sa = v_scale_axis(scale_page, token_local, slice);
        zp = v_zp_axis(scale_page, token_local, slice);
    }
    return (static_cast<float>(q) * sa + zp) * so;
}

// ---- 逐 token 解出 D256（旋转域 f32）----
inline void decode_token_rotated(bool is_value, const uint8_t* code_page, int bits,
                                 const uint16_t* scale_page, int token_local, float* out256) {
    for (int d = 0; d < kHeadDim; ++d)
        out256[d] = decode_element(is_value, code_page, bits, scale_page, token_local, d);
}

// ---- 逆旋转：normalized Hadamard-256（自逆，精确 1/16）----
//   与 device 侧 ninfer 版结构同（stride 1,2,4,...,128 递增；标准 Sylvester 符号），
//   逐 stage 的蝶形互相独立 ⇒ 舍入与寄存器/内存布局无关。
inline void inverse_rotate_d256(float* v) {
    for (int s = 1; s < kHeadDim; s <<= 1) {
        for (int i = 0; i < kHeadDim; i += 2 * s) {
            for (int j = i; j < i + s; ++j) {
                const float a = v[j], b = v[j + s];
                v[j] = a + b;
                v[j + s] = a - b;
            }
        }
    }
    for (int i = 0; i < kHeadDim; ++i) v[i] *= 0.0625f;
}

}  // namespace kvarn128_cpu

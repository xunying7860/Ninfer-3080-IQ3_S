// llama_oracle.cpp — 对照 oracle：llama.cpp 原生 KV 量化档 vs beellama KVarN
// ---------------------------------------------------------------------------
// 目的：在同一份真实 tile、同一 K/V 分工（K: tok=col / V: tok=row）下，把
//       llama.cpp 原生 KV 量化档（q4_0/q4_1/q5_0/q5_1/q8_0/q4_K）与 KVarN
//       （K4V4/K4V3/K4V2/K3V3/K6V6）以及四个消融档放在一起对拍。
//
// 口径来源（逐条给行号，全部只读 ref-beellama/，未改任何引擎源码）：
//   ggml-quants.c:113-148   quantize_row_q4_0_ref
//   ggml-quants.c:150-185   quantize_row_q4_1_ref
//   ggml-quants.c:187-229   quantize_row_q5_0_ref
//   ggml-quants.c:231-273   quantize_row_q5_1_ref
//   ggml-quants.c:532-555   quantize_row_q8_0_ref
//   ggml-quants.c:715-733   dequantize_row_q4_0
//   ggml-quants.c:735-754   dequantize_row_q4_1
//   ggml-quants.c:756-780   dequantize_row_q5_0
//   ggml-quants.c:782-807   dequantize_row_q5_1
//   ggml-quants.c:934-948   dequantize_row_q8_0
//   ggml-quants.c:1838-1908 quantize_row_q4_K_ref
//   ggml-quants.c:1910-1932 dequantize_row_q4_K
//   ggml-quants.c:1002-1007 nearest_int
//   ggml-quants.c:1180-1259 make_qkx2_quants
//   ggml-quants.c:1261-1268 get_scale_min_k4
//   ggml-quants.c:28-37     best_index_int8（IQ4_NL 的码本最近邻）
//   ggml-quants.c:20        GROUP_MAX_EPS 1e-15f
//   ggml-quants.c:5389-5498 quantize_row_iq4_nl_impl（super=32/block=32 ⇒ 单块分支）
//   ggml-quants.c:5500-5520 quantize_iq4_nl（带 imatrix 权重，本工具不用）
//   ggml-quants.c:5522-5536 quantize_row_iq4_nl_ref（from_float_ref，ntry=-1/无权重）
//   ggml-quants.c:3148-3164 dequantize_row_iq4_nl
//   ggml-quants.c:376-415   quantize_row_q3_0_ref
//   ggml-quants.c:417-456   quantize_row_q3_1_ref
//   ggml-quants.c:854-873   dequantize_row_q3_0
//   ggml-quants.c:875-895   dequantize_row_q3_1
//   ggml-common.h:1203-1206 kvalues_iq4nl[16] 非线性码本 {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}
//   ggml-common.h:530-536  QK4_NL=32 / block_iq4_nl {half d; u8 qs[16]}    = 18 B
//   ggml-common.h:291-298  QK3_0=32 / block_q3_0 {half d; u8 qh[4]; u8 qs[8]} = 14 B
//   ggml-common.h:300-312  QK3_1=32 / block_q3_1 {half d,m; u8 qh[4]; u8 qs[8]} = 16 B
//   ggml.c:912-919         type_traits[IQ4_NL]（blck=QK4_NL, from_float_ref=quantize_row_iq4_nl_ref）
//   ggml.c:709-724         type_traits[Q3_0]/[Q3_1]
//   ggml-common.h:212-217  QK4_0=32 / block_q4_0 {half d; u8 qs[16]}      = 18 B
//   ggml-common.h:219-230  QK4_1=32 / block_q4_1 {half d,m; u8 qs[16]}    = 20 B
//   ggml-common.h:247-253  QK5_0=32 / {half d; u8 qh[4]; u8 qs[16]}       = 22 B
//   ggml-common.h:255-267  QK5_1=32 / {half d,m; u8 qh[4]; u8 qs[16]}     = 24 B
//   ggml-common.h:335-340  QK8_0=32 / {half d; i8 qs[32]}                 = 34 B
//   ggml-common.h:89-90    QK_K=256 / K_SCALE_SIZE=12
//   ggml-common.h:411-422  block_q4_K {half d,dmin; u8 scales[12]; u8 qs[128]} = 144 B
//   ggml-cuda/cpy-utils.cuh:17-44   quantize_f32_q4_0_block   (KV cache 实际写入口径)
//   ggml-cuda/cpy-utils.cuh:46-72   quantize_f32_q4_1_block
//   ggml-cuda/cpy-utils.cuh:74-104  quantize_f32_q5_0_block
//   ggml-cuda/cpy-utils.cuh:106-135 quantize_f32_q5_1_block
//   ggml-cuda/cpy-utils.cuh:352-369 quantize_f32_q8_0_block
//   ggml-cuda/cpy.cu:250-297        f32->q8_0/q4_0/q4_1 的 cpy 入口
//   ggml-cuda/cpy.cu:659-731        类型分派表
//   common/arg.cpp:341-357          kv_cache_types 允许集合（**q4_K 不在其中**）
//   common/arg.cpp:380-389          + kvarn2..kvarn8 伪类型
//   common/common.h:344-345         cache_type_k = cache_type_v = F16（默认同方案）
//   llama-context.cpp:4692-4695     MLA/DeepSeek4 强制 K/V 同方案
//   llama-context.cpp:4708-4728     K/V 量化档的 blck_size 必须整除 head_dim
//
// 旋转口径（beellama D256）：make_tiles.py 逐行 —— 每 128 片做归一化 WHT128
//   (×1/sqrt(128))，再跨 2 片做 Hadamard(×1/sqrt(2))。两段都是自逆正交变换，
//   所以 R^-1 = R（先跨片再逐片 WHT）。本文件用 unrotate 把入库的（已旋转）
//   tile 还原成 raw，从而能在同一份原始数据上比较"旋转档"与"不旋转档"。
//
// 子命令：
//   selftest   旋转往返 / unrotate 精度 / llama 码本往返 / 字节账自检
//   bytes      各档每 token·head-slice 字节账（含 K+V 的 D256 口径）
//   quant      量化→反量化误差（真实 tile 或合成数据）
//
// quant 选项：
//   --mode MODE   kvarn | llama | llama-rot | rot-only | bal-only | neither
//   --bits-k N --bits-v N      KVarN 档位宽（kvarn / rot-only / bal-only / neither）
//   --lk T --lv T              llama 档类型（llama / llama-rot），T∈q4_0,q4_1,iq4_nl,q5_0,q5_1,q8_0,q3_0,q3_1,q4_K
//   --group N                  token 组长度（128 / 64）
//   --iters N                  Sinkhorn 迭代（bal-only 默认 16；kvarn 默认 16）
//   --tiles PATH               真实 tile 二进制
//   --gauss / --outlier        合成数据分布
//   --seed N                   合成种子（默认 20261013）
//   --stage-f16                输入先过一遍 fp16（beellama stage 口径）
//   --ref-h16                  误差参照取 h16(码域 tile)（复刻旧 oracle 的输出口径）
//   --tag STR                  输出里的数据标签（默认取自文件名）
// 编译（容器内）：g++ -O2 -std=c++17 -o llama_oracle llama_oracle.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <climits>
#include <cctype>
#include <vector>
#include <string>
#include <array>
#include <map>
#include <algorithm>
#include <random>

// ===========================================================================
// 0. fp16：__float2half_rn 的标量等价（照抄 bee_oracle.cpp:49-100，未改）
// ===========================================================================
static uint16_t f32_to_f16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t exp  = (u >> 23) & 0xFFu;
    const uint32_t m    = u & 0x7FFFFFu;

    if (exp == 0xFFu) {
        if (m == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7E00u);
    }
    const int32_t e = (int32_t)exp - 127;
    if (e > 15) return (uint16_t)(sign | 0x7C00u);
    if (e >= -14) {
        const uint32_t mant = m >> 13;
        const uint32_t rest = m & 0x1FFFu;
        uint16_t base = (uint16_t)(sign | ((uint32_t)(e + 15) << 10) | mant);
        if (rest > 0x1000u || (rest == 0x1000u && (mant & 1u))) {
            base = (uint16_t)(base + 1u);
        }
        return base;
    }
    if (e < -25) return (uint16_t)sign;
    const uint32_t sig = m | 0x800000u;
    const int j = -(e + 1);
    uint32_t q = sig >> j;
    const uint32_t rem = sig & ((1u << j) - 1u);
    const uint32_t half = 1u << (j - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;
    return (uint16_t)(sign | q);
}
static float f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    uint32_t u;
    if (e == 0) {
        if (m == 0) u = sign;
        else {
            int32_t ee = 127 - 15 + 1;
            uint32_t mm = m;
            while (!(mm & 0x400u)) { mm <<= 1; --ee; }
            mm &= 0x3FFu;
            u = sign | ((uint32_t)ee << 23) | (mm << 13);
        }
    } else if (e == 0x1Fu) u = sign | 0x7F800000u | (m << 13);
    else u = sign | ((e + 127 - 15) << 23) | (m << 13);
    float f; std::memcpy(&f, &u, 4);
    return f;
}
static inline float h16(float x) { return f16_to_f32(f32_to_f16_rn(x)); }

// ===========================================================================
// 1. KVarN 布局 / 位打包（照抄 bee_oracle.cpp:105-188，未改）
// ===========================================================================
struct TileLayout {
    size_t k_payload_off, k_payload_bytes, k_s_col_off, k_zp_off, k_s_row_off;
    size_t v_payload_off, v_payload_bytes, v_s_col_off, v_s_row_off, v_zp_off;
    size_t tile_bytes;
};
static size_t bee_packed_bytes(int n_values, int bits) {
    return (size_t(n_values) * size_t(bits) + 7) / 8;
}
static size_t align_up(size_t v, size_t a) { return (v + a - 1) / a * a; }

// llama-kvarn.cpp:634
static TileLayout bee_layout(int head_dim, int group, int key_bits, int value_bits) {
    TileLayout L{}; size_t off = 0;
    L.k_payload_off = off; L.k_payload_bytes = bee_packed_bytes(head_dim * group, key_bits); off += L.k_payload_bytes;
    L.k_s_col_off = off; off += size_t(head_dim) * 2;
    L.k_zp_off    = off; off += size_t(head_dim) * 2;
    L.k_s_row_off = off; off += size_t(group) * 2;
    L.v_payload_off = off; L.v_payload_bytes = bee_packed_bytes(group * head_dim, value_bits); off += L.v_payload_bytes;
    L.v_s_col_off = off; off += size_t(head_dim) * 2;
    L.v_s_row_off = off; off += size_t(group) * 2;
    L.v_zp_off    = off; off += size_t(group) * 2;
    L.tile_bytes = align_up(off, 8);
    return L;
}
struct RecordLayout { int rows, cols; size_t payload_bytes, scale_off, zp_off, other_off, record_bytes; };
// llama-kvarn.cpp:673
static RecordLayout make_record_layout(int record_dim, int bits, bool value) {
    RecordLayout R{};
    R.rows = value ? 128 : record_dim;
    R.cols = value ? record_dim : 128;
    R.payload_bytes = bee_packed_bytes(R.rows * R.cols, bits);
    R.scale_off = R.payload_bytes;
    R.zp_off    = R.scale_off + size_t(R.rows) * 2;
    R.other_off = R.zp_off    + size_t(R.rows) * 2;
    R.record_bytes = R.other_off + size_t(R.cols) * 2;
    return R;
}
static size_t record_tile_bytes(int record_dim, int head_slices, int kb, int vb) {
    return size_t(head_slices) * (make_record_layout(record_dim, kb, false).record_bytes
                                + make_record_layout(record_dim, vb, true ).record_bytes);
}

static void bee_pack_bits(const uint8_t * values, int n_values, int bits, uint8_t * dst) {
    std::memset(dst, 0, bee_packed_bytes(n_values, bits));
    const uint8_t mask = uint8_t((1u << bits) - 1u);
    for (int i = 0; i < n_values; ++i) {
        const uint8_t value = uint8_t(values[i] & mask);
        const size_t bit_offset = size_t(i) * size_t(bits);
        for (int bit = 0; bit < bits; ++bit) {
            const size_t dst_bit = bit_offset + size_t(bit);
            dst[dst_bit / 8] |= uint8_t(((value >> bit) & 1u) << (dst_bit % 8));
        }
    }
}
static uint8_t bee_unpack_bits_value(const uint8_t * src, int index, int bits) {
    uint8_t value = 0;
    const size_t bit_offset = size_t(index) * size_t(bits);
    for (int bit = 0; bit < bits; ++bit) {
        const size_t src_bit = bit_offset + size_t(bit);
        value |= uint8_t(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }
    return value;
}

// ===========================================================================
// 2. Sinkhorn + KVarN 量化/反量化（照抄 bee_oracle.cpp:195-374，未改）
// ===========================================================================
struct Sink { std::vector<float> s_col, s_row; float best_imbalance; int best_iter; };

static inline float std_of(const float * p, int n, int stride, const float * s_col, const float * s_row,
                           int row_index, bool is_col) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float v = is_col ? p[i * stride] / (s_col[row_index] * s_row[i])
                               : p[i * stride] / (s_col[i] * s_row[row_index]);
        sum += v; sum_sq += v * v;
    }
    const float mean = sum / float(n);
    return sqrtf(fmaxf((sum_sq - float(n) * mean * mean) / float(n - 1), 0.0f));
}

static void sinkhorn_gpu(const float * tile, int R, int C, int iters, Sink & out) {
    std::vector<float> log_s_col(C, 0.f), log_s_row(R, 0.f), s_col(C, 1.f), s_row(R, 1.f);
    std::vector<float> best_col(C, 1.f), best_row(R, 1.f);
    std::vector<float> col_std(C), row_std(R);

    auto update_best = [&](int iter) {
        float cmin = FLT_MAX, cmax = 0.f, rmin = FLT_MAX, rmax = 0.f;
        for (int c = 0; c < C; ++c) { cmin = fminf(cmin, col_std[c]); cmax = fmaxf(cmax, col_std[c]); }
        for (int r = 0; r < R; ++r) { rmin = fminf(rmin, row_std[r]); rmax = fmaxf(rmax, row_std[r]); }
        const float imb = cmax / fmaxf(cmin, 1e-8f) + rmax / fmaxf(rmin, 1e-8f);
        if (imb <= out.best_imbalance) {
            out.best_imbalance = imb; out.best_iter = iter;
            best_col = s_col; best_row = s_row;
        }
    };

    for (int c = 0; c < C; ++c) col_std[c] = std_of(tile + c, R, C, s_col.data(), s_row.data(), c, true);
    for (int r = 0; r < R; ++r) row_std[r] = std_of(tile + r * C, C, 1, s_col.data(), s_row.data(), r, false);
    out.best_imbalance = FLT_MAX; out.best_iter = -1;
    update_best(-1);

    for (int iter = 0; iter < iters; ++iter) {
        for (int c = 0; c < C; ++c) {
            const float col = fminf(fmaxf(col_std[c], 1e-3f), 1e3f);
            log_s_col[c] = fminf(fmaxf(log_s_col[c] + logf(col), -0.3f), 10.0f);
            s_col[c] = expf(log_s_col[c]);
        }
        for (int r = 0; r < R; ++r) row_std[r] = std_of(tile + r * C, C, 1, s_col.data(), s_row.data(), r, false);
        for (int r = 0; r < R; ++r) {
            const float row = fminf(fmaxf(row_std[r], 1e-3f), 1e3f);
            log_s_row[r] = fminf(fmaxf(log_s_row[r] + logf(row), -0.3f), 10.0f);
            s_row[r] = expf(log_s_row[r]);
        }
        for (int r = 0; r < R; ++r) row_std[r] = std_of(tile + r * C, C, 1, s_col.data(), s_row.data(), r, false);
        for (int c = 0; c < C; ++c) col_std[c] = std_of(tile + c, R, C, s_col.data(), s_row.data(), c, true);
        update_best(iter);
    }
    out.s_col = best_col; out.s_row = best_row;
}

struct QuantMeta { std::vector<float> best_col, best_row; std::vector<uint8_t> q; float best_imbalance; int best_iter; };

static void bee_quant_tile(const float * tile, int rows, int cols, int bits, int iters,
                           std::vector<uint8_t> & record, QuantMeta * meta) {
    Sink S;
    sinkhorn_gpu(tile, rows, cols, iters, S);

    const int qmax = (1 << bits) - 1;
    std::vector<uint8_t> q(size_t(rows) * cols);
    std::vector<float> scale_axis(rows), zp_axis(rows);

    for (int r = 0; r < rows; ++r) {
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            const float x = tile[size_t(r) * cols + c] / (S.s_col[c] * S.s_row[r]);
            lo = fminf(lo, x); hi = fmaxf(hi, x);
        }
        const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
        for (int c = 0; c < cols; ++c) {
            const float x = tile[size_t(r) * cols + c] / (S.s_col[c] * S.s_row[r]);
            q[size_t(r) * cols + c] = (uint8_t) fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float) qmax);
        }
        scale_axis[r] = h16(S.s_row[r] * scale);
        zp_axis[r]    = h16(S.s_row[r] * lo);
    }

    const size_t payload_bytes = bee_packed_bytes(rows * cols, bits);
    record.assign(payload_bytes + size_t(2 * rows + cols) * 2, 0);
    bee_pack_bits(q.data(), rows * cols, bits, record.data());
    for (int r = 0; r < rows; ++r) {
        uint16_t h;
        h = f32_to_f16_rn(scale_axis[r]); std::memcpy(&record[payload_bytes + size_t(r) * 2], &h, 2);
        h = f32_to_f16_rn(zp_axis[r]);    std::memcpy(&record[payload_bytes + size_t(rows) * 2 + size_t(r) * 2], &h, 2);
    }
    for (int c = 0; c < cols; ++c) {
        const uint16_t h = f32_to_f16_rn(S.s_col[c]);
        std::memcpy(&record[payload_bytes + size_t(2 * rows) * 2 + size_t(c) * 2], &h, 2);
    }
    if (meta) { meta->best_col = S.s_col; meta->best_row = S.s_row; meta->q = q;
                meta->best_imbalance = S.best_imbalance; meta->best_iter = S.best_iter; }
}

static void bee_dequant_tile(const uint8_t * record, int rows, int cols, int bits, float * tile) {
    const size_t payload_bytes = bee_packed_bytes(rows * cols, bits);
    for (int r = 0; r < rows; ++r) {
        uint16_t hs, hz;
        std::memcpy(&hs, record + payload_bytes + size_t(r) * 2, 2);
        std::memcpy(&hz, record + payload_bytes + size_t(rows) * 2 + size_t(r) * 2, 2);
        const float scale = f16_to_f32(hs), zp = f16_to_f32(hz);
        for (int c = 0; c < cols; ++c) {
            uint16_t ho;
            std::memcpy(&ho, record + payload_bytes + size_t(2 * rows) * 2 + size_t(c) * 2, 2);
            const float other = f16_to_f32(ho);
            const uint8_t q = bee_unpack_bits_value(record, r * cols + c, bits);
            tile[size_t(r) * cols + c] = (float(q) * scale + zp) * other;
        }
    }
}

// ===========================================================================
// 3. D256 旋转（beellama 口径；make_tiles.py 逐行）
//    自逆：forward = cross(WHT(a),WHT(b))；inverse = WHT(cross(a,b))，两者都是 R
// ===========================================================================
static const float INV_SQRT_128 = 0.08838834764831845f;   // 1/sqrt(128)
static const float INV_SQRT_2   = 0.7071067811865475f;    // 1/sqrt(2)

// 归一化 WHT128（自逆）。与 make_tiles.py:wht128 的 butterfly 分组一致。
static void wht128_norm(float * y) {
    int stride = 1;
    while (stride < 128) {
        for (int base = 0; base < 128; base += 2 * stride) {
            for (int i = 0; i < stride; ++i) {
                const float a = y[base + i], b = y[base + stride + i];
                y[base + i]          = a + b;
                y[base + stride + i] = a - b;
            }
        }
        stride *= 2;
    }
    for (int i = 0; i < 128; ++i) y[i] *= INV_SQRT_128;
}
// forward：两片各自 WHT 后跨片 Hadamard（= make_tiles.py:53-63 rotate_d256）
static void rot_pair_fwd(const float * a, const float * b, float * o0, float * o1) {
    float p[128], q[128];
    std::memcpy(p, a, sizeof(p)); std::memcpy(q, b, sizeof(q));
    wht128_norm(p); wht128_norm(q);
    for (int i = 0; i < 128; ++i) { o0[i] = (p[i] + q[i]) * INV_SQRT_2; o1[i] = (p[i] - q[i]) * INV_SQRT_2; }
}
// inverse：先跨片再逐片 WHT（与 forward 同式，因为两段都自逆）
static void rot_pair_inv(const float * o0, const float * o1, float * a, float * b) {
    float p[128], q[128];
    for (int i = 0; i < 128; ++i) { p[i] = (o0[i] + o1[i]) * INV_SQRT_2; q[i] = (o0[i] - o1[i]) * INV_SQRT_2; }
    wht128_norm(p); wht128_norm(q);
    std::memcpy(a, p, sizeof(p)); std::memcpy(b, q, sizeof(q));
}

// ===========================================================================
// 4. llama.cpp 原生量化档（ggml-quants.c 逐行）
// ===========================================================================
static inline int nearest_int(float fval) {                 // ggml-quants.c:1002
    float val = fval + 12582912.f;
    int i; std::memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}

// ggml-common.h:1203-1206 kvalues_iq4nl[16]（IQ4_NL 的非线性码本，int8_t）
static const int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};
#define GROUP_MAX_EPS 1e-15f                                 // ggml-quants.c:20

// ggml-quants.c:28-37 best_index_int8 —— 在升序码本里找 x 的最近邻下标
static inline int best_index_int8(int n, const int8_t * val, float x) {
    if (x <= val[0]) return 0;
    if (x >= val[n-1]) return n-1;
    int ml = 0, mu = n-1;
    while (mu-ml > 1) {
        int mav = (ml+mu)/2;
        if (x < val[mav]) mu = mav; else ml = mav;
    }
    return x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

// ggml-quants.c:1180 make_qkx2_quants（q4_K 的子块 scale/min 搜索）
static float make_qkx2_quants(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max == min) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_error = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = std::max(0, std::min(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_error += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = std::max(0, std::min(nmax, l));
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float cur_error = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                cur_error += w * diff;
            }
            if (cur_error < best_error) {
                for (int i = 0; i < n; ++i) L[i] = Laux[i];
                best_error = cur_error;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

// ggml-quants.c:1261
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// ---- 各档 block 字节（ggml-common.h 的 static_assert 值）----
enum LType { LT_Q4_0, LT_Q4_1, LT_IQ4_NL, LT_Q5_0, LT_Q5_1, LT_Q8_0, LT_Q3_0, LT_Q3_1, LT_Q4_K, LT_COUNT };
static const char * lt_name[LT_COUNT] = { "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1", "q8_0", "q3_0", "q3_1", "q4_K" };
static int lt_block_size(LType t) { return t == LT_Q4_K ? 256 : 32; }        // ggml-common.h:89 QK_K / :212..335 QK*
static int lt_block_bytes(LType t) {
    switch (t) {                                                          // sizeof(block_q*)
        case LT_Q4_0:  return 18;  // 2 + 16          ggml-common.h:212-217
        case LT_Q4_1:  return 20;  // 2 + 2 + 16      ggml-common.h:219-230
        case LT_IQ4_NL:return 18;  // 2 + 16          ggml-common.h:530-536
        case LT_Q5_0:  return 22;  // 2 + 4 + 16      ggml-common.h:247-253
        case LT_Q5_1:  return 24;  // 2 + 2+4+16      ggml-common.h:255-267
        case LT_Q8_0:  return 34;  // 2 + 32          ggml-common.h:335-340
        case LT_Q3_0:  return 14;  // 2 + 4 + 8       ggml-common.h:291-298
        case LT_Q3_1:  return 16;  // 2 + 2+4+8       ggml-common.h:300-312
        case LT_Q4_K:  return 144; // 2+2+12+128      ggml-common.h:411-422
        default: return 0;
    }
}
static double lt_bpw(LType t) { return 8.0 * lt_block_bytes(t) / lt_block_size(t); }
static bool lt_parse(const char * s, LType & out) {
    for (int i = 0; i < LT_COUNT; ++i) if (!strcmp(s, lt_name[i])) { out = (LType)i; return true; }
    if (!strcmp(s, "q4k") || !strcmp(s, "q4_k") || !strcmp(s, "Q4_K")) { out = LT_Q4_K; return true; }
    return false;
}

// 量化 + 反量化：buf 里放的是**逐位照抄 ggml 布局**的字节流，字节数 = ggml_row_size
// 返回值 = 字节数；n 必须整除 blk。x/deq 都是长度 n 的行（K: 一行 = 一条 token 的 256/128 dim；
// V: 一行 = 一条 token 的 dim 向量）。
static size_t llama_qdq(LType t, const float * x, int n, std::vector<uint8_t> & buf, float * deq) {
    const int qk = lt_block_size(t);
    const int nb = n / qk;
    const size_t bytes = (size_t)nb * lt_block_bytes(t);
    buf.assign(bytes, 0);
    const size_t bb = lt_block_bytes(t);

    for (int i = 0; i < nb; ++i) {
        uint8_t * y = buf.data() + (size_t)i * bb;
        const float * xb = x + (size_t)i * qk;

        if (t == LT_Q4_0) {                       // ggml-quants.c:113-148 / :715-733
            float amax = 0.f, mx = 0.f;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (amax < fabsf(v)) { amax = fabsf(v); mx = v; } }
            const float d = mx / -8.f;
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d); std::memcpy(y, &hd, 2);
            for (int j = 0; j < qk/2; ++j) {
                const float x0 = xb[0     + j]*id;
                const float x1 = xb[qk/2  + j]*id;
                const uint8_t xi0 = (uint8_t)std::min(15, (int)(int8_t)(x0 + 8.5f));
                const uint8_t xi1 = (uint8_t)std::min(15, (int)(int8_t)(x1 + 8.5f));
                y[2 + j]  = xi0;
                y[2 + j] |= (uint8_t)(xi1 << 4);
            }
            const float dd = f16_to_f32(hd);
            for (int j = 0; j < qk/2; ++j) {
                const int a = (y[2 + j] & 0x0F) - 8;
                const int b = (y[2 + j] >>   4) - 8;
                deq[i*qk + j + 0   ] = a*dd;
                deq[i*qk + j + qk/2] = b*dd;
            }
        } else if (t == LT_Q4_1) {                // ggml-quants.c:150-185 / :735-754
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const float d  = (mx - mn) / ((1 << 4) - 1);
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d), hm = f32_to_f16_rn(mn);
            std::memcpy(y, &hd, 2); std::memcpy(y + 2, &hm, 2);
            for (int j = 0; j < qk/2; ++j) {
                const float x0 = (xb[0    + j] - mn)*id;
                const float x1 = (xb[qk/2 + j] - mn)*id;
                const uint8_t xi0 = (uint8_t)std::min(15, (int)(int8_t)(x0 + 0.5f));
                const uint8_t xi1 = (uint8_t)std::min(15, (int)(int8_t)(x1 + 0.5f));
                y[4 + j]  = xi0;
                y[4 + j] |= (uint8_t)(xi1 << 4);
            }
            const float dd = f16_to_f32(hd), mm = f16_to_f32(hm);
            for (int j = 0; j < qk/2; ++j) {
                const int a = (y[4 + j] & 0x0F);
                const int b = (y[4 + j] >>   4);
                deq[i*qk + j + 0   ] = a*dd + mm;
                deq[i*qk + j + qk/2] = b*dd + mm;
            }
        } else if (t == LT_Q5_0) {                // ggml-quants.c:187-229 / :756-780
            float amax = 0.f, mx = 0.f;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (amax < fabsf(v)) { amax = fabsf(v); mx = v; } }
            const float d  = mx / -16.f;
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d); std::memcpy(y, &hd, 2);
            uint32_t qh = 0;
            for (int j = 0; j < qk/2; ++j) {
                const float x0 = xb[0    + j]*id;
                const float x1 = xb[qk/2 + j]*id;
                const uint8_t xi0 = (uint8_t)std::min(31, (int)(int8_t)(x0 + 16.5f));
                const uint8_t xi1 = (uint8_t)std::min(31, (int)(int8_t)(x1 + 16.5f));
                y[6 + j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
                qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
            }
            std::memcpy(y + 2, &qh, 4);
            const float dd = f16_to_f32(hd);
            for (int j = 0; j < qk/2; ++j) {
                const uint8_t xh_0 = (uint8_t)(((qh >> (j +  0)) << 4) & 0x10);
                const uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
                const int32_t a = ((y[6 + j] & 0x0F) | xh_0) - 16;
                const int32_t b = ((y[6 + j] >>   4) | xh_1) - 16;
                deq[i*qk + j + 0   ] = a*dd;
                deq[i*qk + j + qk/2] = b*dd;
            }
        } else if (t == LT_Q5_1) {                // ggml-quants.c:231-273 / :782-807
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const float d  = (mx - mn) / ((1 << 5) - 1);
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d), hm = f32_to_f16_rn(mn);
            std::memcpy(y, &hd, 2); std::memcpy(y + 2, &hm, 2);
            uint32_t qh = 0;
            for (int j = 0; j < qk/2; ++j) {
                const float x0 = (xb[0    + j] - mn)*id;
                const float x1 = (xb[qk/2 + j] - mn)*id;
                const uint8_t xi0 = (uint8_t)(x0 + 0.5f);
                const uint8_t xi1 = (uint8_t)(x1 + 0.5f);
                y[8 + j] = (uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
                qh |= ((xi0 & 0x10u) >> 4) << (j + 0);
                qh |= ((xi1 & 0x10u) >> 4) << (j + qk/2);
            }
            std::memcpy(y + 4, &qh, 4);
            const float dd = f16_to_f32(hd), mm = f16_to_f32(hm);
            for (int j = 0; j < qk/2; ++j) {
                const uint8_t xh_0 = (uint8_t)(((qh >> (j +  0)) << 4) & 0x10);
                const uint8_t xh_1 = (uint8_t)(((qh >> (j + 12))     ) & 0x10);
                const int a = (y[8 + j] & 0x0F) | xh_0;
                const int b = (y[8 + j] >>   4) | xh_1;
                deq[i*qk + j + 0   ] = a*dd + mm;
                deq[i*qk + j + qk/2] = b*dd + mm;
            }
        } else if (t == LT_Q8_0) {                // ggml-quants.c:532-555 / :934-948
            float amax = 0.f;
            for (int j = 0; j < qk; ++j) amax = std::max(amax, fabsf(xb[j]));
            const float d  = amax / ((1 << 7) - 1);
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d); std::memcpy(y, &hd, 2);
            int8_t * qs = (int8_t *)(y + 2);
            for (int j = 0; j < qk; ++j) {
                const float x0 = xb[j]*id;
                qs[j] = (int8_t) roundf(x0);
            }
            const float dd = f16_to_f32(hd);
            for (int j = 0; j < qk; ++j) deq[i*qk + j] = qs[j]*dd;
        } else if (t == LT_IQ4_NL) {              // ggml-quants.c:5522-5536 / :5389-5498 / :3148-3164
            // quantize_row_iq4_nl_ref → quantize_row_iq4_nl_impl(QK4_NL=32, block=32, ntry=-1, quant_weights=NULL)
            // super_block_size/block_size == 1 ⇒ 走 impl 的 else 分支（ggml-quants.c:5483-5490）
            float amax = 0.f, mx = 0.f;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (amax < fabsf(v)) { amax = fabsf(v); mx = v; } }
            uint8_t L[32]; for (int j = 0; j < 32; ++j) L[j] = 0;  // C 里全零块 L 未初始化；q4 被 memset 0 ⇒ 确定化为 0
            float d = 0.f;
            if (!(amax < GROUP_MAX_EPS)) {                       // :5420
                d = mx / (float)kvalues_iq4nl[0];                // :5424（ntry<=0 ⇒ max/values[0]，values[0]=-127）
                const float id = 1.0f/d;
                float sumqx = 0.f, sumq2 = 0.f;
                for (int j = 0; j < 32; ++j) {                   // :5427-5435
                    const float al = id*xb[j];
                    const int l = best_index_int8(16, kvalues_iq4nl, al);
                    L[j] = (uint8_t)l;
                    const float q = (float)kvalues_iq4nl[l];
                    const float w = xb[j]*xb[j];                 // :5411（quant_weights=NULL ⇒ w = x*x）
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
                d = sumq2 > 0 ? sumqx/sumq2 : 0.f;               // :5436
                // :5438-5452 的 itry 搜索在 ntry=-1 时循环体不执行（for itry=1; itry<=-1）
            }
            const uint16_t hd = f32_to_f16_rn(d);                // :5484 dh[0] = FP16(scales[0])
            std::memcpy(y, &hd, 2);
            for (int j = 0; j < 16; ++j) y[2 + j] = (uint8_t)(L[j] | (L[16 + j] << 4));  // :5493-5497
            const float dd = f16_to_f32(hd);                     // dequantize_row_iq4_nl :3156-3159
            for (int j = 0; j < 16; ++j) {
                deq[i*qk + j]      = dd * (float)kvalues_iq4nl[y[2 + j] & 0x0F];
                deq[i*qk + 16 + j] = dd * (float)kvalues_iq4nl[y[2 + j] >>   4];
            }
        } else if (t == LT_Q3_0) {                // ggml-quants.c:376-415 / :854-873
            float amax = 0.f, mx = 0.f;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (amax < fabsf(v)) { amax = fabsf(v); mx = v; } }
            const float d  = mx / -4.f;
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d);
            uint32_t qh = 0;
            uint8_t qs[8]; std::memset(qs, 0, sizeof(qs));
            for (int j = 0; j < qk; ++j) {
                const float x0 = xb[j]*id;
                const uint8_t xi0 = (uint8_t)std::min(7, (int)(int8_t)(x0 + 4.5f));
                qs[j % (qk/4)] |= (uint8_t)((xi0 & 0x03) << (2*(j / (qk/4))));
                qh |= (uint32_t)((xi0 & 0x04u) >> 2) << j;
            }
            std::memcpy(y, &hd, 2); std::memcpy(y + 2, &qh, 4); std::memcpy(y + 6, qs, 8);
            const float dd = f16_to_f32(hd);
            for (int j = 0; j < qk; ++j) {
                const int q = ((qs[j % (qk/4)] >> (2*(j / (qk/4)))) & 0x03) | (((qh >> j) & 1) << 2);
                deq[i*qk + j] = (q - 4)*dd;
            }
        } else if (t == LT_Q3_1) {                // ggml-quants.c:417-456 / :875-895
            float mn = FLT_MAX, mx = -FLT_MAX;
            for (int j = 0; j < qk; ++j) { const float v = xb[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
            const float d  = (mx - mn) / ((1 << 3) - 1);
            const float id = d ? 1.0f/d : 0.0f;
            const uint16_t hd = f32_to_f16_rn(d), hm = f32_to_f16_rn(mn);
            uint32_t qh = 0;
            uint8_t qs[8]; std::memset(qs, 0, sizeof(qs));
            for (int j = 0; j < qk; ++j) {
                const float x0 = (xb[j] - mn)*id;
                const uint8_t xi0 = (uint8_t)std::min(7, (int)(int8_t)(x0 + 0.5f));
                qs[j % (qk/4)] |= (uint8_t)((xi0 & 0x03) << (2*(j / (qk/4))));
                qh |= (uint32_t)((xi0 & 0x04u) >> 2) << j;
            }
            std::memcpy(y, &hd, 2); std::memcpy(y + 2, &hm, 2);
            std::memcpy(y + 4, &qh, 4); std::memcpy(y + 8, qs, 8);
            const float dd = f16_to_f32(hd), mm = f16_to_f32(hm);
            for (int j = 0; j < qk; ++j) {
                const int q = ((qs[j % (qk/4)] >> (2*(j / (qk/4)))) & 0x03) | (((qh >> j) & 1) << 2);
                deq[i*qk + j] = q*dd + mm;
            }
        } else if (t == LT_Q4_K) {                // ggml-quants.c:1838-1908 / :1910-1932
            uint8_t L[256];
            uint8_t Laux[32];
            float   weights[32];
            float   mins[8];
            float   scales[8];
            float max_scale = 0, max_min = 0;
            for (int j = 0; j < 8; ++j) {
                float sum_x2 = 0;
                for (int l = 0; l < 32; ++l) sum_x2 += xb[32*j + l] * xb[32*j + l];
                float av_x = sqrtf(sum_x2/32);
                for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(xb[32*j + l]);
                scales[j] = make_qkx2_quants(32, 15, xb + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
                if (scales[j] > max_scale) max_scale = scales[j];
                if (mins[j]   > max_min  ) max_min   = mins[j];
            }
            float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
            float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
            uint8_t scales_packed[12] = {0};
            for (int j = 0; j < 8; ++j) {
                uint8_t ls = (uint8_t)nearest_int(inv_scale*scales[j]);
                uint8_t lm = (uint8_t)nearest_int(inv_min*mins[j]);
                ls = std::min<uint8_t>(63, ls);
                lm = std::min<uint8_t>(63, lm);
                if (j < 4) {
                    scales_packed[j]   = ls;
                    scales_packed[j+4] = lm;
                } else {
                    scales_packed[j+4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
                    scales_packed[j-4] |= (uint8_t)((ls >> 4) << 6);
                    scales_packed[j-0] |= (uint8_t)((lm >> 4) << 6);
                }
            }
            const uint16_t hd = f32_to_f16_rn(max_scale/63.f);
            const uint16_t hdm = f32_to_f16_rn(max_min/63.f);
            std::memcpy(y, &hd, 2); std::memcpy(y + 2, &hdm, 2);
            std::memcpy(y + 4, scales_packed, 12);
            for (int j = 0; j < 8; ++j) {
                uint8_t sc, m;
                get_scale_min_k4(j, scales_packed, &sc, &m);
                const float d = f16_to_f32(hd) * sc;
                if (!d) continue;
                const float dm = f16_to_f32(hdm) * m;
                for (int ii = 0; ii < 32; ++ii) {
                    int l = nearest_int((xb[32*j + ii] + dm)/d);
                    l = std::max(0, std::min(15, l));
                    L[32*j + ii] = (uint8_t)l;
                }
            }
            uint8_t * q = y + 16;
            for (int j = 0; j < 256; j += 64) {
                for (int l = 0; l < 32; ++l) q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
                q += 32;
            }
            // dequant
            {
                const uint8_t * qq = y + 16;
                const float dd   = f16_to_f32(hd);
                const float dmin = f16_to_f32(hdm);
                int is = 0;
                int out = 0;
                for (int j = 0; j < 256; j += 64) {
                    uint8_t sc, m;
                    get_scale_min_k4(is + 0, scales_packed, &sc, &m);
                    const float d1 = dd * sc; const float m1 = dmin * m;
                    get_scale_min_k4(is + 1, scales_packed, &sc, &m);
                    const float d2 = dd * sc; const float m2 = dmin * m;
                    for (int l = 0; l < 32; ++l) deq[out++] = d1 * (qq[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) deq[out++] = d2 * (qq[l]  >> 4) - m2;
                    qq += 32; is += 2;
                }
            }
        }
    }
    return bytes;
}

// ===========================================================================
// 5. 数据
// ===========================================================================
struct Tile { std::string name; int rows, cols; bool token_is_row; std::vector<float> d; };

static std::vector<Tile> load_tiles_bin(const char * path) {
    std::vector<Tile> out;
    FILE * fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "无法打开 %s\n", path); exit(2); }
    int32_t n = 0;
    if (fread(&n, 4, 1, fp) != 1) { fclose(fp); return out; }
    for (int i = 0; i < n; ++i) {
        int32_t nl = 0, rr = 0, cc = 0, tr = 0;
        if (fread(&nl, 4, 1, fp) != 1) break;
        std::string nm(nl, '\0'); if (fread(&nm[0], 1, nl, fp) != (size_t)nl) break;
        if (fread(&rr, 4, 1, fp) != 1) break;
        if (fread(&cc, 4, 1, fp) != 1) break;
        if (fread(&tr, 4, 1, fp) != 1) break;
        Tile t; t.name = nm; t.rows = rr; t.cols = cc; t.token_is_row = (tr != 0);
        t.d.resize(size_t(rr) * cc);
        if (fread(t.d.data(), 4, t.d.size(), fp) != t.d.size()) break;
        out.push_back(std::move(t));
    }
    fclose(fp);
    return out;
}

static void fill_gauss(std::vector<float> & d, int rows, int cols, uint64_t seed, bool outlier) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    d.resize(size_t(rows) * cols);
    for (auto & v : d) v = nd(rng);
    if (outlier) {
        for (int c = 0; c < cols; ++c) {
            d[0 * cols + c] *= 20.f;
            d[7 * cols + c] *= 20.f;
            d[63 * cols + c] *= 20.f;
            for (int r = 0; r < rows; ++r) {
                std::student_t_distribution<float> td(3.0);
                d[r * cols + c] *= (1.f + 0.25f * std::fabs(td(rng)));
            }
        }
    }
}

// 合成 tile：与 bee_oracle 的 split_run 同布局，之后按 beellama D256 口径旋转成"入库态"
static std::vector<Tile> build_synth_tiles(int group, uint64_t seed, bool outlier);
static void rot_apply(std::vector<Tile> & v, bool inverse);

// 名称里去掉 _sN 段：L35_k_h0_s0_g0 -> L35_k_h0_g0
static std::string tile_key(const std::string & name) {
    size_t p = name.find("_s");
    while (p != std::string::npos) {
        size_t q = p + 2;
        while (q < name.size() && isdigit((unsigned char)name[q])) ++q;
        if (q > p + 2 && q < name.size() && name[q] == '_') return name.substr(0, p) + name.substr(q);
        p = name.find("_s", p + 1);
    }
    return name;
}

// 按 (h,g) 把 tile 对（s0/s1）做旋转：inverse=false → raw 变入库态；true → 入库态还原 raw
// dim_axis 由 token_is_row 决定：K(rows=dim, dim 轴=0) / V(cols=dim, dim 轴=1)
static void rot_apply(std::vector<Tile> & v, bool inverse) {
    std::map<std::string, std::vector<int>> bykey;
    for (int i = 0; i < (int)v.size(); ++i) bykey[tile_key(v[i].name)].push_back(i);
    float p[128], q[128], a[128], b[128];
    for (auto & kv : bykey) {
        if (kv.second.size() != 2) {
            fprintf(stderr, "警告：%s 的 slice 数 = %zu（期望 2），该组跳过旋转\n", kv.first.c_str(), kv.second.size());
            continue;
        }
        const int i0 = kv.second[0], i1 = kv.second[1];
        Tile & A = v[i0]; Tile & B = v[i1];
        if (A.rows != B.rows || A.cols != B.cols || A.token_is_row != B.token_is_row) {
            fprintf(stderr, "警告：%s 两片形状不一致\n", kv.first.c_str()); continue;
        }
        const bool V = A.token_is_row;
        const int ntok = V ? A.rows : A.cols;
        for (int t = 0; t < ntok; ++t) {
            if (!V) { for (int r = 0; r < 128; ++r) { p[r] = A.d[size_t(r) * A.cols + t]; q[r] = B.d[size_t(r) * B.cols + t]; } }
            else    { for (int c = 0; c < 128; ++c) { p[c] = A.d[size_t(t) * A.cols + c];   q[c] = B.d[size_t(t) * B.cols + c]; } }
            if (inverse) rot_pair_inv(p, q, a, b); else rot_pair_fwd(p, q, a, b);
            if (!V) { for (int r = 0; r < 128; ++r) { A.d[size_t(r) * A.cols + t] = a[r]; B.d[size_t(r) * B.cols + t] = b[r]; } }
            else    { for (int c = 0; c < 128; ++c) { A.d[size_t(t) * A.cols + c] = a[c];   B.d[size_t(t) * B.cols + c] = b[c]; } }
        }
    }
}

// 入库态（已旋转）→ raw
static std::vector<Tile> unrotate_tiles(const std::vector<Tile> & rot) {
    std::vector<Tile> raw = rot;
    rot_apply(raw, /*inverse=*/true);
    return raw;
}

// 合成 tile：与 bee_oracle 的 split_run 同布局，之后按 beellama D256 口径旋转成"入库态"
static std::vector<Tile> build_synth_tiles(int group, uint64_t seed, bool outlier) {
    const int nh = 4, nt = 256, dm = 256;
    std::vector<Tile> out;
    const char * tag = outlier ? "synth_outlier" : "synth_gauss";
    for (int h = 0; h < nh; ++h) {
        std::vector<float> a, b;
        fill_gauss(a, nt, dm, seed + h * 977ull, outlier);
        fill_gauss(b, nt, dm, seed + 100000 + h * 977ull, outlier);
        for (int g = 0; g * group + group <= nt; ++g) {
            for (int s = 0; s < 2; ++s) {
                const int dimlo = s * 128;
                std::vector<float> tk(size_t(128) * group), tv(size_t(group) * 128);
                for (int r = 0; r < 128; ++r)
                    for (int c = 0; c < group; ++c) {
                        tk[size_t(r) * group + c] = a[size_t(g * group + c) * dm + dimlo + r];
                        tv[size_t(c) * 128 + r]    = b[size_t(g * group + c) * dm + dimlo + r];
                    }
                char nm[128];
                snprintf(nm, sizeof nm, "%s_k_h%d_s%d_g%d", tag, h, s, g);
                out.push_back({ nm, 128, group, false, tk });
                snprintf(nm, sizeof nm, "%s_v_h%d_s%d_g%d", tag, h, s, g);
                out.push_back({ nm, group, 128, true, tv });
            }
        }
    }
    rot_apply(out, /*inverse=*/false);
    return out;
}


// ===========================================================================
// 6. 误差指标（照抄 bee_oracle.cpp:423-464，未改）
// ===========================================================================
struct Metrics { double cos_global, cos_vec_mean, cos_vec_min, rel_l2, max_abs, max_rel; };

static Metrics measure(const std::vector<float> & orig, const std::vector<float> & deq,
                       int rows, int cols, bool token_is_row) {
    Metrics M{};
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < orig.size(); ++i) {
        dot += (double)deq[i] * orig[i];
        na += (double)deq[i] * deq[i];
        nb += (double)orig[i] * orig[i];
    }
    M.cos_global = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
    M.rel_l2 = std::sqrt([&] { double s = 0; for (size_t i = 0; i < orig.size(); ++i) { const double d = deq[i] - orig[i]; s += d * d; } return s; }()) / (std::sqrt(nb) + 1e-300);
    M.max_abs = 0;
    const double rms = std::sqrt(nb / orig.size());
    M.max_rel = 0;
    for (size_t i = 0; i < orig.size(); ++i) {
        const double d = std::fabs((double)deq[i] - orig[i]);
        if (d > M.max_abs) M.max_abs = d;
        if (std::fabs(orig[i]) > 1e-3 * rms) {
            const double rel = d / std::fabs(orig[i]);
            if (rel > M.max_rel) M.max_rel = rel;
        }
    }
    const int nvec = token_is_row ? rows : cols;
    const int vlen = token_is_row ? cols : rows;
    double sum = 0, mn = 1.0;
    M.cos_vec_min = 1.0;
    for (int v = 0; v < nvec; ++v) {
        double d2 = 0, a2 = 0, b2 = 0;
        for (int k = 0; k < vlen; ++k) {
            const size_t idx = token_is_row ? size_t(v) * cols + k : size_t(k) * cols + v;
            d2 += (double)deq[idx] * orig[idx]; a2 += (double)deq[idx] * deq[idx]; b2 += (double)orig[idx] * orig[idx];
        }
        const double c = d2 / (std::sqrt(a2) * std::sqrt(b2) + 1e-300);
        sum += c; if (c < mn) mn = c;
    }
    M.cos_vec_mean = sum / nvec;
    M.cos_vec_min = mn;
    return M;
}

// ===========================================================================
// 6b. 以「一条 token 的 256-dim head 向量」为单位测误差
//     这是旋转不变量的正确单位：跨片 Hadamard 会在 s0/s1 之间搬运能量，所以
//     单看 128-dim 片的 cos 在旋转前后不等（|o0| ≠ |v0|），只有把两片合起来
//     （256 = head_dim）才是正交变换的不变量。所有跨档比较都在这个单位上做。
// ===========================================================================
static Metrics measure_pair(const Tile & A, const Tile & B,
                            const std::vector<float> & refA, const std::vector<float> & refB,
                            const std::vector<float> & recA, const std::vector<float> & recB) {
    const bool V = A.token_is_row;
    const int ntok = V ? A.rows : A.cols;
    const int dim  = V ? A.cols : A.rows;      // 128
    const int L    = 2 * dim;                  // 256
    std::vector<double> o(size_t(ntok) * L), q(size_t(ntok) * L);
    for (int t = 0; t < ntok; ++t) {
        for (int k = 0; k < dim; ++k) {
            if (!V) {
                o[size_t(t)*L + k]       = refA[size_t(k)*A.cols + t];
                o[size_t(t)*L + dim + k] = refB[size_t(k)*B.cols + t];
                q[size_t(t)*L + k]       = recA[size_t(k)*A.cols + t];
                q[size_t(t)*L + dim + k] = recB[size_t(k)*B.cols + t];
            } else {
                o[size_t(t)*L + k]       = refA[size_t(t)*A.cols + k];
                o[size_t(t)*L + dim + k] = refB[size_t(t)*B.cols + k];
                q[size_t(t)*L + k]       = recA[size_t(t)*A.cols + k];
                q[size_t(t)*L + dim + k] = recB[size_t(t)*B.cols + k];
            }
        }
    }
    Metrics M{};
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < o.size(); ++i) { dot += q[i]*o[i]; na += q[i]*q[i]; nb += o[i]*o[i]; }
    M.cos_global = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
    double s = 0; for (size_t i = 0; i < o.size(); ++i) { const double d = q[i]-o[i]; s += d*d; }
    M.rel_l2 = std::sqrt(s) / (std::sqrt(nb) + 1e-300);
    M.max_abs = 0; for (size_t i = 0; i < o.size(); ++i) M.max_abs = std::max(M.max_abs, std::fabs(q[i]-o[i]));
    double sm = 0, mn = 1.0;
    for (int t = 0; t < ntok; ++t) {
        double d2 = 0, a2 = 0, b2 = 0;
        for (int k = 0; k < L; ++k) { const size_t i = size_t(t)*L + k; d2 += q[i]*o[i]; a2 += q[i]*q[i]; b2 += o[i]*o[i]; }
        const double c = d2 / (std::sqrt(a2) * std::sqrt(b2) + 1e-300);
        sm += c; if (c < mn) mn = c;
    }
    M.cos_vec_mean = sm / ntok; M.cos_vec_min = mn;
    M.max_rel = 0;
    return M;
}

// ===========================================================================
// 7. 模式
// ===========================================================================
enum Mode { M_KVARN, M_LLAMA, M_LLAMA_ROT, M_ROT_ONLY, M_BAL_ONLY, M_NEITHER, M_COUNT };
static const char * mode_name[M_COUNT] = { "kvarn", "llama", "llama-rot", "rot-only", "bal-only", "neither" };

struct Opts {
    Mode mode = M_KVARN;
    int bits_k = 4, bits_v = 4, group = 128, iters = 16;
    LType lk = LT_Q4_0, lv = LT_Q4_0;
    bool stage_f16 = false, ref_h16 = false, slice_metrics = false;
    const char * tiles = nullptr;
    bool gauss = false, outlier = false;
    uint64_t seed = 20261013ull;
    std::string tag;
};

// 每 token·head-slice 的字节数（K 与 V 分开）
struct ModeOut { std::vector<uint8_t> rec; size_t bytes = 0; size_t bpt_x1000 = 0; };

// 对单个 128-dim tile 跑 KVarN 码本（rot/raw 由调用方决定）
static ModeOut run_kvarn_slice(const Tile & t, int bits, int iters, bool stage_f16, std::vector<float> & deq) {
    std::vector<float> in = t.d;
    if (stage_f16) for (auto & v : in) v = h16(v);
    ModeOut O;
    bee_quant_tile(in.data(), t.rows, t.cols, bits, iters, O.rec, nullptr);
    deq.assign(in.size(), 0.f);
    bee_dequant_tile(O.rec.data(), t.rows, t.cols, bits, deq.data());
    O.bytes = O.rec.size();
    const int ntok = t.token_is_row ? t.rows : t.cols;
    O.bpt_x1000 = (O.bytes * 1000 + ntok/2) / ntok;
    return O;
}

// 对单个 128-dim tile 跑 llama 码本（块沿 dim 轴，每 token 一行 128 元素 = 4 块）
static ModeOut run_llama_slice(const Tile & t, LType ty, bool stage_f16, std::vector<float> & deq) {
    std::vector<float> in = t.d;
    if (stage_f16) for (auto & v : in) v = h16(v);
    const int ntok = t.token_is_row ? t.rows : t.cols;
    const int dim  = t.token_is_row ? t.cols : t.rows;
    deq.assign(in.size(), 0.f);
    std::vector<float> row(dim), drow(dim);
    std::vector<uint8_t> buf;
    size_t row_bytes = 0;
    for (int tk = 0; tk < ntok; ++tk) {
        for (int d = 0; d < dim; ++d) row[d] = t.token_is_row ? in[size_t(tk)*dim + d] : in[size_t(d)*ntok + tk];
        row_bytes = llama_qdq(ty, row.data(), dim, buf, drow.data());
        for (int d = 0; d < dim; ++d) {
            if (t.token_is_row) deq[size_t(tk)*dim + d] = drow[d];
            else                deq[size_t(d)*ntok + tk] = drow[d];
        }
    }
    ModeOut O;
    O.bytes = row_bytes * (size_t)ntok;
    O.bpt_x1000 = (row_bytes * 1000);
    return O;
}

// 对一对 128-dim slice（256 dim 行）跑 llama q4_K（super-block 256，不可切 128）
static ModeOut run_q4k_pair(const Tile & A, const Tile & B, bool stage_f16,
                            std::vector<float> & deqA, std::vector<float> & deqB) {
    std::vector<float> ia = A.d, ib = B.d;
    if (stage_f16) { for (auto & v : ia) v = h16(v); for (auto & v : ib) v = h16(v); }
    const bool V = A.token_is_row;
    const int ntok = V ? A.rows : A.cols;
    deqA.assign(ia.size(), 0.f); deqB.assign(ib.size(), 0.f);
    std::vector<float> row(256), drow(256);
    std::vector<uint8_t> buf;
    size_t row_bytes = 0;
    for (int tk = 0; tk < ntok; ++tk) {
        if (!V) { for (int r = 0; r < 128; ++r) { row[r] = ia[size_t(r)*A.cols + tk]; row[128+r] = ib[size_t(r)*B.cols + tk]; } }
        else    { for (int c = 0; c < 128; ++c) { row[c] = ia[size_t(tk)*128 + c]; row[128+c] = ib[size_t(tk)*128 + c]; } }
        row_bytes = llama_qdq(LT_Q4_K, row.data(), 256, buf, drow.data());
        if (!V) { for (int r = 0; r < 128; ++r) { deqA[size_t(r)*A.cols + tk] = drow[r]; deqB[size_t(r)*B.cols + tk] = drow[128+r]; } }
        else    { for (int c = 0; c < 128; ++c) { deqA[size_t(tk)*128 + c] = drow[c];   deqB[size_t(tk)*128 + c] = drow[128+c]; } }
    }
    ModeOut O;
    O.bytes = row_bytes * (size_t)ntok;
    O.bpt_x1000 = row_bytes * 1000 / 2;    // 256 dim 行 → 每 128-dim head-slice 摊一半
    return O;
}

// 把一对 slice 的重建旋转回 raw 域
static void map_pair_to_raw(const Tile & refA, const Tile & refB,
                            const std::vector<float> & dA, const std::vector<float> & dB,
                            std::vector<float> & rA, std::vector<float> & rB) {
    const bool V = refA.token_is_row;
    const int ntok = V ? refA.rows : refA.cols;
    rA.assign(dA.size(), 0.f); rB.assign(dB.size(), 0.f);
    float p[128], q[128], a[128], b[128];
    for (int t = 0; t < ntok; ++t) {
        if (!V) { for (int r = 0; r < 128; ++r) { p[r] = dA[size_t(r)*refA.cols + t]; q[r] = dB[size_t(r)*refB.cols + t]; } }
        else    { for (int c = 0; c < 128; ++c) { p[c] = dA[size_t(t)*128 + c];       q[c] = dB[size_t(t)*128 + c]; } }
        rot_pair_inv(p, q, a, b);
        if (!V) { for (int r = 0; r < 128; ++r) { rA[size_t(r)*refA.cols + t] = a[r]; rB[size_t(r)*refB.cols + t] = b[r]; } }
        else    { for (int c = 0; c < 128; ++c) { rA[size_t(t)*128 + c] = a[c];       rB[size_t(t)*128 + c] = b[c]; } }
    }
}

// ===========================================================================
// 8. 子命令
// ===========================================================================
static int cmd_selftest(const char * tilespath) {
    int fail = 0;
    printf("== 1. WHT128 自逆性 ==\n");
    {
        std::mt19937_64 rng(7);
        std::normal_distribution<float> nd(0.f, 1.f);
        float x[128], y[128];
        for (auto & v : x) v = nd(rng);
        std::memcpy(y, x, sizeof(x));
        wht128_norm(y); wht128_norm(y);
        double mx = 0; for (int i = 0; i < 128; ++i) mx = std::max(mx, (double)std::fabs(y[i]-x[i]));
        printf("  max|WHT(WHT(x))-x| = %.3e  %s\n", mx, mx < 1e-5 ? "OK" : "FAIL");
        if (!(mx < 1e-5)) ++fail;
    }
    printf("== 2. rot_pair 往返（forward 后 inverse 应还原）==\n");
    {
        std::mt19937_64 rng(11);
        std::normal_distribution<float> nd(0.f, 1.f);
        float a[128], b[128], o0[128], o1[128], aa[128], bb[128];
        for (int i = 0; i < 128; ++i) { a[i] = nd(rng); b[i] = nd(rng); }
        rot_pair_fwd(a, b, o0, o1);
        rot_pair_inv(o0, o1, aa, bb);
        double m1 = 0, m2 = 0;
        for (int i = 0; i < 128; ++i) { m1 = std::max(m1, (double)std::fabs(aa[i]-a[i])); m2 = std::max(m2, (double)std::fabs(bb[i]-b[i])); }
        printf("  max|a'-a| = %.3e  max|b'-b| = %.3e  %s\n", m1, m2, (m1 < 1e-5 && m2 < 1e-5) ? "OK" : "FAIL");
        if (!(m1 < 1e-5 && m2 < 1e-5)) ++fail;
    }
    if (tilespath) {
        printf("== 3. 真实 tile 的 unrotate 往返（对 raw 再 forward，应逐元素还原入库态）==\n");
        auto rot = load_tiles_bin(tilespath);
        auto raw = unrotate_tiles(rot);
        std::map<std::string, std::vector<int>> bykey;
        for (int i = 0; i < (int)rot.size(); ++i) bykey[tile_key(rot[i].name)].push_back(i);
        double mx = 0; int npair = 0;
        float p[128], q[128], o0[128], o1[128];
        for (auto & kv : bykey) {
            if (kv.second.size() != 2) continue;
            const int i0 = kv.second[0], i1 = kv.second[1];
            const Tile & A = raw[i0], & B = raw[i1];
            const bool V = A.token_is_row;
            const int ntok = V ? A.rows : A.cols;
            for (int t = 0; t < ntok; ++t) {
                if (!V) { for (int r = 0; r < 128; ++r) { p[r] = A.d[size_t(r)*A.cols+t]; q[r] = B.d[size_t(r)*B.cols+t]; } }
                else    { for (int c = 0; c < 128; ++c) { p[c] = A.d[size_t(t)*128+c];   q[c] = B.d[size_t(t)*128+c]; } }
                rot_pair_fwd(p, q, o0, o1);
                for (int k = 0; k < 128; ++k) {
                    const double d0 = V ? std::fabs(o0[k] - rot[i0].d[size_t(t)*128+k]) : std::fabs(o0[k] - rot[i0].d[size_t(k)*rot[i0].cols+t]);
                    const double d1 = V ? std::fabs(o1[k] - rot[i1].d[size_t(t)*128+k]) : std::fabs(o1[k] - rot[i1].d[size_t(k)*rot[i1].cols+t]);
                    mx = std::max(mx, std::max(d0, d1));
                }
            }
            ++npair;
        }
        printf("  tiles=%zu 对=%d  max|forward(unrotate(rot)) - rot| = %.3e  %s\n", rot.size(), npair, mx, mx < 1e-5 ? "OK" : "FAIL");
        if (!(mx < 1e-5)) ++fail;
    }
    printf("== 4. llama 码本往返自检（量化→反量化后重跑，码字应不变）==\n");
    {
        std::mt19937_64 rng(23);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int t = 0; t < LT_COUNT; ++t) {
            const LType ty = (LType)t;
            const int n = lt_block_size(ty);
            std::vector<float> x(n), d1(n), d2(n);
            for (auto & v : x) v = nd(rng);
            std::vector<uint8_t> b1, b2;
            const size_t B1 = llama_qdq(ty, x.data(), n, b1, d1.data());
            const size_t B2 = llama_qdq(ty, d1.data(), n, b2, d2.data());
            const bool idem = (b1 == b2);
            double mxd = 0; for (int i = 0; i < n; ++i) mxd = std::max(mxd, (double)std::fabs(d1[i]-d2[i]));
            printf("  %-5s n=%3d bytes=%3zu bpw=%.3f  幂等=%s max|d1-d2|=%.3e\n",
                   lt_name[ty], n, B1, lt_bpw(ty), idem ? "OK" : "DIFF", mxd);
            if (!idem) ++fail;
        }
    }
    printf("== 5. 字节账自检（对照 ggml-common.h 的 static_assert 值）==\n");
    {
        const int expect[LT_COUNT] = { 18, 20, 18, 22, 24, 34, 14, 16, 144 };
        for (int t = 0; t < LT_COUNT; ++t) {
            const int got = lt_block_bytes((LType)t);
            printf("  %-5s blk=%3d block_bytes=%3d (期望 %3d) %s\n", lt_name[t], lt_block_size((LType)t), got, expect[t], got==expect[t]?"OK":"FAIL");
            if (got != expect[t]) ++fail;
        }
    }
    printf("== 6. 误差度量在正交旋转下的不变性（等字节跨档对照成立的前提）==\n");
    {
        // 取一对真实 tile（或合成），在 rot 域跑 KVarN，分别按 rot 域与 raw 域测 cos/relL2
        std::vector<Tile> R;
        if (tilespath) R = load_tiles_bin(tilespath);
        else           R = build_synth_tiles(128, 20261013ull, false);
        std::vector<Tile> W = unrotate_tiles(R);
        std::map<std::string, std::vector<int>> bk;
        for (int i = 0; i < (int)R.size(); ++i) bk[tile_key(R[i].name)].push_back(i);
        double dcos = 0, drel = 0; int n = 0;
        for (auto & kv : bk) {
            if (kv.second.size() != 2) continue;
            const int i0 = kv.second[0], i1 = kv.second[1];
            const Tile & t0 = R[i0], & t1 = R[i1];
            std::vector<float> dA, dB, d1;
            run_kvarn_slice(t0, 4, 16, false, dA);
            run_kvarn_slice(t1, 4, 16, false, d1); dB = d1;
            const Metrics a0 = measure_pair(t0, t1, t0.d, t1.d, dA, dB);
            std::vector<float> mA, mB;
            map_pair_to_raw(t0, t1, dA, dB, mA, mB);
            const Metrics b0 = measure_pair(t0, t1, W[i0].d, W[i1].d, mA, mB);
            dcos = std::max(dcos, std::fabs(a0.cos_global - b0.cos_global));
            drel = std::max(drel, std::fabs(a0.rel_l2 - b0.rel_l2));
            if (getenv("BEE_DEBUG")) {
                static int dbg = 0;
                if (dbg++ < 2) {
                    auto nrm = [](const std::vector<float> & d) {
                        double s = 0; for (float v : d) s += (double)v * v; return std::sqrt(s);
                    };
                    // F(W) 与 R 的逐元素偏差
                    const bool VV = t0.token_is_row;
                    const int nt = VV ? t0.rows : t0.cols;
                    double mxF = 0, mxD = 0;
                    float pp[128], qq[128], o0[128], o1[128];
                    for (int t = 0; t < nt; ++t) {
                        if (!VV) { for (int r = 0; r < 128; ++r) { pp[r] = W[i0].d[size_t(r)*t0.cols+t]; qq[r] = W[i1].d[size_t(r)*t1.cols+t]; } }
                        else     { for (int c = 0; c < 128; ++c) { pp[c] = W[i0].d[size_t(t)*128+c];   qq[c] = W[i1].d[size_t(t)*128+c]; } }
                        rot_pair_fwd(pp, qq, o0, o1);
                        for (int k = 0; k < 128; ++k) {
                            const double e0 = std::fabs(o0[k] - (VV ? t0.d[size_t(t)*128+k] : t0.d[size_t(k)*t0.cols+t]));
                            const double e1 = std::fabs(o1[k] - (VV ? t1.d[size_t(t)*128+k] : t1.d[size_t(k)*t1.cols+t]));
                            mxF = std::max(mxF, std::max(e0, e1));
                        }
                    }
                    for (size_t z = 0; z < t0.d.size(); ++z) mxD = std::max(mxD, (double)std::fabs(W[i0].d[z] - t0.d[z]));
                    printf("    [dbg] %-18s |R0|=%.4f |R1|=%.4f |W0|=%.4f |W1|=%.4f  max|F(W)-R|=%.3e max|W-R|=%.3e\n",
                           kv.first.c_str(), nrm(t0.d), nrm(t1.d), nrm(W[i0].d), nrm(W[i1].d), mxF, mxD);
                }
            }
            ++n;
        }
        const bool ok = (dcos < 1e-5 && drel < 1e-5);
        printf("  对=%d  max|Δcos_glob(rot域 - raw域)| = %.3e  max|ΔrelL2| = %.3e  %s\n", n, dcos, drel, ok ? "OK" : "FAIL");
        printf("  （cos 与 relL2 在正交变换下不变 ⇒ 各档在不同码域测出的数可直接横向比；maxAbs 不是不变量，本文件统一在 raw 域测）\n");
        if (!ok) ++fail;
    }
    printf("\n失败项 = %d\n", fail);
    return fail ? 1 : 0;
}

static int cmd_bytes() {
    printf("== 每 token·head-slice(128 dim) 的字节账 ==\n");
    printf("口径：KVarN = bee record 口径(record_dim=128)；llama = ggml_row_size(type, 128)，K/V 各自按 head_dim 分块\n\n");
    printf("--- A. KVarN（record 口径）---\n");
    printf("bits | payload | scale/zp/other | K 或 V B/token/slice | D256 K+V B/token/head\n");
    for (int bits = 2; bits <= 8; ++bits) {
        const RecordLayout R = make_record_layout(128, bits, false);
        const size_t tot = record_tile_bytes(128, 2, bits, bits);
        printf("%4d | %7zu | %14zu | %20d | %21zu\n", bits, R.payload_bytes,
               2*size_t(R.rows)*2 + size_t(R.cols)*2, (int)(R.record_bytes/128), tot/128);
    }
    printf("\n--- B. llama.cpp 原生 KV 档（QK 块沿 head_dim）---\n");
    printf("type  | blk | block_B | bpw  | 128-dim 片 B/token/slice | D256 K+V B/token/head\n");
    for (int t = 0; t < LT_COUNT; ++t) {
        const LType ty = (LType)t;
        const int blk = lt_block_size(ty);
        const int per_slice = blk <= 128 ? (128 / blk) * lt_block_bytes(ty) : lt_block_bytes(ty) / 2;
        const int per_head  = blk <= 256 ? (256 / blk) * lt_block_bytes(ty) : 0;
        char mark[8] = "";
        if (blk > 128) snprintf(mark, sizeof mark, " *");
        printf("%-5s | %3d | %7d | %4.2f | %24d%s | %21d\n",
               lt_name[t], blk, lt_block_bytes(ty), lt_bpw(ty), per_slice, mark, 2*per_head);
    }
    printf("  * q4_K 的 super-block = 256 元素，128-dim 的 head-slice 装不下（ggml_row_size 会 assert），\n");
    printf("    上表按 256-dim 整行 144 B 折半记 72 B/token/slice；llama.cpp 的 KV cache 类型集合里本来也没有 q4_K。\n");
    printf("\n--- C. 等字节对照的分档点（K 或 V 单侧 B/token/slice）---\n");
    printf("  KVarN K2=38 K3=54 K4=70 K5=86 K6=102 K8=134\n");
    printf("  llama q4_0=72 q4_1=80 iq4_nl=72 q5_0=88 q5_1=96 q8_0=136 q3_0=56 q3_1=64 q4_K=72\n");
    printf("  (q4_0 与 q4_K 与 iq4_nl 同为 4.5 bpw；KVarN K4=70 比 q4_0 少 2 B/token/slice)\n");
    return 0;
}

static int cmd_quant(int argc, char ** argv) {
    Opts o;
    for (int i = 0; i < argc; ++i) {
        if      (!strcmp(argv[i], "--mode")    && i+1 < argc) {
            const char * m = argv[++i]; o.mode = M_COUNT;
            for (int k = 0; k < M_COUNT; ++k) if (!strcmp(m, mode_name[k])) o.mode = (Mode)k;
            if (o.mode == M_COUNT) { fprintf(stderr, "未知 mode %s\n", m); return 2; }
        }
        else if (!strcmp(argv[i], "--bits-k")  && i+1 < argc) o.bits_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits-v")  && i+1 < argc) o.bits_v = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group")   && i+1 < argc) o.group  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")   && i+1 < argc) o.iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--lk")      && i+1 < argc) { if (!lt_parse(argv[++i], o.lk)) { fprintf(stderr,"未知 llama 类型\n"); return 2; } }
        else if (!strcmp(argv[i], "--lv")      && i+1 < argc) { if (!lt_parse(argv[++i], o.lv)) { fprintf(stderr,"未知 llama 类型\n"); return 2; } }
        else if (!strcmp(argv[i], "--tiles")   && i+1 < argc) o.tiles = argv[++i];
        else if (!strcmp(argv[i], "--gauss"))   o.gauss = true;
        else if (!strcmp(argv[i], "--outlier")) { o.outlier = true; o.gauss = true; }
        else if (!strcmp(argv[i], "--seed")    && i+1 < argc) o.seed = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--stage-f16")) o.stage_f16 = true;
        else if (!strcmp(argv[i], "--ref-h16"))   o.ref_h16 = true;
        else if (!strcmp(argv[i], "--slice"))     o.slice_metrics = true;
        else if (!strcmp(argv[i], "--tag")     && i+1 < argc) o.tag = argv[++i];
    }
    // rot-only / neither 强制 iters=0
    int iters = o.iters;
    if (o.mode == M_ROT_ONLY || o.mode == M_NEITHER) iters = 0;

    std::vector<Tile> rot;
    if (o.tiles) rot = load_tiles_bin(o.tiles);
    else         rot = build_synth_tiles(o.group, o.seed, o.outlier);
    if (rot.empty()) { printf("无 tile\n"); return 2; }
    std::vector<Tile> raw = unrotate_tiles(rot);

    const bool codec_in_rot = !(o.mode == M_LLAMA || o.mode == M_BAL_ONLY || o.mode == M_NEITHER);
    const char * dom = codec_in_rot ? "rot" : "raw";
    const char * datatag = o.tag.empty() ? (o.tiles ? "real" : (o.outlier ? "synth_outlier" : "synth_gauss")) : o.tag.c_str();
    // ref_h16 且码域=rot 时按旧 oracle 口径在码域测（复刻归档数字）
    const bool measure_codec_domain = (o.ref_h16 && codec_in_rot);
    const bool q4k_any = (o.mode == M_LLAMA || o.mode == M_LLAMA_ROT) && (o.lk == LT_Q4_K || o.lv == LT_Q4_K);
    printf("# mode=%s dom=%s data=%s group=%d iters=%d stage_f16=%d ref=%s  单位=一条 token 的 256-dim head 向量%s\n",
           mode_name[o.mode], dom, datatag, o.group, iters, (int)o.stage_f16,
           measure_codec_domain ? "h16(rot码域)" : (o.ref_h16 ? "h16(raw)" : "f32(raw)"),
           q4k_any ? "  [q4_K: rec= 为 256-dim 整行的两片合计]" : "");

    std::map<std::string, std::vector<int>> bykey;
    for (int i = 0; i < (int)rot.size(); ++i) bykey[tile_key(rot[i].name)].push_back(i);

    double sk[4] = {0,0,0,0}, sv[4] = {0,0,0,0};
    int nk = 0, nv = 0;
    double bk = 0, bv = 0;

    for (auto & kv : bykey) {
        const std::vector<int> & idx = kv.second;
        if (idx.size() != 2) continue;
        const int i0 = idx[0], i1 = idx[1];
        const Tile & t0 = rot[i0], & t1 = rot[i1];
        const Tile & r0 = raw[i0], & r1 = raw[i1];
        const bool V = t0.token_is_row;
        const int bits = V ? o.bits_v : o.bits_k;
        const LType lty = V ? o.lv : o.lk;

        std::vector<float> dA, dB;      // 重建（码域）
        ModeOut O;
        const bool q4k = (o.mode == M_LLAMA || o.mode == M_LLAMA_ROT) && lty == LT_Q4_K;

        if (o.mode == M_KVARN || o.mode == M_ROT_ONLY) {
            O = run_kvarn_slice(t0, bits, iters, o.stage_f16, dA);
            std::vector<float> d1; run_kvarn_slice(t1, bits, iters, o.stage_f16, d1); dB = d1;
        } else if (o.mode == M_BAL_ONLY || o.mode == M_NEITHER) {
            O = run_kvarn_slice(r0, bits, iters, o.stage_f16, dA);
            std::vector<float> d1; run_kvarn_slice(r1, bits, iters, o.stage_f16, d1); dB = d1;
        } else if (o.mode == M_LLAMA) {
            if (q4k) O = run_q4k_pair(r0, r1, o.stage_f16, dA, dB);
            else { O = run_llama_slice(r0, lty, o.stage_f16, dA); std::vector<float> d1; run_llama_slice(r1, lty, o.stage_f16, d1); dB = d1; }
        } else { // M_LLAMA_ROT
            if (q4k) O = run_q4k_pair(t0, t1, o.stage_f16, dA, dB);
            else { O = run_llama_slice(t0, lty, o.stage_f16, dA); std::vector<float> d1; run_llama_slice(t1, lty, o.stage_f16, d1); dB = d1; }
        }

        std::vector<float> refA, refB, mA, mB;
        if (measure_codec_domain) {
            refA = t0.d; refB = t1.d;
            for (auto & v : refA) v = h16(v);
            for (auto & v : refB) v = h16(v);
            mA = dA; mB = dB;
        } else {
            refA = r0.d; refB = r1.d;
            if (o.ref_h16) { for (auto & v : refA) v = h16(v); for (auto & v : refB) v = h16(v); }
            if (codec_in_rot) map_pair_to_raw(t0, t1, dA, dB, mA, mB);
            else { mA = dA; mB = dB; }
        }

        // 主口径：256-dim head 向量（正交变换的不变量）
        const Metrics MP = measure_pair(t0, t1, refA, refB, mA, mB);

        char bbits[64];
        if (o.mode == M_LLAMA || o.mode == M_LLAMA_ROT)
            snprintf(bbits, sizeof bbits, "type=%s", lt_name[lty]);
        else
            snprintf(bbits, sizeof bbits, "bits=%d", bits);

        const int pr = V ? t0.rows : 256;      // 度量空间的 rows/cols（256-dim head）
        const int pc = V ? 256    : t0.cols;
        printf("%s/%-9s %-24s %s %dx%d tok=%s rec=%6zu B bpt=%7.3f | cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min=%.7f relL2=%.5f maxAbs=%.4g\n",
               datatag, mode_name[o.mode], kv.first.c_str(), bbits, pr, pc,
               t0.token_is_row ? "row" : "col", 2 * O.bytes, O.bpt_x1000 / 1000.0,
               MP.cos_global, MP.cos_vec_mean, MP.cos_vec_min, MP.rel_l2, MP.max_abs);

        if (o.slice_metrics) {   // 归档口径：按 128-dim 片测（只用于复刻旧 oracle 数字）
            const Metrics M0 = measure(refA, mA, t0.rows, t0.cols, t0.token_is_row);
            const Metrics M1 = measure(refB, mB, t1.rows, t1.cols, t1.token_is_row);
            printf("SLICE %s/%-9s %-24s %s %dx%d tok=%s | cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min=%.7f relL2=%.5f maxAbs=%.4g\n",
                   datatag, mode_name[o.mode], kv.first.c_str(), bbits, t0.rows, t0.cols,
                   t0.token_is_row ? "row" : "col",
                   M0.cos_global, M0.cos_vec_mean, M0.cos_vec_min, M0.rel_l2, M0.max_abs);
        }

        if (V) {
            sv[0] += MP.cos_global; sv[1] += MP.cos_vec_mean;
            sv[2] += MP.cos_vec_min; sv[3] += MP.rel_l2;
            ++nv; bv += O.bpt_x1000;
        } else {
            sk[0] += MP.cos_global; sk[1] += MP.cos_vec_mean;
            sk[2] += MP.cos_vec_min; sk[3] += MP.rel_l2;
            ++nk; bk += O.bpt_x1000;
        }
    }
    if (nk) printf(">>> %s/%s 汇总(%d 个 256-dim head) K: cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min(worst)=%.7f relL2=%.5f bpt=%.3f\n",
                   datatag, mode_name[o.mode], nk, sk[0]/nk, sk[1]/nk, sk[2]/nk, sk[3]/nk, bk/nk/1000.0);
    if (nv) printf(">>> %s/%s 汇总(%d 个 256-dim head) V: cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min(worst)=%.7f relL2=%.5f bpt=%.3f\n",
                   datatag, mode_name[o.mode], nv, sv[0]/nv, sv[1]/nv, sv[2]/nv, sv[3]/nv, bv/nv/1000.0);
    return 0;
}

// 独立交叉校验用：把固定测试向量的「输入 / 各档反量化输出 / 码流 hex」dump 出来，
// 由 numpy 侧按 ggml-quants.c 重写一遍做逐元素对拍（见 xcheck_numpy.py）。
static int cmd_dump(const char * outpath) {
    const int N = 256;
    std::vector<float> x(N);
    {
        std::mt19937_64 rng(20261013ull);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto & v : x) v = nd(rng);
        x[0] *= 20.f; x[37] *= 12.f; x[128] *= 8.f; x[255] *= 30.f;
    }
    FILE * fp = fopen(outpath, "w");
    if (!fp) { fprintf(stderr, "无法写 %s\n", outpath); return 2; }
    fprintf(fp, "n %d\n", N);
    fprintf(fp, "in");
    for (int i = 0; i < N; ++i) fprintf(fp, " %.9g", x[i]);
    fprintf(fp, "\n");
    for (int t = 0; t < LT_COUNT; ++t) {
        const LType ty = (LType)t;
        std::vector<uint8_t> buf;
        std::vector<float> dq(N);
        const size_t B = llama_qdq(ty, x.data(), N, buf, dq.data());
        fprintf(fp, "out %s %zu", lt_name[t], B);
        for (int i = 0; i < N; ++i) fprintf(fp, " %.9g", dq[i]);
        fprintf(fp, "\n");
        fprintf(fp, "hex %s ", lt_name[t]);
        for (size_t i = 0; i < buf.size(); ++i) fprintf(fp, "%02x", (unsigned)buf[i]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("已写 %s（%d 元素，%d 档）\n", outpath, N, (int)LT_COUNT);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("子命令: selftest [tiles] | bytes | quant <opts> | dump <file>\n"); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "selftest") return cmd_selftest(argc > 2 ? argv[2] : nullptr);
    if (cmd == "bytes")    return cmd_bytes();
    if (cmd == "quant")    return cmd_quant(argc - 2, argv + 2);
    if (cmd == "dump")     return cmd_dump(argc > 2 ? argv[2] : "dump.txt");
    printf("未知子命令 %s\n", cmd.c_str());
    return 2;
}

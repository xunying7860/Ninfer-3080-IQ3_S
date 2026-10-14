// bee_oracle.cpp — 阶段 1：beellama KVarN G128 量化/存储口径的纯标量 oracle
// ---------------------------------------------------------------------------
// 逐行对照三份上游真值（见 ref/，sha256 记在 README.md 与 manifest 里）：
//   kvarn.cu:545  kvarn_std_col / kvarn_std_row
//   kvarn.cu:593  kvarn_reduce_std_ranges
//   kvarn.cu:624  kvarn_update_best_from_std
//   kvarn.cu:660  kvarn_quantize_tile          <- 默认口径（GPU，生产路径）
//   llama-kvarn.cpp:824 llama_kvarn_variance_normalize / :892 llama_kvarn_quantize_tile
//                                              <- 备选口径（CPU，--cpu-order）
//   llama-kvarn.cpp:634 make_layout            <- doc02 的 36 格字节表口径
//   llama-kvarn.cpp:673 make_record_layout     <- 真实 arena 口径（唯一有调用者的那个）
//   llama-kvarn.cpp:709 packed_bytes / :715 pack_bits / :735 unpack_bits_value
//
// 子命令：
//   table            36 组合字节账（doc02 / make_layout / record 三口径逐格对）
//   packtest         pack_bits/unpack_bits_value 对 bits∈{2,3,4,5,6,8} 自检
//   sinktest         中途 dump：Sinkhorn 的 best_col/best_row/best_imbalance
//   quant <opts>     量化→反量化误差（合成 + 真实数据）
//   derive           自适应：从固定 tile 反推口径（本题不用）
//
// quant 选项：
//   --bits-k N --bits-v N      K/V 位宽
//   --group N                  token 组长度（128=新档 / 64=旧实现对照）
//   --iters N                  Sinkhorn 迭代（beellama 默认 16）
//   --cpu-order                用 CPU 口径（llama-kvarn.cpp）而非 GPU 口径
//   --no-stage-f16             不把输入先过一遍 fp16（beellama stage 是 F16，默认要过）
//   --tiles PATH               真实 tile 二进制（make_tiles.py 产出）
//   --gauss / --outlier        合成数据分布
//   --seed N                   合成数据种子（默认 20261013）
// 编译（容器内，见 README）：
//   g++ -O2 -std=c++17 -o bee bee_oracle.cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <climits>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <random>

// ===========================================================================
// 0. fp16：__float2half_rn 的标量等价（round-to-nearest-even，含次正规）
//    与 nvcc 的 __float2half_rn 逐位对拍见 f16_check.cu
// ===========================================================================
static uint16_t f32_to_f16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t exp  = (u >> 23) & 0xFFu;
    const uint32_t m    = u & 0x7FFFFFu;

    if (exp == 0xFFu) {                                  // inf / nan
        if (m == 0) return (uint16_t)(sign | 0x7C00u);
        return (uint16_t)(sign | 0x7E00u);
    }
    const int32_t e = (int32_t)exp - 127;
    if (e > 15) return (uint16_t)(sign | 0x7C00u);       // 溢出 → inf
    if (e >= -14) {                                      // 正规
        const uint32_t mant = m >> 13;
        const uint32_t rest = m & 0x1FFFu;
        uint16_t base = (uint16_t)(sign | ((uint32_t)(e + 15) << 10) | mant);
        if (rest > 0x1000u || (rest == 0x1000u && (mant & 1u))) {
            base = (uint16_t)(base + 1u);                // 进位自然推高指数，最大推到 inf
        }
        return base;
    }
    if (e < -25) return (uint16_t)sign;                  // 下溢 → ±0
    // 次正规：value = sig*2^(e-23)，目标 mh = value*2^24 = sig*2^(e+1)
    const uint32_t sig = m | 0x800000u;
    const int j = -(e + 1);                              // 右移位数，14..24
    uint32_t q = sig >> j;
    const uint32_t rem = sig & ((1u << j) - 1u);
    const uint32_t half = 1u << (j - 1);
    if (rem > half || (rem == half && (q & 1u))) ++q;     // q 到 1024 时自然变成最小正规
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
// beellama 存轴用 __float2half_rn，回读是"被表示过的值"
static inline float h16(float x) { return f16_to_f32(f32_to_f16_rn(x)); }

// ===========================================================================
// 1. 布局：make_layout（doc02 口径）与 make_record_layout（arena 口径）
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

// llama-kvarn.cpp:634 逐行
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
// llama-kvarn.cpp:673 逐行（token_group 恒为 128）
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
// 记录口径下"一个 head 的 128-token 组"= head_slices 份 K 记录 + head_slices 份 V 记录
static size_t record_tile_bytes(int record_dim, int head_slices, int kb, int vb) {
    return size_t(head_slices) * (make_record_layout(record_dim, kb, false).record_bytes
                                + make_record_layout(record_dim, vb, true ).record_bytes);
}

// ===========================================================================
// 2. 位串打包：llama-kvarn.cpp:715 / :735 逐行
// ===========================================================================
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
// 每行独立打包（kvarn.cu:734 的 row_bytes/bit_offset 写法）—— 用于验证与扁平打包等价
static void bee_pack_bits_perrow(const uint8_t * values, int rows, int cols, int bits, uint8_t * dst) {
    const int row_bytes = cols * bits / 8;
    std::memset(dst, 0, size_t(rows) * row_bytes);
    for (int r = 0; r < rows; ++r) {
        uint8_t * row_payload = dst + size_t(r) * row_bytes;
        for (int c = 0; c < cols; ++c) {
            const uint8_t q = values[r * cols + c];
            const int bit_offset = c * bits;
            for (int bit = 0; bit < bits; ++bit) {
                const int dst_bit = bit_offset + bit;
                row_payload[dst_bit / 8] |= uint8_t(((q >> bit) & 1u) << (dst_bit % 8));
            }
        }
    }
}

// ===========================================================================
// 3. Sinkhorn（方差归一化）：GPU 口径（kvarn.cu）与 CPU 口径（llama-kvarn.cpp）
//    参数化 R（行数）/C（列数）：beellama 恒为 128/128；G64 对照时 C=64。
//    std 分母 = 轴长-1（样本标准差），照抄。
// ===========================================================================
struct Sink { std::vector<float> s_col, s_row; float best_imbalance; int best_iter; };

static inline float std_of(const float * p, int n, int stride, const float * s_col, const float * s_row,
                           int row_index, bool is_col) {
    // is_col: p 是某一列（沿行跳 stride），除以 s_col[col]*s_row[row]
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
        if (getenv("BEE_DEBUG")) {
            float cmin = FLT_MAX, cmax = 0.f, rmin = FLT_MAX, rmax = 0.f;
            for (int c = 0; c < C; ++c) { cmin = fminf(cmin, col_std[c]); cmax = fmaxf(cmax, col_std[c]); }
            for (int r = 0; r < R; ++r) { rmin = fminf(rmin, row_std[r]); rmax = fmaxf(rmax, row_std[r]); }
            printf("  [gpu] iter=%d imb=%.9g cmin=%.9g cmax=%.9g rmin=%.9g rmax=%.9g s_col[0]=%.9g s_row[0]=%.9g\n",
                   iter, cmax / fmaxf(cmin, 1e-8f) + rmax / fmaxf(rmin, 1e-8f), cmin, cmax, rmin, rmax, s_col[0], s_row[0]);
        }
        update_best(iter);
    }
    out.s_col = best_col; out.s_row = best_row;
}

// llama-kvarn.cpp:790/804/824 逐行（std 用 double 累加，imbalance 同样口径）
static float sample_std_d(const float * p, int n, int stride) {
    double sum = 0.0, sum_sq = 0.0;
    for (int i = 0; i < n; ++i) { const double v = p[i * stride]; sum += v; sum_sq += v * v; }
    const double mean = sum / n;
    const double var = std::max(0.0, (sum_sq - n * mean * mean) / (n - 1));
    return float(std::sqrt(var));
}
static void sinkhorn_cpu(const float * tile, int R, int C, int iters, Sink & out) {
    std::vector<float> log_s_col(C, 0.f), log_s_row(R, 0.f);
    std::vector<float> cur(tile, tile + size_t(R) * C);
    std::vector<float> best_col(C, 1.f), best_row(R, 1.f);
    auto imbalance = [&](const std::vector<float> & t) {
        float cmin = INFINITY, cmax = 0.f, rmin = INFINITY, rmax = 0.f;
        for (int c = 0; c < C; ++c) { const float v = sample_std_d(t.data() + c, R, C); cmin = std::min(cmin, v); cmax = std::max(cmax, v); }
        for (int r = 0; r < R; ++r) { const float v = sample_std_d(t.data() + r * C, C, 1); rmin = std::min(rmin, v); rmax = std::max(rmax, v); }
        return cmax / std::max(cmin, 1e-8f) + rmax / std::max(rmin, 1e-8f);
    };
    float imb_best = imbalance(cur);
    out.best_imbalance = imb_best; out.best_iter = -1;
    auto rebuild = [&]() {
        for (int r = 0; r < R; ++r) {
            const float sr = std::exp(log_s_row[r]);
            for (int c = 0; c < C; ++c) cur[size_t(r) * C + c] = tile[size_t(r) * C + c] / (std::exp(log_s_col[c]) * sr);
        }
    };
    for (int iter = 0; iter < iters; ++iter) {
        for (int c = 0; c < C; ++c) {
            const float sd = std::clamp(sample_std_d(cur.data() + c, R, C), 1e-3f, 1e3f);
            log_s_col[c] = std::clamp(log_s_col[c] + std::log(sd), -0.3f, 10.0f);
        }
        rebuild();
        for (int r = 0; r < R; ++r) {
            const float sd = std::clamp(sample_std_d(cur.data() + size_t(r) * C, C, 1), 1e-3f, 1e3f);
            log_s_row[r] = std::clamp(log_s_row[r] + std::log(sd), -0.3f, 10.0f);
        }
        rebuild();
        const float imb = imbalance(cur);
        if (getenv("BEE_DEBUG")) printf("  [cpu] iter=%d imb=%.9g cur[0]=%.9g cur[129]=%.9g s_col[0]=%.9g s_row[0]=%.9g\n",
                                        iter, imb, cur[0], cur[129], std::exp(log_s_col[0]), std::exp(log_s_row[0]));
        if (imb <= imb_best) {
            imb_best = imb; out.best_iter = iter;
            for (int i = 0; i < C; ++i) best_col[i] = std::exp(log_s_col[i]);
            for (int i = 0; i < R; ++i) best_row[i] = std::exp(log_s_row[i]);
        }
    }
    out.best_imbalance = imb_best;   // <- 修复：原来漏了这一行，报告的初值会被误当成本次最优
    out.s_col = best_col; out.s_row = best_row;
}

// ===========================================================================
// 4. 量化 / 反量化
//    record 布局（arena 口径，records->ne[0] 被 GGML_ASSERT 锁定）：
//      [0, payload)                        codes（按 row 打包，row_bytes = cols*bits/8）
//      payload + r*2                       scale_axis[r] = fp16(best_row[r]*scale)
//      payload + rows*2 + r*2              zp_axis[r]    = fp16(best_row[r]*lo)
//      payload + 2*rows*2 + k*2            other_axis[k] = fp16(best_col[k])
//    反量化：x[r][c] = (q*scale_axis[r] + zp_axis[r]) * other_axis[c]
// ===========================================================================
struct QuantMeta { std::vector<float> best_col, best_row; std::vector<uint8_t> q; float best_imbalance; int best_iter; };

static void bee_quant_tile(const float * tile, int rows, int cols, int bits, int iters,
                           std::vector<uint8_t> & record, QuantMeta * meta, bool cpu_order) {
    Sink S;
    if (cpu_order) sinkhorn_cpu(tile, rows, cols, iters, S);
    else           sinkhorn_gpu(tile, rows, cols, iters, S);

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
        fread(&nl, 4, 1, fp);
        std::string nm(nl, '\0'); fread(&nm[0], 1, nl, fp);
        fread(&rr, 4, 1, fp); fread(&cc, 4, 1, fp); fread(&tr, 4, 1, fp);
        Tile t; t.name = nm; t.rows = rr; t.cols = cc; t.token_is_row = (tr != 0);
        t.d.resize(size_t(rr) * cc);
        fread(t.d.data(), 4, t.d.size(), fp);
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
        // 模拟"离群通道"（§21 的 6% 提位场景）：3 行 ×20，另 1 行进 heavy tail
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

// ===========================================================================
// 6. 误差指标
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
    // 逐 token 向量 cosine
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
// 7. 子命令
// ===========================================================================
static const int KB_LIST[6] = {2, 3, 4, 5, 6, 8};
static const int DOC02[6][6] = {
    {146, 178, 210, 242, 274, 338},
    {178, 210, 242, 274, 306, 370},
    {210, 242, 274, 306, 338, 402},
    {242, 274, 306, 338, 370, 434},
    {274, 306, 338, 370, 402, 466},
    {338, 370, 402, 434, 466, 530},
};

static int cmd_table() {
    printf("== A. 36 组合字节账（D=256, G128；单位 B/token/KV head，= tile_bytes/128） ==\n");
    printf("口径① doc02 = beellama make_layout(head_dim=256, group=128)\n");
    printf("口径② record = beellama make_record_layout(record_dim=128) x head_slices(2)  <- arena 真口径\n\n");
    printf("Kb Vb | doc02 | mL(B/tok) | d | mL(tile) | rec(B/tok) | d | rec(tile) | Krec | Vrec\n");
    printf("------+-------+-----------+---+----------+------------+---+-----------+------+------\n");
    int mismatch_doc = 0, mismatch_ml = 0;
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        const int kb = KB_LIST[i], vb = KB_LIST[j];
        const TileLayout L = bee_layout(256, 128, kb, vb);
        const size_t rk = make_record_layout(128, kb, false).record_bytes;
        const size_t rv = make_record_layout(128, vb, true ).record_bytes;
        const size_t tot = record_tile_bytes(128, 2, kb, vb);
        const int ml_tok = (int)(L.tile_bytes / 128);
        const int rec_tok = (int)(tot / 128);
        if (ml_tok != DOC02[i][j]) ++mismatch_doc;
        if (rec_tok == DOC02[i][j]) ++mismatch_ml;   // 计数：record 口径反而等于 doc02 的格数（应为 0）
        printf("%2d %2d | %5d | %9d |%+3d| %8zu | %10d |%+3d| %9zu | %4zu | %4zu\n",
               kb, vb, DOC02[i][j], ml_tok, ml_tok - DOC02[i][j], L.tile_bytes,
               rec_tok, rec_tok - DOC02[i][j], tot, rk, rv);
    }
    printf("\ndoc02 表 vs make_layout 不一致格数 = %d / 36\n", mismatch_doc);
    printf("record 口径 = doc02 - 6 B/token（每格）；record 口径等于 doc02 的格数 = %d / 36\n", mismatch_ml);
    printf("\n== B. 每格 tile 码流长度与三段偏移（record 口径，record_dim=128） ==\n");
    printf("Kb Vb | K: payload scale_off zp_off other_off total | V: payload scale_off zp_off other_off total | K段起点 V段起点 tile总\n");
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        const int kb = KB_LIST[i], vb = KB_LIST[j];
        const RecordLayout RK = make_record_layout(128, kb, false);
        const RecordLayout RV = make_record_layout(128, vb, true);
        const size_t kseg = 2 * RK.record_bytes, vseg = 2 * RV.record_bytes;
        int dummy_ok = (int)(RK.other_off + size_t(RK.cols) * 2 == RK.record_bytes);
        bool ok = dummy_ok && (RV.other_off + size_t(RV.cols) * 2 == RV.record_bytes)
               && RK.scale_off == RK.payload_bytes
               && RK.zp_off == RK.scale_off + size_t(RK.rows) * 2
               && RV.scale_off == RV.payload_bytes
               && RV.zp_off == RV.scale_off + size_t(RV.rows) * 2
               && RK.rows == 128 && RK.cols == 128 && RV.rows == 128 && RV.cols == 128;
        if (!ok) { printf("%2d %2d | 偏移自检失败!\n", kb, vb); continue; }
        printf("%2d %2d | %7zu %9zu %6zu %9zu %5zu | %7zu %9zu %6zu %9zu %5zu | %6d %6zu %6zu\n",
               kb, vb, RK.payload_bytes, RK.scale_off, RK.zp_off, RK.other_off, RK.record_bytes,
               RV.payload_bytes, RV.scale_off, RV.zp_off, RV.other_off, RV.record_bytes,
               0, kseg, kseg + vseg);
    }
    printf("\n== C. tile 口径（make_layout）八段偏移示范：K4V2_g128 ==\n");
    {
        const TileLayout L = bee_layout(256, 128, 4, 2);
        printf("  k_payload off=%zu bytes=%zu\n", L.k_payload_off, L.k_payload_bytes);
        printf("  k_s_col   off=%zu bytes=%d\n", L.k_s_col_off, 512);
        printf("  k_zp      off=%zu bytes=%d\n", L.k_zp_off, 512);
        printf("  k_s_row   off=%zu bytes=%d\n", L.k_s_row_off, 256);
        printf("  v_payload off=%zu bytes=%zu\n", L.v_payload_off, L.v_payload_bytes);
        printf("  v_s_col   off=%zu bytes=%d\n", L.v_s_col_off, 512);
        printf("  v_s_row   off=%zu bytes=%d\n", L.v_s_row_off, 256);
        printf("  v_zp      off=%zu bytes=%d\n", L.v_zp_off, 256);
        printf("  tile_bytes=%zu (align8) = %d B/token\n", L.tile_bytes, (int)(L.tile_bytes / 128));
        printf("  record 口径同档 = %zu = %d B/token  差 %zu B\n",
               record_tile_bytes(128, 2, 4, 2), (int)(record_tile_bytes(128, 2, 4, 2) / 128),
               record_tile_bytes(128, 2, 4, 2) - L.tile_bytes);
    }
    printf("\n== D. G64 对照（doc02 §2.3 的表 = make_layout(head_dim=256, group=64)） ==\n");
    printf("Kb Vb | doc02_G64 | mL_G64(B/tok) | G128(mL) | 差\n");
    for (int i = 0; i < 6; ++i) {
        const int kb = 4, vb = KB_LIST[i];
        const TileLayout L64 = bee_layout(256, 64, kb, vb);
        const TileLayout L128 = bee_layout(256, 128, kb, vb);
        printf("%2d %2d |         - | %13d | %8d | %+3d\n", kb, vb,
               (int)(L64.tile_bytes / 64), (int)(L128.tile_bytes / 128),
               (int)(L128.tile_bytes / 128) - (int)(L64.tile_bytes / 64));
    }
    return 0;
}

static int cmd_packtest() {
    printf("== pack_bits / unpack_bits_value 自检（bits 全覆盖）+ 每行打包等价性 ==\n");
    int fail = 0;
    for (int bits : {2, 3, 4, 5, 6, 8}) {
        for (int rows : {128}) for (int cols : {128, 64}) {
            const int qmax = (1 << bits) - 1;
            std::vector<uint8_t> q(size_t(rows) * cols);
            std::mt19937_64 rng(1234 + bits * 10 + cols);
            for (auto & v : q) v = (uint8_t)(rng() % (qmax + 1));
            std::vector<uint8_t> flat(bee_packed_bytes(rows * cols, bits), 0), perrow(size_t(rows) * (cols * bits / 8), 0);
            bee_pack_bits(q.data(), rows * cols, bits, flat.data());
            bee_pack_bits_perrow(q.data(), rows, cols, bits, perrow.data());
            bool same = (flat == perrow);
            bool rt = true;
            for (int i = 0; i < rows * cols; ++i) if (bee_unpack_bits_value(flat.data(), i, bits) != q[i]) { rt = false; break; }
            printf("bits=%d rows=%d cols=%d bytes=%zu  roundtrip=%s  flat==perrow=%s\n",
                   bits, rows, cols, flat.size(), rt ? "OK" : "FAIL", same ? "OK" : "FAIL");
            if (!rt || !same) ++fail;
        }
    }
    printf("失败项 = %d\n", fail);
    return fail ? 1 : 0;
}

static int cmd_sinktest(const char * tilepath, int rows, int cols, int bits, int iters) {
    std::vector<float> d;
    if (tilepath) {
        auto ts = load_tiles_bin(tilepath);
        if (ts.empty()) { printf("无 tile\n"); return 2; }
        d = ts[0].d; rows = ts[0].rows; cols = ts[0].cols;
    } else fill_gauss(d, rows, cols, 20261013ull, false);
    Sink g, c;
    sinkhorn_gpu(d.data(), rows, cols, iters, g);
    sinkhorn_cpu(d.data(), rows, cols, iters, c);
    printf("Sinkhorn dump (%dx%d, iters=%d)\n", rows, cols, iters);
    printf("  GPU-order: best_imbalance=%.9g best_iter=%d\n", g.best_imbalance, g.best_iter);
    printf("  CPU-order: best_imbalance=%.9g best_iter=%d\n", c.best_imbalance, c.best_iter);
    double dm = 0, dr = 0;
    for (int i = 0; i < cols; ++i) dm = std::max(dm, std::fabs((double)g.s_col[i] - c.s_col[i]));
    for (int i = 0; i < rows; ++i) dr = std::max(dr, std::fabs((double)g.s_row[i] - c.s_row[i]));
    printf("  max|s_col_gpu - s_col_cpu| = %.3e   max|s_row_gpu - s_row_cpu| = %.3e\n", dm, dr);
    printf("  s_col[0..3] = %.7g %.7g %.7g %.7g\n", g.s_col[0], g.s_col[1], g.s_col[2], g.s_col[3]);
    printf("  s_row[0..3] = %.7g %.7g %.7g %.7g\n", g.s_row[0], g.s_row[1], g.s_row[2], g.s_row[3]);
    printf("\n  逐迭代对照（iters=0 即只算初值，用来判断两口径的 imbalance 定义是否同源）:\n");
    printf("  iters |  GPU best_imbalance  best_iter |  CPU best_imbalance  best_iter | maxd_s_col  maxd_s_row\n");
    for (int it = 0; it <= iters; ++it) {
        Sink gg, cc;
        sinkhorn_gpu(d.data(), rows, cols, it, gg);
        sinkhorn_cpu(d.data(), rows, cols, it, cc);
        double m1 = 0, m2 = 0;
        for (int i = 0; i < cols; ++i) m1 = std::max(m1, std::fabs((double)gg.s_col[i] - cc.s_col[i]));
        for (int i = 0; i < rows; ++i) m2 = std::max(m2, std::fabs((double)gg.s_row[i] - cc.s_row[i]));
        printf("  %5d | %19.9g %10d | %19.9g %10d | %.3e %.3e\n",
               it, gg.best_imbalance, gg.best_iter, cc.best_imbalance, cc.best_iter, m1, m2);
    }
    return 0;
}

static int cmd_quant(int argc, char ** argv) {
    int bits_k = 4, bits_v = 2, group = 128, iters = 16;
    bool cpu_order = false, stage_f16 = true, use_gauss = false, use_outlier = false;
    uint64_t seed = 20261013ull;
    const char * tilepath = nullptr;
    for (int i = 0; i < argc; ++i) {
        if (!strcmp(argv[i], "--bits-k") && i + 1 < argc) bits_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits-v") && i + 1 < argc) bits_v = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group") && i + 1 < argc) group = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters") && i + 1 < argc) iters = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cpu-order")) cpu_order = true;
        else if (!strcmp(argv[i], "--no-stage-f16")) stage_f16 = false;
        else if (!strcmp(argv[i], "--tiles") && i + 1 < argc) tilepath = argv[++i];
        else if (!strcmp(argv[i], "--gauss")) use_gauss = true;
        else if (!strcmp(argv[i], "--outlier")) { use_outlier = true; use_gauss = true; }
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
    }
    printf("# 口径: %s   group=%d iters=%d stage_f16=%d\n",
           cpu_order ? "CPU(llama-kvarn.cpp:824)" : "GPU(kvarn.cu:660)", group, iters, (int)stage_f16);

    std::vector<Tile> tiles;
    if (tilepath) tiles = load_tiles_bin(tilepath);

    auto run_one = [&](const char * name, int rows, int cols, bool token_is_row, int bits,
                       std::vector<float> orig) {
        if (stage_f16) for (auto & v : orig) v = h16(v);      // beellama stage 是 F16
        std::vector<uint8_t> rec;
        bee_quant_tile(orig.data(), rows, cols, bits, iters, rec, nullptr, cpu_order);
        std::vector<float> deq(size_t(rows) * cols);
        bee_dequant_tile(rec.data(), rows, cols, bits, deq.data());
        std::vector<float> thin(size_t(rows) * cols);
        // 只过 fp16、不量化的对照基线（stage 舍入本身贡献多少）
        for (size_t i = 0; i < orig.size(); ++i) thin[i] = h16(orig[i]);
        const Metrics M = measure(thin, deq, rows, cols, token_is_row);
        printf("%-26s bits=%d %dx%d rec=%5zu B | cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min=%.7f relL2=%.5f maxAbs=%.4g maxRel=%.3g\n",
               name, bits, rows, cols, rec.size(), M.cos_global, M.cos_vec_mean, M.cos_vec_min, M.rel_l2, M.max_abs, M.max_rel);
        (void)use_gauss; (void)use_outlier;
        return M;
    };

    auto split_run = [&](const char * tag, std::vector<float> dk, std::vector<float> dv,
                         int dim, int ntok, int nheads) {
        // dk: [nheads][ntok][dim]  (K: row=dim, col=token)，dv 同布局
        std::vector<Metrics> mk, mv;
        for (int h = 0; h < nheads; ++h) {
            for (int g = 0; g * group + group <= ntok; ++g) {
                // K: 取 dim 切片 [slice*128, slice*128+128) x token[group]
                for (int slice = 0; slice * 128 + 128 <= dim; ++slice) {
                    std::vector<float> tk(size_t(128) * group);
                    for (int r = 0; r < 128; ++r)
                        for (int c = 0; c < group; ++c)
                            tk[size_t(r) * group + c] = dk[(size_t(h) * ntok + g * group + c) * dim + slice * 128 + r];
                    char nm[128]; snprintf(nm, sizeof nm, "%s_k_s%d_g%d", tag, slice, g);
                    mk.push_back(run_one(nm, 128, group, false, bits_k, tk));
                    // V: row=token, col=dim → 转置
                    std::vector<float> tv(size_t(group) * 128);
                    for (int r = 0; r < group; ++r)
                        for (int c = 0; c < 128; ++c)
                            tv[size_t(r) * 128 + c] = dv[(size_t(h) * ntok + g * group + r) * dim + slice * 128 + c];
                    char nm2[128]; snprintf(nm2, sizeof nm2, "%s_v_s%d_g%d", tag, slice, g);
                    mv.push_back(run_one(nm2, group, 128, true, bits_v, tv));
                }
            }
        }
        auto avg = [](const std::vector<Metrics> & v, int which) {
            double s = 0; for (auto & m : v) s += which == 0 ? m.cos_global : (which == 1 ? m.cos_vec_mean : m.cos_vec_min);
            return s / v.size();
        };
        printf(">>> %s 汇总(%zu tiles): K cos_glob=%.7f cos_vec_mean=%.7f | V cos_glob=%.7f cos_vec_mean=%.7f\n",
               tag, mk.size(), avg(mk, 0), avg(mk, 1), avg(mv, 0), avg(mv, 1));
    };

    if (tilepath) {
        // 真实数据：make_tiles.py 已按 beellama 口径（WHT128 x2 + 跨 slice）旋好并切好
        for (auto & t : tiles) {
            std::vector<float> o = t.d;
            if (stage_f16) for (auto & v : o) v = h16(v);
            const int bits = (t.name.find("_v_") != std::string::npos) ? bits_v : bits_k;
            std::vector<uint8_t> rec;
            bee_quant_tile(o.data(), t.rows, t.cols, bits, iters, rec, nullptr, cpu_order);
            std::vector<float> deq(o.size());
            bee_dequant_tile(rec.data(), t.rows, t.cols, bits, deq.data());
            std::vector<float> thin(o.size());
            for (size_t i = 0; i < o.size(); ++i) thin[i] = h16(o[i]);
            const Metrics M = measure(thin, deq, t.rows, t.cols, t.token_is_row);
            printf("%-30s bits=%d %dx%d %s rec=%5zu B | cos_glob=%.7f cos_vec_mean=%.7f cos_vec_min=%.7f relL2=%.5f maxAbs=%.4g\n",
                   t.name.c_str(), bits, t.rows, t.cols, t.token_is_row ? "tok=row" : "tok=col",
                   rec.size(), M.cos_global, M.cos_vec_mean, M.cos_vec_min, M.rel_l2, M.max_abs);
        }
        return 0;
    }

    // 合成数据
    const int nh = 4, nt = 256, dm = 256;
    std::vector<float> dk(size_t(nh) * nt * dm), dv(size_t(nh) * nt * dm);
    for (int h = 0; h < nh; ++h) {
        std::vector<float> a, b;
        fill_gauss(a, nt, dm, seed + h * 977ull, use_outlier);
        fill_gauss(b, nt, dm, seed + 100000 + h * 977ull, use_outlier);
        std::copy(a.begin(), a.end(), dk.begin() + size_t(h) * nt * dm);
        std::copy(b.begin(), b.end(), dv.begin() + size_t(h) * nt * dm);
    }
    split_run(use_outlier ? "synth_outlier" : "synth_gauss", dk, dv, dm, nt, nh);
    return 0;
}

// ===========================================================================
// [GATE2 追加段 —— 本文件相对上游 bee_oracle.cpp 的唯一改动]
// 新增子命令 dumprec：把 beellama 量化口径的【输入 tile（过 f16 后）】+
// 【record 全文】+【best_col/best_row/best_iter/best_imbalance】落盘，
// 供 CUDA kernel 侧逐字节对拍。
//   原文件 sha256 = 2c1cb2470a21e0a5965a3d9bcb431b62d6bbd14c101944cfd07257a565fda3a2
//   路径        = kvarn-全档位移植-20261013/oracle/bee_oracle.cpp（只读复制，未修改）
// 用法：./dump <out.bin> [--tiles PATH | --gauss | --outlier]
//                       [--bits-k N] [--bits-v N] [--iters N] [--seed N]
//                       [--no-stage-f16]      (group 恒 128)
// 落盘格式（小端）：u32 magic='GVK2' | u32 ver=2 | u32 n
//   per rec: u32 namelen, name[], i32 rows,i32 cols,i32 bits,i32 iters,
//            i32 best_iter, f32 best_imbalance,
//            f32 s_col[cols], f32 s_row[rows], f32 tile[rows*cols],
//            u32 rec_len, u8 rec[rec_len]
// ===========================================================================
struct RecDump {
    std::string name; int rows, cols, bits, iters, best_iter; float best_imb;
    std::vector<float> s_col, s_row, tile; std::vector<uint8_t> rec;
};
static void g2_w_u32(FILE * f, uint32_t v) { fwrite(&v, 4, 1, f); }
static void g2_w_i32(FILE * f, int32_t v)  { fwrite(&v, 4, 1, f); }
static void g2_w_f32(FILE * f, const float * p, int n) { fwrite(p, 4, size_t(n), f); }

static int cmd_dumprec(int argc, char ** argv) {
    if (argc < 1) { fprintf(stderr, "用法: dumprec <out.bin> [...]\n"); return 2; }
    const char * outpath = argv[0];
    int bits_k = 4, bits_v = 4, group = 128, iters = 16;
    bool stage_f16 = true, use_gauss = false, use_outlier = false;
    uint64_t seed = 20261013ull;
    const char * tilepath = nullptr;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "--bits-k") && i + 1 < argc) bits_k = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bits-v") && i + 1 < argc) bits_v = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--group")  && i + 1 < argc) group  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")  && i + 1 < argc) iters  = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-stage-f16")) stage_f16 = false;
        else if (!strcmp(argv[i], "--tiles")  && i + 1 < argc) tilepath = argv[++i];
        else if (!strcmp(argv[i], "--gauss")) use_gauss = true;
        else if (!strcmp(argv[i], "--outlier")) { use_outlier = true; use_gauss = true; }
        else if (!strcmp(argv[i], "--seed")   && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else { fprintf(stderr, "未知参数 %s\n", argv[i]); return 2; }
    }
    (void)use_gauss;
    if (group != 128) { fprintf(stderr, "dumprec 只支持 group=128（kernel 恒 128x128）\n"); return 2; }

    std::vector<RecDump> recs;
    auto push = [&](const char * name, int rows, int cols, int bits, std::vector<float> tile) {
        if (stage_f16) for (auto & v : tile) v = h16(v);   // beellama stage 是 F16
        RecDump D; D.name = name; D.rows = rows; D.cols = cols; D.bits = bits; D.iters = iters;
        QuantMeta meta;
        bee_quant_tile(tile.data(), rows, cols, bits, iters, D.rec, &meta, false);  // GPU 口径
        D.best_iter = meta.best_iter; D.best_imb = meta.best_imbalance;
        D.s_col = meta.best_col; D.s_row = meta.best_row; D.tile = std::move(tile);
        recs.push_back(std::move(D));
    };

    if (tilepath) {
        auto tiles = load_tiles_bin(tilepath);
        for (auto & t : tiles) {
            const int bits = (t.name.find("_v_") != std::string::npos) ? bits_v : bits_k;
            push(t.name.c_str(), t.rows, t.cols, bits, t.d);
        }
    } else {
        const int nh = 4, nt = 256, dm = 256;
        for (int h = 0; h < nh; ++h) {
            std::vector<float> a, b;
            fill_gauss(a, nt, dm, seed + h * 977ull, use_outlier);
            fill_gauss(b, nt, dm, seed + 100000 + h * 977ull, use_outlier);
            for (int g = 0; g * group + group <= nt; ++g)
            for (int slice = 0; slice * 128 + 128 <= dm; ++slice) {
                std::vector<float> tk(size_t(128) * group), tv(size_t(group) * 128);
                for (int r = 0; r < 128; ++r) for (int c = 0; c < group; ++c)
                    tk[size_t(r) * group + c] = a[(size_t(g * group + c) * dm) + slice * 128 + r];
                for (int r = 0; r < group; ++r) for (int c = 0; c < 128; ++c)
                    tv[size_t(r) * 128 + c] = b[(size_t(g * group + r) * dm) + slice * 128 + c];
                char nm[128];
                snprintf(nm, sizeof nm, "%s_k_h%d_s%d_g%d", use_outlier ? "outlier" : "gauss", h, slice, g);
                push(nm, 128, group, bits_k, std::move(tk));
                snprintf(nm, sizeof nm, "%s_v_h%d_s%d_g%d", use_outlier ? "outlier" : "gauss", h, slice, g);
                push(nm, group, 128, bits_v, std::move(tv));
            }
        }
    }

    FILE * fp = fopen(outpath, "wb");
    if (!fp) { fprintf(stderr, "无法写 %s\n", outpath); return 2; }
    g2_w_u32(fp, 0x324B5647u); g2_w_u32(fp, 2u); g2_w_u32(fp, (uint32_t)recs.size());
    for (auto & D : recs) {
        g2_w_u32(fp, (uint32_t)D.name.size());
        fwrite(D.name.data(), 1, D.name.size(), fp);
        g2_w_i32(fp, D.rows); g2_w_i32(fp, D.cols); g2_w_i32(fp, D.bits); g2_w_i32(fp, D.iters);
        g2_w_i32(fp, D.best_iter); fwrite(&D.best_imb, 4, 1, fp);
        g2_w_f32(fp, D.s_col.data(), D.cols);
        g2_w_f32(fp, D.s_row.data(), D.rows);
        g2_w_f32(fp, D.tile.data(), D.rows * D.cols);
        g2_w_u32(fp, (uint32_t)D.rec.size());
        fwrite(D.rec.data(), 1, D.rec.size(), fp);
    }
    fclose(fp);
    printf("dumprec: %zu records -> %s  (bits_k=%d bits_v=%d iters=%d stage_f16=%d)\n",
           recs.size(), outpath, bits_k, bits_v, iters, (int)stage_f16);
    return 0;
}

int main(int argc, char ** argv) {
    if (argc < 2) { printf("子命令: table | packtest | sinktest [tilebin rows cols bits iters] | quant <opts> | dumprec <out.bin> <opts>\n"); return 2; }
    const std::string cmd = argv[1];
    if (cmd == "dumprec")  return cmd_dumprec(argc - 2, argv + 2);
    if (cmd == "table")    return cmd_table();
    if (cmd == "packtest") return cmd_packtest();
    if (cmd == "sinktest") {
        const char * tp = (argc > 2) ? argv[2] : nullptr;
        return cmd_sinktest(tp, argc > 3 ? atoi(argv[3]) : 128, argc > 4 ? atoi(argv[4]) : 128,
                            argc > 5 ? atoi(argv[5]) : 4, argc > 6 ? atoi(argv[6]) : 16);
    }
    if (cmd == "quant")    return cmd_quant(argc - 2, argv + 2);
    printf("未知子命令 %s\n", cmd.c_str());
    return 2;
}

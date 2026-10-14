// kvarn_g128_verify.cpp —— KVarN G128 引擎级对拍工具（"record 解码 == 主机反量化"）
// ===========================================================================
// 目的：把"喂给 attention 的那份值"从未被对拍过的教训（实机取证-20261013.md §10/§11）
//       变成一条可复现的判据：
//         主机反量化(record)  ==  record 解码(引擎口径，含列因子/行 scale 折叠 + 逆旋转)
//       逐向量（每个 token 的 D256）cosine、最大绝对/相对误差、非有限值计数。
//
// 三个输入入口（同一套分析，同一套输出）：
//   --oracle <dump.bin>     GATE2 的 dumprec 文件（magic 'GVK2' v2）：
//                           每条含 tile(量化前原值,旋转域) + rec(record 全文) + s_col/s_row
//   --from-dump <file>      KVD1 dump（本工具定义的引擎 plane dump 最小格式，见 README）
//                           一段 record 码字节 + 一段 scale 字节 + 页数/token 数
//   --sweep                 内部造全矩阵（K4V4/K4V2 × {128,127,51,1} token × {gauss,outlier}）
//
// 判据（详见 README §判据）：
//   cos(record解码, 主机反量化) ≥ 0.999  —— 这条不过 = 解码端（引擎读路径）病
//   cos(主机反量化, 原值)      ≥ 0.999  —— 这条不过 = 编码端（量化/旋转/存储）病
// ===========================================================================
//
// 只读 import（逐字复制，未改动）自：
//   H:\Hermes工作区\Hermes\kvarn-全档位移植-20261013\gate2-kvarn-quant\dump_oracle.cpp
//     f32_to_f16_rn / f16_to_f32 / h16                  (其 §0,  :49-100)
//     bee_packed_bytes / bee_pack_bits / bee_unpack_bits_value (其 §2, :110-172)
//     std_of / sinkhorn_gpu                             (其 §3,  :197-255)
//     QuantMeta / bee_quant_tile                        (其 §4,  :316-357)
//     bee_dequant_tile                                  (其 §4,  :359-374)
//     fill_gauss                                        (其 §5,  :401-418)
//   因为该文件本身是 oracle/bee_oracle.cpp 的只读复制 + dumprec 追加段，所以这里
//   的复制 = 对 oracle 的只读 import。run.sh 的 --selftest 会把这些复制再跑一遍，
//   逐字节比回 GATE2 dump，证明副本没有走样。
//
// 本文件新增（不来自 oracle）：
//   * norm_hadamard_256 —— ninfer 侧旋转 normalized_hadamard_d256_inplace 的标量等价
//                         （两道gate结论-20261013.md §gate1 已定案：照搬 ninfer 侧）
//   * engine_decode_*   —— 引擎口径的解码：按行字节寻址(merge) + 逐 token 组装 D256
//                         + 列因子/行 scale 折叠（§2.2）+ 逆旋转
//   * 指标 / 报告 / 三种 reader / KVD1 dump 读写 / sweep
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <string>
#include <vector>
#include <algorithm>
#include <random>

// ===========================================================================
// 【只读 import 区开始】—— 逐字来自 oracle（dump_oracle.cpp），勿改
// ===========================================================================

// ---- 0. fp16：__float2half_rn 的标量等价（round-to-nearest-even，含次正规）----
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

// ---- 2. 位串打包：llama-kvarn.cpp:715 / :735 逐行 ----
static size_t bee_packed_bytes(int n_values, int bits) {
    return (size_t(n_values) * size_t(bits) + 7) / 8;
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

// ---- 3. Sinkhorn（方差归一化）：GPU 口径（kvarn.cu）----
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

// ---- 4. 量化/反量化：llama-kvarn.cpp:999/1019 + kvarn.cu materialize 的读法 ----
//   反量化：x[r][c] = (q*scale_axis[r] + zp_axis[r]) * other_axis[c]
struct QuantMeta { std::vector<float> best_col, best_row; std::vector<uint8_t> q; float best_imbalance; int best_iter; };

static void bee_quant_tile(const float * tile, int rows, int cols, int bits, int iters,
                           std::vector<uint8_t> & record, QuantMeta * meta, bool cpu_order) {
    (void)cpu_order;                       // 本工具只用 GPU 口径（生产路径）
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

// ---- 5. 数据 ----
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
// 【只读 import 区结束】
// ===========================================================================


// ===========================================================================
// 6. 旋转：normalized Hadamard-256（ninfer `normalized_hadamard_d256_inplace` 的标量等价）
//    两道gate结论-20261013.md §gate1 定案：结构层与 beellama 的
//    (kvarn_wht_128 ×0.08838834764831845) + (cross-slice H2 ×0.7071067811865475) 逐位相同，
//    且耦合映射是恒等 map[j]=j ⇒ D256 全程就这一个正交自逆变换（H256/16，(H/16)²=I）。
//    逆旋转 == 正旋转（自逆）。
// ===========================================================================
static constexpr int D256 = 256;
static void norm_hadamard_256(float * v) {
    for (int s = 1; s < D256; s <<= 1) {
        for (int i = 0; i < D256; i += 2 * s) {
            for (int j = i; j < i + s; ++j) {
                const float a = v[j], b = v[j + s];
                v[j] = a + b; v[j + s] = a - b;
            }
        }
    }
    for (int i = 0; i < D256; ++i) v[i] *= 0.0625f;      // 1/16
}


// ===========================================================================
// 7. 解码器
// ===========================================================================
// RecView：一条 record 拆成"码平面 + scale 平面"（= KVD1 dump 的两段字节）
//   code  -> payload（rows*cols*bits/8 字节，低位先）
//   sc/zp -> 每行 1 个 fp16（scale_axis / zp_axis，已折进行因子 best_row）
//   other -> 每列 1 个 fp16（列因子 best_col）
struct RecView {
    const uint8_t * code = nullptr;
    const uint16_t * sc = nullptr;
    const uint16_t * zp = nullptr;
    const uint16_t * other = nullptr;
    int rows = 0, cols = 0, bits = 0;
    bool is_v = false;                 // false=K(row=dim,col=token) true=V(row=token,col=dim)
};

// --- 主机反量化（顺序、逐元素；= oracle 的 bee_dequant_tile 读法）---
static inline float host_elem(const RecView & R, int row, int col) {
    const uint8_t q = bee_unpack_bits_value(R.code, row * R.cols + col, R.bits);
    return (float(q) * f16_to_f32(R.sc[row]) + f16_to_f32(R.zp[row])) * f16_to_f32(R.other[col]);
}

// --- 引擎口径解码（按"行内字节 + 半字节槽位"寻址，merge 式；低位先）---
//     与 host_elem 数学等价但是另一条代码路径：
//       byte = row*row_bytes + (col*bits)/8 ; shift = (col*bits)%8
//     即引擎码平面的"每 token 行内合并"寻址（sinkhorn_g64_codec.cuh:355-385 的 Merge 约定）。
static inline float engine_elem(const RecView & R, int row, int col) {
    const int row_bytes = R.cols * R.bits / 8;
    const int bit_in_row = col * R.bits;
    const uint8_t byte = R.code[size_t(row) * row_bytes + bit_in_row / 8];
    const unsigned shift = (unsigned)(bit_in_row % 8);
    const unsigned mask = (1u << R.bits) - 1u;
    const int q = (int)((byte >> shift) & mask);
    // 折叠口径（§2.2）：scale_axis/zp_axis 已含行因子 best_row ⇒ 不得再乘 s_row；列因子只乘一次
    const float sc = f16_to_f32(R.sc[row]);
    const float zp = f16_to_f32(R.zp[row]);
    const float other = f16_to_f32(R.other[col]);
    return (sc * (float)q + zp) * other;
}

// --- 把 record 存成"引擎 plane 视图"（offset 寻址用）---
// P1 页几何（接线细案-20261013.md §2.3）：每页 64 token，plane 形状 {leading, 64, heads, pages}；
// element offset = leading*64*(head + heads*page) + leading*token + idx。
// 本工具只做数值对拍，故取 head=0/pages=0 的单页视图，leading=rows/cols 的元素步长。
static inline int64_t plane_offset(int leading_extent, int page, int head, int token, int idx) {
    const int64_t page_size = 64;
    return int64_t(leading_extent) * page_size * (int64_t(head)) + int64_t(leading_extent) * page + int64_t(idx);
}
static inline int plane_off_single(int leading_extent, int token, int idx) {
    return int(plane_offset(leading_extent, token, 0, token, idx));   // 单 head/单页自足
}

// --- D256 组装：一个 head 的 2 个 slice record（K 或 V）---
//   K: rec.rows=dim_local(128), rec.cols=token(128)  -> k[t][d] = elem(row=d_local, col=t)
//   V: rec.rows=token(128),    rec.cols=dim_local   -> v[t][d] = elem(row=t,       col=d_local)
static inline float host_head(const RecView * pair, bool is_v, int token, int d) {
    const int s = (d >= 128) ? 1 : 0, dl = d & 127;
    const RecView & R = pair[s];
    return is_v ? host_elem(R, token, dl) : host_elem(R, dl, token);
}
static inline float engine_head(const RecView * pair, bool is_v, int token, int d) {
    const int s = (d >= 128) ? 1 : 0, dl = d & 127;
    const RecView & R = pair[s];
    return is_v ? engine_elem(R, token, dl) : engine_elem(R, dl, token);
}


// ===========================================================================
// 8. 指标：逐向量（每个 token 的 D256）cosine + 最大绝对/相对误差 + 非有限值计数
// ===========================================================================
struct VecMetrics {
    int nvec = 0;
    double cos_min = 1.0, cos_max = -1.0, cos_mean = 0.0, cos_global = 0.0;
    double max_abs = 0.0, max_rel = 0.0, ref_scale = 0.0;
    long long nonfinite = 0;                                  // 两路里任一非有限
    std::vector<double> per_vec_cos;                          // --verbose 用
};

static VecMetrics metrics(const std::vector<float> & a, const std::vector<float> & b, int nvec, int width) {
    VecMetrics M; M.nvec = nvec;
    double dot_g = 0, na_g = 0, nb_g = 0;
    for (int t = 0; t < nvec; ++t) {
        double dot = 0, na = 0, nb = 0, ma = 0;
        for (int d = 0; d < width; ++d) {
            const float x = a[size_t(t) * width + d], y = b[size_t(t) * width + d];
            if (!std::isfinite(x) || !std::isfinite(y)) { M.nonfinite++; continue; }
            dot += double(x) * y; na += double(x) * x; nb += double(y) * y;
            const double ad = std::fabs(double(x) - y);
            M.max_abs = std::max(M.max_abs, ad);
            ma = std::max(ma, std::fabs(double(y)));
        }
        dot_g += dot; na_g += na; nb_g += nb;
        M.ref_scale = std::max(M.ref_scale, ma);
        const double c = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 1.0;
        M.per_vec_cos.push_back(c);
        M.cos_min = std::min(M.cos_min, c);
        M.cos_max = std::max(M.cos_max, c);
        M.cos_mean += c;
    }
    if (nvec > 0) M.cos_mean /= nvec;
    M.cos_global = (na_g > 0 && nb_g > 0) ? dot_g / std::sqrt(na_g * nb_g) : 1.0;
    return M;
}
// 逐向量最大相对误差（分母 = 该向量的 max|ref|）
static double max_rel_vec(const std::vector<float> & a, const std::vector<float> & b, int nvec, int width) {
    double worst = 0.0;
    for (int t = 0; t < nvec; ++t) {
        double ma = 0, md = 0;
        for (int d = 0; d < width; ++d) {
            const float x = a[size_t(t) * width + d], y = b[size_t(t) * width + d];
            if (!std::isfinite(x) || !std::isfinite(y)) continue;
            ma = std::max(ma, std::fabs(double(y)));
            md = std::max(md, std::fabs(double(x) - double(y)));
        }
        if (ma > 0) worst = std::max(worst, md / ma);
    }
    return worst;
}


// ===========================================================================
// 9. 记录容器 + 三种 reader
// ===========================================================================
struct Rec {
    std::string name;              // 用于报告/配对
    int kind = 0;                  // 0=K 1=V
    int bits = 4, rows = 128, cols = 128;
    int tokens = 128;              // 有效 token 数（未满 tile < 128）
    int pages = 2;                 // 覆盖页数（每页 64 token）
    int head_slices = 2;
    int iters = 16;                // 量化时的 Sinkhorn 迭代数（selftest 用）
    std::vector<uint8_t> code;     // record 码字节（payload）
    std::vector<uint16_t> scale;   // scale 轴：rows 个 sc | rows 个 zp | cols 个 other
    std::vector<float> orig;       // 量化前原值（旋转域），rows*cols
    bool has_orig = false;
    std::vector<float> orig_unrot; // 量化前原值（未旋转域），tokens*256（可选）
    RecView view() const {
        RecView R; R.code = code.data();
        R.sc = scale.data(); R.zp = scale.data() + rows; R.other = scale.data() + 2 * rows;
        R.rows = rows; R.cols = cols; R.bits = bits; R.is_v = (kind == 1);
        return R;
    }
};

static bool rdall(void * dst, size_t n, FILE * f) { return fread(dst, 1, n, f) == n; }

// ---- reader A：GATE2 dumprec（magic 'GVK2'=0x324B5647, ver 2）----
static bool load_gate2(const char * path, std::vector<Rec> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", path); return false; }
    uint32_t magic = 0, ver = 0, n = 0;
    if (!rdall(&magic, 4, f) || !rdall(&ver, 4, f) || !rdall(&n, 4, f)) { fclose(f); return false; }
    if (magic != 0x324B5647u || ver != 2u) {
        fprintf(stderr, "magic/ver 不对 (%08x,%u)，不是 GATE2 dumprec\n", magic, ver); fclose(f); return false;
    }
    for (uint32_t k = 0; k < n; ++k) {
        Rec D; uint32_t nl = 0;
        if (!rdall(&nl, 4, f)) { fclose(f); return false; }
        D.name.resize(nl); if (nl && !rdall(&D.name[0], nl, f)) { fclose(f); return false; }
        int32_t a = 0;
        rdall(&a, 4, f); D.rows = a;
        rdall(&a, 4, f); D.cols = a;
        rdall(&a, 4, f); D.bits = a;
        rdall(&a, 4, f); D.iters = a;
        rdall(&a, 4, f);                       // best_iter
        rdall(&a, 4, f);                       // best_imbalance
        std::vector<float> s_col(D.cols), s_row(D.rows), tile(size_t(D.rows) * D.cols);
        if (!rdall(s_col.data(), 4 * size_t(D.cols), f)) { fclose(f); return false; }
        if (!rdall(s_row.data(), 4 * size_t(D.rows), f)) { fclose(f); return false; }
        if (!rdall(tile.data(), 4 * tile.size(), f)) { fclose(f); return false; }
        uint32_t rl = 0; if (!rdall(&rl, 4, f)) { fclose(f); return false; }
        std::vector<uint8_t> rec(rl);
        if (rl && !rdall(rec.data(), rl, f)) { fclose(f); return false; }
        const size_t payload = bee_packed_bytes(D.rows * D.cols, D.bits);
        if (rec.size() != payload + size_t(2 * D.rows + D.cols) * 2) {
            fprintf(stderr, "%s: record 长度 %zu 与布局 %zu 不符，跳过\n", D.name.c_str(), rec.size(),
                    payload + size_t(2 * D.rows + D.cols) * 2);
            continue;
        }
        // 拆成 code / scale 两段
        D.code.assign(rec.begin(), rec.begin() + payload);
        D.scale.resize(size_t(2 * D.rows + D.cols));
        std::memcpy(D.scale.data(), rec.data() + payload, D.scale.size() * 2);
        D.orig = std::move(tile); D.has_orig = true;
        D.kind = (D.name.find("_v_") != std::string::npos) ? 1 : 0;
        D.tokens = std::min(D.rows, D.cols);
        D.pages = (D.tokens + 63) / 64;
        D.head_slices = 2;
        out.push_back(std::move(D));
    }
    fclose(f);
    return true;
}

// ---- reader B：KVD1 dump（本工具定义的引擎 plane dump 最小格式）----
// 头：u32 'KVD1' | u32 ver=1 | u32 n_records
// 逐条：u32 name_len; u8 name[]; i32 kind; i32 bits; i32 rows; i32 cols; i32 tokens;
//       i32 pages; i32 head_slices; u32 flags(bit0=带原值);
//       u32 code_bytes; u8 code[]; u32 scale_halfs; u16 scale[]; u32 orig_floats; f32 orig[]
static constexpr uint32_t KVD1_MAGIC = 0x3144564Bu;
static bool load_kvd1(const char * path, std::vector<Rec> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", path); return false; }
    uint32_t magic = 0, ver = 0, n = 0;
    if (!rdall(&magic, 4, f) || !rdall(&ver, 4, f) || !rdall(&n, 4, f)) { fclose(f); return false; }
    if (magic != KVD1_MAGIC || ver != 1u) {
        fprintf(stderr, "magic/ver 不对 (%08x,%u)，不是 KVD1 dump\n", magic, ver); fclose(f); return false;
    }
    for (uint32_t k = 0; k < n; ++k) {
        Rec D; uint32_t nl = 0;
        if (!rdall(&nl, 4, f)) { fclose(f); return false; }
        D.name.resize(nl); if (nl && !rdall(&D.name[0], nl, f)) { fclose(f); return false; }
        int32_t a = 0;
        rdall(&a, 4, f); D.kind = a;
        rdall(&a, 4, f); D.bits = a;
        rdall(&a, 4, f); D.rows = a;
        rdall(&a, 4, f); D.cols = a;
        rdall(&a, 4, f); D.tokens = a;
        rdall(&a, 4, f); D.pages = a;
        rdall(&a, 4, f); D.head_slices = a;
        uint32_t flags = 0; rdall(&flags, 4, f);
        uint32_t cb = 0; rdall(&cb, 4, f);
        D.code.resize(cb); if (cb && !rdall(D.code.data(), cb, f)) { fclose(f); return false; }
        uint32_t sh = 0; rdall(&sh, 4, f);
        D.scale.resize(sh); if (sh && !rdall(D.scale.data(), sh * 2, f)) { fclose(f); return false; }
        uint32_t of = 0; rdall(&of, 4, f);
        D.orig.resize(of); if (of && !rdall(D.orig.data(), of * 4, f)) { fclose(f); return false; }
        D.has_orig = (of > 0);
        out.push_back(std::move(D));
    }
    fclose(f);
    return true;
}
static bool save_kvd1(const char * path, const std::vector<Rec> & recs) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "写不了 %s\n", path); return false; }
    const uint32_t m = KVD1_MAGIC, ver = 1, n = (uint32_t)recs.size();
    fwrite(&m, 4, 1, f); fwrite(&ver, 4, 1, f); fwrite(&n, 4, 1, f);
    for (const Rec & D : recs) {
        const uint32_t nl = (uint32_t)D.name.size();
        fwrite(&nl, 4, 1, f); if (nl) fwrite(D.name.data(), 1, nl, f);
        const int32_t a[7] = { D.kind, D.bits, D.rows, D.cols, D.tokens, D.pages, D.head_slices };
        fwrite(a, 4, 7, f);
        const uint32_t flags = D.has_orig ? 1u : 0u;
        fwrite(&flags, 4, 1, f);
        const uint32_t cb = (uint32_t)D.code.size(); fwrite(&cb, 4, 1, f);
        if (cb) fwrite(D.code.data(), 1, cb, f);
        const uint32_t sh = (uint32_t)D.scale.size(); fwrite(&sh, 4, 1, f);
        if (sh) fwrite(D.scale.data(), 2, sh, f);
        const uint32_t of = D.has_orig ? (uint32_t)D.orig.size() : 0u; fwrite(&of, 4, 1, f);
        if (of) fwrite(D.orig.data(), 4, of, f);
    }
    fclose(f);
    return true;
}


// ---- 引擎解码输出文件 KVO1（把引擎 kernel 解出来的值 dump 下来，喂给本工具对拍）----
// 头：u32 'KVO1' | u32 ver=1 | u32 n_heads
// 逐 head：u32 name_len; u8 name[]; i32 kind; i32 tokens; i32 width; i32 domain(0=旋转域 1=未旋转域);
//          u32 n_floats; f32 data[tokens*width]
struct KvoHead {
    std::string name; int kind = 0, tokens = 0, width = 256, domain = 0;
    std::vector<float> data;
};
static constexpr uint32_t KVO1_MAGIC = 0x314F564Bu;
static bool load_kvo1(const char * path, std::vector<KvoHead> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "打不开 %s\n", path); return false; }
    uint32_t magic = 0, ver = 0, n = 0;
    if (!rdall(&magic, 4, f) || !rdall(&ver, 4, f) || !rdall(&n, 4, f)) { fclose(f); return false; }
    if (magic != KVO1_MAGIC || ver != 1u) { fprintf(stderr, "%s 不是 KVO1\n", path); fclose(f); return false; }
    for (uint32_t k = 0; k < n; ++k) {
        KvoHead H; uint32_t nl = 0;
        if (!rdall(&nl, 4, f)) { fclose(f); return false; }
        H.name.resize(nl); if (nl && !rdall(&H.name[0], nl, f)) { fclose(f); return false; }
        int32_t a = 0;
        rdall(&a, 4, f); H.kind = a;
        rdall(&a, 4, f); H.tokens = a;
        rdall(&a, 4, f); H.width = a;
        rdall(&a, 4, f); H.domain = a;
        uint32_t nf = 0; rdall(&nf, 4, f);
        H.data.resize(nf); if (nf && !rdall(H.data.data(), nf * 4, f)) { fclose(f); return false; }
        out.push_back(std::move(H));
    }
    fclose(f);
    return true;
}
static bool save_kvo1(const char * path, const std::vector<KvoHead> & heads) {
    FILE * f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "写不了 %s\n", path); return false; }
    const uint32_t m = KVO1_MAGIC, ver = 1, n = (uint32_t)heads.size();
    fwrite(&m, 4, 1, f); fwrite(&ver, 4, 1, f); fwrite(&n, 4, 1, f);
    for (const KvoHead & H : heads) {
        const uint32_t nl = (uint32_t)H.name.size();
        fwrite(&nl, 4, 1, f); if (nl) fwrite(H.name.data(), 1, nl, f);
        const int32_t a[4] = { H.kind, H.tokens, H.width, H.domain };
        fwrite(a, 4, 4, f);
        const uint32_t nf = (uint32_t)H.data.size(); fwrite(&nf, 4, 1, f);
        if (nf) fwrite(H.data.data(), 4, nf, f);
    }
    fclose(f);
    return true;
}


// ===========================================================================
// 10. 分析：一个 head 的 D256 逐 token 对拍（主判据在这里）
// ===========================================================================
// pair：同一 head 的 2 个 slice record（K4V4/K4V2 位宽下 record 恒 128×128）
// 输出（全部为原始输出行）：
//   cos_adq_orig  = 主机反量化 vs 量化前原值（旋转域）   —— 编码端质量
//   cos_eng_orig  = record 解码 vs 量化前原值（旋转域）  —— 读路径质量
//   cos_eng_adq   = record 解码 vs 主机反量化  <== 主判据 "record 解码 == 主机反量化"
//   cos_unr       = 逆旋转(record 解码) vs 未旋转原值（若有 orig_unrot）
//   maxabs / maxrel / nonfinite
struct HeadResult {
    std::string label;
    double cos_adq_orig = 0, cos_eng_orig = 0, cos_eng_adq = 0, cos_unr = -1;
    double cos_file_adq = -1, cos_file_orig = -1;      // 有 --decoded 时才有意义
    double maxabs_eng_adq = 0, maxrel_eng_adq = 0;
    long long nonfinite = 0;
    int tokens = 0, bits = 4;
};

static HeadResult analyze_pair(const Rec & a, const Rec & b, const char * label, bool verbose,
                               const KvoHead * engfile = nullptr) {
    HeadResult H; H.label = label;
    const bool is_v = (a.kind == 1);
    const int ntok = std::min(a.tokens, b.tokens);
    const int W = 256;
    const RecView v0 = a.view(), v1 = b.view();
    const RecView pair[2] = { v0, v1 };

    std::vector<float> adq(size_t(ntok) * W), eng(size_t(ntok) * W), orig(size_t(ntok) * W);
    for (int t = 0; t < ntok; ++t) {
        for (int d = 0; d < W; ++d) {
            adq[size_t(t) * W + d] = host_head(pair, is_v, t, d);
            eng[size_t(t) * W + d] = engine_head(pair, is_v, t, d);
        }
    }
    // 量化前原值（旋转域）：按 head 的 [token][d] 取向拼 2 个 slice
    for (int t = 0; t < ntok; ++t) {
        for (int s = 0; s < 2; ++s) {
            const Rec & R = s ? b : a;
            for (int dl = 0; dl < 128; ++dl) {
                const float x = is_v ? R.orig[size_t(t) * 128 + dl]      // V: rows=token, cols=dim
                                     : R.orig[size_t(dl) * 128 + t];     // K: rows=dim,  cols=token
                orig[size_t(t) * W + s * 128 + dl] = x;
            }
        }
    }
    VecMetrics m1 = metrics(adq, orig, ntok, W);   // 主机反量化 vs 原值
    VecMetrics m2 = metrics(eng, orig, ntok, W);   // 引擎解码   vs 原值
    VecMetrics m3 = metrics(eng, adq, ntok, W);    // 引擎解码   vs 主机反量化  <== 主判据

    H.cos_adq_orig = m1.cos_global;
    H.cos_eng_orig = m2.cos_global;
    H.cos_eng_adq  = m3.cos_global;
    H.maxabs_eng_adq = m3.max_abs;
    H.maxrel_eng_adq = max_rel_vec(eng, adq, ntok, W);
    H.nonfinite = m3.nonfinite + m2.nonfinite + m1.nonfinite;
    H.tokens = ntok;
    H.bits = a.bits;

    // 引擎解码输出（--decoded）：主判据的真身 —— 引擎 kernel 解出来的值 vs 主机反量化
    if (engfile && engfile->width == W && engfile->tokens >= ntok
        && (int)engfile->data.size() >= ntok * W) {
        std::vector<float> ef(size_t(ntok) * W), ef_ref(size_t(ntok) * W);
        for (int t = 0; t < ntok; ++t)
            for (int d = 0; d < W; ++d) ef[size_t(t) * W + d] = engfile->data[size_t(t) * W + d];
        if (engfile->domain == 0) {
            ef_ref = adq;                                         // 引擎输出在旋转域 ⇒ 直接用主机反量化比
        } else {
            for (int t = 0; t < ntok; ++t) {                      // 引擎输出已逆旋转 ⇒ 主机反量化也逆旋转
                float buf[D256];
                for (int d = 0; d < W; ++d) buf[d] = adq[size_t(t) * W + d];
                norm_hadamard_256(buf);
                for (int d = 0; d < W; ++d) ef_ref[size_t(t) * W + d] = buf[d];
            }
        }
        VecMetrics m6 = metrics(ef, ef_ref, ntok, W);
        H.cos_file_adq = m6.cos_global;
        if (engfile->domain == 0) {
            H.cos_file_orig = metrics(ef, orig, ntok, W).cos_global;
        } else if (!a.orig_unrot.empty() && a.orig_unrot.size() >= size_t(ntok) * W) {
            std::vector<float> ur(size_t(ntok) * W);
            for (int t = 0; t < ntok; ++t)
                for (int d = 0; d < W; ++d) ur[size_t(t) * W + d] = a.orig_unrot[size_t(t) * W + d];
            H.cos_file_orig = metrics(ef, ur, ntok, W).cos_global;
        }
        if (verbose) {
            printf("      [dec ] per-token cos(引擎解码文件, 主机反量化) min=%.9f mean=%.9f max=%.9f "
                   "maxabs=%.3e nonfinite=%lld\n",
                   m6.cos_min, m6.cos_mean, m6.cos_max, m6.max_abs, m6.nonfinite);
        }
    }

    // 逆旋转臂：逆旋转(引擎解码) vs 未旋转原值（sweep 模式才有 orig_unrot）
    if (!a.orig_unrot.empty() && a.orig_unrot.size() >= size_t(ntok) * W) {
        std::vector<float> unrot(size_t(ntok) * W);
        for (int t = 0; t < ntok; ++t) {
            for (int d = 0; d < W; ++d) unrot[size_t(t) * W + d] = a.orig_unrot[size_t(t) * W + d];
        }
        std::vector<float> eng_inv(size_t(ntok) * W);
        for (int t = 0; t < ntok; ++t) {
            float buf[D256];
            for (int d = 0; d < W; ++d) buf[d] = eng[size_t(t) * W + d];
            norm_hadamard_256(buf);                        // 逆旋转（自逆）
            for (int d = 0; d < W; ++d) eng_inv[size_t(t) * W + d] = buf[d];
        }
        VecMetrics m4 = metrics(eng_inv, unrot, ntok, W);
        H.cos_unr = m4.cos_global;
        if (verbose) {
            VecMetrics m5 = metrics(eng, unrot, ntok, W);  // 反证：不做逆旋转时该有多差
            printf("      [rot] 逆旋转(解码) vs 未旋转原值 cos=%.9f | 反证 不逆旋转 vs 未旋转原值 cos=%.9f\n",
                   m4.cos_global, m5.cos_global);
        }
    }

    printf("  %-34s tok=%3d | cos_adq_orig=%.9f cos_eng_orig=%.9f | cos_ENG_ADQ=%.9f | "
           "maxabs=%.3e maxrel=%.3e nonfinite=%lld\n",
           label, ntok, H.cos_adq_orig, H.cos_eng_orig, H.cos_eng_adq,
           H.maxabs_eng_adq, H.maxrel_eng_adq, H.nonfinite);
    if (H.cos_eng_adq < 1.0 - 1e-12)
        printf("  %-34s [!] 内部两条解码路径不逐位一致：这是【本工具】的问题，不是引擎的\n", label);
    if (H.cos_file_adq >= 0)
        printf("  %-34s [主判据] cos(引擎解码输出, 主机反量化)=%.9f   cos(引擎解码输出, 原值)=%.9f\n",
               label, H.cos_file_adq, H.cos_file_orig);
    if (H.cos_unr >= 0) printf("  %-34s [rot]  cos_逆旋转解码_vs_未旋转原值=%.9f\n", label, H.cos_unr);
    if (verbose) {
        printf("      per-token cos(ENG,ADQ): min=%.9f mean=%.9f max=%.9f\n",
               m3.cos_min, m3.cos_mean, m3.cos_max);
    }
    return H;
}

// 把 recs 按 slice 配对（相邻两条 = s0,s1）后逐 head 分析
static std::vector<HeadResult> analyze_all(std::vector<Rec> & recs, bool verbose,
                                           const std::vector<KvoHead> * decoded = nullptr) {
    std::vector<HeadResult> out;
    size_t i = 0, hi = 0;
    while (i < recs.size()) {
        const KvoHead * ef = nullptr;
        if (decoded && hi < decoded->size()) ef = &(*decoded)[hi];
        if (i + 1 < recs.size() && recs[i].head_slices == 2 && recs[i + 1].head_slices == 2
            && recs[i].kind == recs[i + 1].kind) {
            char lab[256];
            snprintf(lab, sizeof lab, "%s", recs[i].name.c_str());
            out.push_back(analyze_pair(recs[i], recs[i + 1], lab, verbose, ef));
            i += 2;
        } else {
            // 落单的 record：只能做 slice 内自对拍（cos_eng_adq 仍有效）
            out.push_back(analyze_pair(recs[i], recs[i], (recs[i].name + "[single]").c_str(), verbose, ef));
            i += 1;
        }
        ++hi;
    }
    if (decoded && decoded->size() != hi)
        fprintf(stderr, "[warn] --decoded 的 head 数(%zu) 与 record 配对数(%zu) 不一致\n", decoded->size(), hi);
    return out;
}


// ===========================================================================
// 11. sweep：内部造全矩阵（K4V4/K4V2 × {128,127,51,1} token × {gauss,outlier}）
// ===========================================================================
struct SweepCfg { int bits_k = 4, bits_v = 4; int iters = 16; uint64_t seed = 20261013ull; };

// 造一个 head 的未旋转原值 K/V（tokens×256，N 之后的 token 置 0 = 未满 tile 的 padding）
static void make_head(std::vector<float> & k, std::vector<float> & v, int ntok, uint64_t seed, bool outlier) {
    std::vector<float> raw;
    fill_gauss(raw, 128, 256, seed, outlier);
    k.assign(size_t(128) * 256, 0.f);
    for (int t = 0; t < ntok; ++t)
        for (int d = 0; d < 256; ++d) k[size_t(t) * 256 + d] = raw[size_t(t) * 256 + d];
    fill_gauss(raw, 128, 256, seed + 777777ull, outlier);
    v.assign(size_t(128) * 256, 0.f);
    for (int t = 0; t < ntok; ++t)
        for (int d = 0; d < 256; ++d) v[size_t(t) * 256 + d] = raw[size_t(t) * 256 + d];
}

// 前向旋转（未旋转 -> 旋转域），逐 token 的 D256
static void rotate_head(std::vector<float> & x) {
    for (int t = 0; t < 128; ++t) {
        float buf[D256];
        for (int d = 0; d < D256; ++d) buf[d] = x[size_t(t) * D256 + d];
        norm_hadamard_256(buf);
        for (int d = 0; d < D256; ++d) x[size_t(t) * D256 + d] = buf[d];
    }
}

// 一个 kind（K 或 V）的 sweep：造 -> 旋转 -> h16 stage -> 量化 -> 解码 -> 对拍
static HeadResult sweep_one(const char * kindname, bool is_v, int bits, int ntok, bool outlier,
                            const SweepCfg & C, bool verbose) {
    std::vector<float> k, v;
    make_head(k, v, ntok, C.seed, outlier);
    std::vector<float> & head = is_v ? v : k;
    std::vector<float> unrot = head;                 // 未旋转原值
    rotate_head(head);                               // 旋转域
    for (auto & x : head) x = h16(x);                // beellama stage = F16（dump_oracle push():763 同口径）

    // 切成 2 个 slice 的 tile 并量化
    Rec rec[2];
    for (int s = 0; s < 2; ++s) {
        std::vector<float> tile(size_t(128) * 128);
        if (is_v) {                                   // V: rows=token, cols=dim_local
            for (int t = 0; t < 128; ++t)
                for (int dl = 0; dl < 128; ++dl) tile[size_t(t) * 128 + dl] = head[size_t(t) * 256 + s * 128 + dl];
        } else {                                      // K: rows=dim_local, cols=token
            for (int dl = 0; dl < 128; ++dl)
                for (int t = 0; t < 128; ++t) tile[size_t(dl) * 128 + t] = head[size_t(t) * 256 + s * 128 + dl];
        }
        std::vector<uint8_t> recbytes;
        QuantMeta meta;
        bee_quant_tile(tile.data(), 128, 128, bits, C.iters, recbytes, &meta, false);
        const size_t payload = bee_packed_bytes(128 * 128, bits);
        Rec & R = rec[s];
        R.name = std::string(kindname) + "_s" + std::to_string(s);
        R.kind = is_v ? 1 : 0; R.bits = bits; R.rows = 128; R.cols = 128;
        R.tokens = ntok; R.pages = (ntok + 63) / 64; R.head_slices = 2;
        R.code.assign(recbytes.begin(), recbytes.begin() + payload);
        R.scale.resize(size_t(2 * 128 + 128));
        std::memcpy(R.scale.data(), recbytes.data() + payload, R.scale.size() * 2);
        R.orig = std::move(tile); R.has_orig = true;
    }
    rec[0].orig_unrot = unrot;                        // 只挂在 s0 上（配对时用 a.orig_unrot）
    char lab[128];
    snprintf(lab, sizeof lab, "%s_%s tok=%d b=%d it%d", outlier ? "outlier" : "gauss", kindname, ntok, bits, C.iters);
    return analyze_pair(rec[0], rec[1], lab, verbose);
}


// ===========================================================================
// 12. --selftest：证明"只读 import 的量化器副本"与 GATE2 dump 逐字节一致
// ===========================================================================
static int cmd_selftest(const char * path) {
    std::vector<Rec> recs;
    if (!load_gate2(path, recs)) return 3;
    int nrec = 0, nbad = 0; long long badbytes = 0;
    int iters_seen = -1;
    for (auto & D : recs) {
        std::vector<uint8_t> recbytes;
        QuantMeta meta;
        std::vector<float> t = D.orig;
        bee_quant_tile(t.data(), D.rows, D.cols, D.bits, D.iters, recbytes, &meta, false);
        iters_seen = D.iters;
        const size_t payload = bee_packed_bytes(D.rows * D.cols, D.bits);
        std::vector<uint8_t> got(payload + D.scale.size() * 2);
        std::memcpy(got.data(), recbytes.data(), got.size());
        // 与 dump 里的 code+scale 逐字节比
        int nd = 0;
        for (size_t i = 0; i < payload; ++i) if (D.code[i] != got[i]) ++nd;
        for (size_t i = 0; i < D.scale.size() * 2; ++i)
            if (((const uint8_t *)D.scale.data())[i] != got[payload + i]) ++nd;
        nrec++; badbytes += nd;
        if (nd) nbad++;
    }
    printf("[selftest] %s : records=%d iters=%d 不一致=%d 差异字节=%lld\n",
           path, nrec, iters_seen, nbad, badbytes);
    printf("[selftest] 结论：%s\n", (nbad == 0)
           ? "PASS —— import 的量化器副本逐字节复现 GATE2 dump，副本没走样"
           : "FAIL —— 副本与 dump 不一致（iters=16 时可能是 Sinkhorn 病态轨迹，见两道gate结论 §gate2）");
    return nbad == 0 ? 0 : 1;
}

// ---- --gen-fake：逐字段手写一条最小 KVD1 dump + 两个 KVO1（不依赖 python）----
// 与 tools/make_fake_dump.py 完全同构（互为独立实现，证明格式不是纸面的）
static int cmd_gen_fake(const char * outpath) {
    const int R = 128, Cc = 128, B = 4;
    auto qof = [](int r, int c, int s) { return (r * 3 + c * 5 + 7 * s + 1) & 0xF; };
    auto pack = [&](int s) {
        std::vector<uint8_t> buf(size_t(R) * Cc * B / 8, 0);
        for (int r = 0; r < R; ++r) {
            const size_t base = size_t(r) * (Cc / 2);
            for (int c = 0; c < Cc; ++c) {
                const int q = qof(r, c, s);
                if (c % 2 == 0) buf[base + c / 2] |= uint8_t(q);
                else            buf[base + c / 2] |= uint8_t(q << 4);
            }
        }
        return buf;
    };
    // 统一口径：tile[row][col] 的码 = qof(row,col,s)，原值 = 2q，按 tile 行主序存
    auto origv = [&](int s) {
        std::vector<float> o(size_t(R) * Cc);
        for (int i = 0; i < R; ++i) for (int j = 0; j < Cc; ++j)
            o[size_t(i) * Cc + j] = 2.0f * qof(i, j, s);
        return o;
    };

    std::vector<Rec> recs;
    for (int isv = 0; isv < 2; ++isv) for (int s = 0; s < 2; ++s) {
        Rec D;
        char nm[64]; snprintf(nm, sizeof nm, "fake_%s_s%d", isv ? "V" : "K", s);
        D.name = nm; D.kind = isv; D.bits = B; D.rows = R; D.cols = Cc;
        D.tokens = R; D.pages = 2; D.head_slices = 2; D.iters = 16;
        D.code = pack(s);
        D.scale.resize(size_t(2 * R + Cc));
        for (int i = 0; i < R; ++i) { D.scale[i] = f32_to_f16_rn(1.0f); D.scale[R + i] = f32_to_f16_rn(0.0f); }
        for (int j = 0; j < Cc; ++j) D.scale[2 * R + j] = f32_to_f16_rn(2.0f);
        D.orig = origv(s); D.has_orig = true;
        recs.push_back(std::move(D));
    }
    if (!save_kvd1(outpath, recs)) return 3;

    std::vector<KvoHead> heads;
    for (int isv = 0; isv < 2; ++isv) {
        KvoHead H; H.name = isv ? "fake_V" : "fake_K"; H.kind = isv;
        H.tokens = R; H.width = 256; H.domain = 0;
        H.data.assign(size_t(R) * 256, 0.f);
        for (int t = 0; t < R; ++t) for (int d = 0; d < 256; ++d) {
            const int s = d / 128, dl = d % 128;
            H.data[size_t(t) * 256 + d] = 2.0f * (isv ? qof(t, dl, s) : qof(dl, t, s));
        }
        heads.push_back(std::move(H));
    }
    std::string dec = std::string(outpath) + ".dec";
    if (!save_kvo1(dec.c_str(), heads)) return 3;
    std::vector<KvoHead> bad = heads;                       // 把 K 的 token7 整体取反
    for (int d = 0; d < 256; ++d) bad[0].data[7 * 256 + d] = -bad[0].data[7 * 256 + d];
    std::string badp = std::string(outpath) + ".bad";
    if (!save_kvo1(badp.c_str(), bad)) return 3;

    printf("[gen-fake] %s : %zu records (K s0/s1 + V s0/s1, bits=4, rows=cols=128)\n", outpath, recs.size());
    printf("[gen-fake] %s : %zu heads（正确引擎解码, domain=0） -> 应 PASS\n", dec.c_str(), heads.size());
    printf("[gen-fake] %s : %zu heads（K 的 token7 取反）     -> 应 FAIL(解码端)\n", badp.c_str(), bad.size());
    printf("[gen-fake] 解析式预期：解码值 = (q*1 + 0)*2 = 2q，q=(3r+5c+7s+1)&15；三路 cosine 全 1.0，maxabs=0\n");
    return 0;
}


// ===========================================================================
// main
// ===========================================================================
static void usage() {
    printf(
        "kvarn_g128_verify —— KVarN G128 引擎级对拍：\"record 解码 == 主机反量化\"\n"
        "\n"
        "  --oracle <dump.bin>        读 GATE2 dumprec（magic GVK2），逐 head 对拍\n"
        "  --from-dump <file.dump>    读 KVD1 dump（引擎 plane 最小格式），逐 head 对拍\n"
        "  --decoded <file.kvo>       引擎自己解码出来的值（KVO1），作为主判据的对照\n"
        "  --make-dump <in> <out>     把 GATE2 dumprec 转成 KVD1 dump\n"
        "  --gen-dump <out.dump>      内部造一个 KVD1 dump（含真值），证明 --from-dump 入口可用\n"
        "  --gen-fake <out.dump>      逐字段手写最小 dump + .dec/.bad（证明两个入口不是纸面的）\n"
        "  --sweep                    全矩阵：K4V4/K4V2 × {128,127,51,1} token × {gauss,outlier}\n"
        "  --selftest <dump.bin>      验证 import 的量化器副本能复现该 dump（结构证明）\n"
        "\n"
        "  选参：--bits-k N --bits-v N --iters N --seed N --tokens N --verbose\n");
}

int main(int argc, char ** argv) {
    std::string mode, path, path2, decoded_path;
    SweepCfg C; bool verbose = false; int tokens_override = -1;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if      (a == "--oracle"    && i + 1 < argc) { mode = a; path = argv[++i]; }
        else if (a == "--from-dump" && i + 1 < argc) { mode = a; path = argv[++i]; }
        else if (a == "--selftest"  && i + 1 < argc) { mode = a; path = argv[++i]; }
        else if (a == "--gen-dump"  && i + 1 < argc) { mode = a; path = argv[++i]; }
        else if (a == "--gen-fake"  && i + 1 < argc) { mode = a; path = argv[++i]; }
        else if (a == "--decoded"   && i + 1 < argc) { decoded_path = argv[++i]; }
        else if (a == "--make-dump" && i + 2 < argc) { mode = a; path = argv[++i]; path2 = argv[++i]; }
        else if (a == "--sweep")   { mode = a; }
        else if (a == "--bits-k" && i + 1 < argc) C.bits_k = atoi(argv[++i]);
        else if (a == "--bits-v" && i + 1 < argc) C.bits_v = atoi(argv[++i]);
        else if (a == "--iters"  && i + 1 < argc) C.iters  = atoi(argv[++i]);
        else if (a == "--seed"   && i + 1 < argc) C.seed   = strtoull(argv[++i], nullptr, 10);
        else if (a == "--tokens" && i + 1 < argc) tokens_override = atoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else { fprintf(stderr, "未知参数 %s\n", a.c_str()); usage(); return 2; }
    }
    if (mode.empty()) { usage(); return 2; }

    if (mode == "--selftest") return cmd_selftest(path.c_str());
    if (mode == "--gen-fake") return cmd_gen_fake(path.c_str());

    if (mode == "--make-dump") {
        std::vector<Rec> recs;
        if (!load_gate2(path.c_str(), recs)) return 3;
        if (!save_kvd1(path2.c_str(), recs)) return 3;
        printf("[make-dump] %zu records: %s -> %s\n", recs.size(), path.c_str(), path2.c_str());
        return 0;
    }

    if (mode == "--gen-dump") {
        // 造一个 KVD1 dump：K4V4 + K4V2、tok=128/51，带真值（原值 + 未旋转原值）
        std::vector<Rec> all;
        const int lens[2] = { 128, 51 };
        for (int bi = 0; bi < 2; ++bi) {
            const int bits = bi ? 2 : 4;
            for (int li = 0; li < 2; ++li) {
                const int ntok = lens[li];
                for (int isv = 0; isv < 2; ++isv) {
                    std::vector<float> k, v;
                    make_head(k, v, ntok, C.seed, false);
                    std::vector<float> & head = isv ? v : k;
                    std::vector<float> unrot = head;
                    rotate_head(head);
                    for (auto & x : head) x = h16(x);
                    for (int s = 0; s < 2; ++s) {
                        std::vector<float> tile(size_t(128) * 128);
                        if (isv) {
                            for (int t = 0; t < 128; ++t) for (int dl = 0; dl < 128; ++dl)
                                tile[size_t(t) * 128 + dl] = head[size_t(t) * 256 + s * 128 + dl];
                        } else {
                            for (int dl = 0; dl < 128; ++dl) for (int t = 0; t < 128; ++t)
                                tile[size_t(dl) * 128 + t] = head[size_t(t) * 256 + s * 128 + dl];
                        }
                        std::vector<uint8_t> rb; QuantMeta meta;
                        bee_quant_tile(tile.data(), 128, 128, bits, C.iters, rb, &meta, false);
                        const size_t payload = bee_packed_bytes(128 * 128, bits);
                        Rec R;
                        char nm[128];
                        snprintf(nm, sizeof nm, "%s_k%dv%d_tok%d_s%d", isv ? "V" : "K", 4, bits, ntok, s);
                        R.name = nm; R.kind = isv ? 1 : 0; R.bits = bits; R.rows = 128; R.cols = 128;
                        R.tokens = ntok; R.pages = (ntok + 63) / 64; R.head_slices = 2;
                        R.code.assign(rb.begin(), rb.begin() + payload);
                        R.scale.resize(size_t(2 * 128 + 128));
                        std::memcpy(R.scale.data(), rb.data() + payload, R.scale.size() * 2);
                        R.orig = std::move(tile); R.has_orig = true;
                        if (s == 0) R.orig_unrot = unrot;
                        all.push_back(std::move(R));
                    }
                }
            }
        }
        if (!save_kvd1(path.c_str(), all)) return 3;
        printf("[gen-dump] %zu records -> %s（含 K4V4/K4V2 × tok=128/51 × K/V，带真值）\n", all.size(), path.c_str());
        return 0;
    }

    if (mode == "--sweep") {
        printf("================ KVarN G128 引擎级对拍 · sweep ================\n");
        printf("口径：主机反量化 = record 的 rank-1 逐元素读法（oracle bee_dequant_tile）\n");
        printf("      record 解码 = 引擎口径（行内 merge 字节寻址 + 逐 token 组装 D256 + 折叠 + 逆旋转）\n");
        printf("旋转：norm_hadamard_256（(H256/16)²=I，自逆），与 ninfer normalized_hadamard_d256 同构\n");
        printf("stage：F16（beellama stage 口径）  iters=%d  seed=%llu\n", C.iters, (unsigned long long)C.seed);
        const int lens[4] = { 128, 127, 51, 1 };
        const char * kinds[2] = { "gauss", "outlier" };
        for (int oi = 0; oi < 2; ++oi) {
            for (int bi = 0; bi < 2; ++bi) {
                const int bits = bi ? 2 : 4;
                const char * armname = bi ? "K4V2" : "K4V4";
                printf("\n---- 臂 %s · %s · bits=%d ----\n", armname, kinds[oi], bits);
                for (int li = 0; li < 4; ++li) {
                    sweep_one("K", false, bits, lens[li], oi == 1, C, verbose);
                    sweep_one("V", true,  bits, lens[li], oi == 1, C, verbose);
                }
            }
        }
        printf("\n[注] 单条 record 只有 128 个 token 槽位；未满 tile（127/51/1）按【其余 token 置 0】量化，\n"
               "     统计只取前 N 个 token —— 与引擎 B/C 路（尾块先写 stage、封口时整 128 槽编码）同口径。\n");
        return 0;
    }

    // 两个 reader 走同一条分析
    std::vector<Rec> recs;
    bool ok = (mode == "--oracle") ? load_gate2(path.c_str(), recs) : load_kvd1(path.c_str(), recs);
    if (!ok) return 3;
    if (tokens_override > 0) for (auto & R : recs) R.tokens = std::min(R.tokens, tokens_override);

    std::vector<KvoHead> decoded;
    const std::vector<KvoHead> * pdec = nullptr;
    if (!decoded_path.empty()) {
        if (!load_kvo1(decoded_path.c_str(), decoded)) return 3;
        pdec = &decoded;
    }

    printf("================ KVarN G128 引擎级对拍 · %s ================\n",
           mode == "--oracle" ? "GATE2 dumprec" : "KVD1 engine-plane dump");
    printf("文件=%s  records=%zu", path.c_str(), recs.size());
    if (pdec) printf("  引擎解码文件=%s  heads=%zu", decoded_path.c_str(), decoded.size());
    printf("\n");
    printf("判据：cos(引擎解码输出, 主机反量化) ≥ 0.999（有 --decoded 时为主判据）；"
           "cos_adq_orig ≥ 0.999 才算编码端无病\n");
    auto res = analyze_all(recs, verbose, pdec);

    // 汇总
    const bool have_file = (pdec != nullptr);
    double min_eng_adq = 1.0, min_adq_orig = 1.0, maxabs = 0, min_file_adq = 1.0;
    long long nf = 0;
    int fails_eng = 0, fails_adq = 0, fails_file = 0, enc_low4 = 0, enc_low2 = 0;
    for (auto & H : res) {
        min_eng_adq = std::min(min_eng_adq, H.cos_eng_adq);
        min_adq_orig = std::min(min_adq_orig, H.cos_adq_orig);
        maxabs = std::max(maxabs, H.maxabs_eng_adq);
        nf += H.nonfinite;
        if (H.cos_eng_adq < 0.999) fails_eng++;
        if (H.cos_adq_orig < 0.999) fails_adq++;
        if (H.cos_file_adq >= 0) {
            min_file_adq = std::min(min_file_adq, H.cos_file_adq);
            if (H.cos_file_adq < 0.999) fails_file++;
        }
        if (H.cos_adq_orig < 0.99) { if (H.bits >= 4) enc_low4++; else enc_low2++; }
    }
    printf("\n================ 汇总 ================\n");
    printf("heads=%zu  min cos_ENG_ADQ=%.9f  min cos_adq_orig=%.9f  maxabs_ENG_ADQ=%.3e  nonfinite=%lld\n",
           res.size(), min_eng_adq, min_adq_orig, maxabs, nf);
    if (have_file)
        printf("min cos(引擎解码输出, 主机反量化)=%.9f  <0.999 的 head 数=%d\n", min_file_adq, fails_file);
    printf("cos_ENG_ADQ<0.999 的 head 数=%d   cos_adq_orig<0.999 的 head 数=%d"
           "（其中 bits>=4：%d，bits=2：%d）\n", fails_eng, fails_adq, enc_low4, enc_low2);

    // 判据（无形容词）：
    //   主判据 A：cos(引擎解码输出, 主机反量化) ≥ 0.999（有 --decoded 时）
    //             —— 无 --decoded 时退化为工具自洽见证：两条独立解码路径 maxabs 必须 == 0
    //   健全性 B：nonfinite == 0
    //   诊断 C：cos(主机反量化, 原值) —— 编码端质量；bits>=4 基线 0.99（K4V4 合成 ≈0.995、真实 ≈0.998）；
    //           bits=2 天然 ~0.90，不作判据（阶段 0：K4V2 超判据 25-29 倍）
    const bool A = have_file ? (fails_file == 0) : (maxabs == 0.0 && min_eng_adq == 1.0);
    const bool B = (nf == 0);
    const bool pass = A && B;
    printf("\n判据A（record 解码 == 主机反量化）：%s\n",
           have_file ? (A ? "PASS" : "FAIL") : (A ? "N/A(无 --decoded)→工具自洽见证 PASS" : "FAIL"));
    printf("判据B（无 nan/inf）：%s\n", B ? "PASS" : "FAIL");
    printf("总判定：%s\n", pass ? "PASS" : "FAIL");
    if (!A) printf("  -> 病灶在【解码端】：cos(引擎解码输出, 主机反量化) < 0.999"
                   "（引擎读路径：列因子/行 scale 折叠、位序寻址、逆旋转）\n");
    if (enc_low4 > 0) printf("  -> 病灶在【编码端】：bits>=4 的主机反量化已偏离原值 cos<0.99（量化 kernel / 旋转 / stage）\n");
    if (enc_low2 > 0 && enc_low4 == 0)
        printf("  -> [诊断·非故障] bits=2 的 cos≈0.90 是 2-bit 量化噪声本身（阶段 0 已定 K4V2 不上线）\n");
    return pass ? 0 : 1;
}

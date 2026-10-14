// =============================================================================
// decode_harness.cu  ——  P1 页解码器件的 (a) 数据生产 (b) 主机参照 (c) GPU/CPU 对拍
//
// 链路：
//   oracle 口径量化（逐字只读复制） → record（slice0/slice1）
//     ├─ 路径 A：主机反量化（record 读法）+ 逆旋转          = 「期望」（任务口径）
//     ├─ 路径 B：record → P1 两页 plane → P1 CPU 参照解码 + 逆旋转（独立第二实现）
//     └─ 路径 C：P1 两页 plane → GPU device 解码（kvarn128_decode.cuh）+ 逆旋转
//
// 判据：
//   * device(C) 与 CPU 参照(B) 逐元素：0 差（f32 旋转域）或 ≤1e-6
//   * B 与 A 逐元素：应 0 差（证明 P1 重打包是忠实的位双射，不是自洽）
//   * 逐 token cos(C, A) ≥ 0.999，nonfinite == 0
//   * 逆旋转反证：C 不做逆旋转 vs A（含逆旋转）→ cos ≈ 0
//   * 槽位表交叉核验：本文件的槽位公式 vs offsets-g128 机器生成的 p1_slot_map_g128.csv
//
// 设备：本机 RTX 3080 / sm_86（docker --gpus all）。不 ssh x99，不碰引擎目录。
// 编译：nvcc -O2 -std=c++17 -arch=sm_86 -Xcompiler -ffp-contract=off（禁用 FMA 收缩，
//       让主机与设备的 (q*sa+zp)*so 与蝶形舍入完全一致；不得开 -use_fast_math）
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <cfloat>
#include <climits>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <random>
#include <map>
#include <tuple>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "kvarn128_decode.cuh"
#include "kvarn128_cpu_reference.h"

// =============================================================================
// §0 oracle 口径量化器 —— 逐字只读复制自
//    oracle/bee_oracle.cpp (sha256 2c1cb247..., 只读，未改动)
//    f32_to_f16_rn / f16_to_f32 / h16 / bee_packed_bytes / make_record_layout /
//    bee_pack_bits / bee_unpack_bits_value / std_of / sinkhorn_gpu / QuantMeta /
//    bee_quant_tile / bee_dequant_tile / fill_gauss / load_tiles_bin
//    （verify-g128 也做了同样的只读 import，gate2 §selftest 已证副本不走样）
// =============================================================================
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
        if (rest > 0x1000u || (rest == 0x1000u && (mant & 1u))) base = (uint16_t)(base + 1u);
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

static size_t bee_packed_bytes(int n_values, int bits) { return ((size_t)n_values * (size_t)bits + 7) / 8; }

struct RecordLayout { int rows, cols; size_t payload_bytes, scale_off, zp_off, other_off, record_bytes; };
static RecordLayout make_record_layout(int record_dim, int bits, bool value) {
    RecordLayout R{};
    R.rows = value ? 128 : record_dim;
    R.cols = value ? record_dim : 128;
    R.payload_bytes = bee_packed_bytes(R.rows * R.cols, bits);
    R.scale_off = R.payload_bytes;
    R.zp_off    = R.scale_off + (size_t)R.rows * 2;
    R.other_off = R.zp_off + (size_t)R.rows * 2;
    R.record_bytes = R.other_off + (size_t)R.cols * 2;
    return R;
}

static void bee_pack_bits(const uint8_t* values, int n_values, int bits, uint8_t* dst) {
    std::memset(dst, 0, bee_packed_bytes(n_values, bits));
    const uint8_t mask = (uint8_t)((1u << bits) - 1u);
    for (int i = 0; i < n_values; ++i) {
        const uint8_t value = (uint8_t)(values[i] & mask);
        const size_t bit_offset = (size_t)i * (size_t)bits;
        for (int bit = 0; bit < bits; ++bit) {
            const size_t dst_bit = bit_offset + (size_t)bit;
            dst[dst_bit / 8] |= (uint8_t)(((value >> bit) & 1u) << (dst_bit % 8));
        }
    }
}
static uint8_t bee_unpack_bits_value(const uint8_t* src, int index, int bits) {
    uint8_t value = 0;
    const size_t bit_offset = (size_t)index * (size_t)bits;
    for (int bit = 0; bit < bits; ++bit) {
        const size_t src_bit = bit_offset + (size_t)bit;
        value |= (uint8_t)(((src[src_bit / 8] >> (src_bit % 8)) & 1u) << bit);
    }
    return value;
}

struct Sink { std::vector<float> s_col, s_row; float best_imbalance; int best_iter; };
static inline float std_of(const float* p, int n, int stride, const float* s_col, const float* s_row,
                           int row_index, bool is_col) {
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = 0; i < n; ++i) {
        const float v = is_col ? p[i * stride] / (s_col[row_index] * s_row[i])
                               : p[i * stride] / (s_col[i] * s_row[row_index]);
        sum += v; sum_sq += v * v;
    }
    const float mean = sum / (float)n;
    return sqrtf(fmaxf((sum_sq - (float)n * mean * mean) / (float)(n - 1), 0.0f));
}
static void sinkhorn_gpu(const float* tile, int R, int C, int iters, Sink& out) {
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
static void bee_quant_tile(const float* tile, int rows, int cols, int bits, int iters,
                           std::vector<uint8_t>& record, QuantMeta* meta) {
    Sink S;
    sinkhorn_gpu(tile, rows, cols, iters, S);
    const int qmax = (1 << bits) - 1;
    std::vector<uint8_t> q((size_t)rows * cols);
    std::vector<float> scale_axis(rows), zp_axis(rows);
    for (int r = 0; r < rows; ++r) {
        float lo = FLT_MAX, hi = -FLT_MAX;
        for (int c = 0; c < cols; ++c) {
            const float x = tile[(size_t)r * cols + c] / (S.s_col[c] * S.s_row[r]);
            lo = fminf(lo, x); hi = fmaxf(hi, x);
        }
        const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
        for (int c = 0; c < cols; ++c) {
            const float x = tile[(size_t)r * cols + c] / (S.s_col[c] * S.s_row[r]);
            q[(size_t)r * cols + c] = (uint8_t)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
        }
        scale_axis[r] = h16(S.s_row[r] * scale);
        zp_axis[r]    = h16(S.s_row[r] * lo);
    }
    const size_t payload_bytes = bee_packed_bytes(rows * cols, bits);
    record.assign(payload_bytes + (size_t)(2 * rows + cols) * 2, 0);
    bee_pack_bits(q.data(), rows * cols, bits, record.data());
    for (int r = 0; r < rows; ++r) {
        uint16_t h;
        h = f32_to_f16_rn(scale_axis[r]); std::memcpy(&record[payload_bytes + (size_t)r * 2], &h, 2);
        h = f32_to_f16_rn(zp_axis[r]);    std::memcpy(&record[payload_bytes + (size_t)rows * 2 + (size_t)r * 2], &h, 2);
    }
    for (int c = 0; c < cols; ++c) {
        const uint16_t h = f32_to_f16_rn(S.s_col[c]);
        std::memcpy(&record[payload_bytes + (size_t)(2 * rows) * 2 + (size_t)c * 2], &h, 2);
    }
    if (meta) { meta->best_col = S.s_col; meta->best_row = S.s_row; meta->q = q;
                meta->best_imbalance = S.best_imbalance; meta->best_iter = S.best_iter; }
}
static void bee_dequant_tile(const uint8_t* record, int rows, int cols, int bits, float* tile) {
    const size_t payload_bytes = bee_packed_bytes(rows * cols, bits);
    for (int r = 0; r < rows; ++r) {
        uint16_t hs, hz;
        std::memcpy(&hs, record + payload_bytes + (size_t)r * 2, 2);
        std::memcpy(&hz, record + payload_bytes + (size_t)rows * 2 + (size_t)r * 2, 2);
        const float scale = f16_to_f32(hs), zp = f16_to_f32(hz);
        for (int c = 0; c < cols; ++c) {
            uint16_t ho;
            std::memcpy(&ho, record + payload_bytes + (size_t)(2 * rows) * 2 + (size_t)c * 2, 2);
            const float other = f16_to_f32(ho);
            const uint8_t q = bee_unpack_bits_value(record, r * cols + c, bits);
            tile[(size_t)r * cols + c] = (float(q) * scale + zp) * other;
        }
    }
}
static void fill_gauss(std::vector<float>& d, int rows, int cols, uint64_t seed, bool outlier) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    d.resize((size_t)rows * cols);
    for (auto& v : d) v = nd(rng);
    if (outlier) {
        for (int c = 0; c < cols; ++c) {
            d[0 * cols + c] *= 20.f; d[7 * cols + c] *= 20.f; d[63 * cols + c] *= 20.f;
            for (int r = 0; r < rows; ++r) {
                std::student_t_distribution<float> td(3.0);
                d[r * cols + c] *= (1.f + 0.25f * std::fabs(td(rng)));
            }
        }
    }
}
// 真实 tile（oracle/make_tiles.py 口径）：int32 n; repeat{ int32 name_len, name[], i32 rows,i32 cols,i32 token_is_row, f32 data[] }
struct RawTile { std::string name; int rows = 0, cols = 0, token_is_row = 0; std::vector<float> d; };
static std::vector<RawTile> load_tiles_bin(const char* path) {
    std::vector<RawTile> out;
    FILE* fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "无法打开 %s\n", path); exit(2); }
    int32_t n = 0;
    if (fread(&n, 4, 1, fp) != 1) { fprintf(stderr, "读 n 失败\n"); exit(2); }
    for (int i = 0; i < n; ++i) {
        int32_t nl = 0; if (fread(&nl, 4, 1, fp) != 1) exit(2);
        std::string nm(nl, '\0'); if (nl) { if (fread(&nm[0], 1, nl, fp) != (size_t)nl) exit(2); }
        RawTile t; t.name = nm;
        int32_t a[3];
        if (fread(a, 4, 3, fp) != 3) exit(2);
        t.rows = a[0]; t.cols = a[1]; t.token_is_row = a[2];
        t.d.resize((size_t)t.rows * t.cols);
        if (fread(t.d.data(), 4, t.d.size(), fp) != t.d.size()) exit(2);
        out.push_back(std::move(t));
    }
    fclose(fp);
    return out;
}

// =============================================================================
// §1 P1 生产/消费共用常量
// =============================================================================
static constexpr int PAGE   = 64;
static constexpr int DIM    = 256;
static constexpr int SLICES = 2;
static constexpr int SDIM   = 128;
static constexpr int KSL    = 10;   // k_scale_leading
static constexpr int VSL    = 8;    // v_scale_leading
static constexpr int HEADS  = 4;    // KV head 数
static constexpr int PHYSP  = 8;    // plane 物理页数（>2，用来考寻址）

// record 拆段视图
struct RecView {
    const uint8_t* code = nullptr; const uint16_t* sa = nullptr; const uint16_t* zp = nullptr;
    const uint16_t* other = nullptr;
    int rows = 0, cols = 0, bits = 0; bool is_v = false;
    size_t payload_bytes = 0;
};
static RecView view_record(const std::vector<uint8_t>& rec, int bits, bool is_v) {
    const RecordLayout L = make_record_layout(128, bits, is_v);
    RecView V;
    V.code = rec.data(); V.payload_bytes = L.payload_bytes;
    V.sa    = (const uint16_t*)(rec.data() + L.scale_off);
    V.zp    = (const uint16_t*)(rec.data() + L.zp_off);
    V.other = (const uint16_t*)(rec.data() + L.other_off);
    V.rows = L.rows; V.cols = L.cols; V.bits = bits; V.is_v = is_v;
    return V;
}

// ---- P1 双页生产：两条 slice record → 四条 plane 的两页 ----
// 码 plane：页内行 = token%64；行内位偏移 = d*bits（d = slice*128 + r），LSB-first
//   K：record 码索引 = r*cols + T（row=dim）
//   V：record 码索引 = T*cols + r（row=token）
// scale plane：见 P1几何与偏移表 §3.2/§3.3
struct P1Pages {
    int bits = 0; bool is_v = false;
    int code_leading = 0, scale_leading = 0;
    std::vector<uint8_t>  code;    // 2 页 × 64 行 × code_leading
    std::vector<uint16_t> scale;   // 2 页 × 64 行 × scale_leading
};
static P1Pages make_p1_pages(const RecView& s0, const RecView& s1, bool is_v) {
    P1Pages P; P.is_v = is_v; P.bits = s0.bits;
    P.code_leading  = 32 * s0.bits;
    P.scale_leading = is_v ? VSL : KSL;
    P.code.assign((size_t)2 * PAGE * P.code_leading, 0);
    P.scale.assign((size_t)2 * PAGE * P.scale_leading, 0);
    const RecView* R[2] = { &s0, &s1 };
    for (int p = 0; p < 2; ++p) {
        uint8_t*  cp = P.code.data()  + (size_t)p * PAGE * P.code_leading;
        uint16_t* sp = P.scale.data() + (size_t)p * PAGE * P.scale_leading;
        for (int lt = 0; lt < PAGE; ++lt) {
            const int T = p * PAGE + lt;                 // tile 内 token 号 0..127
            uint8_t* row = cp + (size_t)lt * P.code_leading;
            for (int s = 0; s < SLICES; ++s) {
                for (int r = 0; r < SDIM; ++r) {
                    const int d = s * SDIM + r;
                    const int src_index = is_v ? (T * SDIM + r) : (r * SDIM + T);
                    const uint8_t q = bee_unpack_bits_value(R[s]->code, src_index, P.bits);
                    const int bit = d * P.bits;
                    for (int b = 0; b < P.bits; ++b) {
                        const int db = bit + b;
                        row[db >> 3] |= (uint8_t)(((q >> b) & 1u) << (db & 7));
                    }
                    if (!is_v) {  // K：scale/zp 维轴
                        const int arid = d >> 2, acol = d & 3;
                        sp[(size_t)arid * KSL + 0 + acol] = R[s]->sa[r];
                        sp[(size_t)arid * KSL + 4 + acol] = R[s]->zp[r];
                    } else {      // V：other 维轴
                        const int arid = d >> 2, acol = d & 3;
                        sp[(size_t)arid * VSL + 0 + acol] = R[s]->other[r];
                    }
                }
                if (!is_v) {      // K：other token 轴（列 8+slice）
                    sp[(size_t)lt * KSL + 8 + s] = R[s]->other[T];
                } else {          // V：scale/zp token 轴（列 4+slice / 6+slice）
                    sp[(size_t)lt * VSL + 4 + s] = R[s]->sa[T];
                    sp[(size_t)lt * VSL + 6 + s] = R[s]->zp[T];
                }
            }
        }
    }
    return P;
}

// ---- 反向重建：P1 双页 → 两条 record（双射断言用）----
static std::vector<uint8_t> p1_to_record(const P1Pages& P, int slice, int bits, bool is_v) {
    const RecordLayout L = make_record_layout(128, bits, is_v);
    std::vector<uint8_t> rec(L.record_bytes, 0);
    RecView V = view_record(rec, bits, is_v);
    for (int p = 0; p < 2; ++p) {
        const uint8_t*  cp = P.code.data()  + (size_t)p * PAGE * P.code_leading;
        const uint16_t* sp = P.scale.data() + (size_t)p * PAGE * P.scale_leading;
        for (int lt = 0; lt < PAGE; ++lt) {
            const int T = p * PAGE + lt;
            const uint8_t* row = cp + (size_t)lt * P.code_leading;
            for (int r = 0; r < SDIM; ++r) {
                const int d = slice * SDIM + r;
                uint8_t q = 0;
                const int bit = d * bits;
                for (int b = 0; b < bits; ++b) { const int db = bit + b; q |= (uint8_t)(((row[db >> 3] >> (db & 7)) & 1u) << b); }
                const int dst_index = is_v ? (T * SDIM + r) : (r * SDIM + T);
                for (int b = 0; b < bits; ++b) {
                    const size_t db = (size_t)dst_index * bits + b;
                    rec[db / 8] |= (uint8_t)(((q >> b) & 1u) << (db % 8));
                }
            }
        }
    }
    // 轴条目：K 的维轴在页 0/1 应相同，token 轴按 T 取本页；V 反之
    uint16_t* sa = const_cast<uint16_t*>(V.sa);
    uint16_t* zp = const_cast<uint16_t*>(V.zp);
    uint16_t* ot = const_cast<uint16_t*>(V.other);
    for (int r = 0; r < SDIM; ++r) {
        const int d = slice * SDIM + r;
        const int arid = d >> 2, acol = d & 3;
        if (!is_v) {
            sa[r] = P.scale[(size_t)arid * KSL + 0 + acol];
            zp[r] = P.scale[(size_t)arid * KSL + 4 + acol];
        } else {
            ot[r] = P.scale[(size_t)arid * VSL + 0 + acol];
        }
    }
    for (int p = 0; p < 2; ++p) {
        const uint16_t* sp = P.scale.data() + (size_t)p * PAGE * P.scale_leading;
        for (int lt = 0; lt < PAGE; ++lt) {
            const int T = p * PAGE + lt;
            if (!is_v) ot[T] = sp[(size_t)lt * KSL + 8 + slice];
            else { sa[T] = sp[(size_t)lt * VSL + 4 + slice]; zp[T] = sp[(size_t)lt * VSL + 6 + slice]; }
        }
    }
    return rec;
}

// 维轴在两页的副本一致性（offsets-g128 §5-④ 的同一断言）
static long long dim_axis_dup_check(const P1Pages& P) {
    long long bad = 0;
    const int SL = P.is_v ? VSL : KSL;
    for (int lt = 0; lt < PAGE; ++lt) {
        for (int c = 0; c < 4; ++c) {                        // other/scale+zp 维轴列
            if (P.scale[(size_t)lt * SL + c] != P.scale[(size_t)PAGE * SL + (size_t)lt * SL + c]) ++bad;
            if (!P.is_v) {
                if (P.scale[(size_t)lt * SL + 4 + c] != P.scale[(size_t)PAGE * SL + (size_t)lt * SL + 4 + c]) ++bad;
            }
        }
    }
    return bad;
}

// =============================================================================
// §2 设备核（薄封装；真正逻辑在 kvarn128_decode.cuh）
// =============================================================================
template <bool IsValue, int Warps>
__global__ void decode_page_kernel(const uint8_t* __restrict__ code_plane, int code_leading, int bits,
                                   const uint16_t* __restrict__ scale_plane, int scale_leading, int heads,
                                   const int32_t* __restrict__ block_table, int page_index, int kv_head,
                                   bool inverse_rotate, __nv_bfloat16* __restrict__ out_tile,
                                   float* __restrict__ out_rot) {
    const int32_t phys = block_table[page_index];
    const uint8_t*  cp = code_plane  + kvarn128::page_head_offset_rt(code_leading, heads, phys, kv_head);
    const uint16_t* sp = scale_plane + kvarn128::page_head_offset_rt(scale_leading, heads, phys, kv_head);
    kvarn128::decode_page_bf16<IsValue, Warps>(cp, code_leading, bits, sp, scale_leading, inverse_rotate,
                                               out_tile, out_rot);
}
// 逆旋转自检核：同一行跑两次 H（应回到输入，验证自逆+归一化）
__global__ void hadamard_selfcheck_kernel(const float* __restrict__ in, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    float v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = in[lane + 32 * r];
    kvarn128::inverse_rotate_d256_row(v, lane);
    kvarn128::inverse_rotate_d256_row(v, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) out[lane + 32 * r] = v[r];
}

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA 错误 %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e_)); exit(3); } } while (0)

// =============================================================================
// §3 指标
// =============================================================================
struct CosStat { double min_cos = 2.0, mean_cos = 0.0, max_abs = 0.0; int worst_tok = -1;
                 long long nonfinite = 0, n = 0; std::vector<double> per_tok; };
static double cosine256(const float* a, const float* b, int off, int len, long long& nonfinite, double& maxabs) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < len; ++i) {
        const double x = a[off + i], y = b[off + i];
        if (!std::isfinite(x) || !std::isfinite(y)) ++nonfinite;
        const double d = x - y; maxabs = std::max(maxabs, std::fabs(d));
        dot += x * y; na += x * x; nb += y * y;
    }
    return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-300);
}
// 逐 token 余弦（tok0 起 ntok 个 token；off/len 选定 slice）
static CosStat token_cos(const std::vector<float>& dec, const std::vector<float>& ref,
                         int tok0, int ntok, int off, int len) {
    CosStat S; S.per_tok.resize(ntok, 0.0);
    double sum = 0;
    for (int i = 0; i < ntok; ++i) {
        const int t = tok0 + i;
        long long nf = 0; double ma = 0;
        const double c = cosine256(dec.data(), ref.data(), t * DIM + off, len, nf, ma);
        S.per_tok[i] = c; sum += c;
        if (c < S.min_cos) { S.min_cos = c; S.worst_tok = t; }
        S.nonfinite += nf; S.max_abs = std::max(S.max_abs, ma);
        ++S.n;
    }
    S.mean_cos = S.n ? sum / (double)S.n : 0.0;
    return S;
}
static long long exact_mismatch_f32(const std::vector<float>& a, const std::vector<float>& b, double& maxabs) {
    long long m = 0; maxabs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) ++m;
        maxabs = std::max(maxabs, (double)std::fabs((double)a[i] - (double)b[i]));
    }
    return m;
}
static std::string bf16_bits(__nv_bfloat16 v) { uint16_t u; std::memcpy(&u, &v, 2); char b[8]; snprintf(b, 8, "0x%04x", u); return b; }

// =============================================================================
// §4 主流程
// =============================================================================
struct TierDef { const char* name; int kb, vb; };

struct RunCtx {
    // 数据
    std::vector<float> x;                    // [128 token][256 dim] 量化前（旋转域，未截断的）
    std::string src_name;
    int ntok = 128;
};

static std::vector<float> g_x;        // 当前 run 的量化前数据（旋转域）[128 token][256 dim]
static std::vector<float> g_x_unrot;  // 同一数据的未旋转原值（对比用）

static void run_one(const TierDef& tier, bool kind_v, const std::string& src_name, int ntok,
                    FILE* per_tok_fp, std::vector<std::string>& summary) {
    const int bits = kind_v ? tier.vb : tier.kb;
    const char kindch = kind_v ? 'V' : 'K';

    // ---- 1) 两条 slice record ----
    std::vector<uint8_t> rec[2];
    for (int s = 0; s < SLICES; ++s) {
        std::vector<float> rtile((size_t)SDIM * SDIM, 0.f);
        for (int t = 0; t < 128; ++t) {
            const bool valid = (t < ntok);
            for (int r = 0; r < SDIM; ++r) {
                const int d = s * SDIM + r;
                const float v = valid ? g_x[(size_t)t * DIM + d] : 0.f;
                if (kind_v) rtile[(size_t)t * SDIM + r] = v;     // V: rows=token, cols=dim
                else        rtile[(size_t)r * SDIM + t] = v;     // K: rows=dim,   cols=token
            }
        }
        bee_quant_tile(rtile.data(), SDIM, SDIM, bits, 16, rec[s], nullptr);
    }
    RecView R0 = view_record(rec[0], bits, kind_v);
    RecView R1 = view_record(rec[1], bits, kind_v);

    // ---- 2) 路径 B：P1 双页 → CPU 参照解码 → 逆旋转 → bf16 参照 ----
    P1Pages P = make_p1_pages(R0, R1, kind_v);

    // 双射断言：P1 → record 逐字节重建
    long long bij_bad = 0;
    for (int s = 0; s < SLICES; ++s) {
        std::vector<uint8_t> back = p1_to_record(P, s, bits, kind_v);
        if (back.size() != rec[s].size()) { bij_bad += 1; continue; }
        for (size_t i = 0; i < back.size(); ++i) if (back[i] != rec[s][i]) { ++bij_bad;
            fprintf(stderr, "[bij] tier=%s kind=%c slice=%d 首差异 off=%zu (rec=0x%02x back=0x%02x) size=%zu\n",
                    tier.name, kindch, s, i, rec[s][i], back[i], rec[s].size());
            break; }
    }

    // ---- 3) 路径 A：主机反量化（record 读法）+ 逆旋转 = 期望 ----
    std::vector<float> exp_rot((size_t)128 * DIM, 0.f);
    for (int s = 0; s < SLICES; ++s) {
        std::vector<float> dq((size_t)SDIM * SDIM);
        bee_dequant_tile(rec[s].data(), SDIM, SDIM, bits, dq.data());
        for (int t = 0; t < 128; ++t)
            for (int r = 0; r < SDIM; ++r) {
                const float v = kind_v ? dq[(size_t)t * SDIM + r] : dq[(size_t)r * SDIM + t];
                exp_rot[(size_t)t * DIM + s * SDIM + r] = v;
            }
    }
    std::vector<float> exp_bf16f((size_t)128 * DIM);
    std::vector<__nv_bfloat16> exp_bf16((size_t)128 * DIM);
    for (int t = 0; t < 128; ++t) {
        float buf[DIM];
        for (int d = 0; d < DIM; ++d) buf[d] = exp_rot[(size_t)t * DIM + d];
        kvarn128_cpu::inverse_rotate_d256(buf);
        for (int d = 0; d < DIM; ++d) {
            exp_bf16f[(size_t)t * DIM + d] = __bfloat162float(__float2bfloat16_rn(buf[d]));
            exp_bf16[(size_t)t * DIM + d] = __float2bfloat16_rn(buf[d]);
        }
    }

    // ---- 4) 路径 B 落盘：CPU 参照（P1 页）f32 旋转域 + bf16（逆旋转后）----
    std::vector<float> ref_rot((size_t)128 * DIM);
    std::vector<float> ref_bf16f((size_t)128 * DIM);
    for (int p = 0; p < 2; ++p) {
        const uint8_t*  cp = P.code.data()  + (size_t)p * PAGE * P.code_leading;
        const uint16_t* sp = P.scale.data() + (size_t)p * PAGE * P.scale_leading;
        for (int lt = 0; lt < PAGE; ++lt) {
            const int t = p * PAGE + lt;
            float buf[DIM];
            kvarn128_cpu::decode_token_rotated(kind_v, cp, bits, sp, lt, buf);
            for (int d = 0; d < DIM; ++d) ref_rot[(size_t)t * DIM + d] = buf[d];
            kvarn128_cpu::inverse_rotate_d256(buf);
            for (int d = 0; d < DIM; ++d) ref_bf16f[(size_t)t * DIM + d] = __bfloat162float(__float2bfloat16_rn(buf[d]));
        }
    }

    // ---- 5) P1（路径 B）vs 主机反量化（路径 A）逐元素 ----
    double ab_maxabs = 0; long long ab_mismatch = 0;
    for (size_t i = 0; i < ref_rot.size(); ++i) {
        if (ref_rot[i] != exp_rot[i]) ++ab_mismatch;
        ab_maxabs = std::max(ab_maxabs, (double)std::fabs((double)ref_rot[i] - (double)exp_rot[i]));
    }

    // ---- 6) GPU：路径 C ----
    const int code_leading  = P.code_leading;
    const int scale_leading = P.scale_leading;
    const size_t code_elems  = (size_t)code_leading  * PAGE * HEADS * PHYSP;
    const size_t scale_elems = (size_t)scale_leading * PAGE * HEADS * PHYSP;
    uint8_t*  d_code = nullptr; uint16_t* d_scale = nullptr; int32_t* d_bt = nullptr;
    __nv_bfloat16* d_out = nullptr; __nv_bfloat16* d_out_nr = nullptr; float* d_rot = nullptr;
    __nv_bfloat16* d_outb = nullptr; __nv_bfloat16* d_out_nr_b = nullptr; float* d_rotb = nullptr;
    CK(cudaMalloc(&d_code,  code_elems));
    CK(cudaMalloc(&d_scale, scale_elems * 2));
    CK(cudaMalloc(&d_bt, 8 * 4));
    CK(cudaMalloc(&d_out,    (size_t)128 * DIM * 2));
    CK(cudaMalloc(&d_out_nr, (size_t)128 * DIM * 2));
    CK(cudaMalloc(&d_rot,    (size_t)128 * DIM * 4));
    CK(cudaMalloc(&d_outb,    (size_t)128 * DIM * 2));
    CK(cudaMalloc(&d_out_nr_b,(size_t)128 * DIM * 2));
    CK(cudaMalloc(&d_rotb,    (size_t)128 * DIM * 4));

    // 页散布：把 tile 的 2 页放到物理页 5 与 2，头 3（考寻址式）
    const int phys_of[2] = { 5, 2 };
    const int kv_head = 3;
    std::vector<int32_t> bt(8, 0); bt[0] = phys_of[0]; bt[1] = phys_of[1];
    std::vector<uint8_t>  code_w(code_elems, 0);
    std::vector<uint16_t> scale_w(scale_elems, 0);
    for (int p = 0; p < 2; ++p) {
        const size_t cbase = kvarn128::page_head_offset_rt(code_leading, HEADS, phys_of[p], kv_head);
        const size_t sbase = kvarn128::page_head_offset_rt(scale_leading, HEADS, phys_of[p], kv_head);
        std::memcpy(&code_w[cbase],  P.code.data()  + (size_t)p * PAGE * code_leading,  (size_t)PAGE * code_leading);
        std::memcpy(&scale_w[sbase], P.scale.data() + (size_t)p * PAGE * scale_leading, (size_t)PAGE * scale_leading * 2);
    }
    CK(cudaMemcpy(d_code, code_w.data(), code_elems, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_scale, scale_w.data(), scale_elems * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_bt, bt.data(), bt.size() * 4, cudaMemcpyHostToDevice));

    constexpr int WARPS = 8;
    const dim3 blk(32 * WARPS), grd(1);
    for (int p = 0; p < 2; ++p) {
        if (kind_v) {
            decode_page_kernel<true, WARPS><<<grd, blk>>>(d_code, code_leading, bits, d_scale, scale_leading,
                                                          HEADS, d_bt, p, kv_head, true,
                                                          d_out + (size_t)p * PAGE * DIM,
                                                          d_rot + (size_t)p * PAGE * DIM);
            decode_page_kernel<true, WARPS><<<grd, blk>>>(d_code, code_leading, bits, d_scale, scale_leading,
                                                          HEADS, d_bt, p, kv_head, false,
                                                          d_out_nr + (size_t)p * PAGE * DIM, nullptr);
        } else {
            decode_page_kernel<false, WARPS><<<grd, blk>>>(d_code, code_leading, bits, d_scale, scale_leading,
                                                           HEADS, d_bt, p, kv_head, true,
                                                           d_out + (size_t)p * PAGE * DIM,
                                                           d_rot + (size_t)p * PAGE * DIM);
            decode_page_kernel<false, WARPS><<<grd, blk>>>(d_code, code_leading, bits, d_scale, scale_leading,
                                                           HEADS, d_bt, p, kv_head, false,
                                                           d_out_nr + (size_t)p * PAGE * DIM, nullptr);
        }
    }
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    std::vector<float> dev_rot((size_t)128 * DIM), dev_bf16f((size_t)128 * DIM), dev_nr_bf16f((size_t)128 * DIM);
    std::vector<__nv_bfloat16> dev_bf16((size_t)128 * DIM), dev_nr_bf16((size_t)128 * DIM);
    CK(cudaMemcpy(dev_rot.data(), d_rot, dev_rot.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dev_bf16.data(), d_out, dev_bf16.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(dev_nr_bf16.data(), d_out_nr, dev_nr_bf16.size() * 2, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < dev_bf16.size(); ++i) dev_bf16f[i] = __bfloat162float(dev_bf16[i]);
    for (size_t i = 0; i < dev_nr_bf16.size(); ++i) dev_nr_bf16f[i] = __bfloat162float(dev_nr_bf16[i]);

    // ---- 7) 对拍 ----
    double maxabs_rot = 0; const long long mm_rot = exact_mismatch_f32(dev_rot, ref_rot, maxabs_rot);
    long long mm_bf16 = 0; double maxabs_bf16 = 0;
    for (size_t i = 0; i < dev_bf16.size(); ++i) { if (dev_bf16[i] != exp_bf16[i]) ++mm_bf16;
        maxabs_bf16 = std::max(maxabs_bf16, (double)std::fabs((double)dev_bf16f[i] - (double)exp_bf16f[i])); }

    // 报告
    std::string line = std::string("[") + tier.name + "|" + kindch + "|" + src_name +
                       "|tok=" + std::to_string(ntok) + "]";
    for (int p = 0; p < 2; ++p) {
        const int nv = std::max(0, std::min(PAGE, ntok - p * PAGE));
        char buf[512];
        if (nv > 0) {
            const int tok0 = p * PAGE;
            const CosStat Sall = token_cos(dev_bf16f, exp_bf16f, tok0, nv, 0, DIM);
            const CosStat S0   = token_cos(dev_bf16f, exp_bf16f, tok0, nv, 0, SDIM);
            const CosStat S1   = token_cos(dev_bf16f, exp_bf16f, tok0, nv, SDIM, SDIM);
            const CosStat Srv  = token_cos(dev_nr_bf16f, exp_bf16f, tok0, nv, 0, DIM);
            const CosStat Srv0 = token_cos(dev_nr_bf16f, exp_bf16f, tok0, nv, 0, SDIM);
            const CosStat Srv1 = token_cos(dev_nr_bf16f, exp_bf16f, tok0, nv, SDIM, SDIM);
            // 与「量化前原值」（未旋转）比：诊断 D（编码质量）+ 强反证
            const CosStat Sorig_w = token_cos(dev_bf16f, g_x_unrot, tok0, nv, 0, DIM);
            const CosStat Sorig_nw = token_cos(dev_nr_bf16f, g_x_unrot, tok0, nv, 0, DIM);
            snprintf(buf, sizeof buf,
                     "  page%d valid=%2d  cos_min=%.9f cos_mean=%.9f (worst tok %d) maxabs=%.3e nonfinite=%lld"
                     " | slice0 cos_min=%.9f slice1 cos_min=%.9f"
                     " | 反证(不逆旋转) cos_min=%.9f cos_mean=%.9f s0=%.9f s1=%.9f"
                     " | vs原值: 逆旋转后 cos_min=%.9f / 不逆旋转 cos_min=%.9f cos_mean=%.9f",
                     p, nv, Sall.min_cos, Sall.mean_cos, Sall.worst_tok, Sall.max_abs, Sall.nonfinite,
                     S0.min_cos, S1.min_cos, Srv.min_cos, Srv.mean_cos, Srv0.min_cos, Srv1.min_cos,
                     Sorig_w.min_cos, Sorig_nw.min_cos, Sorig_nw.mean_cos);
            line += "\n" + std::string(buf);
            if (per_tok_fp) {
                for (int i = 0; i < nv; ++i)
                    fprintf(per_tok_fp, "%s %s page=%d tok=%d cos=%.9f cos_s0=%.9f cos_s1=%.9f cos_norot=%.9f cos_norot_vs_orig=%.9f\n",
                            tier.name, kindch == 'V' ? "V" : "K", p, tok0 + i,
                            Sall.per_tok[i], S0.per_tok[i], S1.per_tok[i], Srv.per_tok[i], Sorig_nw.per_tok[i]);
            }
        } else {
            snprintf(buf, sizeof buf, "  page%d valid= 0 (无有效 token，未计入判据)", p);
            line += "\n" + std::string(buf);
        }
    }
    {
        char buf[768];
        long long nf_full = 0;
        for (size_t i = 0; i < dev_bf16f.size(); ++i) if (!std::isfinite(dev_bf16f[i])) ++nf_full;
        const long long dup_bad = dim_axis_dup_check(P);
        snprintf(buf, sizeof buf,
                 "  对拍 device_vs_cpuP1(f32旋转域) mismatch=%lld/%zu maxabs=%.3e | P1(路径B)_vs_主机反量化(路径A) mismatch=%lld maxabs=%.3e |"
                 " device_bf16_vs_期望_bf16 mismatch=%lld/%zu maxabs=%.3e | P1→record 重建坏=%lld | 两页维轴副本失配=%lld | 全页(含未满槽位) nonfinite=%lld",
                 mm_rot, dev_rot.size(), maxabs_rot, ab_mismatch, ab_maxabs, mm_bf16, dev_bf16.size(),
                 maxabs_bf16, bij_bad, dup_bad, nf_full);
        line += "\n" + std::string(buf);
    }

    summary.push_back(line);
    printf("%s\n", line.c_str());

    cudaFree(d_code); cudaFree(d_scale); cudaFree(d_bt);
    cudaFree(d_out); cudaFree(d_out_nr); cudaFree(d_rot);
    cudaFree(d_outb); cudaFree(d_out_nr_b); cudaFree(d_rotb);
}

// =============================================================================
// §5 槽位表交叉核验：本文件公式 vs offsets-g128/out/p1_slot_map_g128.csv
// =============================================================================
static int check_slot_map_csv(const char* path, FILE* outp) {
    FILE* fp = fopen(path, "r");
    if (!fp) { fprintf(outp, "[slotmap] 打不开 %s（跳过）\n", path); return -1; }
    char line[512];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return -1; }
    long long rows = 0, bad = 0, checked = 0;
    std::map<std::string, std::pair<int,int>> tier_bits = {
        {"K4V4", {4,4}}, {"K4V2", {4,2}}, {"K4V3", {4,3}},
        {"K3V3", {3,3}}, {"K5V5", {5,5}}, {"K6V6", {6,6}} };
    while (fgets(line, sizeof line, fp)) {
        ++rows;
        char* tok[16] = { nullptr };
        int nt = 0;
        char* p = line;
        while (nt < 16) { tok[nt++] = p; char* c = strchr(p, ','); if (!c) break; *c = 0; p = c + 1; }
        if (nt < 13) continue;
        const std::string tier = tok[0], plane = tok[1], row_s = tok[2], col_s = tok[3];
        const std::string item = tok[5];
        const int slice = atoi(tok[6]), index = atoi(tok[7]);
        const int dim = atoi(tok[8]);
        const int lead = atoi(tok[9]);
        const long long eng_off = atoll(tok[12]);
        if (plane != "k_scale_pages" && plane != "v_scale_pages") continue;
        const bool is_v = (plane == "v_scale_pages");
        const int SL = is_v ? VSL : KSL;
        int mrow = -1, mcol = -1, mdim = dim;
        const int d = slice * SDIM + index;
        if (item == "k_scale_axis")      { mrow = d >> 2; mcol = 0 + (d & 3); mdim = d; }
        else if (item == "k_zp_axis")    { mrow = d >> 2; mcol = 4 + (d & 3); mdim = d; }
        else if (item == "k_other_axis") { mrow = index & (PAGE - 1); mcol = 8 + slice; mdim = -1; }
        else if (item == "v_other_axis") { mrow = d >> 2; mcol = 0 + (d & 3); mdim = d; }
        else if (item == "v_scale_axis") { mrow = index & (PAGE - 1); mcol = 4 + slice; mdim = -1; }
        else if (item == "v_zp_axis")    { mrow = index & (PAGE - 1); mcol = 6 + slice; mdim = -1; }
        else continue;
        ++checked;
        const int expect_lead = is_v ? VSL : KSL;
        const long long my_eng = (long long)SL * mrow + mcol;
        if (atoi(row_s.c_str()) != mrow || atoi(col_s.c_str()) != mcol || lead != expect_lead ||
            eng_off != my_eng || mdim != dim) {
            if (bad < 5)
                fprintf(outp, "[slotmap] 失配: %s %s csv(row=%s col=%s dim=%d lead=%d off=%lld) vs mine(row=%d col=%d dim=%d lead=%d off=%lld)\n",
                        tier.c_str(), item.c_str(), row_s.c_str(), col_s.c_str(), dim, lead, eng_off,
                        mrow, mcol, mdim, expect_lead, my_eng);
            ++bad;
        }
    }
    fclose(fp);
    fprintf(outp, "[slotmap] p1_slot_map_g128.csv 核验：读 %lld 行，纳入 %lld（scale 平面槽位），失配 %lld\n",
            rows, checked, bad);
    return (int)bad;
}

// =============================================================================
// §6 main
// =============================================================================
int main(int argc, char** argv) {
    const char* bin_path  = (argc > 1) ? argv[1] : "/oracle/data/real_g128.bin";
    const char* slot_csv  = (argc > 2) ? argv[2] : "/offsets/out/p1_slot_map_g128.csv";
    const char* out_dir   = (argc > 3) ? argv[3] : ".";

    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, dev));
    printf("================================================================================\n");
    printf("[device] %s sm_%d%d | kPageTokens=64 shift=6 | KV heads=%d | PHYSP=%d\n",
           prop.name, prop.major, prop.minor, HEADS, PHYSP);

    char per_tok_path[1024];
    snprintf(per_tok_path, sizeof per_tok_path, "%s/per_token.txt", out_dir);
    FILE* per_tok_fp = fopen(per_tok_path, "w");
    char summary_path[1024];
    snprintf(summary_path, sizeof summary_path, "%s/summary.txt", out_dir);
    FILE* summary_fp = fopen(summary_path, "w");

    std::vector<std::string> summary;

    // ---- 0) 逆旋转自检（device：H∘H = I）----
    {
        std::mt19937_64 rng(20261013ull);
        std::uniform_real_distribution<float> ud(-1.f, 1.f);
        std::vector<float> in(256), out(256, 0.f);
        for (auto& v : in) v = ud(rng);
        float* di = nullptr; float* dov = nullptr;
        CK(cudaMalloc(&di, 256 * 4)); CK(cudaMalloc(&dov, 256 * 4));
        CK(cudaMemcpy(di, in.data(), 256 * 4, cudaMemcpyHostToDevice));
        hadamard_selfcheck_kernel<<<1, 32>>>(di, dov);
        CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(out.data(), dov, 256 * 4, cudaMemcpyDeviceToHost));
        double ma = 0; for (int i = 0; i < 256; ++i) ma = std::max(ma, (double)std::fabs(out[i] - in[i]));
        printf("[hadamard] device 版 H∘H 自逆 max|Δ| = %.3e（应 ~1e-7，即 ninfer 精确 2^-4）\n", ma);
        cudaFree(di); cudaFree(dov);
    }

    // ---- 1) 槽位表核验 ----
    const int slot_bad = check_slot_map_csv(slot_csv, stdout);

    const TierDef tiers[] = { {"K4V4", 4, 4}, {"K4V2", 4, 2} };
    const int ntoks[] = { 128, 64, 51, 1 };
    const char* srcs[] = { "real", "gauss", "outlier" };

    // 真实 tile 载入并配对 slice
    std::vector<RawTile> raws = load_tiles_bin(bin_path);
    printf("[data] %s 载入 %zu 条 record\n", bin_path, raws.size());
    // (layer, kind, head) -> slice0/slice1
    struct PairK { int layer; char kind; int head; };
    std::map<std::tuple<int,char,int>, std::array<int,2>> groups;
    for (size_t i = 0; i < raws.size(); ++i) {
        int l = 0, h = 0, s = 0, g = 0; char k = '?';
        if (sscanf(raws[i].name.c_str(), "L%d_%c_h%d_s%d_g%d", &l, &k, &h, &s, &g) != 5) continue;
        auto key = std::make_tuple(l, k, h);
        if (!groups.count(key)) groups[key] = { -1, -1 };
        groups[key][s] = (int)i;
    }

    for (const TierDef& tier : tiers) {
        for (int kind_v = 0; kind_v < 2; ++kind_v) {
            for (const char* sn : srcs) {
                const std::string src_name = sn;
                // 构造 g_x[128][256] 量化前数据（旋转域）
                std::vector<float> x, xu;
                std::string label = src_name;
                if (src_name == "real") {
                    // 真实 tile 已是 make_tiles.py 旋转后的值 ⇒ g_x = 载入值；未旋转原值 = 逆旋转
                    const char want = kind_v ? 'v' : 'k';
                    bool got = false;
                    for (auto& kv : groups) {
                        if (std::get<1>(kv.first) != want) continue;
                        if (kv.second[0] < 0 || kv.second[1] < 0) continue;
                        x.assign((size_t)128 * DIM, 0.f);
                        for (int s = 0; s < SLICES; ++s) {
                            const RawTile& T = raws[kv.second[s]];
                            for (int t = 0; t < 128; ++t)
                                for (int r = 0; r < SDIM; ++r) {
                                    const float v = T.token_is_row ? T.d[(size_t)t * SDIM + r] : T.d[(size_t)r * SDIM + t];
                                    x[(size_t)t * DIM + s * SDIM + r] = v;
                                }
                        }
                        char nm[64];
                        snprintf(nm, sizeof nm, "real L%d_%c_h%d", std::get<0>(kv.first), want, std::get<2>(kv.first));
                        label = nm;
                        got = true;
                        break;
                    }
                    if (!got) { printf("[warn] real %c 无配对组，跳过\n", want); continue; }
                    xu.assign(x.size(), 0.f);
                    for (int t = 0; t < 128; ++t) {
                        float buf[DIM];
                        for (int d = 0; d < DIM; ++d) buf[d] = x[(size_t)t * DIM + d];
                        kvarn128_cpu::inverse_rotate_d256(buf);        // 自逆：还原未旋转原值
                        for (int d = 0; d < DIM; ++d) xu[(size_t)t * DIM + d] = buf[d];
                    }
                } else {
                    // 合成：先生成未旋转原值 xu，再旋转成 x 喂进量化器（与 make_tiles.py 同口径）
                    const bool outlier = (src_name == "outlier");
                    std::vector<float> g;
                    fill_gauss(g, DIM, 128, 20261013ull, outlier);     // dim 主序（离群在维上）
                    xu.assign((size_t)128 * DIM, 0.f);
                    for (int t = 0; t < 128; ++t) for (int d = 0; d < DIM; ++d) xu[(size_t)t * DIM + d] = g[(size_t)d * 128 + t];
                    x.assign(xu.size(), 0.f);
                    for (int t = 0; t < 128; ++t) {
                        float buf[DIM];
                        for (int d = 0; d < DIM; ++d) buf[d] = xu[(size_t)t * DIM + d];
                        kvarn128_cpu::inverse_rotate_d256(buf);
                        for (int d = 0; d < DIM; ++d) x[(size_t)t * DIM + d] = buf[d];
                    }
                }
                g_x = x; g_x_unrot = xu;
                for (int ntok : ntoks) {
                    printf("--------------------------------------------------------------------------------\n");
                    printf("=== tier=%s kind=%c src=%s ntok=%d ===\n", tier.name, kind_v ? 'V' : 'K', label.c_str(), ntok);
                    run_one(tier, kind_v != 0, label, ntok, per_tok_fp, summary);
                }
            }
        }
    }

    printf("\n================================================================================\n");
    printf("汇总（每行 = 一条 run）\n");
    for (const std::string& s : summary) { printf("%s\n", s.c_str()); if (summary_fp) fprintf(summary_fp, "%s\n", s.c_str()); }
    printf("\n[slotmap] 失配 = %d\n", slot_bad);
    if (per_tok_fp) fclose(per_tok_fp);
    if (summary_fp) fclose(summary_fp);
    printf("[out] summary.txt / per_token.txt 已写入 %s\n", out_dir);
    return 0;
}

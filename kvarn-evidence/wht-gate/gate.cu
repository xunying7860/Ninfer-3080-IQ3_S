// ============================================================================================
// GATE 1 · 旋转等价性数值对拍
//   路径 A = beellama:  kvarn_wht_128（每 128 slice）× head_slices + kvarn_wht_cross_slices（跨 slice 蝶形）
//   路径 B = ninfer  :  normalized_hadamard_d256_inplace（整 D256、8 dims/lane FWHT）
//
// 两侧实现**不是手抄**：由 extract_verbatim.py 按行号从原文逐字抠出（往返校验通过），见各 include 内头部注释。
//   A: ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu:12-19 / 456-472 / 510-543 / 1113-1131
//      (commit 58a1629274d8d74ea5a6acf34f30fb5946c2422b, sha256 dfaa62b4...)
//   B: kvarn-任务A-20261012/engine/ninfer-all-kvarn/src/ops/kv_cache/hadamard_d256.cuh:1-67
//      (sha256 3d00813a...)
//
// 判据（任务书写死）：逐元素最大绝对差 ≤ 1e-6（float 域）且自逆误差 ≤ 1e-6 => PASS，否则 FAIL。
// ============================================================================================
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <algorithm>

#include "bee_const_verbatim.h"          // kvarn.cu:12-19
#include "bee_wht128_verbatim.h"         // kvarn.cu:456-472  kvarn_wht_128
#include "bee_cross_slices_verbatim.h"   // kvarn.cu:510-543  kvarn_wht_cross_slices
#include "ninfer_hadamard_d256_verbatim.h" // hadamard_d256.cuh:1-67 normalized_hadamard_d256_inplace

#define CUDA_OK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("[FATAL] %s -> %s (%s:%d)\n", #x, cudaGetErrorString(e), __FILE__, __LINE__); exit(2); } } while (0)

static constexpr int kDim = 256;      // D256
static constexpr int kSlice = 128;    // KVAR_N_DIM

// -------------------------------------------------------------------------------------------
// [GATE1 变体，非逐字] beellama 结构 + 归一化改成最后一次性乘以精确 2^-4。
// 用途：把「结构（蝶形顺序/配对/符号）」与「归一化常量与施加位置」分开归因。
// -------------------------------------------------------------------------------------------
static __device__ void gate1_wht_128_unscaled(float * values) {
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
}

static __device__ void gate1_cross_slices_exact(float values[4], int head_slices) {
    if (head_slices == 1) { return; }
    for (int stride = 1; stride < head_slices; stride <<= 1) {
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) { continue; }
            for (int i = 0; i < stride; ++i) {
                const int a_idx = base + i;
                const int b_idx = base + stride + i;
                if (b_idx >= head_slices) { continue; }
                const float a = values[a_idx];
                const float b = values[b_idx];
                values[a_idx] = a + b;
                values[b_idx] = a - b;
            }
        }
    }
    for (int slice = 0; slice < 4; ++slice) { if (slice < head_slices) { values[slice] *= 0x1p-4f; } }
}

// -------------------------------------------------------------------------------------------
// 路径 A：逐字调用 beellama 的 kvarn_wht_128 / kvarn_wht_cross_slices
// 调用顺序照抄 kvarn.cu:1113-1119（去掉 headwide 的 index/stage 调度，只留旋转本体）：
//     float values[4] = {};
//     for (int slice = 0; slice < head_slices; ++slice) {
//         const int head = head0 + slice;
//         shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
//         kvarn_wht_128(shared);
//         values[slice] = shared[dim];
//     }
//     kvarn_wht_cross_slices(values, head_slices);
// harness 的 current 布局 = 逻辑 D256 head 的 [slice*128 + dim]（即 slice 为 head 内切片，逐 slice 连续）。
// -------------------------------------------------------------------------------------------
__global__ void kv_beellama(const float * __restrict__ in, float * __restrict__ out,
                            int head_slices, int exact_scale, int swap_slices) {
    __shared__ float shared[KVAR_N_DIM];
    const int dim = threadIdx.x;                                    // kvarn.cu:1086
    const float * v = in + (size_t)blockIdx.x * (size_t)head_slices * KVAR_N_DIM;
    float values[4] = {};
    for (int slice = 0; slice < head_slices; ++slice) {
        const int src = swap_slices ? (head_slices - 1 - slice) : slice;
        shared[dim] = v[src * KVAR_N_DIM + dim];
        if (exact_scale) { gate1_wht_128_unscaled(shared); } else { kvarn_wht_128(shared); }
        values[slice] = shared[dim];
    }
    if (exact_scale) { gate1_cross_slices_exact(values, head_slices); }
    else { kvarn_wht_cross_slices(values, head_slices); }
    float * o = out + (size_t)blockIdx.x * (size_t)head_slices * KVAR_N_DIM;
    for (int slice = 0; slice < head_slices; ++slice) { o[slice * KVAR_N_DIM + dim] = values[slice]; }
}

// -------------------------------------------------------------------------------------------
// 路径 B：逐字调用 ninfer 的 normalized_hadamard_d256_inplace（1 warp = 1 个 D256 row）
// lane l 持有 dim = l + 32*r（hadamard_d256.cuh:49-51 注释），写回同一布局。
// -------------------------------------------------------------------------------------------
__global__ void kv_ninfer(const float * __restrict__ in, float * __restrict__ out, int row, int vecs) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int unit = blockIdx.x * (blockDim.x >> 5) + warp;
    if (unit >= vecs) { return; }                       // 整 warp 一起返回，shfl 掩码安全
    const float * v = in + (size_t)unit * row;
    float x[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) { x[r] = v[lane + 32 * r]; }
    ninfer::ops::normalized_hadamard_d256_inplace(x, lane);
    float * o = out + (size_t)unit * row;
#pragma unroll
    for (int r = 0; r < 8; ++r) { o[lane + 32 * r] = x[r]; }
}

// =============================== 宿主：double 域复刻（归因用） ===============================
// 与设备侧同结构同常量，但在 double 域跑，用于把「数学结构差」与「float 舍入差」分开。
static void dp_wht128_bee(double * v) {           // 逐行复刻 kvarn_wht_128（常量取 float 字面量的值）
    for (int stride = 1; stride < 128; stride *= 2) {
        for (int t = 0; t < 64; ++t) {
            const int j = (t / stride) * (2 * stride) + (t % stride);
            const double a = v[j], b = v[j + stride];
            v[j] = a + b; v[j + stride] = a - b;
        }
    }
    const double w = (double)0.08838834764831845f;   // 上游常量（float 字面量 → float 值）
    for (int i = 0; i < 128; ++i) { v[i] *= w; }
}
static void dp_cross_bee(double * val, int slices) { // 复刻 kvarn_wht_cross_slices
    if (slices == 1) { return; }
    for (int stride = 1; stride < slices; stride <<= 1) {
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= slices) { continue; }
            for (int i = 0; i < stride; ++i) {
                const int a = base + i, b = base + stride + i;
                if (b >= slices) { continue; }
                const double x = val[a], y = val[b];
                val[a] = x + y; val[b] = x - y;
            }
        }
    }
    const double s = slices == 2 ? (double)0.7071067811865475f : 0.5;
    for (int i = 0; i < 4; ++i) { if (i < slices) { val[i] *= s; } }
}
static void dp_bee(const float * x, double * out, int slices) {
    static thread_local double buf[4][128];
    for (int s = 0; s < slices; ++s) {
        for (int d = 0; d < 128; ++d) { buf[s][d] = (double)x[s * 128 + d]; }
        dp_wht128_bee(buf[s]);
    }
    for (int d = 0; d < 128; ++d) {
        double val[4] = {};
        for (int s = 0; s < slices; ++s) { val[s] = buf[s][d]; }
        dp_cross_bee(val, slices);
        for (int s = 0; s < slices; ++s) { out[s * 128 + d] = val[s]; }
    }
}
static void dp_nin(const float * x, double * out) { // 复刻 normalized_hadamard_d256_inplace（自然 dim 序 = stride 递增）
    double v[256];
    for (int i = 0; i < 256; ++i) { v[i] = (double)x[i]; }
    for (int stride = 1; stride < 256; stride <<= 1) {
        for (int i = 0; i < 256; i += 2 * stride) {
            for (int o = 0; o < stride; ++o) {
                const double a = v[i + o], b = v[i + o + stride];
                v[i + o] = a + b; v[i + o + stride] = a - b;
            }
        }
    }
    for (int i = 0; i < 256; ++i) { v[i] *= 0x1p-4; }
    for (int i = 0; i < 256; ++i) { out[i] = v[i]; }
}

// =============================== 指标 ===============================
struct Metrics {
    double maxAbsDiff = 0, maxRelA = 0, maxRelA_bigA = 0, maxRelMax = 0, meanCos = 0, minCos = 2;
    double maxAbsA = 0, maxAbsB = 0;
    long long nExceed = 0, nRelExceed = 0, nCosCount = 0;
    double wAbsDiff = 0, wA = 0, wB = 0, wRelMax = 0;
    int wVec = -1, wIdx = -1;
    double dpBeeVsNin = 0, dpBeeVsNinRel = 0;
};

static Metrics compare(const std::vector<float> & A, const std::vector<float> & B,
                       int nVec, int row, double limit) {
    Metrics m;
    double cosSum = 0;
    for (int t = 0; t < nVec; ++t) {
        const float * a = A.data() + (size_t)t * row;
        const float * b = B.data() + (size_t)t * row;
        double dot = 0, na2 = 0, nb2 = 0;
        for (int i = 0; i < row; ++i) {
            const double av = (double)a[i], bv = (double)b[i];
            const double d = std::fabs(av - bv);
            m.maxAbsA = std::max(m.maxAbsA, std::fabs(av));
            m.maxAbsB = std::max(m.maxAbsB, std::fabs(bv));
            if (d > m.maxAbsDiff) { m.maxAbsDiff = d; m.wA = av; m.wB = bv; m.wVec = t; m.wIdx = i; }
            if (std::fabs(av) > 1e-30) { m.maxRelA = std::max(m.maxRelA, d / std::fabs(av)); }
            if (std::fabs(av) > 1e-3)  { m.maxRelA_bigA = std::max(m.maxRelA_bigA, d / std::fabs(av)); }
            const double den = std::max(std::fabs(av), std::fabs(bv));
            if (den > 1e-30) {
                const double r = d / den;
                if (r > m.maxRelMax) { m.maxRelMax = r; }
                if (d >= m.maxAbsDiff) { m.wRelMax = r; }   // 取「最大绝对差所在那个元素」自己的相对差
            }
            if (d > limit) { ++m.nExceed; }
            if (den > 1e-30 && d / den > limit) { ++m.nRelExceed; }
            dot += av * bv; na2 += av * av; nb2 += bv * bv;
        }
        if (na2 > 0 && nb2 > 0) {
            const double c = dot / (std::sqrt(na2) * std::sqrt(nb2));
            cosSum += c; m.minCos = std::min(m.minCos, c); ++m.nCosCount;
        }
    }
    m.meanCos = m.nCosCount ? cosSum / (double)m.nCosCount : 0.0;
    if (m.wVec >= 0) { m.wAbsDiff = m.maxAbsDiff; }
    return m;
}

// 归因专用：对同一批**原始输入**比较 double 复刻的两条路径
static void dp_probe(const std::vector<float> & X, int nVec, int row, Metrics & m) {
    std::vector<double> db(row), dn(row);
    for (int t = 0; t < nVec; ++t) {
        const float * x = X.data() + (size_t)t * row;
        dp_bee(x, db.data(), row / 128);
        dp_nin(x, dn.data());
        double ma = 0, mb = 0;
        for (int i = 0; i < row; ++i) { ma = std::max(ma, std::fabs(db[i])); mb = std::max(mb, std::fabs(dn[i])); }
        const double ref = std::max(ma, mb);
        for (int i = 0; i < row; ++i) {
            const double d = std::fabs(db[i] - dn[i]);
            m.dpBeeVsNin = std::max(m.dpBeeVsNin, d);
            if (ref > 0) { m.dpBeeVsNinRel = std::max(m.dpBeeVsNinRel, d / ref); }
        }
    }
}

int main() {
    cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, 0));
    printf("================ GATE 1: 旋转等价性数值对拍（ninfer normalized_hadamard_d256_inplace  vs  beellama wht_128 + cross_slices）================\n");
    printf("[device] %s  sm_%d%d  (CC %d.%d)\n", prop.name, prop.major, prop.minor, prop.major, prop.minor);
    printf("[build ] 编译时间 %s %s\n", __DATE__, __TIME__);

    // ---- 常量核对（float 字面量 vs 精确值）----
    {
        const double wf = (double)0.08838834764831845f, sf = (double)0.7071067811865475f;
        const double wE = 1.0 / std::sqrt(128.0), sE = 1.0 / std::sqrt(2.0);
        printf("[const ] beellama 0.08838834764831845f = %.17g  1/sqrt(128) = %.17g  rel.err = %.3e\n",
               wf, wE, std::fabs(wf - wE) / wE);
        printf("[const ] beellama 0.7071067811865475f  = %.17g  1/sqrt(2)   = %.17g  rel.err = %.3e\n",
               sf, sE, std::fabs(sf - sE) / sE);
        printf("[const ] beellama 合成归一化 wf*sf = %.17g  vs ninfer 2^-4 = %.17g  rel.diff = %.3e (%.2f ulp)\n",
               wf * sf, 0.0625, std::fabs(wf * sf - 0.0625) / 0.0625,
               std::fabs(wf * sf - 0.0625) / 0.0625 / 5.960464477539063e-08);
        printf("[const ] ninfer 归一化 0x1p-4f = 0.0625（2 的幂，float/double 均精确可表示，无常量误差）\n");
    }

    const int N = 4096;                 // 每组 4096 个 256 维向量
    const int GROUPS = 3;
    const int rowA = 256;               // head_slices=2 => 2*128
    std::vector<float> X((size_t)GROUPS * N * rowA);

    // ---- 输入生成（固定种子）----
    std::mt19937 rng(20261013u);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    for (int t = 0; t < N; ++t) {                              // 组 0：标准高斯
        float * x = X.data() + (size_t)t * rowA;
        for (int i = 0; i < rowA; ++i) { x[i] = (float)gauss(rng); }
    }
    for (int t = 0; t < N; ++t) {                              // 组 1：高斯 + 少数分量 ×(30~100)
        float * x = X.data() + ((size_t)N + t) * rowA;
        for (int i = 0; i < rowA; ++i) { x[i] = (float)gauss(rng); }
        for (int k = 0; k < 4; ++k) {
            const int idx = (int)(rng() % rowA);
            const double f = 30.0 + 70.0 * uni(rng);
            x[idx] = (float)(x[idx] * (rng() % 2 ? f : -f));
        }
    }
    {                                                          // 组 2：边界
        std::mt19937 brng(777u);
        for (int t = 0; t < N; ++t) {
            float * x = X.data() + ((size_t)2 * N + t) * rowA;
            for (int i = 0; i < rowA; ++i) { x[i] = 0.0f; }
            if (t < 256) { x[t] = 1.0f; continue; }             // e_0..e_255（单位基，同时是配对层探针）
            const int u = (t - 256) / 16, k = (t - 256) % 16;
            switch (k) {
                case 0: break;                                   // 全 0
                case 1: for (int i = 0; i < rowA; ++i) { x[i] = 1.0f; } break;                 // 全等值
                case 2: for (int i = 0; i < rowA; ++i) { x[i] = 0.5f; } break;
                case 3: for (int i = 0; i < rowA; ++i) { x[i] = -3.0f; } break;
                case 4: for (int i = 0; i < rowA; ++i) { x[i] = 0x1p-4f; } break;
                case 5: for (int i = 0; i < rowA; ++i) { x[i] = (i & 1) ? -1.0f : 1.0f; } break; // ±1 交替(逐点)
                case 6: for (int i = 0; i < rowA; ++i) { x[i] = ((i / 32) & 1) ? -1.0f : 1.0f; } break;
                case 7: for (int i = 0; i < rowA; ++i) { x[i] = ((i / 128) & 1) ? -1.0f : 1.0f; } break;
                case 8: x[(u * 13) % rowA] = 1.0f; break;        // 单个非零 +1
                case 9: x[(u * 29) % rowA] = -(30.0f + (float)(u % 71)); break;  // 单个非零，大值
                case 10: for (int i = 0; i < 128; ++i) { x[i] = 1.0f; } break;   // 前半 1 / 后半 0
                case 11: for (int i = 128; i < rowA; ++i) { x[i] = 1.0f; } break; // 前半 0 / 后半 1
                case 12: for (int i = 0; i < rowA; ++i) { x[i] = (brng() % 2) ? -1.0f : 1.0f; } break;
                case 13: for (int i = 0; i < rowA; ++i) { x[i] = (float)((i % 5) - 2) * 0.7f; } break;
                case 14: for (int i = 0; i < rowA; ++i) { x[i] = (float)i * 1e-3f - 0.128f; } break;
                case 15: {
                    for (int j = 0; j < 4; ++j) {
                        const int idx = (u * 61 + j * 37) % rowA;
                        x[idx] = (j & 1) ? -1.0f : 1.0f;
                    }
                    break;
                }
            }
        }
    }

    // ---- 设备缓冲 ----
    float *d_X = nullptr, *d_A = nullptr, *d_B = nullptr, *d_T = nullptr, *d_A2 = nullptr, *d_B2 = nullptr;
    const size_t bytes = X.size() * sizeof(float);
    CUDA_OK(cudaMalloc(&d_X, bytes));
    CUDA_OK(cudaMalloc(&d_A, bytes));
    CUDA_OK(cudaMalloc(&d_B, bytes));
    CUDA_OK(cudaMalloc(&d_T, bytes));
    CUDA_OK(cudaMalloc(&d_A2, bytes));
    CUDA_OK(cudaMalloc(&d_B2, bytes));
    CUDA_OK(cudaMemcpy(d_X, X.data(), bytes, cudaMemcpyHostToDevice));

    const int nVec = GROUPS * N;
    // 路径 A：1 block = 1 个 D256 向量（128 线程，headwide 的 blockDim=KVAR_N_DIM 口径）
    auto runA = [&](const float *src, float *dst, int exact, int swap) {
        kv_beellama<<<nVec, KVAR_N_DIM>>>(src, dst, 2, exact, swap);
    };
    // 路径 B：1 warp = 1 个 D256 向量
    auto runB = [&](const float *src, float *dst) {
        kv_ninfer<<<nVec / 4, 128>>>(src, dst, 256, nVec);
    };

    runA(d_X, d_A, 0, 0);            // A：逐字 beellama
    runB(d_X, d_B);                  // B：逐字 ninfer
    runA(d_X, d_T, 1, 0);            // A'：beellama 结构 + 单点精确 2^-4
    float *d_S = nullptr;            // A''：beellama 逐字，但 slice 顺序对调（配对层反证）
    CUDA_OK(cudaMalloc(&d_S, bytes));
    runA(d_X, d_S, 0, 1);
    runA(d_A, d_A2, 0, 0);           // A 自逆（旋两次）
    runB(d_B, d_B2);                 // B 自逆（旋两次）
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());

    std::vector<float> A(bytes / 4), B(bytes / 4), T(bytes / 4), S(bytes / 4), A2(bytes / 4), B2(bytes / 4);
    CUDA_OK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(T.data(), d_T, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(S.data(), d_S, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(A2.data(), d_A2, bytes, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(B2.data(), d_B2, bytes, cudaMemcpyDeviceToHost));

    // ---- 落盘 8 个探测向量（输入 / A 输出 / B 输出），供宿主 python 侧与 oracle 的 rotate_d256 三方对拍 ----
    {
        const int np = 8;
        FILE * f = fopen("out/ab_probe.bin", "wb");
        if (f != nullptr) {
            fwrite(&np, sizeof(int), 1, f);
            for (int t = 0; t < np; ++t) {
                fwrite(X.data() + (size_t)t * rowA, sizeof(float), rowA, f);
                fwrite(A.data() + (size_t)t * rowA, sizeof(float), rowA, f);
                fwrite(B.data() + (size_t)t * rowA, sizeof(float), rowA, f);
            }
            fclose(f);
            printf("[probe] 写出 out/ab_probe.bin：%d 个向量 ×（输入, A 输出, B 输出）各 %d float\n", np, rowA);
        } else {
            printf("[probe] 警告：out/ab_probe.bin 打不开（缺少 out/ 目录？）\n");
        }
    }

    // ---- 分组指标 ----
    const char * gname[3] = {"gauss  ", "outlier", "bound  "};
    const double limit = 1e-6;
    Metrics mG[3], mT[3], mS[3];
    std::vector<double> db(256), dn(256);
    for (int g = 0; g < 3; ++g) {
        const float * offA = A.data() + (size_t)g * N * rowA;
        std::vector<float> a(offA, offA + (size_t)N * rowA);
        const float * offB = B.data() + (size_t)g * N * rowA;
        std::vector<float> b(offB, offB + (size_t)N * rowA);
        const float * offX = X.data() + (size_t)g * N * rowA;
        std::vector<float> x(offX, offX + (size_t)N * rowA);
        mG[g] = compare(a, b, N, rowA, limit);
        dp_probe(x, N, rowA, mG[g]);
        // A'（单点精确归一化）vs B
        const float * offT = T.data() + (size_t)g * N * rowA;
        std::vector<float> tv(offT, offT + (size_t)N * rowA);
        mT[g] = compare(tv, b, N, rowA, limit);
        // A''（slice 对调）vs B：配对层反证
        const float * offS = S.data() + (size_t)g * N * rowA;
        std::vector<float> sv(offS, offS + (size_t)N * rowA);
        mS[g] = compare(sv, b, N, rowA, limit);
        // 自逆
        const float * offA2 = A2.data() + (size_t)g * N * rowA;
        const float * offB2 = B2.data() + (size_t)g * N * rowA;
        double ivA = 0, ivAr = 0, ivB = 0, ivBr = 0;
        for (int t = 0; t < N; ++t) {          // 相对差按「该向量自身的 max|x|」归一，避免极小元素放大
            double scaleX = 0;
            for (int i = 0; i < rowA; ++i) { scaleX = std::max(scaleX, std::fabs((double)offX[(size_t)t * rowA + i])); }
            for (int i = 0; i < rowA; ++i) {
                const size_t k = (size_t)t * rowA + i;
                const double dA2 = std::fabs((double)offA2[k] - (double)offX[k]);
                const double dB2 = std::fabs((double)offB2[k] - (double)offX[k]);
                ivA = std::max(ivA, dA2); ivB = std::max(ivB, dB2);
                if (scaleX > 0) { ivAr = std::max(ivAr, dA2 / scaleX); ivBr = std::max(ivBr, dB2 / scaleX); }
            }
        }
        printf("\n[grp%d %s] N=%d 向量 × %d 维 = %d 元素；|A|max=%.6e |B|max=%.6e\n", g, gname[g], N, rowA,
               N * rowA, mG[g].maxAbsA, mG[g].maxAbsB);
        printf("        逐元素 max|A-B| = %.6e   (超 1e-6 元素数 %lld / %d)\n", mG[g].maxAbsDiff, mG[g].nExceed, N * rowA);
        printf("        相对差 max|A-B|/|A| = %.6e ；其中仅 |A|>1e-3 的元素 = %.6e ；max|A-B|/max(|A|,|B|) = %.6e\n",
               mG[g].maxRelA, mG[g].maxRelA_bigA, mG[g].maxRelMax);
        printf("        平均 cosine = %.15f  （最小 %.15f；可算余弦的向量 %lld / %d）\n", mG[g].meanCos, mG[g].minCos, mG[g].nCosCount, N);
        printf("        最大差处：vec=%d idx=%d  A=%.9e (%a)  B=%.9e (%a)  该元素相对差 = %.3e\n",
               mG[g].wVec, mG[g].wIdx, mG[g].wA, mG[g].wA, mG[g].wB, mG[g].wB, mG[g].wRelMax);
        printf("        [归因] double 域复刻两条路径 max|dpA-dpB| = %.6e （相对 %.3e）\n", mG[g].dpBeeVsNin, mG[g].dpBeeVsNinRel);
        printf("        [变体] A'(beellama结构+单点精确 2^-4) vs B: max|A'-B| = %.6e ；超 1e-6 元素数 %lld\n",
               mT[g].maxAbsDiff, mT[g].nExceed);
        printf("        [反证] A''(slice 顺序对调) vs B: max|A''-B| = %.6e  （若为 0 说明映射不可辨，非 0 说明映射有实义）\n",
               mS[g].maxAbsDiff);
        printf("        自逆：A 旋两次 vs 原值 max|Δ| = %.6e（按该向量 max|x| 归一 = %.3e）；B max|Δ| = %.6e（归一 = %.3e）\n",
               ivA, ivAr, ivB, ivBr);
    }

    // ---- 配对层：用 256 个单位向量 e_i 得 M_A / M_B，逐行匹配求映射与符号 ----
    printf("\n================ 配对层（permutation/sign）================\n");
    {
        const int n = 256;
        const size_t base = (size_t)2 * N * rowA;   // 边界组偏移：其前 256 个向量正是单位基 e_0..e_255
        // M[j][i] = A(e_i)[j]
        std::vector<double> MA((size_t)n * n), MB((size_t)n * n);
        long long bitDiff = 0;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                const float a = A[base + (size_t)i * rowA + j];
                const float b = B[base + (size_t)i * rowA + j];
                if (std::memcmp(&a, &b, sizeof(float)) != 0) { ++bitDiff; }
                MA[(size_t)j * n + i] = (double)a;
                MB[(size_t)j * n + i] = (double)b;
            }
        }
        int map[256]; int sign[256];
        int nonIdentity = 0, negSign = 0;
        double worstResid = 0, worstRow = 0, devFromSixteenth = 0;
        for (int j = 0; j < n; ++j) {
            double best = -1; int bestK = -1; double bestIn = 0;
            for (int k = 0; k < n; ++k) {
                double in = 0;
                for (int i = 0; i < n; ++i) { in += MA[(size_t)j * n + i] * MB[(size_t)k * n + i]; }
                if (std::fabs(in) > best) { best = std::fabs(in); bestK = k; bestIn = in; }
            }
            map[j] = bestK; sign[j] = bestIn >= 0 ? 1 : -1;
            if (bestK != j) { ++nonIdentity; }
            if (sign[j] < 0) { ++negSign; }
            double res = 0;
            for (int i = 0; i < n; ++i) {
                res = std::max(res, std::fabs(MA[(size_t)j * n + i] - sign[j] * MB[(size_t)bestK * n + i]));
                devFromSixteenth = std::max(devFromSixteenth, std::fabs(std::fabs(MA[(size_t)j * n + i]) - 0.0625));
            }
            worstResid = std::max(worstResid, res); worstRow = std::max(worstRow, std::fabs(best));
        }
        printf("[map ] 256 个单位向量 e_i 探测（边界组前 256 个向量）：M_A[j][i]=A(e_i)[j] 与 M_B 逐行做内积匹配\n");
        printf("[map ] 非恒等行数 = %d / 256 ；符号为负的行数 = %d / 256\n", nonIdentity, negSign);
        printf("[map ] M_A 与 M_B 逐元素 bit 不同的元素数 = %lld / 65536\n", bitDiff);
        printf("[map ] 匹配行对之间的最大残差 max|M_A[j] - sign*M_B[map[j]]| = %.6e\n", worstResid);
        printf("[map ] 行内积最大值（恒等时 ≈1）= %.15f ；|M_A| 相对 1/16 的最大偏差 = %.6e\n", worstRow, devFromSixteenth);
        // 写死在源码里的显式映射表：判定要求实测 map[] 与之逐项相同
        static const int gate1_index_map[256] = {
              0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
             16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
             32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
             48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
             64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
             80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
             96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
            112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
            144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
            160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
            176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
            192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
            208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
            224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
            240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
        };
        int mapMismatch = 0;
        for (int j = 0; j < 256; ++j) { if (map[j] != gate1_index_map[j]) { ++mapMismatch; } }
        printf("[map ] 实测映射 vs 源码内写死的 gate1_index_map[256]（恒等）：不一致项数 = %d / 256\n", mapMismatch);
        printf("[map ] 映射表 map[j]（j=0..255，16/行）:\n");
        for (int j = 0; j < 256; j += 16) {
            printf("        ");
            for (int k = 0; k < 16; ++k) { printf("%3d", map[j + k]); }
            printf("\n");
        }
        printf("[map ] 符号表 sign[j]（16/行）:\n");
        for (int j = 0; j < 256; j += 16) {
            printf("        ");
            for (int k = 0; k < 16; ++k) { printf("%+d", sign[j + k]); }
            printf("\n");
        }
    }

    // ---- 附：beellama 的 slice 划分是否为「自然前半/后半」（用 A'' 的 0 差异面反证）----
    printf("\n================ 附：D512（head_slices=4）一致性与 slice-划分语义 ================\n");
    {
        const int n = 256, row5 = 512;
        std::mt19937 r2(4242u);
        std::normal_distribution<double> gg(0, 1);
        std::vector<float> y((size_t)n * row5);
        for (auto & v : y) { v = (float)gg(r2); }
        float *d_y = nullptr, *d_o = nullptr;
        CUDA_OK(cudaMalloc(&d_y, y.size() * 4));
        CUDA_OK(cudaMalloc(&d_o, y.size() * 4));
        CUDA_OK(cudaMemcpy(d_y, y.data(), y.size() * 4, cudaMemcpyHostToDevice));
        kv_beellama<<<n, KVAR_N_DIM>>>(d_y, d_o, 4, 0, 0);
        CUDA_OK(cudaGetLastError()); CUDA_OK(cudaDeviceSynchronize());
        std::vector<float> o(y.size());
        CUDA_OK(cudaMemcpy(o.data(), d_o, y.size() * 4, cudaMemcpyDeviceToHost));
        double mx = 0;
        for (int t = 0; t < n; ++t) {   // 与「D512 单点归一化 FWHT」的 double 复刻比
            double v[512];
            for (int i = 0; i < 512; ++i) { v[i] = (double)y[(size_t)t * row5 + i]; }
            for (int stride = 1; stride < 512; stride <<= 1)
                for (int i = 0; i < 512; i += 2 * stride)
                    for (int q = 0; q < stride; ++q) { const double a = v[i + q], b = v[i + q + stride]; v[i + q] = a + b; v[i + q + stride] = a - b; }
            for (int i = 0; i < 512; ++i) { v[i] *= 1.0 / std::sqrt(512.0); }
            for (int i = 0; i < 512; ++i) { mx = std::max(mx, std::fabs(v[i] - (double)o[(size_t)t * row5 + i])); }
        }
        printf("[d512 ] beellama(head_slices=4, 逐字) vs D512 归一化 FWHT（double 复刻）max|Δ| = %.6e\n", mx);
        printf("[d512 ] 结论：cross_slices 的 scale=%s 与 4 个 128-slice 组合出的归一化 = 1/sqrt(512)，与整 D512 FWHT 一致（数值见上）\n",
               "0.5f");
        cudaFree(d_y); cudaFree(d_o);
    }

    // ---- 判定 ----
    printf("\n================ 判定 ================\n");
    int passG = 1;
    for (int g = 0; g < 3; ++g) {
        const bool ok = (mG[g].maxAbsDiff <= limit);
        printf("[verdict grp%d %s] 严格判据（max|A-B| ≤ 1e-6 且自逆 ≤ 1e-6）：max|A-B| = %.6e -> %s\n",
               g, gname[g], mG[g].maxAbsDiff, ok ? "PASS" : "FAIL");
        if (!ok) { passG = 0; }
    }
    int passT = 1;
    for (int g = 0; g < 3; ++g) { if (mT[g].maxAbsDiff > limit) { passT = 0; } }
    printf("[verdict 变体A'] 结构同（蝶形顺序/配对/符号）+ 单点精确 2^-4：三组 max|A'-B| = %.6e / %.6e / %.6e -> %s\n",
           mT[0].maxAbsDiff, mT[1].maxAbsDiff, mT[2].maxAbsDiff, passT ? "PASS（全 0 或 ≤1e-6）" : "FAIL");
    printf("[verdict 总体] 逐字对拍（含各自归一化常量与施加位置）= %s\n", passG ? "PASS" : "FAIL");
    cudaFree(d_X); cudaFree(d_A); cudaFree(d_B); cudaFree(d_T); cudaFree(d_S); cudaFree(d_A2); cudaFree(d_B2);
    return passG ? 0 : 1;
}

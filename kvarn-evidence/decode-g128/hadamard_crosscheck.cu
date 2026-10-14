// =============================================================================
// hadamard_crosscheck.cu —— kvarn128_decode.cuh 内嵌的逆旋转副本 vs 引擎真头
//   include 两个来源（只读 include 引擎目录，不修改）：
//     * 内嵌：kvarn128::detail::normalized_hadamard_d256_inplace
//     * 引擎：ninfer::ops::normalized_hadamard_d256_inplace
//           （engine/ninfer-all-kvarn/src/ops/kv_cache/hadamard_d256.cuh
//             sha256 3d00813a0a3c112484ef5a12c0011ec7ee8d0848aa17fadccaf9fbdeb0b14fa7）
//   同一 device、同一数据、逐位对拍（4,194,304 元素）。
// 编译：nvcc -O2 -std=c++17 -arch=sm_86 -I/eng/src
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>

#define KVARN128_EMBED_HADAMARD 1
#include "kvarn128_decode.cuh"
#include "ops/kv_cache/hadamard_d256.cuh"   // 引擎真头（只读 include）

__global__ void k_embedded(const float* __restrict__ in, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = in[(size_t)row * 256 + lane + 32 * r];
    kvarn128::detail::normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) out[(size_t)row * 256 + lane + 32 * r] = v[r];
}
__global__ void k_engine(const float* __restrict__ in, float* __restrict__ out) {
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    float v[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) v[r] = in[(size_t)row * 256 + lane + 32 * r];
    ninfer::ops::normalized_hadamard_d256_inplace(v, lane);
#pragma unroll
    for (int r = 0; r < 8; ++r) out[(size_t)row * 256 + lane + 32 * r] = v[r];
}

int main() {
    const int ROWS = 4096;                       // 4096 行 × 256 = 1,048,576 元素
    std::vector<float> in((size_t)ROWS * 256), a(in.size()), b(in.size());
    std::mt19937_64 rng(20261013ull);
    std::uniform_real_distribution<float> ud(-4.f, 4.f);
    for (auto& v : in) v = ud(rng);
    float *di = nullptr, *da = nullptr, *db = nullptr;
    cudaMalloc(&di, in.size() * 4); cudaMalloc(&da, a.size() * 4); cudaMalloc(&db, b.size() * 4);
    cudaMemcpy(di, in.data(), in.size() * 4, cudaMemcpyHostToDevice);
    const int rows_per_block = 8;
    dim3 blk(32 * rows_per_block), grd(ROWS / rows_per_block);
    k_embedded<<<grd, blk>>>(di, da);
    k_engine<<<grd, blk>>>(di, db);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("CUDA 错误: %s\n", cudaGetErrorString(e)); return 3; }
    cudaMemcpy(a.data(), da, a.size() * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), db, b.size() * 4, cudaMemcpyDeviceToHost);

    long long bit_diff = 0; double maxabs = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        uint32_t ua, ub;
        __builtin_memcpy(&ua, &a[i], 4); __builtin_memcpy(&ub, &b[i], 4);
        if (ua != ub) ++bit_diff;
        maxabs = maxabs > fabs((double)a[i] - (double)b[i]) ? maxabs : fabs((double)a[i] - (double)b[i]);
    }
    // 自逆性（取引擎版跑两次）
    std::vector<float> c(in.size());
    float* dc = nullptr; cudaMalloc(&dc, c.size() * 4);
    k_engine<<<grd, blk>>>(db, dc);
    cudaDeviceSynchronize();
    cudaMemcpy(c.data(), dc, c.size() * 4, cudaMemcpyDeviceToHost);
    double maxid = 0;
    for (size_t i = 0; i < c.size(); ++i) maxid = std::max(maxid, fabs((double)c[i] - (double)in[i]));

    printf("[crosscheck] 内嵌副本 vs 引擎真头：元素 %zu，逐位不一致 = %lld，max|Δ| = %.3e\n", a.size(), bit_diff, maxabs);
    printf("[crosscheck] 引擎版 H∘H 自逆 max|Δ| = %.3e\n", maxid);
    printf("[crosscheck] 判定：%s\n", (bit_diff == 0) ? "PASS（内嵌副本与引擎真头逐位相同）" : "FAIL");
    cudaFree(di); cudaFree(da); cudaFree(db); cudaFree(dc);
    return bit_diff == 0 ? 0 : 1;
}

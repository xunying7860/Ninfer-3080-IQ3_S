// kvarn_store_gate.cu — GATE 2 附件：存储路径的【旋转 + F16 stage】段
// ---------------------------------------------------------------------------
// 覆盖 kvarn.cu:1059 kvarn_store_kernel_headwide 里除「record 封口调度」以外的核心段，
// 逐行照抄自 ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu（commit 58a1629…）：
//   kvarn.cu:456-472  kvarn_wht_128
//   kvarn.cu:510-543  kvarn_wht_cross_slices
//   kvarn.cu:1113-1131
//        float values[4] = {};
//        for (int slice = 0; slice < head_slices; ++slice) {
//            const int head = head0 + slice;
//            shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
//            kvarn_wht_128(shared);
//            values[slice] = shared[dim];
//        }
//        kvarn_wht_cross_slices(values, head_slices);
//        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
//        for (int slice = 0; slice < head_slices; ++slice) {
//            const int head = head0 + slice;
//            stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + dim] =
//                __float2half_rn(values[slice]);
//        }
//   去掉的只有：indices/stage_slot/flush 的调度（与本 gate 的数值无关）。
// 对照物：同一 bit 的蝶形在宿主机上逐行转写（wht_128 = 纯 add/sub + 一次 mul，
//         cross_slices = 纯 add/sub + 一次 mul）=> 两条路径应【逐字节】相同。
// ===========================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CUDA_CHECK(x) do { cudaError_t _e = (x); if (_e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); exit(3); } } while (0)

static constexpr int KVAR_N_DIM = 128;

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

// ---- kvarn.cu:510-543 ----
static __device__ void kvarn_wht_cross_slices(float values[4], int head_slices) {
    if (head_slices == 1) { return; }
    for (int stride = 1; stride < head_slices; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) { continue; }
#pragma unroll
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
    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        if (slice < head_slices) { values[slice] *= scale; }
    }
}

// ---- kvarn.cu:1113-1131 的旋转 + stage 写（仅去掉 record 调度）----
__global__ void kvarn_store_rotate_stage_kernel(
        const float * __restrict__ current,
        half * __restrict__ stage,
        int n_tokens,
        int n_heads,
        int head_slices) {
    extern __shared__ float shared[];
    const int head0 = blockIdx.x * head_slices;
    if (head0 + head_slices > n_heads) { return; }
    const int dim = threadIdx.x;
    const int stage_base = 0;

    for (int token = 0; token < n_tokens; ++token) {
        const int pos = token;                       // 本 gate 用 pos=token 的直接映射
        const int stage_slot = 0;
        float values[4] = {};
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
            kvarn_wht_128(shared);
            values[slice] = shared[dim];
        }
        kvarn_wht_cross_slices(values, head_slices);
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + dim] =
                __float2half_rn(values[slice]);
        }
        __syncthreads();
    }
}

// ---- 宿主机转写（同一 bit 的蝶形，顺序逐一对应）----
static void host_wht_128(float * values) {
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        for (int t = 0; t < 64; ++t) {
            const int j = (t / stride) * (2 * stride) + (t % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
    }
    for (int i = 0; i < KVAR_N_DIM; ++i) { values[i] *= 0.08838834764831845f; }
}
static void host_cross_slices(float values[4], int head_slices) {
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
    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
    for (int slice = 0; slice < 4; ++slice) { if (slice < head_slices) values[slice] *= scale; }
}

int main() {
    const int n_tokens = 128, head_slices = 2, n_heads = 4;   // 2 个逻辑 D256 head
    const int n_logical = n_heads / head_slices;

    int dev = 0; CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{}; CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int smpbo = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&smpbo, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

    printf("================ GATE2 附件：存储路径（旋转 + F16 stage）================\n");
    printf("[device] %s sm_%d%d ; smem/block opt-in = %d B\n", prop.name, prop.major, prop.minor, smpbo);
    printf("[sel   ] kvarn.cu:2536 选 headwide 的条件：head_slices(%d) > 1 && smpbo(%d) >= KVAR_N_SHARED_BYTES(69704) => %s\n",
           head_slices, smpbo, (head_slices > 1 && smpbo >= 69704) ? "走 headwide（每 block 一个 D256 head，block 内跨 2 slice）" : "不走");

    std::vector<float> cur(size_t(n_tokens) * n_heads * KVAR_N_DIM);
    std::mt19937_64 rng(20261013ull);
    std::normal_distribution<float> gauss(0.f, 1.f);
    for (auto & v : cur) v = gauss(rng);
    for (int t = 0; t < n_tokens; ++t) {                       // 掺几个离群值/边界值
        cur[(size_t(t) * n_heads + 0) * KVAR_N_DIM + 0] = 0.f;
        cur[(size_t(t) * n_heads + 1) * KVAR_N_DIM + 7] *= 1000.f;
        cur[(size_t(t) * n_heads + 2) * KVAR_N_DIM + 63] = (t & 1) ? 1.f : -1.f;
    }

    float * d_cur = nullptr; half * d_stage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_cur, sizeof(float) * cur.size()));
    CUDA_CHECK(cudaMalloc(&d_stage, sizeof(half) * size_t(n_tokens) * n_heads * KVAR_N_DIM));
    CUDA_CHECK(cudaMemcpy(d_cur, cur.data(), sizeof(float) * cur.size(), cudaMemcpyHostToDevice));
    kvarn_store_rotate_stage_kernel<<<n_logical, KVAR_N_DIM, sizeof(float) * KVAR_N_DIM>>>(d_cur, d_stage, n_tokens, n_heads, head_slices);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<half> got(size_t(n_tokens) * n_heads * KVAR_N_DIM);
    CUDA_CHECK(cudaMemcpy(got.data(), d_stage, sizeof(half) * got.size(), cudaMemcpyDeviceToHost));

    std::vector<half> ref(got.size());
    for (int h0 = 0; h0 < n_heads; h0 += head_slices) {
        for (int token = 0; token < n_tokens; ++token) {
            float rotated[KVAR_N_DIM][4];
            for (int slice = 0; slice < head_slices; ++slice) {
                float buf[KVAR_N_DIM];
                for (int q = 0; q < KVAR_N_DIM; ++q)
                    buf[q] = cur[((size_t)token * n_heads + (h0 + slice)) * KVAR_N_DIM + q];
                host_wht_128(buf);
                for (int d = 0; d < KVAR_N_DIM; ++d) {
                    rotated[d][slice] = buf[d];       // 线程 dim 读回 shared[dim]
                }
            }
            for (int d = 0; d < KVAR_N_DIM; ++d) {
                float v4[4] = { rotated[d][0], rotated[d][1], 0.f, 0.f };
                host_cross_slices(v4, head_slices);
                for (int slice = 0; slice < head_slices; ++slice) {
                    ref[((size_t)token * n_heads + (h0 + slice)) * KVAR_N_DIM + d] = __float2half_rn(v4[slice]);
                }
            }
        }
    }

    int nd = 0, first = -1;
    const uint8_t * a = (const uint8_t *) got.data();
    const uint8_t * b = (const uint8_t *) ref.data();
    for (size_t i = 0; i < got.size() * 2; ++i) {
        if (a[i] != b[i]) { if (first < 0) first = (int)i; ++nd; }
    }
    printf("[stage ] tokens=%d heads=%d dim=%d slices=%d ; stage 元素=%zu\n",
           n_tokens, n_heads, KVAR_N_DIM, head_slices, got.size());
    printf("[stage ] CUDA 旋转+F16 stage vs 宿主转写：差异字节 = %d / %zu  首差异@%d\n",
           nd, got.size() * 2, first);
    printf("判定：%s\n", nd == 0 ? "PASS（存储路径旋转 + F16 stage 逐字节一致）" : "FAIL");
    cudaFree(d_cur); cudaFree(d_stage);
    return nd == 0 ? 0 : 1;
}

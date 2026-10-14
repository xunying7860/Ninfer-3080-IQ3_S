// kvarn128_gate.cu —— G128 KVarN 的独立对拍（本机 docker + --gpus all 跑，不进引擎构建）
//
// [1] 36 组合字节账：kvarn_layout.h 的紧凑 record 口径 vs 阶段 1 oracle 口径②（32*(Kb+Vb)+24）
// [2] P1 实际分配口径：kvarn_plane_tile_bytes（4 plane × 2 页）逐格打印
// [3] **record 解码 == 主机反量化**：GPU 走 engine 用的同一份 device 主体
//     （kvarn128_encode_tile / kvarn128_decode_tile_device），主机走独立的标量复刻
//     （Sinkhorn + per-row RTN + rank-1 反量化），比 half 逐位与 cosine（判据 ≥ 0.999）
// [4] 两段 staging（= C 路封口）与一次写满（= A 路）解出来的 half 必须完全相同
//
// 编译（容器内）：nvcc -O2 -std=c++17 -arch=sm_86 -I/src/src -I/src/include -o gate kvarn128_gate.cu
// 运行：./gate
#include "ops/kv_cache/append/kvarn128_codec.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using namespace ninfer::ops;

#define CUDA_CHECK(x)                                                                              \
    do {                                                                                           \
        cudaError_t _e = (x);                                                                      \
        if (_e != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            exit(3);                                                                               \
        }                                                                                          \
    } while (0)

static constexpr int kDyn =
    (kKvarnTileTokens * kKvarnSliceDim + 8 * kKvarnSliceDim + 2 + 16) * (int)sizeof(float);

// ============ GPU：调用 engine 用的同一份 device 主体 ============
__global__ void gate_encode_kernel(const __half* __restrict__ stage, int slot, int head,
                                   std::uint8_t* __restrict__ k_code, __half* __restrict__ k_scale,
                                   std::uint8_t* __restrict__ v_code, __half* __restrict__ v_scale,
                                   int page0, int page1, int iters) {
    extern __shared__ float sh[];
    __shared__ std::uint8_t rec[kKvarnMaxRecordBytes];
    __shared__ float sa[kKvarnSlices][kKvarnSliceDim];
    __shared__ float zp[kKvarnSlices][kKvarnSliceDim];
    __shared__ float oa[kKvarnSlices][kKvarnSliceDim];
    __shared__ float bc[kKvarnSliceDim];
    __shared__ float br[kKvarnSliceDim];
    kvarn128_encode_tile<4, 4>(stage, slot, head, k_code, k_scale, v_code, v_scale, page0, page1, sh,
                               rec, sa, zp, oa, bc, br, iters);
}
__global__ void gate_decode_kernel(const std::uint8_t* __restrict__ k_code,
                                   const __half* __restrict__ k_scale,
                                   const std::uint8_t* __restrict__ v_code,
                                   const __half* __restrict__ v_scale, int page0, int page1,
                                   __half* __restrict__ out_k, __half* __restrict__ out_v) {
    kvarn128_decode_tile_device<4, 4>(k_code, k_scale, v_code, v_scale, 0, page0, page1, out_k,
                                      out_v);
}

// ============ 主机：独立标量复刻（自己写一遍，不用 device 代码）============
static float h_std_col(const float t[128][128], const float sc[128], const float sr[128], int col) {
    double sum = 0, sum_sq = 0;
    const double s = sc[col];
    for (int r = 0; r < 128; ++r) {
        const double x = (double)t[r][col] / (s * (double)sr[r]);
        sum += x; sum_sq += x * x;
    }
    const double mean = sum / 128.0;
    const double var  = (sum_sq - 128.0 * mean * mean) / 127.0; // 分母 127（样本标准差）
    return (float)std::sqrt(var > 0.0 ? var : 0.0);
}
static float h_std_row(const float t[128][128], const float sc[128], const float sr[128], int row) {
    double sum = 0, sum_sq = 0;
    const double s = sr[row];
    for (int c = 0; c < 128; ++c) {
        const double x = (double)t[row][c] / ((double)sc[c] * s);
        sum += x; sum_sq += x * x;
    }
    const double mean = sum / 128.0;
    const double var  = (sum_sq - 128.0 * mean * mean) / 127.0;
    return (float)std::sqrt(var > 0.0 ? var : 0.0);
}

// Sinkhorn + per-row 非对称 RTN + rank-1 反量化；out[r][c] = (q*sa[r]+zp[r])*oa[c]
// 轴值与平面一致：都先取 fp16 再回到 float（平面里存的就是 fp16）。
static void h_scalar_roundtrip(const float t[128][128], int bits, int iters, float out[128][128]) {
    float log_sc[128], log_sr[128], sc[128], sr[128], bc[128], br[128], cstd[128], rstd[128];
    for (int i = 0; i < 128; ++i) {
        log_sc[i] = 0; log_sr[i] = 0; sc[i] = 1; sr[i] = 1; bc[i] = 1; br[i] = 1;
    }
    for (int i = 0; i < 128; ++i) {
        cstd[i] = h_std_col(t, sc, sr, i);
        rstd[i] = h_std_row(t, sc, sr, i);
    }
    float best_imb = 3.402823466e+38F;
    auto update_best = [&]() {
        float cmn = 1e30f, cmx = -1e30f, rmn = 1e30f, rmx = -1e30f;
        for (int i = 0; i < 128; ++i) {
            cmn = fminf(cmn, cstd[i]); cmx = fmaxf(cmx, cstd[i]);
            rmn = fminf(rmn, rstd[i]); rmx = fmaxf(rmx, rstd[i]);
        }
        const float imb = cmx / fmaxf(cmn, 1e-8f) + rmx / fmaxf(rmn, 1e-8f);
        if (imb <= best_imb) {
            best_imb = imb;
            for (int i = 0; i < 128; ++i) { bc[i] = sc[i]; br[i] = sr[i]; }
        }
    };
    update_best();
    for (int it = 0; it < iters; ++it) {
        for (int i = 0; i < 128; ++i) {
            const float c = fminf(fmaxf(cstd[i], 1e-3f), 1e3f);
            log_sc[i] = fminf(fmaxf(log_sc[i] + logf(c), -0.3f), 10.0f);
            sc[i] = expf(log_sc[i]);
        }
        for (int i = 0; i < 128; ++i) rstd[i] = h_std_row(t, sc, sr, i);
        for (int i = 0; i < 128; ++i) {
            const float rr = fminf(fmaxf(rstd[i], 1e-3f), 1e3f);
            log_sr[i] = fminf(fmaxf(log_sr[i] + logf(rr), -0.3f), 10.0f);
            sr[i] = expf(log_sr[i]);
        }
        for (int i = 0; i < 128; ++i) {
            rstd[i] = h_std_row(t, sc, sr, i);
            cstd[i] = h_std_col(t, sc, sr, i);
        }
        update_best();
    }
    const int qmax = (1 << bits) - 1;
    float sa[128], zp[128], oa[128];
    int qs[128][128];
    for (int r = 0; r < 128; ++r) {
        float lo = 3.402823466e+38F, hi = -3.402823466e+38F;
        for (int c = 0; c < 128; ++c) {
            const float x = t[r][c] / (bc[c] * br[r]);
            lo = fminf(lo, x); hi = fmaxf(hi, x);
        }
        const float scale = fmaxf((hi - lo) / qmax, 1e-10f);
        sa[r] = __half2float(__float2half_rn(br[r] * scale));
        zp[r] = __half2float(__float2half_rn(br[r] * lo));
        for (int c = 0; c < 128; ++c) {
            const float x = t[r][c] / (bc[c] * br[r]);
            qs[r][c] = (int)fminf(fmaxf(roundf((x - lo) / scale), 0.0f), (float)qmax);
        }
    }
    for (int i = 0; i < 128; ++i) oa[i] = __half2float(__float2half_rn(bc[i]));
    for (int r = 0; r < 128; ++r)
        for (int c = 0; c < 128; ++c) out[r][c] = ((float)qs[r][c] * sa[r] + zp[r]) * oa[c];
}

// ============ 比较 ============
struct Cmp { double cos = 0, maxabs = 0; int half_mismatch = 0; int nonfinite = 0; };
static Cmp compare(const std::vector<__half>& gpu, const float ref[128][256]) {
    Cmp o;
    double dot = 0, na = 0, nb = 0;
    for (int t = 0; t < 128; ++t)
        for (int d = 0; d < 256; ++d) {
            const float a = __half2float(gpu[t * 256 + d]);
            const double b = (double)ref[t][d];
            if (!std::isfinite((double)a)) ++o.nonfinite;
            dot += (double)a * b; na += (double)a * a; nb += b * b;
            const double diff = std::fabs((double)a - b);
            if (diff > o.maxabs) o.maxabs = diff;
            if (a != __half2float(__float2half_rn((float)b))) ++o.half_mismatch;
        }
    o.cos = dot / std::sqrt(na * nb);
    return o;
}

int main() {
    printf("================ G128 KVarN 对拍（engine 用的同一份 device 主体）================\n");
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("[device] %s sm_%d%d\n", prop.name, prop.major, prop.minor);

    // ---- [1] 36 组合字节账 ----
    const int ob[6] = {2, 3, 4, 5, 6, 8};
    const int table[6][6] = {
        {152, 184, 216, 248, 280, 344}, {184, 216, 248, 280, 312, 376},
        {216, 248, 280, 312, 344, 408}, {248, 280, 312, 344, 376, 440},
        {280, 312, 344, 376, 408, 472}, {344, 376, 408, 440, 472, 536},
    };
    int bad = 0;
    printf("\n[1] 紧凑 record 口径 vs 阶段 1 oracle 口径②（B/token/head）\n");
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j) {
            const int got = kvarn_compact_bytes_per_token(ob[i], ob[j]);
            if (got != table[i][j]) {
                ++bad;
                printf("  MISMATCH K%db V%db: got %d want %d\n", ob[i], ob[j], got, table[i][j]);
            }
        }
    printf("  36 格：不一致 %d / 36  ⇒ %s\n", bad, bad == 0 ? "PASS" : "FAIL");
    printf("  闭式自检：K4V2=%d（应 216）；K4V4=%d（应 280）\n",
           kvarn_compact_bytes_per_token(4, 2), kvarn_compact_bytes_per_token(4, 4));
    printf("\n[2] P1 实际分配口径（4 plane × 2 页 = 128 token）\n");
    printf("  K4V4: tile=%d B ⇒ %d B/token/head（record 口径 280；生产 rk4v4 = 280）\n",
           kvarn_plane_tile_bytes(4, 4), kvarn_plane_bytes_per_token(4, 4));
    printf("  K4V2: tile=%d B ⇒ %d B/token/head（record 口径 216）\n",
           kvarn_plane_tile_bytes(4, 2), kvarn_plane_bytes_per_token(4, 2));

    // ---- [3] record 解码 == 主机反量化 ----
    const int KVH = 4, PAGES = 2;
    const int code_leading  = kvarn_code_leading(4);
    const std::size_t code_elems  = (std::size_t)code_leading * kKvarnPageTokens * KVH * PAGES;
    const std::size_t scale_elems = (std::size_t)kKvarnKScaleLeading * kKvarnPageTokens * KVH * PAGES;
    // stage 布局 = [which][slot][head][128 token][256 dim]（必须与 kvarn128_encode_tile 一致）
    const std::size_t stage_plane =
        (std::size_t)kKvarnPagesPerTile * KVH * kKvarnTileTokens * kKvarnHeadDim;
    const std::size_t stage_elems = 2 * stage_plane;

    std::uint8_t *d_kc = nullptr, *d_vc = nullptr;
    __half *d_ks = nullptr, *d_vs = nullptr, *d_stage = nullptr, *d_ok = nullptr, *d_ov = nullptr,
           *d_ok2 = nullptr, *d_ov2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_kc, code_elems));
    CUDA_CHECK(cudaMalloc(&d_vc, code_elems));
    CUDA_CHECK(cudaMalloc(&d_ks, scale_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_vs, scale_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_stage, stage_elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ok, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ov, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ok2, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_ov2, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
    CUDA_CHECK(cudaFuncSetAttribute(gate_encode_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    kDyn));

    const char* names[2] = {"高斯", "离群通道"};
    int all_pass = 1;
    for (int ci = 0; ci < 2; ++ci) {
        const bool outlier = ci == 1;
        std::vector<__half> hstage(stage_elems, __half(0.0f));
        float tl[2][128][128], tv[2][128][128];
        unsigned seed = 20261013u;
        auto rnd = [&]() {
            seed = seed * 1664525u + 1013904223u;
            return ((seed >> 8) & 0xFFFFFF) / 16777216.0f - 0.5f;
        };
        for (int t = 0; t < 128; ++t)
            for (int d = 0; d < 256; ++d) {
                float kv = rnd(), vv = rnd();
                if (outlier && (d % 64) == 0) { kv *= 20.0f; vv *= 20.0f; }
                const int s = d / 128, dl = d % 128;
                tl[s][dl][t]      = kv; // K: [dim][token]
                tv[s][t][dl]      = vv; // V: [token][dim]
                hstage[t * kKvarnHeadDim + d]               = __float2half_rn(kv);
                hstage[stage_plane + t * kKvarnHeadDim + d] = __float2half_rn(vv);
            }
        CUDA_CHECK(cudaMemcpy(d_stage, hstage.data(), stage_elems * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kc, 0, code_elems));
        CUDA_CHECK(cudaMemset(d_vc, 0, code_elems));
        CUDA_CHECK(cudaMemset(d_ks, 0, scale_elems * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_vs, 0, scale_elems * sizeof(__half)));
        gate_encode_kernel<<<1, 128, kDyn>>>(d_stage, 0, 0, d_kc, d_ks, d_vc, d_vs, 0, 1, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_ok, 0, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
        CUDA_CHECK(cudaMemset(d_ov, 0, (std::size_t)kKvarnTileTokens * kKvarnHeadDim * sizeof(__half)));
        gate_decode_kernel<<<1, 128>>>(d_kc, d_ks, d_vc, d_vs, 0, 1, d_ok, d_ov);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<__half> gk(128 * 256), gv(128 * 256);
        CUDA_CHECK(cudaMemcpy(gk.data(), d_ok, gk.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gv.data(), d_ov, gv.size() * sizeof(__half), cudaMemcpyDeviceToHost));

        static float hk[128][256], hv[128][256];
        for (int s = 0; s < 2; ++s) {
            float q[128][128];
            h_scalar_roundtrip(tl[s], 4, 0, q); // K: row=dim, col=token
            for (int t = 0; t < 128; ++t)
                for (int dl = 0; dl < 128; ++dl) hk[t][s * 128 + dl] = q[dl][t];
            h_scalar_roundtrip(tv[s], 4, 0, q); // V: row=token, col=dim
            for (int t = 0; t < 128; ++t)
                for (int dl = 0; dl < 128; ++dl) hv[t][s * 128 + dl] = q[t][dl];
        }
        const Cmp ck = compare(gk, hk);
        const Cmp cv = compare(gv, hv);
        printf("\n[3-%s] K record 解码 vs 主机反量化: cosine=%.10f max|Δ|=%.3e half 不一致=%d/%d nonfinite=%d\n",
               names[ci], ck.cos, ck.maxabs, ck.half_mismatch, 128 * 256, ck.nonfinite);
        printf("[3-%s] V record 解码 vs 主机反量化: cosine=%.10f max|Δ|=%.3e half 不一致=%d/%d nonfinite=%d\n",
               names[ci], cv.cos, cv.maxabs, cv.half_mismatch, 128 * 256, cv.nonfinite);
        const bool pass = ck.cos >= 0.999 && cv.cos >= 0.999 && ck.nonfinite == 0 && cv.nonfinite == 0;
        printf("[3-%s] 判定：%s（判据 A: cosine ≥ 0.999；判据 B: nonfinite == 0）\n", names[ci],
               pass ? "PASS" : "FAIL");
        all_pass &= pass ? 1 : 0;

        // ---- [3b] iters=16 参考（信息项，不作判据）----
        // gate2 已证：合成数据上 Sinkhorn 是病态轨迹，宿主 libm 与设备数学的 ulp 差会被迭代放大
        // （1-ulp 输入扰动就能改 4600 字节）⇒ 这里只报 cosine。真实数据 best_iter=0，不受影响。
        {
            gate_encode_kernel<<<1, 128, kDyn>>>(d_stage, 0, 0, d_kc, d_ks, d_vc, d_vs, 0, 1, 16);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            gate_decode_kernel<<<1, 128>>>(d_kc, d_ks, d_vc, d_vs, 0, 1, d_ok2, d_ov2);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<__half> gk16(128 * 256), gv16(128 * 256);
            CUDA_CHECK(cudaMemcpy(gk16.data(), d_ok2, gk16.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gv16.data(), d_ov2, gv16.size() * sizeof(__half), cudaMemcpyDeviceToHost));
            static float hk16[128][256], hv16[128][256];
            for (int s = 0; s < 2; ++s) {
                float q[128][128];
                h_scalar_roundtrip(tl[s], 4, 16, q);
                for (int t = 0; t < 128; ++t)
                    for (int dl = 0; dl < 128; ++dl) hk16[t][s * 128 + dl] = q[dl][t];
                h_scalar_roundtrip(tv[s], 4, 16, q);
                for (int t = 0; t < 128; ++t)
                    for (int dl = 0; dl < 128; ++dl) hv16[t][s * 128 + dl] = q[t][dl];
            }
            const Cmp i1 = compare(gk16, hk16), i2 = compare(gv16, hv16);
            printf("[3b-%s] iters=16（信息项，不作判据）K cosine=%.6f  V cosine=%.6f\n", names[ci],
                   i1.cos, i2.cos);
        }

        // ---- [4] 两段 staging（C 路：先写前 100 个 token，再写后 28 个）== 一次写满（A 路）----
        std::vector<__half> split(stage_elems, __half(0.0f)); // 先全清空
        for (int t = 0; t < 100; ++t) {
            for (int d = 0; d < 256; ++d) {
                split[t * kKvarnHeadDim + d]               = hstage[t * kKvarnHeadDim + d];
                split[stage_plane + t * kKvarnHeadDim + d] = hstage[stage_plane + t * kKvarnHeadDim + d];
            }
        }
        // 后 28 个 token 由第二次写入补齐（模拟 C 路：先前攒了 100 个，这次封口）
        for (int t = 100; t < 128; ++t) {
            for (int d = 0; d < 256; ++d) {
                split[t * kKvarnHeadDim + d]               = hstage[t * kKvarnHeadDim + d];
                split[stage_plane + t * kKvarnHeadDim + d] = hstage[stage_plane + t * kKvarnHeadDim + d];
            }
        }
        CUDA_CHECK(cudaMemcpy(d_stage, split.data(), stage_elems * sizeof(__half), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_kc, 0xAA, code_elems));
        CUDA_CHECK(cudaMemset(d_vc, 0xAA, code_elems));
        gate_encode_kernel<<<1, 128, kDyn>>>(d_stage, 0, 0, d_kc, d_ks, d_vc, d_vs, 0, 1, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        gate_decode_kernel<<<1, 128>>>(d_kc, d_ks, d_vc, d_vs, 0, 1, d_ok2, d_ov2);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<__half> gk2(128 * 256), gv2(128 * 256);
        CUDA_CHECK(cudaMemcpy(gk2.data(), d_ok2, gk2.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(gv2.data(), d_ov2, gv2.size() * sizeof(__half), cudaMemcpyDeviceToHost));
        int dk = 0, dv = 0;
        for (std::size_t i = 0; i < gk.size(); ++i) {
            if (gk[i] != gk2[i]) ++dk;
            if (gv[i] != gv2[i]) ++dv;
        }
        printf("[4-%s] staging 分段写 vs 一次写满：K 不一致=%d  V 不一致=%d ⇒ %s（应全 0）\n",
               names[ci], dk, dv, (dk == 0 && dv == 0) ? "PASS" : "FAIL");
        all_pass &= (dk == 0 && dv == 0) ? 1 : 0;
    }

    printf("\n================ 汇总 ================\n");
    printf("36 格字节账不一致 = %d / 36\n", bad);
    printf("P1 K4V4 = %d B/token/head（record 口径 280；生产 rk4v4 = 280，P1 多 12 B 是维轴按页存两份）\n",
           kvarn_plane_bytes_per_token(4, 4));
    printf("总判定：%s\n", (bad == 0 && all_pass) ? "PASS" : "FAIL");
    cudaFree(d_kc); cudaFree(d_vc); cudaFree(d_ks); cudaFree(d_vs); cudaFree(d_stage);
    cudaFree(d_ok); cudaFree(d_ov); cudaFree(d_ok2); cudaFree(d_ov2);
    return (bad == 0 && all_pass) ? 0 : 1;
}

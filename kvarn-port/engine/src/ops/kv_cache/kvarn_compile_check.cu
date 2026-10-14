// kvarn_compile_check.cu —— 只做编译检查（本机不跑，不占 GPU）。
// 共享内存预算的实证：cur 瓦片 64 KB 走 extern __shared__；orig 每线程一行 64 float 放寄存器。
#include "ops/kv_cache/append/sinkhorn_kernel.cuh"
#include "ops/kv_cache/sinkhorn_g64_codec.cuh"

using namespace ninfer::ops;

__global__ void kvarn_tile_probe(const float* __restrict__ in_tile, float* __restrict__ out_bal,
                                 __half* __restrict__ scales, __half* __restrict__ zps, int dyn_bytes,
                                 unsigned char* __restrict__ out_bytes, int physical_page, int kv_head, int BitsProbe) {
    extern __shared__ float cur[];                                   // 64 KB：256 通道 × 64 token
    __shared__ float log_col[kKVarNTileTokens], log_row[kKVarNHeadDim];
    __shared__ float best_col[kKVarNTileTokens], best_row[kKVarNHeadDim];
    float mine[kKVarNTileTokens];                                    // 本线程那一行（= 一个通道）的原始值

    const int r = threadIdx.x;                                       // 256 线程 = 256 通道
    for (int c = 0; c < kKVarNTileTokens; ++c) {
        mine[c] = in_tile[r * kKVarNTileTokens + c];
        cur[r * kKVarNTileTokens + c] = mine[c];
    }
    __syncthreads();

    kvarn_hadamard_channels(cur);                                    // 旋转（含 1/16 归一）
    __syncthreads();
    // 旋转后每线程重新取回自己那一行（寄存器里的 orig 就是旋转后的值，语义与参考一致）
    for (int c = 0; c < kKVarNTileTokens; ++c) mine[c] = cur[r * kKVarNTileTokens + c];
    __syncthreads();

    __shared__ float row_std_sh[kKVarNHeadDim];
    __shared__ float imb_box[4];
    kvarn_sinkhorn_tile_rows(cur, mine, log_col, log_row, best_col, best_row, row_std_sh, imb_box, kKVarNIters);
    __syncthreads();

    __half sc, zp;
    unsigned char codes[kKVarNTileTokens];
    kvarn_encode_row_quant(&cur[r * kKVarNTileTokens], kKVarNTileTokens, 3, codes, &sc, &zp);
    scales[r] = sc; zps[r] = zp;
    // 探针：把码与 scale 按真实 plane 索引写出去（4-bit / 9 槽位口径），只为取证编译
    kvarn_write_codes<4, 128, 4>(codes, out_bytes, physical_page, kv_head, r);
    kvarn_write_scale_slots<9, 4>(sc, zp, out_bytes, physical_page, kv_head, r);
    // 2-bit 口径也要编过（Merge=4，只有 (d&3)==0 的通道写整字节）
    if (BitsProbe == 2) { kvarn_write_codes<2, 64, 4>(codes, out_bytes, physical_page, kv_head, r); }
    if (threadIdx.x < kKVarNTileTokens) out_bal[r * kKVarNTileTokens + threadIdx.x] = cur[r * kKVarNTileTokens + threadIdx.x];
    (void)dyn_bytes;
}

static_assert(kKVarNSlotsPerToken == 9 && kKVarNSlotsSymmetric == 5, "slot layout");
static_assert(kKVarNTileTokens == 64, "tile must equal one ninfer page");

// 页级 kernel 本体也要实例化（只为取证编译；运行留给内核级对拍那一轮）
void kvarn_kernel_launch_probe(const __half* staging, unsigned char* codes, unsigned char* scales,
                               int page, int kv_head, cudaStream_t stream) {
    const int dyn = ninfer::ops::kKVarNHeadDim * ninfer::ops::kKVarNTileTokens * (int)sizeof(float);
    ninfer::ops::kv_cache_append_kvarn_kernel<4, true, 4><<<1, ninfer::ops::kKVarNHeadDim, dyn, stream>>>(
        staging, codes, scales, page, kv_head);
    ninfer::ops::kv_cache_append_kvarn_kernel<2, false, 4><<<1, ninfer::ops::kKVarNHeadDim, dyn, stream>>>(
        staging, codes, scales, page, kv_head);
}

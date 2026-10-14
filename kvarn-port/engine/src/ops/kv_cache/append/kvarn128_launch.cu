// kvarn128_launch.cu —— G128 KVarN 的 append（A 满瓦片直写 / B 未满只写 stage / C 封口）
// + device 侧页表解析 + 消费端 bf16 窗口。新 TU，与 G64 旧路线（kvarn_launch.cu）并存但互不影响。
//
// 硬约束复述（违反会重演静默死）：
//   * host 绝不解引用任何**设备**指针：block table（KvarnTile128BlockTable::resolve() 只有 device 版）
//     以及 **positions**。实机取证（LD_PRELOAD 的 SIGSEGV handler + backtrace + addr2line）：
//     kv_cache_append_kvarn128_launch 里 `positions[0]` 的 host 读直接 SIGSEGV，故障指令
//     `mov (%rax),%r12d`（rax 来自 `positions`），栈上调用链
//     kv_cache_append_batch_launch ← causal_attention_prompt_launch ← causal_softmax_attention
//     ← qwen3_5 TextContext::attn_mix ← target_verify_batch ← ProgramImpl::prepare_graphs。
//     ⇒ 所有"位置 → tile/page"的推导一律在 device 上做。
//   * B 路不碰 metadata（尾块还没落到页上）。
//   * staging 必须在 CUDA Graph capture 之前预分配，且双槽 ping-pong
//     （MTP 拒 draft 回滚会重开上一组，单槽会静默读错源）。
//   * 每个 tile 的 rotate → encode 必须**成对**按 stream 顺序做：stage 只有 2 个 slot（tile%2），
//     一次 append 跨多个 tile 时先做完所有 rotate 会让 tile T 的 slot 被 tile T+2 覆盖 ⇒ 静默读错。
#include "ops/kv_cache/append/kvarn128_launch.h"

#include "core/device.h"
#include "core/paged_kv_cache.h"
#include "core/vmm.h"
#include "ops/kv_cache/append/kvarn128_codec.cuh"
#include "ops/kv_cache/kvarn_layout.h"

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::ops {
namespace {

// 与上游一致：128 线程（threadIdx.x 即 row）、动态 smem = 一个 128×128 fp32 瓦片 + 归约暂存。
constexpr int kThreads          = kKvarnSliceDim; // 128
constexpr int kDynamicSmemBytes = (kKvarnTileTokens * kKvarnSliceDim + 8 * kKvarnSliceDim + 2 +
                                   16) *
                                  static_cast<int>(sizeof(float)); // 69,704 B（= gate2 实测 required）
constexpr int kMaxKVHeads = 4; // 引擎侧只注册 4/2；staging/窗口按 4 预留

// ---------------------------------------------------------------------------
// 位置 → tile 的推导**全部在 device 上**（host 读 positions = SIGSEGV）。
// 前提：本次 append 的位置序列在 positions[0..count) 里是连续的（引擎的单序列 append 语义）。
// index 是本次启动要处理的"本 append 覆盖范围内第 index 个 tile"（host 用 tokens/128+2 上界扫）。
// ---------------------------------------------------------------------------
struct KvarnAppendTile {
    int  tile     = 0;
    int  base     = 0; // tile * 128
    int  tt_begin = 0; // 本 tile 内本次要被 rotate 的起点（tile 内下标）
    int  tt_end   = 0; // 终点（不含）
    bool sealed   = false;
    bool in_range = false;
};

__device__ __forceinline__ KvarnAppendTile kvarn128_append_tile(const std::int32_t* positions,
                                                                int count, int index) {
    KvarnAppendTile r;
    const int pos0  = positions[0];
    const int last  = positions[count - 1];
    const int t0    = pos0 / kKvarnTileTokens;
    const int tlast = last / kKvarnTileTokens;
    r.tile          = t0 + index;
    if (r.tile < t0 || r.tile > tlast) { return r; }
    r.in_range = true;
    r.base     = r.tile * kKvarnTileTokens;
    r.tt_begin = pos0 > r.base ? pos0 - r.base : 0;
    r.tt_end   = (last + 1 < r.base + kKvarnTileTokens) ? last + 1 - r.base : kKvarnTileTokens;
    r.sealed   = (r.base + kKvarnTileTokens <= last + 1);
    return r;
}

// ---------------------------------------------------------------------------
// 旋转 + 落 stage：一个 warp 一个 token（lane l 载 dims l+32r），K/V 各转一次。
// stage 布局 = [which][slot][head][tile_token 0..127][dim 0..255]，旋转域 fp16。
// grid = (kv_heads)，block = 128 线程（4 warps）；本 tile 的 token 由 warp 轮流搬。
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(kThreads) kvarn128_rotate_tile_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, int count, int index, int kv_heads,
    __half* __restrict__ stage) {
    const KvarnAppendTile t = kvarn128_append_tile(positions, count, index);
    if (!t.in_range || t.tt_begin >= t.tt_end) { return; }
    const int head = blockIdx.x;
    const int slot = t.tile % kKvarnPagesPerTile;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int pos0 = positions[0];

    const std::int64_t stage_plane =
        static_cast<std::int64_t>(kKvarnTileTokens) * kKvarnHeadDim * kv_heads * kKvarnPagesPerTile;
    for (int tt = t.tt_begin + warp; tt < t.tt_end; tt += kThreads / 32) {
        const int token = t.base + tt - pos0; // positions[] 下标（连续假设）
        for (int which = 0; which < 2; ++which) {
            const __nv_bfloat16* src = which ? v : k;
            __half* dst =
                stage + which * stage_plane +
                (static_cast<std::int64_t>(slot) * kv_heads + head) * kKvarnTileTokens *
                    kKvarnHeadDim +
                static_cast<std::int64_t>(tt) * kKvarnHeadDim;
            float values[8];
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                const int d = lane + 32 * r;
                values[r]   = __bfloat162float(
                    src[static_cast<std::int64_t>(d) +
                        static_cast<std::int64_t>(kKvarnHeadDim) * (head + kv_heads * token)]);
            }
            kvarn128_rotate_token(values, lane);
#pragma unroll
            for (int r = 0; r < 8; ++r) { dst[lane + 32 * r] = __float2half_rn(values[r]); }
        }
    }
}

// ---------------------------------------------------------------------------
// 编码：一个 block 干 (tile, head)。只在本 tile 真的攒满（sealed）时才写码流平面。
// 主体在 kvarn128_codec.cuh 的 kvarn128_encode_tile —— 与独立 harness 共用同一份。
// ---------------------------------------------------------------------------
template <int Bits, int KVHeads>
__global__ void __launch_bounds__(kThreads) kvarn128_encode_kernel(
    const __half* __restrict__ stage, const std::int32_t* __restrict__ positions, int count,
    int index, std::uint8_t* __restrict__ k_code, __half* __restrict__ k_scale,
    std::uint8_t* __restrict__ v_code, __half* __restrict__ v_scale, KvarnTile128BlockTable table,
    int iters) {
    const KvarnAppendTile t = kvarn128_append_tile(positions, count, index);
    if (!t.in_range || !t.sealed) { return; }
    extern __shared__ float sh[]; // 69,704 B：瓦片 + quantize 的归约暂存
    __shared__ std::uint8_t rec[kKvarnMaxRecordBytes];
    __shared__ float sa[kKvarnSlices][kKvarnSliceDim];
    __shared__ float zp[kKvarnSlices][kKvarnSliceDim];
    __shared__ float oa[kKvarnSlices][kKvarnSliceDim];
    __shared__ float bc[kKvarnSliceDim];
    __shared__ float br[kKvarnSliceDim];

    // block table 只在 device 上解析（host 解析 = 设备函数 host 调用 ⇒ exit(1) 陷阱）。
    const std::int32_t* bt  = table.resolve();
    const std::int32_t page0 = bt[2 * t.tile];
    const std::int32_t page1 = bt[2 * t.tile + 1];
    const int slot           = t.tile % kKvarnPagesPerTile;

    kvarn128_encode_tile<Bits, KVHeads>(stage, slot, blockIdx.x, k_code, k_scale, v_code, v_scale,
                                        page0, page1, sh, rec, sa, zp, oa, bc, br, iters);
}

// ---------------------------------------------------------------------------
// 消费端窗口：把本 tile 的值写进 bf16 窗口平面（页表/物理页与量化平面一致）。
//   * sealed ⇒ 从码流平面**真解码**（kvarn128_decode_plane 的同一套槽位映射）；
//   * 未封口 ⇒ 从 stage 取（旋转域原值）；两者都做一次 Hadamard 解旋转（H 对合）后落 bf16。
// 一个 warp 一个 token，lane l 载 dims l+32r。grid = (kv_heads)，block = 128 线程。
// ---------------------------------------------------------------------------
template <int Bits, int KVHeads>
__global__ void __launch_bounds__(kThreads) kvarn128_window_kernel(
    const std::uint8_t* __restrict__ k_code, const __half* __restrict__ k_scale,
    const std::uint8_t* __restrict__ v_code, const __half* __restrict__ v_scale,
    const __half* __restrict__ stage, const std::int32_t* __restrict__ positions, int count,
    int index, KvarnTile128BlockTable table, __nv_bfloat16* __restrict__ win_k,
    __nv_bfloat16* __restrict__ win_v) {
    constexpr int CodeLeading = kvarn_code_leading(Bits);
    constexpr int DimsPerByte = 8 / Bits;
    const KvarnAppendTile t = kvarn128_append_tile(positions, count, index);
    if (!t.in_range) { return; }
    const int hi = t.sealed ? kKvarnTileTokens : t.tt_end;
    if (hi <= 0) { return; }

    const int head = blockIdx.x;
    const int slot = t.tile % kKvarnPagesPerTile;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* bt  = table.resolve();
    const std::int32_t page0 = bt[2 * t.tile];
    const std::int32_t page1 = bt[2 * t.tile + 1];
    const std::int64_t stage_plane =
        static_cast<std::int64_t>(kKvarnTileTokens) * kKvarnHeadDim * KVHeads * kKvarnPagesPerTile;
    const std::int64_t stage_head =
        static_cast<std::int64_t>(slot) * KVHeads * kKvarnTileTokens * kKvarnHeadDim +
        static_cast<std::int64_t>(head) * kKvarnTileTokens * kKvarnHeadDim;

    for (int tt = warp; tt < hi; tt += kThreads / 32) {
        const int page = (tt < kKvarnPageTokens) ? page0 : page1;
        const int off  = tt & (kKvarnPageTokens - 1);
        float kk[8];
        float vv[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d     = lane + 32 * r;
            const int slice = d / kKvarnSliceDim; // 0/1
            if (t.sealed) {
                const int b  = d / DimsPerByte;
                const int qk = (static_cast<int>(k_code[kvarn_plane_element_offset(
                                   CodeLeading, KVHeads, page, head, off, b)]) >>
                                ((d % DimsPerByte) * Bits)) &
                               ((1 << Bits) - 1);
                const int qv = (static_cast<int>(v_code[kvarn_plane_element_offset(
                                   CodeLeading, KVHeads, page, head, off, b)]) >>
                                ((d % DimsPerByte) * Bits)) &
                               ((1 << Bits) - 1);
                const float ks = __half2float(k_scale[kvarn_plane_element_offset(
                    kKvarnKScaleLeading, KVHeads, page, head, kvarn_axis_row(d),
                    kvarn_k_scale_col_scale(d))]);
                const float kz = __half2float(k_scale[kvarn_plane_element_offset(
                    kKvarnKScaleLeading, KVHeads, page, head, kvarn_axis_row(d),
                    kvarn_k_scale_col_zp(d))]);
                const float ko = __half2float(k_scale[kvarn_plane_element_offset(
                    kKvarnKScaleLeading, KVHeads, page, head, off,
                    kvarn_k_scale_col_other(slice))]);
                kk[r] = (qk * ks + kz) * ko;
                const float vo = __half2float(v_scale[kvarn_plane_element_offset(
                    kKvarnVScaleLeading, KVHeads, page, head, kvarn_axis_row(d),
                    kvarn_v_scale_col_other(d))]);
                const float vsc = __half2float(v_scale[kvarn_plane_element_offset(
                    kKvarnVScaleLeading, KVHeads, page, head, off,
                    kvarn_v_scale_col_scale(slice))]);
                const float vz = __half2float(v_scale[kvarn_plane_element_offset(
                    kKvarnVScaleLeading, KVHeads, page, head, off,
                    kvarn_v_scale_col_zp(slice))]);
                vv[r] = (qv * vsc + vz) * vo;
            } else {
                kk[r] = __half2float(
                    stage[stage_head + static_cast<std::int64_t>(tt) * kKvarnHeadDim + d]);
                vv[r] = __half2float(stage[stage_plane + stage_head +
                                           static_cast<std::int64_t>(tt) * kKvarnHeadDim + d]);
            }
        }
        // H 对合（symmetric unitary）⇒ 再乘一次就是解旋转回真实域。
        kvarn128_rotate_token(kk, lane);
        kvarn128_rotate_token(vv, lane);
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            const int d = lane + 32 * r;
            const std::int64_t at =
                kvarn_plane_element_offset(kKvarnHeadDim, KVHeads, page, head, off, d);
            win_k[at] = __float2bfloat16_rn(kk[r]);
            win_v[at] = __float2bfloat16_rn(vv[r]);
        }
    }
}

// ---------------------------------------------------------------------------
// 解码对拍：plane → 旋转域 fp16 [128 token][256 dim]
// ---------------------------------------------------------------------------
template <int Bits, int KVHeads>
__global__ void __launch_bounds__(kThreads) kvarn128_decode_kernel(
    const std::uint8_t* __restrict__ k_code, const __half* __restrict__ k_scale,
    const std::uint8_t* __restrict__ v_code, const __half* __restrict__ v_scale,
    KvarnTile128BlockTable table, int tile, __half* __restrict__ out_k,
    __half* __restrict__ out_v) {
    const std::int32_t* bt  = table.resolve();
    const std::int32_t page0 = bt[2 * tile];
    const std::int32_t page1 = bt[2 * tile + 1];
    kvarn128_decode_tile_device<Bits, KVHeads>(k_code, k_scale, v_code, v_scale, blockIdx.x, page0,
                                               page1, out_k, out_v);
}

// ---------------------------------------------------------------------------
// staging 池：**每层一份**（键 = 量化 K 平面基址，与窗口同键），双槽 ping-pong
// （MTP 拒 draft 回滚会重开上一组，单槽会静默读错源）。
//
// 为什么必须按层分：stage 承载的语义是「**这块 tile 的旋转域累积值**」，不是「本次 append 的暂存」——
//   * rotate 只写 [tt_begin, tt_end)（本次 append 覆盖的那一段）；
//   * encode 在**封口**时读满 128 槽（整块 tile 都要）；
//   * 未封口窗口读 [0, tt_end)（这块 tile 到目前为止的全部 token）。
// 用一块全局 stage 时，后两处读到的 [0, tt_begin) 是**别的 layer / 别的 append 的残留**。
// 实测（KVARN128_DUMP_DIR 的 pre/cont 对拍）：未封口窗口的「保持段」[0,51) 被 51/51 个 token
// 覆盖成别人的值（cos 由 1.000000 掉到 -0.237~-0.103）⇒ 窗口里已经不是这块 tile 的真值。
// 按层分之后 stage 就是这块 tile 的累积值：rotate 写一段、其余保持上一次 append 写的内容。
// 必须在 CUDA Graph capture 之前分配（capture 期禁 malloc）。
// ---------------------------------------------------------------------------
constexpr int kMaxStages = 256;

struct KvarnStage {
    const void* key   = nullptr;
    int         heads = 0;
    __half*     buf   = nullptr;
};

KvarnStage g_stages[kMaxStages];
int        g_stage_count = 0;

// 取本层 staging；不存在则建（含 capture 前置检查）。成功时 *out 有效并返回 nullptr。
const char* ensure_stage_for(const void* key, int kv_heads, cudaStream_t stream, __half** out) {
    *out = nullptr;
    const int heads = kv_heads > 0 ? kv_heads : kMaxKVHeads;
    for (int i = 0; i < g_stage_count; ++i) {
        if (g_stages[i].key == key) {
            if (g_stages[i].heads < heads) {
                return "KVarN staging 形状不一致（同一层的 KVHeads 变大了）";
            }
            *out = g_stages[i].buf;
            return nullptr;
        }
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        return "KVarN staging 必须在 CUDA Graph capture 之前完成分配（warmup 里第一次 append 就会触发）";
    }
    if (g_stage_count >= kMaxStages) { return "KVarN staging 池已满"; }
    // 布局 = [which(K/V)][slot][head][token][dim]，rotate/encode/window 三处同一式
    //   stage[which * 128*256*heads*2 + (slot*heads + head)*128*256 + tt*256 + d]
    // ⇒ 需要 2(which) × kKvarnPagesPerTile(slot) × heads × 128 × 256 个 half。
    // （旧值 2*heads*128*256 漏了 which 那一维 ⇒ V 平面整段落在分配尾后：heads=4 时每个
    //   rotate/encode/window 都越界读写 512 KiB；cudaMemcpy 从该地址取数直接报错。）
    const std::size_t elements = static_cast<std::size_t>(2) * kKvarnPagesPerTile * heads *
                                 kKvarnTileTokens * kKvarnHeadDim;
    const std::size_t bytes = elements * sizeof(__half);
    __half* fresh = nullptr;
    if (cudaMalloc(&fresh, bytes) != cudaSuccess) {
        cudaGetLastError();
        return "KVarN staging cudaMalloc 失败（2 which × 2 slot × KVHeads × 128 × 256 × 2 B）";
    }
    if (cudaMemset(fresh, 0, bytes) != cudaSuccess) { cudaGetLastError(); }
    g_stages[g_stage_count++] = KvarnStage{key, heads, fresh};
    *out = fresh;
    return nullptr;
}

void ensure_encode_opt_in() {
    static bool done = [] {
        cudaFuncSetAttribute(kvarn128_encode_kernel<4, 4>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
        cudaFuncSetAttribute(kvarn128_encode_kernel<4, 2>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, kDynamicSmemBytes);
        return true;
    }();
    (void)done;
}

// ---------------------------------------------------------------------------
// 消费端窗口池：按量化 K 平面基址（每层唯一）索引。窗口页表与量化平面**完全一致**，
// 所以 bf16 消费核能直接读。
//
// 物理后备是**惰性**的（NINFER_KVARN128_WINDOW_FULL=1 退回旧的整块 cudaMalloc）：
//   * 窗口的**虚拟地址**一次预留（张量形状、块表、CUDA Graph 抓到的指针全部不变）；
//   * 物理块按 KV 页**实际落地量**分批提交（2 MiB = VMM 最小粒度）；
//   * 启动热身/抓图期写入的块在 ProgramImpl 构造末尾释放（虚拟地址保留），此后由
//     ensure_sequence_kv_mapped 按序列覆盖量重新提交。
//   ⇒ 空闲态窗口不占显存；占用随上下文增长，与 KV 池自身的 materialize 同口径。
// ---------------------------------------------------------------------------
constexpr int kMaxWindows = 256;
// 覆盖量之外多提交几页：物理页号相对逻辑页号可能有偏移（抓图期占过页）。
constexpr std::size_t kWindowCommitMarginPages = 2;

struct KvarnWindow {
    const void*    key      = nullptr;
    int            pages    = 0;
    int            kv_heads = 0;
    __nv_bfloat16* k        = nullptr;
    __nv_bfloat16* v        = nullptr;
    // 惰性路账目；granule == 0 ⇒ 整块 cudaMalloc（无物理块账）。
    std::size_t page_bytes   = 0; // 单 plane 一页（64 token × heads × 256 dim）的字节数
    int         device       = -1;
    std::size_t granule      = 0;
    std::size_t reserved     = 0; // 虚拟预留字节（granule 整数倍）
    std::size_t k_committed  = 0; // K 已提交字节（K/V 各自记账：共用会漏掉一片）
    std::size_t v_committed  = 0;
    CUdeviceptr k_base       = 0;
    CUdeviceptr v_base       = 0;
    std::vector<CUmemGenericAllocationHandle> k_handles;
    std::vector<CUmemGenericAllocationHandle> v_handles;
};

KvarnWindow g_windows[kMaxWindows];
int         g_window_count = 0;
// 已请求过的最大覆盖量（tokens）。窗口是**第一次 append** 时才建的，而 scoring/graft 路会在那之前
// 就把覆盖量推到整窗（causal_score 里 create_active 后立刻 ensure_mapped_to_tokens(8192)），
// 所以建窗时必须按这个高水位提交，否则那次提交会落空、随后 append 直接越过已提交区（= illegal address）。
int g_commit_tokens = 0;

std::size_t kvarn128_wanted_pages(int tokens) {
    if (tokens <= 0) { return 0; }
    return static_cast<std::size_t>(tokens) / kKvarnPageTokens + 1 + kWindowCommitMarginPages;
}

bool kvarn128_window_full_mode() {
    static const bool on = [] {
        const char* e = std::getenv("NINFER_KVARN128_WINDOW_FULL");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

// 预留一段虚拟地址（不提交物理）。失败返回 false ⇒ 调用方退回整块 cudaMalloc。
bool kvarn128_reserve_va(std::size_t bytes, int device, CUdeviceptr* out, std::size_t* reserved,
                         std::size_t* granule) {
    if (!vmm::supported(device)) { return false; }
    const std::size_t g = vmm::granularity(device);
    if (g == 0) { return false; }
    vmm::ensure_driver_initialized();
    const std::size_t r = (bytes + g - 1) / g * g;
    CUdeviceptr       va = 0;
    try {
        NINFER_CU_CHECK(cuMemAddressReserve(&va, r, g, 0, 0));
    } catch (...) {
        return false;
    }
    *out      = va;
    *reserved = r;
    *granule  = g;
    return true;
}

// 把某一 plane 的物理后备提交到至少 want 字节（按 granule 对齐）。false = 分配/映射失败。
// K/V **各自记账**：共用一份 committed 会让第二次调用直接短路 ⇒ 另一片永不映射（实机 = illegal address）。
bool kvarn128_commit_plane(KvarnWindow& w, bool is_k, std::size_t want) {
    std::size_t& committed = is_k ? w.k_committed : w.v_committed;
    if (w.granule == 0 || want <= committed) { return true; }
    const std::size_t target  = (want + w.granule - 1) / w.granule * w.granule;
    const CUdeviceptr base    = is_k ? w.k_base : w.v_base;
    auto&             handles = is_k ? w.k_handles : w.v_handles;
    while (committed < target && committed < w.reserved) {
        const std::size_t piece = std::min(w.granule, w.reserved - committed);
        CUmemGenericAllocationHandle handle{};
        try {
            handle = vmm::create(piece, w.device);
        } catch (...) {
            return false;
        }
        const int devices[1] = {w.device};
        try {
            vmm::map(base + committed, piece, handle, std::span<const int>(devices, 1));
        } catch (...) {
            (void)cuMemRelease(handle);
            return false;
        }
        handles.push_back(handle);
        committed += piece;
    }
    return true;
}

// 释放一个窗口的物理后备（虚拟地址与内容指针保留）；两片记账归零。
void kvarn128_release_window_backing(KvarnWindow& w) {
    if (w.granule == 0) { return; }
    const auto release = [&](CUdeviceptr base, std::vector<CUmemGenericAllocationHandle>& handles) {
        std::size_t offset = 0;
        for (const CUmemGenericAllocationHandle handle : handles) {
            const std::size_t piece = std::min(w.granule, w.reserved - offset);
            (void)cuMemUnmap(base + offset, piece);
            (void)cuMemRelease(handle);
            offset += piece;
        }
        handles.clear();
    };
    release(w.k_base, w.k_handles);
    release(w.v_base, w.v_handles);
    w.k_committed = 0;
    w.v_committed = 0;
}

} // namespace

int kvarn128_window_count() { return g_window_count; }
int kvarn128_window_leading() { return kKvarnHeadDim; }

bool kvarn128_window(const void* k_code, int pages, int kv_heads, cudaStream_t stream, void** win_k,
                     void** win_v) {
    if (win_k != nullptr) { *win_k = nullptr; }
    if (win_v != nullptr) { *win_v = nullptr; }
    if (k_code == nullptr || pages <= 0 || kv_heads <= 0 || kv_heads > kMaxKVHeads) {
        std::fprintf(stderr, "ninfer: KVarN128 窗口参数非法（k=%p pages=%d kv_heads=%d）\n", k_code,
                     pages, kv_heads);
        std::fflush(stderr);
        return false;
    }
    for (int i = 0; i < g_window_count; ++i) {
        if (g_windows[i].key == k_code) {
            if (g_windows[i].pages != pages || g_windows[i].kv_heads != kv_heads) {
                std::fprintf(stderr,
                             "ninfer: KVarN128 窗口形状不一致（key=%p 已登记 pages=%d heads=%d，"
                             "本次 pages=%d heads=%d）\n",
                             k_code, g_windows[i].pages, g_windows[i].kv_heads, pages, kv_heads);
                std::fflush(stderr);
                return false;
            }
            if (win_k != nullptr) { *win_k = g_windows[i].k; }
            if (win_v != nullptr) { *win_v = g_windows[i].v; }
            return true;
        }
    }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        std::fprintf(
            stderr, "ninfer: KVarN128 窗口必须在 CUDA Graph capture 之前分配（第一次 append 就会触发）\n");
        std::fflush(stderr);
        return false;
    }
    if (g_window_count >= kMaxWindows) {
        std::fprintf(stderr, "ninfer: KVarN128 窗口池已满（%d）\n", kMaxWindows);
        std::fflush(stderr);
        return false;
    }
    const std::size_t bytes = static_cast<std::size_t>(kKvarnHeadDim) * kKvarnPageTokens * kv_heads *
                              pages * sizeof(__nv_bfloat16);
    const std::size_t page_bytes =
        static_cast<std::size_t>(kKvarnHeadDim) * kKvarnPageTokens * kv_heads * sizeof(__nv_bfloat16);
    __nv_bfloat16* k = nullptr;
    __nv_bfloat16* v = nullptr;

    // 惰性路优先：虚拟地址一次预留，只提交「这一块 tile（2 页）」所需的物理块。
    // warmup 的 eager append 会真写这一页，所以首块必须立刻提交；此后由
    // kvarn128_window_commit_tokens 按覆盖量追加。
    if (!kvarn128_window_full_mode()) {
        int device = -1;
        if (cudaGetDevice(&device) != cudaSuccess) {
            cudaGetLastError();
            device = -1;
        }
        CUdeviceptr kb = 0;
        CUdeviceptr vb = 0;
        std::size_t res = 0;
        std::size_t gran = 0;
        if (device >= 0 && kvarn128_reserve_va(bytes, device, &kb, &res, &gran)) {
            if (!kvarn128_reserve_va(bytes, device, &vb, &res, &gran)) {
                (void)cuMemAddressFree(kb, res);
            } else {
                KvarnWindow w;
                w.key       = k_code;
                w.pages     = pages;
                w.kv_heads  = kv_heads;
                w.k         = reinterpret_cast<__nv_bfloat16*>(kb);
                w.v         = reinterpret_cast<__nv_bfloat16*>(vb);
                w.page_bytes = page_bytes;
                w.device    = device;
                w.granule   = gran;
                w.reserved  = res;
                w.k_base    = kb;
                w.v_base    = vb;
                // 首块：按「已请求过的最大覆盖量」提交，且至少 4 个 granule。
                // 抓图热身期块表是「全部逻辑页 → 抓图行的同一物理页」（见 graphs.cpp
                // reserve_capture_rows 的 publish_repeated），只会碰那一页；而 scoring 路会在建窗前
                // 就把覆盖量推到整窗（g_commit_tokens）⇒ 两者都要覆盖。
                // **不能**无条件按整池提交（131072 时整池 = 8.5 GiB，直接 OOM，就是原病灶）。
                const std::size_t by_coverage =
                    kvarn128_wanted_pages(g_commit_tokens) * page_bytes;
                const std::size_t first = std::min(
                    w.reserved,
                    std::max(static_cast<std::size_t>(4) * w.granule, by_coverage));
                if (!kvarn128_commit_plane(w, true, first) ||
                    !kvarn128_commit_plane(w, false, first)) {
                    kvarn128_release_window_backing(w);
                    (void)cuMemAddressFree(kb, res);
                    (void)cuMemAddressFree(vb, res);
                    std::fprintf(stderr,
                                 "ninfer: KVarN128 窗口物理后备提交失败（pages=%d heads=%d 首块 %zu B）\n",
                                 pages, kv_heads, first);
                    std::fflush(stderr);
                    return false;
                }
                if (cudaMemset(reinterpret_cast<void*>(kb), 0, w.k_committed) != cudaSuccess ||
                    cudaMemset(reinterpret_cast<void*>(vb), 0, w.v_committed) != cudaSuccess) {
                    cudaGetLastError();
                }
                g_windows[g_window_count++] = w;
                std::fprintf(stderr,
                             "ninfer: KVarN128 窗口已分配 #%d pages=%d heads=%d 虚拟 %.1f MiB×2 "
                             "已提交 %.1f MiB×2（惰性物理后备，与量化平面同页表）\n",
                             g_window_count - 1, pages, kv_heads,
                             static_cast<double>(res) / (1024.0 * 1024.0),
                             static_cast<double>(w.k_committed) / (1024.0 * 1024.0));
                std::fflush(stderr);
                if (win_k != nullptr) { *win_k = w.k; }
                if (win_v != nullptr) { *win_v = w.v; }
                return true;
            }
        }
        std::fprintf(stderr,
                     "ninfer: KVarN128 窗口惰性后备不可用（device=%d），退回整块 cudaMalloc\n",
                     device);
        std::fflush(stderr);
    }

    if (cudaMalloc(&k, bytes) != cudaSuccess || cudaMalloc(&v, bytes) != cudaSuccess) {
        cudaGetLastError();
        if (k != nullptr) { cudaFree(k); }
        if (v != nullptr) { cudaFree(v); }
        std::fprintf(stderr, "ninfer: KVarN128 窗口 cudaMalloc 失败（%zu B × 2）\n", bytes);
        std::fflush(stderr);
        return false;
    }
    if (cudaMemset(k, 0, bytes) != cudaSuccess || cudaMemset(v, 0, bytes) != cudaSuccess) {
        cudaGetLastError();
    }
    g_windows[g_window_count++] = KvarnWindow{k_code, pages, kv_heads, k, v};
    std::fprintf(stderr,
                 "ninfer: KVarN128 窗口已分配 #%d pages=%d heads=%d %.1f MiB×2"
                 "（真解码，与量化平面同页表）\n",
                 g_window_count - 1, pages, kv_heads,
                 static_cast<double>(bytes) / (1024.0 * 1024.0));
    std::fflush(stderr);
    if (win_k != nullptr) { *win_k = k; }
    if (win_v != nullptr) { *win_v = v; }
    return true;
}

// 按 KV 页实际落地量提交窗口的物理后备（惰性窗口）。tokens ≤ 0 或没有窗口时 no-op。
// 提交失败**大声失败**：窗口内容会被 attention 真读，静默留洞 = 静默读错。
void kvarn128_window_commit_tokens(int tokens) {
    if (tokens <= 0) { return; }
    if (tokens > g_commit_tokens) { g_commit_tokens = tokens; }
    if (g_window_count == 0) { return; }
    const std::size_t pages_wanted = kvarn128_wanted_pages(tokens);
    for (int i = 0; i < g_window_count; ++i) {
        KvarnWindow& w = g_windows[i];
        if (w.granule == 0) { continue; }
        const std::size_t want = std::min(pages_wanted * w.page_bytes, w.reserved);
        if (want <= w.k_committed && want <= w.v_committed) { continue; }
        if (!kvarn128_commit_plane(w, true, want) || !kvarn128_commit_plane(w, false, want)) {
            std::fprintf(stderr,
                         "ninfer: KVarN128 窗口物理后备提交失败（层 #%d pages=%d 需要 %zu B，"
                         "K 已提交 %zu B / V 已提交 %zu B）\n",
                         i, w.pages, want, w.k_committed, w.v_committed);
            std::fflush(stderr);
            throw std::runtime_error("kvarn128 window backing commit failed");
        }
    }
}

// 按**实际落地的物理页号**提交窗口后备。见头文件说明：只按逻辑页数提交会在「第二个请求拿到一段
// 物理页号更靠后的新地址空间」时漏掉整段（实机 = 第二次整窗 prefill 报 illegal address）。
void kvarn128_window_commit_physical_pages(const std::int32_t* pages, int count) {
    if (pages == nullptr || count <= 0 || g_window_count == 0) { return; }
    std::int32_t max_page = -1;
    for (int i = 0; i < count; ++i) {
        if (pages[i] > max_page) { max_page = pages[i]; }
    }
    if (max_page < 0) { return; }
    const std::size_t wanted_pages =
        static_cast<std::size_t>(max_page) + 1 + kWindowCommitMarginPages;
    for (int i = 0; i < g_window_count; ++i) {
        KvarnWindow& w = g_windows[i];
        if (w.granule == 0) { continue; }
        const std::size_t want = std::min(wanted_pages * w.page_bytes, w.reserved);
        if (want <= w.k_committed && want <= w.v_committed) { continue; }
        if (!kvarn128_commit_plane(w, true, want) || !kvarn128_commit_plane(w, false, want)) {
            std::fprintf(stderr,
                         "ninfer: KVarN128 窗口物理后备提交失败（层 #%d pages=%d 需要 %zu B，"
                         "K 已提交 %zu B / V 已提交 %zu B）\n",
                         i, w.pages, want, w.k_committed, w.v_committed);
            std::fflush(stderr);
            throw std::runtime_error("kvarn128 window backing commit failed");
        }
    }
}

// 释放所有窗口的物理后备（虚拟地址保留）。启动热身/抓图结束后调用：空闲态不占显存。
// 调用后窗口内容未定义 ⇒ 后续 append 必须先写后读（attention 只读序列写过的页）。
void kvarn128_window_release_backing() {
    if (g_window_count == 0) { return; }
    for (int i = 0; i < g_window_count; ++i) { kvarn128_release_window_backing(g_windows[i]); }
    std::fprintf(stderr,
                 "ninfer: KVarN128 窗口物理后备已释放（%d 份，虚拟地址保留；按覆盖量重新提交）\n",
                 g_window_count);
    std::fflush(stderr);
}

// ===========================================================================
// 对拍 dump（KVARN128_DUMP_DIR=<dir> 打开；只 dump 一次；不改数值路径）
//   <dir>/kvarn128_p1.dump —— KVD1(layout=1)：tile0/head0 的 K、V 两条（各 2 页）
//   <dir>/kvarn128_win.kvo —— KVO1(domain=1)：引擎窗口里 tile0/head0 的 128×256 bf16→f32
// host 侧只做 cudaMemcpy，绝不解引用设备指针（block table 也走 D2H 拷回）。
// ===========================================================================
namespace {

bool kvarn128_d2h(void* dst, const void* src, std::size_t bytes) {
    if (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return true;
}

void kvarn128_write_kvd1(FILE* f, const char* name, int kind, int bits, int code_leading,
                         int scale_leading, const std::uint8_t* code, const std::uint16_t* scale) {
    const std::uint32_t nl = static_cast<std::uint32_t>(std::strlen(name));
    std::fwrite(&nl, 4, 1, f);
    std::fwrite(name, 1, nl, f);
    const std::int32_t a[7] = {kind,       bits,               kKvarnSliceDim, kKvarnSliceDim,
                               kKvarnTileTokens, kKvarnPagesPerTile, kKvarnSlices};
    std::fwrite(a, 4, 7, f);
    const std::uint32_t flags  = 0;
    const std::uint32_t layout = 1; // P1 页自足槽位布局（decode-g128 §4 建议位）
    const std::uint32_t cl     = static_cast<std::uint32_t>(code_leading);
    const std::uint32_t sl     = static_cast<std::uint32_t>(scale_leading);
    std::fwrite(&flags, 4, 1, f);
    std::fwrite(&layout, 4, 1, f);
    std::fwrite(&cl, 4, 1, f);
    std::fwrite(&sl, 4, 1, f);
    const std::uint32_t cb =
        static_cast<std::uint32_t>(kKvarnPagesPerTile * kKvarnPageTokens * code_leading);
    std::fwrite(&cb, 4, 1, f);
    std::fwrite(code, 1, cb, f);
    const std::uint32_t sh =
        static_cast<std::uint32_t>(kKvarnPagesPerTile * kKvarnPageTokens * scale_leading);
    std::fwrite(&sh, 4, 1, f);
    std::fwrite(scale, 2, sh, f);
    const std::uint32_t of = 0;
    std::fwrite(&of, 4, 1, f);
}

void kvarn128_write_kvo1_head(FILE* f, const char* name, int kind, int domain, const float* data) {
    const std::uint32_t nl = static_cast<std::uint32_t>(std::strlen(name));
    std::fwrite(&nl, 4, 1, f);
    std::fwrite(name, 1, nl, f);
    // domain: 1 = 已逆旋转（真实域，引擎窗口的值）；0 = 旋转域（stage 的原始值）
    const std::int32_t h[4] = {kind, kKvarnTileTokens, kKvarnHeadDim, domain};
    std::fwrite(h, 4, 4, f);
    const std::uint32_t nf = static_cast<std::uint32_t>(kKvarnTileTokens * kKvarnHeadDim);
    std::fwrite(&nf, 4, 1, f);
    std::fwrite(data, 4, nf, f);
}

float kvarn128_bf16_bits_to_f32(std::uint16_t h) {
    const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
    float               f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool kvarn128_dump_verbose() {
    static const bool on = [] {
        const char* e = std::getenv("KVARN128_DUMP_VERBOSE");
        return e != nullptr && e[0] != '\0' && e[0] != '0';
    }();
    return on;
}

// 窗口 tile/head0 的 128×256 bf16 → f32（页表与量化平面一致；host 只 cudaMemcpy）。
void kvarn128_read_window(const __nv_bfloat16* win_k, const __nv_bfloat16* win_v, int kv_heads,
                          const std::int32_t* phys, std::vector<float>& wk,
                          std::vector<float>& wv) {
    for (int tt = 0; tt < kKvarnTileTokens; ++tt) {
        const int          p   = tt / kKvarnPageTokens;
        const int          off = tt % kKvarnPageTokens;
        const std::int64_t base =
            static_cast<std::int64_t>(kKvarnHeadDim) * kKvarnPageTokens *
                (static_cast<std::int64_t>(kv_heads) * phys[p]) +
            static_cast<std::int64_t>(kKvarnHeadDim) * off;
        std::uint16_t tmp[2 * kKvarnHeadDim];
        if (!kvarn128_d2h(tmp, win_k + base, kKvarnHeadDim * 2) ||
            !kvarn128_d2h(tmp + kKvarnHeadDim, win_v + base, kKvarnHeadDim * 2)) {
            return;
        }
        for (int d = 0; d < kKvarnHeadDim; ++d) {
            wk[static_cast<std::size_t>(tt) * kKvarnHeadDim + d] =
                kvarn128_bf16_bits_to_f32(tmp[d]);
            wv[static_cast<std::size_t>(tt) * kKvarnHeadDim + d] =
                kvarn128_bf16_bits_to_f32(tmp[kKvarnHeadDim + d]);
        }
    }
}

// ===========================================================================
// 未封口 tile 的判定 dump：stage（旋转域 fp16→f32，domain=0）与窗口（bf16→f32，domain=1）
// 各 128×256，主机侧对 stage 做一次 inverse_rotate_d256 后与窗口逐 token 比 cos：
//   ≈1 ⇒ 窗口的未封口分支写的就是 stage 的值（分支本身没错）；
//   ≪1 ⇒ 窗口写进了非本次 append 的 stage 内容（陈旧/越界写）。
// 触发：KVARN128_DUMP_DIR 非空、非 capture、且本次 append 的 tile 未封口。
//   首次 tt_begin==0（起始 append）写 kvarn128_unsealed_pre_{stage,win}.kvo；
//   之后 tt_begin>0（续写 append）写 kvarn128_unsealed_cont<N>_{stage,win}.kvo（最多 4 次）。
// ===========================================================================
void kvarn128_dbg(const char* msg, int a, int b) {
    if (!kvarn128_dump_verbose()) { return; }
    std::fprintf(stderr, "ninfer: KVarN128 dump-dbg %s a=%d b=%d\n", msg, a, b);
    std::fflush(stderr);
}

void kvarn128_dump_unsealed(const void* k_code, const void* v_code, const __nv_bfloat16* win_k,
                            const __nv_bfloat16* win_v, const __half* stage,
                            KvarnTile128BlockTable table, int kv_heads,
                            const std::int32_t* positions, int tokens, cudaStream_t stream,
                            const char* dir) {
    kvarn128_dbg("enter", tokens, kv_heads);
    if (positions == nullptr || win_k == nullptr || win_v == nullptr) {
        kvarn128_dbg("bail-null", positions == nullptr, win_k == nullptr);
        return;
    }
    std::int32_t pos0 = 0, poslast = 0;
    if (!kvarn128_d2h(&pos0, positions, 4) ||
        !kvarn128_d2h(&poslast, positions + (tokens - 1), 4)) {
        kvarn128_dbg("bail-pos-d2h", tokens, 0);
        return;
    }
    const int  tile     = pos0 / kKvarnTileTokens;
    const int  base     = tile * kKvarnTileTokens;
    const bool sealed   = (base + kKvarnTileTokens <= poslast + 1);
    const int  tt_begin = pos0 > base ? pos0 - base : 0;
    const int  tt_end   = (poslast + 1 < base + kKvarnTileTokens) ? poslast + 1 - base
                                                                 : kKvarnTileTokens;
    kvarn128_dbg("geom", pos0, poslast);
    if (sealed) {
        kvarn128_dbg("bail-sealed", tile, tt_end);
        return; // 已封口走 KVD1 路
    }

    static int prefill_done = 0, cont_count = 0;
    const bool continuation = tt_begin > 0;
    if (!continuation) {
        if (prefill_done != 0) { return; }
        prefill_done = 1;
    } else if (cont_count >= 4) {
        return;
    }
    const int idx = continuation ? cont_count++ : 0;
    kvarn128_dbg("accepted", continuation ? 1 : 0, idx);

    std::int32_t        row = 0, page0 = -1, page1 = -1;
    const std::int32_t* bt  = table.direct;
    if (bt == nullptr) {
        if (!kvarn128_d2h(&row, table.table_rows, sizeof(row))) {
            kvarn128_dbg("bail-row-d2h", tokens, 0);
            return;
        }
        bt = table.tables + static_cast<std::int64_t>(row) * table.table_stride;
    }
    if (!kvarn128_d2h(&page0, bt + 2 * tile, 4) ||
        !kvarn128_d2h(&page1, bt + 2 * tile + 1, 4)) {
        kvarn128_dbg("bail-page-d2h", tile, row);
        return;
    }
    if (page0 < 0 || page1 < 0) {
        kvarn128_dbg("bail-page-neg", page0, page1);
        return;
    }
    const std::int32_t phys[2] = {page0, page1};

    const std::size_t elems = static_cast<std::size_t>(kKvarnTileTokens) * kKvarnHeadDim;
    std::vector<float> sk(elems), sv(elems), wk(elems), wv(elems);

    // stage 布局 = [which][slot][head][tile_token][dim]，slot = tile % 2，head = 0
    const int          slot = tile % kKvarnPagesPerTile;
    const std::int64_t stage_plane = static_cast<std::int64_t>(kKvarnTileTokens) * kKvarnHeadDim *
                                     kv_heads * kKvarnPagesPerTile;
    const std::int64_t stage_head =
        static_cast<std::int64_t>(slot) * kv_heads * kKvarnTileTokens * kKvarnHeadDim;
    if (stage == nullptr) {
        kvarn128_dbg("bail-stage-null", tokens, 0);
        return;
    }
    std::vector<std::uint16_t> raw(2 * elems);
    if (!kvarn128_d2h(raw.data(), stage + stage_head, elems * 2) ||
        !kvarn128_d2h(raw.data() + elems, stage + stage_plane + stage_head, elems * 2)) {
        kvarn128_dbg("bail-stage-d2h", slot, kv_heads);
        return;
    }
    for (std::size_t i = 0; i < elems; ++i) {
        const __half hk = *reinterpret_cast<const __half*>(&raw[i]);
        const __half hv = *reinterpret_cast<const __half*>(&raw[elems + i]);
        sk[i]           = __half2float(hk);
        sv[i]           = __half2float(hv);
    }
    kvarn128_read_window(win_k, win_v, kv_heads, phys, wk, wv);

    char path[1024];
    const char* tag = continuation ? "cont" : "pre";
    std::snprintf(path, sizeof(path), "%s/kvarn128_unsealed_%s%d_stage.kvo", dir, tag, idx);
    FILE* f = std::fopen(path, "wb");
    if (f != nullptr) {
        const std::uint32_t magic = 0x314F564Bu, ver = 1, n = 2;
        std::fwrite(&magic, 4, 1, f);
        std::fwrite(&ver, 4, 1, f);
        std::fwrite(&n, 4, 1, f);
        kvarn128_write_kvo1_head(f, "L0-K-h0", 0, 0 /*旋转域*/, sk.data());
        kvarn128_write_kvo1_head(f, "L0-V-h0", 1, 0 /*旋转域*/, sv.data());
        std::fclose(f);
    }
    std::snprintf(path, sizeof(path), "%s/kvarn128_unsealed_%s%d_win.kvo", dir, tag, idx);
    f = std::fopen(path, "wb");
    if (f != nullptr) {
        const std::uint32_t magic = 0x314F564Bu, ver = 1, n = 2;
        std::fwrite(&magic, 4, 1, f);
        std::fwrite(&ver, 4, 1, f);
        std::fwrite(&n, 4, 1, f);
        kvarn128_write_kvo1_head(f, "L0-K-h0", 0, 1 /*已逆旋转*/, wk.data());
        kvarn128_write_kvo1_head(f, "L0-V-h0", 1, 1 /*已逆旋转*/, wv.data());
        std::fclose(f);
    }
    std::fprintf(stderr,
                 "ninfer: KVarN128 unsealed dump[%s#%d] tokens=%d pos0=%d poslast=%d tile=%d "
                 "tt=[%d,%d) heads=%d pages=%d/%d\n",
                 tag, idx, tokens, pos0, poslast, tile, tt_begin, tt_end, kv_heads, page0, page1);
    std::fflush(stderr);
}

void kvarn128_dump_probe(const void* k_code, const void* k_scale, const void* v_code,
                         const void* v_scale, KvarnTile128BlockTable table, int kv_heads,
                         const __nv_bfloat16* win_k, const __nv_bfloat16* win_v,
                         const __half* stage, const std::int32_t* positions, int tokens,
                         cudaStream_t stream) {
    const char* dir = std::getenv("KVARN128_DUMP_DIR");
    if (dir == nullptr || dir[0] == '\0') { return; }
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    cudaStreamIsCapturing(stream, &cap);
    if (cap != cudaStreamCaptureStatusNone) {
        std::fprintf(stderr, "ninfer: KVarN128 dump 跳过（本次 append 在 CUDA Graph capture 中）\n");
        std::fflush(stderr);
        return;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { cudaGetLastError(); return; }
    kvarn128_dbg("probe-enter", tokens, kv_heads);
    if (tokens < kKvarnTileTokens) { // 未封口 tile：走 stage/窗口对拍分支
        kvarn128_dump_unsealed(k_code, v_code, win_k, win_v, stage, table, kv_heads, positions,
                               tokens, stream, dir);
        return;
    }
    static bool done = false;
    if (done) { return; }
    done = true;
    constexpr int KB = 4, VB = 4;
    constexpr int CK = 32 * KB, CV = 32 * VB;
    constexpr int SK = kKvarnKScaleLeading, SV = kKvarnVScaleLeading;

    // 物理页：block table 只在 device 上解析 ⇒ host 只用 cudaMemcpy 取回（tile 0 ⇒ 条目 0/1）
    std::int32_t        row = 0, page0 = -1, page1 = -1;
    const std::int32_t* bt  = table.direct;
    if (bt == nullptr) {
        if (!kvarn128_d2h(&row, table.table_rows, sizeof(row))) { return; }
        bt = table.tables + static_cast<std::int64_t>(row) * table.table_stride;
    }
    if (!kvarn128_d2h(&page0, bt + 0, 4) || !kvarn128_d2h(&page1, bt + 1, 4)) { return; }
    if (page0 < 0 || page1 < 0) { return; }

    // 每个 plane 的「页块」基址 = leading*64*(head + heads*phys)，head = 0
    auto page_block = [kv_heads](int leading, std::int32_t phys) -> std::int64_t {
        return static_cast<std::int64_t>(leading) * kKvarnPageTokens *
               (static_cast<std::int64_t>(kv_heads) * phys);
    };
    const std::int32_t phys[2] = {page0, page1};

    std::uint8_t  kc[2 * kKvarnPageTokens * CK];
    std::uint8_t  vc[2 * kKvarnPageTokens * CV];
    std::uint16_t ks[2 * kKvarnPageTokens * SK];
    std::uint16_t vs[2 * kKvarnPageTokens * SV];
    for (int p = 0; p < 2; ++p) {
        if (!kvarn128_d2h(kc + p * kKvarnPageTokens * CK,
                          static_cast<const std::uint8_t*>(k_code) + page_block(CK, phys[p]),
                          kKvarnPageTokens * CK) ||
            !kvarn128_d2h(vc + p * kKvarnPageTokens * CV,
                          static_cast<const std::uint8_t*>(v_code) + page_block(CV, phys[p]),
                          kKvarnPageTokens * CV) ||
            !kvarn128_d2h(ks + p * kKvarnPageTokens * SK,
                          static_cast<const __half*>(k_scale) + page_block(SK, phys[p]),
                          kKvarnPageTokens * SK * 2) ||
            !kvarn128_d2h(vs + p * kKvarnPageTokens * SV,
                          static_cast<const __half*>(v_scale) + page_block(SV, phys[p]),
                          kKvarnPageTokens * SV * 2)) {
            return;
        }
    }

    // 窗口值（bf16，已逆旋转）→ f32
    std::vector<float> wk(kKvarnTileTokens * kKvarnHeadDim);
    std::vector<float> wv(kKvarnTileTokens * kKvarnHeadDim);
    kvarn128_read_window(win_k, win_v, kv_heads, phys, wk, wv);

    char path[1024];
    std::snprintf(path, sizeof(path), "%s/kvarn128_p1.dump", dir);
    FILE* f = std::fopen(path, "wb");
    if (f != nullptr) {
        const std::uint32_t magic = 0x3144564Bu, ver = 1, n = 2;
        std::fwrite(&magic, 4, 1, f);
        std::fwrite(&ver, 4, 1, f);
        std::fwrite(&n, 4, 1, f);
        kvarn128_write_kvd1(f, "L0-K-h0", 0, KB, CK, SK, kc, ks);
        kvarn128_write_kvd1(f, "L0-V-h0", 1, VB, CV, SV, vc, vs);
        std::fclose(f);
        std::fprintf(stderr, "ninfer: KVarN128 dump 写出 %s（KVD1 layout=1，pages %d/%d）\n", path,
                     page0, page1);
    } else {
        std::fprintf(stderr, "ninfer: KVarN128 dump 打不开 %s\n", path);
    }
    std::snprintf(path, sizeof(path), "%s/kvarn128_win.kvo", dir);
    f = std::fopen(path, "wb");
    if (f != nullptr) {
        const std::uint32_t magic = 0x314F564Bu, ver = 1, n = 2;
        std::fwrite(&magic, 4, 1, f);
        std::fwrite(&ver, 4, 1, f);
        std::fwrite(&n, 4, 1, f);
        kvarn128_write_kvo1_head(f, "L0-K-h0", 0, 1 /*已逆旋转*/, wk.data());
        kvarn128_write_kvo1_head(f, "L0-V-h0", 1, 1 /*已逆旋转*/, wv.data());
        std::fclose(f);
        std::fprintf(stderr, "ninfer: KVarN128 dump 写出 %s\n", path);
    } else {
        std::fprintf(stderr, "ninfer: KVarN128 dump 打不开 %s\n", path);
    }
    std::fflush(stderr);
}

} // namespace

// 与 ops/kernel/paged_kv_address.cuh 的寻址逐格相同 —— 单一真值来源的机器校验。
static_assert(kvarn_plane_element_offset(128, 4, /*page=*/3, /*head=*/2, /*page_off=*/5,
                                         /*slot=*/7) ==
                  static_cast<std::int64_t>(128) * kKvarnPageTokens *
                          (2 + static_cast<std::int64_t>(4) * 3) +
                      static_cast<std::int64_t>(128) * 5 + 7,
              "kvarn_layout.h 的页内寻址必须与引擎 paged_kv_element_offset 同式");
static_assert(kKvarnPageTokens == kPagedKVPageSize, "KVarN 的页粒度必须等于引擎的分页粒度");

int kvarn128_stage_slots() { return kKvarnPagesPerTile; }

// staging 现在**按层**在第一次 append 时分配（键 = 该层量化 K 平面基址），那次必然发生在
// CUDA Graph capture 之前（capture 期禁 malloc 的检查在 ensure_stage_for 里）。所以这里
// 不再需要"提前预热"；保留入口只为不破坏调用方（当前全树无调用）。
bool kvarn128_ensure_stage(int kv_heads, cudaStream_t stream) {
    (void)kv_heads;
    (void)stream;
    return true;
}

void kv_cache_append_kvarn128_launch(const void* k_bf16, const void* v_bf16,
                                     const std::int32_t* positions, KvarnTile128BlockTable table,
                                     unsigned char* k_code, void* k_scale, unsigned char* v_code,
                                     void* v_scale, int tokens, int kv_heads, int pages,
                                     cudaStream_t stream) {
    if (tokens <= 0) { return; }
    if (kv_heads != 4 && kv_heads != 2) {
        std::fprintf(stderr, "ninfer: KVarN128 append 只支持 KVHeads 4/2（收到 %d）\n", kv_heads);
        std::fflush(stderr);
        throw std::invalid_argument("kv_cache_append: KVarN128 unsupported KV head count");
    }
    if (positions == nullptr || k_code == nullptr || k_scale == nullptr || v_code == nullptr ||
        v_scale == nullptr) {
        std::fprintf(stderr, "ninfer: KVarN128 append 实参缺失（positions/plane 有空指针）\n");
        std::fflush(stderr);
        throw std::invalid_argument("kv_cache_append: KVarN128 missing arguments");
    }
    // 本层的 staging（键 = k_code，与窗口同键）：stage 是"这块 tile 的旋转域累积值"，
    // 不是"本次 append 的暂存" —— encode 封口时读满 128 槽、未封口窗口读 [0,tt_end)，
    // 两处都需要 [0,tt_begin) 那段上一次 append 写进去的值。
    __half*     stage = nullptr;
    const char* why   = ensure_stage_for(k_code, kv_heads, stream, &stage);
    if (why != nullptr) {
        std::fprintf(stderr, "ninfer: KVarN128 append 前置条件不满足: %s\n", why);
        std::fflush(stderr);
        throw std::invalid_argument("kv_cache_append: KVarN128 staging unavailable");
    }
    ensure_encode_opt_in();

    const auto* kp = static_cast<const __nv_bfloat16*>(k_bf16);
    const auto* vp = static_cast<const __nv_bfloat16*>(v_bf16);
    auto* kc = static_cast<std::uint8_t*>(k_code);
    auto* ks = static_cast<__half*>(k_scale);
    auto* vc = static_cast<std::uint8_t*>(v_code);
    auto* vs = static_cast<__half*>(v_scale);

    // 消费端窗口（bf16，同页表/同物理页）。拿不到就**大声失败**：宁可报错也不静默读错。
    void* win_k = nullptr;
    void* win_v = nullptr;
    if (!kvarn128_window(k_code, pages, kv_heads, stream, &win_k, &win_v)) {
        std::fprintf(stderr, "ninfer: KVarN128 消费端窗口不可用 ⇒ 拒绝 append\n");
        std::fflush(stderr);
        throw std::invalid_argument("kv_cache_append: KVarN128 consumer window unavailable");
    }
    auto* wk = static_cast<__nv_bfloat16*>(win_k);
    auto* wv = static_cast<__nv_bfloat16*>(win_v);

    // host 不读 positions ⇒ 只能给一个上界：tokens 个**连续**位置最多跨 tokens/128 + 2 个 tile。
    const int  tiles_span = tokens / kKvarnTileTokens + 2;
    const int  n          = tokens;
    const dim3 grid(static_cast<unsigned>(kv_heads));
    // 每个 tile：rotate(slot) → encode(读同一 slot) → window(读码流平面或 stage)。
    // 必须逐 tile 成对做：stage 只有 2 个 slot，先把所有 tile 的 rotate 做完会被 T+2 覆盖。
    for (int index = 0; index < tiles_span; ++index) {
        if (kv_heads == 4) {
            kvarn128_rotate_tile_kernel<<<grid, kThreads, 0, stream>>>(kp, vp, positions, n, index, 4,
                                                                       stage);
            CUDA_CHECK(cudaGetLastError());
            kvarn128_encode_kernel<4, 4><<<grid, kThreads, kDynamicSmemBytes, stream>>>(
                stage, positions, n, index, kc, ks, vc, vs, table, kKvarnIters);
            CUDA_CHECK(cudaGetLastError());
            kvarn128_window_kernel<4, 4><<<grid, kThreads, 0, stream>>>(
                kc, ks, vc, vs, stage, positions, n, index, table, wk, wv);
        } else {
            kvarn128_rotate_tile_kernel<<<grid, kThreads, 0, stream>>>(kp, vp, positions, n, index, 2,
                                                                       stage);
            CUDA_CHECK(cudaGetLastError());
            kvarn128_encode_kernel<4, 2><<<grid, kThreads, kDynamicSmemBytes, stream>>>(
                stage, positions, n, index, kc, ks, vc, vs, table, kKvarnIters);
            CUDA_CHECK(cudaGetLastError());
            kvarn128_window_kernel<4, 2><<<grid, kThreads, 0, stream>>>(
                kc, ks, vc, vs, stage, positions, n, index, table, wk, wv);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // 对拍 dump（默认关闭；KVARN128_DUMP_DIR 打开时只写一次，不改数值路径）
    kvarn128_dump_probe(k_code, k_scale, v_code, v_scale, table, kv_heads, wk, wv, stage, positions,
                        tokens, stream);
}

void kvarn128_decode_tile(const void* k_code, const void* k_scale, const void* v_code,
                          const void* v_scale, KvarnTile128BlockTable table, int tile, int head,
                          int kv_heads, void* out_k, void* out_v, cudaStream_t stream) {
    (void)head;
    if (kv_heads == 4) {
        kvarn128_decode_kernel<4, 4><<<1, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(k_code), static_cast<const __half*>(k_scale),
            static_cast<const std::uint8_t*>(v_code), static_cast<const __half*>(v_scale), table,
            tile, static_cast<__half*>(out_k), static_cast<__half*>(out_v));
    } else {
        kvarn128_decode_kernel<4, 2><<<1, kThreads, 0, stream>>>(
            static_cast<const std::uint8_t*>(k_code), static_cast<const __half*>(k_scale),
            static_cast<const std::uint8_t*>(v_code), static_cast<const __half*>(v_scale), table,
            tile, static_cast<__half*>(out_k), static_cast<__half*>(out_v));
    }
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops

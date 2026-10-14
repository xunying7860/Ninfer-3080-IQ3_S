#pragma once

// ===========================================================================
// kvarn128_launch.h —— G128 KVarN 的 host 侧入口（append 三路 + 解码对拍）
//
// **硬约束（G64 旧路线真金白银换来的）**：
//   * block table 只允许在 device 上解析。引擎的 PagedKVBatchMetadata::block_table()
//     （ops/kernel/paged_kv_address.cuh:37-39）是 __device__-only 并且要读 table_rows[0]
//     （设备地址）；host 侧调用它 = 编译器给那条路径生成 exit(1) 陷阱，现象是
//     "无信号、无异常文本、atexit 跑了、main 没返回、rc=1"。KvarnTile128BlockTable::resolve()
//     只给 __device__，host 误用直接编译错。
//   * positions 是**绝对位置**向量（kv_cache_append.cpp: require_shape(positions, tokens,1,1,1)）。
//   * 尾块（本组未攒满）必须走 B 路：只写 stage、不碰 metadata。
// ===========================================================================

#include <cstdint>
#include <cuda_runtime.h>

namespace ninfer::ops {

struct KvarnTile128BlockTable {
    const std::int32_t* direct       = nullptr; // PagedKVDirectMetadata::table（已解析）
    const std::int32_t* tables       = nullptr; // PagedKVBatchMetadata::tables
    const std::int32_t* table_rows   = nullptr; // PagedKVBatchMetadata::table_rows（设备数组）
    std::int32_t        table_stride = 0;

    KvarnTile128BlockTable() = default;
    __host__ __device__ KvarnTile128BlockTable(const std::int32_t* table) : direct(table) {}

    // 只允许 device 调用（host 调用会读设备地址；故意不给 host 版本，让误用变成编译错误）。
    __device__ __forceinline__ const std::int32_t* resolve() const {
        return direct != nullptr
                   ? direct
                   : tables + static_cast<std::int64_t>(table_rows[0]) * table_stride;
    }
};

// 引擎侧唯一入口（ops/kv_cache/append/launch.cu 调用）。四个 plane 都是整块基底指针，
// 页内寻址由 kvarn_layout.h 的槽位映射算。pages = k_pages.ne[3]（物理页数）。
void kv_cache_append_kvarn128_launch(const void* k_bf16, const void* v_bf16,
                                     const std::int32_t* positions, KvarnTile128BlockTable table,
                                     unsigned char* k_code, void* k_scale, unsigned char* v_code,
                                     void* v_scale, int tokens, int kv_heads, int pages,
                                     cudaStream_t stream);

// staging（旋转域 fp16）现在**按层**（键 = 量化 K 平面基址，与窗口同键）在第一次 append 时
// 分配，那次必然在 CUDA Graph capture 之前（capture 期禁 malloc 的检查在分配点）。
// 本入口保留为兼容 no-op（当前全树无调用）。
bool kvarn128_ensure_stage(int kv_heads, cudaStream_t stream);
// 当前 staging 的 slot 数（诊断用）。
int kvarn128_stage_slots();

// 把某个 tile 解码成**旋转域** fp16 [128 token][256 dim]（out_k/out_v 各 128*256 个 half）。
// 与 kv_cache_append 共用 kvarn_layout.h 的槽位映射；这是"record 解码 == 主机反量化"对拍的
// 唯一解码实现，也是后续融合读核（消费端路线 A）的参考实现。
void kvarn128_decode_tile(const void* k_code, const void* k_scale, const void* v_code,
                          const void* v_scale, KvarnTile128BlockTable table, int tile, int head,
                          int kv_heads, void* out_k, void* out_v, cudaStream_t stream);

// ---------------------------------------------------------------------------
// 消费端窗口（bf16，与量化平面同页表/同物理页）
//
// 量化平面读不了现成内核（U8 码 + 跨 token 打包的轴向量），所以 append 时把每个已封口 tile
// **真解码**（走 kvarn128_decode_plane）成 bf16 写进一个"窗口"平面：窗口的页表/物理页与量化
// 缓存完全一致，因此任何 bf16 消费核（prompt/bf16、small_t/bf16）可以直接读它。
// 未封口 tile 的 token 仍只有 stage 里的旋转域原值（KVarN 的量化只在 tile 粒度发生），
// 这一步按同样方式解旋转后写入窗口 —— 这是 tile 粒度量化的固有语义，不是绕过 codec。
// ---------------------------------------------------------------------------
// 按量化 K 平面基址（= 每层唯一）取/建窗口；返回 false 时已 fprintf 说明原因。
bool kvarn128_window(const void* k_code, int pages, int kv_heads, cudaStream_t stream, void** win_k,
                     void** win_v);
// 已注册的窗口数（诊断用）。
int kvarn128_window_count();
// 窗口每 token 每 head 的 bf16 元素数（= kKvarnHeadDim）。
int kvarn128_window_leading();

// ---------------------------------------------------------------------------
// 窗口的**惰性物理后备**（默认开；NINFER_KVARN128_WINDOW_FULL=1 退回整块 cudaMalloc）。
// 窗口虚拟地址一次预留，物理块按 KV 页实际落地量分批提交：
//   * kvarn128_window_commit_tokens —— 把窗口覆盖量提到「tokens 个位置所需页数 + 余量」；
//     由引擎在序列覆盖量扩展时调用（ProgramImpl::ensure_sequence_kv_mapped），必须发生在
//     读到这些页的 attention 之前。提交失败会 fprintf+fflush 后 throw（不许静默留洞）。
//   * kvarn128_window_release_backing —— 释放全部窗口的物理后备（虚拟地址保留）；启动
//     热身/抓图结束后调用，空闲态窗口不占显存。调用后窗口内容未定义。
// ---------------------------------------------------------------------------
void kvarn128_window_commit_tokens(int tokens);
// 按**实际落地的物理页号**提交窗口后备。逻辑页数不够用：第二个请求拿到的新地址空间，其物理页号
// 可能整段落在「逻辑页数」推出来的已提交区之外（实机 = 第二次整窗 prefill 报 illegal address）。
// pages/count 为 DeviceKVPagePool 的物理页号数组（block table 里写的就是它们）。
void kvarn128_window_commit_physical_pages(const std::int32_t* pages, int count);
void kvarn128_window_release_backing();

} // namespace ninfer::ops

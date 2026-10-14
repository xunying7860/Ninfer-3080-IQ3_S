#pragma once

#include <cstdint>

// ===========================================================================
// KVarN G128 —— 几何与字节账的**单一真值来源**（single source of truth）
//
// 本文件是 G128 路线的唯一几何账：record 字节、页内 leading 份额、scale 页槽位映射、
// 页内元素寻址。encode 核、decode 核、profile 注册、字节账对拍**都只能从这里取**，
// 任何一处各写一份都会让"能跑但数值错"重新变成可能（这正是 G64 路线的头号静默风险）。
//
// 与上游 beellama.cpp 的对应：
//   tile（组）   = 128 token（upstream token_group / KVAR_N_DIM）
//   一个 record  = 一个 slice 的 128×128（rows×cols 随 K/V 语义轴互换）
//   D256         = 2 个 slice（head_slices=2），K/V 各 2 条 record
//
// 与上游的**唯一刻意偏离**：ninfer 的分页核固定按「页 = 64 token」组织平面
// （ops/kernel/paged_kv_address.cuh 的 paged_kv_element_offset），所以这里取 **P1 页自足**：
// 一个 tile = 2 页，维轴（每 tile 一份）在两个页里**各存一份**换取"任意一页可独立解码"，
// 代价 = 每 tile 1536 B（12 B/token）。紧凑 record 口径与 P1 分配口径都在此文件给出，谁都能对。
//
// 本文件刻意只依赖 <cstdint>（core/ 与独立对拍 harness 都要能包含它），所以 device 侧
// 用得到的 constexpr 函数都挂 NINFER_KVARN_HD（host+device），不把 cuda_runtime 拖进来。
// ===========================================================================

#if defined(__CUDACC__)
#define NINFER_KVARN_HD __host__ __device__
#else
#define NINFER_KVARN_HD
#endif

namespace ninfer::ops {

inline constexpr int kKvarnTileTokens   = 128; // 组大小 = upstream token_group
inline constexpr int kKvarnPageTokens   = 64;  // ninfer 分页粒度（core/paged_kv_cache.h: kPagedKVPageSize）
inline constexpr int kKvarnPagesPerTile = 2;   // 128 / 64
inline constexpr int kKvarnSlices       = 2;   // D256 / 128
inline constexpr int kKvarnSliceDim     = 128;
inline constexpr int kKvarnHeadDim      = 256;
// 迭代数：上游 KVAR_N_OP_PARAM_ITERS（8 或 16 都无损）。取 16 = gate 2 实跑口径；
// 真实数据 best_iter=0，迭代数对真实数据的 record 无影响（oracle 交叉校验已验证）。
inline constexpr int kKvarnIters = 16;

// 维轴每 token 的槽位数 = 128 维 / 64 token = 2（**与码位宽无关**，位宽只改码流长度）。
inline constexpr int kKvarnAxisPerToken = kKvarnSliceDim / kKvarnPageTokens;

// ---- record 口径（upstream make_record_layout(record_dim=128).record_bytes × head_slices）----
// record_bytes = rows*cols*bits/8 + (2*rows + cols)*2，rows=cols=KVarN_DIM=128
NINFER_KVARN_HD constexpr int kvarn_record_bytes(int bits) {
    return kKvarnSliceDim * kKvarnSliceDim * bits / 8 + (2 * kKvarnSliceDim + kKvarnSliceDim) * 2;
}
// 一个 tile（128 token × 1 KV head）的紧凑 record 账：K/V 各 2 条 record。
NINFER_KVARN_HD constexpr int kvarn_compact_tile_bytes(int k_bits, int v_bits) {
    return kKvarnSlices * (kvarn_record_bytes(k_bits) + kvarn_record_bytes(v_bits));
}
// 每 token 每 head 的紧凑账 = 32*(Kb+Vb) + 24（与阶段 1 oracle 口径②的 36 格逐格相同）。
NINFER_KVARN_HD constexpr int kvarn_compact_bytes_per_token(int k_bits, int v_bits) {
    return kvarn_compact_tile_bytes(k_bits, v_bits) / kKvarnTileTokens;
}

// ---- 码流 leading：每 token 每 head 的字节数 = 256 维 × bits / 8 ----
NINFER_KVARN_HD constexpr int kvarn_code_leading(int bits) { return kKvarnHeadDim * bits / 8; }

// ---- P1 页自足方案：scale 平面 leading（每 token 每 head 的 FP16 元素数，全是推导值）----
// K：维轴 (scale_axis + zp_axis) 2 slice × 128 维 × 2 = 512 项，按每页 64 token 摊 ⇒ 每 token 8；
//    列因子 other 2 slice ⇒ 每 token 2；合计 10。
// V：维轴 other 2 slice × 128 维 = 256 项摊到 64 token ⇒ 每 token 4；
//    token 轴 (scale_axis + zp_axis) 2 slice × 2 ⇒ 每 token 4；合计 8。
inline constexpr int kKvarnKScaleLeading =
    kKvarnSlices * kKvarnAxisPerToken * 2 /*维轴 scale+zp 摊到每页 64 token*/ + kKvarnSlices
    /*列因子*/;
inline constexpr int kKvarnVScaleLeading =
    kKvarnSlices * kKvarnAxisPerToken /*维轴 other*/ + kKvarnSlices * 2 /*token 轴 scale+zp*/;
static_assert(kKvarnKScaleLeading == 10, "K scale 页 leading 推导错误");
static_assert(kKvarnVScaleLeading == 8, "V scale 页 leading 推导错误");

// ---- 实际分配的 tile 字节（4 个 plane × 2 页）----
NINFER_KVARN_HD constexpr int kvarn_plane_tile_bytes(int k_bits, int v_bits) {
    return kKvarnPagesPerTile * kKvarnPageTokens *
           (kvarn_code_leading(k_bits) + kvarn_code_leading(v_bits) + 2 * kKvarnKScaleLeading +
            2 * kKvarnVScaleLeading);
}
NINFER_KVARN_HD constexpr int kvarn_plane_bytes_per_token(int k_bits, int v_bits) {
    return kvarn_plane_tile_bytes(k_bits, v_bits) / kKvarnTileTokens;
}

// ---- 页内元素寻址（与 ops/kernel/paged_kv_address.cuh 的 paged_kv_page_head_offset 同式）----
// 平面形状 = {leading_extent, 64, kv_heads, pages}，页内是 [token 0..63][slot 0..leading)。
// 本文件刻意不 include 引擎头，便于独立对拍；.cu 里有 static_assert 钉住两者相等。
NINFER_KVARN_HD constexpr std::int64_t kvarn_plane_element_offset(int leading_extent,
                                                                  int head_extent, int page,
                                                                  int head, int page_off,
                                                                  int slot) {
    return static_cast<std::int64_t>(leading_extent) * kKvarnPageTokens *
               (static_cast<std::int64_t>(head) + static_cast<std::int64_t>(head_extent) * page) +
           static_cast<std::int64_t>(leading_extent) * page_off + slot;
}

// ---- 槽位分段（列）—— P1 定案（`P1几何与偏移表-20261013.md` §3）----
//   页 = 64 token 行 × leading 列。**维轴条目按全局维顺序每行 4 条填满 64 行**：
//   全局维 d（0..255）的条目在 row = d/4、col = d%4。
//   K：col 0..3 = scale_axis[d] / 4..7 = zp_axis[d] / 8..9 = other（列 8+slice，行 = 页内 token）。
//   V：col 0..3 = other[d]（列因子）/ 4..5 = scale_axis（列 4+slice，行 = 页内 token）/ 6..7 = zp_axis。
//   （旧版把维轴摊成 row = dim_local/2 + col = slice*2+parity —— 与 §3 的定案表逐格不符，已修。）
inline constexpr int kKvarnKScaleSlotScale = 0;
inline constexpr int kKvarnKScaleSlotZp    = 4;
inline constexpr int kKvarnKScaleSlotOther = 8;
inline constexpr int kKvarnVScaleSlotOther = 0;
inline constexpr int kKvarnVScaleSlotScale = 4;
inline constexpr int kKvarnVScaleSlotZp    = 6;

NINFER_KVARN_HD constexpr int kvarn_axis_row(int dim_global) { return dim_global >> 2; }
NINFER_KVARN_HD constexpr int kvarn_axis_col(int dim_global) { return dim_global & 3; }
NINFER_KVARN_HD constexpr int kvarn_k_scale_col_scale(int dim_global) {
    return kKvarnKScaleSlotScale + kvarn_axis_col(dim_global);
}
NINFER_KVARN_HD constexpr int kvarn_k_scale_col_zp(int dim_global) {
    return kKvarnKScaleSlotZp + kvarn_axis_col(dim_global);
}
NINFER_KVARN_HD constexpr int kvarn_k_scale_col_other(int slice) {
    return kKvarnKScaleSlotOther + slice;
}
NINFER_KVARN_HD constexpr int kvarn_v_scale_col_other(int dim_global) {
    return kKvarnVScaleSlotOther + kvarn_axis_col(dim_global);
}
NINFER_KVARN_HD constexpr int kvarn_v_scale_col_scale(int slice) {
    return kKvarnVScaleSlotScale + slice;
}
NINFER_KVARN_HD constexpr int kvarn_v_scale_col_zp(int slice) {
    return kKvarnVScaleSlotZp + slice;
}

// ---- 位置 → tile / 页映射（1 tile = 2 个**逻辑**页；物理页由 block table 各自映射，不必相邻）----
NINFER_KVARN_HD constexpr int kvarn_tile_of(int position) { return position / kKvarnTileTokens; }
NINFER_KVARN_HD constexpr int kvarn_tile_base(int position) {
    return kvarn_tile_of(position) * kKvarnTileTokens;
}
NINFER_KVARN_HD constexpr int kvarn_page_of_token(int tile_token) {
    return tile_token / kKvarnPageTokens; // 0 或 1
}
NINFER_KVARN_HD constexpr int kvarn_page_off(int tile_token) {
    return tile_token % kKvarnPageTokens;
}
// 逻辑页号（block table 的索引）= 2*tile + 页内序。
NINFER_KVARN_HD constexpr int kvarn_logical_page(int position) {
    return 2 * kvarn_tile_of(position) + kvarn_page_of_token(position % kKvarnTileTokens);
}

} // namespace ninfer::ops

#undef NINFER_KVARN_HD

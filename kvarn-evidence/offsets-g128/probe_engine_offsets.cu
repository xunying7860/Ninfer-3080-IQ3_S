// probe_engine_offsets.cu —— 用**引擎真头**（只读 include）在 RTX 3080 上求值 P1 的 leading/偏移，
//                           与 offsets_g128_check.cpp 的 CPU 侧表逐格对拍。
// ---------------------------------------------------------------------------
// include 的引擎头（只读，未修改任何引擎文件）：
//   src/ops/kernel/paged_kv_address.cuh  —— paged_kv_element_offset<Leading,Heads>（:47-61）、
//                                            paged_kv_physical_page（:42-45）、kPagedKVPageShift（:9）
//   src/core/paged_kv_cache.h            —— kPagedKVPageSize=64（:19）、KVPlaneGeometry::alignment=256（:53）
// 编译：nvcc -std=c++17 -arch=sm_86 -I<engine>/src -o probe_engine_offsets probe_engine_offsets.cu
// 运行：需 GPU（--gpus all）
// ===========================================================================
#include "ops/kernel/paged_kv_address.cuh"

#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>

using namespace ninfer;
using namespace ninfer::ops;

static int g_fail = 0;

// CPU 侧的「照抄表」（与 offsets_g128_check.cpp 同一算式，逐项写开便于核对）
static std::int64_t cpu_offset(int leading, int heads, int page, int head, int row, int col) {
    return (std::int64_t)leading * kPagedKVPageSize *
               ((std::int64_t)head + (std::int64_t)heads * page) +
           (std::int64_t)leading * row + col;
}

// 用引擎模板实例在**设备**上求 offset（模板参数 = 本档 P1 的 leading）
template <int Leading, int Heads>
__global__ void tpl_probe(std::int64_t* out, int n_page, int n_row) {
    for (int p = 0; p < n_page; ++p) {
        for (int h = 0; h < Heads; ++h) {
            for (int r = 0; r < n_row; ++r) {
                out[(p * Heads + h) * n_row + r] = paged_kv_element_offset<Leading, Heads>(p, h, r, 0);
            }
        }
    }
}

template <int Leading, int Heads>
static void check_one(const char* tag) {
    const int n_page = 2, n_row = kPagedKVPageSize;
    std::vector<std::int64_t> host((size_t)n_page * Heads * n_row, -1), got(host.size(), -1);
    std::int64_t* dev = nullptr;
    cudaMalloc(&dev, host.size() * sizeof(std::int64_t));
    tpl_probe<Leading, Heads><<<1, 1>>>(dev, n_page, n_row);
    cudaDeviceSynchronize();
    cudaMemcpy(got.data(), dev, got.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    for (int p = 0; p < n_page; ++p)
        for (int h = 0; h < Heads; ++h)
            for (int r = 0; r < n_row; ++r)
                host[(size_t)(p * Heads + h) * n_row + r] = cpu_offset(Leading, Heads, p, h, r, 0);
    int bad = 0;
    for (size_t i = 0; i < host.size(); ++i) { if (host[i] != got[i]) { ++bad; } }
    g_fail += bad;
    std::printf("[tpl  ] %-26s Leading=%-4d Heads=%d  引擎模板 vs CPU 表：%zu 格，不一致=%d\n",
                tag, Leading, Heads, host.size(), bad);
}

// 「1 tile = 2 page，P1 页自足 ⇒ 消费核不需要 sibling」的寻址验证：
// 用引擎的第二个重载 paged_kv_element_offset(block_table, head, position, leading)，
// 直接确认地址只由 position 自己的物理页决定，公式里没有 sibling 项。
template <int Leading, int Heads>
__global__ void sib_probe(const std::int32_t* bt, const std::int32_t* pos, std::int64_t* out, int n) {
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) { out[i] = paged_kv_element_offset<Leading, Heads>(bt, /*head=*/0, pos[i], /*leading=*/0); }
}

int main() {
    cudaDeviceProp prop{};
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
        std::printf("[FAIL] 没有可用 CUDA 设备（本探针需要 --gpus all）\n");
        return 2;
    }
    std::printf("================ 引擎真头 offset 探针 ================\n");
    std::printf("[device] %s sm_%d%d | kPagedKVPageSize=%d | KVPlaneGeometry::alignment=%zu | page shift=%d\n",
                prop.name, prop.major, prop.minor, kPagedKVPageSize, (size_t)KVPlaneGeometry{}.alignment,
                kPagedKVPageShift);

    static_assert(kPagedKVPageSize == 64, "引擎页必须是 64 token");   // core/paged_kv_cache.h:19
    static_assert(kPagedKVPageShift == 6, "引擎页 shift 必须是 6");    // ops/kernel/paged_kv_address.cuh:9

    // 6 档 × 4 平面：模板参数逐个实例化为本档 P1 的 leading
    check_one< 64, 4>("code plane, kb/vb=2");        // 32*2
    check_one< 96, 4>("code plane, kb/vb=3");        // 32*3
    check_one<128, 4>("code plane, kb/vb=4");        // 32*4
    check_one<160, 4>("code plane, kb/vb=5");        // 32*5
    check_one<192, 4>("code plane, kb/vb=6");        // 32*6
    check_one<256, 4>("code plane, kb/vb=8");        // 32*8
    check_one< 10, 4>("k_scale 平面（10 half/行）");
    check_one<  8, 4>("v_scale 平面（8 half/行）");
    check_one<256, 2>("code plane kb=8, KV head=2"); // 换 head 数复核 head/page 步长

    {
        std::vector<std::int32_t> hb(8), hpos(128);
        for (int i = 0; i < 8; ++i) hb[i] = 100 + 2 * i;   // 成对（偶数起始）物理页
        for (int i = 0; i < 128; ++i) hpos[i] = i;         // 逻辑位置 0..127（= 1 tile）
        std::int32_t *db = nullptr, *dp = nullptr;
        std::int64_t* dout = nullptr;
        std::vector<std::int64_t> got(128, -1);
        cudaMalloc(&db, hb.size() * 4);
        cudaMalloc(&dp, hpos.size() * 4);
        cudaMalloc(&dout, hpos.size() * sizeof(std::int64_t));
        cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dp, hpos.data(), hpos.size() * 4, cudaMemcpyHostToDevice);
        const int Leading = 128, Heads = 4;               // kb=4 档
        sib_probe<Leading, Heads><<<2, 64>>>(db, dp, dout, (int)hpos.size());
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), dout, got.size() * sizeof(std::int64_t), cudaMemcpyDeviceToHost);
        int bad = 0;
        for (int i = 0; i < 128; ++i) {
            const int page = hb[i >> 6];
            const std::int64_t expect = cpu_offset(Leading, Heads, page, 0, i & 63, 0);
            if (expect != got[i]) { ++bad; }
        }
        g_fail += bad;
        std::printf("[sib  ] 引擎重载 <128,4>：128 个位置（1 tile = 2 页）不一致=%d\n", bad);
        std::printf("        公式 = Leading*64*(head + Heads*block_table[position>>6]) + Leading*(position&63) + leading\n");
        std::printf("        页内任一位置的地址只由**它自己那一页**（block_table[position>>6]）决定，"
                    "无 sibling 项 ⇒ P1 下消费核不需要 sibling\n");
        cudaFree(db); cudaFree(dp); cudaFree(dout);
    }

    std::printf("\n引擎模板 × CPU 表 全格核对：不一致总数 = %d\n", g_fail);
    std::printf("判定：%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}

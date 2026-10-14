// f16_check.cu —— 用 nvcc 的 __float2half_rn 逐位校验 bee_oracle.cpp 里的标量转换器
// 编译： nvcc -O2 -std=c++17 -o f16check f16_check.cu
// 跑：   ./f16check 10000000
// 输出：不一致条数 + 前若干条反例（hex）。0 即为逐位一致。
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_fp16.h>
#include <random>

// 与 bee_oracle.cpp 中同名函数完全相同的实现（复制粘贴，勿改动）
static uint16_t f32_to_f16_rn(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    const uint32_t sign = (u >> 16) & 0x8000u;
    const uint32_t exp = (u >> 23) & 0xFFu;
    const uint32_t m = u & 0x7FFFFFu;
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

int main(int argc, char ** argv) {
    const long n = argc > 1 ? atol(argv[1]) : 2000000L;
    std::mt19937_64 rng(0xC0FFEEull);
    std::uniform_int_distribution<uint32_t> bits(0, 0xFFFFFFFFu);
    long bad = 0, shown = 0;
    // 1) 纯随机位型（覆盖 nan/inf/次正规/溢出）
    for (long i = 0; i < n; ++i) {
        uint32_t u = bits(rng);
        float f; std::memcpy(&f, &u, 4);
        if (f != f) continue;                                  // NaN 的载荷位不保证一致，跳过
        const uint16_t a = f32_to_f16_rn(f);
        const uint16_t b = __half_as_ushort(__float2half_rn(f));
        if (a != b) { ++bad; if (shown < 12) { printf("随机 u=%08x mine=%04x cuda=%04x (f=%.9g)\n", u, a, b, f); ++shown; } }
    }
    // 2) 量化网格上的值（KV 里实际出现的量级：1e-8 .. 1e4）
    for (long i = 0; i < n; ++i) {
        const double mag = std::pow(10.0, -8.0 + 12.0 * (double)(rng() % 1000000) / 1000000.0);
        const double s = (rng() & 1) ? 1.0 : -1.0;
        const float f = (float)(s * mag * (1.0 + (double)(rng() % 1000) / 1000.0));
        if (!std::isfinite(f)) continue;
        const uint16_t a = f32_to_f16_rn(f);
        const uint16_t b = __half_as_ushort(__float2half_rn(f));
        if (a != b) { ++bad; if (shown < 12) { printf("量级 f=%.9g mine=%04x cuda=%04x\n", f, a, b); ++shown; } }
    }
    // 3) 半精度边界与舍入中点
    const float edge[] = {0.f, -0.f, 1.f, -1.f, 65504.f, 65519.99f, 65520.f, 65536.f, 1e30f,
                          5.960464477539063e-08f, 2.9802322387695312e-08f, 1.4901161193847656e-08f,
                          6.097555160522461e-05f, 6.097555160522461e-05f * 0.5f, 1e-8f, 0.1f, 6.103515625e-05f,
                          2049.f, 2050.f, 0.0001220703125f, 3.402823466e+38f};
    for (float f : edge) {
        const uint16_t a = f32_to_f16_rn(f), b = __half_as_ushort(__float2half_rn(f));
        printf("edge %-24.9g mine=%04x cuda=%04x %s\n", f, a, b, a == b ? "" : "  <-- MISMATCH");
        if (a != b) ++bad;
    }
    printf("\n总计不一致 = %ld / %ld\n", bad, 2 * n + (long)(sizeof(edge) / sizeof(edge[0])));
    return bad ? 1 : 0;
}

#define KVARN_TEST_SKIP_ROTATE
#include <math.h>
#include <cuda_bf16.h>
#include "ops/kv_cache/append/kvarn_prefill_kernel.cuh"
#include "ops/kv_cache/append/kvarn_launch.h"

#include "ops/kv_cache/append/sinkhorn_kernel.cuh"
#include <cstdio>
#include <string>
#include <cstdlib>
#include <vector>
#include <cmath>
using namespace ninfer::ops;

// 消费端解码核：每线程一个 token，用 kvarn_dequant_token 读回 256 维
template <int Bits, int Slots>
__global__ void dequant_probe_kernel(const unsigned char* codes, const unsigned char* scales, float* out) {
    extern __shared__ float dummy[]; (void)dummy; (void)out;
    const int t = threadIdx.x;
    float v[kKVarNHeadDim];
    kvarn_dequant_token<Bits, Slots, 4>(codes, scales, 0, 0, t, v);
    for (int d = 0; d < kKVarNHeadDim; ++d) out[(size_t)t * kKVarNHeadDim + d] = v[d];
}
// 读 dump_tiles.py 的 tile：首行 "R C bits mode"，第二块 = 已旋转的 [通道][token]，第三块 = 标准答案
static double sd_unbiased(const float* p, int n) {
    double s = 0, q = 0; for (int i = 0; i < n; ++i) { s += p[i]; q += (double)p[i] * p[i]; }
    double v = (q - s * s / n) / (n - 1); return sqrt(v < 0 ? 0 : v);
}
// 主机参考：照 proto_codec.cpp 的 log 域 Sinkhorn（无偏 std、钳制 [-0.3,10]、best-so-far）
static void host_sinkhorn(const float* in, int R, int C, int iters, std::vector<float>& bal,
                          std::vector<float>& ccol, std::vector<float>& crow) {
    std::vector<float> cur(in, in + (size_t)R * C), lc(C, 0.f), lr(R, 0.f), bc(C, 1.f), br(R, 1.f);
    auto imb = [&](const std::vector<float>& t) {
        double cmin = 1e30, cmax = 0, rmin = 1e30, rmax = 0;
        for (int c = 0; c < C; ++c) { std::vector<float> col(R); for (int r = 0; r < R; ++r) col[r] = t[(size_t)r * C + c];
            double sd = sd_unbiased(col.data(), R); if (sd < cmin) cmin = sd; if (sd > cmax) cmax = sd; }
        for (int r = 0; r < R; ++r) { double sd = sd_unbiased(&t[(size_t)r * C], C); if (sd < rmin) rmin = sd; if (sd > rmax) rmax = sd; }
        return cmax / (cmin < 1e-8 ? 1e-8 : cmin) + rmax / (rmin < 1e-8 ? 1e-8 : rmin);
    };
    double best = imb(cur);
    for (int it = 0; it < iters; ++it) {
        for (int c = 0; c < C; ++c) { std::vector<float> col(R); for (int r = 0; r < R; ++r) col[r] = cur[(size_t)r * C + c];
            double sd = sd_unbiased(col.data(), R); if (sd < 1e-3) sd = 1e-3; if (sd > 1e3) sd = 1e3;
            double ls = lc[c] + log((double)sd); if (ls < -0.3) ls = -0.3; if (ls > 10.0) ls = 10.0; lc[c] = (float)ls; }
        for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) cur[(size_t)r * C + c] = in[(size_t)r * C + c] / expf(lc[c]) / expf(lr[r]);
        for (int r = 0; r < R; ++r) { double sd = sd_unbiased(&cur[(size_t)r * C], C); if (sd < 1e-3) sd = 1e-3; if (sd > 1e3) sd = 1e3;
            double ls = lr[r] + log((double)sd); if (ls < -0.3) ls = -0.3; if (ls > 10.0) ls = 10.0; lr[r] = (float)ls; }
        for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) cur[(size_t)r * C + c] = in[(size_t)r * C + c] / expf(lc[c]) / expf(lr[r]);
        double now = imb(cur);
        if (now <= best) { best = now; for (int c = 0; c < C; ++c) bc[c] = expf(lc[c]); for (int r = 0; r < R; ++r) br[r] = expf(lr[r]); }
    }
    bal.assign(in, in + (size_t)R * C);
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c) bal[(size_t)r * C + c] = in[(size_t)r * C + c] / (bc[c] * br[r]);
    ccol = bc; crow = br;
}

// 主机逐行 RTN：scale/zp 先经 fp16 表示再反量化（与 proto_codec / kernel 同口径）
static void host_rtn(const std::vector<float>& bal, int R, int C, int levels, std::vector<float>& deq_bal,
                     std::vector<unsigned char>& codes) {
    codes.assign((size_t)R * C, 0);
    deq_bal.assign((size_t)R * C, 0.f);
    for (int r = 0; r < R; ++r) {
        const float* p = &bal[(size_t)r * C];
        float lo = p[0], hi = p[0];
        for (int c = 1; c < C; ++c) { if (p[c] < lo) lo = p[c]; if (p[c] > hi) hi = p[c]; }
        float sc = (hi - lo) / (float)levels; if (sc < 1e-10f) sc = 1e-10f;
        const float scf = __half2float(__float2half_rn(sc));
        const float zpf = __half2float(__float2half_rn(lo));
        for (int c = 0; c < C; ++c) {
            float q = (float)(long long)floor((p[c] - zpf) / scf + 0.5f);   // half-to-even 近似
            if (q < 0) q = 0; if (q > (float)levels) q = (float)levels;
            codes[(size_t)r * C + c] = (unsigned char)q;
            deq_bal[(size_t)r * C + c] = scf * q + zpf;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: harness tile.txt\n"); return 2; }
    FILE* fp = std::fopen(argv[1], "r");
    if (!fp) { std::printf("cannot open %s\n", argv[1]); return 2; }
    int R, C, bits, mode;
    if (std::fscanf(fp, "%d %d %d %d", &R, &C, &bits, &mode) != 4) return 2;
    const int n = R * C;
    std::vector<float> in(n), ref(n);
    for (int i = 0; i < n; ++i) if (std::fscanf(fp, "%f", &in[i]) != 1) return 2;
    for (int i = 0; i < n; ++i) if (std::fscanf(fp, "%f", &ref[i]) != 1) return 2;
    std::fclose(fp);
    // staging 是 [token][dim]：把 [通道][token] 转置回去
    std::vector<__half> h_stage(n);
    for (int c = 0; c < C; ++c) for (int d = 0; d < R; ++d) h_stage[c * R + d] = __float2half_rn(in[d * C + c]);
    const int code_bytes = C * (kKVarNHeadDim * bits / 8);
    const int slot_half  = C * 9;                       // 1 token 9 槽
    std::vector<unsigned char> h_codes(code_bytes, 0), h_scales(slot_half * 2, 0);
    std::vector<float> h_bal((size_t)R * C, 0.f);
    // 主机参考全链：Sinkhorn + 逐行 RTN（fp16 表示后）+ 列因子还原
    std::vector<float> h_bal_ref; std::vector<float> cc, cr; host_sinkhorn(in.data(), R, C, 16, h_bal_ref, cc, cr);
    __half* d_stage; unsigned char *d_codes, *d_scales;
    cudaMalloc(&d_stage, n * 2); cudaMalloc(&d_codes, code_bytes); cudaMalloc(&d_scales, slot_half * 2);
    cudaMemcpy(d_stage, h_stage.data(), n * 2, cudaMemcpyHostToDevice);
    cudaMemset(d_codes, 0, code_bytes); cudaMemset(d_scales, 0, slot_half * 2);
    const int dyn = kKVarNHeadDim * kKVarNTileTokens * (int)sizeof(float);
    cudaFuncSetAttribute(kv_cache_append_kvarn_kernel<2, true, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
    cudaFuncSetAttribute(kv_cache_append_kvarn_kernel<4, true, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn);
    float* d_bal = nullptr; cudaMalloc(&d_bal, n * sizeof(float));
    if (bits == 2) kv_cache_append_kvarn_kernel<2, true, 4><<<1, 256, dyn>>>(d_stage, d_codes, d_scales, 0, d_bal);
    else           kv_cache_append_kvarn_kernel<4, true, 4><<<1, 256, dyn>>>(d_stage, d_codes, d_scales, 0, d_bal);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { std::printf("kernel error: %s\n", cudaGetErrorString(err)); return 3; }
    std::vector<unsigned char> codes(code_bytes); std::vector<__half> scales(slot_half);
    cudaMemcpy(h_bal.data(), d_bal, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(codes.data(), d_codes, code_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(scales.data(), d_scales, slot_half * 2, cudaMemcpyDeviceToHost);
    // 按 §7 折叠口径解码：deq = (q·scaleF + zpF) · colF（不再乘 s_row）
    std::vector<float> deq(n);
    const int mask = (1 << bits) - 1;
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (int d = 0; d < R; ++d) {
        const int ts = d >> 2, off = d & 3;
        const float sc = __half2float(scales[ts * 9 + off]);
        const float zp = __half2float(scales[ts * 9 + 4 + off]);
        for (int c = 0; c < C; ++c) {
            const int byte_i = c * (kKVarNHeadDim * bits / 8) + (bits == 4 ? (d >> 1) : (d >> 2));
            const int q = (codes[byte_i] >> ((bits == 4 ? (d & 1) : (d & 3)) * bits)) & mask;
            const float col = __half2float(scales[c * 9 + 8]);
            const float deq_v = (q * sc + zp) * col;
            deq[(size_t)d * C + c] = deq_v;
            const double a = deq_v, b = ref[d * C + c];
            dot += a * b; na += a * a; nb += b * b;
            const double dd = std::fabs(a - b); if (dd > mx) mx = dd;
        }
    }
    auto cos_of = [&](const std::vector<float>& a, const std::vector<float>& b) {
        double dd = 0, x = 0, y = 0; for (int i = 0; i < n; ++i) { dd += (double)a[i] * b[i]; x += (double)a[i] * a[i]; y += (double)b[i] * b[i]; }
        return dd / (sqrt(x) * sqrt(y) + 1e-30); };
    std::vector<float> h_deq_bal; std::vector<unsigned char> h_codes_out;
    host_rtn(h_bal_ref, R, C, (1 << bits) - 1, h_deq_bal, h_codes_out);
    std::vector<float> host_full(n);
    for (int r = 0; r < R; ++r) for (int c = 0; c < C; ++c)
        host_full[(size_t)r * C + c] = h_deq_bal[(size_t)r * C + c] * cc[c] * cr[r];
    std::printf("%-42s bits=%d | cos(host全链,ref)=%.7f  cos(host全链,kernel解码)=%.7f  cos(kernbal,hostbal)=%.7f\n",
                argv[1], bits, cos_of(host_full, ref), cos_of(host_full, deq), cos_of(h_bal, h_bal_ref));

    // ---- 跨 page / 跨 KV head 的 plane 偏移验证：拆档跑，每档独立缓冲 + 独立查错（粘性错误会污染后续读数）----
    for (int GH = 1; GH <= 4; GH <<= 1) {
        const int pg = 3, hd = (GH > 2) ? 2 : 0;                       // GH=1/2 查 head 0；GH=4 查 head 2
        const size_t bc_codes = (size_t)code_bytes * 4 * 4;            // Leading*64 已含每页 64 token
        const size_t bc_scale = (size_t)slot_half * 4 * 4;
        unsigned char* Xc; unsigned char* Xs; __half* Xst;
        cudaMalloc(&Xc, bc_codes); cudaMalloc(&Xs, bc_scale * 2); cudaMalloc(&Xst, (size_t)4 * n * 2);
        cudaMemset(Xc, 0, bc_codes); cudaMemset(Xs, 0, bc_scale * 2); cudaMemset(Xst, 0, (size_t)4 * n * 2);
        for (int hh = 0; hh < 4; ++hh)
            cudaMemcpy((char*)Xst + (size_t)hh * n * 2, h_stage.data(), n * 2, cudaMemcpyHostToDevice);
        if (bits == 2) kv_cache_append_kvarn_kernel<2, true, 4><<<dim3(1, (unsigned)GH), 256, dyn>>>(Xst, Xc, Xs, pg, nullptr);
        else           kv_cache_append_kvarn_kernel<4, true, 4><<<dim3(1, (unsigned)GH), 256, dyn>>>(Xst, Xc, Xs, pg, nullptr);
        cudaError_t ge = cudaDeviceSynchronize();
        std::string msg = "ok";
        std::vector<float> deqX(n, 0.f);
        if (ge != cudaSuccess) {
            msg = cudaGetErrorString(ge); cudaGetLastError();           // 清粘性错误，别让后续读数被污染
        } else {
            std::vector<unsigned char> xc(bc_codes); std::vector<__half> xs(bc_scale);
            cudaMemcpy(xc.data(), Xc, bc_codes, cudaMemcpyDeviceToHost);
            cudaMemcpy(xs.data(), Xs, bc_scale * 2, cudaMemcpyDeviceToHost);
            const size_t cbase = (size_t)code_bytes * (hd + 4 * pg), sbase = (size_t)slot_half * (hd + 4 * pg);
            for (int d = 0; d < R; ++d) {
                const float sc = __half2float(xs[sbase + (size_t)(d >> 2) * 9 + (d & 3)]);
                const float zp = __half2float(xs[sbase + (size_t)(d >> 2) * 9 + 4 + (d & 3)]);
                for (int c = 0; c < C; ++c) {
                    const size_t bi = cbase + (size_t)c * (kKVarNHeadDim * bits / 8) + (bits == 4 ? (d >> 1) : (d >> 2));
                    const int q = (xc[bi] >> ((bits == 4 ? (d & 1) : (d & 3)) * bits)) & mask;
                    deqX[(size_t)d * C + c] = (q * sc + zp) * __half2float(xs[sbase + (size_t)c * 9 + 8]);
                }
            }
            msg = "cos=" + [&]{ double dd=0,x=0,y=0; for (int i=0;i<n;++i){dd+=(double)deqX[i]*host_full[i];x+=(double)deqX[i]*deqX[i];y+=(double)host_full[i]*host_full[i];} char bf[32]; std::snprintf(bf,32,"%.7f",dd/(std::sqrt(x)*std::sqrt(y)+1e-30)); return std::string(bf); }();
        }
        std::printf("    [跨页拆档] grid=(1,%d) page=%d 查 head=%d -> %s\n", GH, pg, hd, msg.c_str());
        cudaFree(Xc); cudaFree(Xs); cudaFree(Xst);
    }

    // ---- 消费端设备解码函数自证：GPU 用 kvarn_dequant_token 解码，与 host 解码比（同口径应逐位一致）----
    unsigned char* d_c2; unsigned char* d_s2; float* d_out;
    cudaMalloc(&d_c2, code_bytes); cudaMalloc(&d_s2, slot_half * 2); cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_c2, codes.data(), code_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_s2, scales.data(), slot_half * 2, cudaMemcpyHostToDevice);
    if (bits == 2) dequant_probe_kernel<2, 9><<<1, 256>>>(d_c2, d_s2, d_out);
    else           dequant_probe_kernel<4, 9><<<1, 256>>>(d_c2, d_s2, d_out);
    cudaError_t de = cudaDeviceSynchronize();
    if (de != cudaSuccess) { std::printf("consumer dequant launch failed: %s\n", cudaGetErrorString(de)); return 4; }
    std::vector<float> dev(n); cudaMemcpy(dev.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    // host 解码是 [通道][token] 布局，设备输出是 [token][通道]，比之前按同一布局重排
    std::vector<float> dev_cm(n);
    for (int c = 0; c < C; ++c) for (int d = 0; d < R; ++d) dev_cm[(size_t)d * C + c] = dev[(size_t)c * kKVarNHeadDim + d];
    // ---- 引擎布局 prefill kernel 自证：喂 [dim][KVHeads][T][B] 的 bf16，读回后与 host 全链比 ----
    {
        const int KVH = 4;
        std::vector<__nv_bfloat16> h_eng((size_t)kKVarNHeadDim * KVH * C);
        for (int t = 0; t < C; ++t)
            for (int hd = 0; hd < KVH; ++hd)
                for (int dd = 0; dd < kKVarNHeadDim; ++dd)
                    h_eng[dd + kKVarNHeadDim * (hd + KVH * t)] = __float2bfloat16(in[(size_t)dd * C + t]);
        __nv_bfloat16 *e_k, *e_v; int* e_pos; int* e_bt; unsigned char *e_c, *e_s;
        const size_t esz = (size_t)kKVarNHeadDim * KVH * C * 2;
        cudaMalloc(&e_k, esz); cudaMalloc(&e_v, esz);
        cudaMemcpy(e_k, h_eng.data(), esz, cudaMemcpyHostToDevice); cudaMemcpy(e_v, h_eng.data(), esz, cudaMemcpyHostToDevice);
        cudaMalloc(&e_pos, C * 4); cudaMalloc(&e_bt, 4);
        int posv[64]; for (int t = 0; t < C; ++t) posv[t] = t;
        cudaMemcpy(e_pos, posv, C * 4, cudaMemcpyHostToDevice);
        int bt0 = 0; cudaMemcpy(e_bt, &bt0, 4, cudaMemcpyHostToDevice);
        // plane 偏移含 head ⇒ 缓冲要按 KVHeads 个 head 一起算（之前只算单 head，写越界）
        const int cb = C * (kKVarNHeadDim * bits / 8) * KVH, sb = C * 9 * KVH;
        cudaMalloc(&e_c, cb); cudaMalloc(&e_s, (size_t)sb * 2); cudaMemset(e_c, 0, cb); cudaMemset(e_s, 0, (size_t)sb * 2);
            float* e_bal = nullptr; cudaMalloc(&e_bal, (size_t)n * sizeof(float));
            unsigned char *e_c2, *e_s2; cudaMalloc(&e_c2, cb); cudaMalloc(&e_s2, (size_t)sb * 2);
            cudaMemset(e_c2, 0, cb); cudaMemset(e_s2, 0, (size_t)sb * 2);
        const int dyn2 = kKVarNHeadDim * kKVarNTileTokens * (int)sizeof(float);
        cudaFuncSetAttribute(kvarn_append_prefill_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
        cudaFuncSetAttribute(kvarn_append_prefill_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
        if (bits == 2) kvarn_append_prefill_kernel<2, 4><<<dim3(1, 1), 256, dyn2>>>(e_k, e_v, e_pos, e_bt, e_c, e_s, e_c2, e_s2, C, e_bal);
        else           kvarn_append_prefill_kernel<4, 4><<<dim3(1, 1), 256, dyn2>>>(e_k, e_v, e_pos, e_bt, e_c, e_s, e_c2, e_s2, C, e_bal);
        cudaError_t ee = cudaDeviceSynchronize();
        std::string m2 = "ok";
        if (ee != cudaSuccess) { m2 = cudaGetErrorString(ee); cudaGetLastError(); }
        else {
            std::vector<unsigned char> ec(cb); std::vector<__half> es(sb);
            cudaMemcpy(ec.data(), e_c, cb, cudaMemcpyDeviceToHost); cudaMemcpy(es.data(), e_s, (size_t)sb * 2, cudaMemcpyDeviceToHost);
            std::vector<float> deqE(n, 0.f);
            const int per = 8 / bits, msk = (1 << bits) - 1;
            for (int dd = 0; dd < R; ++dd) {
                const float sc = __half2float(es[(size_t)(dd >> 2) * 9 + (dd & 3)]);
                const float zp = __half2float(es[(size_t)(dd >> 2) * 9 + 4 + (dd & 3)]);
                for (int t = 0; t < C; ++t) {
                    const size_t bi = (size_t)t * (kKVarNHeadDim * bits / 8) + (bits == 4 ? (dd >> 1) : (dd >> 2));
                    const int q = (ec[bi] >> ((bits == 4 ? (dd & 1) : (dd & 3)) * bits)) & msk;
                    deqE[(size_t)dd * C + t] = (q * sc + zp) * __half2float(es[(size_t)t * 9 + 8]);
                }
            }
            std::vector<float> ebal(n); cudaMemcpy(ebal.data(), e_bal, (size_t)n * sizeof(float), cudaMemcpyDeviceToHost);
            m2 = "cos(引擎布局balance,host balance)=" + [&]{ double bb=0,bx=0,by=0; for (int i=0;i<n;++i){bb+=(double)ebal[i]*h_bal_ref[i];bx+=(double)ebal[i]*ebal[i];by+=(double)h_bal_ref[i]*h_bal_ref[i];} char b3[32]; std::snprintf(b3,32,"%.7f",bb/(std::sqrt(bx)*std::sqrt(by)+1e-30)); return std::string(b3); }() + "  解码 ";
            m2 += "cos=" + [&]{ double dd2=0,x2=0,y2=0; for (int i=0;i<n;++i){dd2+=(double)deqE[i]*host_full[i];x2+=(double)deqE[i]*deqE[i];y2+=(double)host_full[i]*host_full[i];} char bf[32]; std::snprintf(bf,32,"%.7f",dd2/(std::sqrt(x2)*std::sqrt(y2)+1e-30)); return std::string(bf); }();
        }
        std::printf("%-42s          | 引擎布局 prefill kernel: %s\n", "(engine layout)", m2.c_str());

        // ---- 解码路径自证：先 append 40 个 token（页未满，只暂存），再 append 24 个（第 64 个落地→编码）----
        {
            __nv_bfloat16* pool; cudaMalloc(&pool, (size_t)1 * 4 * 2 * kKVarNTileTokens * kKVarNHeadDim * 2);
            cudaMemset(pool, 0, (size_t)1 * 4 * 2 * kKVarNTileTokens * kKVarNHeadDim * 2);
            unsigned char* kc; unsigned char* ks; unsigned char* vc; unsigned char* vs;
            cudaMalloc(&kc, cb); cudaMalloc(&ks, sb * 2); cudaMalloc(&vc, cb); cudaMalloc(&vs, sb * 2);
            cudaMemset(kc, 0, cb); cudaMemset(ks, 0, sb * 2); cudaMemset(vc, 0, cb); cudaMemset(vs, 0, sb * 2);
            int bt4[4] = {0, 0, 0, 0};                                  // 位置 0..63 全落 page 0
            int* e_bt2; cudaMalloc(&e_bt2, 16); cudaMemcpy(e_bt2, bt4, 16, cudaMemcpyHostToDevice);
            int pos1[64], pos2[64];
            for (int t = 0; t < 64; ++t) { pos1[t] = t; pos2[t] = 40 + t; }
            int *ep1, *ep2; cudaMalloc(&ep1, 256); cudaMalloc(&ep2, 256);
            cudaMemcpy(ep1, pos1, 40 * 4, cudaMemcpyHostToDevice);
            cudaMemcpy(ep2, pos2, 24 * 4, cudaMemcpyHostToDevice);
            // 第二段输入：token 40..63 另起一份 [dim][KVHeads][24]
            std::vector<__nv_bfloat16> h2((size_t)kKVarNHeadDim * KVH * 24);
            for (int t = 0; t < 24; ++t)
                for (int hd = 0; hd < KVH; ++hd)
                    for (int dd = 0; dd < kKVarNHeadDim; ++dd)
                        h2[dd + kKVarNHeadDim * (hd + KVH * t)] = __float2bfloat16(in[(size_t)dd * C + 40 + t]);
            __nv_bfloat16 *k2, *v2; cudaMalloc(&k2, h2.size() * 2); cudaMalloc(&v2, h2.size() * 2);
            cudaMemcpy(k2, h2.data(), h2.size() * 2, cudaMemcpyHostToDevice);
            cudaMemcpy(v2, h2.data(), h2.size() * 2, cudaMemcpyHostToDevice);
            cudaFuncSetAttribute(kvarn_append_decode_kernel<4, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
            cudaFuncSetAttribute(kvarn_append_decode_kernel<2, 4>, cudaFuncAttributeMaxDynamicSharedMemorySize, dyn2);
            std::string dm = "ok";
            const int first = (bits == 2) ? 2 : 4;
            for (int stage = 0; stage < 2 && dm == "ok"; ++stage) {
                const bool second = (stage == 1);
                const void* kk = second ? (const void*)k2 : (const void*)e_k;
                const void* vv = second ? (const void*)v2 : (const void*)e_v;
                const int* pp  = second ? ep2 : ep1;
                const int ntok = second ? 24 : 40;
                if (first == 2) kvarn_append_decode_kernel<2, 4><<<dim3(1, 4), 256, dyn2>>>((const __nv_bfloat16*)kk, (const __nv_bfloat16*)vv, pp, e_bt2, pool, kc, ks, vc, vs, ntok);
                else            kvarn_append_decode_kernel<4, 4><<<dim3(1, 4), 256, dyn2>>>((const __nv_bfloat16*)kk, (const __nv_bfloat16*)vv, pp, e_bt2, pool, kc, ks, vc, vs, ntok);
                cudaError_t de = cudaDeviceSynchronize();
                if (de != cudaSuccess) { dm = cudaGetErrorString(de); cudaGetLastError(); }
            }
            std::vector<float> deqA(n, 0.f), deqB(n, 0.f);
            if (dm == "ok") {
                std::vector<unsigned char> xcdA(cb), xcdB(cb); std::vector<__half> xsdA(sb), xsdB(sb);
                // K：page 0 / head 0
                cudaMemcpy(xcdA.data(), kc, cb, cudaMemcpyDeviceToHost);
                cudaMemcpy(xsdA.data(), ks, (size_t)sb * 2, cudaMemcpyDeviceToHost);
                const int msk = (1 << bits) - 1;
                for (int dd = 0; dd < R; ++dd) {
                    const float s1 = __half2float(xsdA[(size_t)(dd >> 2) * 9 + (dd & 3)]);
                    const float z1 = __half2float(xsdA[(size_t)(dd >> 2) * 9 + 4 + (dd & 3)]);
                    for (int t = 0; t < C; ++t) {
                        const size_t bi = (size_t)t * (kKVarNHeadDim * bits / 8) + (bits == 4 ? (dd >> 1) : (dd >> 2));
                        const int q = (xcdA[bi] >> ((bits == 4 ? (dd & 1) : (dd & 3)) * bits)) & msk;
                        deqA[(size_t)dd * C + t] = (q * s1 + z1) * __half2float(xsdA[(size_t)t * 9 + 8]);
                    }
                }
                cudaMemcpy(xcdB.data(), vc, cb, cudaMemcpyDeviceToHost);
                cudaMemcpy(xsdB.data(), vs, (size_t)sb * 2, cudaMemcpyDeviceToHost);
                for (int dd = 0; dd < R; ++dd) {
                    const float s1 = __half2float(xsdB[(size_t)(dd >> 2) * 9 + (dd & 3)]);
                    const float z1 = __half2float(xsdB[(size_t)(dd >> 2) * 9 + 4 + (dd & 3)]);
                    for (int t = 0; t < C; ++t) {
                        const size_t bi = (size_t)t * (kKVarNHeadDim * bits / 8) + (bits == 4 ? (dd >> 1) : (dd >> 2));
                        const int q = (xcdB[bi] >> ((bits == 4 ? (dd & 1) : (dd & 3)) * bits)) & msk;
                        deqB[(size_t)dd * C + t] = (q * s1 + z1) * __half2float(xsdB[(size_t)t * 9 + 8]);
                    }
                }
                auto cs3 = [&](const std::vector<float>& a) { double dd3 = 0, x3 = 0, y3 = 0; for (int i = 0; i < n; ++i) { dd3 += (double)a[i] * host_full[i]; x3 += (double)a[i] * a[i]; y3 += (double)host_full[i] * host_full[i]; } char b4[48]; std::snprintf(b4, 48, "%.7f", dd3 / (std::sqrt(x3) * std::sqrt(y3) + 1e-30)); return std::string(b4); };
                int badA = 0, badS = 0; float maxS = 0.f;
                for (int i = 0; i < n; ++i) { if (!std::isfinite(deqA[i])) ++badA; }
                for (int i = 0; i < (int)(C * 9); ++i) { const float sv = __half2float(xsdA[i]); if (!std::isfinite(sv)) ++badS; if (std::fabs(sv) > maxS) maxS = std::fabs(sv); }
                std::printf("   [探针 K] 非有限 deqA=%d/%d  scale 非有限=%d  max|scale|=%.6g  codes[0..3]=%02x %02x %02x %02x\n",
                            badA, n, badS, maxS, xcdA[0], xcdA[1], xcdA[2], xcdA[3]);
                dm = "K cos(两段式 decode 读回, host全链)=" + cs3(deqA) + "  V cos=" + cs3(deqB);
            }
            std::printf("%-42s          | %s\n", "(decode staging 两段式)", dm.c_str());
            cudaFree(pool); cudaFree(kc); cudaFree(ks); cudaFree(vc); cudaFree(vs); cudaFree(e_bt2); cudaFree(ep1); cudaFree(ep2); cudaFree(k2); cudaFree(v2);
        }
        cudaFree(e_k); cudaFree(e_v); cudaFree(e_pos); cudaFree(e_bt); cudaFree(e_c); cudaFree(e_s); cudaFree(e_c2); cudaFree(e_s2); cudaFree(e_bal);
    }

    std::printf("%-42s          | cos(设备解码, host解码)=%.7f  cos(设备解码, host全链)=%.7f\n",
                "(consumer dequant)", cos_of(dev_cm, deq), cos_of(dev_cm, host_full));
    cudaFree(d_c2); cudaFree(d_s2); cudaFree(d_out);
    cudaFree(d_bal);
    cudaFree(d_stage); cudaFree(d_codes); cudaFree(d_scales);
    return 0;
}

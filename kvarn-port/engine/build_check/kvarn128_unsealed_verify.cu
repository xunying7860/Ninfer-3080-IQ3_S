// kv128_verify_unsealed2.cu —— 未封口 tile 的引擎级判定（纯 host，无 kernel）
//
//   用法: kv128_unsealed2 <pre_stage.kvo> <pre_win.kvo> <cont_stage.kvo> <cont_win.kvo>
//                        <cont_tt_begin> <cont_tt_end>
//
// 两份 dump 都来自**同一个 layer**（probe 取"第一次 tt_begin==0 的 append"与"第一次
// tt_begin>0 的 append"，而 prefill/decode 都是逐 layer 推进 ⇒ 都是 layer 0 的同一个窗口）：
//   *_stage.kvo = 引擎 g_stage 里 tile/head0 的 128×256 旋转域 fp16→f32（domain=0）
//   *_win.kvo   = 引擎窗口同 tile/head0 的 128×256 bf16→f32（domain=1，已逆旋转）
//
// 四个数（主机对 stage 做一次 inverse_rotate_d256 后比 cos）：
//   A = cos(inv_rot(pre_stage[tt]),  pre_win[tt])   prefill 那次：窗口是不是 stage 的忠实拷贝
//   B = cos(inv_rot(cont_stage[tt]), cont_win[tt])  续写那次：同上
//   C = cos(inv_rot(cont_stage[tt]), pre_win[tt])   续写那次的 stage 是不是"这块 tile 的真值"
//                                                   （tt<tt_begin 段：真值 = pre_win，因为
//                                                    同一 layer、同一 tile、同一批 token）
//   D = cos(cont_win[tt], pre_win[tt])              窗口内容有没有被改写
//
// 判读：
//   A≈1 且 B≈1           ⇒ 未封口分支"照抄 stage"这一步没写错（拷贝/寻址/逆旋转都对）；
//   C≪1 (tt<tt_begin)    ⇒ 续写那次读到的 stage 不是这块 tile 的真值（陈旧：那是别的 layer/
//                          别的 append 留下的内容）；
//   D≪1 (tt<tt_begin)    ⇒ 窗口[0,tt_begin) 被那次 append 改写掉了 ⇒ 真值被覆盖 = 损坏。
#include "kvarn128_cpu_reference.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using kvarn128_cpu::inverse_rotate_d256;
using kvarn128_cpu::kHeadDim;

static bool rd(void* p, std::size_t n, FILE* f) { return std::fread(p, 1, n, f) == n; }

struct Head {
    std::string        name;
    int                kind = 0, tokens = 0, width = 0, domain = 0;
    std::vector<float> data;
};

static bool load_kvo1(const char* path, std::vector<Head>& out) {
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) { std::fprintf(stderr, "打不开 %s\n", path); return false; }
    std::uint32_t magic = 0, ver = 0, n = 0;
    if (!rd(&magic, 4, f) || !rd(&ver, 4, f) || !rd(&n, 4, f)) { std::fclose(f); return false; }
    if (magic != 0x314F564Bu || ver != 1u) {
        std::fprintf(stderr, "KVO1 magic/ver 不对 (%08x,%u)\n", magic, ver);
        std::fclose(f);
        return false;
    }
    for (std::uint32_t i = 0; i < n; ++i) {
        Head          H;
        std::uint32_t nl = 0;
        if (!rd(&nl, 4, f)) { std::fclose(f); return false; }
        H.name.resize(nl);
        if (nl != 0 && !rd(&H.name[0], nl, f)) { std::fclose(f); return false; }
        std::int32_t a = 0;
        rd(&a, 4, f); H.kind = a;
        rd(&a, 4, f); H.tokens = a;
        rd(&a, 4, f); H.width = a;
        rd(&a, 4, f); H.domain = a;
        std::uint32_t nf = 0;
        rd(&nf, 4, f);
        H.data.resize(nf);
        if (nf != 0 && !rd(H.data.data(), nf * 4, f)) { std::fclose(f); return false; }
        out.push_back(std::move(H));
    }
    std::fclose(f);
    return true;
}

static double cosine(const float* a, const float* b, int n) {
    double dot = 0, na = 0, nb = 0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(a[i]) * b[i];
        na += static_cast<double>(a[i]) * a[i];
        nb += static_cast<double>(b[i]) * b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "用法: %s <pre_stage.kvo> <pre_win.kvo> <cont_stage.kvo> <cont_win.kvo> "
                     "<cont_tt_begin> <cont_tt_end>\n",
                     argv[0]);
        return 2;
    }
    const int tt_begin = std::atoi(argv[5]);
    const int tt_end   = std::atoi(argv[6]);
    std::vector<Head> ps, pw, cs, cw;
    if (!load_kvo1(argv[1], ps) || !load_kvo1(argv[2], pw) || !load_kvo1(argv[3], cs) ||
        !load_kvo1(argv[4], cw))
        return 3;
    std::printf("pre_stage %zu head  pre_win %zu  cont_stage %zu  cont_win %zu   "
                "cont tt=[%d,%d)\n",
                ps.size(), pw.size(), cs.size(), cw.size(), tt_begin, tt_end);
    if (ps.size() != pw.size() || cs.size() != cw.size() || ps.size() != cs.size()) {
        std::printf("head 数不一致\n");
        return 4;
    }

    for (std::size_t i = 0; i < ps.size(); ++i) {
        const bool  is_value = (ps[i].kind == 1);
        const char* nm       = is_value ? "V" : "K";
        std::printf("\n================ head[%zu] %s  (%s) ================\n", i, nm,
                    ps[i].name.c_str());
        std::vector<float> pr(kHeadDim), cr(kHeadDim);
        double amin = 1e9, bmin = 1e9, cmin = 1e9, dmin = 1e9;
        double amin_k = 1e9, bmin_k = 1e9, amin_f = 1e9, bmin_f = 1e9;
        double csum = 0, dsum = 0;
        int    nkept = 0, bad_d = 0, nfresh = 0;
        for (int tt = 0; tt < ps[i].tokens; ++tt) {
            const std::size_t off = static_cast<std::size_t>(tt) * kHeadDim;
            const float* pst = ps[i].data.data() + off;
            const float* pwn = pw[i].data.data() + off;
            const float* cst = cs[i].data.data() + off;
            const float* cwn = cw[i].data.data() + off;
            for (int d = 0; d < kHeadDim; ++d) { pr[d] = pst[d]; cr[d] = cst[d]; }
            inverse_rotate_d256(pr.data());
            inverse_rotate_d256(cr.data());
            const double A = cosine(pr.data(), pwn, kHeadDim);
            const double B = cosine(cr.data(), cwn, kHeadDim);
            const double C = cosine(cr.data(), pwn, kHeadDim);
            const double D = cosine(cwn, pwn, kHeadDim);
            if (A < amin) amin = A;
            if (B < bmin) bmin = B;
            if (tt < tt_begin) {
                if (C < cmin) cmin = C;
                if (D < dmin) dmin = D;
                if (A < amin_k) amin_k = A;
                if (B < bmin_k) bmin_k = B;
                csum += C;
                dsum += D;
                ++nkept;
                if (D < 0.99) ++bad_d;
            } else if (tt < tt_end) {
                if (A < amin_f) amin_f = A;
                if (B < bmin_f) bmin_f = B;
                ++nfresh;
            }
            std::printf("  %s tt=%3d A=%.6f B=%.6f C=%.6f D=%.6f%s\n", nm, tt, A, B, C, D,
                        (tt < tt_begin && D < 0.99) ? "   <-- 窗口被改写" : "");
        }
        std::printf("  [%s] A_min=%.9f (prefill 照抄 stage)   B_min=%.9f (续写照抄 stage)\n", nm,
                    amin, bmin);
        std::printf("  [%s] 保持段 [0,%d)：A_min=%.9f B_min=%.9f  C_min=%.9f C_mean=%.9f  "
                    "D_min=%.9f D_mean=%.9f  被改写的 token=%d/%d\n",
                    nm, tt_begin, amin_k, bmin_k, cmin, nkept ? csum / nkept : 0.0, dmin,
                    nkept ? dsum / nkept : 0.0, bad_d, nkept);
        std::printf("  [%s] 新写段 [%d,%d)：A_min=%.9f B_min=%.9f（窗口=fresh stage 的忠实拷贝）\n",
                    nm, tt_begin, tt_end, amin_f, bmin_f);
    }
    return 0;
}

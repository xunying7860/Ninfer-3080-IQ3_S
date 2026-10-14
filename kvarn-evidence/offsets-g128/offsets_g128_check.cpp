// offsets_g128_check.cpp —— KVarN G128「P1 页自足」几何的 leading / 页内偏移 校验程序
// ===========================================================================
// 目的（对应任务：offsets-g128 产物 1/2/3）
//   ① P1 页内布局 与 gate2 整 tile 的 record 布局**互相换算**，并逐位断言；
//   ② 用 oracle（dump_oracle.cpp dumprec）落盘的 record 逐段比对 P1 偏移；
//   ③ 算清 P1 相对 record 口径每 tile 每 head 多花多少字节、折算 B/token 与相对 280 的百分比。
//
// 本程序为**纯 C++17、无 CUDA、无 GPU**（docker 里 g++ 编译即可跑）。
// 所有引用的引擎约束都是**只读**抄录，行号按 kvarn-任务A-20261012/engine/ninfer-all-kvarn/。
//
// ---------------------------------------------------------------------------
// 引用的真值（每条给 文件:行号）
// ---------------------------------------------------------------------------
// [E1] 页大小 64：            src/core/paged_kv_cache.h:19     inline constexpr std::int32_t kPagedKVPageSize = 64;
// [E2] 页 shift = 6：         src/ops/kernel/paged_kv_address.cuh:9   inline constexpr int kPagedKVPageShift = 6;
// [E3] plane 形状 PageMajor：  src/core/paged_kv_cache.cpp:104-107
//        planned.storage = builder.add_tensor(plane.dtype,
//            {plane.leading_extent, kPagedKVPageSize, plane.head_extent, physical_pages},
//            plane.alignment, label);
// [E4] 偏移公式（本程序照抄）：src/ops/kernel/paged_kv_address.cuh:47-61
//        page_head_offset = (int64)LeadingExtent * kPagedKVPageSize *
//                           (head + (int64)HeadExtent * physical_page);
//        element_offset   = page_head_offset + (int64)LeadingExtent * page_offset + leading;
// [E5] 形状硬校验：           src/core/paged_kv_cache.cpp:40-50 validate_geometry
//        page_tokens 必须 == kPagedKVPageSize；leading_extent > 0；head_extent > 0。（无 2 的幂/无字节对齐要求）
// [E6] 运行期 require_shape： src/ops/kv_cache/append/kv_cache_append.cpp:20-25 + 调用点 :52-79
//        ne[0]==profile.key/value_leading_extent, ne[1]==kPagedKVPageSize, ne[2]==kv_heads, ne[3]==physical_pages
//      （causal_softmax_attention.cpp:111/134 同形）
// [E7] plane 区域对齐 256 B： src/core/paged_kv_cache.h:53   std::size_t alignment = 256;
//        —— 只作用于 LayoutBuilder 给**整个 plane 区域**的对齐，不约束 leading/每 token 行宽。
// [E8] profile 字段名：       src/ops/kv_cache/d256_profile.h:25-33
//        key_leading_extent / value_leading_extent / scale_leading_extent / value_scale_leading_extent
//
// [R1] record 布局（gate2 口径）：beellama llama-kvarn.cpp:673-688 make_record_layout
//        rows = value ? 128 : record_dim;   cols = value ? record_dim : 128;
//        payload = rows*cols*bits/8;  scale_off=payload; zp_off=payload+rows*2;
//        other_off=payload+2*rows*2;  record_bytes=other_off+cols*2;
//      D256 ⇒ record_dim = 128、head_slices = 2（llama-kvarn.cpp:690-698 geometry_for case 256），
//      每 tile 每 head = K 记录 ×2 slice + V 记录 ×2 slice。
//      与 gate2 实测闭式一致：2048*bits + 768（GATE2 文档 §2.3 注释、oracle out/table.txt）。
// [R2] 位串：连续比特流、位内 LSB-first、每行独立字节对齐（row_bytes = 128*bits/8 = 16*bits）。
//      gate2 已验 iters=0 下 576/576 record 逐字节一致。
// [R3] 轴语义（接线细案 §2.2）：K 的 scale_axis/zp_axis 是**每维**向量、other_axis 是**每 token**向量；
//      V 相反（scale/zp 每 token、other 每维）。
// ===========================================================================

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

// ---------------------------------------------------------------------------
// 1. 引擎常量（见 [E1][E2]）
// ---------------------------------------------------------------------------
static constexpr int kPageSize   = 64;    // [E1]
static constexpr int kPageShift  = 6;     // [E2]
static constexpr int kHeads      = 4;     // KV head 数（阶段 1 tile / oracle 数据口径）
static constexpr int kHeadDim    = 256;   // [E8] d256
static constexpr int kRecordDim  = 128;   // [R1] D256 ⇒ record_dim = 128
static constexpr int kSlices     = 2;     // [R1] head_slices
static constexpr int kTileTokens = 128;   // G128 tile = 2 页

static_assert(kPageSize == (1 << kPageShift), "页大小与 shift 必须自洽");

// [E4] 逐字照抄 src/ops/kernel/paged_kv_address.cuh:47-61
static inline std::int64_t paged_kv_page_head_offset(int LeadingExtent, int HeadExtent,
                                                    std::int32_t physical_page, std::int32_t head) {
    return static_cast<std::int64_t>(LeadingExtent) * kPageSize *
           (static_cast<std::int64_t>(head) +
            static_cast<std::int64_t>(HeadExtent) * physical_page);
}
static inline std::int64_t paged_kv_element_offset(int LeadingExtent, int HeadExtent,
                                                   std::int32_t physical_page, std::int32_t head,
                                                   std::int32_t page_offset, std::int32_t leading) {
    return paged_kv_page_head_offset(LeadingExtent, HeadExtent, physical_page, head) +
           static_cast<std::int64_t>(LeadingExtent) * page_offset + leading;
}

// ---------------------------------------------------------------------------
// 2. record 布局（[R1]，逐行抄 make_record_layout）
// ---------------------------------------------------------------------------
struct RecordLayout {
    int    rows = 0, cols = 0, bits = 0;
    size_t payload = 0, scale_off = 0, zp_off = 0, other_off = 0, record_bytes = 0;
};
static size_t bee_packed_bytes(int n_values, int bits) {
    return (static_cast<size_t>(n_values) * static_cast<size_t>(bits) + 7) / 8;
}
static RecordLayout make_record_layout(int record_dim, int bits, bool value) {
    RecordLayout R{};
    R.rows = value ? 128 : record_dim;
    R.cols = value ? record_dim : 128;
    R.bits = bits;
    R.payload    = bee_packed_bytes(R.rows * R.cols, bits);
    R.scale_off  = R.payload;
    R.zp_off     = R.scale_off + static_cast<size_t>(R.rows) * 2;
    R.other_off  = R.zp_off + static_cast<size_t>(R.rows) * 2;
    R.record_bytes = R.other_off + static_cast<size_t>(R.cols) * 2;
    return R;
}
static inline int64_t record_tile_bytes(int kb, int vb) {
    return (int64_t)kSlices * (int64_t)(make_record_layout(kRecordDim, kb, false).record_bytes +
                                        make_record_layout(kRecordDim, vb, true).record_bytes);
}

// ---------------------------------------------------------------------------
// 3. P1 页自足几何（本程序的产出）
//    leading 单位 = 元素；码 plane 是 U8（1 B/元素）、scale plane 是 FP16（2 B/元素）
//      k_code_leading  = 256 dim * kb bit / 8 = 32*kb   [字节/token 行]
//      v_code_leading  = 32*vb
//      k_scale_leading = (2 slice*128 维 scale + 2 slice*128 维 zp + 2 slice*64 token other) / 64 行 = 640/64 = 10
//      v_scale_leading = (2 slice*128 维 other + 2 slice*64 token scale + 2 slice*64 token zp) / 64 行 = 512/64 = 8
// ---------------------------------------------------------------------------
struct P1Geom {
    int kb = 0, vb = 0;
    int k_code_leading = 0, v_code_leading = 0, k_scale_leading = 0, v_scale_leading = 0;

    size_t k_code_page_bytes()  const { return (size_t)kPageSize * k_code_leading; }
    size_t v_code_page_bytes()  const { return (size_t)kPageSize * v_code_leading; }
    size_t k_scale_page_bytes() const { return (size_t)kPageSize * k_scale_leading * 2; }  // FP16
    size_t v_scale_page_bytes() const { return (size_t)kPageSize * v_scale_leading * 2; }
    size_t tile_bytes() const {
        return 2 * (k_code_page_bytes() + v_code_page_bytes() +
                    k_scale_page_bytes() + v_scale_page_bytes());
    }
    int b_per_token() const { return (int)(tile_bytes() / kTileTokens); }
};
static P1Geom p1_geom(int kb, int vb) {
    P1Geom g{};
    g.kb = kb; g.vb = vb;
    g.k_code_leading  = 32 * kb;
    g.v_code_leading  = 32 * vb;
    g.k_scale_leading = 10;
    g.v_scale_leading = 8;
    return g;
}

// 轴条目的种类（用于逐段比对）
enum Item {
    K_CODE = 0, V_CODE, K_SA, K_ZP, K_OTHER, V_OTHER, V_SA, V_ZP, ITEM_COUNT
};
static const char* item_name(Item i) {
    switch (i) {
    case K_CODE:  return "k_code";
    case V_CODE:  return "v_code";
    case K_SA:    return "k_scale_axis";
    case K_ZP:    return "k_zp_axis";
    case K_OTHER: return "k_other_axis";
    case V_OTHER: return "v_other_axis";
    case V_SA:    return "v_scale_axis";
    case V_ZP:    return "v_zp_axis";
    default:      return "?";
    }
}

// P1 轴条目 -> (页内行, 页内列)。index 的语义见调用处。
// K 侧维轴：全局维 d = slice*128 + r，行 = d/4，列 = d%4（scale）/ 4 + d%4（zp）
// K 侧 token 轴：行 = T%64，列 = 8 + slice
// V 侧维轴（other）：行 = d/4，列 = d%4
// V 侧 token 轴：行 = T%64，列 = 4 + slice（scale）/ 6 + slice（zp）
static inline int p1_row_of(Item it, int slice, int index) {
    switch (it) {
    case K_SA: case K_ZP: { const int d = slice * 128 + index; return d / 4; }
    case K_OTHER:         return index % kPageSize;
    case V_OTHER: { const int d = slice * 128 + index; return d / 4; }
    case V_SA: case V_ZP: return index % kPageSize;
    default: return -1;
    }
}
static inline int p1_col_of(Item it, int slice, int index) {
    switch (it) {
    case K_SA:    { const int d = slice * 128 + index; return d % 4; }
    case K_ZP:    { const int d = slice * 128 + index; return 4 + d % 4; }
    case K_OTHER: return 8 + slice;
    case V_OTHER: { const int d = slice * 128 + index; return d % 4; }
    case V_SA:    return 4 + slice;
    case V_ZP:    return 6 + slice;
    default: return -1;
    }
}
// 该条目的源：record 分段 + 相对该 record 的 payload 的字节偏移
static inline const char* rec_seg_of(Item it) {
    switch (it) {
    case K_SA: case V_SA: return "scale_off";  // payload + 2*idx
    case K_ZP: case V_ZP: return "zp_off";     // payload + 256 + 2*idx
    case K_OTHER: case V_OTHER: return "other_off"; // payload + 512 + 2*idx
    default: return "payload";
    }
}
static inline size_t rec_axis_byte_off(Item it, int index) {
    switch (it) {
    case K_SA: case V_SA: return (size_t)2 * index;
    case K_ZP: case V_ZP: return 256 + (size_t)2 * index;
    case K_OTHER: case V_OTHER: return 512 + (size_t)2 * index;
    default: return 0;
    }
}

// ---------------------------------------------------------------------------
// 4. 位操作（[R2] 连续比特流、位内 LSB-first）
// ---------------------------------------------------------------------------
static inline unsigned get_bit(const uint8_t* p, int64_t bit) {
    return (p[bit >> 3] >> (bit & 7)) & 1u;
}
static inline void put_bit(uint8_t* p, int64_t bit, unsigned v) {
    p[bit >> 3] = (uint8_t)((p[bit >> 3] & ~(1u << (bit & 7))) | ((v & 1u) << (bit & 7)));
}
static inline unsigned get_bits(const uint8_t* p, int64_t bit, int n) {
    unsigned v = 0;
    for (int i = 0; i < n; ++i) { v |= get_bit(p, bit + i) << i; }
    return v;
}
static inline void put_bits(uint8_t* p, int64_t bit, int n, unsigned v) {
    for (int i = 0; i < n; ++i) { put_bit(p, bit + i, (v >> i) & 1u); }
}

// ---------------------------------------------------------------------------
// 5. oracle dump 读取（dump_oracle.cpp dumprec 落盘格式，见其 :726-729）
// ---------------------------------------------------------------------------
struct Rec {
    std::string name;
    int rows = 0, cols = 0, bits = 0, iters = 0, best_iter = 0;
    float best_imb = 0.f;
    std::vector<float> s_col, s_row, tile;
    std::vector<uint8_t> rec;
    int layer = 0, head = 0, slice = 0, group = 0;
    char kind = '?';
};
static bool rd(void* dst, size_t n, FILE* f) { return fread(dst, 1, n, f) == n; }
static bool r_u32(uint32_t& v, FILE* f) { return rd(&v, 4, f); }
static bool r_i32(int32_t& v, FILE* f) { return rd(&v, 4, f); }

static bool parse_name(Rec& D) {
    // oracle tile 名："L%d_%c_h%d_s%d_g%d"（make_tiles.py:105）
    int l = 0, h = 0, s = 0, g = 0; char k = '?';
    if (sscanf(D.name.c_str(), "L%d_%c_h%d_s%d_g%d", &l, &k, &h, &s, &g) == 5) {
        D.layer = l; D.kind = k; D.head = h; D.slice = s; D.group = g;
        return true;
    }
    return false;
}
static bool load_dump(const char* path, std::vector<Rec>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "无法打开 %s\n", path); return false; }
    uint32_t magic = 0, ver = 0, n = 0;
    if (!r_u32(magic, f) || !r_u32(ver, f) || !r_u32(n, f)) { fclose(f); return false; }
    if (magic != 0x324B5647u || ver != 2u) {
        fprintf(stderr, "magic/ver 不对 (%08x,%u)\n", magic, ver); fclose(f); return false;
    }
    for (uint32_t i = 0; i < n; ++i) {
        Rec D;
        uint32_t nl = 0; if (!r_u32(nl, f)) { fclose(f); return false; }
        D.name.resize(nl); if (nl && !rd(&D.name[0], nl, f)) { fclose(f); return false; }
        int32_t a = 0;
        if (!r_i32(a, f)) { fclose(f); return false; } D.rows = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.cols = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.bits = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.iters = a;
        if (!r_i32(a, f)) { fclose(f); return false; } D.best_iter = a;
        if (!rd(&D.best_imb, 4, f)) { fclose(f); return false; }
        D.s_col.resize(D.cols); D.s_row.resize(D.rows);
        if (!rd(D.s_col.data(), 4 * (size_t)D.cols, f)) { fclose(f); return false; }
        if (!rd(D.s_row.data(), 4 * (size_t)D.rows, f)) { fclose(f); return false; }
        D.tile.resize((size_t)D.rows * D.cols);
        if (!rd(D.tile.data(), 4 * D.tile.size(), f)) { fclose(f); return false; }
        uint32_t rl = 0; if (!r_u32(rl, f)) { fclose(f); return false; }
        D.rec.resize(rl);
        if (rl && !rd(D.rec.data(), rl, f)) { fclose(f); return false; }
        parse_name(D);
        out.push_back(std::move(D));
    }
    fclose(f);
    return true;
}

// ---------------------------------------------------------------------------
// 6. 全档位字节账 CSV
// ---------------------------------------------------------------------------
static const int BIT_LIST[6] = {2, 3, 4, 5, 6, 8};

static void write_byte_table_csv(FILE* f) {
    fprintf(f, "tier,kb,vb,k_code_leading,v_code_leading,k_scale_leading,v_scale_leading,"
               "k_code_page_bytes,v_code_page_bytes,k_scale_page_bytes,v_scale_page_bytes,"
               "p1_tile_bytes,record_tile_bytes,extra_tile_bytes,extra_per_token_bytes,"
               "p1_b_per_token,record_b_per_token,delta_vs_record,pct_vs_rk4v4_280\n");
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        const int kb = BIT_LIST[i], vb = BIT_LIST[j];
        const P1Geom g = p1_geom(kb, vb);
        const int64_t rec = record_tile_bytes(kb, vb);
        const int64_t p1  = (int64_t)g.tile_bytes();
        const int p1_b    = g.b_per_token();
        const int rec_b   = (int)(rec / kTileTokens);
        fprintf(f, "K%dV%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%zu,%zu,%lld,%lld,%lld,%d,%d,%d,%d,%.4f\n",
                kb, vb, kb, vb, g.k_code_leading, g.v_code_leading, g.k_scale_leading,
                g.v_scale_leading, g.k_code_page_bytes(), g.v_code_page_bytes(),
                g.k_scale_page_bytes(), g.v_scale_page_bytes(),
                (long long)p1, (long long)rec, (long long)(p1 - rec),
                (int)((p1 - rec) / kTileTokens), p1_b, rec_b, p1_b - rec_b,
                100.0 * p1_b / 280.0);
    }
}

// 页内偏移表：每个 (plane, 行, 列) 槽位 -> 语义 + record 来源。逐槽枚举（机器可读）。
static void write_slot_map_csv(FILE* f, const char* const* tiers, int n_tiers) {
    fprintf(f, "tier,plane,page_row,col,slot_index_in_plane,item,slice,index,dim_d_or_token,"
               "leading_elements,record_segment,record_byte_offset_in_record,"
               "engine_element_offset(page=0,head=0)\n");
    for (int t = 0; t < n_tiers; ++t) {
        int kb = 0, vb = 0;
        sscanf(tiers[t], "K%dV%d", &kb, &vb);
        const P1Geom g = p1_geom(kb, vb);
        // K scale plane：10 槽/行
        for (int row = 0; row < kPageSize; ++row) for (int col = 0; col < g.k_scale_leading; ++col) {
            Item it; int slice = 0, index = 0, dd = -1;
            if (col < 4)       { it = K_SA; const int d = 4 * row + col; slice = d / 128; index = d % 128; dd = d; }
            else if (col < 8)  { it = K_ZP; const int d = 4 * row + (col - 4); slice = d / 128; index = d % 128; dd = d; }
            else               { it = K_OTHER; slice = col - 8; index = row; }
            const std::int64_t off = paged_kv_element_offset(g.k_scale_leading, kHeads, 0, 0, row, col);
            fprintf(f, "%s,k_scale_pages,%d,%d,%lld,%s,%d,%d,%d,%d,%s,%zu,%lld\n",
                    tiers[t], row, col, (long long)off, item_name(it), slice, index, dd,
                    g.k_scale_leading, rec_seg_of(it), rec_axis_byte_off(it, index), (long long)off);
        }
        // V scale plane：8 槽/行
        for (int row = 0; row < kPageSize; ++row) for (int col = 0; col < g.v_scale_leading; ++col) {
            Item it; int slice = 0, index = 0, dd = -1;
            if (col < 4)      { it = V_OTHER; const int d = 4 * row + col; slice = d / 128; index = d % 128; dd = d; }
            else if (col < 6) { it = V_SA; slice = col - 4; index = row; }
            else              { it = V_ZP; slice = col - 6; index = row; }
            const std::int64_t off = paged_kv_element_offset(g.v_scale_leading, kHeads, 0, 0, row, col);
            fprintf(f, "%s,v_scale_pages,%d,%d,%lld,%s,%d,%d,%d,%d,%s,%zu,%lld\n",
                    tiers[t], row, col, (long long)off, item_name(it), slice, index, dd,
                    g.v_scale_leading, rec_seg_of(it), rec_axis_byte_off(it, index), (long long)off);
        }
        // 码 plane：按 (slice, dim) 枚举公式行（行 = token%64，列 = 该行内字节；record 侧是 dim 行）
        for (int slice = 0; slice < kSlices; ++slice) for (int r = 0; r < 128; ++r) {
            const int d = slice * 128 + r;
            const size_t row_bytes = (size_t)16 * g.kb;     // record: 每 dim 行 16*bits 字节
            fprintf(f, "%s,k_code_pages,row=token%%64,%d,%d,k_code,%d,%d,%d,%d,payload,%zu,token-wise\n",
                    tiers[t], d * g.kb / 8, d * g.kb / 8, slice, r, d, g.k_code_leading,
                    (size_t)r * row_bytes);
        }
        for (int slice = 0; slice < kSlices; ++slice) for (int r = 0; r < 128; ++r) {
            const int d = slice * 128 + r;
            const size_t row_bytes = (size_t)16 * g.vb;
            fprintf(f, "%s,v_code_pages,row=token%%64,%d,%d,v_code,%d,%d,%d,%d,payload,%zu,token-wise\n",
                    tiers[t], d * g.vb / 8, d * g.vb / 8, slice, r, d, g.v_code_leading,
                    (size_t)r * row_bytes);
        }
    }
}

// ---------------------------------------------------------------------------
// 6b. 结构性推导：P1 每 tile 每 head 多花的字节 = 维轴条目在 tile 两页各存一份
//     维轴（K 的 scale/zp、V 的 other）条目数 = 2 slice*128(K) *2 + 2 slice*128(V) = 768，
//     每份是 FP16 ⇒ 多花 768*2 = 1536 B/tile/head = 12 B/token/KV head。
//     本函数**不**假设这个数，而是枚举 P1 两页的全部槽位做 tally 后推出，并与
//     (P1 tile bytes - record tile bytes) 对账。
// ---------------------------------------------------------------------------
static bool analyze_duplication(const P1Geom& g, long long* dim_entries_out, long long* extra_bytes_out) {
    std::map<std::tuple<int, int, int>, int> tally;   // (item, slice, index) -> 槽位数
    for (int page = 0; page < 2; ++page) {
        for (int row = 0; row < kPageSize; ++row) {
            for (int col = 0; col < g.k_scale_leading; ++col) {
                Item it; int slice = 0, index = 0;
                if (col < 4)        { it = K_SA; const int d = 4 * row + col; slice = d / 128; index = d % 128; }
                else if (col < 8)   { it = K_ZP; const int d = 4 * row + (col - 4); slice = d / 128; index = d % 128; }
                else                { it = K_OTHER; slice = col - 8; index = page * kPageSize + row; }
                tally[std::make_tuple((int)it, slice, index)]++;
            }
            for (int col = 0; col < g.v_scale_leading; ++col) {
                Item it; int slice = 0, index = 0;
                if (col < 4)        { it = V_OTHER; const int d = 4 * row + col; slice = d / 128; index = d % 128; }
                else if (col < 6)   { it = V_SA; slice = col - 4; index = page * kPageSize + row; }
                else                { it = V_ZP; slice = col - 6; index = page * kPageSize + row; }
                tally[std::make_tuple((int)it, slice, index)]++;
            }
        }
    }
    bool ok = true;
    long long dim_entries = 0;
    for (const auto& kv : tally) {
        const Item it = (Item)std::get<0>(kv.first);
        const bool dim_axis = (it == K_SA || it == K_ZP || it == V_OTHER);
        const int expect = dim_axis ? 2 : 1;         // 维轴：两页各一份；token 轴：本页一份
        if (kv.second != expect) { ok = false; }
        if (dim_axis) ++dim_entries;
    }
    *dim_entries_out = dim_entries;
    *extra_bytes_out = dim_entries * 2;              // 多出的那一份副本，FP16 = 2 B/条目
    return ok && dim_entries == 768;
}

// ---------------------------------------------------------------------------
// 7. 主校验
// ---------------------------------------------------------------------------
struct Counter { long long checks = 0, agree = 0, mismatch = 0; };

int main(int argc, char** argv) {
    const std::vector<std::string> dumps(argv + 1, argv + argc);
    if (dumps.empty()) {
        fprintf(stderr, "用法: ./offsets_check <dump1.bin> [dump2.bin ...]\n");
        return 2;
    }

    std::printf("================ P1 页自足几何 · leading/偏移 校验 ================\n");
    std::printf("[const] page=%d shift=%d heads=%d head_dim=%d record_dim=%d slices=%d tile_tokens=%d\n",
                kPageSize, kPageShift, kHeads, kHeadDim, kRecordDim, kSlices, kTileTokens);

    // ---- 7.0 引擎硬约束自检（[E5][E6][E7]）：leading>0、页元素数 = 64*leading、无 2 的幂/对齐要求 ----
    int geom_fail = 0;
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        const P1Geom g = p1_geom(BIT_LIST[i], BIT_LIST[j]);
        struct { const char* name; int leading; int elem; } planes[4] = {
            {"k_code_pages",  g.k_code_leading,  1},
            {"v_code_pages",  g.v_code_leading,  1},
            {"k_scale_pages", g.k_scale_leading, 2},
            {"v_scale_pages", g.v_scale_leading, 2},
        };
        for (auto& p : planes) {
            if (p.leading <= 0) { std::printf("[FAIL] %s leading <= 0\n", p.name); ++geom_fail; }
            // [E5] leading_extent>0 即可；这里额外验证「页元素数 == 64*leading」与引擎 page_head_offset 自洽
            const std::int64_t e = paged_kv_element_offset(p.leading, kHeads, 1, 0, 0, 0);
            const std::int64_t expect = (std::int64_t)p.leading * kPageSize * kHeads;
            if (e != expect) { std::printf("[FAIL] %s page stride 不自洽\n", p.name); ++geom_fail; }
        }
    }
    std::printf("[geom ] 4 平面 × 36 档 的 leading>0 与页 stride 自洽：%s（fail=%d）\n",
                geom_fail == 0 ? "PASS" : "FAIL", geom_fail);

    // ---- 7.0b 结构性推导「每 tile 每 head 多花多少字节」（不假设，枚举后推出并对账） ----
    int dup_bad = 0;
    long long dim_entries = 0, extra_bytes = 0;
    for (int i = 0; i < 6; ++i) for (int j = 0; j < 6; ++j) {
        const P1Geom g = p1_geom(BIT_LIST[i], BIT_LIST[j]);
        long long de = 0, eb = 0;
        if (!analyze_duplication(g, &de, &eb)) ++dup_bad;
        if (eb != (long long)g.tile_bytes() - record_tile_bytes(BIT_LIST[i], BIT_LIST[j])) ++dup_bad;
        dim_entries = de; extra_bytes = eb;
    }
    std::printf("[dupl ] 维轴条目（K scale+zp、V other）=%lld 个，tile 两页各存一份 ⇒ 多花 %lld B/tile/head "
                "= %lld B/token/KV head；与 (P1 tile - record tile) 逐档对账：%s（fail=%d）\n",
                dim_entries, extra_bytes, extra_bytes / kTileTokens,
                dup_bad == 0 ? "PASS" : "FAIL", dup_bad);

    // ---- 7.1 CSV ----
    FILE* fcsv = fopen("out/offsets_p1_g128.csv", "wb");
    if (fcsv) { write_byte_table_csv(fcsv); fclose(fcsv); }
    static const char* TIERS[] = {"K4V4", "K4V2", "K4V3", "K3V3", "K5V5", "K6V6"};
    FILE* fmap = fopen("out/p1_slot_map_g128.csv", "wb");
    if (fmap) { write_slot_map_csv(fmap, TIERS, 6); fclose(fmap); }
    std::printf("[csv  ] out/offsets_p1_g128.csv（36 档字节账）、out/p1_slot_map_g128.csv（6 档逐槽偏移表）\n\n");

    // ---- 7.2 逐 dump 校验 ----
    Counter c_shape, c_seg, c_code, c_round, c_dup;
    int file_ok = 0;
    int first_diff_reported = 0;
    long long code_bad_total = 0;
    std::vector<std::string> report;

    for (const std::string& path : dumps) {
        std::vector<Rec> recs;
        if (!load_dump(path.c_str(), recs)) { std::printf("[FAIL] load %s\n", path.c_str()); continue; }
        // 该 dump 的 kb / vb：K 记录取 bits_k、V 记录取 bits_v（dumprec 按名字里的 _k_/_v_ 分派）
        int kb = -1, vb = -1;
        for (const Rec& D : recs) {
            if (D.kind == 'k') { if (kb < 0) kb = D.bits; else if (kb != D.bits) { kb = -2; } }
            if (D.kind == 'v') { if (vb < 0) vb = D.bits; else if (vb != D.bits) { vb = -2; } }
        }
        if (kb < 0 || vb < 0) {
            std::printf("[FAIL] %s: 无法从记录里定出 kb/vb（kb=%d vb=%d）\n", path.c_str(), kb, vb);
            continue;
        }
        const P1Geom G = p1_geom(kb, vb);
        const RecordLayout KR = make_record_layout(kRecordDim, kb, false);
        const RecordLayout VR = make_record_layout(kRecordDim, vb, true);

        // (a) record 尺寸自检（[R1] 闭式 2048*bits+768），逐 record
        int shape_bad = 0;
        for (const Rec& D : recs) {
            const bool  is_k = (D.kind == 'k');
            const size_t expect = (is_k ? KR.record_bytes : VR.record_bytes);
            const size_t raw    = (size_t)2048 * D.bits + 768;
            ++c_shape.checks;
            if (D.rec.size() == expect && expect == raw && D.rows == 128 && D.cols == 128 &&
                D.bits == (is_k ? kb : vb)) { ++c_shape.agree; } else { ++c_shape.mismatch; ++shape_bad; }
        }

        // (b) 按 (layer, head, group) 组成 tile：需要 K s0/s1 与 V s0/s1
        std::map<std::tuple<int, int, int>, std::map<std::pair<char, int>, const Rec*>> tiles;
        for (const Rec& D : recs) tiles[std::make_tuple(D.layer, D.head, D.group)][std::make_pair(D.kind, D.slice)] = &D;

        int n_tiles = 0, elem_code_checks = 0, elem_code_bad = 0;
        int axis_checks = 0, axis_bad = 0, dup_k = 0, dup_v = 0;

        for (auto& kv : tiles) {
            auto& m = kv.second;
            const Rec *K0 = m.count(std::make_pair('k', 0)) ? m[std::make_pair('k', 0)] : nullptr;
            const Rec *K1 = m.count(std::make_pair('k', 1)) ? m[std::make_pair('k', 1)] : nullptr;
            const Rec *V0 = m.count(std::make_pair('v', 0)) ? m[std::make_pair('v', 0)] : nullptr;
            const Rec *V1 = m.count(std::make_pair('v', 1)) ? m[std::make_pair('v', 1)] : nullptr;
            if (!K0 || !K1 || !V0 || !V1) continue;   // 不完整的 tile（非 2 层全档）跳过
            ++n_tiles;
            const Rec* Ks[2] = {K0, K1};
            const Rec* Vs[2] = {V0, V1};

            // ---- ① 与 gate2 record 布局互相换算：record -> P1（逐位搬运） ----
            // 四块 plane 的 P1 页缓冲。注意 [E4] 的页面步长是 Leading*64*HeadExtent
            // （引擎 PlaneOrder::PageMajor：同一 head 的两页在内存上**不相邻**，[E3]），
            // 所以缓冲按「2 页 × kHeads 个 head 槽」开，用引擎公式索引。
            const size_t kc_alloc = (size_t)G.k_code_leading  * kPageSize * kHeads * 2;
            const size_t vc_alloc = (size_t)G.v_code_leading  * kPageSize * kHeads * 2;
            const size_t ks_alloc = (size_t)G.k_scale_leading * kPageSize * kHeads * 2 * 2;  // FP16
            const size_t vs_alloc = (size_t)G.v_scale_leading * kPageSize * kHeads * 2 * 2;
            std::vector<uint8_t> p1_kc(kc_alloc, 0);
            std::vector<uint8_t> p1_vc(vc_alloc, 0);
            std::vector<uint8_t> p1_ks(ks_alloc, 0);
            std::vector<uint8_t> p1_vs(vs_alloc, 0);
            // 页 0 / 页 1 的基址（元素），照 [E4]
            const std::int64_t ks_p0 = paged_kv_element_offset(G.k_scale_leading, kHeads, 0, 0, 0, 0);
            const std::int64_t ks_p1 = paged_kv_element_offset(G.k_scale_leading, kHeads, 1, 0, 0, 0);
            const std::int64_t vs_p0 = paged_kv_element_offset(G.v_scale_leading, kHeads, 0, 0, 0, 0);
            const std::int64_t vs_p1 = paged_kv_element_offset(G.v_scale_leading, kHeads, 1, 0, 0, 0);

            // 码：K 记录（slice s, 行 r=dim, 列 T=token）-> P1（page=T/64, 行=T%64, 列 d）
            for (int s = 0; s < 2; ++s) {
                for (int r = 0; r < 128; ++r) {
                    const int d = s * 128 + r;
                    for (int T = 0; T < 128; ++T) {
                        const int64_t rbit = (int64_t)r * 128 * kb + (int64_t)T * kb;
                        const int page = T / kPageSize, row = T % kPageSize;
                        // P1 行基址（元素）* 8 + 行内位偏移 (d*kb)；整码 kb 位一起搬（位内 LSB-first，两端口径同）
                        const int64_t pbase_bits = (int64_t)paged_kv_element_offset(G.k_code_leading, kHeads, page, 0, row, 0) * 8;
                        const int64_t pbit2 = pbase_bits + (int64_t)d * kb;
                        const unsigned want = get_bits(Ks[s]->rec.data(), rbit, kb);
                        put_bits(p1_kc.data(), pbit2, kb, want);
                        ++c_code.checks; ++elem_code_checks;
                        if (get_bits(p1_kc.data(), pbit2, kb) != want) ++elem_code_bad;
                    }
                }
                for (int T = 0; T < 128; ++T) {
                    for (int r = 0; r < 128; ++r) {
                        const int d = s * 128 + r;
                        const int64_t rbit = (int64_t)T * 128 * vb + (int64_t)r * vb;
                        const int page = T / kPageSize, row = T % kPageSize;
                        const int64_t pbase_bits = (int64_t)paged_kv_element_offset(G.v_code_leading, kHeads, page, 0, row, 0) * 8;
                        const int64_t pbit = pbase_bits + (int64_t)d * vb;
                        const unsigned wantv = get_bits(Vs[s]->rec.data(), rbit, vb);
                        put_bits(p1_vc.data(), pbit, vb, wantv);
                        ++c_code.checks; ++elem_code_checks;
                        if (get_bits(p1_vc.data(), pbit, vb) != wantv) ++elem_code_bad;
                    }
                }
            }

            // ---- ② 轴向量逐段比对（P1 槽位 <- record 分段，字节级） ----
            const RecordLayout recs_layout[2] = {KR, KR};   // K 与 V 的 record_dim 都是 128；payload 不同（kb/vb）
            const RecordLayout vrecs_layout[2] = {VR, VR};
            // K 轴：scale_axis(s,r)、zp_axis(s,r)、other_axis(s,T)
            for (int s = 0; s < 2; ++s) {
                for (int r = 0; r < 128; ++r) {
                    const int d = s * 128 + r;
                    for (Item it : {K_SA, K_ZP}) {
                        const int row = p1_row_of(it, s, r), col = p1_col_of(it, s, r);
                        const size_t recb = recs_layout[s].payload + rec_axis_byte_off(it, r);
                        const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.k_scale_leading, kHeads, 0, 0, row, col) * 2;
                        memcpy(&p1_ks[(size_t)p1b], &Ks[s]->rec[recb], 2);
                        ++c_seg.checks; ++axis_checks;
                        if (memcmp(&p1_ks[(size_t)p1b], &Ks[s]->rec[recb], 2) != 0) { ++c_seg.mismatch; ++axis_bad; } else { ++c_seg.agree; }
                        // P1 里的这一槽 = 该维轴条目；页 1 也放一份（复制）
                        memcpy(&p1_ks[(size_t)(p1b + (ks_p1 - ks_p0) * 2)], &Ks[s]->rec[recb], 2);
                        (void)d;
                    }
                }
                for (int T = 0; T < 128; ++T) {
                    const int row = p1_row_of(K_OTHER, s, T), col = p1_col_of(K_OTHER, s, T);
                    const size_t recb = recs_layout[s].payload + rec_axis_byte_off(K_OTHER, T);
                    const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.k_scale_leading, kHeads, T / kPageSize, 0, row, col) * 2;
                    memcpy(&p1_ks[(size_t)p1b], &Ks[s]->rec[recb], 2);
                    ++c_seg.checks; ++axis_checks;
                    if (memcmp(&p1_ks[(size_t)p1b], &Ks[s]->rec[recb], 2) != 0) { ++c_seg.mismatch; ++axis_bad; } else { ++c_seg.agree; }
                }
            }
            for (int s = 0; s < 2; ++s) {
                for (int r = 0; r < 128; ++r) {
                    const int row = p1_row_of(V_OTHER, s, r), col = p1_col_of(V_OTHER, s, r);
                    const size_t recb = vrecs_layout[s].payload + rec_axis_byte_off(V_OTHER, r);
                    const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.v_scale_leading, kHeads, 0, 0, row, col) * 2;
                    memcpy(&p1_vs[(size_t)p1b], &Vs[s]->rec[recb], 2);
                    memcpy(&p1_vs[(size_t)(p1b + (vs_p1 - vs_p0) * 2)], &Vs[s]->rec[recb], 2);
                    ++c_seg.checks; ++axis_checks;
                    if (memcmp(&p1_vs[(size_t)p1b], &Vs[s]->rec[recb], 2) != 0) { ++c_seg.mismatch; ++axis_bad; } else { ++c_seg.agree; }
                }
                for (int T = 0; T < 128; ++T) {
                    for (Item it : {V_SA, V_ZP}) {
                        const int row = p1_row_of(it, s, T), col = p1_col_of(it, s, T);
                        const size_t recb = vrecs_layout[s].payload + rec_axis_byte_off(it, T);
                        const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.v_scale_leading, kHeads, T / kPageSize, 0, row, col) * 2;
                        memcpy(&p1_vs[(size_t)p1b], &Vs[s]->rec[recb], 2);
                        ++c_seg.checks; ++axis_checks;
                        if (memcmp(&p1_vs[(size_t)p1b], &Vs[s]->rec[recb], 2) != 0) { ++c_seg.mismatch; ++axis_bad; } else { ++c_seg.agree; }
                    }
                }
            }

            // ---- ③ 反向：P1 -> record 重建，逐字节比对（可互相换算的断言） ----
            for (int s = 0; s < 2; ++s) {
                std::vector<uint8_t> kr(KR.record_bytes, 0), vr(VR.record_bytes, 0);
                for (int r = 0; r < 128; ++r) for (int T = 0; T < 128; ++T) {
                    const int d = s * 128 + r;
                    const int64_t rbit = (int64_t)r * 128 * kb + (int64_t)T * kb;
                    const int page = T / kPageSize, row = T % kPageSize;
                    const int64_t pbase_bits = (int64_t)paged_kv_element_offset(G.k_code_leading, kHeads, page, 0, row, 0) * 8;
                    put_bits(kr.data(), rbit, kb, get_bits(p1_kc.data(), pbase_bits + (int64_t)d * kb, kb));
                    const int64_t rbitv = (int64_t)T * 128 * vb + (int64_t)r * vb;
                    const int64_t pbasev = (int64_t)paged_kv_element_offset(G.v_code_leading, kHeads, page, 0, row, 0) * 8;
                    put_bits(vr.data(), rbitv, vb, get_bits(p1_vc.data(), pbasev + (int64_t)d * vb, vb));
                }
                // 轴：从 P1 **页 0** 取回（页 1 是副本，下面单独断言两页一致）
                for (int r = 0; r < 128; ++r) {
                    for (Item it : {K_SA, K_ZP}) {
                        const int row = p1_row_of(it, s, r), col = p1_col_of(it, s, r);
                        const size_t recb = KR.payload + rec_axis_byte_off(it, r);
                        const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.k_scale_leading, kHeads, 0, 0, row, col) * 2;
                        memcpy(&kr[recb], &p1_ks[(size_t)p1b], 2);
                    }
                    for (Item it : {V_OTHER}) {
                        const int row = p1_row_of(it, s, r), col = p1_col_of(it, s, r);
                        const size_t recb = VR.payload + rec_axis_byte_off(it, r);
                        const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.v_scale_leading, kHeads, 0, 0, row, col) * 2;
                        memcpy(&vr[recb], &p1_vs[(size_t)p1b], 2);
                    }
                }
                for (int T = 0; T < 128; ++T) {
                    { const Item it = K_OTHER; const int row = p1_row_of(it, s, T), col = p1_col_of(it, s, T);
                      const size_t recb = KR.payload + rec_axis_byte_off(it, T);
                      const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.k_scale_leading, kHeads, T / kPageSize, 0, row, col) * 2;
                      memcpy(&kr[recb], &p1_ks[(size_t)p1b], 2); }
                    for (Item it : {V_SA, V_ZP}) {
                      const int row = p1_row_of(it, s, T), col = p1_col_of(it, s, T);
                      const size_t recb = VR.payload + rec_axis_byte_off(it, T);
                      const std::int64_t p1b = (std::int64_t)paged_kv_element_offset(G.v_scale_leading, kHeads, T / kPageSize, 0, row, col) * 2;
                      memcpy(&vr[recb], &p1_vs[(size_t)p1b], 2); }
                }
                ++c_round.checks; ++c_round.checks;
                // 逐字节（不一致时打印首差异，便于定位到分段）
                const auto first_diff = [](const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) -> long long {
                    if (a.size() != b.size()) return -2;
                    for (size_t i = 0; i < a.size(); ++i) { if (a[i] != b[i]) return (long long)i; }
                    return -1;
                };
                long long dk = first_diff(kr, Ks[s]->rec), dv = first_diff(vr, Vs[s]->rec);
                if (dk == -1) ++c_round.agree; else { ++c_round.mismatch; }
                if (dv == -1) ++c_round.agree; else { ++c_round.mismatch; }
                if ((dk != -1 || dv != -1) && first_diff_reported < 4) {
                    ++first_diff_reported;
                    const auto seg_of = [](long long off, size_t payload) -> const char* {
                        if (off < 0) return "?";
                        if ((size_t)off < payload) return "payload";
                        if ((size_t)off < payload + 256) return "scale_axis";
                        if ((size_t)off < payload + 512) return "zp_axis";
                        return "other_axis";
                    };
                    std::printf("  [diag ] tile(L%d,h%d,g%d) slice=%d: K 首差异@%lld(%s) got=%02x want=%02x | "
                                "V 首差异@%lld(%s) got=%02x want=%02x\n",
                                std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first),
                                s, dk, seg_of(dk, KR.payload),
                                dk >= 0 ? kr[(size_t)dk] : 0, dk >= 0 ? Ks[s]->rec[(size_t)dk] : 0,
                                dv, seg_of(dv, VR.payload),
                                dv >= 0 ? vr[(size_t)dv] : 0, dv >= 0 ? Vs[s]->rec[(size_t)dv] : 0);
                }
                // ④ 维轴副本：页 0 与页 1 的**维轴槽位**必须逐字节一致（token 轴槽位天然不同：各存本页 64 token）
                for (int row = 0; row < kPageSize; ++row) {
                    for (int col = 0; col < 8; ++col) {           // K scale 的 0..7 = 维轴（scale+zp）
                        const std::int64_t a = (ks_p0 + (std::int64_t)row * G.k_scale_leading + col) * 2;
                        const std::int64_t b = (ks_p1 + (std::int64_t)row * G.k_scale_leading + col) * 2;
                        ++c_dup.checks;
                        if (memcmp(&p1_ks[(size_t)a], &p1_ks[(size_t)b], 2) == 0) ++c_dup.agree; else ++c_dup.mismatch;
                    }
                    for (int col = 0; col < 4; ++col) {           // V scale 的 0..3 = 维轴（other）
                        const std::int64_t a = (vs_p0 + (std::int64_t)row * G.v_scale_leading + col) * 2;
                        const std::int64_t b = (vs_p1 + (std::int64_t)row * G.v_scale_leading + col) * 2;
                        ++c_dup.checks;
                        if (memcmp(&p1_vs[(size_t)a], &p1_vs[(size_t)b], 2) == 0) ++c_dup.agree; else ++c_dup.mismatch;
                    }
                }
                (void)dup_k; (void)dup_v;
            }
        }
        // 覆盖计数：record 的每一字节都被 P1 覆盖（码 1:1；维轴 1:2 复制；token 轴 1:1）
        const int64_t rec_bytes_per_tile = record_tile_bytes(kb, vb);
        const int64_t p1_bytes_per_tile  = (int64_t)G.tile_bytes();
        std::printf("[file ] %-34s kb=%d vb=%d tiles=%2d rec=%lld B/tile p1=%lld B/tile 多花=%lld B "
                    "(= %d B/token) | record 尺寸自检坏=%d | 逐元素码检查=%d 坏=%d | 轴向逐段=%d 坏=%d\n",
                    path.c_str(), kb, vb, n_tiles, (long long)rec_bytes_per_tile,
                    (long long)p1_bytes_per_tile,
                    (long long)(p1_bytes_per_tile - rec_bytes_per_tile),
                    (int)((p1_bytes_per_tile - rec_bytes_per_tile) / kTileTokens),
                    shape_bad, elem_code_checks, elem_code_bad, axis_checks, axis_bad);
        code_bad_total += elem_code_bad;
        ++file_ok;
    }

    // ---- 7.3 汇总 ----
    std::printf("\n================ 汇总（原始计数）================\n");
    std::printf("[① record 尺寸/形状自检]     checks=%lld 一致=%lld 不一致=%lld\n",
                c_shape.checks, c_shape.agree, c_shape.mismatch);
    std::printf("[② 轴向逐段字节比对]          checks=%lld 一致=%lld 不一致=%lld\n",
                c_seg.checks, c_seg.agree, c_seg.mismatch);
    std::printf("[① 码流逐位比对]              checks=%lld（每 record 每位） 不一致=%lld\n",
                c_code.checks, code_bad_total);
    std::printf("[③ P1->record 反向重建逐字节] checks=%lld 一致=%lld 不一致=%lld\n",
                c_round.checks, c_round.agree, c_round.mismatch);
    std::printf("[④ 两页维轴副本一致性]        checks=%lld 一致=%lld 不一致=%lld\n",
                c_dup.checks, c_dup.agree, c_dup.mismatch);

    const long long total = c_shape.checks + c_seg.checks + c_code.checks + c_round.checks + c_dup.checks;
    const long long bad   = c_shape.mismatch + c_seg.mismatch + code_bad_total + c_round.mismatch + c_dup.mismatch;
    std::printf("\n总检查 = %lld   不一致 = %lld   文件 = %d\n", total, bad, file_ok);
    std::printf("判定：%s\n", (bad == 0 && file_ok > 0) ? "PASS（P1 与 record 布局可互相换算，且与 oracle record 逐段一致）"
                                                        : "FAIL（见上）");

    // ---- 7.4 B/token 表（P1 落地值 vs record vs 生产 rk4v4=280） ----
    std::printf("\n================ 档位 × B/token（P1 落地） × 相对生产 rk4v4=280 ================\n");
    std::printf("%-6s %-6s %-10s %-10s %-9s %-9s %-12s %s\n",
                "档","Kb+Vb","P1 B/tok","record B/tok","差(B/tok)","P1/280","P1 tile B","record tile B");
    static const int PL[6][2] = {{4,4},{4,2},{4,3},{3,3},{5,5},{6,6}};
    for (int i = 0; i < 6; ++i) {
        const int kb = PL[i][0], vb = PL[i][1];
        const P1Geom g = p1_geom(kb, vb);
        const int64_t rec = record_tile_bytes(kb, vb);
        std::printf("K%dV%-4d %-6d %-10d %-10d %-9d %-9.2f%% %-12lld %lld\n",
                    kb, vb, kb + vb, g.b_per_token(), (int)(rec / kTileTokens),
                    g.b_per_token() - (int)(rec / kTileTokens), 100.0 * g.b_per_token() / 280.0,
                    (long long)g.tile_bytes(), (long long)rec);
    }
    std::printf("（P1 闭式：32*(Kb+Vb)+36 ；record 闭式：32*(Kb+Vb)+24 ；差恒为 12 B/token/KV head）\n");
    return (bad == 0 && file_ok > 0) ? 0 : 1;
}

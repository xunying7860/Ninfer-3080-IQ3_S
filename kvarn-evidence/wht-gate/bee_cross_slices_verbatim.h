// ============================================================================
// GATE 1 逐字取证：本文件不是手抄，由 extract_verbatim.py 按行号从原文抠出。
// 来源：kvarn-全档位移植-20261013/ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu:510-543  （beellama kvarn_wht_cross_slices（跨 slice 蝶形 + 1/sqrt(head_slices)））
// 源文件 sha256：dfaa62b40e80a2951a3bf1fd1d642e83ce00a2f691d55081a523bf1f3da3dd54
// 内容：beellama kvarn_wht_cross_slices（跨 slice 蝶形 + 1/sqrt(head_slices)）
// ============================================================================
static __device__ void kvarn_wht_cross_slices(float values[4], int head_slices) {
    if (head_slices == 1) {
        return;
    }

    for (int stride = 1; stride < head_slices; stride <<= 1) {
#pragma unroll
        for (int base = 0; base < 4; base += 2 * stride) {
            if (base + stride >= head_slices) {
                continue;
            }
#pragma unroll
            for (int i = 0; i < stride; ++i) {
                const int a_idx = base + i;
                const int b_idx = base + stride + i;
                if (b_idx >= head_slices) {
                    continue;
                }
                const float a = values[a_idx];
                const float b = values[b_idx];
                values[a_idx] = a + b;
                values[b_idx] = a - b;
            }
        }
    }

    const float scale = head_slices == 2 ? 0.7071067811865475f : 0.5f;
#pragma unroll
    for (int slice = 0; slice < 4; ++slice) {
        if (slice < head_slices) {
            values[slice] *= scale;
        }
    }
}

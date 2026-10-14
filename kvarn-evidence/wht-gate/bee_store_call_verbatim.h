// ============================================================================
// GATE 1 逐字取证：本文件不是手抄，由 extract_verbatim.py 按行号从原文抠出。
// 来源：kvarn-全档位移植-20261013/ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu:1113-1131  （beellama headwide store 的调用点（旋转 + 跨 slice））
// 源文件 sha256：dfaa62b40e80a2951a3bf1fd1d642e83ce00a2f691d55081a523bf1f3da3dd54
// 内容：beellama headwide store 的调用点（旋转 + 跨 slice）
// ============================================================================
        float values[4] = {};
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            shared[dim] = current[((int64_t) token * n_heads + head) * KVAR_N_DIM + dim];
            kvarn_wht_128(shared);
            values[slice] = shared[dim];
        }

        kvarn_wht_cross_slices(values, head_slices);

        const int stage_slot = assigned_slot >= 0 ? assigned_slot :
            (swa ? (group % stage_groups) : (group == 0 ? 0 : 1 + ((group - 1) % tail_groups)));
        const int stage_pos = stage_base + stage_slot * KVAR_N_DIM + pos;
        for (int slice = 0; slice < head_slices; ++slice) {
            const int head = head0 + slice;
            stage[((int64_t) stage_pos * n_heads + head) * KVAR_N_DIM + dim] =
                __float2half_rn(values[slice]);
        }
        __syncthreads();

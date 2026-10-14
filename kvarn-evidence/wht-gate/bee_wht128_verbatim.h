// ============================================================================
// GATE 1 逐字取证：本文件不是手抄，由 extract_verbatim.py 按行号从原文抠出。
// 来源：kvarn-全档位移植-20261013/ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu:456-472  （beellama kvarn_wht_128（128 点 Sylvester FWHT + 1/sqrt(128)））
// 源文件 sha256：dfaa62b40e80a2951a3bf1fd1d642e83ce00a2f691d55081a523bf1f3da3dd54
// 内容：beellama kvarn_wht_128（128 点 Sylvester FWHT + 1/sqrt(128)）
// ============================================================================
static __device__ void kvarn_wht_128(float * values) {
    __syncthreads();
    for (int stride = 1; stride < KVAR_N_DIM; stride *= 2) {
        if (threadIdx.x < 64) {
            const int j = (threadIdx.x / stride) * (2 * stride) + (threadIdx.x % stride);
            const float a = values[j];
            const float b = values[j + stride];
            values[j] = a + b;
            values[j + stride] = a - b;
        }
        __syncthreads();
    }
    if (threadIdx.x < KVAR_N_DIM) {
        values[threadIdx.x] *= 0.08838834764831845f;
    }
    __syncthreads();
}

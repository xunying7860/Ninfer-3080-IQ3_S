// ============================================================================
// GATE 1 逐字取证：本文件不是手抄，由 extract_verbatim.py 按行号从原文抠出。
// 来源：kvarn-全档位移植-20261013/ref-beellama/src/ggml/src/ggml-cuda/kvarn.cu:12-19  （beellama KVarN 常量（KVAR_N_DIM 等））
// 源文件 sha256：dfaa62b40e80a2951a3bf1fd1d642e83ce00a2f691d55081a523bf1f3da3dd54
// 内容：beellama KVarN 常量（KVAR_N_DIM 等）
// ============================================================================
static constexpr int KVAR_N_DIM = 128;
// Dynamic stage/tail depth is carried explicitly in op_params[7]/[8].
static constexpr int KVAR_N_TILE_VALUES = KVAR_N_DIM * KVAR_N_DIM;
static constexpr int KVAR_N_REDUCE_FLOATS = 4 * 4;
static constexpr int KVAR_N_SHARED_FLOATS = KVAR_N_TILE_VALUES + 8 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_SHARED_BYTES = KVAR_N_SHARED_FLOATS * sizeof(float);
static constexpr int KVAR_N_LOWSHMEM_FLOATS = 6 * KVAR_N_DIM + 2 + KVAR_N_REDUCE_FLOATS;
static constexpr int KVAR_N_LOWSHMEM_BYTES = KVAR_N_LOWSHMEM_FLOATS * sizeof(float);

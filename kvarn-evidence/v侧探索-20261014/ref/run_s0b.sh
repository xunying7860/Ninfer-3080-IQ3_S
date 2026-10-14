#!/bin/bash
# 阶段 0 误差归因 · 语料 B（ppl_doc2.txt，6016 token，post-RoPE 口径）
cd /data/kvarn/g128
export PYTHONPATH=/data/kvarn/pylibs19:/data/kvarn/pylibs:/data/kvarn:/data/kvarn/g128
export PPL_DOC=/data/kvarn/ppl_doc2.txt
export PPL_TOK=6016
export S0_ARMS=baseline_fp16,rk4v4_post,K4only_g128,V2only_g128,K2only_g128,V4only_g128,K4V2_g128
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec /data/comfy/venv/bin/python -u /data/kvarn/g128/stage0_run.py

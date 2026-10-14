#!/bin/bash
# run.sh —— P1 页自足几何 leading/偏移 校验：一条命令全量复现（容器内跑，无 GPU）
#   docker run --rm -v "H:/Hermes工作区/Hermes/kvarn-全档位移植-20261013/offsets-g128:/w" -w /w \
#     nvidia/cuda:13.1.2-devel-ubuntu24.04 bash -lc "./run.sh"
set -u

echo "== toolchain =="
g++ --version | head -1
echo "host: $(uname -a)"
echo

echo "== 0) 生成 oracle dump（dump_oracle.cpp dumprec，iters=0，tiles/real_g128.bin） =="
g++ -O2 -std=c++17 -w -o dump dump_oracle.cpp || exit 2
for kv in "4 4" "4 2" "4 3" "3 3" "5 5" "6 6"; do
  set -- $kv
  ./dump dumprec "data/z0_real_k$1v$2.bin" --tiles tiles/real_g128.bin --bits-k "$1" --bits-v "$2" --iters 0
done
echo

echo "== 1) 编译校验程序（纯 C++17，无 CUDA） =="
g++ -O2 -std=c++17 -w -o offsets_check offsets_g128_check.cpp || exit 2
echo "build ok"
echo

echo "== 2) 跑校验（6 档真实数据 dump） =="
./offsets_check data/z0_real_k4v4.bin data/z0_real_k4v2.bin data/z0_real_k4v3.bin \
                data/z0_real_k3v3.bin data/z0_real_k5v5.bin data/z0_real_k6v6.bin
rc=$?
echo "run rc=$rc"
exit $rc

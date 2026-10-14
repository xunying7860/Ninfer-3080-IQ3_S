# KVarN128（K4V4）引擎侧验收 · 结果汇总
时间：2026-10-14 11:56 ~ 12:48 CST ｜ 机器 x99 ｜ GPU0（基线 21 MiB）
serve md5 2532acf8fefbcca01e429095c4bdf173（build 0.12.0-rtx3090+8186a2e-dirty）
perplexity md5 eb1f413c653c5e33f64d0c84db1937f6

## 实验1 · 引擎侧 PPL（--device 0 --context 8192 --disjoint --gdn-state-fp16，仅 --kv-dtype/语料变）
语料 A = /data/kvarn/ppl_doc.txt   54,347 B / 36,685 字符 / sha256 48297f63cf8a0e34f85c51b345b76585a1d910e8ed969383af5af37dcb10592b
       引擎 input 18,992 tok → scored 16,382（2 个 8192 disjoint 窗）
语料 B = /data/kvarn/ppl_doc2.txt  32,136 B / 20,097 字符 / sha256 125363a6196747ea3b9a592643ce7b3583fd08267ceec7a5dfb3d533c502f36d
       引擎 input 10,792 tok → scored 8,191（1 个窗）

| 语料 | kvarn128 | rk4v4 | int8 | bf16 |
|---|---|---|---|---|
| A PPL | 5.848024 | 5.849927 | 5.831034 | 5.829839 |
| B PPL | 6.609538 | 6.610634 | 6.595432 | 6.588490 |

Δ_X = PPL(kvarn128)/PPL(X) − 1
- A：Δ_int8 = **+0.29137%**，Δ_rk4v4 = **−0.03253%**，Δ_bf16 = +0.31193%
- B：Δ_int8 = **+0.21388%**，Δ_rk4v4 = **−0.01658%**，Δ_bf16 = +0.31947%

判定：两份语料一致 ⇒ 略逊 int8（落在 +0.10%~+0.50%）；与生产 rk4v4 打平（≤ +0.10%）。
per-arm 峰值显存（1Hz 采样，MiB）：kvarn128 14,901 / bf16 14,739 / int8 14,493 / rk4v4 14,369。

## 实验2 · 门禁（GATE_DOC_LINES=1200、GATE_NONCE=gate-prod，9 套件）
| 臂 | ctrl-rst | rst-rst2 | toolsctrl-tools | apitoolctrl-apitool | visctrl-vis |
|---|---|---|---|---|---|
| kvarn128（ngram 开） | PASS | PASS | FAIL(R1,R4) | PASS | PASS |
| rk4v4（ngram 开） | FAIL(R4) | FAIL(R4) | PASS | PASS | PASS |
| int8（ngram 开） | PASS | PASS | FAIL(R1) | PASS | PASS |
| kvarn128（--ngram-draft-tokens 0） | PASS | PASS | FAIL(R1,R4) | PASS | PASS |
| rk4v4（--ngram-draft-tokens 0） | PASS | PASS | PASS | PASS | PASS |

五轮均无 FATAL / worker crash。profile 副本起前起后 md5 恒为 7277b862f5c7319fc534c50bbc57a436（未被重标定覆写）。

## 实验3 · 容量阶梯（可用 runtime 预留 = 7,553,613,824 B，各臂相同）
| 臂 | 65536 | 131072 | 196608 | 262144 | 317440 | 最大窗口 |
|---|---|---|---|---|---|---|
| kvarn128 | OK 2.63/1.09 GiB | FAIL | FAIL | FAIL | FAIL | **65,536** |
| rk4v4 | OK 2.58/5.42 | OK 3.74/4.26 | OK 4.90/3.10 | OK 6.06/1.93 | FAIL | **262,144** |
| int8 | OK 3.61/4.39 | OK 5.80/2.20 | FAIL | FAIL | FAIL | **131,072** |

（runtime/free 为 capacity 行原文；nvidia-smi 65536 时 kvarn128 18,987 / int8 15,611 / rk4v4 14,557 MiB）

## 实验4 · 吞吐与一致性（65536 窗口，8 题 × 2 遍，temperature=0）
Q1/Q2（1024 tok 长答）run1 均值：
| 臂 | TTFT ms | prefill tok/s | decode tok/s |
|---|---|---|---|
| kvarn128 | 159.0 | 741.0 | 89.85 |
| rk4v4 | 155.0 | 757.85 | 93.25 |
| int8 | 155.0 | 758.3 | 93.45 |

32K 长上下文（needle req#17，prompt 32,635 / output 64）引擎自报：
- kvarn128：TTFT 32.3s｜prefill 1.01k tok/s｜decode 89.6 tok/s
- rk4v4  ：TTFT 23.2s｜prefill 1.41k tok/s｜decode 99.8 tok/s
- int8   ：TTFT 23.0s｜prefill 1.42k tok/s｜decode 115.6 tok/s
⇒ 短题同级；32K 上 kvarn128 TTFT +39%、prefill −28%（超 15% 阈值）。

一致性（combined 前 200 字逐字）：同臂 run-to-run 三臂均 100%（8/8）。
- kvarn128 vs rk4v4：4/16 = 25.0%（全文 2/16）
- kvarn128 vs int8 ：8/16 = 50.0%（全文 8/16）

三针（needle，窗口 65536）：三臂 32K 档 prompt 32,635、content 空（finish=length）、reasoning 含 BLUE-FALCON-7419 ⇒ 命中；196K 档三臂均 HTTP 400（预期）。

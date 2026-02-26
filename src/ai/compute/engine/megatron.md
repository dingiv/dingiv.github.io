---
title: Megatron-LM
---

# Megatron-LM

Megatron-LM 是 NVIDIA 开发的超大规模模型训练框架，首创了张量并行技术，并系统性地提出了 3D 并行策略。它是训练 GPT-3 175B、Megatron-Turing NLG 530B 等里程碑模型的基石。

## 张量并行原理

张量并行的核心是将矩阵乘法算子切分到多个 GPU。对于 Transformer 的 MLP 层 $Y = XW$，其中 $X \in [B, S, H]$，$W \in [H, 4H]$。列并行将 $W$ 按列切分为 $W_1, W_2, \dots, W_n$，每张卡计算 $Y_i = XW_i$，最后通过 AllConcat 拼接 $Y = [Y_1, Y_2, \dots, Y_n]$。

多头注意力更适合张量并行：$n$ 个头天然可以分配到 $n$ 张卡，每张卡计算自己的 QKV 投影和注意力输出，最后通过 AllReduce 聚合。这种切分完全符合 Transformer 的数学结构，通信开销极小。

## 3D 并行

3D 并行在同一集群内同时使用三种并行策略，针对不同层次的通信特点优化拓扑。

- **节点内张量并行**：利用 NVLink 400GB/s 高带宽，通信开销低
- **节点间流水线并行**：跨节点通信少，仅在层边界通信
- **全局数据并行**：最外层包裹，通过梯度同步实现简单

例如 64 张 A100 训练 175B 模型：每节点 8 张卡，内部做 8 路张量并行（TP=8），节点间做 8 路流水线并行（PP=8），最后做 1 路数据并行（DP=1）。若模型更大，可增加 DP，此时 8 个节点各处理不同数据分片，通过 AllReduce 同步梯度。

## Sequence Parallel

长序列训练时，KV Cache 和注意力计算在序列维度上的内存和计算压力巨大。Megatron-LM 提出的 Sequence Parallel 将序列维度也进行切分，配合 Ring Attention 将通信复杂度从 $O(n^2)$ 降至 $O(n)$。这使得 128K 上下文的 GPT-3 训练成为可能。

## 使用成本

Megatron-LM 的工程复杂度远高于 DeepSpeed 和 FSDP。它要求模型代码按照特定的并行模式重写，且不支持即插即用的模型加载。但对于训练千亿级以上参数的模型，其性能优化是无可替代的。NVIDIA 的 NGC 容器预装了 Megatron-LM，可直接在 Base Command 平台上启动训练。

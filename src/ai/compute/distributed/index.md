---
title: 集群
order: 60
---

# 集群
AI 的计算需要消耗大量的计算资源，AI 引擎需要基于分布式集群为前提进行设计和实现。单张 GPU 的算力和显存有限，训练大模型（如 GPT-3 175B）需要数千张 GPU 协同工作，推理高并发请求也需要多 GPU 甚至多机集群。分布式集群涉及通信拓扑、硬件互联、集合通信等多个层面，需要在算法、系统和工程三个层面进行协同优化。

分布式 AI 训练依赖高效的通信基础设施。三种通信场景各有特点：机内通信（CPU ↔ GPU，通过 PCIe）是数据进入计算单元的通道，卡间通信（GPU ↔ GPU，通过 NVLink/NVSwitch）支撑张量并行和流水线并行，机间通信（节点 ↔ 节点，通过 InfiniBand/RoCE）承载数据并行的梯度同步。

各类通信技术的原理和协议分层分析见 [通信技术](/kernel/netlink/) 专题，本文聚焦于这些技术在 AI 训练场景中的使用方式。

## 通信模式与训练策略

不同并行策略对通信的需求差异巨大。张量并行每层前向/反向传播都需要 AllReduce，通信频率最高，因此必须使用 NVLink 这类高带宽低延迟的卡间互联。流水线并行每个 microbatch 才通信一次，频率较低，可以用 InfiniBand 承载。数据并行每个 iteration 梯度同步一次，频率最低，带宽要求也最宽松。

Megatron-LM 的 3D 并行策略是对这三种通信需求进行分层映射的典范：高频通信走 NVLink（节点内），中频走 InfiniBand（节点间），低频走以太网（跨机架）。这种"通信分层"设计的核心原则是让最贵的带宽服务最频繁的通信。

## 关键通信技术

NVLink 和 NVSwitch 是单机内多 GPU 通信的基础。NVSwitch 将多个 GPU 全互联，任意两卡之间带宽一致，消除了传统 PCIe tree 拓扑的带宽不均问题。在 H100 上，NVLink 4.0 提供 900 GB/s 的单向带宽，支持最多 8 个 GPU 的全互联。

InfiniBand 和 RoCE 是节点间通信的主力。InfiniBand NDR 提供 400 Gbps 带宽和约 1 μs 延迟，原生支持 RDMA，是大规模集群的首选。RoCE v2 在以太网上实现 RDMA，成本更低但需要配置 PFC 和 ECN 保证无损传输。

NCCL 是 GPU 集合通信的标准库。它针对 NVLink 和 InfiniBand 做了深度优化，实现了 Ring AllReduce、Tree AllReduce 等算法。NCCL 的拓扑感知能力使其能够根据 GPU 的物理连接关系自动选择最优通信路径——比如通过 NVSwitch 连接的 GPU 使用 Tree 算法，通过 PCIe 连接的 GPU 使用 Ring 算法。

## 硬件互联
集群内部的通信带宽直接影响分布式训练和推理的性能。从 GPU 到 GPU，从节点到节点，不同层次的通信技术带宽差异巨大。

| 互联技术       | 带宽                        | 延迟   | 覆盖范围          | 适用场景             |
| -------------- | --------------------------- | ------ | ----------------- | -------------------- |
| NVLink         | 450-900 GB/s                | <1 μs  | 节点内（8 卡）    | 张量并行、高频通信   |
| PCIe 5.0       | 64 GB/s                     | ~1 μs  | 节点内（CPU-GPU） | 数据传输             |
| InfiniBand NDR | 400 Gbps (50 GB/s)          | ~1 μs  | 节点间            | 数据并行、跨节点通信 |
| RoCE v2        | 100-200 Gbps (12.5-25 GB/s) | ~2 μs  | 节点间            | 以太网 RDMA          |
| 以太网         | 25-100 Gbps (3-12 GB/s)     | ~10 μs | 节点间            | 成本敏感场景         |

NVLink 是 NVIDIA 的 GPU 间高速互联技术，带宽远超 PCIe，适用于节点内的高频通信（如张量并行）。InfiniBand 是数据中心级的高性能网络，带宽接近 PCIe、延迟低，适用于跨节点通信。RoCE（RDMA over Converged Ethernet）允许在以太网上进行 RDMA，成本低于 InfiniBand 但性能也略低。

## 通信层次
分布式训练和推理的通信可分为三个层次：机内通信（GPU-GPU）、机间通信（节点-节点）、跨机房通信（数据中心-数据中心）。

机内通信通过 NVLink 或 PCIe 完成，带宽高延迟低，适合张量并行等高频通信场景。机间通信通过 InfiniBand 或以太网完成，带宽较低，适合数据并行等低频通信场景。跨机房通信用于跨数据中心训练（如联邦学习），带宽最低且延迟最高，需要专门的优化算法（如梯度压缩、异步训练）。

## 通信优化
通信与计算重叠（overlap）是将通信时间隐藏在计算时间中的关键技术。DeepSpeed 的梯度预取在前向传播计算层 i 的梯度时，预取层 i+1 的参数到显存，同时同步层 i-1 的梯度，实现三级流水。梯度累积是简单示例：在前向传播计算 mini-batch 1 的同时，同步 mini-batch 0 的梯度。

拓扑感知通信根据网络拓扑优化通信路径。8 张 GPU 组成的单机，ring allreduce 比树状通信效率更高；64 台机器组成的集群，hierarchical allreduce（节点内用 ring，节点间用 tree）最优。NCCL 会自动检测拓扑，选择最优通信算法。

## 通信库
NCCL（NVIDIA Collective Communications Library）是 NVIDIA 提供的集合通信库，针对 NVIDIA GPU 和网络优化，性能远高于开源实现（如 Gloo、MPI）。RCCL 是 AMD GPU 的对应实现。Gloo 是 PyTorch 的通用通信后端，支持 CPU 和 GPU，适合开发测试环境。

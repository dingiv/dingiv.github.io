---
title: 通信链路
order: 65
---

# 通信链路
分布式 AI 系统的通信链路分为机内通信（GPU 互联）和机间通信（网络互联）两类。机内通信带宽高延迟低，机间通信带宽低延迟高。合理规划通信拓扑是分布式系统设计的基础，通信带宽往往成为分布式训练的瓶颈。

## 机内通信
机内通信指同一服务器内 GPU 之间的数据传输，主要通过 PCIe 总线或专用高速互联实现。

### PCIe
PCIe（Peripheral Component Interconnect Express）是通用的总线标准，用于 CPU 与 GPU 之间的数据传输。PCIe 5.0 x16 带宽为 64 GB/s，延迟约 1 μs。PCIe 的优势是兼容性好，几乎所有的服务器和 GPU 都支持；劣势是带宽远低于专用互联技术（如 NVLink），且需要经过 CPU，增加了数据拷贝开销。

PCIe 的典型用途是 CPU 与 GPU 间的数据传输：训练数据从磁盘加载到 CPU 内存，然后通过 PCIe 拷贝到 GPU 显存；训练完成后，模型从 GPU 拷贝回 CPU 保存。对于多 GPU 训练，PCIe 也可用于 GPU 间通信，但带宽瓶颈明显。

## 机间通信

机间通信指不同服务器之间的数据传输，主要通过高性能网络实现。

### 以太网

以太网是最通用的网络技术，成本较低但性能有限。25 Gigabit 以太网带宽为 3.125 GB/s，100 Gigabit 以太网带宽为 12.5 GB/s。以太网的延迟通常在 10-100 μs，远高于机内通信。以太网的优势是兼容性好、成本低，适合预算敏感的中小规模集群。但对于大规模分布式训练，以太网的带宽瓶颈明显。

以太网在 AI 集群中的应用包括：数据并行梯度同步（可通过梯度压缩减少通信量）、模型并行的前向/反向传播（通信频率低，带宽要求较低）、推理服务的负载均衡。

### InfiniBand

InfiniBand 是高性能网络技术，专为数据中心设计。InfiniBand HDR（200 Gbps）带宽为 25 GB/s，NDR（400 Gbps）带宽为 50 GB/s，下一代 XDR 将达到 800 Gbps（100 GB/s）。InfiniBand 的延迟低至 1 μs，与机内通信相当。

InfiniBand 的优势在于高带宽低延迟，且支持 RDMA（Remote Direct Memory Access），允许应用直接访问远程内存，无需经过 CPU。这使得 InfiniBand 非常适合大规模分布式训练。NVIDIA 的 DGX SuperPOD 就使用 InfiniBand NDR 进行节点间通信，在 512 张 A100 上训练 GPT-3 175B。

InfiniBand 的劣势是成本高、配置复杂。需要专门的网卡（ConnectX）、交换机（Quantum）、线缆，且需要配置 IPoIB（IP over InfiniBand）或 RDMA 协议。对于预算有限的团队，InfiniBand 可能过于昂贵。

### RDMA 和 RoCE

RDMA（Remote Direct Memory Access）是一种直接访问远程内存的技术，无需经过远程 CPU。RDMA 由网卡硬件实现，通信双方建立连接后，可直接读写远程内存，延迟低且 CPU 开销小。

InfiniBand 原生支持 RDMA。以太网上的 RDMA 称为 RoCE（RDMA over Converged Ethernet），RoCE v2 带宽约 100-200 Gbps（12.5-25 GB/s），RoCE v3 将进一步提升到 400 Gbps。RoCE 的优势在于基于以太网，成本低于 InfiniBand；劣势是依赖无损网络（lossless network），需要配置 PFC（Priority Flow Control）和 ECN（Explicit Congestion Notification）。

### 无带宽缩放

Scaling Law 表明，模型的性能与计算量（FLOPs）和训练数据量密切相关。但分布式训练的通信开销随着 GPU 数量增加而增加，如果通信带宽不足，增加 GPU 可能无法线性提升训练速度。这就是"通信墙"（communication wall）问题。

为了突破通信墙，需要采用通信与计算重叠（overlap）、梯度压缩、拓扑感知通信、混合精度训练等技术。Megatron-DeepSpeed 的 3D 并行策略就是针对通信优化的典范：节点内使用 NVLink 进行张量并行（高频通信），节点间使用流水线并行（低频通信），最外层使用数据并行（最低频通信）。

## 加速卡互联
除了 NVIDIA 的 NVLink，其他厂商也有各自的加速卡互联技术。AMD 的 Infinity Fabric 用于 GPU 间和 CPU-GPU 间通信，带宽约 200-400 GB/s。Intel 的 CXL（Compute Express Link）是新一代互联标准，支持 CPU 和加速器间的内存共享，带宽可达 128 GB/s。Intel 的 Xeon Phi（KNM）使用 KNL 互联，带宽约 100 GB/s。

华为昇腾 NPU 的 HCCS（Huawei Cube Collective Communication on Scale）互联带宽约 200-300 GB/s，用于昇腾 910 系列 NPU 间通信。Google TPU 的 ICI（Inter-Chip Interconnect）互联带宽约 600 GB/s，用于 TPU Pod 内的 TPU 芯片间通信。

### NVLink
NVLink 是 NVIDIA 的 GPU 间高速互联技术，带宽远超 PCIe。A100 的 NVLink 3.0 带宽为 450 GB/s（单向），H100 的 NVLink 4.0 带宽为 900 GB/s。NVLink 是双向的，两个方向可同时传输数据，总带宽翻倍。

NVLink 的优势在于高带宽低延迟，且无需经过 CPU，GPU 间直接通信。这使得 NVLink 非常适合张量并行（tensor parallel）等高频通信场景。Megatron-LM 的张量并行就依赖 NVLink 来实现高效的 allreduce 通信。

NVLink 的拓扑通常采用全连接（每个 GPU 与其他所有 GPU 直连）或环形连接（每个 GPU 只与相邻 GPU 连接）。全连接带宽最高但成本高，环形连接成本较低但通信可能需要多跳。8 卡 GPU 的服务器通常采用 NVLink Switch（全连接），16 卡以上的服务器可能采用混合拓扑。

### CXL

---
title: 集合通信
---

# 集合通信

分布式训练和推理需要在多张 GPU 甚至多台机器间协同工作，这带来了通信开销。集群通信研究如何高效地在设备间传输数据，将通信与计算重叠，最小化通信对性能的影响。

## 通信原语
集合通信（collective communication）定义了多个进程间的协同通信模式，是分布式计算的基石。最常见的原语包括：

- **Broadcast**：根节点将数据广播到所有节点，用于同步模型权重或超参数
- **AllReduce**：所有节点计算数据的和/最大值/最小值，并将结果分发到所有节点，用于数据并行的梯度同步
- **ReduceScatter**：所有节点计算数据的和，然后将结果切分到不同节点（节点 i 获得分片 i），是 ZeRO-2 的核心通信模式
- **AllGather**：所有节点将自己的数据分片发送到所有节点，拼接为完整数据，是 ZeRO-3 的核心通信模式
- **Send/Recv**：点对点通信，用于流水线并行的层间数据传递

这些原语的性能受限于网络带宽和延迟。A100 的 NVLink 带宽为 450GB/s，PCIe 5.0 为 64GB/s，InfiniBand NDR400 为 400Gbps（50GB/s）。跨节点通信的带宽远低于节点内，因此分布式训练会尽量将张量并行（通信频繁）放在节点内（利用 NVLink），数据并行（通信频率低）放在节点间。

## 通信优化
通信与计算重叠（overlap）是将通信时间隐藏在计算时间中的关键技术。梯度累积是简单示例：在前向传播计算 mini-batch 1 的同时，同步 mini-batch 0 的梯度，两者并行执行。DeepSpeed 的梯度预取进一步优化：在反向传播计算层 $i$ 的梯度时，预取层 $i+1$ 的参数到显存，同时同步层 $i-1$ 的梯度，实现三级流水。

通信压缩通过减少传输数据量来降低通信时间。梯度稀疏化仅传递绝对值大于阈值的梯度（95% 的梯度接近零），量化将 FP32 梯度压缩为 FP16 甚至 INT8。TopK 保留梯度中绝对值最大的 K 个元素，其余置零，解压时使用动量（momentum）补偿被丢弃的梯度。这些方法可将通信量降低 4-10 倍，但可能影响收敛速度，需要调整学习率。

拓扑感知通信根据网络拓扑优化通信路径。例如 8 张 GPU 组成的单机，ring allreduce（环状通信）比 tree allreduce（树状通信）效率更高；而 64 台机器组成的集群，fat-tree（胖树）拓扑下 hierarchical allreduce（分层通信，节点内用 ring，节点间用 tree）最优。NCCL 会自动检测拓扑，但手动配置（如 `NCCL_SOCKET_IFNAME` 指定网络接口）可进一步提升性能。

## NCCL
NVIDIA Collective Communications Library (NCCL) 是 NVIDIA 提供的集合通信库，针对 NVIDIA GPU 和网络（NVLink、InfiniBand、Ethernet）优化。它实现了上述所有通信原语，且针对不同 GPU 架构（Volta、Ampere、Hopper）和不同网络拓扑（单机、多机、跨机房）分别优化，性能远高于开源实现（如 Gloo、MPI）。

NCCL 的核心优势是**自动调优**。初始化时，NCCL 会检测硬件拓扑（GPU 间的 NVLink 连接、节点的 InfiniBand 配置），选择最优通信算法（ring/tree/hierarchical），并在运行时动态调整通信粒度（chunk size）以隐藏延迟。这种"即插即用"的设计使得开发者无需关心底层通信细节，只需调用 `ncclAllReduce` 即可获得接近理论峰值的性能。

NCCL 的另一个优势是**与 CUDA 集成**。通信 kernel 可与计算 kernel 共享 CUDA stream，实现通信与计算的细粒度重叠。PyTorch DDP、DeepSpeed、Megatron-LM 都使用 NCCL 作为后端，通过 `torch.distributed.all_reduce` 调用 NCCL，无需手动编写 CUDA 代码。

## 通信库选择
| 通信库 | 优势               | 劣势                 | 适用场景                   |
| ------ | ------------------ | -------------------- | -------------------------- |
| NCCL   | 性能最优，自动调优 | 仅支持 NVIDIA GPU    | NVIDIA GPU 集群            |
| Gloo   | 开源，支持 CPU/GPU | 性能低于 NCCL        | 开发测试、CPU 训练         |
| MPI    | 成熟稳定，支持异构 | 配置复杂，需手动调优 | 超大规模集群（1000+ 节点） |

对于 NVIDIA GPU 集群，NCCL 是唯一选择。对于 AMD GPU，RCCL（Radeon Collective Communications Library）是 NCCL 的替代品。对于 CPU 训练或开发环境，Gloo 提供了足够的性能和更好的兼容性。

## 通信性能分析
集合通信的理论性能可通过带宽-延迟模型分析。对于 ring allreduce，$T = 2 \times (n-1) \times (\alpha + \beta \times m / n)$，其中 $n$ 为节点数，$m$ 为数据量，$\alpha$ 为延迟，$\beta$ 为带宽倒数。当 $m$ 较小时，延迟主导；当 $m$ 较大时，带宽主导。因此，小张量（如层归一化参数）的同步效率低，大张量（如全连接层权重）的同步效率高。

实际性能可通过 `torch.distributed.collective_profiler` 或 NCCL 的 `NCCL_DEBUG=INFO` 环境变量分析。如果通信时间占比超过 30%，说明通信成为瓶颈，需要考虑增加梯度累积步数（减少通信频率）、使用模型并行（减少跨节点通信）或升级网络硬件（从 Ethernet 升级到 InfiniBand）。

## 流水线并行通信
流水线并行的通信模式与数据并行不同。数据并行在每次迭代后同步梯度，通信集中在迭代末尾；流水线并行在每个 micro-batch 的层间传递激活值（前向传播）和梯度（反向传播），通信分散在整个迭代中。这使得流水线并行的通信与计算更容易重叠，但引入了气泡（bubble）空转——下游 GPU 等待上游 GPU 完成的空闲时间。

1F1B（One Forward One Backward）调度通过交错不同 micro-batch 的前向和反向传播来填充气泡。具体来说，GPU 1 在处理 micro-batch 1 的反向传播时，GPU 2 可同时处理 micro-batch 2 的前向传播，两者并行执行。这将气泡从 $O(n \times t)$ 降低到 $O(t)$，其中 $n$ 为流水线深度，$t$ 为单层计算时间。PipeDream-Flush、PipeDream-2BW 是早期的流水线调度优化，Megatron-LM 的 interleaved pipeline 进一步改进了 1F1B，将气泡进一步降低到近乎消失。

流水线并行的通信量小于数据并行（每个 micro-batch 只需传递层间激活值，而非完整梯度），但通信频率更高（每层都需要通信）。因此，流水线并行更适合跨节点部署（节点间带宽低），而数据并行更适合节点内部（节点内带宽高）。Megatron-DeepSpeed 的 3D 并行正是基于这一洞察：节点内张量并行（高带宽 NVLink），节点间流水线并行（低带宽但通信量小），最外层数据并行（最简单且容错）。

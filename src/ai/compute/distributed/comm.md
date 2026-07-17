---
title: 集合通信
---

# 集合通信
分布式训练和推理需要在多张 GPU 甚至多台机器间协同工作，这带来了通信开销。集群通信研究如何高效地在设备间传输数据，将通信与计算重叠，最小化通信对性能的影响。

集合通信主要针对于多张 GPU 间的通信进行讨论

## 通信原语

集合通信（collective communication）定义了多个进程间的协同通信模式，是分布式计算的基石。五种基本原语各自有不同的数据流动方向、触发时机和应用场景，理解它们有助于把握分布式训练的通信开销来源。

### AllReduce

AllReduce 是最核心的集合通信原语。所有参与节点各自持有一部分数据，通信完成后每个节点都得到全局归约结果（和、最大值、最小值等）。

在数据并行中，每个 GPU 处理不同的 mini-batch，计算出各自的梯度。AllReduce 将所有 GPU 的梯度求和后分发回每个 GPU，使各 GPU 拥有相同的全局梯度。触发时机在每个 iteration 的反向传播完成后。以 8 卡数据并行为例，每张卡计算出本地梯度 $g_i$，AllReduce(sum) 后每张卡得到 $\sum g_i$，然后各自更新权重。

在张量并行的行并行中，权重矩阵按行切分到不同 GPU，每张卡计算 $Y_i = X W_i$。AllReduce 将所有卡的 $Y_i$ 求和得到完整输出 $Y = \sum Y_i$。触发时机在每层的前向和反向传播中——张量并行的通信是嵌入在计算图中的，每一层都要触发。

AllReduce 的两种主要算法：Ring AllReduce 将所有 GPU 排成环形，数据沿环传递并累加，时间复杂度 $O(N)$，适合节点内 GPU 数量较少时；Tree AllReduce 构建树形拓扑，先在子树内归约再向上传播，时间复杂度 $O(\log N)$，适合节点数量多时。NCCL 会根据 GPU 拓扑自动选择：NVSwitch 全互联时用 Tree，环形连接时用 Ring。

### ReduceScatter

ReduceScatter 执行归约后将结果切分，每个节点只得到结果的一部分。与 AllReduce 的区别在于：AllReduce 最终每个节点都有完整结果，ReduceScatter 每个节点只持有结果的 $1/N$。

在数据并行混合 ZeRO-2 时，ReduceScatter 替代了 AllReduce。每个 GPU 计算出完整梯度后，通过 ReduceScatter 将梯度归约并按参数分片分发——GPU 0 得到参数分片 0 的全局梯度，GPU 1 得到分片 1 的全局梯度，以此类推。每张卡只保留自己负责的那部分梯度的全局归约结果，其余丢弃。触发时机与 AllReduce 相同（反向传播完成后），但通信量减少了 $(N-1)/N$。

在张量并行的列并行中，输入 $X$ 按列切分到不同 GPU——但列并行的输入不需要切分，每张卡都有完整的 $X$。权重 $W$ 按列切分，输出 $Y = XW$ 自然也被切分——GPU i 持有 $Y$ 的列分片 i。如果下一层需要完整 $Y$，就要通过 AllGather 拼接。

### AllGather

AllGather 是 ReduceScatter 的逆操作。每个节点持有数据的一个分片，通信完成后所有节点拥有完整拼接数据。

在 ZeRO-3 中，模型参数被分片存储。前向传播计算到第 $k$ 层时，每张卡只有该层权重的 $1/N$，需要通过 AllGather 从其他 GPU 拉取该层的剩余权重分片，拼接为完整权重矩阵。计算完成后立即释放拉取的部分。触发时机在每个 layer 的前向计算前和反向计算前——ZeRO-3 的通信是"按需获取"模式，每层计算前都要触发一次 AllGather。

在张量并行的列并行中，上一层的输出 $Y$ 被切分在各 GPU 上，如果下一层是行并行需要完整的输入 $X$（即 $Y$），就要通过 AllGather 将切分的 $Y$ 拼接为完整输入。这是张量并行中列并行接行并行的标准模式。

通信量方面，AllGather 传输的数据量为 $(N-1)/N \times M$（每个节点只缺少其余 $N-1$ 个分片），与 ReduceScatter 对称。

### Broadcast

Broadcast 是最简单的集合原语：根节点将数据发送给所有其他节点，通信完成后所有节点持有相同数据。

在分布式训练的初始化阶段，Broadcast 用于将模型权重从 rank 0 广播到所有 GPU，确保训练的初始状态一致。触发时机在训练开始时，仅执行一次。

在数据并行中，如果使用 BatchNorm 且跨 GPU 同步统计量（running mean/variance），会通过 Broadcast 将计算出的统计量分发到所有 GPU。触发时机在每个 iteration 的 forward 中，频率较低。

Broadcast 的通信量为 $M$（根节点发送，其余节点接收），是集合通信中通信量最小的原语。

### Send/Recv（点对点通信）

Send/Recv 是一对一的通信原语，与前面四种"集体"通信不同。发送方指定目标 rank 发送数据，接收方指定来源 rank 接收数据。

在流水线并行中，模型被按层拆分到不同 GPU。前向传播时，GPU $i$ 计算完本段的最后一层后，通过 Send 将激活值发送给 GPU $i+1$，GPU $i+1$ 通过 Recv 接收后继续计算。反向传播时方向相反，GPU $i+1$ 将梯度通过 Send 发回给 GPU $i$。触发时机在每个 microbatch 的段边界——以 1F1B 调度为例，每个 microbatch 的 forward 结束时触发一次 Send/Recv，backward 结束时再触发一次。

Send/Recv 的通信量为 $B \times H$（batch size × hidden dimension），与模型层数无关。这也是流水线并行通信量低于张量并行的原因——张量并行每层都要通信完整矩阵，流水线并行只在段边界通信激活值。

### 原语总结

| 原语 | 数据流 | 通信量 | 典型触发场景 |
|------|--------|--------|-------------|
| AllReduce | 归约后全员分发 | $2(N-1)/N \times M$ | 数据并行梯度同步、TP 行并行输出合并 |
| ReduceScatter | 归约后切分分发 | $(N-1)/N \times M$ | ZeRO-2 梯度分片、ZeRO-3 梯度同步 |
| AllGather | 拼接后全员分发 | $(N-1)/N \times M$ | ZeRO-3 参数按需拉取、TP 列并行输出拼接 |
| Broadcast | 根→所有 | $M$ | 初始化权重复制、BatchNorm 统计量同步 |
| Send/Recv | 点对点 | $B \times H$ | 流水线并行层间激活值/梯度传递 |

通信原语的选择直接决定了训练的通信瓶颈在哪。张量并行用 AllReduce + AllGather，特点是高频但数据量可控（单层矩阵大小）；ZeRO-3 用 AllGather + ReduceScatter，特点是有参数分片粒度控制通信量；流水线并行用 Send/Recv，特点是低频且通信量小但引入了计算气泡。3D 并行的核心思路就是将每种通信原语分配到最适合它的硬件层次上。

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

# KV Cache

KV Cache 是 Transformer 推理阶段的核心优化技术，通过缓存历史 Token 的 Key 和 Value 矩阵，避免了重复计算，将生成每个新 Token 的计算复杂度从 $O(n^2)$ 降至 $O(n)$。对于长序列推理，KV Cache 的显存占用往往超过模型权重本身，成为推理性能的主要瓶颈。

## 基本原理
在 Transformer 推理的 Decode 阶段，每生成一个新 Token，都需要和之前所有的 Token 进行 Attention 计算。Attention 机制的核心公式是 $\text{Attention}(Q, K, V) = \text{softmax}(QK^T/\sqrt{d_k})V$，其中 $Q$ 是当前 Token 的 Query，$K$ 和 $V$ 是所有历史 Token 的 Key 和 Value。

如果不使用 KV Cache，生成第 $n$ 个 Token 时需要重新计算前 $n-1$ 个 Token 的 $K$ 和 $V$ 矩阵，计算量随序列长度呈 $O(n^2)$ 增长。使用 KV Cache 后，只需计算当前 Token 的 $k_{new}$ 和 $v_{new}$，并将其追加到之前的矩阵中。公式变为 $\text{Attention}(q_{new}, [K_{old}, k_{new}], [V_{old}, v_{new}])$，每次生成只需计算当前 Token 的 Attention，复杂度降至 $O(n)$。

### 显存占用

KV Cache 的显存占用与序列长度、batch size、隐藏层维度、层数成正比。对于 Llama-2-7B 模型，隐藏层维度为 4096，32 层，序列长度 2048，batch size 32 时，FP16 精度的 KV Cache 约需 8GB 显存，超过模型权重本身（约 14GB）。当序列长度扩展到 32K 时，KV Cache 占用会增至 128GB 以上，远超单卡容量。

## vLLM 的 PagedAttention

传统推理引擎将每个请求的 KV Cache 作为连续显存块管理，这要求在推理开始前预分配足够大的显存块。但序列长度无法预测——预分配过多浪费显存，预分配过少则生成中断。vLLM 通过 PagedAttention 解决了这个问题，借鉴了操作系统的虚拟内存管理思想。

### 逻辑块与物理块

PagedAttention 将 KV Cache 分页，每页固定大小（如 16 个 Token）。每个请求的 KV Cache 是一组页面的链表，页面可分散在显存任意位置。逻辑块是模型认为的连续数组，物理块是 vLLM 在显存里维护的固定大小 Block Pool。映射表（Block Table）类似于操作系统的页表，记录逻辑 Token 索引对应的物理显存地址。

这种设计消除了预分配问题，按需申请页面即可。也解决了内存碎片，因为页面大小统一，可自由复用。

### 动态分配与共享

当一个请求生成到第 17 个 Token 时（假设块大小为 16），vLLM 才从空闲队列中分配第二个物理块。如果两个请求有相同的 System Prompt（前缀），它们可以共享同一组物理块。只有当其中一个请求开始生成不同的内容时，才会触发 Copy-on-Write（写时复制）。

### 前缀缓存优化

vLLM 使用基数树（Radix Tree）来存储已计算的前缀，匹配时间复杂度为 $O(1)$。对于客服机器人等场景（所有用户共享相同的系统 prompt），首 Token 延迟可降低 50% 以上。当新请求到达时，Scheduler 会检查其 prompt 前缀是否与已有请求的前缀匹配，如果匹配直接复用已有请求的 KV Cache。

## 显存管理策略

KV Cache 的显存管理是推理引擎的核心挑战。当显存不足时，引擎需要决定哪些 Cache 保留、哪些释放、哪些交换到 CPU。

### 缓存驱逐策略

常见的驱逐策略包括 LRU（Least Recently Used）和 FIFO（First In First Out）。LRU 优先淘汰最久未使用的请求 Cache，适合突发流量场景。FIFO 按请求到达顺序淘汰，实现简单但可能淘汰活跃请求。更高级的策略会考虑请求的优先级、剩余生成长度、重新计算成本等因素。

### CPU 卸载

当 GPU 显存不足时，vLLM 可以将暂时不活跃的 KV Cache 交换到 CPU 内存。底层使用异步 IO 和预取逻辑，在调度器预测到该请求即将执行时，提前将其换入显存。这类似于操作系统的 swap 机制，实现了用时间换空间。

### 多级缓存

现代推理引擎支持多级缓存：GPU → CPU → NVMe SSD。GPU 缓存最快但容量有限，CPU 内存容量较大但带宽较低（PCIe 3.0 x16 约 12GB/s，远低于 GPU 显存带宽的 2TB/s），NVMe SSD 容量最大但延迟最高。引擎需要根据请求的热度自动在不同层级间迁移数据。

## 分布式 KV Cache

在多机多卡场景下，KV Cache 变成了分布式的，需要考虑数据分片和通信开销。

### 张量并行下的 KV Cache

在张量并行（Tensor Parallel）模式下，每个 GPU 只存储一部分 Attention Head 的 KV Cache。计算 Attention 时，每张卡只计算自己负责的 Head，然后通过 AllReduce 聚合结果。这种分片方式通信量小，但要求每张卡都有完整的序列 KV Cache。

### 序列并行下的 KV Cache

在序列并行（Sequence Parallel）模式下，KV Cache 在序列维度上切分到多张卡。每张卡只存储部分序列的 KV Cache，计算时通过 Ring Attention 在环状拓扑上通信。这避免了完整序列的 KV Cache 存储，将显存占用从 $O(n^2)$ 降至 $O(n^2/p)$（$p$ 为卡数），但通信开销更大。

### Prefill 与 Decode 分离

2026 年的高性能集群倾向于让一部分卡专做 Prefill（处理新请求的前向传播，吞吐型），另一部分卡专做 Decode（生成 Token，延迟型）。引擎通过高速网络协议（如基于 RDMA 的 Nixl 库）在不同节点间实时同步 KV Block。这种分离架构可以针对不同阶段优化硬件配置，Prefill 节点使用高带宽显存，Decode 节点使用高容量显存。

## 硬件加速

KV Cache 的读写是推理性能的关键瓶颈，现代硬件通过多种方式加速。

### FlashAttention

FlashAttention 通过分块计算减少显存访问次数，将显存带宽利用率从 20% 提升到 80% 以上。它不将完整的 Attention 矩阵加载到显存，而是分块加载到 SRAM 中计算，充分利用 GPU 的片上缓存。对于长序列（8K+），性能提升尤为显著。

### PagedAttention Kernel

vLLM 使用专门优化的 PagedAttention Kernel（通常由 Triton 或 CUTLASS 编写）。这些 Kernel 能够直接读取离散的物理块地址，在 GPU 寄存器中完成流水线化的 Attention 计算，避免了将零散块拼凑成大矩阵的昂贵显存拷贝。Kernel 通过 tiling 和 shared memory 优化，将分页开销降到最低。

### INT8 量化

KV Cache 量化是降低显存占用的有效手段。将 FP16 的 KV Cache 量化为 INT8 可将显存减半，配合 INT8 算子可实现接近原始精度的性能。量化策略包括静态量化（使用固定 scale）和动态量化（根据激活值动态调整 scale）。动态量化精度更高但需要额外的量化计算开销。

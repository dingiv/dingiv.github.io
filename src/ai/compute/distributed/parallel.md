# 模型拆分

模型拆分是突破单卡显存限制的关键技术。当一个大型模型（如 Llama-3-70B 或 DeepSeek-V3）的显存占用超过单张显卡的容量（H100 为 80GB，而 70B 模型 FP16 权重就需 140GB+），必须将模型切分到多卡甚至多机。从分布式系统的视角来看，这是将一个巨大的计算图（DAG）进行跨节点/跨进程的解耦与流水线化。

目前主流的拆分方式有四种：数据并行、张量并行、流水线并行、专家并行。

## 数据并行

数据并行是最基础的分布式策略。每张 GPU 持有完整模型副本，处理不同的 mini-batch 数据。反向传播后，所有 GPU 通过 AllReduce 将梯度求和，使各 GPU 上的模型保持一致。通信量取决于模型参数量 $P$，每个 iteration 在反向传播完成后触发一次 AllReduce。

梯度累积是数据并行最常用的优化——多个 micro-batch 的梯度在本地累加，达到累积步数后才触发一次同步，减少通信频率。FSDP 是数据并行的升级版，将参数/梯度/优化器状态全部切分到各 GPU，前向通过 AllGather 按需拉取，反向通过 ReduceScatter 分片梯度。通信模式从集中的 AllReduce 变为分散的 AllGather + ReduceScatter，更容易与计算重叠。

数据并行的目的在于将多张较弱算力的卡的算力集中起来，大家一起算一个问题的不同部分，最后将结果进行汇总，从而加速计算。数据并行简单且通信频率低，适合 GPU 数量多但模型规模适中（单卡能装下完整模型）的场景。在实践中通常作为 3D 并行的最外层。

## 张量并行

张量并行将模型每一层内部的巨大矩阵切开，分配到不同显卡上。以 Transformer 中的全连接层 $Y = XW$ 为例，有两种切分方式。

**行并行**（row-based parallel）将权重矩阵 $W$ 按行切分，每张卡计算一部分乘加，最后通过 AllReduce 将各卡结果求和。**列并行**将权重矩阵 $W$ 按列切分，每张卡输出 $Y$ 的一部分通道，最后通过 AllGather 拼接。

张量并行是通信频率最高的策略——每一层前向和反向都需要 AllReduce/AllGather——因此必须使用 NVLink 这类高带宽低延迟的卡间互联。vLLM 在推理中使用列并行作为第一层、行并行作为第二层，中间结果无需通信，仅在输出层同步。

## 流水线并行

流水线并行将模型的层拆分到不同机器。例如 80 层 Transformer Block 拆为：第 1-20 层放 0 号机，21-40 层放 1 号机，以此类推。数据像流水线一样流动——GPU $i$ 算完本段的最后一层后，通过 Send/Recv 将激活值发送给 GPU $i+1$。

流水线并行的通信量小于张量并行（每个 microbatch 只需在段边界传递激活值，而非每层都通信完整的矩阵），但引入了气泡问题——后端 GPU 等待前端 GPU 时空闲。1F1B（One Forward One Backward）调度通过交错不同 microbatch 的前向和反向来填充气泡，将气泡从 $O(n \times t)$ 降低到 $O(t)$。

流水线并行更适合跨节点部署（节点间带宽低但通信量小），而张量并行更适合节点内部（带宽高）。Megatron-DeepSpeed 的 3D 并行中，流水线并行位于中间层。

## 专家并行

专家并行是 MoE（混合专家）模型的专属并行策略。MoE 层包含多个 Expert（如 8-160 个独立 FFN），每个 token 通过 Router 只激活其中 top-k 个。专家并行将不同专家放置在不同 GPU 上，通过 All-to-All 通信将 token 路由到目标专家。

与前面三种策略不同，专家并行的通信是 All-to-All（全交换）而非 AllReduce。通信量为 $B \times H \times k$（batch × hidden × top-k），当专家数量很大时可能成为瓶颈。DeepSeekMoE 引入了共享专家（所有 token 始终经过）和细粒度专家切分来提升专家利用率和负载均衡。

专家并行可与数据并行组合：数据并行组内做专家并行（All-to-All），组间做数据并行（AllReduce），减少 All-to-All 的参与节点数。

## 推理部署

推理的核心指标是首 token 延迟（TTFT）和每 token 生成时间（TPOT）。vLLM 是当前分布式推理的事实标准，它的张量并行实现可以视为工程落地的最佳参考。

vLLM 使用列并行-行并行交织设计：Transformer 的 Attention 层用列并行（权重按列切分，输出自然分片），后续的 FFN 层用行并行（权重按行切分，AllReduce 求和）。列并行和行并行交替排列，使得层间的中间结果无需通信——列并行的输出已经按列切分，恰好是行并行的自然输入。仅在 Attention 和 FFN 的边界处触发一次 AllReduce。这种设计将通信频率降到最低，充分利用了 NVLink 的高带宽。

vLLM 的底层实现基于 SPMD（Single Program Multiple Data）模型：每个 GPU 运行完全相同的代码，通过 `torch.distributed` 获取自己的 rank，据此决定处理权重矩阵的哪一部分。为了抵消频繁 Launch Kernel 带来的 CPU 开销，vLLM 使用 CUDA Graphs 将整条分布式计算路径录制为一张图，后续迭代直接回放，消除 kernel launch 延迟。

不同场景下的策略选择：

单用户实时对话对延迟最敏感。用户期望 200-500ms 首字延迟，30-50 token/s 生成速度。小模型（7B 及以下）单张消费级 GPU（24GB）即可，不需并行。中等模型（13B-34B）单张 A100/H100 可运行，长上下文时优先 KV Cache 量化而非模型并行——量化省显存且零通信开销。大模型（70B+）单卡装不下 FP16 权重，必须用张量并行：2×H100 TP=2 放 70B，4×H100 TP=4 支持 128K 上下文。

多用户高并发对吞吐量敏感。vLLM 的 continuous batching 在 TP 组内动态调度请求。当单组 GPU 吞吐量不够时，水平扩展多组独立推理实例（组内 TP，组间各自独立服务），通过负载均衡分发。

大规模离线推理对延迟不敏感，吞吐为王。将每张 GPU 作为独立推理单元，通过量化装入完整模型，各自处理不同数据批次——完全消除跨卡通信，吞吐量最大化。

vLLM 的张量并行实现基于 SPMD 编程模型，每个显卡运行完全相同的代码，但处理不同的数据分片。在每一层计算结束时，通过 NCCL 进行跨卡的集合通信。

vLLM 通常为每张显卡 fork 一个独立进程，通过 Ray 或 PyTorch Distributed 管理。进程间通过共享内存交换控制信息，通过 NCCL 传输张量数据。为了抵消分布式环境下频繁 Launch Kernel 带来的 CPU 开销，vLLM 会使用 CUDA Graphs 将整条分布式路径录制下来。

### 分布式 PagedAttention
切分模型只是第一步，分布式场景下最难的是 KV Cache 的管理。vLLM 采用集中式调度，有一个 Master 节点负责计算每个请求的 Token 应该放在哪张卡的哪个物理块上。各卡在自己的显存空间里找到对应的局部 KV Cache 进行计算，然后通过 TP 进行结果同步。

### 流水线并行
当张量并行达到 8 卡（单机上限）依然放不下模型时，就需要跨机使用流水线并行。流水线并行将模型的层拆分到不同机器，例如 80 层 Transformer Block 可以拆分为：第 1-20 层放 0 号机，21-40 层放 1 号机，以此类推。数据像流水线一样流动，0 号机算完输出后通过网络传给 1 号机。

流水线并行的痛点是气泡问题，后端显卡在等待前端显卡计算时是空闲的。vLLM 较少在高性能场景下单独依赖流水线并行，通常采用张量并行加流水线并行的混合策略。

### 张量并行 vs 流水线并行
张量并行是算子内部矩阵的切分，通信频率极高（每一层都要同步），硬件要求极高带宽（如 NVLink），主要瓶颈是通信延迟。流水线并行是算子层与层之间的切分，通信频率较低（每组层算完才同步），可以使用 RoCE 或 InfiniBand，主要瓶颈是负载均衡与气泡。

单机多卡场景主要靠张量并行，利用 NVLink 把多张卡连成一张逻辑大显卡。多机多卡场景主要靠张量并行加流水线并行，利用网络让模型在不同机器间像流水线一样接力。

### 训练落地

训练的核心约束是显存——参数、梯度、优化器状态的总和远超模型权重本身。以 Adam 优化器为例：FP16 权重 $2 \times P$，FP16 梯度 $2 \times P$，FP32 Master Weights + Momentum + Variance 合计 $12 \times P$，总计约 $16 \times P$。一个 70B 模型训练时需要约 1.1TB 显存，远非单卡能承受。

DeepSpeed ZeRO 是训练并行的工业标准，通过三级分片逐步降低单卡显存占用。ZeRO-1 分片优化器状态，每卡只维护 $1/N$，几乎零额外通信，省约 80% 优化器显存。ZeRO-2 进一步分片梯度，反向传播后通过 ReduceScatter 归约并分发，每卡只保留自己负责的那部分梯度。ZeRO-3 将参数也分片，前向时通过 AllGather 按需拉取当前层参数，计算完立即释放——理论上可训练无限大的模型，只要集群总显存够。ZeRO-Offload 进一步将优化器状态卸载到 CPU 内存甚至 NVMe，训练速度下降 30-50% 但 GPU 需求可降一个数量级。

并行策略的决策链路：单卡装得下 → 纯数据并行。单卡装不下 → FSDP/ZeRO（通用性好，通信分散）。FSDP 不够 → 加张量并行（通信重，但限制在 NVLink 内）。还放不下 → 加流水线并行（通信轻但有气泡）。MoE 模型额外叠专家并行。

3D 并行是这一决策链路的最终形态：张量并行在节点内利用 NVLink 承载高频计算，流水线并行在节点间利用 InfiniBand 承载中频通信，数据并行在最外层以最低频率同步梯度。这是将模型推向万卡规模的关键架构。

在实践中，配置这些策略通常通过框架的配置文件完成：

DeepSpeed 使用 JSON 配置 ZeRO 阶段。ZeRO-2 适合中等规模训练，ZeRO-3 适合超大规模，offload 在显存极度紧张时启用：

```json
{
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu" },
    "overlap_comm": true,
    "reduce_scatter": true
  },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 1
}
```

vLLM 通过 `--tensor-parallel-size` 指定张量并行度，`--pipeline-parallel-size` 指定流水线并行度。两者的乘积不能超过总 GPU 数：

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3-70B \
  --tensor-parallel-size 4 \
  --max-model-len 32768
```

Megatron-LM 通过 `--tensor-model-parallel-size` 和 `--pipeline-model-parallel-size` 设置两种并行度，数据并行度由总 GPU 数除以两者乘积自动计算。`--num-layers-per-virtual-pipeline-stage` 控制交错式流水线调度（interleaved schedule）的粒度，进一步减少气泡：

```bash
python pretrain_gpt.py \
  --tensor-model-parallel-size 4 \
  --pipeline-model-parallel-size 8 \
  --num-layers-per-virtual-pipeline-stage 2
```


### 配置选择的原则

并行策略没有银弹，每次选择都是一次显存-通信-算力利用率的三角权衡。

显存优先原则：先确保模型能跑起来。如果单卡能装下，不做模型并行。如果装不下，优先用 FSDP/ZeRO（通用性好），不行再加张量并行（通信重但高效），最后加流水线并行（通信轻但有气泡）。

通信层级原则：通信越频繁的策略放在带宽越高的连接上。节点内 NVLink（900GB/s）承载张量并行，节点间 InfiniBand（50GB/s）承载流水线并行和数据并行。跨机架以太网（12GB/s）仅适合低频的梯度同步或参数服务器通信。

算力利用率原则：每添加一层并行策略都会引入通信开销和调度气泡。并行度不是越高越好——2 卡 TP 通信开销占 5-10%，8 卡 TP 可能占 20-30%。找到一个并行度，使得通信开销的增长不超过算力增加带来的收益。

实际调试时最可靠的方法是测量而非估算。用 `torch.distributed.collective_profiler` 或 `NCCL_DEBUG=INFO` 打印每次 AllReduce/AllGather 的耗时，确认通信时间占比不超过 20-30%。如果超过，减少该层的并行度或升级硬件。

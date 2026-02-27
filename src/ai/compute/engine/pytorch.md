---
title: PyTorch
order: 0
---

# PyTorch

从 AI infra 工程师的视角来看，PyTorch 不仅是一个深度学习框架，更是一个复杂的分布式计算系统。它的实现涉及硬件后端抽象、动态计算图的执行、自动微分系统、内存管理、分布式训练等多个子系统。理解这些实现原理，对于开发 AI 引擎、优化模型性能、调试底层问题至关重要。

## 硬件后端抽象

PyTorch 的一个精妙设计是 Dispatcher 机制，它实现了多后端的统一抽象。上层 API（如 `torch.add`）不直接调用 CUDA kernel，而是通过 Dispatcher 分发到具体后端。Dispatcher 根据张量的 device 类型查找对应的 kernel 实现，然后将调用转发过去。

这种设计使得 PyTorch 可以无缝支持多种硬件平台，开发者无需修改上层代码，只需安装对应的后端包即可。

| 平台              | 后端                              | 底层调用                |
| ----------------- | --------------------------------- | ----------------------- |
| NVIDIA GPU        | CUDA                              | cuBLAS、cuDNN、TensorRT |
| AMD GPU           | ROCm                              | hipBLAS、MIOpen         |
| Apple M 芯片      | MPS                               | Metal Compute           |
| Intel GPU / CPU   | XPU (oneAPI)                      | oneDNN                  |
| Huawei Ascend NPU | Ascend C                          | CANN                    |
| Google TPU        | XLA                               | HLO / MLIR              |
| CPU               | Native                            | OpenMP / MKL / BLAS     |

### Dispatcher 实现原理

Dispatcher 的核心是一个基于类型注册的分发表。每个张量操作（如 `torch.add`）在初始化时会注册不同设备类型的 kernel 实现。运行时，Dispatcher 根据输入张量的 device 属性查找对应的 kernel，然后通过函数指针直接调用。这种间接调用层的设计使得添加新后端变得简单——只需为新设备注册 kernel 实现即可，无需修改上层 API。

对于 AI infra 开发者来说，这意味着可以为定制硬件（如国产 AI 芯片）开发 PyTorch 后端。实现一个后端需要定义 DeviceGuard、Allocator、Stream 等接口，然后为每个算子编写 kernel 实现。PyTorch 的 native_functions.yaml 文件定义了所有算子的接口规范，是实现后端的重要参考。

## 自动微分系统

PyTorch 的自动微分系统是其最精巧的设计之一。Autograd 的核心思想是：在前向传播时记录操作的依赖关系，在反向传播时自动应用链式法则计算梯度。

### 计算图构建

从实现角度看，每个张量都有一个 `grad_fn` 属性指向创建它的函数。例如 `y = x + 2` 会创建一个 `AddBackward` 节点，记录输入 `x` 和常量 `2`。当调用 `y.backward()` 时，PyTorch 会从 `y` 开始反向遍历计算图，对每个 `grad_fn` 调用其 `backward()` 方法，计算梯度并累积到输入张量的 `.grad` 属性。

这种设计有几个关键点。梯度累积意味着多次调用 `backward()` 会让梯度累加，需要手动 `zero_grad()` 清零，这是实现梯度累积训练的基础。计算图保留机制默认在前向传播后释放图，需要设置 `create_graph=True` 才能进行高阶微分。原位操作（如 `x += 1`）会修改原值导致梯度计算错误，Autograd 通过版本检测机制标记这类操作的梯度为无效。

### Autograd 引擎

Autograd 引擎是一个基于拓扑排序的执行引擎。反向传播时，引擎首先对计算图进行拓扑排序，确保节点按照依赖关系正确执行。然后为每个节点分配计算资源，通过线程池并行执行独立的梯度计算。引擎还支持梯度检查点（gradient checkpointing），通过牺牲计算换显存，只保存部分中间结果，反向传播时重新计算被丢弃的激活值。

## 内存管理

PyTorch 的内存管理是性能优化的关键。理解内存分配器的行为，对于解决 OOM 问题、优化显存利用率至关重要。

### 缓存分配器

PyTorch 使用缓存分配器（Caching Allocator）来管理 GPU 显存，避免频繁的 malloc/free 开销。当请求分配显存时，分配器优先从缓存中查找合适大小的空闲块；如果没有，则从 CUDA 申请新块。释放时不立即归还给 CUDA，而是放入缓存供后续复用。

缓存分配器的配置对性能影响很大。`max_split_size_mb` 限制单个缓存块的最大大小，避免内存碎片；`garbage collection threshold` 控制何时触发缓存整理；`expandable_segments` 允许缓存段动态扩展。显存不足时（OOM），可以尝试增加 `max_split_size_mb` 或启用 `expandable_segments`。

### 内存碎片

内存碎片是导致 OOM 的常见原因。外部碎片由大量小块显存分配导致，缓存分配器通过合并相邻空闲块来缓解。内部碎片由请求大小与实际分配大小不匹配导致，可以通过调整缓存块大小策略来优化。

PyTorch 提供了 `torch.cuda.memory_summary()` API 来分析显存使用情况，包括缓存大小、碎片率、分配统计等。对于调试显存问题，这个 API 比 nvidia-smi 更精准。

## 分布式训练

PyTorch 提供了多种分布式训练策略，从简单到复杂依次是 DataParallel、DistributedDataParallel (DDP)、FullyShardedDataParallel (FSDP)。

### DDP 实现

DistributedDataParallel 是生产级的分布式方案，每张卡运行独立的进程，通过 AllReduce 同步梯度。DDP 使用高效的通信后端（NCCL、Gloo），支持多机多卡。

DDP 的实现涉及两个关键机制。bucket 机制将多个参数的梯度打包成一个 bucket，减少通信次数。梯度累积在反向传播时异步通信，通过通信与计算的重叠（overlap）隐藏通信延迟。具体来说，DDP 在反向传播计算层 $i$ 的梯度时，同时同步层 $i-1$ 的梯度，将通信开销从串行的 30% 降至并行的 10% 以下。

### FSDP 实现

FullyShardedDataParallel 是 PyTorch 2.0 引入的大模型训练方案，借鉴了 DeepSpeed ZeRO-3 的设计。它将参数、梯度、优化器状态全部分片到多张卡，前向传播时通过 AllGather 重建参数，反向传播后立即释放。

FSDP 的设计更贴近 PyTorch 的模块化哲学。通过 `torch.distributed.fsdp.FullyShardedDataParallel` 包装模块，自动处理前向传播时的 AllGather 和反向传播后的 ReduceScatter。迁移现有代码非常简单——只需将 `nn.DataParallel` 替换为 `FSDP`，无需重构模型定义。

FSDP 与 `torch.compile` 深度集成，通过算子融合和通信计算重叠，在保持易用性的同时实现了与 DeepSpeed 相当的性能。FSDP 原生支持 BF16 混合精度训练，BF16 与 FP16 的指数位相同（8 位），但尾数位减少到 7 位，数值范围更大，不需要 loss scaling，对于训练稳定性敏感的大模型是更安全的选择。

## 编译优化

PyTorch 2.0 引入的 `torch.compile` 是近年来最重要的更新之一。它通过编译优化将动态图的性能提升到接近静态图的水平。

### 编译流水线

`torch.compile` 的核心组件包括 TorchDynamo、AOTAutograd、PrimTorch、Inductor。TorchDynamo 负责捕获 Python bytecode，将其转换为 FX Graph。它使用字节码分析和 guard 机制，在输入变化时自动重新编译。AOTAutograd 提前执行 autograd，生成前向和反向的计算图，允许跨前向反向的优化。PrimTorch 定义了算子的原语集合，将不同后端的算子统一到一组原语。Inductor 是编译后端，将 FX Graph 编译为高效的 Triton kernel 或 C++ 代码。

### 优化技术

`torch.compile` 的收益来自多方面。算子融合减少 kernel 启动开销和显存访问，例如 LayerNorm 后接 Residual 可融合为一个 kernel，只需读写一次显存。内存规划减少中间结果的存储，通过活跃度分析确定变量的生命周期，复用存储空间。常量折叠在编译时计算常量表达式。死代码消除删除无用计算。对于 Transformer 等模型，`torch.compile` 可带来 30% 以上的性能提升。

## 模型并行

当模型太大无法放入单卡显存时，需要使用模型并行。PyTorch 提供了多种模型并行策略。

### 序列并行

对于 Transformer 模型，序列并行是高效的方案。序列并行将序列维度切分到多张卡，每张卡只计算部分序列的 Attention，然后通过 Ring Attention 在环状拓扑上通信。这避免了完整序列的 KV Cache 存储和计算，将显存占用从 $O(n^2)$ 降至 $O(n^2/p)$（$p$ 为卡数）。

### 张量并行

张量并行是另一种模型并行策略，将单个算子的张量切分到多张卡。例如矩阵乘法 $Y = XW$，可将权重 $W$ 按列切分，每张卡计算 $Y_i = XW_i$，最后拼接结果。张量并行的通信频率高（每个算子都需要通信），因此需要高带宽互联（如 NVLink）才能发挥性能。

PyTorch 的 `torch.distributed.tensor` 模块提供了原生的张量并行支持，通过 `DTensor` 抽象简化了张量并行的实现。DTensor 自动处理张量的分片、通信、聚合，使得编写张量并行代码像编写单卡代码一样简单。

## 通信后端

PyTorch 的分布式训练依赖高效的通信后端。NCCL (NVIDIA Collective Communications Library) 是 NVIDIA GPU 上的默认后端，针对 NVLink 和 InfiniBand 优化，性能最佳。Gloo 是通用的后端，支持 TCP 和 RDMA，适合非 NVIDIA 场景。MPI 后端适合已有 MPI 集群的 HPC 环境。

### 通信原语

通信原语包括点对点通信和集合通信。AllReduce 将所有卡的梯度聚合并分发，是数据并行的核心。AllGather 将分片的参数聚合到每张卡，是 FSDP 前向传播的关键。ReduceScatter 将梯度聚合并分片到不同卡，是 FSDP 反向传播的关键。

理解这些通信原语的特性，对于优化分布式训练性能至关重要。例如，AllReduce 的带宽需求高但延迟敏感度低，适合通过增加 batch size 来摊薄通信开销；AllGather 的数据量随模型大小线性增长，在超大模型场景下会成为瓶颈。

### ProcessGroup

ProcessGroup 是 PyTorch 分布式通信的抽象层，屏蔽了不同通信后端的差异。实现自定义通信后端需要实现 ProcessGroup 接口，包括 `broadcast`、`allreduce`、`allgather` 等方法。这对于适配国产通信库（如华为的 HCCS）或优化特定拓扑（如树形网络）非常有用。

## 模型格式与序列化

PyTorch 的原生模型格式是基于 pickle 的序列化格式，文件扩展名通常为 `.pt`、`.pth` 或 `.pkl`。这种格式将 Python 对象（包括张量、模块、字典等）序列化为二进制文件，加载时通过反序列化重建对象。

### state_dict 机制

PyTorch 推荐通过 `state_dict()` 保存模型参数，而非直接保存整个模型对象。`state_dict()` 返回一个有序字典，包含所有可学习参数（权重和偏置）以及持久化缓冲区（如 BatchNorm 的 running mean）。这种设计的优势是：只保存参数不保存代码，避免了版本兼容性问题；加载时可以灵活处理参数映射（如加载预训练权重到自定义模型）。

### 自定义算子扩展

PyTorch 提供了多层自定义算子扩展机制。最简单的方式是通过 `torch.autograd.Function` 定义自定义的前向和反向传播，适合快速原型开发。生产级性能需要使用 C++/CUDA 扩展，通过 `torch.utils.cpp_extension` 加载编译好的共享库。

PyTorch 2.0 引入了 `torch.library` API，简化了自定义算子的注册流程。开发者只需定义算子的内核实现和元数据（如类型推断、别名分析），PyTorch 会自动生成对应的 Python 绑定、Autograd 函数、Dispatch Key 分发。这使得为国产 AI 芯片开发算子后端变得更加规范。

对于需要跨平台部署的场景，TorchScript 是另一种选择。它将 Python 代码编译为中间表示，可以在无 Python 环境的 C++ 程序中加载执行。但 TorchScript 的类型系统较为严格，动态特性支持有限，代码迁移成本较高。

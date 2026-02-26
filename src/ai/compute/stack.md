---
order: 0
title: 计算栈
---

# 科学计算栈
AI 算力软件层次结构，从上层应用到硬件的计算资源调用链路如下。注意 GPU 计算为异构计算，包含运行在 CPU 上的管控面代码和运行在 GPU 上的数据面代码。

| 层次              | CPU 管控面                                    | GPU 数据面                            |
| ----------------- | --------------------------------------------- | ------------------------------------- |
| 大模型应用层      | 提示词与记忆管理                              | 不感知                                |
| 模型算法层        | 实现具体的 AI 模型算法                        | 不感知                                |
| AI 引擎层         | 按照模型结构定义运行 AI 模型                  | 按照模型算法要求实现张量计算的算子    |
| 科学计算标准接口  | 规定跨硬件厂商的加速计算调用规范              | 提供 GPU 编程语言，供上层进行算子编写 |
| HAL 层 + 系统调用 | 各硬件厂商提供的用户态 API 接口，封装硬件调用 | 透传 GPU 目标程序                     |
| 内核驱动层        | 实现硬件驱动管理                              | 透传 GPU 目标程序                     |
| 硬件层            | 实现具体的 AI 模型算法                        | 执行 GPU 目标程序                     |

+ 模型算法层：实现具体的 AI 模型算法，如 Transformer、CNN 等；
+ 模型部署层：；
+ 深度学习框架层（PyTorch）：屏蔽底层硬件差异，定义统一的张量算子接口，由各厂商的后端实现具体调用逻辑；静态编译能力集成，增加编译期技术，进一步优化算法层代码；
+ 算子层：实现 PyTorch 的张量操作，调用厂商提供的加速计算 API。算子通常基于特定 DSL 编写，如 NVIDIA 的 CUDA C++；
+ 加速计算 API 接口层：用户态的硬件加速 DSL，如 CUDA C++，负责将高层语法编译为中间表示（PTX），供驱动程序处理。还有比较新的 OpenAI 的 Triton;
+ HAL 层/系统调用层：封装 `/dev/nvidia0` 等设备节点，提供 `libcuda.so`（计算）、`libnvidia-glcore.so`（图形）等闭源动态库，与内核驱动通信；
+ 内核驱动层：将 PTX 中间码编译为 GPU 机器码（SASS），管理硬件资源。NVIDIA 驱动曾是数百 MB 的巨型二进制，后通过 GSP 架构重构；
+ 硬件层：NVIDIA GPU、AMD GPU、Google TPU、华为昇腾 NPU 等物理设备；

![](./compute.dio.svg)

AI 算力技术分层概述
1. 大模型应用层：大模型部署、调度与记忆管理
这一层处理大型AI模型（如GPT系列或Llama）的实际运行环境。

部署：将模型加载到服务器或集群中，使用工具如Kubernetes或Ray进行分布式部署。示例：使用Hugging Face Transformers库部署Transformer模型，支持量化（如INT8）以减少内存占用。
调度：管理任务分配，如在多GPU环境中使用负载均衡器（如NVIDIA MIG）分割GPU资源。优先级队列确保高优先级推理任务先执行。
记忆管理：处理KV缓存（Key-Value cache）以加速连续对话，避免重复计算。示例：vLLM框架优化内存，通过分页和预取减少OOM（Out of Memory）错误。
这一层强调效率，目标是低延迟和高吞吐量。

2. 模型算法层：实现具体的AI模型算法，如Transformer、CNN等
这一层定义模型的核心逻辑，使用数学和算法构建网络结构。

Transformer：基于自注意力机制，用于NLP和多模态任务。示例：实现Vaswani et al.的Attention Is All You Need论文中的架构，包括多头注意力、位置编码和前馈网络。
CNN：卷积神经网络，用于图像处理。示例：ResNet架构，通过残差连接解决梯度消失问题。
算法在这里用Python或框架API编码，但不直接涉及硬件优化。重点是模型设计和训练逻辑，如梯度下降或Adam优化器。

3. 深度学习框架层（PyTorch）：屏蔽底层硬件差异，定义统一的张量算子接口
框架如PyTorch或TensorFlow提供抽象接口，让开发者无需关心硬件细节。

统一接口：定义张量操作（如add、matmul），厂商后端（如CUDA或ROCm）实现具体调用。
算子编写：自定义算子封装PyTorch的torch.Tensor操作，调用下层API。示例：用torch.autograd.Function编写前向/反向传播，内部调用CUDA kernel。
DSL示例：NVIDIA的CUDA C++用于编写内核，如__global__函数定义GPU并行计算。框架编译这些为可执行代码。
这一层桥接算法和硬件，允许跨设备（如GPU/TPU）移植代码。

4. 加速计算API接口层：用户态的硬件加速DSL，如CUDA C++，负责将高层语法编译为中间表示（PTX）
这一层提供编程接口，直接操作加速硬件。

CUDA C++：NVIDIA的DSL，编写内核代码编译为PTX（Parallel Thread Execution），支持线程块和共享内存管理。示例：矩阵乘法内核使用cudaMalloc分配内存，cudaMemcpy传输数据。
Triton：OpenAI的开源DSL，简化内核编写，支持自动调优。示例：用Triton编写自定义融合操作（如LayerNorm + Add），比纯CUDA更高效。
接口层编译高层代码为中间码，交给驱动处理。重点是性能优化，如减少内存访问延迟。

5. HAL层：封装/dev/nvidia0等设备节点，提供libcuda.so（计算）、libnvidia-glcore.so（图形）等闭源动态库，与内核驱动通信
硬件抽象层（Hardware Abstraction Layer）提供用户空间库，隐藏内核细节。

封装设备：通过/dev/nvidia*节点访问GPU，提供libcuda.so用于计算任务（如cuBLAS库的线性代数）。
图形库：libnvidia-glcore.so处理OpenGL/Vulkan渲染，但AI更侧重计算库。
这一层是闭源的，确保兼容性。示例：PyTorch通过torch.cuda调用libcuda.so启动内核。

6. 内核驱动层：将PTX中间码编译为GPU机器码（SASS），管理硬件资源
操作系统内核模块处理最终编译和资源分配。

编译PTX：驱动将PTX转为SASS（Streamline Assembly），特定于GPU架构（如Ampere或Hopper）。
资源管理：调度SM（Streaming Multiprocessor）、内存分配和上下文切换。NVIDIA驱动曾是大型blob（数百MB），2020年代通过GSP（GPU System Processor）重构为模块化，提高安全性。
示例：驱动处理多进程共享GPU，避免冲突。AMD的ROCm驱动类似，但开源程度更高。

7. 硬件层：NVIDIA GPU、AMD GPU、Google TPU、华为昇腾NPU等物理设备
底层物理芯片执行计算。

NVIDIA GPU：如A100/H100，支持高并行浮点运算（TFLOPS级）。
AMD GPU：如MI300系列，兼容ROCm框架。
Google TPU：ASIC设计，优化矩阵乘法，用于Cloud TPU Pods。
华为昇腾NPU：如Ascend 910B，集成CANN框架，支持AI训练/推理。
硬件决定峰值性能，如H100的80TFLOPS FP16。趋势：向定制ASIC（如TPU）发展，以降低功耗。

这个分层栈确保从抽象算法到高效执行的端到端优化。如果您想深入某个层（如代码示例或最新发展），请指定。

## 硬件加速
使用 GPU 的加速可并行执行的计算任务，目前主要包括俩个领域：图形渲染和科学计算。人工智能领域主要使用科学计算 API 进行加速。

然而，硬件加速的现状并不乐观，各个硬件厂商纷纷使用自家独立的 GPU API，并且同是自家的 API 同样也被迭代和变更，导致不同的硬件设备的差异直接就被暴露到了应用层。应用层的软件编写者需要直面硬件差异。

图形加速计算使用的计算栈大体类似，不过略有不同，具体参考图形渲染章节[硬件加速](/client/render/stack)。

> OpenCL
> 
> 曾经的 GPU 跨平台统一 API，但是随着各家的硬件生态不断割裂，分歧再次扩大，OpenCL 已逐渐退出历史舞台，但仍然被 AMD 和 Intel 所支持，不过性能往往不如各家的专用 API。

## 加速计算 API

| 厂商             | 图形 API                  | 通用计算 API                     |
| ---------------- | ------------------------- | -------------------------------- |
| Apple（苹果）    | Metal Graphics            | Metal Compute（+ Core ML / ANE） |
| NVIDIA（英伟达） | OpenGL / Vulkan / DirectX | CUDA                             |
| AMD（超微）      | OpenGL / Vulkan / DirectX | ROCm（Radeon Open Compute）      |
| Intel            | OpenGL / Vulkan / DirectX | oneAPI（DPC++ / SYCL）           |

其中，Vulkan 作为现代的 OpenGL 的继任者，且支持跨厂商硬件的图形 API 标准，是学习现代渲染的必备技术。在 AI 计算领域，目前由 OpenAI 主导的 Triton 有望成为加速计算领域的 Vulkan，引领跨厂商间的 API 互通，成为 OpenCL 的现代继任者。

## Torch
Torch 框架为了使用硬件加速计算，规定各个 GPU 厂商的封装层，将各家的硬件 API 进行屏蔽，从而让上层的数据科学家无需触及糟心而混乱的 GPU 生态，专注于数据训练即可，在调用 torch 的 API 时，torch 将帮助识别当前的硬件环境，使用对应的硬件进行加速，常见的硬件平台包括：

| 平台              | 后端                              | 底层调用                |
| ----------------- | --------------------------------- | ----------------------- |
| NVIDIA GPU        | CUDA                              | cuBLAS、cuDNN、TensorRT |
| AMD GPU           | ROCm                              | hipBLAS、MIOpen         |
| Apple M 芯片      | MPS （Metal Performance Shaders） | Metal Compute           |
| Intel GPU / CPU   | XPU  （oneAPI）                   | oneDNN                  |
| Huawei Ascend NPU | Ascend C                          | CANN                    |
| Google TPU        | XLA                               | HLO / MLIR              |
| CPU               | Native                            | OpenMP / MKL / BLAS     |

包括国产的华为昇腾 NPU 生态。

## GSP 架构
GSP（GPU System Processor）是 NVIDIA 从 Turing 架构开始在 GPU 芯片上集成的专用 RISC-V 处理器，将原本运行在内核驱动中的复杂逻辑下沉到 GPU 固件中执行，然后将计算语法编译的任务上移到用户态去做。

### 背景
早期 NVIDIA 驱动的 `.ko` 内核模块体积高达数百 MB，原因在于它承担了过多职责：接收 GLSL、HLSL、PTX 等多种上层代码，在内核态完成编译并生成机器码。这种设计带来两个严重问题：
+ 驱动膨胀：内核模块承载编译器、调度器等复杂逻辑，代码体积难以控制。
+ 维护困境：闭源驱动与 Linux 内核演进冲突频发，社区无法介入修复。

GSP 的引入改变了这一局面。内核驱动现在只负责透传命令到 GSP 固件，由固件完成 GPU 初始化、任务调度、PTX 编译等核心工作。这带来以下收益：
+ 驱动瘦身：内核模块仅保留必要的信令逻辑，体积大幅缩减。
+ 固件可控：NVIDIA 可通过固件更新修复问题，无需重发驱动。
+ 开源契机：内核态逻辑简化后，NVIDIA 得以发布开源内核模块（Open Kernel Modules），改善了与 Linux 社区的关系。

### 现状
GSP 固件由内核驱动在初始化阶段加载到 GPU 的专用显存区域执行。它接管了显示引擎、电源管理、上下文切换等核心职能，内核驱动通过 RPC 机制与 GSP 通信，提交计算任务和查询状态。

在 GSP 架构下，编译职责被重新分配：
+ **GSP 固件**：负责 GPU 硬件初始化、任务调度、电源管理、显示输出控制等底层职能。
+ **内核驱动**：仅保留设备枚举、内存映射、中断处理等薄层逻辑，透传命令到 GSP。
+ **用户态运行时**：负责将上层计算语法（CUDA、HLSL、GLSL）编译为中间表示（PTX 或 SPIR-V），再通过 HAL 层接口提交给驱动。

这种分层使得 NVIDIA 能够开放内核驱动源码（Open Kernel Modules），同时将编译器等复杂逻辑保留在用户态闭源库中。对于上层开发者，NVIDIA 推荐直接通过 HAL 层提交 PTX（计算）或 SPIR-V（图形），而算子语言（如 Triton）和着色器语言则可作为前端 DSL 自由定制，只要最终编译到这些中间表示即可。

### PTX 和 SPIR-V
PTX（Parallel Thread Execution）是 NVIDIA 的虚拟指令集架构，类似于 Java Bytecode 或 WebAssembly。它以可读的文本/汇编形式存在，核心价值是**跨代兼容**——从 Maxwell 到 Blackwell，GPU 硬件架构差异巨大，PTX 提供了一套稳定的带寄存器抽象的指令，让开发者无需为每代显卡重写代码。GSP 内置的 `ptxas` 编译器会在运行时将 PTX 即时编译为 GPU 真正执行的机器码 SASS（Streaming Assembly）。

SPIR-V（Standard Portable Intermediate Representation）则是 Khronos 制定的跨厂商中间表示，服务于 Vulkan 和 OpenCL 生态。与 PTX 的 NVIDIA 专有定位不同，SPIR-V 的目标是**跨硬件兼容**——同一份 SPIR-V 二进制可在 AMD、Intel、NVIDIA 等不同 GPU 上执行。SPIR-V 采用二进制格式而非文本格式，更紧凑但不可直接阅读。

| 特性       | PTX            | SPIR-V                   |
| ---------- | -------------- | ------------------------ |
| 制定方     | NVIDIA         | Khronos                  |
| 格式       | 文本/汇编      | 二进制                   |
| 兼容范围   | 仅 NVIDIA GPU  | 跨厂商（AMD/Intel 等）   |
| 主要用途   | CUDA 计算      | Vulkan 图形、OpenCL 计算 |
| 运行时编译 | 驱动内置 ptxas | 驱动内置 SPIR-V 编译器   |

在新架构下，NVIDIA 的 HAL 层同时支持接收 PTX 和 SPIR-V，开发者可根据目标平台选择合适的中间表示。

## Triton
当前的加速计算接口往往局限于特定的 GPU 硬件平台，缺乏跨厂商的 API 语言，为此，OpenAI 推出了
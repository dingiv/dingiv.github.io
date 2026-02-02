---
title: 算力
order: 60
---

# AI 算力
目前的 AI 训练需要大量的计算资源，是阻碍 AI 发展的重大绊脚石。

## 结构层次
AI 算力的层次结构如下：
+ 大模型应用层：大模型部署，调度，记忆
+ 模型算法层：实现 AI 模型算法
+ Pytorch 框架层：屏蔽下层的不同的硬件生态，通过规定张量算子接口，要求下层的后端胶水层来实现这些算子接口，从而让 Pytorch 进行调用；
+ 算子层：具体负责将一个 Pytorch 张量批量操作进行封装，调用自家生态的硬件加速计算接口进行提交；
+ 加速计算接口层：用户态硬件加速计算接口，提供科学计算语法，以一个 DSL 语言的形式存在，例如 CUDA 是一个类似于 C++ 的扩展语法，它负责将 CUDA 语言代码转换为驱动程序能够看懂的中间表示语言 PTX；
+ 系统调用层：由内核实现；
+ 加速卡驱动层：由各硬件厂商按照操作系统的驱动接口进行实现；针对于 nVidia 的闭源显卡驱动，该驱动负责将 PTX 中间标识码表示成 GPU 能够听懂的机器码；
+ 硬件层：Nvidia GPU，AMD GPU，Google TPU...

![](./compute.dio.svg)

> 虚拟指令集 PTX (Parallel Thread Execution)，PTX 是 NVIDIA 的虚拟 ISA（指令集架构）。它类似于 Java 的 Bytecode 或 WebAssembly。PTX 还是可读的文本/汇编形式。
> 
> 它是为了“兼容性”而存在的。不同代的 N 卡（从 Maxwell 到 Pascal，再到最新的 Blackwell）底层硬件架构差异巨大。PTX 提供了一套稳定的、带寄存器抽象的指令，让开发者（或编译器）不需要为每一代新显卡重写代码。
>
> 原生机器码 SASS (Streaming Assembly)，SASS 是 GPU 硬件真正执行的机器指令。由 N 卡驱动程序内置的编译器（ptxas）在后台执行 JIT（即时编译）。

## 硬件加速
使用 GPU 的加速可并行执行的计算任务，目前主要包括俩个领域：图形渲染和科学计算。人工智能领域主要使用科学计算 API 进行加速。

然而，硬件加速的现状并不乐观，各个硬件厂商纷纷使用自家独立的 GPU API，并且同是自家的 API 同样也被迭代和变更，导致不同的硬件设备的差异直接就被暴露到了应用层。应用层的软件编写者需要直面硬件差异。

图形加速计算使用的计算栈大体类似，不过略有不同，具体参考图形渲染章节[硬件加速](/client/render/gpu)。

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
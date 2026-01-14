---
title: 加速技术
order: 60
---

# 科学计算加速技术
目前的 AI 训练需要大量的计算资源，是阻碍 AI 发展的重大绊脚石。通过计算加速技术提高大模型训练和部署的资源需求，从而大力推动 AI 进化和商业化落地。

科学计算加速技术的层次结构如下：
+ 模型算法层：实现 AI 模型算法
+ Pytorch 框架层：屏蔽下层的不同的硬件生态，通过规定张量算子接口，要求下层的后端胶水层来实现这些算子接口，从而让 Pytorch 进行调用；
+ 算子层：具体负责将一个 Pytorch 张量批量操作进行封装，调用自家生态的硬件加速计算接口进行提交；
+ 加速计算接口层：用户态硬件加速计算接口，提供科学计算语法，例如 CUDA 是一个类似于 C++ 的扩展语法；
+ 系统调用层：由内核实现；
+ 加速卡驱动层：由各硬件厂商按照操作系统的驱动接口进行实现；
+ 硬件层

## 硬件加速
使用 GPU 的加速可并行执行的计算任务，目前主要包括俩个领域：图形渲染和科学计算。人工智能领域主要使用科学计算 API 进行加速。

然而，硬件加速的现状并不乐观，各个硬件厂商纷纷使用自家独立的 GPU API，并且同是自家的 API 同样也被迭代和变更，导致不同的硬件设备的差异直接就被暴露到了应用层。应用层的软件编写者需要直面硬件差异。

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
Torch 框架为了使用硬件加速计算，主动实现各个 GPU 厂商的封装层，将各家的硬件 API 进行屏蔽，从而让上层的数据科学家无需触及糟心而混乱的 GPU 生态，专注于数据训练即可，在调用 torch 的 API 时，torch 将自动识别当前的硬件环境，使用对应的硬件进行加速，当前支持的硬件平台包括：

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
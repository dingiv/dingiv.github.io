---
title: 硬件加速
order: 60
---

# 硬件加速
使用 GPU 的加速可并行执行的计算任务，目前主要包括俩个领域：图形渲染和科学计算。人工智能领域主要使用科学计算 API 进行加速。

然而，硬件加速的现状并不乐观，各个硬件厂商纷纷使用自家独立的 GPU API，并且同是自家的 API 同样也被迭代和变更，导致不同的硬件设备的差异直接就被暴露到了应用层。应用层的软件编写者需要直面硬件差异。

> OpenCL
> 
> 曾经的 GPU 跨平台统一 API，但是随着各家的硬件生态不断割裂，分歧再次扩大，OpenCL 已逐渐退出历史舞台，但仍然被 AMD 和 Intel 所支持，不过性能往往不如各家的专用 API。

## 硬件 API

| 厂商             | 图形 API                  | 通用计算 API                     |
| ---------------- | ------------------------- | -------------------------------- |
| Apple（苹果）    | Metal Graphics            | Metal Compute（+ Core ML / ANE） |
| NVIDIA（英伟达） | OpenGL / Vulkan / DirectX | CUDA（核心）                     |
| AMD（超微）      | OpenGL / Vulkan / DirectX | ROCm（Radeon Open Compute）      |
| Intel            | OpenGL / Vulkan / DirectX | oneAPI（DPC++ / SYCL）           |

## Torch
Torch 框架为了使用硬件加速计算，主动实现各个 GPU 厂商的封装层，将各家的硬件 API 进行屏蔽，从而让上层的数据科学家无需触及糟心而混乱的 GPU 生态，专注于数据训练即可，在调用 torch 的 API 时，torch 将自动识别当前的硬件环境，使用对应的硬件进行加速，当前支持的硬件平台包括：

| 平台            | 后端                              | 底层调用                |
| --------------- | --------------------------------- | ----------------------- |
| NVIDIA GPU      | CUDA                              | cuBLAS、cuDNN、TensorRT |
| AMD GPU         | ROCm                              | hipBLAS、MIOpen         |
| Apple M 芯片    | MPS （Metal Performance Shaders） | Metal Compute           |
| Intel GPU / CPU | XPU  （oneAPI）                   | oneDNN                  |
| CPU             | Native                            | OpenMP / MKL / BLAS     |

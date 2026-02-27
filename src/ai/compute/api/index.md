---
title: 标准接口
order: 16
---
# 科学计算 API
科学计算标准接口是连接上层算法与底层硬件的桥梁。不同的硬件平台提供了各自的编程接口，从 CUDA 的显式并行到 Metal 的声明式风格，开发者需要根据目标平台选择合适的 API。这些接口的核心价值在于将硬件的并行计算能力暴露给上层应用，同时尽可能屏蔽底层硬件细节，简化用户使用加速硬件的过程。

## 各平台接口
| 接口     | 硬件平台          | 底层调用                | 详细介绍           |
| -------- | ----------------- | ----------------------- | ------------------ |
| CUDA     | NVIDIA GPU        | cuBLAS、cuDNN、TensorRT | [CUDA](./cuda)     |
| Triton   | 跨平台            | CUDA、ROCm...           | [Triton](./triton) |
| OpenCL   | 跨平台            | *                       | [OpenCL](./opencl) |
| SIMD     | CPU               | OpenMP、MKL、BLAS       | [SIMD](./cpu)      |
| ROCm     | AMD GPU           | hipBLAS、MIOpen         | -                  |
| Metal    | Apple M 芯片      | MPS                     | -                  |
| oneAPI   | Intel GPU/CPU     | XPU、oneDNN             | -                  |
| Ascend C | Huawei Ascend NPU | CANN                    | -                  |
| XLA      | Google TPU        | HLO、MLIR               | -                  |

硬件加速的现状并不乐观，各个硬件厂商纷纷使用自家独立的 GPU API，导致设备差异直接暴露到应用层。OpenCL 曾是跨平台统一 API 的希望，但随着各厂商生态割裂，已逐渐退出历史舞台。在 AI 计算领域，目前由 OpenAI 主导的 Triton 有望接下 OpenCL 的重任，成为加速计算领域的 Vulkan，引领跨厂商间的 API 互通。

## ROCm
ROCm（Radeon Open Compute）是 AMD 的 GPU 计算平台，类似于 NVIDIA 的 CUDA 体系，但采用开源策略。ROCm 的编程语言 HIP 与 CUDA 高度兼容，大部分 CUDA 代码可通过 hipify 工具自动转换。ROCm 的库包括 rocBLAS（类似 cuBLAS）、rocFFT（类似 cuFFT）、MIOpen（类似 cuDNN），提供了完整的深度学习加速栈。

ROCm 是 AMD GPU 计算的主力生态，尤其在数据中心、AI 训练、科研计算等领域。与 NVIDIA 的闭源策略不同，ROCm 的开源使得开发者可深入理解底层实现，也便于学术研究定制优化。然而，ROCm 的生态系统仍不如 CUDA 成熟，工具链和第三方库的支持有限，对于需要稳定支持的生产环境，NVIDIA GPU 仍是更安全的选择。

近年来，ROCm 在性能和兼容性上取得了显著进展。AMD Instinct MI300X（CDNA 3 架构）的 FP16 矩阵乘法性能可达 2 PFLOPS，约为 NVIDIA H100（FP16 BF16）的 80%。PyTorch 的 ROCm 后端已进入主分支，大部分 CUDA 算子可通过 HIP 运行在 AMD GPU 上，无需修改代码。对于预算受限的学术机构，AMD GPU + ROCm 是 NVIDIA GPU 的可行替代方案。

## Metal
Metal 是 Apple 的图形和计算 API，为 macOS、iOS、tvOS 提供了硬件加速的统一接口。Metal 不同于 OpenGL 的跨平台定位，专注于 Apple 硬件的深度优化，通过更贴近硬件的设计降低了 CPU 开销，提升了 GPU 利用率。

Metal 的核心优势是与 Apple 硬件的深度集成。在 Apple M 系列芯片上，Metal 可利用统一内存架构，CPU 和 GPU 共享内存，无需数据传输。Metal Performance Shaders（MPS）提供了高性能的卷积、矩阵乘法、归约等算子库，性能接近手写 Metal 代码。PyTorch 的 MPS 后端通过 Metal Performance Shaders 调用 GPU，在 Apple M 系列芯片上可获得显著加速。

ResNet-50 推理在 M1 Max 上的性能约为 800 FPS（图像尺寸 224×224），约为 CPU（Intel MKL）的 10 倍。对于小模型推理（如 BERT-base），MPS 的性能约为 A100 GPU 的 5-10%，但功耗仅为其十分之一，适合本地部署场景。

## XLA（Google TPU）
XLA（Accelerated Linear Algebra）是 Google 开发的线性代数编译器，专为 TPU（Tensor Processing Unit）优化。TPU 是 Google 的 ASIC 芯片，针对矩阵乘法进行了深度优化，采用 systolic array（脉动阵列）架构，FP16 矩阵乘法效率极高。

XLA 的输入是 HLO（High Level Optimizer）中间表示，输出是 TPU 的机器码或 LLVM IR（支持 CPU/GPU）。XLA 的工作流程是：接收 TensorFlow/JAX 的计算图，转换为 HLO，进行图优化（融合、常量折叠、内存规划），然后编译为目标代码。XLA 的优势在于**算子融合**——将多个连续算子编译为一个 kernel，减少内存访问。

TPA v4 Pod（4096 个 TPU 芯片）的峰值性能可达 13 EFLOPS（BF16），是 AI 训练的霸主。Google 内部使用 TPA 训练 PaLM、Gemini 等超大模型，外部可通过 Google Cloud TPA 服务访问。TPA 的局限是只能在 Google Cloud 上使用，无法本地部署，且只支持 TensorFlow/JAX（PyTorch 支持需要通过 PyTorch/XLA）。

## CANN（华为昇腾）
CANN（Compute Architecture for Neural Networks）是华为昇腾 NPU 的计算架构，类似于 NVIDIA 的 CUDA 体系。昇腾 NPU 采用达芬奇架构（DaVinci Architecture），专为深度学习优化，支持 INT8、FP16、FP32 等数据格式。

CANN 的核心组件包括：Ascend CL（C 语言编程接口，类似 CUDA Runtime）、算子开发工具（TBE、TIK）、图编译器（GE，Graph Engine）、推理引擎（ACL）。CANN 提供了完整的深度学习加速栈，包括算子库（HCCL，类似 NCCL）、框架适配层（对接 PyTorch、TensorFlow、MindSpore）。

昇腾 NPU 的主力产品是 Ascend 910 系列（训练）和 Ascend 310 系列（推理）。Ascend 910B 的 FP16 性能可达 750 TFLOPS，约为 NVIDIA A100 的 1/3。华为的优势在于垂直整合——从芯片（昇腾）、框架（MindSpore）、到云服务（ModelArts），形成了完整的国产化生态。对于有国产化需求的政企客户，昇腾 NPU + CANN 是 NVIDIA GPU 的替代方案。

## oneAPI（Intel）
oneAPI 是 Intel 的跨架构编程平台，旨在为 CPU、GPU、FPGA 等不同硬件提供统一的编程接口。oneAPI 的核心是 DPC++（Data Parallel C++），基于 Khronos 的 SYCL 标准，类似于 CUDA C++ 但支持多硬件后端。

oneAPI 的库包括 oneDNN（深度学习算子，原 MKL-DNN）、oneMKL（数学核心库，原 MKL）、oneDAL（数据分析库）等。这些库针对 Intel 的 Xeon CPU（AVX-512）、Arc GPU（Xe-HPG 架构）、FPGA（OpenCL）进行了深度优化。

Intel GPU 在数据中心市场仍在追赶中，Arc A770 的 FP32 性能约为 16 TFLOPS，远低于 NVIDIA A100（30 TFLOPS FP64）。oneAPI 的优势在于 CPU/GPU 统一编程，以及与 Intel x86 服务器的深度整合。对于已部署 Intel 服务器集群的用户，使用 Arc GPU + oneAPI 可以在不增加异构架构成本的情况下提升 AI 计算能力。
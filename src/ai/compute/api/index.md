---
title: 标准接口
order: 15
---
# 科学计算 API
科学计算标准接口是连接上层算法与底层硬件的桥梁。

不同的硬件平台提供了各自的编程接口，从 CUDA 的显式并行到 Metal 的声明式风格，开发者需要根据目标平台选择合适的 API。这些接口的核心价值在于将硬件的并行计算能力暴露给上层应用，同时尽可能屏蔽底层差异。

## Triton

Triton 是 OpenAI 推出的 GPU 编程语言，旨在提供比 CUDA 更高层的抽象，同时保持接近手写的性能。与 CUDA 需要手动管理线程块、shared memory、warp shuffle 等底层概念不同，Triton 让开发者只需描述"每个输出元素如何计算"，编译器自动处理线程映射和内存优化。

Triton 的核心价值在于开发效率。一个矩阵乘法 kernel 在 Triton 中仅需约 50 行代码，而手写 CUDA 需要数百行。对于常见算子（逐元素操作、Softmax、LayerNorm），Triton 的性能可达手写 CUDA 的 85-95%。更重要的是，Triton 支持自动调优，通过 `triton.autotune` 自动搜索最优的 block size、num_stages 等配置，无需人工干预。

Triton 正在成为加速计算领域的"Vulkan"——跨平台、高性能、易用性兼备的 GPU 编程语言。目前支持 NVIDIA GPU 和 AMD GPU（通过 ROCm），PyTorch 2.0 原生支持 Triton，`torch.compile` 会自动将符合模式的算子编译为 Triton kernel。

## CUDA

CUDA (Compute Unified Device Architecture) 是 NVIDIA 的 GPU 编程平台，通过扩展 C++ 语法，让开发者能够编写在 GPU 上执行的并行计算程序。CUDA 是深度学习加速的基石，PyTorch、TensorFlow 等框架的底层都调用 CUDA kernel。

CUDA 的编程模型是异构计算：CPU（host）负责串行逻辑和任务调度，GPU（device）负责并行计算。GPU 的计算单元组织为 SM（Streaming Multiprocessor），每个 SM 包含多个 CUDA core、Tensor Core（矩阵乘法加速单元）、SFU（特殊函数单元）。CUDA 的执行模型是 SIMT（Single Instruction Multiple Threads）：32 个线程组成一个 warp，warp 内的所有线程执行相同的指令，但操作不同的数据。

CUDA 的内存层次从快到慢依次为：register（寄存器）、shared memory（片上内存）、L2 cache、global memory（HBM）。优化 CUDA kernel 的关键是尽可能使用快的内存，如使用 shared memory 缓存频繁访问的数据，保证 global memory 的合并访问。

CUDA 的学习曲线陡峭，需要理解 GPU 架构、内存层次、并行模式等复杂概念。但也因此能榨干硬件性能，对于需要极致优化的场景（如 FlashAttention-2 的手写 assembly），CUDA 仍然是不可替代的选择。

## CPU 并行计算

虽然 GPU 在深度学习训练中占据主导地位，但 CPU 的并行计算能力仍然不可忽视。现代 CPU 通过 SIMD（Single Instruction Multiple Data）指令集实现数据级并行，一条指令同时处理多个数据元素，显著提升计算密集型任务的性能。

### SIMD 指令集

x86 架构的 SIMD 指令集经历了多次演进：

- **MMX**：1996 年引入，64 位寄存器，只能处理整数，与浮点寄存器共享导致状态切换开销
- **SSE**：1999 年引入，128 位寄存器（XMM），支持浮点运算，SSE2 增加整数支持，SSE4.1 增加混合运算指令
- **AVX**：2008 年引入，256 位寄存器（YMM），AVX2 支持整数运算，AVX-512 将宽度扩展到 512 位（ZMM），但功耗和发热问题导致 Intel 在后续产品中逐步缩减 AVX-512 支持

ARM 架构的 SIMD 指令集称为 **NEON**，128 位寄存器，支持整数和浮点运算，在移动设备和 Apple Silicon 芯片中广泛使用。Apple M 系列芯片的 ARM 架构配合 AMX（Apple Matrix Multiply）协处理器，在矩阵乘法上性能可达传统 CPU 的数倍。

SIMD 编程的核心思想是向量化：将标量运算转换为向量运算。例如，计算两个数组的元素和，标量代码需要循环逐个处理，而 SIMD 代码可一次处理 8 个 float（AVX2 的 256 位寄存器可容纳 8 个 32-bit float）。编译器可通过自动向量化（auto-vectorization）将简单循环转换为 SIMD 指令，但对于复杂模式（如依赖、分支），仍需使用内联汇编（`_mm256_add_ps` 等 intrinsics）手动编写。

### 并行编程框架

除了 SIMD，CPU 还可通过多线程并行实现任务级并行。OpenMP 是最常用的并行编程框架，通过 `#pragma omp parallel for` 编译制导指令，将循环分配到多个线程执行。对于 NUMA 架构的多路服务器，还需考虑 NUMA 亲和性，将线程绑定到特定 CPU 核心，减少跨 socket 的内存访问延迟。

BLAS（Basic Linear Algebra Subprograms）是线性代数的标准 API，各大厂商提供了优化的实现：Intel MKL（Math Kernel Library）、OpenBLAS（开源）、Apple Accelerate（macOS）。这些库使用 SIMD 指令和汇编优化，矩阵乘法性能可达手写代码的数十倍。PyTorch 的 CPU 后端默认使用 Intel MKL，在 AVX-512 支持的 CPU 上，FP32 矩阵乘法性能可达 500 GFLOPS，约为 A100 GPU（20 TFLOPS）的 2.5%，对于小模型推理已足够。

### 集成显卡加速

集成显卡（Integrated GPU，iGPU）是与 CPU 共享内存和封装的 GPU，虽然性能远不如独立显卡，但在推理场景中仍可提供显著加速。Intel 的集成显卡支持 OpenCL 和 oneAPI（DPC++/SYCL），可通过 GPU 分享系统内存无需数据传输，延迟远低于独立显卡的 PCIe 传输。Apple M 系列芯片的 GPU 核心通过统一内存架构与 CPU 共享内存，Metal Performance Shaders（MPS）提供了类似 CUDA 的计算 API，PyTorch 的 MPS 后端可直接调用。

集成显卡的优势在于功耗和成本。对于边缘设备（如摄像头、机器人），使用集成显卡进行推理无需额外的独立显卡，可降低功耗和硬件成本。Intel 的 OpenVINO 工具包专门优化了集成显卡上的推理性能，支持模型量化（INT8）和算子融合，在 Intel Iris Xe 集成显卡上，ResNet-50 推理性能可达 500 FPS（图像尺寸 224×224），足以满足实时应用需求。

## Metal

Metal 是 Apple 的图形和计算 API，为 macOS、iOS、tvOS 提供了硬件加速的统一接口。Metal 不同于 OpenGL 的跨平台定位，专注于 Apple 硬件的深度优化，通过更贴近硬件的设计降低了 CPU 开销，提升了 GPU 利用率。

Metal 的编程模型与 CUDA 类似，也采用 host-device 异构执行、kernel 启动、内存管理等概念。Metal Shading Language（MSL）是基于 C++14 的着色语言，用于编写在 GPU 上执行的代码。与 CUDA 不同的是，Metal 使用声明式资源管理（通过 MTLBuffer、MTLTexture 对象），由驱动自动管理内存生命周期，减少了手动管理的错误。

Metal 的核心优势是与 Apple 硬件的深度集成。在 Apple M 系列芯片上，Metal 可利用统一内存架构，CPU 和 GPU 共享内存，无需数据传输。Metal Performance Shaders（MPS）提供了高性能的卷积、矩阵乘法、归约等算子库，性能接近手写 Metal 代码。Metal Performance Shaders Graph 进一步支持算子融合，可将多个算子编译为一个 Metal kernel，减少内存访问。

PyTorch 的 MPS 后端通过 Metal Performance Shaders 调用 GPU，在 Apple M 系列芯片上可获得显著加速。例如，ResNet-50 推理在 M1 Max 上的性能约为 800 FPS（图像尺寸 224×224），约为 CPU（Intel MKL）的 10 倍。对于小模型推理（如 BERT-base），MPS 的性能约为 A100 GPU 的 5-10%，但功耗仅为其十分之一，适合本地部署场景。

## ROCm

ROCm (Radeon Open Compute) 是 AMD 的 GPU 计算平台，类似于 NVIDIA 的 CUDA 体系。ROCm 的核心定位是面向高性能计算（HPC）和 AI 的统一平台，提供开源的驱动、编译器、运行时和库。

ROCm 的编程语言是 HIP (Heterogeneous-compute Interface for Portability)，语法与 CUDA 高度兼容，大部分 CUDA 代码可通过 hipify 工具自动转换为 HIP 代码。ROCm 还支持 OpenCL，但由于性能不如 HIP，主要用于跨平台场景。ROCm 的库包括 rocBLAS（类似 cuBLAS）、rocFFT（类似 cuFFT）、MIOpen（类似 cuDNN），提供了完整的深度学习加速栈。

ROCm 是 AMD GPU 计算的主力生态，尤其在数据中心、AI 训练、科研计算等领域。与 NVIDIA 的闭源策略不同，ROCm 的开源使得开发者可深入理解底层实现，也便于学术研究定制优化。然而，ROCm 的生态系统仍不如 CUDA 成熟，工具链（如 ROCgdb、ROCprofiler）和第三方库的支持有限，对于需要稳定支持的生产环境，NVIDIA GPU 仍是更安全的选择。

近年来，ROCm 在性能和兼容性上取得了显著进展。AMD Instinct MI300X（CDNA 3 架构）的 FP16 矩阵乘法性能可达 2 PFLOPS，约为 NVIDIA H100（FP16 BF16）的 80%。PyTorch 的 ROCm 后端已进入主分支，大部分 CUDA 算子可通过 HIP 运行在 AMD GPU 上，无需修改代码。对于预算受限的学术机构，AMD GPU + ROCm 是 NVIDIA GPU 的可行替代方案。

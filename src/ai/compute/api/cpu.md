---
title: SIMD
order: 50
---

# SIMD
SIMD（Single Instruction Multiple Data）是 CPU 实现数据级并行的核心技术。与 GPU 的粗粒度并行（数千个线程同时执行）不同，SIMD 在单个指令周期内同时处理多个数据元素，通过向量化操作提升计算密集型任务的性能。虽然 GPU 在深度学习训练中占据主导地位，但 CPU 的 SIMD 能力在推理场景中仍然不可忽视。

## x86 SIMD 指令集演进
x86 架构的 SIMD 指令集经历了多次演进，每次更新都带来更大的寄存器宽度和更丰富的指令集。

| 指令集  | 年份 | 寄存器宽度    | 寄存器数 | 主要特性                         |
| ------- | ---- | ------------- | -------- | -------------------------------- |
| MMX     | 1996 | 64-bit (MM)   | 8        | 仅整数，与浮点寄存器共享         |
| SSE     | 1999 | 128-bit (XMM) | 8        | 引入浮点支持                     |
| SSE2    | 2001 | 128-bit (XMM) | 8        | 扩展到 64-bit 系统，增加整数操作 |
| SSE3    | 2004 | 128-bit (XMM) | 8        | 增加水平操作（hadd、hsub）       |
| SSE4.1  | 2006 | 128-bit (XMM) | 8        | 增加混合运算、点积指令           |
| AVX     | 2008 | 256-bit (YMM) | 16       | 寄存器宽度翻倍                   |
| AVX2    | 2013 | 256-bit (YMM) | 16       | 增加整数操作                     |
| AVX-512 | 2017 | 512-bit (ZMM) | 32       | 寄存器宽度再次翻倍               |

MMX 是最早的 SIMD 指令集，但设计缺陷导致它很快被 SSE 取代。MMX 的寄存器（MM0-MM7）实际上复用了 x87 浮点寄存器，这意味着在 MMX 代码和浮点代码间切换时需要调用 `EMMS` 指令清空寄存器状态，带来额外开销。SSE 引入了独立的 XMM 寄存器（XMM0-XMM7），解决了这个问题，并首次引入了浮点 SIMD 操作。

AVX 将寄存器宽度从 128 位扩展到 256 位，使得一条指令可以处理 8 个 float（32-bit）或 4 个 double（64-bit）。AVX2 进一步增加了整数 SIMD 操作，使得整数和浮点都可用向量指令处理。AVX-512 将寄存器宽度扩展到 512 位，但由于功耗和发热问题，Intel 在后续产品中逐步缩减 AVX-512 支持（如 Alder Lake 只在特定 SKU 上启用）。

## ARM SIMD 指令集
ARM 架构的 SIMD 指令集称为 **NEON**，128 位寄存器，支持整数和浮点运算。NEON 在移动设备和 Apple Silicon 芯片中广泛使用，是移动端图像处理、音频编解码的核心加速技术。

```asm
// ARM NEON 示例：两个向量相加
vld1.32 {q0}, [r0]    @ 加载 4 个 float 到 q0
vld1.32 {q1}, [r1]    @ 加载 4 个 float 到 q1
vadd.f32 q0, q0, q1   @ q0 = q0 + q1（4 个 float 并行相加）
vst1.32 {q0}, [r2]    @ 存储结果
```

Apple M 系列芯片的 ARM 架构配合 **AMX（Apple Matrix Multiply）** 协处理器，在矩阵乘法上性能可达传统 CPU 的数倍。AMX 是 Apple 专有的矩阵加速单元，支持 8×8 或 16×16 矩阵乘法，主要用于神经网络的加速。PyTorch 的 MPS 后端会优先使用 AMX 进行矩阵运算。

## SIMD 编程
SIMD 编程的核心思想是**向量化**：将标量运算转换为向量运算。例如，计算两个数组的元素和，标量代码需要循环逐个处理，而 SIMD 代码可一次处理 8 个 float（AVX2 的 256 位寄存器可容纳 8 个 32-bit float）。

```c
// 标量代码
void add_scalar(float* x, float* y, float* z, int n) {
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}

// SIMD 代码（使用 AVX2 intrinsics）
#include <immintrin.h>

void add_simd(float* x, float* y, float* z, int n) {
    int i = 0;
    // 每次处理 8 个 float（256 位 / 32 位 = 8）
    for (; i + 8 <= n; i += 8) {
        __m256 a = _mm256_loadu_ps(x + i);  // 加载 8 个 float
        __m256 b = _mm256_loadu_ps(y + i);
        __m256 c = _mm256_add_ps(a, b);     // 向量加法
        _mm256_storeu_ps(z + i, c);         // 存储 8 个 float
    }
    // 处理剩余元素
    for (; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
```

编译器可通过自动向量化（auto-vectorization）将简单循环转换为 SIMD 指令，无需手动编写 intrinsics。GCC 的 `-O3 -ftree-vectorize`、Clang 的 `-O3 -Rpass=loop-vectorize` 会启用自动向量化。但自动向量化对循环模式有严格要求：循环体内不能有函数调用、分支依赖、复杂的内存访问模式，否则编译器会放弃向量化。

```c
// 可自动向量化
for (int i = 0; i < n; i++) {
    z[i] = x[i] + y[i];
}

// 难以自动向量化（有分支依赖）
for (int i = 0; i < n; i++) {
    if (x[i] > 0) {
        z[i] = y[i];
    } else {
        z[i] = -y[i];
    }
}
```

对于复杂模式，仍需使用 intrinsics 手动编写 SIMD 代码。主流编译器（GCC、Clang、MSVC）都支持 `immintrin.h` 头文件，提供了 MMX、SSE、AVX、AVX-512 的 intrinsics。

## 多线程并行
除了 SIMD，CPU 还可通过多线程并行实现任务级并行。**OpenMP** 是最常用的并行编程框架，通过 `#pragma omp parallel for` 编译制导指令，将循环分配到多个线程执行。

```c
#include <omp.h>

void add_parallel(float* x, float* y, float* z, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        z[i] = x[i] + y[i];
    }
}
```

OpenMP 会根据 CPU 核心数自动创建线程池，并将循环迭代分配给不同线程。对于 NUMA 架构的多路服务器，还需要考虑 NUMA 亲和性，将线程绑定到特定 CPU 核心，减少跨 socket 的内存访问延迟。Linux 的 `numactl` 命令或 `pthread_setaffinity_np` 函数可用于绑定线程到 CPU 核心。

SIMD 和多线程可以结合使用，实现两级并行：线程级并行（多核）+ 数据级并行（SIMD）。这种组合在现代 CPU 上可获得接近理论峰值的性能。

## BLAS 和线性代数库
BLAS（Basic Linear Algebra Subprograms）是线性代数的标准 API，定义了向量运算（Level 1）、矩阵-向量运算（Level 2）、矩阵-矩阵运算（Level 3）的接口。各大厂商提供了优化的 BLAS 实现：

| 库               | 平台          | 特点                   |
| ---------------- | ------------- | ---------------------- |
| Intel MKL        | Intel CPU     | 优化最深，支持 AVX-512 |
| OpenBLAS         | 开源，跨平台  | 性能接近 MKL           |
| Apple Accelerate | Apple Silicon | 针对 AMX 优化          |
| BLIS             | 开源，跨平台  | 模块化设计，易于移植   |

这些库使用 SIMD 指令和汇编优化，矩阵乘法性能可达手写代码的数十倍。PyTorch 的 CPU 后端默认使用 Intel MKL，在 AVX-512 支持的 CPU 上，FP32 矩阵乘法性能可达 500 GFLOPS，约为 A100 GPU（20 TFLOPS）的 2.5%，对于小模型推理已足够。

```python
import torch

# 使用 CPU 后端进行矩阵乘法
x = torch.randn(1024, 1024)
y = torch.randn(1024, 1024)
z = torch.mm(x, y)  # 调用 MKL 的 sgemm
```

## 集成显卡加速
集成显卡是与 CPU 共享内存和封装的 GPU，虽然性能远不如独立显卡，但在推理场景中仍可提供显著加速。Intel 的集成显卡支持 OpenCL 和 oneAPI（DPC++/SYCL），可通过 GPU 分享系统内存无需数据传输，延迟远低于独立显卡的 PCIe 传输。

Apple M 系列芯片的 GPU 核心通过统一内存架构与 CPU 共享内存，Metal Performance Shaders（MPS）提供了类似 CUDA 的计算 API，PyTorch 的 MPS 后端可直接调用。在 M1 Max 上，ResNet-50 推理性能约为 800 FPS（图像尺寸 224×224），约为 CPU（Intel MKL）的 10 倍。

集成显卡的优势在于功耗和成本。对于边缘设备（如摄像头、机器人），使用集成显卡进行推理无需额外的独立显卡，可降低功耗和硬件成本。Intel 的 OpenVINO 工具包专门优化了集成显卡上的推理性能，支持模型量化（INT8）和算子融合，在 Intel Iris Xe 集成显卡上，ResNet-50 推理性能可达 500 FPS。

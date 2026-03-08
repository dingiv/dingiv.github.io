---
title: SIMT
order: 55
---

# SIMT

SIMT（Single Instruction Multiple Threads）是 GPU 并行计算的基础架构模型，与 CPU 的 SIMD（Single Instruction Multiple Data）有相似之处，但编程模型和硬件实现存在本质差异。对于习惯了 CPU 编程的开发者来说，理解 SIMT 的关键是转变思维模式：从"手动向量化的单线程"转向"大规模并行的多线程"。

## 从 SIMD 到 SIMT
CPU 的 SIMD 要求开发者显式管理向量寄存器，将多个数据打包后执行单条指令。以 AVX-512 为例，开发者需要手动将 16 个 float32 打包进 512 位寄存器，调用 `_mm512_add_ps` 执行加法，再手动解包结果。这种方式虽然精确可控，但编程复杂度高，需要处理数据对齐、打包解包、掩码操作等细节。

GPU 的 SIMT 采用了不同的思路。开发者编写的是标量代码——描述单个线程如何处理单个数据元素，GPU 硬件负责将大量线程（数万至数百万）编组执行。NVIDIA 将 32 个线程编组为一个 warp，AMD 将 64 个线程编组为一个 wavefront，编组内的线程同时执行相同指令但操作不同数据。这种隐式并行的模型降低了编程门槛，开发者无需关心底层向量宽度，只需关注任务如何划分。

```cpp
// CPU SIMD 代码（需要手动处理向量化）
#include <immintrin.h>
void add_simd(float* x, float* y, float* z, int n) {
    for (int i = 0; i < n; i += 16) {
        __m512 a = _mm512_loadu_ps(x + i);
        __m512 b = _mm512_loadu_ps(y + i);
        __m512 c = _mm512_add_ps(a, b);
        _mm512_storeu_ps(z + i, c);
    }
}

// GPU SIMT 代码（标量思维）
__global__ void add_simt(const float* x, const float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}
```

这种差异反映了两种硬件的设计目标：CPU SIMD 追求低延迟，通过复杂的分支预测和缓存层次减少单个任务的执行时间；GPU SIMT 追求高吞吐，通过大量线程掩盖内存延迟，当某个 warp 等待内存数据时，调度器会切换到另一个 warp 执行，确保计算单元始终忙碌。

## 线程层次结构
CUDA 的线程模型分为三层：thread（线程）、block（线程块）、grid（网格）。thread 是最基本的执行单元，多个 thread 组成一个 block，多个 block 组成一个 grid。这种层次结构不仅是编程抽象，也映射到 GPU 的物理硬件——block 映射到 SM（Streaming Multiprocessor），thread 在 warp 级别调度。

```cpp
// 启动配置：grid 有 (n + 255) / 256 个 block，每个 block 有 256 个 thread
int threads_per_block = 256;
int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
add_kernel<<<blocks_per_grid, threads_per_block>>>(x, y, z, n);
```

每个 thread 可通过内置变量获取自身标识：`threadIdx` 是 thread 在 block 内的索引，`blockIdx` 是 block 在 grid 内的索引，`blockDim` 是 block 的维度（如 256），`gridDim` 是 grid 的维度。全局线程 ID 通常通过 `idx = blockIdx.x * blockDim.x + threadIdx.x` 计算，用于定位 thread 处理的数据元素。

block 内的 thread 可以通过 shared memory 通信和同步。shared memory 是片上内存，延迟低（~100 clock cycles）但容量小（每 SM 约 64KB），适合缓存 block 内频繁访问的数据。`__syncthreads()` 用于同步 block 内的 thread，确保所有 thread 都执行到同步点后才继续执行。

```cpp
__global__ void reduce_kernel(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 加载到 shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // 写回结果
    if (tid == 0) output[blockIdx.x] = sdata[0];
}
```

## 内存层次与访问优化
GPU 的内存层次从快到慢依次为：register（寄存器）、shared memory（片上内存）、L2 cache、global memory（HBM）。register 是 thread 私有的，速度最快但数量有限（每个 thread 约 255 个 32-bit register），超出限制会触发 register spilling，溢出到 global memory 导致性能骤降。

global memory 是 GPU 的主存（显存），容量大（A100 80GB）但延迟高（~300 clock cycles）。优化 global memory 访问的关键是合并访问（coalescing）：warp 内的 32 个 thread 应访问连续的 32 个元素，硬件可将这 32 次访问合并为 1 次（或少数几次）内存事务。如果 thread 访问跨步较大的地址（如每 128 个元素访问一次），会触发多次独立事务，浪费带宽。

```cpp
// 好的访问模式（连续访问）
__global__ void good_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx];  // warp 内访问连续地址
}

// 差的访问模式（跨步访问）
__global__ void bad_access(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = data[idx * 128];  // warp 内访问跨步地址
}
```

shared memory 的另一个重要用途是缓存 tile（数据块）。以矩阵乘法为例，可将矩阵 $A$ 和 $B$ 分块为 $A_{tile}$ 和 $B_{tile}$（如 32×32），每个 block 加载一个 tile 到 shared memory 后，block 内计算无需重复访问 global memory。这种 tiling 策略可显著减少内存访问次数，提升带宽利用率。

## 分支发散

分支发散是 SIMT 编程中最常见的性能陷阱。当 warp 内的 thread 走入不同分支（如部分 thread 执行 if，部分执行 else），GPU 需要串行执行各分支——先执行 if 分支（未进入该分支的 thread 被禁用），再执行 else 分支。如果 warp 内频繁出现分支发散，并行性会退化为串行，效率大幅下降。

```cpp
// 可能产生分支发散的代码
__global__ void branch_divergence(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (data[idx] > 0.5f) {
            data[idx] = sqrtf(data[idx]);
        } else {
            data[idx] = data[idx] * data[idx];
        }
    }
}
```

处理分支发散的策略包括：重新组织数据使 warp 内 thread 走相同分支、使用算术指令替代分支（如通过 `min`/`max` 或三元运算符）、使用掩码操作（类似 CPU SIMD 的 mask）。现代 GPU（如 NVIDIA Ampere）引入了独立线程调度，可在一定程度上缓解分支发散的影响，但仍需尽量避免。

## 吞吐量导向设计

CPU 设计追求低延迟，通过复杂的分支预测、乱序执行、大容量缓存减少单个任务的执行时间。GPU 设计追求高吞吐，通过大量线程掩盖延迟，当某个 warp 等待内存数据时，调度器会切换到另一个 warp，确保计算单元始终忙碌。这种差异意味着 GPU 编程需要以吞吐量为导向，而非延迟。

衡量 GPU 利用率的指标是 occupancy（占用率），即实际驻留在 SM 上的 warp 数量与 SM 可容纳的最大 warp 数量之比。occupancy 受限于 block 大小、register 使用量、shared memory 使用量。理论上 occupancy 越高越好，但并非绝对——高 occupancy 不等于高性能，有时减少每个 block 的 thread 数可降低 resource contention（资源争用），反而提升性能。

```cpp
// 使用 CUDA 工具分析 occupancy
// nvcc --ptxas-options=-v your_kernel.cu  # 查看 register 和 shared memory 使用量
// cudaOccupancyMaxActiveBlocksPerMultiprocessor  # 计算最大 active block 数
```

GPU 编程的黄金法则是：暴露足够的并行度、优化内存访问、减少分支发散。具体来说，确保有足够的 thread 和 block 填满所有 SM、保证 global memory 访问合并、合理使用 shared memory 缓存热数据、尽量避免 warp 内分支。对于深度学习算子开发，这些原则尤为重要——矩阵乘法、卷积、归约等核心算子的性能直接影响模型训练和推理速度。

## 工具与调试

CUDA 提供了丰富的工具链用于性能分析和调试。`nvcc` 编译器可将 `.cu` 文件编译为 PTX（中间表示）或 cubin（二进制），PTX 在运行时由驱动编译为 SASS（GPU 机器码），实现跨代兼容。`cuda-gdb` 是 GDB 的 CUDA 扩展，支持设置断点、检查变量、单步执行 kernel。

`nsys`（Nsight Systems）分析 kernel 的执行时间、内存传输、CPU-GPU 并行情况，可识别性能瓶颈。`ncu`（Nsight Compute）深入分析 kernel 的内存带宽、occupancy、指令混合，提供更细致的优化建议。对于初次接触 GPU 编程的开发者，建议从 `nsys` 开始，先整体把握程序性能分布，再针对热点 kernel 使用 `ncu` 深入分析。

```bash
# Nsight Systems 分析
nsys profile --stats=true python your_script.py

# Nsight Compute 分析
ncu --set full python your_script.py
```

理解 SIMT 模型是 GPU 编程的第一步。对于有 CPU SIMD 经验的开发者，关键转变是从"手动管理向量寄存器"到"编写标量代码、让硬件自动并行"；对于没有并行编程经验的开发者，需要理解线程层次结构、内存访问模式、分支发散等概念。GPU 编程的学习曲线比 CPU SIMD 更平缓，但要写出高性能 kernel 仍需深入理解硬件架构和反复调优。

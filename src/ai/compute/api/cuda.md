# CUDA
# CUDA
CUDA (Compute Unified Device Architecture) 是 NVIDIA 的 GPU 编程平台，通过扩展 C++ 语法，让开发者能够编写在 GPU 上执行的并行计算程序。CUDA 是深度学习加速的基石，PyTorch、TensorFlow 等框架的底层都调用 CUDA kernel。

## 编程模型

CUDA 的编程模型是**异构计算**：CPU（host）负责串行逻辑和任务调度，GPU（device）负责并行计算。CPU 通过 `cudaMalloc` 在 GPU 上分配内存，通过 `cudaMemcpy` 在 CPU 和 GPU 间传输数据，通过 `kernel<<<grid, block>>>(...)` 启动 GPU 计算。

GPU 的计算单元组织为 **SM（Streaming Multiprocessor）**，每个 SM 包含多个 CUDA core（FP32 单元）、Tensor Core（FP16/BF16/INT8 矩阵乘法加速单元）、SFU（Special Function Unit，计算 sin/cos/exp 等特殊函数）。A100 有 108 个 SM，每个 SM 有 64 个 FP32 core、4 个 Tensor Core。

CUDA 的执行模型是 **SIMT（Single Instruction Multiple Threads）**：32 个线程组成一个 warp，warp 内的所有线程执行相同的指令，但操作不同的数据。如果 warp 内的线程走不同分支（如 if-else），则需要串行执行各分支，降低性能。因此，编写 CUDA kernel 时应尽量保证 warp 内线程路径一致，避免分支发散。

## 内存层次

GPU 的内存层次从快到慢依次为：register（寄存器，~200TB/s）、shared memory（片上内存，~20TB/s）、L2 cache（~5TB/s）、global memory（HBM，~2TB/s）。优化 CUDA kernel 的关键是尽可能使用快的内存。

register 是线程私有的，速度最快但数量有限（每个线程约 255 个 32-bit register）。超出限制会触发 register spilling（溢出到 global memory），性能急剧下降。shared memory 是 block 内线程共享的，可通过 `__shared__` 声明，用于块内线程通信和缓存频繁访问的数据。矩阵乘法的 tiling 算法就是经典例子：将矩阵块加载到 shared memory 后，块内计算无需访问 global memory。

global memory 是 GPU 的主存（显存），容量大（A100 80GB）但延迟高（~300 clock cycles）。优化 global memory 访问的关键是**合并访问**（coalescing）：warp 内的 32 个线程应访问连续的 32 个元素，这样硬件可将这 32 次访问合并为 1 次事务。如果线程访问跨步较大的地址（如每 128 个元素访问一次），会触发多次事务，降低带宽利用率。

## Kernel 编写

CUDA kernel 以 `__global__` 修饰的函数表示，该函数在 GPU 上执行，由 CPU 调用。kernel 内部可通过 `threadIdx`（线程在线程块内的索引）、`blockIdx`（线程块在网格内的索引）、`blockDim`（线程块大小）、`gridDim`（网格大小）计算全局线程 ID。

```cpp
__global__ void add_kernel(const float* x, const float* y, float* z, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        z[idx] = x[idx] + y[idx];
    }
}

// CPU 调用
int threads_per_block = 256;
int blocks_per_grid = (n + threads_per_block - 1) / threads_per_block;
add_kernel<<<blocks_per_grid, threads_per_block>>>(x, y, z, n);
```

矩阵乘法的 kernel 更复杂，需要使用 shared memory tiling 减少全局显存访问。基本思想：将矩阵 $A$ 和 $B$ 分块为 $A_{tile}$ 和 $B_{tile}$（如 32×32），每个 block 加载一个 tile 到 shared memory，然后计算 tile 的部分积，累加到输出 $C$。这样每个元素只需访问 global memory 2 次（加载 $A_{tile}$ 和 $B_{tile}$），而非 $K$ 次（标准算法需要 $K$ 次加载）。

## Tensor Core

Tensor Core 是 NVIDIA GPU 上专门的矩阵乘法加速单元，支持 FP16/BF16/INT8/TF32 等格式的矩阵乘法累加（MMA，Matrix Multiply Accumulate）。使用 Tensor Core 需要调用 WMMA（Warp Matrix Multiply Accumulate）API：

```cpp
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void matmul_tensor_core(half* A, half* B, half* C, int M, int N, int K) {
    // 初始化 fragment（warp 级别的矩阵块）
    fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, half, col_major> b_frag;
    fragment<accumulator, 16, 16, 16, half> c_frag;

    // 加载矩阵块到 fragment
    load_matrix_sync(a_frag, A + ...);
    load_matrix_sync(b_frag, B + ...);
    fill_fragment(c_frag, 0.0f);

    // 矩阵乘法累加（使用 Tensor Core）
    mma_sync(c_frag, a_frag, b_frag, c_frag);

    // 存储结果
    store_matrix_sync(C + ..., c_frag, mem_row_major);
}
```

Tensor Core 要求矩阵维度是 16 的倍数（fragment 大小为 16×16），且数据格式为 FP16/BF16/INT8。对于不满足条件的矩阵，需要 padding 填充到 16 的倍数。Tensor Core 的性能是 FP32 的 8 倍（FP16）或 32 倍（INT8），是深度学习加速的核心。

## 工具链

CUDA 的工具链包括编译器（`nvcc`）、调试器（`cuda-gdb`）、性能分析器（`nsys`、`ncu`）。`nvcc` 将 `.cu` 文件编译为 PTX（中间表示）或 cubin（二进制），PTX 在运行时由驱动编译为 SASS（GPU 机器码），这实现了跨代兼容（同一 PTX 可在不同 GPU 上运行）。`cuda-gdb` 是 GDB 的 CUDA 扩展，支持设置断点、检查变量、单步执行 kernel。`nsys`（Nsight Systems）分析 kernel 的执行时间、内存传输、CPU-GPU 并行情况；`ncu`（Nsight Compute）深入分析 kernel 的内存带宽、occupancy、指令混合，指导优化。

## 学习路径

CUDA 的学习曲线陡峭，建议从简单到复杂逐步深入：

1. **基础概念**：理解 host-device 异构执行、kernel 启动、线程索引计算
2. **内存管理**：掌握 `cudaMalloc`、`cudaMemcpy`、内存层次（register、shared、global）
3. **并行模式**：学习归约（reduce）、前缀和（scan）、直方图（histogram）等常见并行算法
4. **性能优化**：理解 occupancy（SM 占用率）、合并访问、warp divergence、shared memory bank conflict
5. **Tensor Core**：学习 WMMA API、 CUTLASS 库（NVIDIA 的模板化矩阵乘法库）

编写高效 CUDA kernel 需要对 GPU 架构有深入理解，但也因此能榨干硬件性能。对于深度学习开发者，直接编写 CUDA 的场景不多（大部分时间使用 PyTorch），但理解 CUDA 有助于调试性能问题、阅读 `torch.nn.functional` 的底层实现、编写自定义算子。


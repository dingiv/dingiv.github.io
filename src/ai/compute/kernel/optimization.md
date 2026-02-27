---
title: 算子优化
order: 30
---

# 算子优化
算子是深度学习模型计算的基本单元，如矩阵乘法、卷积、注意力等。算子优化通过改进 CUDA kernel 实现，提升单算子的计算效率，是提升模型性能的基础。相比于框架层面的优化（如分布式策略），算子优化更贴近硬件，收益稳定且可迁移。

## 性能瓶颈

GPU 计算的理论性能由 FLOPS 衡量，但实际性能受限于**内存带宽**和**延迟**。A100 的 FP16 理论性能为 312 TFLOPS，显存带宽 2TB/s，这意味着每个浮点运算需要访存约 6.4 字节才能充分利用计算单元。但深度学习算子的算术强度（arithmetic intensity，计算量/访存量）往往低于此阈值，导致性能受限于内存带宽而非计算单元。

以矩阵乘法 $C = AB$ 为例，其中 $A \in [M, K]$，$B \in [K, N]$，$C \in [M, N]$。标准算法需要 $2MNK$ 次 FLOPs，访存 $MN + NK + MK$ 个元素。当 $M=N=K=4096$ 时，算术强度为 $2 \times 4096 / 3 \approx 2730$ FLOPs/byte，远超 A100 的 6.4 FLOPs/byte 阈值，因此矩阵乘法是计算密集型（compute-bound）算子，性能受限于计算单元而非显存带宽。

但对于逐元素操作（如 ReLU、Add），算术强度接近于 0，因为每个元素需要读写显存但计算量很少。这类算子的性能受限于显存带宽，优化方向是减少访存次数（如算子融合）。

另一个瓶颈是 **warp divergence**。CUDA 以 warp（32 个线程）为单位执行指令，如果 warp 中的线程走不同分支（如 if-else），则需要串行执行各分支，降低并行度。这要求 kernel 设计时尽量保证 warp 内线程路径一致。

## 优化技术

共享内存（shared memory）是 GPU 片上内存，带宽远高于全局显存（A100 的 shared memory 带宽约 20TB/s，global memory 约 2TB/s）。通过将频繁访问的数据加载到 shared memory，可大幅减少全局显存访问。矩阵乘法的 tiling 算法就是典型例子：将矩阵块加载到 shared memory 后，块内计算无需再次访问全局显存。

算子融合通过将多个连续算子合并为一个 kernel 来减少显存读写。例如 LayerNorm 后接 Residual ($y = \text{LayerNorm}(x + z)$)，标准实现需要读写显存三次（读 $x, z$，写中间结果，读中间结果，写 $y$），融合后只需一次读写（读 $x, z$，写 $y$）。FlashAttention 更是将 Attention 的多次分块计算融合为单个 kernel，将显存访问减少 10 倍以上。

指令级并行（ILP）通过在单个线程内发射多条独立指令来隐藏延迟。例如在等待显存加载时，可执行与已加载无关的计算。CUDA 编译器会自动进行指令调度，但手动使用 `#pragma unroll` 展开循环、减少分支、向量化操作可进一步提升 ILP。

Tensor Core 是 NVIDIA GPU 上专门的矩阵乘法加速单元，通过牺牲精度（FP16/BF16/INT8）换取速度（FP16 矩阵乘法比 FP32 快 8 倍）。使用 Tensor Core 需要满足特定条件：矩阵维度是 16 的倍数、数据格式为 FP16/BF16/INT8、调用 WMMA（Warp Matrix Multiply Accumulate）API。现代深度学习框架会自动使用 Tensor Core，但自定义算子需要手动调用。

## 工具与生态

编写高效 CUDA kernel 需要深入理解硬件架构，门槛较高。为此，一系列高层抽象工具应运而生。

Triton 是 OpenAI 开发的类 Python 语言，用于编写 GPU 算子。它的抽象级别高于 CUDA，无需手动管理 shared memory、thread block、warp shuffle，只需编写计算逻辑，编译器自动优化为高效 kernel。Triton 的性能可达到手写 CUDA 的 90% 以上，但开发效率提升 5-10 倍。

cuDNN 是 NVIDIA 提供的深度学习算子库，包含卷积、池化、激活、归一化等常见算子的高度优化实现。这些实现针对不同 GPU 架构（Volta、Turing、Ampere、Hopper）分别优化，性能远超开源实现。PyTorch 的 `torch.nn.functional.conv2d` 底层就调用 cuDNN。

cutlass 是 NVIDIA 开源的模板库，用于编写高性能的矩阵乘法、卷积 kernel。它封装了 Tensor Core、shared memory tiling、指令级优化等底层细节，开发者只需配置矩阵形状、数据类型、分块大小即可生成高效 kernel。cutlass 常用于自定义算子中的矩阵乘法部分。

## 优化流程

算子优化的第一步是**性能分析**。使用 Nsight Compute、nvprof、PyTorch profiler 等工具定位热点算子。A100 的理论 FLOPS 为 312 TFLOPS（FP16），如果某算子实测仅 50 TFLOPS，说明优化空间很大。

第二步是**算法优化**。选择更优的算法可减少 FLOPs，如 Winograd 卷积将 FLOPs 降低 2-3 倍，FFT 卷积将 $O(n^2)$ 降为 $O(n \log n)$。但算法优化可能改变数值精度，需要验证。

第三步是**实现优化**。使用 shared memory tiling、Tensor Core、指令级并行等技术提升 kernel 效率。这一步需要反复 benchmark，调整分块大小、展开循环、融合算子，直到接近理论性能峰值。

最后是**集成测试**。将优化后的算子集成到模型中，验证端到端性能提升和数值正确性。有时算子层面优化了 50%，但模型层面仅提升 5%，因为瓶颈转移到其他算子。

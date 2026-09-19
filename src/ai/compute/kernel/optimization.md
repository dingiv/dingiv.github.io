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

## 硬件层细节
**Warp 与调度器**：CUDA 以 warp（32 个线程）为调度单位，每个 SM 有 4 个 warp scheduler。同一 scheduler 每周期只能发射一条指令到它负责的 8 个 warp 中——这意味着一个 SM 最多同时跑 32 个 warp 才能喂饱 4 个 scheduler。Ampere 每个 SM 最多 64 个 warp，Hopper 提升到 64 个 warp（但单 warp 调度能力因 wgma 提升）。Kernel 设计需要保证 occupancy（每个 SM 活跃 warp 数）足够高，否则 scheduler 会“闲”住。

**寄存器压力与 occupancy 平衡**：每个线程占用的寄存器越多，同一个 SM 能容纳的 warp 数越少。例如每个线程 64 个寄存器时，Ampere SM 可容纳 64 个 warp；每个线程 128 个寄存器时只能容纳 32 个。Llama 70B 的 attention kernel 为了走 FlashAttention 路径，需要寄存器存累积的 softmax 分母，寄存器压力较高——这是为什么 FA2 在大 batch 下 occupancy 反而下降、加速比衰减的根源。优化手段是**双缓冲**（用两套缓冲区交替，让一组算的时候另一组加载）或**寄存器溢出到 SMEM**（强制但低效）。

**Shared Memory 容量约束**：Ampere 每个 SM 有 164KB SMEM（但单 kernel 可用通常 96KB），Hopper 提升到 228KB。SMEM 是片上内存（带宽 ~20TB/s，远超 HBM 的 2TB/s），但总量有限——如果一个 kernel 用了 100KB SMEM，同一 SM 上只能跑 1 个 thread block，occupancy 直接腰斩。FlashAttention、Marlin、CUTLASS GEMM 都是 SMEM 重度使用者，参数调优（block size、stage 数量）要在性能和 occupancy 之间找平衡点。

**Latency Hiding 与流水线**：GPU 通过在等待显存加载时调度其他 warp 来隐藏延迟。理想情况是每个 warp 都不在等待数据——这要求足够多的活跃 warp 数量，或者异步加载指令。Ampere 的 `cp.async` 和 Hopper 的 `TMA` 把异步加载从“软件模拟”变成“硬件支持”，让流水线设计更简单且更高效。Marlin 的 4 级流水线正是利用 `cp.async` 实现的。

## 推理优化中的算子例子
理解了通用优化原则后，看几个推理栈里的具体例子。

**GEMM 通用**：矩阵乘法是 LLM 里最频繁的算子（每个 Transformer 层几十次）。FlashAttention 内部就有 3 次 QK^T、softmax、PV 矩阵乘。CUTLASS 提供了高度参数化的 GEMM 模板，Marlin、Machete 等专用内核都基于它定制。Llama-2-70B 的 prefill 时间中，约 60-70% 在 GEMM 上。

**RMSNorm / LayerNorm**：LLaMA 用 RMSNorm（比 LayerNorm 少一个减均值操作），kernel 设计上要做按行 reduce 并广播。优化点是 warp-level reduce（warp shuffle 指令），避免跨 warp 同步。FlashAttention 的并行 softmax 也用 warp shuffle 做行最大值和归一化因子的聚合。

**RoPE 位置编码**：旋转位置编码需要在 attention 之前对 Q、K 做原地修改。优化点是把它和 Q/K 的 reshape、transpose 融合到同一个 kernel 里，避免中间张量写回 HBM。vLLM 的 chunked prefill 实现里，RoPE 融合能省 1 次 HBM 读写。

**SiLU / GeLU 激活**：逐元素算子，算术强度接近 0，本身不耗时，但与周围算子分离时会浪费一次 HBM 读写。融合到 GEMM 的 epilogue 阶段（GEMM 输出后立即激活）是现代推理框架的标准做法——cuBLASLt、CUTLASS、Marlin 都支持。

**KV Cache 读写**：Decode 阶段每生成一个 token 都要读完整历史 KV Cache，是 Memory-Bound 的核心来源。PagedAttention 把 KV 分页存储到非连续显存块，配合 FA2 让分页加载与 attention 计算重叠。FP8 KV Cache 进一步减半读写量，代价是要在 attention kernel 入口做 FP8 → FP16 的转换。

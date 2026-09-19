---
title: Marlin
order: 25
---

# Marlin
Marlin（Mixed Auto-Regressive Linear）是 IST-DASLab 在 2024 年开源的 FP16 × INT4 矩阵乘法 CUDA 内核，专门为大语言模型的 4-bit 权重量化推理设计。它是目前 W4A16 推理的事实标准后端，被 vLLM、SGLang、TensorRT-LLM、LMDeploy 等主流框架深度集成，是消费级 NVIDIA 显卡（RTX 3090/4090）跑 70B 以下模型的关键加速组件。

## 诞生背景
W4A16（权重 4-bit、激活 16-bit）格式的优势非常明确——模型权重压缩到原来的 1/4，在 decode 这种 memory-bound 的场景下能直接带来接近 4 倍的吞吐加速。但早期的 INT4 内核（bitsandbytes、ExLlamaV1、AWQ 官方实现）在 batch size = 1~2 时确实接近理论加速比，batch 一旦增大到 8~16 就迅速退化——原因在于反量化开销与 GEMM 计算无法充分流水线化，整体吞吐被内存子系统卡住。

Marlin 的核心目标正是把"接近理想 4× 加速"的有效范围扩展到 batch size 16~32 甚至更高。它通过精心设计的异步内存访问、Tensor Core 深度融合的反量化、多级共享内存流水线、寄存器内高效 unpack，把全局显存、L2、SMEM、Tensor Core、向量单元全部榨干。论文 MARLIN: Mixed-Precision Auto-Regressive Parallel Inference on Large Language Models（2024）的实测数据显示，Marlin 在 A100 和 RTX 3090 上相对 FP16 基线实现了 2.8~3.9 倍的 decode 加速，batch 16 时仍能保持 80% 以上的理论加速比。

## 关键设计

### 异步内存拷贝
Marlin 最关键的硬件依赖是 Ampere 架构引入的 `cp.async`（`cuda::memcpy_async`）指令。该指令允许从全局内存（GMEM）直接异步拷贝数据到共享内存（SMEM），不占用寄存器和线程资源，且支持 L1 缓存旁路。Marlin 大量使用这种异步加载来流水线化权重读取，让计算与数据加载真正重叠，实现 latency hiding。

Turing 及更早架构没有这条指令，只能用同步加载或占用寄存器的方式模拟异步，无法实现 Marlin 设计的深度流水线。这是 Marlin 必须 Ampere 起跑的根本原因。

### 反量化与 GEMM 融合
Marlin 的第二个核心设计是把 INT4 → FP16 的反量化和矩阵乘法融合在同一个 kernel 里完成。传统做法是先把 INT4 权重反量化到 FP16 写回全局内存，再调用标准 GEMM——这一步额外的全局内存读写是性能杀手。Marlin 在 Tensor Core 计算之前直接在寄存器内完成 unpack 和 dequant：INT4 权重从共享内存加载到寄存器后，立即用 group-wise scale 还原成 FP16，然后送入 Tensor Core 的 `mma.sync` 指令。整个过程没有任何中间结果写回全局内存，权重只读一次。

这种融合带来的副作用是 group size 必须是 128 的倍数（实际最常用 128），因为 SMEM 的 bank conflict 优化和寄存器布局都围绕这个粒度设计。

### Striped 划分与流水线
Marlin 把矩阵乘法按输出维度切成多个 stripe，每个 stripe 由独立的 warp group 负责。Stripe 内部使用 4 级共享内存队列（4-stage pipeline），让当前 stripe 计算的同时预加载下一 stripe 的权重到 SMEM。这种 striped 划分让 L2 缓存命中率显著提升，因为相邻 stripe 共享大量激活值数据。

整个 kernel 的设计目标是让 Tensor Core 的计算流水线和 SMEM 的加载流水线同时饱和——任何一侧空闲都意味着性能损失。

### 支持 2:4 稀疏加速
Marlin 还有扩展版本 Sparse-Marlin，结合 NVIDIA Ampere 引入的 2:4 稀疏 Tensor Core（每 4 个权重中 2 个必须为零），可以在原 Marlin 基础上再额外加速约 1.2 倍。代价是模型权重必须预先按 2:4 模式稀疏化，且推理时需要保证非零权重位置固定——这对未经稀疏训练的稠密模型不适用，但对专门稀疏化过的模型（如某些 NVIDIA 优化版本）非常有效。

## 为什么必须 Ampere 起跑
Marlin 官方 README 明确要求 compute capability ≥ 8.0（Ampere 或 Ada）。这不是"能跑就行"的最低限制，而是"必须用这些新硬件能力把带宽和计算同时推到极限"的硬约束。

| 架构 | compute capability | Marlin 支持 | 关键差异 |
| ---- | ------------------ | ----------- | -------- |
| Turing | 7.5 | 不支持 | 没有 cp.async，SMEM 流水线能力不足 |
| Ampere | 8.0 / 8.6 | 原生支持 | cp.async + 增强 Tensor Core + 更大 L2 |
| Ada Lovelace | 8.9 | 完整支持 | 同 Ampere 能力，针对 40 系进一步调优 |
| Hopper | 9.0 | 部分/未完全优化 | 官方原版未专门优化，社区/vLLM 已扩展（被 Machete 替代） |

最关键的是 `cp.async`。其次是 Ampere Tensor Core 增强的 `mma.sync.aligned.m16n8k16` FP16 输入 + FP32 累加指令，Marlin 把 INT4 权重反量化到 Tensor Core 期望的寄存器布局后，能与这些 mma 指令完美对齐。再次是 Ampere 的 L2 缓存更大、更高效，Marlin 依赖把激活尽量留在 L2 来减少全局内存访问。最后是 Ampere 的共享内存容量和访问特性更适合 Marlin 的 4 级流水线（kernel 代码中直接写了 `SHARED_MEM = 96 * 1024`，针对 SM 8.x 优化）。

Turing 虽然也有 INT4 Tensor Core，但缺少异步拷贝这套核心能力，无法实现 Marlin 设计的高吞吐流水线。社区曾尝试移植到 Turing，最终性能远不及 Ampere 版本，官方放弃支持。

## 几个关键设计细节

### Group size 为什么是 128
Marlin 要求 group-wise scale 且粒度是 128 的倍数。Group size 是量化算法层的一个参数：每 128 个连续权重共用一个 scale 和 zero point。group size 越小（如 32），精度越高但 scale 表占用越多（每 128 个权重需要 4 个 scale 项）；group size 越大（如 256），scale 表压缩但量化误差累积。

Marlin 选 128 不是因为精度最优，而是因为**硬件流水线对齐**。原因有三。

第一，SMEM bank conflict 优化。Marlin 的 SMEM 布局按 128 权重 + 1 scale + 1 zp 组织，128 字节 = 32 个 float，正好是一个 warp 的访问粒度，避免 SMEM bank conflict。如果改成 64 或 256，要么 bank 冲突要么浪费 SMEM 容量。

第二，寄存器反量化粒度。Marlin 在寄存器内一次性 unpack 128 个 INT4 权重，用对应 scale 反量化成 128 个 FP16，正好填入 4 个 `mma.sync.m16n8k16` 指令的 K 维度（每个指令 K=16，4 个指令走 64，8 个指令走 128）。如果 group size 不是 128，寄存器布局需要重新设计，额外增加指令数。

第三，Tensor Core MMA 指令期望的 K 维度。`mma.sync.m16n8k16` 的 K=16，Marlin 走 8 条指令串起来走完 K=128。改 K 粒度会打破 Tensor Core 的最优指令序列。

AutoAWQ 和 AutoGPTQ 默认都用 group_size=128。如果用户手调成 64 或 256，Marlin 能跑但性能略低；如果改成 32，可能完全跑不起来（kernel 编译期硬编码为 128）。

### 反量化精度权衡
Marlin 把 INT4 权重反量化到 **FP16** 而不是 FP32——这看似反直觉（精度更高的累加器不是更好吗？）。实际上 FP16 已经足够，因为：

1. INT4 → FP16 的转换是简单的整数→浮点映射，不会产生舍入误差（FP16 能精确表示 INT4 的全部 16 个离散值）
2. 权重反量化后进入 Tensor Core 做矩阵乘，Tensor Core 内部使用 FP32 累加器（`mma.sync` 累加器是 FP32），精度无损
3. 如果中间结果是 FP32，反而需要再降回 FP16 才能送进下一层，损失一次精度

所以路径是 **INT4 → FP16 → Tensor Core (FP16 输入 + FP32 累加) → FP16 输出 → 下一层**。FP32 只在 Tensor Core 内部短暂存在，不写回 SMEM/HBM。

Blackwell NVFP4 走了相反的路径：**反量化完全在 Tensor Core 内部**完成，不需要在寄存器中先转 FP16 再喂给 MMA。`tcgen05.mma` 指令直接接受 NVFP4 权重 + 微块缩放因子，在 MMA 电路里实时反量化。这进一步压低了对寄存器容量的需求——Marlin 需要在寄存器里维护 FP16 副本，Blackwell NVFP4 不需要。

### 与 Blackwell NVFP4 的对比
| 维度 | Marlin (Ampere) | CUTLASS NVFP4 (Blackwell) |
| ---- | ---------------- | --------------------------- |
| 权重格式 | INT4 + 128 组 scale | NVFP4 + 微块 E4M3 scale |
| 反量化位置 | 寄存器内 | Tensor Core 内部 |
| 累加器 | FP32（`mma.sync` 内部） | FP32（`tcgen05.mma` 内部） |
| SMEM 占用 | 96KB（4 级流水线） | 类似（按 CUTLASS 配置） |
| 关键指令 | `cp.async` + `mma.sync.m16n8k16` | `tcgen05.mma.block_scale` + `TMA` |

两者的设计哲学一脉相承——避免反量化后的中间结果写回 HBM/SMEM——但 Blackwell 走得更远：直接让 Tensor Core 自己吃 FP4 权重。这反映了硬件从“通用的 FP16 Tensor Core”向“专用量化 Tensor Core”的演进趋势。

## 与其他内核的关系

### ExLlamaV2（EXL2 格式）
ExLlamaV2 是面向消费级单卡、单用户场景的高度优化引擎，核心目标是 decode 极致速度。它的 CUDA 内核（q_gemm 等）针对 batch = 1 的小 M 场景做了特殊优化，在单用户连续对话中往往比 Marlin 更快。但 prefill 时 batch 变大（M = 序列长度 × batch 很大），为小 M 设计的路径效率下降，Marlin 的优势就出来了。

EXL2 是 ExLlamaV2 引擎的私有量化格式（支持 2~8 bpw 混合比特），与 Marlin 期望的标准 W4A16 GPTQ/AWQ 权重布局完全不同——EXL2 的权重存储方式（包括混合比特、scale 布局、列重排）和 Marlin 不兼容，所以 EXL2 模型不能直接用 Marlin 跑，反之亦然。两者是独立的优化路径：EXL2 + ExLlamaV2 适合单用户极致速度，AWQ/GPTQ + Marlin 适合多用户服务（vLLM 的连续批处理场景）。

### Machete（Hopper 上的继任者）
Machete 是 vLLM 团队针对 Hopper 架构（SM 9.0）专门优化的 W4A16 内核，可以看作是 Marlin 在 Hopper 上的继任者。Hopper 引入了新的 wgmma（warp group MMA）指令和 TMA（Tensor Memory Accelerator），Marlin 的 Ampere 优化路径不能直接利用这些新硬件能力。Machete 重新设计了流水线以适配 wgmma + TMA，在 H100/H200 上的表现优于 Marlin。

如果你的硬件是 H100/H200，应该使用 Machete；如果是 RTX 3090/4090/A100，Marlin 仍是首选。

### CUTLASS W4A8（混合精度）
CUTLASS 在 Hopper 上还提供了 W4A8（INT4 权重 × FP8 激活）的混合精度 GEMM 实现，用于同时追求低显存（W4）和高计算吞吐（FP8 Tensor Core）。vLLM 已经集成了 `CutlassW4A8LinearKernel`，但要求 compute capability ≥ 90 且不支持 zero-point 和 act-order。消费级 40 系虽然有原生 FP8 Tensor Core，但 CUTLASS W4A8 是为 Hopper 量身定制的，移植到 Ada 性能反而不如原生 FP8（W8A8）或 Marlin。

## 实际部署选型

### Ampere（RTX 3090 / A100 / A10）
Marlin + W4A16（AWQ 或 GPTQ）是当前最优解。vLLM 中 AWQ/GPTQ 模型在 Ampere 上默认走 Marlin 后端（或 `awq_marlin` / `gptq_marlin`），无需手动配置。medium batch（4~16）时吞吐显著高于 ExLlamaV2 等其他内核。

启动示例：
```bash
vllm serve <awq-model> --quantization awq_marlin
vllm serve <gptq-model> --quantization gptq_marlin
```

### Ada Lovelace（RTX 4090 / L40S / L40）
两个推荐方案二选一：显存充足优先 FP8（W8A8，原生硬件支持，精度无损）；显存紧张选 Marlin 加速的 AWQ/GPTQ（W4A16，显存约为 FP8 的一半，decode 速度极快）。Ada 上 FP8 在 vLLM 中的优化成熟度仍略逊于 Hopper，但已经可用且效果不错。

启动示例：
```bash
vllm serve <model> --quantization fp8 --kv-cache-dtype fp8
```

### Hopper（H100 / H200）
首选 Machete 或原生 FP8，Marlin 已被 Machete 替代。FP8 在 Hopper 上是最成熟的方案，Machete 适合坚持 W4A16 的场景。

### Blackwell（B200 / RTX 5090）
首选 NVFP4（NVIDIA 专为 Blackwell FP4 Tensor Core 设计的高精度 4-bit 浮点格式，使用微块缩放 + E4M3 scale），精度接近 FP8 但速度更快。Marlin 在 Blackwell 上不是最优解。

## 工程实践要点
Marlin 在生产环境中实际跑起来后，工程层面的几个关键细节值得关注。

**模型格式兼容性**。Marlin 期望标准的 GPTQ/AWQ 权重布局，group size 128 最常见且最优。EXL2、GGUF 等私有格式无法直接使用。AutoAWQ、AutoGPTQ 工具链默认产出的就是 Marlin 兼容格式。

**Hopper 上慎用**。虽然 Marlin 也能在 Hopper 上跑，但官方原版没有针对 Hopper 的 wgmma + TMA 做专门优化，性能不如原生 FP8 或 Machete。如果硬件是 H100，强烈建议切到 FP8 或 Machete 路径。

**Prefill vs Decode 的偏好**。Marlin 在 decode（生成阶段）和小到中等 batch 的 prefill（提示处理）上都很强，但如果你的负载以超长上下文 prefill 为主（比如 RAG 灌入几万 token 的文档），可以考虑用 vLLM 的 chunked prefill 特性将 prefill 切成小块，与 decode 任务并发执行，进一步压榨硬件利用率。

**与 Continuous Batching 的协同**。vLLM 的连续批处理（continuous batching）天然适合 Marlin 的中等 batch 优势——多个请求的 decode 步骤被合并成同一个 batch，Marlin 在 batch 16~32 时仍能保持高加速比的特性正好被充分利用。这是 Marlin 在服务场景（vLLM）比 ExLlamaV2 更受欢迎的根本原因——ExLlamaV2 为单用户极致优化，反而在多用户合并 batch 的场景下退路较少。

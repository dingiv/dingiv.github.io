---
title: 量化选型
order: 26
---

# 量化方案选型矩阵
选一张 4-bit 量化模型跑推理时，最常被问的问题是"我这块卡到底该用哪个内核"。答案不是看 GPU 显存够不够，而是看 GPU 架构代际——Ampere 起才开始有原生的 INT4 Tensor Core，Ada 起才有原生 FP8 Tensor Core，Blackwell 起才有原生 FP4。每张卡的最优内核完全不同。

本文以"GPU 架构 × 量化方案"的二维矩阵为主线，给出生产环境落地的选型决策树。

## 位宽命名法
在进入选型之前，必须先澄清一套贯穿整个行业的命名规则：W、A、FP 三种前缀分别代表什么。

W 和 A 后跟数字描述位宽。W 是 Weight（权重）的首字母，A 是 Activation（激活值）的首字母。例如 W4A16 表示权重 4-bit、激活 16-bit；W8A8 表示权重 8-bit、激活 8-bit。这是"weight-only 量化"（只压缩权重、激活保持高精度）和"对等量化"（权重和激活同等压缩）的核心区分。

FP 后跟数字表示浮点格式。例如 FP8 是 8-bit 浮点数（常见 E4M3 / E5M2 两种变体），FP16 / BF16 是 16-bit 浮点。FP8 不指定位宽，只指定数字格式——它既可以用于权重（写成 W8A8-FP8，权重和激活都用 FP8），也可以独立存在。

两者可以组合。W4A8-FP8 表示权重 4-bit 整数、激活 8-bit 浮点；NVFP4 是 NVIDIA 专为 Blackwell 设计的 4-bit 浮点格式（用微块缩放 + E4M3 scale）。理解了这套命名规则，所有量化方案的"是什么"问题就回答了一半。

| 名称 | 类型 | 含义 | 典型硬件 |
| ---- | ---- | ---- | -------- |
| W4A16 | 量化方案 | 权重 INT4 + 激活 FP16/BF16 | Ampere+ |
| W8A8-FP8 | 量化方案 | 权重 FP8 + 激活 FP8 | Ada/Hopper/Blackwell 原生 |
| W4A8-FP8 | 量化方案 | 权重 INT4 + 激活 FP8 | Hopper（CUTLASS） |
| NVFP4 | 量化格式 | 4-bit 浮点 + 微块缩放 | Blackwell 原生 |

## 选型决策矩阵
把 GPU 架构代际和量化方案画成一张二维表，就是这张矩阵的全部内容。每一格代表"在该架构上跑该量化方案时，使用什么后端、性能如何、推荐度怎样"。

| GPU 架构 | 计算能力 | 最优内核 | 备选内核 | 不推荐内核 | 推荐格式 |
| -------- | -------- | -------- | -------- | ---------- | -------- |
| Ampere（RTX 3090 / A100） | sm_80 / sm_86 | Marlin | W8A8 INT8 | FP8（无硬件支持，会回退） | AWQ / GPTQ (W4A16) |
| Ada Lovelace（RTX 4090 / L40S） | sm_89 | FP8 原生 | Marlin | CUTLASS W4A8（为 Hopper 设计） | FP8 优先，AWQ/GPTQ 备选 |
| Hopper（H100 / H200） | sm_90 | FP8 原生 | Machete / CUTLASS W4A8 | Marlin（已被 Machete 替代） | FP8 优先，W4A8 备选 |
| Blackwell（B200 / RTX 5090） | sm_100+ | CUTLASS NVFP4 / DeepGEMM | FP8 | Marlin（性能次优） | NVFP4 优先 |

每一行的推荐内核都依赖该架构的某项硬件能力，盲目套用别的架构的最优解会得到非常差的结果。

### Ampere：Marlin 是绝对主力
Ampere 没有原生 FP8 Tensor Core——任何 FP8 模型在 Ampere 上都会回退到 W8A16 模式（FP8 权重用 Marlin FP8 内核反量化），损失原 FP8 的吞吐优势。Ampere 上 INT4 推理的最优内核只有 Marlin，其他备选（W8A8 INT8、ExLlamaV2）在中等 batch 的服务场景下吞吐显著低于 Marlin。

启动配置：vLLM 中 AWQ/GPTQ 模型默认走 Marlin 后端。

```bash
vllm serve <awq-model> --quantization awq_marlin
vllm serve <gptq-model> --quantization gptq_marlin
```

ExLlamaV2 在 batch=1 的单用户极致延迟场景下可能反超 Marlin，但一旦进入多用户合并 batch 就被 Marlin 拉开差距。Marlin 的关键技术细节（异步内存拷贝、Tensor Core 融合反量化）见 [Marlin](./marlin)。

### Ada Lovelace：FP8 原生优先，Marlin 备选
Ada 是首个消费级原生支持 FP8 Tensor Core 的架构（第四代）。FP8（W8A8）在精度几乎无损的前提下，吞吐和显存效率都比 Marlin 加速的 W4A16 更好——前提是显存够用。显存紧张时（4090 24G 跑 13B+ 模型）退回 Marlin + AWQ/GPTQ。

启动配置：

```bash
# FP8 优先
vllm serve <model> --quantization fp8 --kv-cache-dtype fp8

# 显存不够时退回 Marlin
vllm serve <awq-model> --quantization awq_marlin
```

Ada 上 FP8 的优化成熟度仍略逊于 Hopper，但已经可用且效果不错。Ada 没有原生 FP4，强行使用 NVFP4 只会模拟、没有硬件加速收益。

### Hopper：FP8 最成熟，Machete 替代 Marlin
Hopper 的 FP8 支持是当前数据中心部署的事实标准。第四代 Tensor Core + Transformer Engine + FA3 让 FP8 推理的性能和精度都达到生产级。如果坚持使用 W4A16，应该选 Machete（Marlin 在 Hopper 上的继任者，针对 wgmma + TMA 优化），而不是继续用 Marlin——Marlin 没有针对 Hopper 的新指令集做优化，在 H100 上性能远不如 Machete。

启动配置：

```bash
# FP8
vllm serve <model> --quantization fp8 --kv-cache-dtype fp8

# W4A16 走 Machete 而非 Marlin
vllm serve <awq-model> --quantization awq_marlin  # vLLM 会自动选 Machete
```

Hopper 还支持 CUTLASS W4A8（INT4 权重 + FP8 激活），用于同时追求低显存和高计算吞吐。`CutlassW4A8LinearKernel` 要求 compute capability ≥ 90 且不支持 zero-point 和 act-order。

### Blackwell：NVFP4 多源分化
Blackwell 第五代 Tensor Core 原生支持 FP4，NVFP4 是 NVIDIA 专为它设计的 4-bit 浮点格式——用微块缩放 + E4M3 scale，精度接近 FP8 但显存和速度比 INT4 还激进。这是当前生产部署的最前沿，B200 和 RTX 5090 都应该优先考虑 NVFP4。

启动配置：

```bash
vllm serve <model> --quantization nvfp4
```

与 W4A16 有 Marlin 这个社区主导的"杀手级内核"不同，NVFP4 目前由多个源头的算子拼接而成，还没有一个一统天下的名字。

**数据中心 B200 / SM100** 走 `tcgen05.mma` block-scaled 指令。CUTLASS 提供 `72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu` 参考实现，使用 Tensor Memory (TMEM) 存累加器、TMA 搬数据，输出可以是 FP32、BF16 或 FP8。DeepSeek-AI 出的 **DeepGEMM** 库走相同的 SM100 路径，专为 MoE/grouped-GEMM 场景优化，是当前 FP4 推理吞吐最高的选择之一（截至 2026 初仍 SM100-only，SM120 移植进行中）。

**消费级 RTX 5090 / SM120** 走 `mma.sync.aligned.block_scale` 指令，没有数据中心卡的 TMEM 和大规模 TMA 路径。CUTLASS 提供 `79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu` 参考实现，并配套 grouped-GEMM 变体（`79d`）给 MoE 用。TensorRT-Edge-LLM 仓库里有 CuTe DSL 写的 `blockscaled_contiguous_grouped_gemm_finalize_fusion.py`，是 SM120 上 MoE FP4 推理的高性能实现。

反量化在所有 Blackwell NVFP4 路径上都**发生在 Tensor Core 内部**，没有独立的"dequant kernel"——这与 Marlin 的设计一致（避免写回全局内存），但实现机制不同：Marlin 在寄存器里反量化，Blackwell 的 NVFP4 直接在 Tensor Core 的微块缩放电路里完成。吞吐量参考上，CUTLASS SM120 example 头部声明 NVFP4 比 MXFP8 MMA 快约 2 倍、比 Ada Tensor Core 快约 4 倍，但这个数字仅来自单一实现，尚未独立验证。

Marlin 在 Blackwell 上不是不能跑，但性能远不如 NVFP4——Blackwell 的 wgmma + TMA + FP4 Tensor Core 路径没有被 Marlin 利用起来。如果在 Blackwell 上继续用 Marlin + W4A16，相当于在 5nm 工艺的卡上跑 8nm 时代的优化思路，硬件潜力只发挥了一小部分。

## 为什么不能跨架构套用
很多用户在选型时会犯"看别人跑得好就直接抄"的错误。常见的反面教材：

**用 FP8 权重跑在 Ampere 上**——Ampere 没有原生 FP8 Tensor Core，FP8 权重会被自动反量化到 FP16 再做矩阵乘法，吞吐甚至不如直接用 FP16 + Marlin 的 W4A16 路径。FP8 模型在 Ampere 上不是"快"，而是"绕了一圈变慢"。

**用 Marlin 跑在 Hopper 上**——Marlin 是为 Ampere 的 cp.async 设计的，没有用到 Hopper 的 wgmma + TMA 新指令。同一个 W4A16 模型在 H100 上用 Marlin 跑，比用 Machete 跑慢 30%-50%。

**用 NVFP4 跑在 Ada 上**——Ada 没有原生 FP4 Tensor Core，NVFP4 会被软件模拟。损失精度的同时，吞吐也不如直接用原生 FP8。NVFP4 必须配 Blackwell。

每个量化方案的"原生硬件"是它在设计阶段就锚定的，跨架构使用就失去了硬件加速的根基。选型的第一步永远是先看 GPU 架构代际，第二步才是看模型有什么量化版本。

## 决策树简化
如果上面的矩阵看着累，可以按以下三步快速决策。

**第一步：确认 GPU 架构**。`nvidia-smi` 查看 GPU 型号，对照 NVIDIA 官方架构图确认是 Ampere / Ada / Hopper / Blackwell 中哪一个。如果是 Volta 或 Turing（A100 之前、RTX 20 系及更早），直接放弃跑现代大模型——BF16 和 FlashAttention-2 都缺失，强行跑只会得到 NaN 输出和长文本卡顿。

**第二步：在该架构的最优内核列里选**。Ampere → Marlin + W4A16；Ada → FP8（W8A8）优先；Hopper → FP8 或 Machete；Blackwell → NVFP4。如果该架构上没有原生的最优格式，退回到 Marlin 通用方案。

**第三步：根据显存调整**。如果最优格式的模型权重加上 KV Cache 超过显存，降到次优格式（通常是 W4A16 系）。如果是 batch=1 的单用户极致延迟场景，可以试试 ExLlamaV2 系（EXL2/EXL3 格式），它在 Ampere/Ada 上有小 M 场景的速度优势。

## 与其他量化内核的关系
这张矩阵之外，还有两个常被提及的内核值得单独说明。

EXL2（ExLlamaV2 专用）是 2~8 bit 混合比特的私有格式，权重存储和 Marlin 期望的 GPTQ/AWQ 布局完全不同——EXL2 模型不能直接用 Marlin 跑，反之亦然。EXL2 在 batch=1 的单用户场景下 decode 速度往往领先 Marlin，但在多用户合并 batch 的服务场景下退路较少。如果你的工作负载是个人本地推理，EXL2 + ExLlamaV2 是优秀选择；如果是多用户服务，AWQ/GPTQ + Marlin 更稳妥。

GGUF（llama.cpp 原生格式）是 CPU+GPU 混合推理的瑞士军刀——`ngl` 参数控制多少层放 GPU、多少层走 CPU 内存。GGUF 不属于本文讨论的 GPU 内核选型范畴，但在显存放不下完整模型、系统内存足够时是决定性能力。GGUF 与 Marlin/FP8 是正交关系——前者是部署架构选择，后者是 GPU 内核选择。

## 工程实践要点
**vLLM 默认会自动选最优后端**。AWQ/GPTQ 模型在 Ampere 上默认走 Marlin，在 Hopper 上默认走 Machete，在 Blackwell 上会优先 NVFP4。手动指定 `--quantization awq_marlin` 是为了显式锁定，避免框架版本升级时默认行为变化。

**Group size 选 128**。Marlin 的反量化流水线和 group-wise scale 都围绕 128 这个粒度优化。64 / 256 等其他值虽然能跑，但性能通常略低。

**KV Cache 量化是独立维度**。本文讨论的是权重量化，KV Cache 量化（FP8 KV Cache、TurboQuant 等）是独立的优化维度，与权重方案可以叠加使用——例如 FP8 权重 + FP8 KV Cache 是 Ada/Hopper 的常见生产配置。

**精度与吞吐的权衡**。W4A16 的精度损失通常在 1-3%（大模型更小），W8A8-FP8 精度几乎无损。如果业务对精度极其敏感（如金融、医疗），优先 FP8 而不是 W4A16；如果只是聊天/写作/代码辅助，W4A16 已经足够。

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
| Blackwell 数据中心（B200） | sm_100 | CUTLASS NVFP4 + DeepGEMM | FP8 | Marlin（性能次优） | NVFP4 优先 |
| Blackwell 消费级（RTX 5090） | sm_120 | CUTLASS NVFP4 SM120 + TensorRT-Edge-LLM | FP8 | Marlin（性能次优） | NVFP4 优先 |
| AMD RDNA3（RX 7900 XTX） | gfx1100 | vLLM Native W4A16 | HybridW4A16 | NVFP4（无原生） | AWQ / GPTQ (W4A16) |
| AMD CDNA4（MI355X） | gfx950 | NVFP4 → MXFP4 在线重量化 | MXFP6+MXFP4 混合 | NVFP4 原生（无硬件） | NVFP4 checkpoint / MXFP4 |
| AMD 跨平台 | 多代 | HybridW4A16 (Triton + HIP skinny) | — | — | AWQ 通用 fallback |
| Hopper（H100 / H200） | sm_90 | FP8 原生 | Machete / CUTLASS W4A8 | Marlin（已被 Machete 替代） | FP8 优先，W4A8 备选 |
| Ada Lovelace（RTX 4090 / L40S） | sm_89 | FP8 原生 | Marlin | CUTLASS W4A8（为 Hopper 设计） | FP8 优先，AWQ/GPTQ 备选 |
| Ampere（RTX 3090 / A100） | sm_80 / sm_86 | Marlin | W8A8 INT8 | FP8（无硬件支持，会回退） | AWQ / GPTQ (W4A16) |
| Turing（RTX 2080 Ti / T4） | sm_75 | 无现代内核，llama.cpp Q4_K_M | bitsandbytes INT8 / vLLM Marlin SM75 fork | FP8（无硬件支持） | Q4_K_M / IQ4_XS |
| Volta（V100） | sm_70 | 无现代内核，llama.cpp Q4_K_M 唯一可靠路径 | 1Cat-vLLM fork (AWQ) / vllm-fp8-w8a16-sm70 | NVFP4（fallback 到 12% 带宽） | Q4_K_M |

每一行的推荐内核都依赖该架构的某项硬件能力，盲目套用别的架构的最优解会得到非常差的结果。

### Ampere：Marlin 是绝对主力
Ampere 没有原生 FP8 Tensor Core——任何 FP8 模型在 Ampere 上都会回退到 W8A16 模式（FP8 权重用 Marlin FP8 内核反量化），损失原 FP8 的吞吐优势。Ampere 上 INT4 推理的最优内核只有 Marlin，其他备选（W8A8 INT8、ExLlamaV2）在中等 batch 的服务场景下吞吐显著低于 Marlin。

启动配置：vLLM 中 AWQ/GPTQ 模型默认走 Marlin 后端。

```bash
vllm serve <awq-model> --quantization awq_marlin
vllm serve <gptq-model> --quantization gptq_marlin
```

ExLlamaV2 在 batch=1 的单用户极致延迟场景下可能反超 Marlin，但一旦进入多用户合并 batch 就被 Marlin 拉开差距。Marlin 的关键技术细节（异步内存拷贝、Tensor Core 融合反量化）见 [Marlin](./marlin)。

### Ada Lovelace：Marlin 和 FP8 原生
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

与 W4A16 有 Marlin 这个社区主导的“杀手级内核”不同，NVFP4 目前由多个源头的算子拼接而成，还没有一个一统天下的名字。SM100（数据中心）和 SM120（消费级）走完全不同的指令路径，两者不能混用。

#### SM100（数据中心 B200）路径
数据中心 Blackwell B200 / B100 / GB200 走 `tcgen05.mma` block-scaled 指令，这是为数据中心推理全新设计的 5 代 Tensor Core 指令。

指令特性：
- `tcgen05.mma` 接受 block-scaled 输入（权重 + 微块缩放因子同时送入）
- 累加器存在专门的 TMEM（Tensor Memory），不是传统寄存器或 SMEM——带宽更高、压力更小
- TMA（Tensor Memory Accelerator）从 Hopper 传承下来，负责大规模异步数据搬运
- 输出可以是 FP32 / BF16 / FP8，下游 GEMM 可以接任何精度

CUTLASS 参考实现：`examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu`。这是 NVIDIA 官方给出的性能基准点。

DeepGEMM（DeepSeek-AI 出品）走相同的 SM100 路径，专为 MoE / grouped-GEMM 场景优化，是当前 FP4 推理吞吐最高的选择之一。截至 2026 初仍 SM100-only，SM120 移植正在进行中。如果在 B200 上跑 MoE（如 DeepSeek-V3 / Qwen-MoE），DeepGEMM 是首选。

#### SM120（消费级 RTX 5090）路径
消费级 RTX 5090 / RTX 50 系移动版 走 `mma.sync.aligned.block_scale` 指令——名字里还能看到 `mma.sync`，说明这是从 Ampere 传承下来的指令家族，不是数据中心卡的全新 `tcgen05`。这是因为消费级 Blackwell 缺少 TMEM 和大规模 TMA 路径，本质上是“带着 FP4 Tensor Core 的 Ampere”。

指令差异：
- 没有 TMEM——累加器还是回到寄存器 / SMEM
- 没有完整的 TMA 异步搬运能力，数据搬运走传统 `cp.async`
- `mma.sync.aligned.block_scale` 是 SM 8.0+ 都能编译的指令家族（不一定只在 SM120），但 FP4 Tensor Core 仅 SM120+ 有

CUTLASS 参考实现：`examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu`（单 GEMM）+ `79d_blackwell_geforce_nvfp4_grouped_gemm.cu`（MoE 用 grouped 变体）。注意文件名里的 `bf16`——SM120 NVFP4 的反量化目标默认是 BF16，不是 NVFP4↔NVFP4（SM100 那个路径）。

TensorRT-Edge-LLM 仓库里有 CuTe DSL 写的 `blockscaled_contiguous_grouped_gemm_finalize_fusion.py`，专门给 SM120 上的 MoE FP4 推理用。这是当前消费级 Blackwell 上 MoE FP4 的最高性能实现。

吞吐量参考（仅来自 CUTLASS example 头部声明，未独立验证）：
- NVFP4 比 MXFP8 MMA 快 ~2 倍
- NVFP4 比 Ada Tensor Core 快 ~4 倍
- 这些数字仅适用于 SM120 上的 FP4 路径

#### 两路径对比
| 维度 | SM100 (B200) | SM120 (RTX 5090) |
| ---- | ------------ | ---------------- |
| MMA 指令 | `tcgen05.mma`（全新 5 代） | `mma.sync.aligned.block_scale`（Ampera 家族延续） |
| 累加器 | TMEM | 寄存器 + SMEM |
| 数据搬运 | 完整 TMA | `cp.async`（传统异步加载） |
| CUTLASS 路径 | `72_blackwell_narrow_precision_gemm` | `79_blackwell_geforce_gemm` |
| DeepGEMM 支持 | ✅ SM100-only 主力 | ❌ 移植进行中 |
| TensorRT-Edge-LLM | ⚠️ 主供 SM120 | ✅ CuTe DSL 主供 SM120 MoE |
| 反量化输出 | FP32 / BF16 / FP8 | 默认 BF16 |

反量化在所有 Blackwell NVFP4 路径上都发生在 Tensor Core 内部，没有独立的“dequant kernel”——这与 Marlin 的设计一致（避免写回全局内存），但实现机制不同：Marlin 在寄存器里反量化，Blackwell 的 NVFP4 直接在 Tensor Core 的微块缩放电路里完成。

Marlin 在 Blackwell 上不是不能跑，但性能远不如 NVFP4——Blackwell 的 wgmma + TMA + FP4 Tensor Core 路径没有被 Marlin 利用起来。如果在 Blackwell 上继续用 Marlin + W4A16，相当于在 5nm 工艺的卡上跑 8nm 时代的优化思路，硬件潜力只发挥了一小部分。

### Volta 与 Turing：老卡的现实处境
Volta（V100，sm_70）和 Turing（RTX 20 系、T4，sm_75）在 2026 年这个时间点跑 LLM 没有任何“高效”算子可选。这不是缺工具，而是硬件能力断档太多：缺 BF16 硬件支持、缺 FlashAttention-2（要 sm_80+）、缺 `cp.async`（Marlin 类优化都跑不了）。但仍有几个降级方案能“勉强跑起来”。

Turing (sm_75):
+ llama.cpp + Q4_K_M / IQ4_XS | 上游支持 | 不走 Tensor Core 加速，CUDA 后端走 ggml 通用 GEMM |
+ bitsandbytes LLM.int8() (INT8) | 上游支持 | INT8 路径，不是 INT4 |
+ vLLM Marlin SM75 后端 | 实验性 fork（PR #29901） | 没有 `cp.async`，用同步 GMEM→SMEM；没有 m16n8k16，要 m16n8k8 串两次凑齐 K=16 |
+ fused-int4-gemm-sm75（独立项目） | 第三方 | 手写 PTX INT4 GEMM，用 WMMA + mma.sync.aligned，已验证 Qwen2 可跑 |

Volta (sm_70):
+ llama.cpp + Q4_K_M | 上游支持 | Volta 没有 INT4 Tensor Core，CUDA 后端走初代 FP16 Tensor Core |
+ 1Cat-vLLM fork | 实验性 | 集成 lmdeploy TurboMind SM70 WMMA 内核，能跑 AWQ 4-bit |
+ vllm-fp8-w8a16-sm70 | 第三方 fork | 跑 FP8 W8A16，反量化权重到 FP16；改善 decode 但不是原生加速 |
+ NVFP4 | ⚠️ 严重回退 | V100 上 NVFP4 自动 fallback 到 Marlin dequant，仅达显存带宽的 12% |
+ vLLM 上游 pip wheels | ⚠️ 已放弃 | 需要从源码编译（CUDA 12.6）才能跑 vLLM |

实际推荐：两张老卡的唯一可靠路径都是 llama.cpp + Q4_K_M——不依赖 Tensor Core、社区维护最稳、跨平台。B200 路上重起炉灶的 Marlin SM75 fork 、vllm-fp8-w8a16-sm70 、fused-int4-gemm-sm75 都处于“能跑但性能不惊人”的状态，不要指望它们能接近主流 30/40 系的体验。T4 上一个 benchmark 报告 INT4 量化在 ~4000 tok/s 平台，FP16 + 投机解码反而能跑到 9000 tok/s——在某些负载下量化不是吞吐赢家。

跳过所有现代内核（FA2、Marlin、NVFP4、FP8）后，长上下文和 BF16 模型会非常慢。建议 Turing 跑 Q4_K_M 4-bit 量化模型（避开 BF16 fallback），V100 跑 Q4_K_M 并接受它的吞吐——这是老卡唯一现实的部署方案。

### AMD：三条路线分开走
AMD 的量化生态不如 NVIDIA 成熟，但 2026 年已经形成三条可用的内核路径，分别面向消费级 RDNA、数据中心 CDNA、以及跨平台量化 checkpoint。三者走的不是同一套指令。

#### 消费级 RDNA3（RX 7900 XTX / W7900）
RDNA3 没有原生的 INT4 Tensor Core（AMD 的 Matrix Core 支持 FP16 / BF16 / INT8，不包含 INT4），但社区在 vLLM 里逐步补上了 W4A16 的高效实现。

Native W4A16 kernel for RDNA3（vLLM PR #41394）：在 gfx1100（RX 7900 XTX）上跑 GPTQ-W4A16-G32 模型，官方 benchmark 报告在所有测试的并发度上都超过现有 ROCm 选项（包括 ExLlama 的 ROCm 后端），且原生支持 BF16——ExLlama 此前不支持 BF16。测试用 Qwen3.6-27B-GPTQ-W4A16-G32 在双卡 RX 7900 XTX、并发 100 下进行。

启动配置（vLLM）：
```bash
vllm serve <awq-or-gptq-model> --quantization awq  # 自动选 RDNA3 native W4A16
```

注意：RDNA4（RX 9070 系）这条路径尚未在社区调查中明确报道，但与 RDNA3 架构同源，预计类似实现能顺接。

#### 数据中心 CDNA（MI300X / MI355X）
数据中心 Instinct 卡走 CDNA 架构，有原生 FP8、MXFP4、MXFP6 硬件支持，但不直接支持 NVFP4——这是与 NVIDIA Blackwell 的关键差异点。

MXFP4 与 MXFP6：AMD 在 MI350X / MI355X 上原生支持 MXFP4（OCP Microscaling 4-bit 浮点）。MXFP6 激活 + MXFP4 权重（W_MXFP4_A_MXFP6）的混合精度在 Llama-3.1-8B 和 Qwen3.6-2x 上报告能恢复部分 MXFP4 损失的精度，且吞吐仅比纯 MXFP4 慢 2-3%。

NVFP4 on AMD：NVFP4 是 NVIDIA 专有格式，AMD 在 MI350X/MI355X 上原生只能算 MXFP4，不能直接跑 NVFP4。AMD 的解决方案是在线重量化（NVFP4 → MXFP4）——加载 NVFP4 checkpoint 时自动转成 MXFP4 跑，不需预转换。这让 AMD 可以消费 NVIDIA 生态的现成 NVFP4 模型（HF 上 amd/DeepSeek-V4-Flash-NVFP4、amd/Qwen3.5-397B-A17B-NVFP4 都是 AMD 发布的 NVFP4 checkpoint），但损失一点吞吐。

W4A16 GEMM（vLLM Issue #34008 / PR #42640）：早期 vLLM 在 AMD 上缺乏高性能 W4A16 GEMM 和 grouped GEMM，后续 CDNA 调优版本补齐。

启动配置：
```bash
# MI355X 上跑 NVFP4 checkpoint（自动 requant 到 MXFP4）
vllm serve amd/DeepSeek-V4-Flash-NVFP4 --quantization nvfp4
```

#### Hybrid W4A16（ROCm 跨架构）
vLLM PR #40977 推出了 HybridW4A16LinearKernel，这个路径不针对特定架构，而是把两种实现拼起来：

- 小 batch（M ≤ 5，单 token decode）：走 HIP 手写的 `wvSplitK_int4_g` skinny decode 路径，专门优化小矩阵乘
- 大 batch（M > 5，prefill 和多 token decode）：走 Triton fused dequant GEMM

两条路径共用同一份 AWQ 风格打包的 `[N, K//8]` 权重张量，不需要双份存储。这是 ROCm 生态里一个比较通用的 W4A16 优化范式——它不依赖特定硬件（不论 CDNA 还是 RDNA 都能跑），但跨架构性能不是每张卡都是最优。

#### 实际推荐
| AMD 硬件 | 推荐路径 | 启动参数 |
| -------- | -------- | -------- |
| RX 7900 XTX / W7900 (gfx1100) | vLLM native W4A16 (PR #41394) | `--quantization awq` 或 `gptq` |
| MI300X (CDNA3) | FP8 原生 / W4A16 CDNA-tuned | 按模型选 |
| MI355X / MI350X (CDNA4) | NVFP4 → MXFP4 在线重量化 | `--quantization nvfp4` |
| MI355X 质量优先 | MXFP6 激活 + MXFP4 权重混合 | 依赖现成 quantized checkpoint |
| AMD 全平台 | HybridW4A16（PR #40977） | 通用 fallback |

关键限制：AMD 没有 Marlin 这样的“社区一统”内核。生态仍处于“多个 PR 多条路径”的阶段，上游 vLLM 主线在 AMD 上的优化成熟度还低于 NVIDIA 1-2 年。如果你选了 AMD 卡（特别是消费级），就要准备踩踩、跑跑社区 fork。

## 为什么不能跨架构套用
很多用户在选型时会犯"看别人跑得好就直接抄"的错误。常见的反面教材：

用 FP8 权重跑在 Ampere 上——Ampere 没有原生 FP8 Tensor Core，FP8 权重会被自动反量化到 FP16 再做矩阵乘法，吞吐甚至不如直接用 FP16 + Marlin 的 W4A16 路径。FP8 模型在 Ampere 上不是"快"，而是"绕了一圈变慢"。

用 Marlin 跑在 Hopper 上——Marlin 是为 Ampere 的 cp.async 设计的，没有用到 Hopper 的 wgmma + TMA 新指令。同一个 W4A16 模型在 H100 上用 Marlin 跑，比用 Machete 跑慢 30%-50%。

用 NVFP4 跑在 Ada 上——Ada 没有原生 FP4 Tensor Core，NVFP4 会被软件模拟。损失精度的同时，吞吐也不如直接用原生 FP8。NVFP4 必须配 Blackwell。

每个量化方案的"原生硬件"是它在设计阶段就锚定的，跨架构使用就失去了硬件加速的根基。选型的第一步永远是先看 GPU 架构代际，第二步才是看模型有什么量化版本。

## 决策树简化
如果上面的矩阵看着累，可以按以下三步快速决策。

第一步：确认 GPU 架构。`nvidia-smi` 查看 GPU 型号，对照 NVIDIA 官方架构图确认是 Ampere / Ada / Hopper / Blackwell 中哪一个。如果是 Volta 或 Turing（A100 之前、RTX 20 系及更早），直接放弃跑现代大模型——BF16 和 FlashAttention-2 都缺失，强行跑只会得到 NaN 输出和长文本卡顿。

第二步：在该架构的最优内核列里选。Ampere → Marlin + W4A16；Ada → FP8（W8A8）优先；Hopper → FP8 或 Machete；Blackwell → NVFP4。如果该架构上没有原生的最优格式，退回到 Marlin 通用方案。

第三步：根据显存调整。如果最优格式的模型权重加上 KV Cache 超过显存，降到次优格式（通常是 W4A16 系）。如果是 batch=1 的单用户极致延迟场景，可以试试 ExLlamaV2 系（EXL2/EXL3 格式），它在 Ampere/Ada 上有小 M 场景的速度优势。

## 与其他量化内核的关系
这张矩阵之外，还有两个常被提及的内核值得单独说明。

EXL2（ExLlamaV2 专用）是 2~8 bit 混合比特的私有格式，权重存储和 Marlin 期望的 GPTQ/AWQ 布局完全不同——EXL2 模型不能直接用 Marlin 跑，反之亦然。EXL2 在 batch=1 的单用户场景下 decode 速度往往领先 Marlin，但在多用户合并 batch 的服务场景下退路较少。如果你的工作负载是个人本地推理，EXL2 + ExLlamaV2 是优秀选择；如果是多用户服务，AWQ/GPTQ + Marlin 更稳妥。

GGUF（llama.cpp 原生格式）是 CPU+GPU 混合推理的瑞士军刀——`ngl` 参数控制多少层放 GPU、多少层走 CPU 内存。GGUF 不属于本文讨论的 GPU 内核选型范畴，但在显存放不下完整模型、系统内存足够时是决定性能力。GGUF 与 Marlin/FP8 是正交关系——前者是部署架构选择，后者是 GPU 内核选择。

## 工程实践要点
vLLM 默认会自动选最优后端。AWQ/GPTQ 模型在 Ampere 上默认走 Marlin，在 Hopper 上默认走 Machete，在 Blackwell 上会优先 NVFP4。手动指定 `--quantization awq_marlin` 是为了显式锁定，避免框架版本升级时默认行为变化。

Group size 选 128。Marlin 的反量化流水线和 group-wise scale 都围绕 128 这个粒度优化。64 / 256 等其他值虽然能跑，但性能通常略低。

KV Cache 量化是独立维度。本文讨论的是权重量化，KV Cache 量化（FP8 KV Cache、TurboQuant 等）是独立的优化维度，与权重方案可以叠加使用——例如 FP8 权重 + FP8 KV Cache 是 Ada/Hopper 的常见生产配置。

精度与吞吐的权衡。W4A16 的精度损失通常在 1-3%（大模型更小），W8A8-FP8 精度几乎无损。如果业务对精度极其敏感（如金融、医疗），优先 FP8 而不是 W4A16；如果只是聊天/写作/代码辅助，W4A16 已经足够。

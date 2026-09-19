---
title: GGUF
order: 45
---

# GGUF 与 K-quant
GGUF 是 llama.cpp 生态的模型存储格式，K-quant / I-quant 是它对应的**量化方案家族**。两者绑定极深——你下载的 GGUF 文件名后缀（`Q4_K_M`、`IQ3_XXS`）就是这个量化方案的标识。理解 GGUF 不只是理解一个文件格式，更是理解 llama.cpp 整个本地推理生态。

本文拆三个层面：**GGUF 文件格式本身**（存什么、怎么组织）、**K-quant 量化方案**（怎么压）、**llama.cpp 推理引擎**（怎么用）。最后与 Marlin/FP8/NVFP4 路径对比，给出选型边界。

## GGUF 文件格式

### 文件名约定
```
<model-name>.<quant-type>.gguf
```

quant-type 直接告诉你这个文件用了什么量化——这是 llama.cpp 社区的命名约定，HF Hub 上的 GGUF 模型普遍遵守：

```
Qwen2.5-7B-Instruct-Q4_K_M.gguf       # K-quant 中等质量
llama-3.1-70b-instruct-q5_k_m.gguf    # 同上
Mistral-7B-Instruct-v0.3-IQ3_M.gguf   # I-quant 中等
deepseek-v2-lite-iq2_xxs.gguf         # I-quant 极限压缩
```

后缀的语义在下文"K-quant / I-quant 类型"节展开。

### 文件结构
GGUF 是二进制格式，包含模型权重、词汇表、量化参数、元数据。文件头是键值对数组，每条 KV 记录一段元信息（模型名、版本、tokenizer 类型、张量数量等），后面紧跟各个张量块。关键特性是 **mmap 友好**——文件布局允许操作系统直接将磁盘页映射到虚拟内存，推理时不复制数据，模型大小超过物理内存时由操作系统自动分页换入换出。

```c
// GGUF 文件结构（简化）
+--------------------+
| Magic: "GGUF"      |  4 bytes
+--------------------+
| Version            |  4 bytes (uint32, 当前 3)
+--------------------+
| Tensor count       |  8 bytes (uint64)
+--------------------+
| Metadata KV count  |  8 bytes (uint64)
+--------------------+
| Metadata KV pairs  |  变长
|   - key (string)   |
|   - value type     |
|   - value data     |
+--------------------+
| Tensor infos       |  每个张量：名称、维度、量化类型、offset
+--------------------+
| Padding            |
+--------------------+
| Tensor data        |  实际权重，按张量顺序排列
+--------------------+
```

每个张量块独立量化、自带 scale/zero-point 等参数——这与 W4A16 GPTQ/AWQ 的"全局 group_size=128 + per-channel scale"不同，GGUF 允许每个张量用不同量化策略（这就是 K-quant 名字的来源）。

### 单文件分发
GGUF 一个文件包含**全部**部署所需：模型权重、tokenizer、chat template、特殊 token、量化参数。这意味着：

- 分发方便——一个 `.gguf` 文件就能跑，无需配套 `config.json` / `tokenizer.json`
- 离线友好——加载后不依赖任何外部资源
- 跨平台——同一文件能在 Apple Silicon、x86 CPU、NVIDIA GPU、AMD GPU 上跑
- 代价是灵活性低——chat template 改了需要重新量化或事后修改

## K-quant 量化方案
K-quant 是 llama.cpp 在 2023 年推出的量化方案家族，命名空间是 `Q{bit}_K_{size}`。

### 设计动机
早期的 `Q4_0` / `Q4_1` / `Q5_0` / `Q5_1` 是"统一 4-bit"或"统一 5-bit"——所有张量用同一种位宽、所有通道用同一种 scale。这种简单方案的 PPL 退化明显，特别是 attention 层的 Q/K 投影对量化敏感。K-quant 的核心改进：**按张量重要性分配不同位宽**。

具体规则：attention 的 `q_proj` / `k_proj` / `v_proj` 用更高位宽（如 6-bit），attention output 和 MLP 的 down_proj 用中等位宽（4-bit），其他张量用更低位宽（如 4-bit 或更低）。这种"层内混合精度"在不增加平均位宽的前提下大幅降低 PPL 退化。

### 类型详解
| 后缀 | 平均 bpw | 张量分配 | 适用场景 |
| ---- | -------- | -------- | -------- |
| `Q2_K` | ~3.4 | 大部分 2-bit，少量 3-4 bit | 极限压缩，7B 模型仅 ~3GB |
| `Q3_K_S/M/L` | ~3.3-3.7 | 3-bit 为主，敏感层 4-6 bit | 显存极紧张 |
| `Q4_K_S` | ~4.1 | 4-bit 为主，敏感层 5-6 bit | 平衡选项 |
| `Q4_K_M` | ~4.6 | 4-bit + 5-6 bit 混合 | **最常用的甜点选项** |
| `Q5_K_S/M` | ~5.3-5.7 | 5-bit 为主，敏感层 6-bit | 高质量，体积略大 |
| `Q6_K` | ~6.6 | 接近全 6-bit | 几乎无损 |
| `Q8_0` | ~8.0 | 全 8-bit | 接近 FP16，调试用 |

后缀字母含义：
- `_S` = Small（小张量化，位宽偏低，体积小但质量略降）
- `_M` = Medium（平衡，**默认推荐**）
- `_L` = Large（大张量化，位宽偏高，质量好但体积大）

`_M` 通常是同 bpw 等级里**质量 / 体积比最优**的选择——这就是为什么社区里 `Q4_K_M` 和 `Q5_K_M` 是两个最常被推荐的后缀。

### 反量化方式
K-quant 的反量化在 llama.cpp 内核中完成——和 Marlin 不同，**llama.cpp 的量化内核不依赖 Tensor Core**，主要走 SIMD 指令（AVX2 / AVX-512 / NEON / Apple AMX）和手写矩阵乘。这意味着：

- CPU 推理性能极强（因为用 CPU 最擅长的 SIMD）
- GPU 加速版本依赖 `ggml` 的 CUDA backend，但不是 Marlin 那种极致优化
- 量化与反量化路径独立——一个 K-quant 文件能在任何后端跑，无需 GPU 特定编译

代价是 GPU 上的 K-quant 推理速度远不及 Marlin/FP8。Marlin 的 W4A16 INT4 路径在 A100 上可达 ~3000 token/s（70B 模型），K-quant 同等模型可能只有 ~500 token/s——但 K-quant 不需要 NVIDIA GPU。

## I-quant 与 importance matrix
I-quant（Importance Quantization）是 2024 年推出的 K-quant 进化版，命名空间是 `IQ{bit}_{size}`。它的核心改进是引入 **importance matrix (imatrix)**——一份预计算的权重重要性表，告诉量化器哪些权重"不能动"。

### imatrix 工作原理
```bash
# 第一步：在 BF16 模型上跑一遍校准数据，统计每个权重通道的重要性
llama-imatrix -m model-bf16.gguf -f calibration-data.txt -o imatrix.dat

# 第二步：用 imatrix 指导量化，让重要权重保留更高精度
llama-quantize -m model-bf16.gguf \
    --imatrix imatrix.dat \
    -o model-iq4_xs.gguf IQ4_XS
```

imatrix 本质上是与权重矩阵同形状的浮点数矩阵，记录每个权重通道对最终 loss 的敏感度。量化时按敏感度排序，敏感通道分配更多比特，钝感通道可以压到 2-3 bit 甚至更低。

这与 AWQ 的"per-channel scaling"思路相似（保护大激活值通道），但 imatrix 更细粒度——保护到**单个权重**而非"整通道"，且是基于 loss 敏感度而非激活值大小。

### I-quant 类型
| 后缀 | bpw 范围 | 相对 Q_K 的质量 |
| ---- | -------- | --------------- |
| `IQ1_S` / `IQ1_M` | 1.5-2 | 实验性，质量下降明显 |
| `IQ2_XXS` / `IQ2_XS` / `IQ2_S` / `IQ2_M` | 2-3 | 比 `Q2_K` 质量好 |
| `IQ3_XXS` / `IQ3_XS` / `IQ3_S` / `IQ3_M` | 3-4 | 比 `Q3_K` 质量好 |
| `IQ4_XS` / `IQ4_NL` | 4-5 | 与 `Q4_K` 相当 |
| `IQ5_XXS` / `IQ5_XS` / `IQ5_M` | 5-6 | 接近 `Q5_K` / `Q6_K` |

**I-quant 的核心优势在极低比特场景**。当目标 bpw 降到 2-3 时，传统 K-quant 质量崩塌（Q2_K 的 PPL 比 FP16 高 3-5 点），I-quant 因为有 imatrix 指导，能在 2-3 bpw 下保持接近 4 bpw 的质量——这正是"2-bit 跑 70B"成为可能的原因。

但 I-quant 在 4 bpw 及以上时优势不明显，与 K-quant 几乎打平。社区共识是：

- **3 bpw 及以下**：选 I-quant
- **4-5 bpw**：K-quant 与 I-quant 几乎等价
- **6 bpw 及以上**：Q6_K 或 Q8_0 更直接

### I-quant 的限制
I-quant 内核使用查找表（LUT）实现反量化，比 K-quant 复杂：

- 计算开销略高（每权重一次 LUT 查表）
- imatrix 必须与目标模型匹配——换一个模型需要重新计算
- 某些 I-quant 变体（特别是 IQ1_*）仍处于实验阶段，质量不稳定
- 主线 llama.cpp 与 ik_llama.cpp（社区 fork）的 I-quant 支持存在差异，部分变体只在 fork 中可用

## llama.cpp 推理引擎
GGUF 文件需要 llama.cpp（或其衍生工具）才能加载推理。引擎本身有几个独有特性。

### CPU+GPU 混合推理
`ngl` 参数（number of GPU layers）控制多少 Transformer 层放 GPU：

```bash
# 全部层放 GPU（-ngl 999 表示尽可能多）
./llama-cli -m qwen3.6-32b-Q4_K_M.gguf -ngl 999 --ctx-size 8192

# 只放 20 层在 GPU，其余走 CPU（适合显存不够的场景）
./llama-cli -m qwen3.6-32b-Q4_K_M.gguf -ngl 20 --ctx-size 4096

# 纯 CPU 推理（Apple Silicon / 无独显）
./llama-cli -m qwen3.6-32b-Q4_K_M.gguf -ngl 0
```

这是 GGUF 最大的杀手锏——**显存放不下完整模型时仍能跑**。一个 24G 显存的卡装不下 70B Q4_K_M（约 40GB），但可以装 30 层到 GPU，剩余 30 层走 CPU 内存。代价是 CPU 层的推理速度比 GPU 慢 2-5 倍，但至少能把模型跑起来——这对消费级硬件跑大模型是决定性能力。

### 多 GPU 切分与投机解码
llama.cpp 原生支持多种跨卡策略：

- **层切分（PP）**：把不同层分配到不同卡——`--tensor-split` 或 `-ts` 参数指定每张卡的比例
- **GPU+CPU 混合**：层在卡间 PP + 剩余层走 CPU
- **投机解码（speculative decoding）**：挂载 draft model（`-md`）加速 decode——draft 用小模型（如 0.5B），target 用大模型，每次 target 验证 draft 的多个预测

这些能力让 llama.cpp 成为消费级单卡 / 多卡混部环境的瑞士军刀。

### 跨平台一致性
同一份 GGUF 文件可以在以下平台运行：

- Apple Silicon（M1/M2/M3/M4）— Metal 后端
- x86 CPU（AVX2 / AVX-512）
- ARM CPU（NEON）
- NVIDIA GPU（CUDA）
- AMD GPU（ROCm / HIP）
- Intel GPU（SYCL）

这是因为 llama.cpp 用 C/C++ 写核心，量化/反量化路径完全自包含，**没有依赖任何特定硬件的指令集**。代价是性能——在 NVIDIA GPU 上不如 Marlin/FP8，但跨平台灵活性无可替代。

## 与 GPU 内核路径的对比
| 维度 | GGUF + llama.cpp | W4A16 + Marlin / FP8 |
| ---- | ---------------- | -------------------- |
| 量化算法 | K-quant / I-quant（混合比特 + imatrix） | GPTQ / AWQ（统一 4-bit） |
| 主要硬件 | CPU + Apple Silicon + 任意 GPU | NVIDIA GPU（sm_80+） |
| GPU 性能 | 中等（手写 CUDA ggml 后端） | 极高（Tensor Core + 流水线） |
| 模型大小 | 同等 bpw 下与 W4A16 接近 | 同 |
| 跨平台 | 一份文件跨所有平台 | 单一平台，需重新量化 |
| 显存放不下时 | 仍能跑（部分层放 CPU） | 完全无法加载 |
| 部署工具链 | llama.cpp / Ollama / LM Studio | vLLM / SGLang / TGI |

**两个生态解决不同问题**：

- GGUF + llama.cpp 是 **CPU+GPU 异构 + 跨平台 + 极限压缩** 的最优解
- W4A16 + Marlin 是 **NVIDIA GPU 服务端高性能** 的最优解

你看到 HF Hub 上**每个主流模型都有 GGUF 版本**——这是因为 GGUF 是消费级本地部署的事实标准。一个普通用户在 MacBook / 普通 PC 上跑模型，几乎只能选 GGUF 路径。

## 选型建议
**用 GGUF 的场景**：
- Apple Silicon / 无独显 PC 上跑模型
- 显存放不下完整模型，需要 CPU offload
- 想在多个平台（Mac / Windows / Linux）用同一个模型文件
- 想要极限压缩（2-3 bpw 跑 70B 模型）
- 想要本地工具链（Ollama / LM Studio / Jan）直接加载

**不用 GGUF 的场景**：
- NVIDIA GPU 服务端部署追求极致吞吐——Marlin / FP8 快 3-6 倍
- 需要 continuous batching / PagedAttention 等高级服务特性——vLLM 路径更成熟
- 多用户高并发——llama.cpp 不是为并发设计的

**量化后缀选择**：
- 显存充足（如 4090 跑 7B）：`Q8_0` 几乎无损
- 主流选择：`Q4_K_M`（质量 / 体积甜点）
- 更高质量：`Q5_K_M` 或 `Q6_K`
- 极限压缩：`IQ3_M` 或 `IQ2_M`（必须用 imatrix）

## 工程实践
**imatrix 生成**：用目标领域的校准数据生成 imatrix 比通用数据好。比如你想跑中文对话，用 wikitext 中文版或中文新闻数据生成 imatrix，比英文 calibration data 跑出的 IQ4_XS 质量更好。

**量化 vs 反量化的不可分**：GGUF 文件一旦量化，文件本身就是量化结果——没法"在加载时再决定用什么量化"。想换 `Q4_K_M` 到 `Q5_K_M` 需要重新跑 `llama-quantize`。这与 GPTQ/AWQ 一样，但与 Marlin 的"加载时自动选择 Marlin 后端"不同。

**chat template 嵌入**：GGUF 文件里可以嵌入 chat template（`--chat-template` 参数指定），加载时自动使用。这避免了分发模型时附带 `tokenizer_config.json` 的麻烦——一份 `.gguf` 文件就能直接 `llama-cli -m ... --chat`。

**Ollama 的封装**：Ollama 是 llama.cpp 的用户友好封装，把 GGUF 模型管理做成 Docker-like 的 `ollama run / pull / list`。如果你不熟悉 llama.cpp 的命令行参数，用 Ollama 是最快的入门方式——但它牺牲了一些灵活度（如自定义量化、特殊 sampling 参数）。

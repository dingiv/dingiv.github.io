---
title: 模型量化
order: 20
---

# 模型量化
模型量化是本地部署最关键的优化手段——它直接决定模型能否装进显存。FP16 下每 1B 参数约需 2GB 显存，一个 70B 模型需要约 140GB。INT4 量化后，同样的模型只需约 35GB，单张 48GB 显卡就能装下。

量化的本质是将模型权重从高精度浮点数（FP32/BF16）映射到低精度表示（INT8/INT4），用极小的精度损失换取 2-4 倍的显存节省。现代量化方案的关键突破在于——不是均匀量化所有权重，而是分析哪些权重对精度敏感、哪些可以大胆压缩。

## GGUF：CPU 友好的瑞士军刀
GGUF 是 llama.cpp 的原生格式，支持 2-bit 到 8-bit 的多种量化变体，量化后模型为单个文件，分发极其方便。它的 K-quant 系列（Q4_K_M、Q5_K_M 等）是社区打磨最充分的量化方案——通过重要性矩阵分析每一层的敏感度，对 Attention 的 Q/K 投影和 MLP 的下投影使用更高精度（6-bit），对不太敏感的 V 投影和上投影使用更低精度（4-bit）。

Q4_K_M 是公认的甜点：PPL 退化通常 < 0.5，模型体积压缩到约原来的 1/4。Q5_K_M 质量略好但体积大 15%。IQ 系列（IQ2_XXS、IQ3_XXS）将位宽压缩到 2-3 bit，适合内存极其紧张的设备，但精度退化明显（PPL +2~3）。

GGUF 的另一关键优势是 CPU+GPU 混合推理。`-ngl`（number of GPU layers）参数控制多少层放在 GPU 上，其余层走 CPU 内存。即使显存放不下完整模型，只要系统内存够大就能跑——虽然 CPU 层的推理速度比 GPU 慢 2-5 倍，但至少能把模型跑起来。这在消费级硬件上运行 70B 模型时是决定性能力。

```bash
# 全部层放 GPU（-ngl 999 表示尽可能多）
./llama-cli -m qwen3.6-32b-Q4_K_M.gguf -ngl 999 --ctx-size 8192

# 只放 20 层在 GPU，其余走 CPU（适合显存不够的场景）
./llama-cli -m qwen3.6-70b-Q4_K_M.gguf -ngl 20 --ctx-size 4096
```

## AWQ：激活值感知量化
AWQ（Activation-aware Weight Quantization）的核心洞察是——权重的重要性不是均匀的，而是由激活值的分布决定的。约 1% 的权重通道承载了绝大部分激活值的能量，这些"显著权重"的量化误差会被激活值放大。

AWQ 的处理流程：分析一批校准数据的激活值分布 → 找到显著权重通道（激活值分布异常大的通道）→ 对这些通道使用 per-channel scaling 提升精度 → 其余通道正常量化。量化过程很快——70B 模型约 10-30 分钟，不需要计算 Hessian 矩阵。INT4 下 PPL 退化通常不到 0.3，优于同位宽的 GPTQ。

AWQ 格式在 vLLM 和 SGLang 中有原生 kernel 支持，推理速度接近 FP16。对于追求推理速度的服务端场景，AWQ 是目前的首选。

## GPTQ：基于 Hessian 的精准量化
GPTQ 是更早的 LLM 量化方案，基于 Optimal Brain Surgeon 框架。对每一层权重，逐列量化后计算 Hessian 矩阵来补偿剩余权重的量化误差——相当于"这一列量化损失了多少精度，在下一列补回来"。

GPTQ INT4 精度与 AWQ 接近，但量化过程慢 2-3 倍（需要计算 Hessian），且对校准数据的分布更敏感。存量模型多、生态成熟，vLLM 的 `gptq_marlin` kernel 推理速度优秀。新模型优先选 AWQ，存量 GPTQ 模型也能用。

## ISQ：加载时自动量化
mistral.rs 的 ISQ（In-Situ Quantization）改变了量化的工作流——不需要预先准备量化文件。下载原始 FP16 safetensors 权重，框架在加载时自动完成 channel-wise 量化。

对每个线性层权重 $W \in \mathbb{R}^{d_{out} \times d_{in}}$，对每个输出通道计算 scale 和零点：

$$
\Delta = \frac{\max(W_i) - \min(W_i)}{2^b - 1}, \quad z = \text{round}\left(\frac{-\min(W_i)}{\Delta}\right)
$$

量化在 CPU 或 GPU 上直接执行，7B 模型约 1-2 分钟，70B 模型约 8-12 分钟。对用户来说消除了"找 GGUF 版本"这一步摩擦。

## KV Cache 量化
除了权重量化，KV Cache 量化是长上下文场景的独立优化维度。32K 上下文的 KV Cache（FP16）约 16GB+——比模型本身还大。在长上下文或高并发推理中，显存真正的"杀手"往往不是模型权重，而是每个请求独立持有的 KV Cache。

KV Cache 量化的核心挑战：一是 Key 和 Value 的数值分布不同，Key 矩阵的某些通道存在异常值（outlier），直接量化精度崩塌——注意力得分计算严重失真，长文本中的逻辑链断裂或盲目幻觉；二是 KV Cache 在推理过程中动态增长，传统离线量化方法（GPTQ/AWQ）需要校准数据集，无法处理动态变化的数据。

### TurboQuant：在线无校准 KV Cache 压缩
TurboQuant（Google Research, ICLR 2026）是目前 KV Cache 量化领域最优雅的方案。核心设计思想：无需任何数据集校准（Data-oblivious），在线实时压缩 3-4 bit，数学上保证极低失真率。

TurboQuant 由两步协同工作：

**第一步：PolarQuant（极坐标量化 / 随机旋转）**。高维空间中的 KV 向量往往存在离群点——某些维度的值异常大——导致普通量化器找不到合适的量化区间。PolarQuant 将向量乘以一个随机正交矩阵（随机旋转变换），把极端离群值均匀分散到所有维度。旋转后各坐标轴的数据近乎独立服从已知分布（如 Beta 分布），此时用极简的标量量化器就能用 3-4 bit 捕捉到向量 95% 以上的主体能量。

**第二步：QJL 误差修正**。第一步量化后留下微小残差。传统量化直接丢弃残差，累积导致点积计算产生系统性偏差——注意力得分整体偏移，softmax 后某些 token 被系统性忽略或过度关注。TurboQuant 用额外的 1 bit 存储残差的正负符号，注意力计算时把这个 1-bit 补丁加回去，实现无偏点积估计。

效果：4-bit KV Cache 后，相同硬件支持 4 倍长的上下文。一张 24GB 的 4090，原来跑 70B 模型只能开 8K 上下文，TurboQuant 4-bit 后可到 32-64K。

### 工程落地的取舍
TurboQuant 并非"免费午餐"——它做的是存储压缩而非计算加速。KV Cache 以 3-4 bit 存入显存，但 Attention 计算前必须动态反量化回 FP16/BF16。这带来额外的算力开销：Decode 阶段延迟上升 10-50%（取决于量化位宽和硬件）。位宽越低，反量化开销越大。

对比硬件级 FP8 KV Cache：NVIDIA H100/4090 等 Ada/Hopper 架构的 Tensor Core 原生支持 FP8 矩阵乘法——KV Cache 以 FP8 存储且直接参与 Attention 计算，几乎零延迟开销。FP8 的压缩比（2x vs FP16）不如 4-bit（4x），但零延迟代价使其在延迟敏感场景更优。

QJL 残差补丁在低于 3 bit 时存在稳定性风险：1-bit 残差符号引入的方差被 softmax 指数放大。实践中很多部署关闭 QJL（仅使用纯 MSE 的 PolarQuant 模式）以换取更高稳定性。

选型决策：

| 场景 | 推荐方案 |
|------|---------|
| 显存刚好被长上下文爆掉（OOM） | TurboQuant 4-bit——以少许计算延迟换取几倍上下文空间 |
| 硬件支持 FP8（H100/4090），延迟敏感 | FP8 KV Cache——零延迟开销，2x 压缩 |
| 4-bit 仍不够 | PolarQuant 纯 MSE 模式（关闭 QJL），换稳定性 |
| 普通 8K 以下上下文，显存充足 | 不需要 KV Cache 量化——FP16 直接计算，零额外开销 |

本地部署时先确认 `nvidia-smi` 查看 GPU 是否支持 FP8——4090 可以，3090 不行（Ampere 部分支持但推理框架兼容性有限）。如果不支持 FP8 且上下文是瓶颈，TurboQuant 是最优解。

llama.cpp 中通过 `--cache-type-k` 和 `--cache-type-v` 控制 KV Cache 量化精度，vLLM 社区已集成 TurboQuant 4-bit 支持。

## 选型指南
单卡本地推理、追求省心：GGUF + llama.cpp/Ollama（Q4_K_M，CPU+GPU 混合推理兜底）。服务端高频调用、追求推理速度：AWQ + SGLang/vLLM（GPU native kernel，速度最优）。不想手动找量化文件：mistral.rs ISQ（自动量化，消除工作流摩擦）。长上下文场景（32K+）：KV Cache 4-bit 量化（显存节省 > 权重优化的收益）。

具体部署工具和启动命令见[部署工具](eco)，多 GPU 场景下量化和并行的配合见[多卡推理](multi-gpu)。

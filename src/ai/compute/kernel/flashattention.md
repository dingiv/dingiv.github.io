---
title: FlashAttention
order: 10
---

# Flash Attention
FlashAttention 是 Stanford 提出的 Attention 优化算法，通过分块计算和在线更新，将 Attention 的显存访问减少 10 倍以上，推理速度提升 2-3 倍。它是 LLM 推理加速的基石，被 vLLM、TGI、TensorRT-LLM 等主流框架采用。

## 标准 Attention 的问题
标准 Attention 的计算公式为 $\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$。朴素实现需要三步：1) 计算 $S = QK^T$（注意力分数矩阵），2) 计算 $P = \text{softmax}(S)$（注意力权重），3) 计算 $O = PV$（输出）。每步都需要读写显存，且 $S$ 和 $P$ 矩阵的大小为 $[n^2, n^2]$（$n$ 为序列长度），显存占用巨大。

对于序列长度 4096、batch size 32、head dim 128 的 Attention，$S$ 矩阵需要 $32 \times 4096 \times 4096 \times 2$ bytes（FP16）≈ 1GB，$P$ 矩阵同样 1GB。这还未计算 KV Cache，仅中间结果就占用 2GB 显存。显存带宽成为瓶颈，因为每次计算都需要读写这些大矩阵。

## FlashAttention 的优化
FlashAttention 的核心洞察是**无需显式构造 $S$ 和 $P$ 矩阵**，而是通过分块计算在线更新 softmax 和输出。具体来说，将 $Q, K, V$ 按序列维度分块（block size 如 128），每个块单独计算 Attention，然后合并块的结果。

这需要解决两个问题：**softmax 的在线更新**和**显存访问的减少**。对于 softmax，标准实现需要先计算所有分数才能归一化，但分块计算时无法获取全局信息。FlashAttention 使用 log-sum-exp 技巧，维护每个块的最大值和归一化因子，增量式地更新全局 softmax。对于显存访问，分块计算使得中间结果只需写入 shared memory，无需写入全局显存，大幅减少 HBM 访问。

算法伪代码如下：

```python
# 初始化
O = zeros([n, d])  # 输出矩阵
l = zeros([n])     # logsumexp 的分母
m = -inf * ones([n])  # softmax 的分子

# 分块计算
for Q_block, K_block, V_block in blocks(Q, K, V):
    # 计算 S_block = Q_block @ K_block.T（写入 shared memory）
    S_block = Q_block @ K_block.T.T / sqrt(d)
    # 计算 P_block = exp(S_block - m)（逐元素操作，无需存储）
    P_block = exp(S_block - m)
    # 更新 m 和 l
    new_m = max(m, max(S_block, axis=-1))
    new_l = exp(m - new_m) * l + sum(exp(S_block - new_m), axis=-1)
    # 更新 O
    O += exp(S_block - new_m) @ V_block
    m, l = new_m, new_l

# 最终归一化
O = O / l
```

关键在于 $m$ 和 $l$ 的增量式更新，使得分块计算的 softmax 结果与全局 softmax 等价。这避免了存储 $S$ 和 $P$ 矩阵，中间结果仅在 shared memory 中流转，显存访问从 $O(n^2)$ 降至 $O(n)$。

## FlashAttention-2
FlashAttention-2 进一步优化了工作负载分配和并行度。原版 FlashAttention 在序列维度并行（每个 block 处理一个序列片段），但这导致 GPU 的 SM（Streaming Multiprocessor）利用率不足。FlashAttention-2 在序列和 head 两个维度并行，每个 block 处理多个 head 的同一个序列片段，增加并行度，提升吞吐量 2 倍。

另一个改进是非矩阵乘法部分的优化。FlashAttention-2 手写 assembly 来优化 softmax 的指数、归一化、求和操作，减少寄存器压力和指令延迟，将非矩阵乘法部分的开销降低 50%。

## 使用方式
FlashAttention 集成在 PyTorch 2.0 中，通过 `torch.nn.functional.scaled_dot_product_attention` 调用：

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

q, k, v = ...  # [batch, heads, seq_len, head_dim]
output = scaled_dot_product_attention(q, k, v, is_causal=True)
```

对于旧版 PyTorch，可通过 pip 安装 `flash-attn` 包：

```python
from flash_attn import flash_attn_func

output = flash_attn_func(q, k, v, causal=True)
```

FlashAttention 对序列长度和 head dim 有要求：序列长度需是 128 的倍数（最优 256），head dim 需是 64/128/256 之一。不符合时性能会下降，但仍比标准 Attention 快。

## 性能对比
FlashAttention 官方 benchmark 显示，相比标准 Attention，FlashAttention-2 在 A100 上将前向传播加速 2-4 倍，反向传播加速 1.5-2 倍。显存占用方面，序列长度 2K 时标准 Attention 需要 16GB，FlashAttention 仅需 2GB。这使得长序列训练（如 32K 上下文的 GPT-3）成为可能。

推理场景下，FlashAttention 的优势更为明显，因为推理的 batch size 通常较小，GPU 的并行度更受限。FlashAttention-2 的高并行度设计使得小 batch 场景下仍能充分利用 GPU，将首 token 延迟降低 30-50%。

## 硬件依赖与版本演进
FlashAttention 不是一个“万能加速”，它的有效性与 GPU 架构代际紧密绑定。三代实现的硬件门槛如下。

| 版本 | 最低架构 | compute capability | 关键依赖 | 代表卡 |
| ---- | -------- | ------------------ | -------- | ------ |
| FA1 | Ampere | sm_80+ | 标准 `mma.sync` 指令 | RTX 30 系、A100 |
| FA2 | Ampere | sm_80+ | 同 FA1，针对计算并行重排 | 3090、4090、A100、H100 |
| FA3 | Hopper | sm_90+ | `wgmma` + `TMA` | H100、H200 |
| FA4 | Blackwell | sm_100+ | `tcgen05.mma` + `TMEM` | B100/B200、RTX 50 系 |

**FA2 是当代推理部署的事实标准**。它需要 sm_80 起跑，这意味着 RTX 20 系（sm_75）和 V100（sm_70）无法运行 FA2，只能跑 FA1 或回退到标准 Attention。RTX 30 系及以上才能完整享受 FA2 的加速。这是为什么本地 LLM 部署的架构底线被定在 Ampere——不是 BF16（虽然也是原因之一），而是 FA2。

**FA3 在 Hopper 上是质的飞跃**，不是 FA2 的小改进。Hopper 引入的 `wgmma` 指令让一个 warp group（4 个 warp）共同完成一次矩阵乘，而不是传统 `mma.sync` 的单个 warp——单条 MMA 指令覆盖的矩阵面积翻 4 倍。配合 `TMA` 异步加载 K/V 块到 SMEM，FA3 在 H100 上把训练 step time 又压到 FA2 的 1/2-1/3。但 FA3 的 `wgmma` 路径在 Ampere 上不能跑——Hopper 专属指令集。

**FA4 是 Blackwell 的专属**，利用了第五代 Tensor Core 的 `tcgen05.mma` 和新的 TMEM（Tensor Memory）存储累加器。它进一步将 SMEM 占用和寄存器压力大幅降低。在 B200 上 FA4 的吞吐量是 FA3 的约 2 倍。

## 在推理中的位置
FlashAttention 在推理栈里的角色不同于在训练中。

**Prefill 阶段**：与训练前向几乎一样，需要算整个 prompt 的注意力矩阵。FA2 在这里节省显存 + 提升吞吐。如果 prompt 长度 8K+，FA2 通常吃掉 Prefill 总时间的 30-40%，不启用 FA2 几乎跑不动现代模型。

**Decode 阶段**：attention 计算本身规模很小（每个新 token 只与历史 KV Cache 算一次 QK^T），FA2 的分块优势在这里不明显。但 **FA2 与 PagedAttention 的交互** 仍重要——vLLM 的 PagedAttention 把 KV Cache 分页存储，FA2 能感知到这种非连续布局，避免重复加载整块 KV。推理框架（vLLM、TGI、SGLang）默认启用 FA2，用户不需要额外配置。

**训练阶段**：反向传播需要保存中间结果，标准 Attention 会保存完整的 $S$ 和 $P$ 矩阵，显存爆炸；FA2 通过重计算（recomputation）避免保存中间值，反向时重算一遍——显存降 5-10 倍。这与推理场景的需求点不同。

**与 KV Cache 量化的交互**：当 KV Cache 启用 FP8 量化（如 `--kv-cache-dtype fp8`），FA2 内核需要支持 FP8 输入的版本。vLLM 默认会根据硬件和量化配置选择对应内核（FA2 FP8 路径需要 Hopper+，Ada 上的 FP8 KV Cache 可能回退到标准 Attention）。

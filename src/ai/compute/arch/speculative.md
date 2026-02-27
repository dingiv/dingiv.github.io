---
title: 投机采样
order: 30
---

# 投机采样

投机采样（Speculative Decoding，也叫投机推理）是一种通过小模型辅助大模型加速生成的技术。它的核心思想是：让一个小而快的 draft model（草稿模型）快速生成多个 token，然后由大模型并行验证，若验证通过则保留，否则回退重新生成。

## 原理

大模型的生成是自回归的：每个 token 的生成都依赖于之前所有 token。这意味着生成 100 个 token 需要 100 次 forward pass，计算量巨大。投机采样通过引入 draft model 来加速这个过程。

投机采样分为两个阶段：

1. **Draft 阶段**：draft model 快速生成 $k$ 个 token（如 $k=8$）。draft model 通常很小（如 1B 参数），推理速度是大模型的 5-10 倍。

2. **Verify 阶段**：将这 $k$ 个 token 和原始 prompt 一起输入大模型，并行计算所有 token 的概率。如果大模型的概率高于 draft model 的预测，则接受这 $k$ 个 token；否则回退到第一个不匹配的 token，由大模型重新生成。

投机采样的收益来自两个方面：一是 draft model 的快速生成（小模型推理快），二是大模型的并行验证（一次 forward pass 验证多个 token）。当 draft model 的准确率较高时（如 >80%），大部分 token 都会通过验证，实际加速比可达 2-3 倍。

## 实现细节

投机采样的关键挑战是**如何选择 draft model**。draft model 需要满足两个条件：一是与大模型的分布一致（否则验证通过率低），二是推理速度足够快（否则无法加速）。常见的 draft model 选择包括：

1. **同一模型的小版本**：如用 Llama-2-7B 作为 Llama-2-70B 的 draft model
2. **蒸馏模型**：通过知识蒸馏专门训练的 draft model，与大模型行为对齐
3. **提前退出**：使用同一模型，但减少层数或使用早退机制

验证阶段的实现有两种方式：并行采样和串行采样。并行采样是一次性计算所有候选 token 的概率，然后与 draft model 的概率比较。串行采样是逐个 token 验证，遇到不匹配就停止。并行采样的 GPU 利用率更高，但需要更多显存（需要存储所有候选 token 的 KV Cache）。

## 验证机制

投机采样的验证有两种主要方法：

1. **概率匹配**：比较 draft model 和大模型在候选 token 上的概率分布。如果大模型的概率高于 draft model，则接受；否则拒绝。这是最原始的验证方法，简单但可能过于保守。

2. **Token 验证**：只验证候选 token 的值是否一致，不考虑概率。如果候选 token 与大模型预测的 token 相同，则接受；否则拒绝。这种方法更激进，加速比更高，但可能降低输出质量。

当前主流框架（如 vLLM、TGI）使用概率匹配的变种：计算候选序列的对数概率比值（logits ratio），如果比值高于阈值则接受。这平衡了加速比和输出质量。

## 性能分析

投机采样的加速比取决于三个因素：draft model 的准确率、speculation length（$k$ 的大小）、大模型的计算效率。

理论上，最大加速比为 $1 + k \times \text{speed\_ratio}$，其中 $\text{speed\_ratio}$ 是 draft model 相比大模型的速度比（如 5 倍）。但实际加速比受限于验证通过率（acceptance rate），如果验证通过率只有 50%，实际加速比会减半。

draft model 的选择至关重要。如果 draft model 与大模型分布差异过大，验证通过率会很低，不仅无法加速，反而会增加计算开销。理想的 draft model 应该是大模型的蒸馏版本，通过知识蒸馏学习大模型的行为。

## 使用方式

vLLM 原生支持投机采样：

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models import draft_model

llm = LLM(
    model="meta-llama/Llama-2-70b",           # 大模型
    draft_model="meta-llama/Llama-2-7b",      # draft model
    num_speculative_tokens=8,                  # 每次投机 8 个 token
)

output = llm.generate("Hello, world", SamplingParams(max_tokens=100))
```

TGI 也支持投机采样，通过 `--speculative-decoding` 参数启用：

```bash
model=meta-llama/Llama-2-70b
draft_model=meta-llama/Llama-2-7b

docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:latest \
  --model-id $model \
  --draft-model-id $draft_model \
  --speculative-decoding \
  --num-speculative-tokens 8
```

## 局限性

投机采样的局限性在于需要额外的 draft model，这增加了部署复杂度。同时，draft model 需要与主模型保持分布一致，这增加了维护成本。对于频繁变化的模型（如定期重新训练），draft model 也需要同步更新。

另一个局限是对于输出质量要求极高的场景（如数学计算、代码生成），投机采样可能降低输出质量。因为 draft model 的错误会通过验证传播到大模型，虽然概率不高，但仍可能发生。

对于这些场景，可以考虑使用更保守的验证策略（如降低阈值、减少 speculation length），或者完全禁用投机采样。

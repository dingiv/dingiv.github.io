---
title: 投机解码
order: 30
---

# 投机解码
投机解码（Speculative Decoding，也叫投机推理）是一种通过小模型辅助大模型加速生成的技术。它的核心思想是：让一个小而快的 draft model（草稿模型）快速生成多个 token，然后由大模型并行验证，若验证通过则保留，否则回退重新生成。

## 原理
大模型的生成是自回归的：每个 token 的生成都依赖于之前所有 token。这意味着生成 100 个 token 需要 100 次 forward pass，计算量巨大。投机解码通过引入 draft model 来加速这个过程。

投机解码分为两个阶段：

1. **Draft 阶段**：draft model 快速生成 $k$ 个 token（如 $k=8$）。draft model 通常很小（如 1B 参数），推理速度是大模型的 5-10 倍。

2. **Verify 阶段**：将这 $k$ 个 token 和原始 prompt 一起输入大模型，并行计算所有 token 的概率。如果大模型的概率高于 draft model 的预测，则接受这 $k$ 个 token；否则回退到第一个不匹配的 token，由大模型重新生成。

投机解码的收益来自两个方面：一是 draft model 的快速生成（小模型推理快），二是大模型的并行验证（一次 forward pass 验证多个 token）。当 draft model 的准确率较高时（如 >80%），大部分 token 都会通过验证，实际加速比可达 2-3 倍。

## 实现细节
投机解码的关键挑战是**如何选择 draft model**。draft model 需要满足两个条件：一是与大模型的分布一致（否则验证通过率低），二是推理速度足够快（否则无法加速）。常见的 draft model 选择包括：

1. **同一模型的小版本**：如用 Llama-2-7B 作为 Llama-2-70B 的 draft model
2. **蒸馏模型**：通过知识蒸馏专门训练的 draft model，与大模型行为对齐
3. **提前退出**：使用同一模型，但减少层数或使用早退机制

验证阶段的实现有两种方式：并行采样和串行采样。并行采样是一次性计算所有候选 token 的概率，然后与 draft model 的概率比较。串行采样是逐个 token 验证，遇到不匹配就停止。并行采样的 GPU 利用率更高，但需要更多显存（需要存储所有候选 token 的 KV Cache）。

## 验证机制
投机解码的验证有两种主要方法：

1. **概率匹配**：比较 draft model 和大模型在候选 token 上的概率分布。如果大模型的概率高于 draft model，则接受；否则拒绝。这是最原始的验证方法，简单但可能过于保守。

2. **Token 验证**：只验证候选 token 的值是否一致，不考虑概率。如果候选 token 与大模型预测的 token 相同，则接受；否则拒绝。这种方法更激进，加速比更高，但可能降低输出质量。

当前主流框架（如 vLLM、TGI）使用概率匹配的变种：计算候选序列的对数概率比值（logits ratio），如果比值高于阈值则接受。这平衡了加速比和输出质量。

## 性能分析
投机解码的加速比取决于三个因素：draft model 的准确率、speculation length（$k$ 的大小）、大模型的计算效率。

理论上，最大加速比为 $1 + k \times \text{speed\_ratio}$，其中 $\text{speed\_ratio}$ 是 draft model 相比大模型的速度比（如 5 倍）。但实际加速比受限于验证通过率（acceptance rate），如果验证通过率只有 50%，实际加速比会减半。

draft model 的选择至关重要。如果 draft model 与大模型分布差异过大，验证通过率会很低，不仅无法加速，反而会增加计算开销。理想的 draft model 应该是大模型的蒸馏版本，通过知识蒸馏学习大模型的行为。

## 使用方式
vLLM 原生支持投机解码：

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

TGI 也支持投机解码，通过 `--speculative-decoding` 参数启用：

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
投机解码的局限性在于需要额外的 draft model，这增加了部署复杂度。同时，draft model 需要与主模型保持分布一致，这增加了维护成本。对于频繁变化的模型（如定期重新训练），draft model 也需要同步更新。

另一个局限是对于输出质量要求极高的场景（如数学计算、代码生成），投机解码可能降低输出质量。因为 draft model 的错误会通过验证传播到大模型，虽然概率不高，但仍可能发生。

对于这些场景，可以考虑使用更保守的验证策略（如降低阈值、减少 speculation length），或者完全禁用投机解码。

## MTP：多 Token 预测
MTP（Multi-Token Prediction）是不依赖独立 draft model 的推理加速方法。传统自回归模型每次 forward 只预测下一个 token，MTP 在模型最后一层之上加多个并行的预测头（prediction head），每个头负责预测不同位置的未来 token——第 1 个 head 预测 token $t+1$，第 2 个 head 预测 token $t+2$，以此类推。推理时一次 forward pass 同时输出当前 token + $N$ 个未来位置的候选。

MTP head 的结构极其轻量——每个 head 通常只是 RMSNorm + 小规模线性投影。所有 head 的参数量合计不到主模型的 1%。DeepSeek-V3 的 MTP 设计更进一步，head 之间建立了因果依赖关系：第 $i$ 个 head 接受第 $i-1$ 个 head 的输出作为额外输入，对未来的预测不再是独立的，而是"已知前一步预测结果后再预测下一步"。这种链式结构使得更远位置的预测质量不会因距离而严重衰减。

MTP 的核心优势在于**零额外显存和零额外词表对齐成本**。传统投机解码需要加载完整 draft model（即使是 1.8B 量化版也要 1-2GB 显存），且 draft model 必须与主模型共享词表。MTP 的 head 已包含在模型权重中，部署即自动生效——不需要 `--draft-model` 参数，不需要管理两个模型的版本匹配，不需要在两个模型之间协调显存分配。

MTP 在 Memory-Bound 设备上的实际加速比往往高于理论预期。因为 MTP head 与主模型共享 hidden representation——主模型正常计算每个 token 的 hidden state 的过程已经完成了绝大部分工作，MTP head 只需要在此基础上做轻量投影。相比独立 draft model 需要从零开始做完整 forward pass，MTP 的计算增量几乎为零——"多预测的 token 几乎是免费的"。在 3090 等显存带宽受限的设备上，MTP 的 Decode 加速比（1.5-2x）接近甚至超过加载额外 draft model 的投机方案（1.5-2.5x），且不需要牺牲显存给 draft model。

MTP 的局限：需要模型本身在训练时就加了 MTP head——Qwen 2.5、DeepSeek-V3 等 2025 年后的新模型大多支持，但 2024 年及更早的模型（Llama-3、Qwen 2、Mistral）均不支持。可预测的额外 token 数受 head 数量限制——通常 1-2 个，而独立 draft model 可预测 5-8 个。单个 head 的预测质量通常低于同系列独立小模型——小模型有完整的多层 Transformer 做推理，MTP head 只有浅层投影。

MTP 与所有并行策略天然兼容——它是模型内部的能力，无论 PP 还是 TP 都不影响 MTP 的生效。这是 MTP 在多卡场景下对独立 draft model 的最大优势：vLLM/SGLang 中 PP 与独立 draft model 投机解码不兼容，但 PP + 原生 MTP 可以同时工作，因为 MTP 不涉及第二个模型的加载和调度。

```bash
# vLLM: MTP 自动检测启用（模型内置 MTP head）
vllm serve Qwen/Qwen3.6-32B --tp 2  # MTP 随 TP 自动生效

# llama.cpp: 原生 MTP
./llama-server -m qwen3.6-32b-Q4_K_M.gguf --speculative-tokens 2
```

Meta 的 Llama 4 在预训练阶段就使用了 MTP——训练时让模型学习同时预测未来的多个 token，推理时直接复用训练好的预测头。MTP 在编码和数学推理等结构化输出任务上表现更好——这类任务的 token 序列有较强的确定性，未来 token 的预测置信度更高。

## Lookahead Decoding
Lookahead Decoding（前向解码）是另一种不依赖 draft model 的并行生成策略。核心思路是在已经生成的 token 序列上构造多个候选 n-gram（连续 n 个 token 的片段），然后一次 forward 并行验证这些候选片段是否匹配模型的输出。

具体过程：给定已生成的 token 序列 `[t1, t2, ..., tk]`，从序列中提取多个 n-gram 作为候选（如 `[t2, t3]`、`[t4, t5, t6]`），然后验证每个候选的下一个 token 是否与模型预测一致。如果一致，这个候选片段直接作为已生成的 token 保留。这个过程可以一次验证多个候选，每次 forward 可接受 2-5 个 token。

Lookahead 的优势是**完全零额外模型**——候选 token 来自模型自己已经生成的序列，不需要任何 draft model、不需要额外的预测头、不需要额外的训练。代价是有效加速比不如投机解码和 MTP——候选 token 的来源是已生成序列中的重复模式，对于对话这类 token 序列规律性较弱的任务，命中率通常只有 40-60%（每 100 个 token 中有 40-60 个来自 n-gram 片段）。但对于代码生成（变量名、API 调用模式高度重复）和 JSON/XML 输出（结构高度模板化）这类场景，命中率可达 70-90%。

在实现上，Lookahead Decoding 可以与投机解码组合使用——Lookahead 产生一部分低成本候选 token，投机解码用 draft model 补充那些 Lookahead 猜不到的部分。DeepSeek-V3 的技术报告中提到了类似的组合策略。

## DFlash：扩散式块投机
DFlash 是 2025-2026 年投机解码领域的重要进展。传统投机方法（EAGLE、Medusa）依赖自回归 draft model，逐 token 生成候选序列——这个过程本身也是串行的。DFlash 用一个轻量级扩散模型（block diffusion）一次性并行生成一整块 tokens（如 8-16 个），然后让大模型并行验证整块。

扩散模型的特点是可以在任意位置"填充"——给定上下文后，扩散过程直接从噪声出发，逐步去噪得到候选 token 块。这完全消除了 draft model 的串行瓶颈。在大模型验证阶段，候选块的每个 token 使用特殊设计的 ancestor-only attention mask，使整块可以在一次 forward pass 中并行验证。

DFlash 在 Qwen、Llama 等模型上实现 2-6 倍的无损加速。加速比取决于生成任务的特征——代码生成等结构化输出场景中 draft 质量更高，加速比也更显著。

## DDTree：树状扩散验证
DDTree（Diffusion Draft Tree）是 DFlash 的扩展。DFlash 一次扩散生成一条候选链（chain），大模型按链验证；DDTree 利用同一次扩散输出中每个位置的概率分布，构建一棵候选树（draft tree），包含多条可能的分支路径。

树状结构的关键价值在于：大模型一次 forward pass 可以并行验证整棵树上的所有分支，而不是只验证一条路径。候选树通过 best-first heap 算法生成，在固定节点预算下选择最有希望的路径分支。验证时使用 tree attention mask，各分支节点只关注其祖先节点，保持线性复杂度。

DDTree 在 DFlash 基础上将有效接受长度（acceptance length）进一步提升 1.5-2 倍，总加速可达 6-8 倍，且保持无损。特别适合代码生成、长上下文推理等对输出质量要求高的场景。

## QuantSpec：自投机量化
QuantSpec（Apple, 2025）结合了自投机解码和分层量化。传统投机需要一个独立的 draft model，增加部署复杂度。自投机解码使用主模型本身作为 draft——通过减少层数或降低精度来获得更快的 draft pass。

QuantSpec 的做法：主模型本身经过 4-bit 权重和 KV Cache 量化后作为 draft model。draft pass 用 4-bit 精度快速生成候选 tokens，验证 pass 用完整精度验证。因为 draft 和 target 共享同一套权重（仅精度不同），分布一致性极好，验证通过率高。

4-bit 量化的 draft model 推理速度提高了约 4 倍，加上自投机避免独立 draft model 的显存开销，QuantSpec 在长上下文场景下特别有优势——节省的显存可以容纳更长的 KV Cache。

## Diff-LLM：扩散语言模型
在投机解码之外，扩散模型直接作为语言模型是一个更激进的方向。传统自回归 LLM 逐 token 生成，扩散语言模型（如 LLaDA、dLLM）通过迭代去噪过程并行生成全部 tokens。理论上可以实现完全并行的生成，推理延迟与序列长度无关。

目前扩散 LLM 的生成质量尚未达到自回归模型的水平，但在某些场景（如短文本补全、代码填充）中已展现出竞争力。DFlash 可以视为扩散模型在"辅助生成"角色上的成功应用——不需要替换主模型，仅用扩散做高效 draft。未来如果扩散 LLM 质量进一步提升，"直接用扩散生成"可能成为新的推理范式。

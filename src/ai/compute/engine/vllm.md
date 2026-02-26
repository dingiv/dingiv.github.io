---
title: vLLM
---

# vLLM

vLLM 是 UC Berkeley 研发的开源推理引擎，通过 PagedAttention 创新性地解决了 KV Cache 管理问题，成为 2023-2024 年最流行的 LLM 推理框架之一。

## PagedAttention

传统推理引擎将每个请求的 KV Cache 作为连续显存块管理，这带来两个问题：一是预分配困难——无法预测序列长度，预分配过多浪费显存，过少则中断生成；二是内存碎片——不同请求的序列长度差异导致显存无法有效复用。

PagedAttention 借鉴操作系统的虚拟内存管理，将 KV Cache 分页（page），每页固定大小（如 16 个 token）。每个请求的 KV Cache 是一组页面的链表，页面可分散在显存任意位置。这消除了预分配问题——按需申请页面；也解决了内存碎片——页面大小统一，可自由复用。

更重要的是，PagedAttention 支持跨请求的 KV Cache 共享。系统 prompt（如"你是一个有用的助手"）在多个请求中完全相同，共享其 KV Cache 可节省大量显存。在多轮对话中，用户 prompt 重复出现时也可共享，这特别适合客服机器人等场景。

## 连续批处理

vLLM 的调度器支持 continuous batching，即当 batch 中某个序列生成结束时（遇到 EOS token 或达到长度限制），立即插入新序列，而非等待整个 batch 完成。这充分利用了 GPU 的计算资源，避免了传统静态 batch 中"短序列完成后 GPU 空转"的问题。

调度器还支持前缀缓存（prefix caching）——重复的 prompt 前缀只计算一次 KV Cache，后续请求直接复用。对于常见的系统 prompt + 用户 prompt 组合，这可将首 token 延迟降低 50% 以上。

## 使用方式

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b", tensor_parallel_size=2)
prompts = ["Hello, my name is", "The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
outputs = llm.generate(prompts, sampling_params)
```

vLLM 兼容 OpenAI API，可通过 `vllm serve` 命令启动兼容服务，直接替换 OpenAI API 的 base_url 即可。这种兼容性使得从 OpenAI 迁移到自托管模型变得非常简单。

## 性能对比

vLLM 官方 benchmark 显示，相比 TGI（Text Generation Inference），vLLM 在吞吐量上提升 2-4 倍，在首 token 延迟上降低 30-50%。这主要归功于 PagedAttention 的高效内存管理和连续批处理的调度优化。但 vLLM 的 GPU 显存占用略高，因为页面元数据（page table）需要额外维护。

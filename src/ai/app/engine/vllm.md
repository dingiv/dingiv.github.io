---
title: vLLM
order: 32
---

# vLLM
vLLM 是 UC Berkeley 开源的高性能 LLM 推理引擎，核心创新是 PagedAttention 技术。它解决了传统推理引擎的两个根本性问题：KV Cache 显存碎片导致利用率低下，以及静态批处理在并发场景下的吞吐量瓶颈。

## PagedAttention：KV Cache 的分页管理
传统推理引擎将每个请求的 KV Cache 作为连续内存块分配。不同请求的序列长度差异巨大——短的几十个 token，长的几千个 token——连续分配导致严重的显存碎片。实际显存利用率通常只有 20-40%，大量显存被已分配但未使用的碎片占据。

PagedAttention 借鉴操作系统的虚拟内存管理：将 KV Cache 切分为固定大小的 block（类似内存页），每个请求按需分配 block，不要求连续存储。block 大小可配置（默认 16 个 token），当请求长度增长时动态分配更多 block——就像进程的堆增长时分配更多内存页。释放时按 block 回收，消除碎片。

效果：显存利用率从 20-40% 提升到 80%+。同样的硬件可以服务更多并发请求，或使用更大的 batch size、更长的上下文。

## 连续批处理
连续批处理（Continuous Batching）是 PagedAttention 之上的调度层创新。传统静态批处理等待固定数量的请求（如 8 个）到齐后才一起提交 GPU——请求到达不均匀时，先到的请求需要等待后到的请求，延迟不可预测。

vLLM 在每次迭代时动态决定本轮处理哪些请求：新请求可以立即加入当前 batch，已完成的请求立即释放资源。一个生成了 500 个 token 的请求完成后，不会浪费剩余空间等 batch 中其他请求结束。

配合 PagedAttention 的 block 级内存管理，请求的加入和退出只需要分配/释放 block 指针，无需整体内存搬移。并发场景下吞吐量可达传统方案的 5-10 倍。

## 并行策略
vLLM 原生支持张量并行（TP）和流水线并行（PP），通过简单的参数配置：

```bash
# TP=2：模型权重切分到 2 张 GPU
vllm serve meta-llama/Llama-4-70B --tensor-parallel-size 2

# TP=2 + PP=2 混合（4 GPU）
vllm serve Qwen/Qwen3.6-72B --tensor-parallel-size 2 --pipeline-parallel-size 2
```

TP 将每层权重矩阵切分到多卡，层内 AllReduce 通信频繁，依赖高带宽互联（NVLink 或高速 PCIe）。PP 按层切分，通信仅在层边界发生，对低带宽环境更友好。

vLLM 的分布式推理依赖 Ray 进行进程管理和通信协调，与 NCCL 配合完成 GPU 间数据传输。

## 量化支持
vLLM 对 AWQ 和 GPTQ 量化格式有原生 kernel 支持。AWQ 模型通过 AutoAWQ 加载，GPTQ 通过 `gptq_marlin` kernel 加速推理。量化模型在 vLLM 中的推理速度接近 FP16，是目前服务端部署的标准方案。

```bash
# 启动 AWQ 量化模型
vllm serve Qwen/Qwen3.6-32B-Instruct-AWQ --max-model-len 8192

# 启动 GPTQ 量化模型
vllm serve TheBloke/Llama-3-70B-GPTQ --quantization gptq \
  --max-model-len 16384 --gpu-memory-utilization 0.90
```

`--gpu-memory-utilization` 控制显存使用率上限（默认 0.90），用于为 KV Cache 和其他运行时开销预留空间。

## OpenAI 兼容 API
vLLM 默认启动 OpenAI 兼容的 HTTP 服务，任何使用 `openai` Python 库的代码都可以直接切换 base_url：

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
response = client.chat.completions.create(
    model="Qwen/Qwen3.6-32B",
    messages=[{"role": "user", "content": "你好"}]
)
```

支持的 API 包括 `/v1/chat/completions`（对话补全）、`/v1/completions`（文本补全）、`/v1/embeddings`（文本嵌入）以及流式输出。

## 与 SGLang 的对比
SGLang 是 vLLM 在单 GPU 场景下的主要竞品。两者的关键差异：

vLLM 对多 GPU 分布式推理（TP/PP）的支持更成熟，依赖 Ray 做进程管理。SGLang 的 RadixAttention 在多轮对话和 RAG 场景中提供更高效的 prefix caching，且部署更简单（不需 Ray）。显存开销方面，SGLang 通常节省 10-20%。单 GPU 多轮对话优先 SGLang，多 GPU 场景优先 vLLM。

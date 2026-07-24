---
title: 部署工具
order: 10
---

# 部署工具生态
本地部署大模型的核心挑战是在消费级硬件有限的显存和算力下实现可用的推理速度。不同的部署工具在量化格式、GPU 后端、并发支持和部署复杂度上各有侧重。本文对比主流部署工具的特性和适用场景——具体引擎的架构和用法见[推理引擎](/ai/app/engine/)。

## 工具对比
llama.cpp 是 C++ 实现的轻量级推理引擎，专为消费级 GPU 和 CPU 设计。核心优势是 GGUF 量化格式（2-8 bit，单文件分发）和 CPU+GPU 混合推理。CPU 推理性能在同类工具中最优。不支持连续批处理，不适合多用户并发场景。适合个人本地推理和对硬件有精细控制需求的用户。

Ollama 是 llama.cpp 的上层封装，提供 Docker 式的模型管理体验。`ollama pull llama3` 下载模型，`ollama run llama3` 启动对话。自动管理量化参数、GPU 层数和 KV Cache 类型。Modelfile 支持自定义 system prompt 和参数。适合非技术用户快速上手，高并发场景性能有限。

SGLang 的核心优势是 RadixAttention——自动前缀缓存。当多个请求共享相同前缀（system prompt、RAG 检索结果、工具定义）时，自动识别并复用 KV Cache，多轮对话吞吐量可达 vLLM 的 5 倍以上。在同等模型和上下文长度下显存开销通常比 vLLM 低 10-20%。部署极简——`pip install sglang && sglang serve`，不需要 Ray 或多进程管理。适合单 GPU 多轮对话和 RAG 场景。

vLLM 通过 PagedAttention 解决 KV Cache 碎片问题，连续批处理支持动态请求调度。多 GPU 分布式推理（TP/PP）支持最成熟，依赖 Ray 做进程管理。兼容 OpenAI API。适合多 GPU 场景和高并发 API 服务。

mistral.rs 的 ISQ 就地量化在加载模型时自动完成 FP16→INT4/INT8 转换，不需要预先准备量化文件。支持 CUDA、Metal、ROCm、CPU 四种加速单元混合调度。兼容 OpenAI API。适合不想手动管理量化文件的用户。

| 工具 | 量化格式 | GPU 后端 | 连续批处理 | 部署复杂度 | 最适合场景 |
|------|---------|---------|:---:|:---:|-----------|
| llama.cpp | GGUF | CUDA/Metal/ROCm/Vulkan | 否 | 低 | 个人本地推理 |
| Ollama | GGUF | CUDA/Metal/ROCm | 否 | 极低 | 非技术用户 |
| SGLang | AWQ/GPTQ/GGUF | CUDA/ROCm | 是 | 低 | 单 GPU 多轮对话 |
| vLLM | AWQ/GPTQ | CUDA/ROCm | 是 | 中 | 多 GPU 并发服务 |
| mistral.rs | ISQ/GGUF/AWQ | CUDA/Metal/ROCm | 否 | 低 | Rust 生态 |

## 启动命令参考
```bash
# llama.cpp: HTTP server 模式
./llama-server -m qwen3.6-32b-Q4_K_M.gguf \
  --ctx-size 8192 --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 --port 8080 -ngl 999

# Ollama: Modelfile 定制
# FROM qwen2.5:32b-q4_K_M
# PARAMETER temperature 0.7
# PARAMETER num_ctx 8192

# SGLang: 单 GPU 一键启动（RadixAttention 自动生效）
sglang serve Qwen/Qwen3.6-32B-Instruct-AWQ --max-total-tokens 8192

# vLLM: 多 GPU TP 模式
vllm serve Qwen/Qwen3.6-72B-Instruct-AWQ \
  --tensor-parallel-size 2 --max-model-len 16384 --gpu-memory-utilization 0.90

# mistral.rs: ISQ 自动量化
mistralrs-server -i plain -m Qwen/Qwen3.6-8B-Instruct --isq Q4K
```

各工具对量化格式的兼容性和选型详见[模型量化](quantization)。多 GPU 并行配置和 TP/PP 参数选择见[多卡推理](multi-gpu)。投机解码、KV Cache 优化等加速技术见[推理优化](optimization)。服务上线后的监控、负载均衡和成本控制见[服务部署](serving)。

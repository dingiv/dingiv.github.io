---
title: 推理引擎
order: 30
---

# 推理引擎
推理引擎是连接模型和应用的桥梁——负责加载权重、管理显存、处理输入输出。不同于关注"怎么部署"的 AI 算力层面，推理引擎聚焦于引擎本身的架构特性、使用方式和选型对比。

关于 GPU 选型、量化策略、多卡并行、推理优化和服务部署，见 [AI 部署](/ai/compute/deploy/) 章节。

## Ollama
Ollama 是最简单的本地推理方案，封装了模型下载、加载和推理的完整流程。`ollama run llama3` 自动下载模型文件并启动交互式 shell。Ollama 提供 HTTP API（默认端口 11434），支持 Modelfile 定制 system prompt 和参数。底层使用 llama.cpp，但屏蔽了量化参数、GPU 层数等配置细节。适合快速体验和开发调试，高并发场景下性能有限。

```bash
ollama run qwen2.5:14b
# 或通过 Modelfile 定制
# FROM qwen2.5:32b-q4_K_M
# PARAMETER temperature 0.7
# PARAMETER num_ctx 8192
```

## vLLM
vLLM 是高性能推理引擎的代表，核心创新是 PagedAttention——借鉴操作系统的分页内存管理，将 KV Cache 切分为固定大小的 block 按需分配，显存利用率从 20% 提升到 80%+。支持连续批处理（Continuous Batching），在生成过程中动态插入新请求，单个请求完成后立即释放资源。支持张量并行和流水线并行，兼容 OpenAI API。

详见 [vLLM](vllm)。

## Text Generation Inference (TGI)
TGI 是 Hugging Face 的推理框架，内置动态批处理、Flash Attention、流式输出和 Token 级别 logprobs。通过 `--quantize` 支持 AWQ、GPTQ、BitsAndBytes 等量化方案。提供 Prometheus 指标端点，方便集成监控。适合追求生产环境稳定性和可观测性的场景。

## llama.cpp
llama.cpp 是 C++ 实现的轻量级推理引擎，专为消费级硬件设计。核心优势是 GGUF 量化格式和 CPU+GPU 混合推理。对 Apple Silicon（Metal）、AMD GPU（ROCm）、NVIDIA GPU（CUDA）均有良好支持。提供 CLI、HTTP server 和多种语言绑定。

详见 [llama.cpp](llama_cpp)。

## mistral.rs
mistral.rs 是 Rust 生态中最成熟的大模型推理引擎之一，底层基于 candle 框架。核心特色是 ISQ（就地量化）——加载模型时自动将 FP16 权重转为 INT4/INT8，不需要预先准备量化文件。支持 CUDA、Metal、ROCm、CPU 四种加速单元并可混合调度。兼容 OpenAI API。

详见 [mistral.rs](mistral_rs)。

## 其他 Rust 推理工具
Rust 生态中还有 candle（纯 Rust 推理框架，极简依赖）、llama-cpp-rs（llama.cpp 的 Rust FFI 绑定）、burn（完整深度学习框架，支持训练+推理）。详见 [Rust 推理引擎](rust_infer)。

## 参数调优
无论使用哪种引擎，以下参数直接影响输出质量：Temperature（0-2，常用 0.7-1.0）控制随机性——代码生成设 < 0.3，创意写作设 0.8-1.0。Top-p（0-1，常用 0.9）从累积概率达到 p 的最小词集合中采样。Repeat penalty（1.0-1.2）惩罚重复输出，避免模型陷入循环。Max tokens 限制输出长度，需根据上下文窗口合理设置。

引擎级别的批处理策略影响吞吐量：静态批处理等待固定数量请求后一起处理，延迟高；连续批处理（vLLM/TGI）动态插入新请求，高并发场景吞吐量可提升 10 倍。

---
title: llama.cpp
order: 34
---

# llama.cpp
llama.cpp 是 C++ 实现的轻量级 LLM 推理引擎，专为消费级硬件和边缘设备设计。它的设计哲学是"在任何有 CPU 的设备上跑模型"——GPU 是加速选项而非硬依赖。

## GGUF 量化格式
GGUF 是 llama.cpp 的原生模型格式。与 safetensors 不同，GGUF 将模型权重、tokenizer、超参数和元数据打包为单个文件，分发极其方便。支持从 2-bit 到 8-bit 的多种量化变体，K-quant 系列（Q4_K_M、Q5_K_M）通过重要性分析对不同层使用不同精度。

GGUF 的核心设计目标之一是支持 CPU+GPU 混合推理——`-ngl`（number of GPU layers）参数指定多少层放在 GPU 上，其余层走 CPU 内存。这突破了纯 GPU 推理的显存上限——如果系统有 128GB 内存和 24GB 显存，可以跑 70B 模型（模型加载在内存中，20-30 层放在 GPU 上加速）。

```bash
# 全部层放 GPU
./llama-cli -m model-Q4_K_M.gguf -ngl 999

# 部分层放 GPU，其余走 CPU
./llama-cli -m model-Q4_K_M.gguf -ngl 20 --ctx-size 4096
```

## 硬件后端
llama.cpp 支持多种 GPU 后端，编译时选择：

- **CUDA**：NVIDIA GPU，通过 cuBLAS 加速。编译选项 `-DGGML_CUDA=ON`
- **Metal**：Apple Silicon（M1/M2/M3），GPU 推理。编译选项 `-DGGML_METAL=ON`
- **ROCm/HIP**：AMD GPU，编译选项 `-DGGML_HIPBLAS=ON`。RDNA 3 上的 GGUF 推理稳定性与 CUDA 几乎无异
- **Vulkan**：跨平台 GPU 后端，支持 NVIDIA/AMD/Intel
- **CPU**：纯 CPU 推理，通过 BLAS 库加速（OpenBLAS、Intel MKL）

Apple Silicon 上的 Metal 后端表现优异——M1 Max 32 核 GPU 上 7B 模型 Q4_K_M 推理速度约 20-30 token/s。

## 推理加速
llama.cpp 集成了投机解码——通过 `--draft-model` 指定同系列的微型模型做 draft：

```bash
./llama-server -m qwen3.6-32b-Q4_K_M.gguf \
  --draft-model qwen3.6-1.8b-Q4_K_M.gguf \
  --speculative-tokens 8 -ngl 999
```

KV Cache 量化通过 `--cache-type-k` 和 `--cache-type-v` 控制，支持 INT8 和 INT4：

```bash
# KV Cache INT8 量化（32K 上下文场景显著节省显存）
./llama-cli -m model.gguf --ctx-size 32768 \
  --cache-type-k q8_0 --cache-type-v q8_0
```

Flash Attention 在编译时启用（`-DGGML_CUDA_FA=ON`），对长上下文推理有明显加速。

## 使用方式
llama.cpp 提供多种交互方式：

`llama-cli`：命令行交互式对话，适合快速测试
`llama-server`：HTTP 服务，兼容 OpenAI API（`/v1/chat/completions`），适合集成到应用
`llama-perplexity`：计算模型在测试集上的困惑度，评估量化质量
`llama-bench`：基准测试工具，测试不同量化格式和硬件的推理速度

C++ API 可直接嵌入应用，Python 绑定通过 `llama-cpp-python` 提供。Rust 绑定 `llama-cpp-rs` 提供安全的 FFI 封装。

## 与 Ollama 的关系
Ollama 是 llama.cpp 的上层封装。它自动管理 GGUF 版本选择、GPU 层数配置、模型下载和更新。Ollama 适合"不想管底层细节"的用户，llama.cpp 适合需要精细控制硬件参数的场景。两者使用相同的 GGUF 模型文件，可以在 Ollama 和 llama.cpp 之间切换。

Ollama 不支持连续批处理（每个请求串行处理），不适合高并发 API 服务。对于这类需求，应使用 vLLM 或 SGLang。

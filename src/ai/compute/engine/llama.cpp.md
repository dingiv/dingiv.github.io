---
title: llama.cpp
---

# llama.cpp

llama.cpp 是 Georgi Gerganov 开发的轻量级 LLM 推理引擎，最初为在 MacBook 上运行 LLaMA 模型而设计，现已发展成为支持 CPU/GPU 混合推理、跨平台部署的通用框架。

## GGUF 格式

GGUF (GPT-Generated Unified Format) 是 llama.cpp 定义的二进制模型格式。它将模型权重、 tokenizer 词表、配置参数打包为单个文件，支持 mmap 内存映射加载，启动速度快。更重要的是，GGUF 原生支持量化——从 Q4_0（4-bit）到 Q8_0（8-bit）再到 F16（16-bit），可根据硬件和精度需求灵活选择。

量化过程简单到极致。给定 GGUF 格式的 FP16 模型，`llama-cli` 工具可一键量化为 Q4_K_M（4-bit，中等映射质量）或 Q5_K_S（5-bit，小模型）。量化算法基于 GPTQ 思想，通过最小化权重误差保持精度，但在实现上更轻量，无需额外的校准数据集。

## CPU 推理

llama.cpp 的核心优势是 CPU 推理。通过充分利用 SIMD 指令（AVX2、AVX512、ARM NEON），在 CPU 上的推理速度远超 PyTorch。例如，在 M2 MacBook 上，llama.cpp 运行 7B Q4 模型可达 30 tokens/秒，而 PyTorch 仅 5 tokens/秒。

CPU 推理的价值在于 democratization——没有 GPU 的开发者也能运行大模型。对于边缘设备、嵌入式系统、开发测试环境，CPU 推理是唯一选择。llama.cpp 让这些场景成为可能。

## 混合推理

llama.cpp 支持 CPU + GPU 混合推理。将部分层（如 30%）卸载到 GPU，剩余层在 CPU 上计算，通过 Metal (Apple Silicon)、CUDA (NVIDIA)、ROCm (AMD) 后端加速。这平衡了显存占用和计算速度，适合显存有限但有 GPU 加速的设备。

混合推理的关键是层卸载策略。卸载太多层会爆显存，太少则加速不明显。llama.cpp 提供了 `--n-gpu-layers` 参数让用户手动调整，最佳值需要通过实验确定。一般来说，7B 模型在 8GB 显存 GPU 上可卸载 20-30 层，推理速度提升 2-3 倍。

## 使用方式

```bash
# 量化模型
llama-cli --model llama-2-7b.gguf --quantize-output --out-file llama-2-7b-q4.gguf --quantize-type Q4_K_M

# 运行推理
llama-cli --model llama-2-7b-q4.gguf --prompt "Hello, world" --n-predict 100

# 启动 OpenAI 兼容服务
llama-server --model llama-2-7b-q4.gguf --port 8080
```

llama.cpp 的设计哲学是简单。单个二进制文件包含所有功能，无需 Python 环境、无需 CUDA toolkit、无需复杂配置，下载即用。这使得它成为本地部署 LLM 的首选工具。

## 适用场景

llama.cpp 最适合资源受限环境：笔记本电脑、边缘设备、开发测试机器。对于生产环境的服务化部署，vLLM/TGI/TensorRT-LLM 更合适，因为它们的并发能力和吞吐量更高。但对于个人使用、离线部署、隐私敏感场景，llama.cpp 是最佳选择。

llama.cpp 的另一个优势是跨平台。Windows、macOS、Linux、Android、iOS 全平台支持，且编译简单（单文件 C++ 代码）。这使得 llama.cpp 成为嵌入式 AI 应用的首选推理引擎。

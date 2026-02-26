---
title: TensorRT-LLM
---

# TensorRT-LLM

TensorRT-LLM 是 NVIDIA 开发的推理优化框架，通过 TensorRT 的图优化和算子融合能力，将 LLM 推理性能推向极致。它是 NVIDIA 在推理领域对抗 vLLM/TGI 的王牌。

## 核心技术

TensorRT-LLM 的核心是 TensorRT，一个深度学习推理优化器。TensorRT 解析 PyTorch/ONNX 模型后，构建计算图，然后进行一系列优化：层融合（LayerNorm + Residual → 单个 kernel）、精度校准（FP32 → FP16/INT8）、内核自动调优（针对不同 GPU 架构选择最优 CUDA kernel）。这些优化将推理延迟降低 15-30%。

对于 LLM，TensorRT-LLM 额外优化了 Attention 算子。通过 in-place 更新 KV Cache（减少内存分配）、masked softmax 融合（减少 kernel 启动开销）、多头 attention 并行（增加 GPU 占用率），将 Attention 的计算效率提升到接近理论峰值。

## INT4 量化

TensorRT-LLM 支持 AWQ（Activation-aware Weight Quantization）INT4 量化。AWQ 的核心洞察是：只有 1% 的权重对量化敏感，这些权重保留高精度（FP16），其余 99% 量化为 INT4。这保持了模型精度的同时，将显存占用降低 75%，计算速度提升 2-3 倍（INT4 矩阵乘法比 FP16 快得多）。

量化过程高度自动化。给定 FP16 模型，TensorRT-LLM 自动计算每层的量化 scale、校准激活值范围、生成量化后的 engine 文件（`.engine`）。engine 文件是针对特定 GPU 架构（如 A100、H100）编译的二进制，加载后直接执行，无需 JIT 编译，启动速度快。

## In-flight Batching

In-flight Batching 是 TensorRT-LLM 的独特优化。当某个序列生成结束时，立即插入新序列，无需等待当前 batch 完成。这与 vLLM 的连续批处理类似，但 TensorRT-LLM 的实现在 CUDA 层面完成，调度开销更低，适合极高并发场景（1000+ 并发请求）。

## 使用方式

```bash
# 构建 INT4 量化引擎
python build.py --model_dir llama-2-7b --quantization int4_awq --output_dir llama-2-7b-int4

# 运行推理
python run.py --engine_dir llama-2-7b-int4 --max_output_len 512
```

TensorRT-LLM 的 API 偏底层，需要手动构建 engine、配置 tokenizer、管理 CUDA stream。这比 vLLM/TGI 的易用性差，但换来的是极致的性能控制力。对于追求极致性能的商业场景，TensorRT-LLM 是不二之选。

## 适用场景

TensorRT-LLM 最适合 NVIDIA GPU 架构（A100、H100、L40S）上的高性能推理。对于非 NVIDIA GPU（如 AMD ROCm），TensorRT-LLM 不支持，需要考虑 vLLM 或 TGI。对于 CPU 推理，llama.cpp 是更合适的选择。

TensorRT-LLM 的性能优势在 H100 上尤为明显，因为 NVIDIA 针对自家的 Transformer Engine（FP8 算子）做了深度优化。在 A100 上，TensorRT-LLM 与 vLLM 性能接近；在 H100 上，TensorRT-LLM 可领先 20-30%。

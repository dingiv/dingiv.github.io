---
title: 算子
order: 20
---

# 张量计算算子
算子是深度学习模型计算的基本单元，是 AI 引擎性能优化的基础。从矩阵乘法到注意力机制，每个算子的实现效率都会影响模型的整体性能。算子层的优化技术主要包括 FlashAttention、算子融合、量化等，这些技术从不同角度提升计算效率：FlashAttention 减少显存访问，算子融合减少 kernel 启动开销，量化降低显存占用和计算量，优化提升单 kernel 的计算效率。

| 类别     | 核心技术       | 适用场景         | 详细介绍                           |
| -------- | -------------- | ---------------- | ---------------------------------- |
| 显存优化 | FlashAttention | 长序列 Attention | [FlashAttention](./flashattention) |
| 计算优化 | 算子融合       | 连续算子合并     | [算子融合](./fusion)               |
| 精度优化 | 量化算法       | GPTQ/AWQ 算法层  | [量化算法](./quantization)         |
| 精度优化 | Marlin 内核    | W4A16 推理后端   | [Marlin](./marlin)                 |
| 精度优化 | 量化选型矩阵   | GPU×量化方案选型 | [量化选型](./quant_matrix)         |
| 格式规范 | 模型格式       | 存储/交换标准    | [模型格式](./format)               |
| 格式规范 | GGUF / K-quant | llama.cpp 生态   | [GGUF](./gguf)                    |
| 格式规范 | ONNX           | 跨框架交换       | [ONNX](./onnx)                    |
| 性能分析 | 算子优化       | Kernel 优化      | [算子优化](./optimization)         |

算子优化贯穿 AI 引擎的整个技术栈。推理引擎（如 vLLM、TGI）依赖 FlashAttention 来优化 Attention 计算；训练引擎（如 DeepSpeed、FSDP）依赖算子融合来减少通信开销；量化技术使得大模型能在显存有限的 GPU 上运行。理解算子层的优化技术，有助于深入理解 AI 引擎的性能瓶颈和优化方向。

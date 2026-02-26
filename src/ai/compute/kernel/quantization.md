---
title: 量化
---

# 量化

量化是将模型权重和激活从高精度（FP32、FP16）转换为低精度（INT8、INT4）的技术，通过牺牲少量精度换取显存占用和计算速度的大幅降低。对于资源受限的部署场景（边缘设备、移动端、显存有限的服务器），量化是不可或缺的优化手段。

## 量化基础

量化的数学定义是将连续的浮点数映射到离散的整数。对于对称量化，公式为 $x_{\text{quant}} = \text{round}(x / s)$，其中 $s$ 是 scale（缩放因子），将 FP32 范围映射到 INT8 范围 [-128, 127]。反量化为 $x_{\text{dequant}} = x_{\text{quant}} \times s$。

非对称量化引入零点（zero point），支持不对称的值域：$x_{\text{quant}} = \text{round}(x / s + z)$，其中 $z$ 是零点偏移。非对称量化在权重分布不均匀时（如 ReLU 后的激活值，全为正）精度更高，但计算稍复杂。

量化误差来源于两个因素：**精度损失**（INT8 仅 256 个离散值，FP32 是连续值）和**溢出截断**（超出范围的值被截断到 [-128, 127]）。前者是不可避免的舍入误差，后者可通过选择合适的 scale 和 zero point 减少超出范围的概率。

## 量化方法

PTQ（Post-Training Quantization，训练后量化）是最简单的方案，直接对训练好的模型进行量化，无需重新训练。PTQ 分为动态量化和静态量化：动态量化在推理时根据激活值的范围动态计算 scale，精度高但推理慢；静态量化使用校准数据集预先计算激活值的 scale，推理快但精度略低。GPTQ、AWQ、SpQR 是 PTQ 的代表，通过最小化权重误差或激活误差来保持精度。

QAT（Quantization-Aware Training，量化感知训练）在训练过程中模拟量化误差，让模型适应量化带来的精度损失。具体做法是在前向传播时插入 fake quantize 算子（模拟量化和反量化的误差），反向传播时通过 Straight-Through Estimator（STE）近似量化操作的梯度。QAT 的精度通常高于 PTQ，但需要重新训练，成本较高。

## LLM量化

大语言模型的量化有独特挑战。Transformer 的激活值范围动态变化（不同 prompt、不同位置的值域差异大），静态量化容易溢出；LayerNorm 和 Softmax 的数值稳定性对量化敏感，需要特殊处理；大模型的参数量大（7B 模型需要 14GB FP16 显存），量化到 INT4 可降至 4GB，这是部署的关键。

GPTQ 是最早的 LLM 量化方法之一，核心思想是**仅 1% 的权重对量化敏感**。通过二分搜索找到每个权重的量化 scale，最小化量化误差（MSE）。GPTQ 的缺点是需要全量模型在显存中才能计算 scale，7B 模型需要 14GB 显存才能量化到 4GB，这在显存受限时是个问题。

AWQ（Activation-aware Weight Quantization）的洞察是**应该根据激活值的大小来量化权重**。激活值大的位置，权重的量化误差会被放大，因此需要保留更高精度。AWQ 对每 1% 的权重保留 FP16 精度，其余量化为 INT4，精度与 GPTQ 相当，但速度快 2-3 倍（无需二分搜索）。

SpQR（Sparse Quantization）将量化问题建模为稀疏优化。大部分权重可以安全量化为 INT4，少量异常值（outliers）保留 FP16。SpQR 自动检测异常值（通过统计权重的分布），构建稀疏矩阵，推理时稀疏矩阵的乘法通过专门优化的 kernel 加速。SpQR 在 4-bit 量化下的精度优于 AWQ，但推理速度较慢（稀疏矩阵乘法比密集矩阵慢）。

## 推理加速

量化的推理加速来自两方面：**显存带宽减少**和**计算单元加速**。INT8 矩阵乘法的显存读写是 FP16 的一半，因此受限于显存带宽的算子（如逐元素操作）可加速 2 倍。对于计算密集型算子（如矩阵乘法），INT8 可使用 Tensor Core（NVIDIA GPU）或 INT8 SIMD 指令（CPU），加速比可达 4-8 倍。

但量化加速的前提是硬件支持。NVIDIA GPU 从 Turing 架构开始支持 INT8 Tensor Core，A100 的 INT8 理论性能为 624 TFLOPS（FP16 是 312 TFLOPS）。CPU 的 AVX512 VNNI、ARM 的 NEON dot 指令也支持 INT8 加速。如果硬件不支持 INT8 加速（如旧款 GPU），量化反而可能变慢（量化和反量化的额外开销）。

## 使用方式

HuggingFace 的 `bitsandbytes` 库提供了最简单的量化方案：

```python
from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b",
    load_in_8bit=True,  # INT8 量化
    torch_dtype=torch.float16,
)
```

对于 INT4 量化，需要使用 AWQ 或 GPTQ：

```python
# AWQ 量化
from awq import AutoAWQForCausalLM
model = AutoAWQForCausalLM.from_quantized("meta-llama/Llama-2-7b", quant_path="llama-2-7b-awq")

# GPTQ 量化
from auto_gptq import AutoGPTQForCausalLM
model = AutoGPTQForCausalLM.from_quantized("meta-llama/Llama-2-7b", use_triton=True)
```

量化后可通过 `torch.save` 保存为 GGUF 格式（llama.cpp）或 `.bin` 格式（HuggingFace），推理时直接加载，无需重新量化。

## 量化精度

量化精度用 PPL（Perplexity，困惑度）衡量，数值越低越好。FP16 的 PPL 视模型和任务而定，INT8 的 PPL 通常与 FP16 接近（差距 < 5%），INT4 的 PPL 可能比 FP16 高 10-20%（视量化方法而定）。AWQ 和 SpQR 在 INT4 下可保持与 INT8 接近的 PPL，但推理速度略慢。

选择量化位宽需要权衡精度、速度、显存。对于生产环境，INT8 是最稳妥的选择，精度损失小且硬件加速成熟。对于边缘设备或显存极度受限的场景，INT4 是可行方案，但需要仔细验证输出质量（尤其是对于数值敏感的任务，如数学计算、代码生成）。

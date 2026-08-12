---
title: ONNX
order: 55
---

# ONNX
ONNX（Open Neural Network Exchange）是 Microsoft 和 Meta 主推的开放式神经网络交换格式。它与当前 HuggingFace safetensors + GGUF 的 LLM 生态呈现出一条清晰的分界线——ONNX 源于传统深度学习时代（ResNet/BERT/YOLO），HF 生态源于大模型时代（Llama/Qwen/DeepSeek）。两者不是替代关系，而是在不同模型规模、不同部署场景下各自占据最优位置。

理解这条分界线是理解 ONNX 价值的起点。

## 两个时代的模型格式
2016-2020 年的深度学习世界由 CV（ResNet、YOLO、EfficientNet）和传统 NLP（BERT、GPT-2、T5）主导。这些模型的共同特征：参数量在百万到亿级、输入输出结构固定（图像→分类、文本→向量）、推理引擎需要嵌入到各种异构硬件中（手机 NPU、汽车芯片、IoT MCU）。ONNX 在这个背景下诞生——它的核心价值是"一次导出，到处运行"。

2023 年后的 LLM 时代完全不同：模型参数量跳跃到十亿到千亿级别、输入输出是变长 token 序列而非固定张量、自回归生成的 KV Cache 管理比单次前向传播的计算图复杂一个数量级。HF 的 safetensors + config.json 成为这个时代的事实标准——不是因为它比 ONNX 更"先进"，而是因为它更"薄"。safetensors 只是权重容器，模型结构由 config.json 和框架代码定义——不需要预先将计算图固化为 ONNX 的静态算子序列。

ONNX 的静态计算图思想恰恰是它在传统 AI 中优势的来源，也是它在 LLM 中不适的来源。传统模型的推理是确定性的单次前向传播——输入固定形状，输出固定形状，没有 KV Cache，没有自回归循环。这种场景下，静态图编译优化（算子融合、常量折叠、内存规划）的收益巨大。LLM 的推理是动态的自回归循环——每步生成改变 KV Cache，batch 中的请求随时加入退出——静态图预设的优化策略无法适应这种动态性。

这不是 ONNX 的失败，只是场景不匹配。就像叉车不能在高速公路上跑——不是叉车不好，是场景错了。

## ONNX 的核心设计
ONNX 使用 Protocol Buffers 序列化模型，包含三部分：

**计算图（Graph）**：用节点（Node）和边（Tensor）描述模型的前向传播路径。每个节点对应一个标准算子（Conv、MatMul、Softmax 等）。图是静态的——导出时形状和控制流已确定（dynamic axes 支持 batch 维度可变，但不支持变化的分支逻辑）。

**算子集（Opset）**：ONNX 的版本控制机制。每个 opset 版本定义了一组标准算子的语义。截至 2026 年最新 opset 约 22，覆盖了传统 CV/NLP 的绝大多数需求，但对 LLM 特有的操作（FlashAttention、RoPE、RMSNorm、SiLU activation）覆盖有限——这些算子在 opset 17+ 才逐步加入。

**权重与元数据**：模型的训练参数（FP32/FP16）和可选的模型描述信息（作者、版本、输入输出名称）。

```python
# PyTorch → ONNX 导出
import torch

model = MyVisionModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model, dummy_input, "model.onnx",
    input_names=["image"], output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
)
```

导出的 `.onnx` 文件是一个自包含的 protobuf 二进制——包含模型结构、权重和元数据。任何支持 ONNX 的运行时都可以加载它，不需要原始训练框架。

## ONNX Runtime：解耦训练与推理
ONNX Runtime（ORT）是微软开发的 ONNX 官方推理引擎。与 PyTorch/TensorFlow 的推理模式不同，ORT 只做推理——不需要 autograd 图、不需要优化器状态、不需要分布式训练基础设施。这种专注让它能做 PyTorch 无法做的激进优化：整图级别的算子融合（将 Conv+BN+ReLU 融合为单个 kernel）、跨层的内存复用（两个不再同时存在的中间张量共享同一块显存）、计算图重排（将独立的分支并行执行）。

ORT 的硬件后端（Execution Provider）覆盖 CPU（MLAS 库优化）、GPU（CUDA/TensorRT/ROCm）、NPU（Qualcomm QNN、Apple CoreML）、边缘设备（ARM NN、XNNPACK）、Web 端（ONNX Runtime Web，通过 WASM + WebGPU）。这和 HF 生态形成鲜明对比——HF 模型只能在 PyTorch/TF/JAX 等训练框架的推理模式下运行，硬件后端受限于框架本身的支持。

```python
import onnxruntime as ort
import numpy as np

# 选择后端——CPU、CUDA、TensorRT 等
session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]  # fallback 链
)

# 推理——输入 numpy 数组，输出 numpy 数组
input_name = session.get_inputs()[0].name
outputs = session.run(None, {input_name: np.random.randn(1, 3, 224, 224).astype(np.float32)})
```

对于传统 CV 模型，ORT + TensorRT 的推理性能通常比 PyTorch 高 1.5-3 倍。这个差距主要来自算子融合和内存优化——PyTorch 的 eager mode 每层独立执行，中间张量反复分配释放；ORT 将多层合并为单个 GPU kernel，中间结果留在寄存器/L1 缓存。

## ONNX 在 LLM 时代的角色
ONNX 在 LLM 生态中的位置是"补充"而非"替代"。HF 的 safetensors + transformers 对于训练、微调、快速实验是最佳选择——不需要导出步骤，修改模型代码后立即生效。GGUF 对于消费级硬件的本地推理是最佳选择——CPU+GPU 混合推理、K-quant 量化、单文件分发。ONNX Runtime GenAI（ort-genai）是微软推出的 LLM 推理 API——直接在 ORT 上加载 HF 模型，支持 continuous batching、KV Cache 管理、beam search 等 LLM 必需的特性。

ort-genai 的价值场景是"已经在用 ORT 部署传统模型的团队，希望用同一套基础设施部署 LLM"。如果你不需要这个统一性，HF + vLLM/llama.cpp 的生态更成熟。如果你需要在 Windows 生态中部署 LLM（DirectML 后端、ONNX Runtime 的 Windows ARM64 原生支持），ort-genai 是 DirectML 的原生路径。

## 模型优化管线
ONNX 生态的优化工具链是它相比 HF 生态的独特优势——静态图格式使得离线优化成为可能：

**ONNX Simplifier（onnxsim）**：自动简化计算图——常量折叠、去除无用节点、合并连续运算。通常在导出后第一步运行，可将模型体积减小 10-30%。

**Olive（Microsoft）**：自动化的模型优化管线——将 PyTorch/HF 模型转换为 ONNX → 图优化 → 量化（INT8/INT4）→ 架构特定优化（TensorRT/OpenVINO/QNN）→ 输出最优配置。Olive 的核心价值是自动化搜索——尝试不同组合（量化方案 × 算子融合策略 × 硬件后端），选取推理速度最快的配置。

**量化工具**：ORT 内置静态量化（需要校准数据集）和动态量化（推理时在线量化）。对于 CV 模型，INT8 量化后精度损失 < 0.5%，推理速度提升 2-3 倍。

```python
from onnxruntime.quantization import quantize_static, QuantType

# 静态 INT8 量化——需要校准数据
quantize_static(
    "model.onnx", "model_int8.onnx",
    calibration_data_reader=MyDataReader(),
    quant_format=QuantType.QInt8,
)
```

## ONNX 的适用边界
**最适合**：传统 CV 模型部署到异构硬件（ResNet/YOLO/EfficientNet → 手机/嵌入式/浏览器）、多框架混合的推理管线（PyTorch 训练 → ONNX 导出 → TensorRT 加速）、需要离线图优化的确定性推理、Windows 生态的 AI 推理。

**不太适合**：LLM 的自回归推理（HF + vLLM/llama.cpp 更成熟）、频繁修改模型结构的实验阶段（导出步骤增加迭代摩擦）、训练和微调（ONNX 只能做推理）。

ONNX 和 HF 生态在未来会继续共存——LLM 社区不会放弃 transformers 的灵活性，传统 CV 和工业部署不会放弃 ONNX 的跨硬件能力。ort-genai 试图在两者之间架桥，但目前 HF + 专用推理引擎（vLLM/SGLang/llama.cpp）的成熟度仍然领先。

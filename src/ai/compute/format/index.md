---
title: 模型格式
order: 50
---

# 模型格式
模型格式是 AI 模型的存储和交换标准，定义了模型权重、架构、元数据如何组织和序列化。一个良好的模型格式应具备：**可移植性**（跨平台兼容）、**安全性**（防止恶意代码注入）、**效率**（加载速度快、占用空间小）、**可扩展性**（支持新架构和新特性）。

## 各格式对比

| 格式             | 推出方             | 特点                             | 适用场景           | 详细介绍                  |
| ---------------- | ------------------ | -------------------------------- | ------------------ | ------------------------- |
| PyTorch .pt/.pth | Meta               | PyTorch 原生格式，支持完整计算图 | PyTorch 训练/推理  | -                         |
| HF 风格          | Hugging Face       | config + safetensors，生态标准   | 开源模型分发       | [HF 风格](./hf) |
| ONNX             | Microsoft/Facebook | 框架无关，跨平台推理             | 跨框架部署         | [ONNX](./onnx)            |
| GGUF             | llama.cpp          | 量化友好，CPU 推理优化           | 本地部署、边缘设备 | -                         |

## PyTorch 原生格式
PyTorch 的原生模型格式包括 `.pt`、`.pth`、`.pkl`（三者本质相同），使用 Python 的 pickle 模块序列化。这种格式可以保存完整的模型对象（包括架构、权重、优化器状态、训练状态），加载后可直接使用，无需重新定义模型结构。

```python
# 保存完整模型（包含架构和权重）
torch.save(model, "model.pth")

# 保存仅权重（state_dict）
torch.save(model.state_dict(), "weights.pth")

# 加载权重（需要先定义模型结构）
model = MyModelClass()
model.load_state_dict(torch.load("weights.pth"))
```

PyTorch 格式的优势是**完整性**和**灵活性**——可以保存任意 Python 对象（包括自定义层、优化器、学习率调度器）。但这也是它的劣势：pickle 格式存在安全风险（反序列化可执行任意代码）、跨版本兼容性差（PyTorch 版本升级可能导致旧模型无法加载）、文件体积大（包含不必要的元数据）。

生产环境中建议使用 `state_dict` 方式保存权重，而非保存完整模型。原因有二：一是安全性（避免执行任意代码），二是兼容性（模型结构由代码定义，而非依赖序列化对象）。对于跨平台部署，建议转换为 ONNX 或其他框架无关格式。

## GGUF
GGUF（GPT-Generated Unified Format）是 llama.cpp 推出的模型格式，专为 CPU 推理优化。GGUF 的核心设计是**量化友好**——支持 INT4、INT5、INT8 等多种量化格式，且在加载时自动反量化到 FP16 进行计算。这使得 GGUF 格式的模型占用空间小（7B 模型的 INT4 版本约 4GB），同时保持较好的推理质量。

GGUF 文件采用二进制格式，包含模型权重、词汇表、量化参数、元数据（如模型名称、训练信息）。与 GGML（llama.cpp 的旧格式）相比，GGUF 改进了内存映射（mmap）支持，允许大模型在内存不足时通过分页加载，降低硬件门槛。

```python
# 使用 llama.cpp 加载 GGUF 模型
from llama_cpp import Llama

model = Llama(
    model_path="llama-2-7b.Q4_K_M.gguf",
    n_ctx=2048,              # 上下文长度
    n_gpu_layers=-1,         # -1 表示将所有层加载到 GPU
    verbose=False
)

output = model("Hello, world!", max_tokens=50)
```

GGUF 格式在本地部署场景中非常流行。对于没有 NVIDIA GPU 的用户（如 Apple Mac、普通 PC），llama.cpp + GGUF 是运行大模型的主要方式。GGUF 的局限是不适合训练（仅支持推理），且量化后精度有所损失（INT4 的 PPL 通常比 FP16 高 10-20%）。

## 格式选择建议

| 场景          | 推荐格式                        | 理由                 |
| ------------- | ------------------------------- | -------------------- |
| PyTorch 训练  | state_dict + safetensors        | 安全、高效           |
| 模型分发      | HF 风格（config + safetensors） | 生态标准、兼容性强   |
| 本地 CPU 推理 | GGUF                            | 量化友好、内存占用小 |
| 跨框架部署    | ONNX                            | 框架无关、硬件支持广 |
| 边缘设备部署  | GGUF 或 TFLite                  | 量化、低资源优化     |

模型格式转换是常见的工程需求。Hugging Face 提供了 `transformers` 库的转换工具，支持从 PyTorch、TensorFlow、JAX 等格式互转。对于 ONNX 转换，可使用 `torch.onnx.export` 或 `onnxruntime` 的转换工具。对于 GGUF 转换，可使用 llama.cpp 的 `quantize` 工具将 HF 模型转换为 GGUF 格式。
